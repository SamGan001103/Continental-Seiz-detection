"""Score this detector the way the published TUH comparators were scored.

WHY THIS EXISTS

The source paper's Table 2 reports **no sensitivity and no FA/24 h for its own TUH row** — only
AUC 0.84. But the same table cites two published TUH event-level results as comparators, and
those ARE the numbers this project can legitimately be measured against:

    TUH v1.1.0   Golmohammadi et al.   OVLP   39.15 %   22.83 FA/24 h
    TUH v1.4.1   Golmohammadi et al.   OVLP   30.83 %    6.75 FA/24 h

Both use **OVLP** (any-overlap) scoring. The paper's own RPAH figures (76.68 % @ 56.55) use
**SDR**, whose footnote states it "combines the false alarms within 30 seconds into one".

This project's default event scoring is neither: it uses greedy one-prediction-per-reference
matching with a 5-second proximity tolerance, and counts every non-matching prediction
separately. Comparing that directly against an OVLP or SDR number compares scoring conventions
as much as detectors — exactly the error the literature review warns about (a gated CNN-LSTM
scores 30.8 % under OVLP and 12.5 % under TAES: same system, different rule).

So this script re-scores the same cached probabilities under each convention:

    project   greedy 1:1 matching, 5 s proximity tolerance, every stray prediction is an alarm
    ovlp      a reference is detected if ANY prediction overlaps it; a prediction is a false
              alarm only if it overlaps NO reference. Multiple hits on one reference count once.
    ovlp+sdr  OVLP, then false alarms closer than 30 s are merged into one, per the SDR footnote.

Nothing about the model changes. This measures how much of the apparent gap is convention.

    python experiments/comparable_scoring.py --manifest artifacts/zuna_thesis/manifest_full.csv
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import eval_config as cfg  # noqa: E402
from gui.io.cache import cache_path_for, load_probability_file  # noqa: E402
from gui.postprocess import events_from_probs  # noqa: E402
from experiments.evaluate_baseline import iter_manifest_rows, load_file  # noqa: E402
from experiments.compare_zuna import match_events  # noqa: E402

# Published comparators, from Table 3 of the PUBLISHED journal version (Expert
# Systems with Applications 207:118083). The preprint's Table 2 differs: it
# attributed the v1.1.0 row to Golmohammadi and labelled the second row v1.4.1.
# Cite the journal version.
PUBLISHED = [
    ('Shah et al. 2017          (TUH v1.1.0)', 'OVLP', 0.3915, 22.83),
    ('Golmohammadi et al. 2020  (TUH v1.4.0)', 'OVLP', 0.3083, 6.75),
    ('This work / RPAH 1006      [NOT comparable]', 'SDR', 0.7668, 56.55),
    ('This work / RPAH 66-pilot  [NOT comparable]', 'SDR', 0.8710, 47.96),
]
# AUC-only rows from the same table, for the window-level comparison.
PUBLISHED_AUC = [
    ('Saab et al. 2020   (TUH v1.4.0)', 0.78),
    ('Tang et al. 2022   (TUH v1.5.2)', 0.82),
    ('This work          (TUH v1.5.1 dev)', 0.84),
    ('This work          (EPILEPSIAE, public)', 0.81),
    ('This work          (RPAH)', 0.82),
]
SDR_MERGE_S = 30.0


def _overlaps(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) > 0.0


def score_ovlp(preds, refs, merge_fa_s=None):
    """Any-Overlap scoring.

    A reference is detected if ANY prediction overlaps it — several predictions
    inside one long seizure count once, which is what the paper's footnote
    describes ("multiple shorter events detected within the long reference
    event"). A prediction is a false alarm only if it overlaps no reference.

    merge_fa_s : if set, false alarms separated by less than this are merged
        into one, reproducing the SDR convention.
    """
    R = [(float(r['start']), float(r['stop'])) for r in refs]
    P = [(float(a), float(b)) for a, b, _s in preds]

    hits = sum(1 for r in R if any(_overlaps(p, r) for p in P))
    fa = [p for p in P if not any(_overlaps(p, r) for r in R)]

    if merge_fa_s and fa:
        fa = sorted(fa)
        merged = [list(fa[0])]
        for s, e in fa[1:]:
            if s - merged[-1][1] < float(merge_fa_s):
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        fa = merged
    return hits, len(fa), len(P)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=[0.5, 0.3, 0.1, 0.05, 0.01])
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    recs = []
    for edf, coh in iter_manifest_rows(os.path.abspath(args.manifest)):
        r = load_file(edf, coh)
        if r is None or not r['has_reference']:
            continue
        c = load_probability_file(cache_path_for(edf))
        if c is None:
            continue
        r['skip'] = np.asarray(c['skip_code'])
        recs.append(r)

    total_hours = sum(r['duration_s'] for r in recs) / 3600.0
    n_refs = sum(len(r['refs']) for r in recs)
    print('\n{} recordings, {:.1f} h, {} reference seizures'.format(
        len(recs), total_hours, n_refs))
    print('post-processing: averaging={}, shaping={} (concat <{:g}s, discard '
          '<{:g}s)'.format(cfg.USE_PER_SECOND_AVERAGING,
                           cfg.USE_SOURCE_POSTPROCESSING,
                           cfg.MAX_MERGE_GAP_S, cfg.MIN_EVENT_DURATION_S))

    rows = []
    for thr in args.thresholds:
        acc = {k: [0, 0] for k in ('project', 'ovlp', 'ovlp_sdr')}
        n_pred = 0
        for r in recs:
            preds = events_from_probs(
                r['starts'], r['probs'], thr, cfg.SEGMENT_S,
                duration_s=r['duration_s'],
                min_duration_s=cfg.MIN_EVENT_DURATION_S,
                max_gap_s=cfg.MAX_MERGE_GAP_S,
                average=cfg.USE_PER_SECOND_AVERAGING)
            n_pred += len(preds)

            pe = [{'start': a, 'stop': b} for a, b, _ in preds]
            m, fp, _miss, _dup = match_events(pe, r['refs'],
                                              tolerance_s=cfg.EVENT_TOLERANCE_S)
            acc['project'][0] += len(m)
            acc['project'][1] += len(fp)

            h, f, _ = score_ovlp(preds, r['refs'])
            acc['ovlp'][0] += h
            acc['ovlp'][1] += f

            h, f, _ = score_ovlp(preds, r['refs'], merge_fa_s=SDR_MERGE_S)
            acc['ovlp_sdr'][0] += h
            acc['ovlp_sdr'][1] += f

        row = {'threshold': thr, 'n_predictions': n_pred}
        for k, (h, f) in acc.items():
            row[k] = {'hits': h, 'sensitivity': h / float(n_refs) if n_refs else None,
                      'false_alarms': f,
                      'fa_per_24h': f / (total_hours / 24.0) if total_hours else None}
        rows.append(row)

    print('\n{:>5} | {:^24} | {:^24} | {:^24}'.format(
        'thr', 'project (greedy 1:1, 5s)', 'OVLP (any-overlap)',
        'OVLP + SDR (FA merged 30s)'))
    print('{:>5} | {:>10} {:>12} | {:>10} {:>12} | {:>10} {:>12}'.format(
        '', 'sens', 'FA/24h', 'sens', 'FA/24h', 'sens', 'FA/24h'))
    print('-' * 90)
    for row in rows:
        print('{:>5.2f} | {:>9.1f}% {:>12.1f} | {:>9.1f}% {:>12.1f} | '
              '{:>9.1f}% {:>12.1f}'.format(
                  row['threshold'],
                  100 * row['project']['sensitivity'], row['project']['fa_per_24h'],
                  100 * row['ovlp']['sensitivity'], row['ovlp']['fa_per_24h'],
                  100 * row['ovlp_sdr']['sensitivity'], row['ovlp_sdr']['fa_per_24h']))

    print('\nPublished comparators (source paper Table 2):')
    for name, method, sens, fa in PUBLISHED:
        print('  {:<38} {:<5} {:>6.1f}% {:>10.2f} FA/24h'.format(
            name, method, 100 * sens, fa))
    print('\nAUC rows from the same table (window level):')
    for name, auc in PUBLISHED_AUC:
        print('  {:<42} {:>5.2f}'.format(name, auc))

    print('\nThe RPAH rows are for completeness ONLY: private clinical data, a')
    print('20-channel model (19 EEG + ECG), and the PWA/PEI lens, which sets its')
    print('thresholds from the last TWO HOURS of signal and therefore cannot run')
    print('on recordings of this length. The two OVLP rows are the legitimate')
    print('public-data comparison.')
    print('\nThe authors themselves write, of the corpus we evaluate on:')
    print('  "the short interictal periods in the TUH dataset do not provide a')
    print('   realistic specificity test venue for any seizure detection')
    print('   research and development"')
    print('TUH is 6.2 % seizure by duration; RPAH is 0.2 %. Cite that whenever')
    print('an FA/24 h figure from this corpus is reported.')

    if args.out:
        payload = {'config': cfg.as_dict(), 'n_recordings': len(recs),
                   'total_hours': total_hours, 'n_reference_seizures': n_refs,
                   'sdr_merge_s': SDR_MERGE_S, 'rows': rows,
                   'published_comparators': [
                       {'name': n, 'method': m, 'sensitivity': s, 'fa_per_24h': f}
                       for n, m, s, f in PUBLISHED]}
        out = os.path.abspath(args.out)
        parent = os.path.dirname(out)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        with open(out, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')
        print('\nwrote {}'.format(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
