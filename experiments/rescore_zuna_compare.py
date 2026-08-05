"""Re-score the stored baseline-vs-ZUNA probability pairs with the fixed scoring.

The ZUNA comparison numbers quoted across the older documents — "sensitivity
26.3% -> 31.6%, false positives 328.7 -> 205.4 per 24 h" — were produced before
three scoring defects were fixed (see docs/reproduction_status.md §3):

  * the source method's decision stage was not applied,
  * detection fragments inside an already-matched seizure were charged as false
    positives, and
  * the two arms could fragment differently, so the *delta* between them was
    partly an artefact of that mis-counting rather than a real difference.

Re-running ZUNA is not necessary and not affordable (~6x real time, ~42 GiB).
The per-file probability caches it produced are still on disk, so this script
re-scores the stored `<stem>.baseline.probs.npz` / `<stem>.zuna.probs.npz` pairs
and regenerates the comparison from them.

    python experiments/rescore_zuna_compare.py \
        --dir artifacts/zuna_thesis/compare_first10 \
        --out artifacts/zuna_thesis/compare_first10/rescored.json

Reads only cached probabilities: no TensorFlow, no ZUNA, no inference.
"""
from __future__ import print_function

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import eval_config as cfg  # noqa: E402
from gui.io.cache import load_probability_file  # noqa: E402
from experiments.compare_zuna import (  # noqa: E402
    read_reference_events, read_reference_duration, summarize_run,
)
from experiments.evaluate_baseline import _safe_auc, window_labels  # noqa: E402


def find_edf_for_stem(stem):
    """Locate the source EDF (for its .csv_bi) by stem under sample_data/."""
    hits = glob.glob(os.path.join(REPO, 'sample_data', '**', stem + '.edf'),
                     recursive=True)
    return hits[0] if hits else None


def collect_pairs(directory):
    pairs = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.probs.npz'))):
        base = os.path.basename(path)
        for suffix, arm in (('.baseline.probs.npz', 'baseline'),
                            ('.zuna.probs.npz', 'zuna')):
            if base.endswith(suffix):
                pairs.setdefault(base[:-len(suffix)], {})[arm] = path
    return {k: v for k, v in pairs.items() if 'baseline' in v and 'zuna' in v}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True,
                    help='Directory of <stem>.{baseline,zuna}.probs.npz pairs.')
    ap.add_argument('--threshold', type=float, default=cfg.THRESHOLD)
    ap.add_argument('--tolerance', type=float, default=cfg.EVENT_TOLERANCE_S)
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    pairs = collect_pairs(os.path.abspath(args.dir))
    if not pairs:
        raise SystemExit('No baseline/ZUNA probability pairs in ' + args.dir)

    acc = {arm: {'hits': 0, 'refs': 0, 'fp': 0, 'dup': 0, 'dur': 0.0,
                 'labels': [], 'scores': []}
           for arm in ('baseline', 'zuna')}
    per_file = []

    for stem in sorted(pairs):
        edf = find_edf_for_stem(stem)
        if edf is None:
            print('  ! no EDF found for {}, skipping'.format(stem))
            continue
        ref_path = os.path.splitext(edf)[0] + '.csv_bi'
        refs = read_reference_events(ref_path)
        duration_s = read_reference_duration(ref_path)

        row = {'stem': stem, 'n_refs': len(refs)}
        for arm in ('baseline', 'zuna'):
            cache = load_probability_file(pairs[stem][arm])
            if cache is None:
                continue
            starts = np.asarray(cache['window_starts'])
            probs = np.asarray(cache['probs'])
            summary = summarize_run(
                stem, starts, probs, refs, threshold=args.threshold,
                duration_s=duration_s, tolerance_s=args.tolerance)

            labels = window_labels(starts, refs)
            keep = probs != 0.0
            acc[arm]['labels'].append(labels[keep])
            acc[arm]['scores'].append(probs[keep])
            acc[arm]['hits'] += summary['hits']
            acc[arm]['refs'] += summary['n_refs']
            acc[arm]['fp'] += summary['false_positives']
            acc[arm]['dup'] += summary['duplicate_detections']
            acc[arm]['dur'] += summary['duration_s']
            row[arm] = {
                'hits': summary['hits'], 'fp': summary['false_positives'],
                'duplicates': summary['duplicate_detections'],
                'max_prob': summary['max_prob'],
            }
        per_file.append(row)

    print('\n{} files re-scored at threshold {:.2f} '
          '(source post-processing {}, FP counting fixed)\n'.format(
              len(per_file), args.threshold,
              'on' if cfg.USE_SOURCE_POSTPROCESSING else 'off'))

    summary = {}
    print('{:<12} {:>10} {:>12} {:>10} {:>6} {:>12}'.format(
        'arm', 'hits/refs', 'sensitivity', 'fp/24h', 'dup', 'window AUC'))
    print('-' * 68)
    for arm in ('baseline', 'zuna'):
        a = acc[arm]
        days = a['dur'] / 86400.0
        sens = (float(a['hits']) / a['refs']) if a['refs'] else None
        fa = (a['fp'] / days) if days > 0 else None
        labels = np.concatenate(a['labels']) if a['labels'] else np.array([])
        scores = np.concatenate(a['scores']) if a['scores'] else np.array([])
        auc = _safe_auc(labels, scores, 'roc')
        summary[arm] = {
            'hits': a['hits'], 'n_refs': a['refs'],
            'sensitivity': sens, 'false_positives': a['fp'],
            'false_alarms_per_24h': fa, 'duplicate_detections': a['dup'],
            'window_auc': auc, 'total_duration_s': a['dur'],
        }
        print('{:<12} {:>10} {:>12} {:>10} {:>6} {:>12}'.format(
            arm, '{}/{}'.format(a['hits'], a['refs']),
            'n/a' if sens is None else '{:.1%}'.format(sens),
            'n/a' if fa is None else '{:.1f}'.format(fa),
            a['dup'],
            'n/a' if auc is None else '{:.4f}'.format(auc)))

    b, z = summary['baseline'], summary['zuna']
    print('\nDelta (ZUNA - baseline):')
    if b['sensitivity'] is not None and z['sensitivity'] is not None:
        print('  sensitivity  {:+.1f} percentage points ({} -> {} hits)'.format(
            100.0 * (z['sensitivity'] - b['sensitivity']), b['hits'], z['hits']))
    if b['false_alarms_per_24h'] is not None:
        print('  FP/24h       {:+.1f}'.format(
            z['false_alarms_per_24h'] - b['false_alarms_per_24h']))
    if b['window_auc'] is not None and z['window_auc'] is not None:
        print('  window AUC   {:+.4f}  ({} on the threshold-independent '
              'measure)'.format(z['window_auc'] - b['window_auc'],
                                'better' if z['window_auc'] > b['window_auc']
                                else 'WORSE'))

    payload = {'threshold': args.threshold, 'config': cfg.as_dict(),
               'n_files': len(per_file), 'summary': summary,
               'per_file': per_file}
    if args.out:
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
