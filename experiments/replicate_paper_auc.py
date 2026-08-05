"""Score the cached probabilities under the source paper's window protocol.

The source paper (Yang et al., *Expert Syst. Appl.* 207:118083, 2022, Table 2)
reports **AUC 0.84 on the TUH EEG Corpus v1.5.1**, 19 channels, 12-second
windows, ICA — the exact configuration these pretrained weights implement. That
number is the model's raw window-level discrimination on the TUH *development*
split (Fig. 5, "TUH-TUH ... trained on the TUH training dataset and tested on the
TUH development dataset"), before the PWA/PEI lens, which the paper applies only
to the RPAH inference.

So the architecture and preprocessing are directly comparable. The **window
sampling protocol is not**, and this script measures how much of the difference
that alone explains.

How the paper sampled dev windows (`utils/ICA_load_data_elec.py`
:115-157, the code that generated the training and dev features):

    while (st + i) * 250 + window_len < sp * 250:      # window must fit ENTIRELY
        s = data[:, (st+i)*250 : (st+i)*250 + window_len]
        ...
        i += 12                                        # dev setting: NO overlap

Each labelled interval `[st, sp]` from the reference file is tiled with
non-overlapping 12-second windows that fit entirely inside it, and each window
inherits that interval's label. A window straddling a seizure boundary is never
generated, so every scored window is unambiguously ictal or background.

How this project scores windows (`experiments/evaluate_baseline.py:window_labels`):
a sliding window every 6 s across the whole recording, labelled positive if it
*overlaps* a seizure by any amount. Boundary windows are therefore included and
labelled positive even when they are mostly background.

That is a harder and noisier task, and the two AUCs are not comparable. This
script recomputes the AUC under both protocols on the same cached probabilities:

    paper    windows fully inside a seizure (1) or touching no seizure (0);
             straddling windows excluded; optionally non-overlapping
    project  every window, positive on any overlap

Reads only `<edf>.probs.npz`, so it needs no TensorFlow and re-runs no inference.

    python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest.csv
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
from experiments.evaluate_baseline import (  # noqa: E402
    iter_manifest_rows, load_file, window_labels,
)

SEGMENT_S = cfg.SEGMENT_S


def paper_protocol_labels(window_starts, refs, segment_s=SEGMENT_S):
    """Label windows the way the source paper's dev-set loader did.

    Returns ``(labels, keep)``. ``keep`` is False for windows that straddle a
    seizure boundary — the paper's loader could not emit those, because a window
    had to fit entirely inside one labelled interval.
    """
    starts = np.asarray(window_starts, dtype=np.float64)
    labels = np.zeros(len(starts), dtype=np.int32)
    keep = np.ones(len(starts), dtype=bool)

    for i, t in enumerate(starts):
        w0, w1 = float(t), float(t) + float(segment_s)
        inside_any = False
        overlaps_any = False
        for ref in refs:
            r0, r1 = float(ref['start']), float(ref['stop'])
            if w0 >= r0 and w1 <= r1:
                inside_any = True
            if min(w1, r1) - max(w0, r0) > 0.0:
                overlaps_any = True
        if inside_any:
            labels[i] = 1                 # fully ictal
        elif not overlaps_any:
            labels[i] = 0                 # fully background
        else:
            keep[i] = False               # straddles a boundary: never sampled
    return labels, keep


def _auc(labels, scores):
    labels = np.asarray(labels)
    if labels.size == 0:
        return None
    pos = int(labels.sum())
    if pos == 0 or pos == labels.size:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))


def bootstrap_ci(per_file_arrays, n_boot=2000, seed=13, target=None):
    """File-level (cluster) bootstrap CI for a pooled AUC.

    Resamples whole FILES, not windows. Windows inside one recording are highly
    correlated — same patient, same montage, same artifact regime — so a
    window-level bootstrap treats them as independent evidence and reports an
    interval far too narrow. The file-level interval is the honest one, and it
    is what should be quoted next to any AUC from a small file set.

    per_file_arrays : list of (labels, scores) per file
    target          : optional reference value; returns P(bootstrap AUC >= it)
    """
    from sklearn.metrics import roc_auc_score
    files = [(np.asarray(a), np.asarray(b, dtype=np.float64))
             for a, b in per_file_arrays if len(a)]
    if len(files) < 2:
        return None
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(files), len(files))
        lab = np.concatenate([files[i][0] for i in idx])
        sc = np.concatenate([files[i][1] for i in idx])
        if 0 < lab.sum() < lab.size:
            boots.append(roc_auc_score(lab, sc))
    if not boots:
        return None
    boots = np.array(boots)
    out = {'n_boot': int(boots.size),
           'ci_lo': float(np.percentile(boots, 2.5)),
           'ci_hi': float(np.percentile(boots, 97.5))}
    if target is not None:
        out['p_ge_target'] = float((boots >= target).mean())
        out['target'] = float(target)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--keep-zero', action='store_true',
                    help='Keep prob==0 (preprocessing-skipped) windows.')
    ap.add_argument('--n-boot', type=int, default=2000,
                    help='Bootstrap resamples for the confidence interval.')
    ap.add_argument('--target-auc', type=float, default=0.84,
                    help="The source paper's published TUH AUC (Table 2).")
    args = ap.parse_args(argv)

    records = []
    for edf_abs, cohort in iter_manifest_rows(os.path.abspath(args.manifest)):
        rec = load_file(edf_abs, cohort, exclude_zero=not args.keep_zero)
        if rec is not None:
            records.append(rec)
    if not records:
        raise SystemExit('No probability caches found.')

    variants = {
        'project_any_overlap': ([], []),
        'paper_pure_windows': ([], []),
        'paper_pure_nonoverlapping': ([], []),
    }
    n_straddle = 0
    n_total = 0
    per_file = []

    for r in records:
        starts, probs, scored = r['starts'], r['probs'], r['keep']
        refs = r['refs']

        proj_lab = window_labels(starts, refs)
        paper_lab, paper_keep = paper_protocol_labels(starts, refs)
        n_straddle += int((~paper_keep).sum())
        n_total += len(starts)

        # Non-overlapping: at the canonical 6 s stride, every second window
        # gives the paper's 12 s tiling.
        step = cfg.STEP_S if cfg.STEP_S else 6
        stride_mult = max(1, int(round(SEGMENT_S / float(step))))
        nonoverlap = np.zeros(len(starts), dtype=bool)
        nonoverlap[::stride_mult] = True

        sel = {
            'project_any_overlap': (proj_lab, scored),
            'paper_pure_windows': (paper_lab, scored & paper_keep),
            'paper_pure_nonoverlapping': (paper_lab,
                                          scored & paper_keep & nonoverlap),
        }
        row = {'stem': r['stem'], 'cohort': r['cohort']}
        for name, (lab, mask) in sel.items():
            variants[name][0].append(lab[mask])
            variants[name][1].append(probs[mask])
            row[name] = _auc(lab[mask], probs[mask])
        per_file.append(row)

    print('{} files, {} windows, {} straddling a seizure boundary '
          '({:.1f}% excluded by the paper protocol)\n'.format(
              len(records), n_total, n_straddle,
              100.0 * n_straddle / n_total if n_total else 0.0))

    payload = {'n_files': len(records), 'n_windows': n_total,
               'n_boundary_windows': n_straddle, 'per_file': per_file,
               'pooled': {}}

    print('{:<30} {:>10} {:>18} {:>8} {:>6}'.format(
        'protocol', 'pooled AUC', '95% CI (by file)', 'windows', 'pos'))
    print('-' * 78)
    for name in ('project_any_overlap', 'paper_pure_windows',
                 'paper_pure_nonoverlapping'):
        labs = np.concatenate(variants[name][0]) if variants[name][0] else np.array([])
        scs = np.concatenate(variants[name][1]) if variants[name][1] else np.array([])
        pooled = _auc(labs, scs)
        per = [row[name] for row in per_file if row[name] is not None]
        mean_pf = float(np.mean(per)) if per else None
        ci = bootstrap_ci(list(zip(variants[name][0], variants[name][1])),
                          n_boot=args.n_boot, target=args.target_auc)
        payload['pooled'][name] = {
            'pooled_auc': pooled, 'n_windows': int(labs.size),
            'n_positive': int(labs.sum()) if labs.size else 0,
            'mean_per_file_auc': mean_pf, 'n_files_with_auc': len(per),
            'bootstrap': ci}
        print('{:<30} {:>10} {:>18} {:>8} {:>6}'.format(
            name,
            'n/a' if pooled is None else '{:.4f}'.format(pooled),
            'n/a' if ci is None else '[{:.3f}, {:.3f}]'.format(ci['ci_lo'],
                                                               ci['ci_hi']),
            int(labs.size), int(labs.sum()) if labs.size else 0))
        if ci and 'p_ge_target' in ci:
            print('{:<30} {:>10} P(AUC >= {:.2f}) = {:.2f}'.format(
                '', '', ci['target'], ci['p_ge_target']))

    print('\nSource paper (Table 2): AUC {:.2f} — TUH EEG Corpus v1.5.1, '
          '19 ch, 12 s window, ICA,\n'
          'measured on the TUH *development* split (Fig. 5 "TUH-TUH"), '
          'before the PWA/PEI lens.'.format(args.target_auc))
    print('The CI resamples whole FILES, not windows: windows within a '
          'recording are highly\ncorrelated, so a window-level interval would '
          'be far too narrow to be honest.')

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
