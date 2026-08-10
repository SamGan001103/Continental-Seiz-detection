"""Put a confidence interval on the ZUNA window-AUC delta.

`docs/RESULTS.md` §5 reports baseline 0.6878 vs ZUNA 0.6466 — a delta of
−0.0412 — over 10 recordings and 19 reference seizures, and describes the
result as "inconclusive at best and mildly negative on the more rigorous
measure". That description is almost certainly right, but it is an assertion:
nothing in the repository says how wide the uncertainty on −0.0412 actually is.

With 19 seizures it could easily be indistinguishable from zero, and a thesis
that reports a negative result should be able to say so quantitatively rather
than hedge in prose.

Method
------
The two arms score the *same* recordings, so the comparison is paired and the
bootstrap must preserve that: resample **recordings** (the cluster), and within
each resample recompute both arms' pooled AUC from the same drawn files. This
is the same file-level cluster bootstrap used for the headline replication
interval, and it is what stops the interval from being falsely narrow —
windows within a recording are not independent.

    python experiments/zuna_auc_interval.py

Reads only the stored probability caches. Runs no inference.
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

import eval_config as cfg                                    # noqa: E402
from gui.io.cache import load_probability_file                # noqa: E402
from gui.io.csv_bi import read_csv_bi                         # noqa: E402


def window_labels(starts, refs, segment_s):
    """Any-overlap labelling, matching the project protocol used in §5."""
    lab = np.zeros(len(starts), dtype=np.int8)
    for a, b in refs:
        lab |= ((starts + segment_s > a) & (starts < b)).astype(np.int8)
    return lab


def load_pair(base_path):
    """Return (labels, baseline_probs, zuna_probs) for one recording."""
    zuna_path = base_path.replace('.baseline.', '.zuna.')
    meta_path = base_path.replace('.baseline.probs.npz', '.zuna_compare.json')
    if not (os.path.exists(zuna_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    ref_csv = meta.get('reference_csv_bi')
    if not ref_csv or not os.path.exists(ref_csv):
        cand = os.path.join(REPO, ref_csv) if ref_csv else None
        if not (cand and os.path.exists(cand)):
            return None
        ref_csv = cand
    events = read_csv_bi(ref_csv)
    refs = [(float(e['start']), float(e['stop'])) for e in events
            if str(e.get('label', '')).lower().startswith('seiz')]
    if not refs:
        return None

    b = load_probability_file(base_path)
    z = load_probability_file(zuna_path)
    if b is None or z is None:
        return None

    # Two independent requirements, and getting either wrong biases the result.
    #
    # 1. Match on window START, not index: the arms need not produce the same
    #    number of windows, and aligning by position would compare different
    #    moments in the recording.
    # 2. Drop a window if EITHER arm failed to score it. Masking each arm
    #    separately — which experiments/rescore_zuna_compare.py does with
    #    `keep = probs != 0.0` — leaves the arms on different window sets:
    #    measured here, 6 windows are refused in the baseline arm and scored in
    #    the ZUNA arm, so those 6 enter one arm's AUC and not the other's.
    #    Keeping them instead, as an earlier version of this script did, is no
    #    better: their stored 0.0 is a *sentinel* for "never scored", and
    #    feeding it to the AUC ranks a refusal as a confident negative.
    bs = {int(s): i for i, s in enumerate(b['window_starts'])}
    zs = {int(s): i for i, s in enumerate(z['window_starts'])}
    b_ok = np.asarray(b['skip_code']) == 0
    z_ok = np.asarray(z['skip_code']) == 0
    shared = sorted(s for s in (set(bs) & set(zs))
                    if b_ok[bs[s]] and z_ok[zs[s]])
    if not shared:
        return None
    starts = np.array(shared, dtype=float)
    bp = np.asarray(b['probs'])[[bs[s] for s in shared]]
    zp = np.asarray(z['probs'])[[zs[s] for s in shared]]
    lab = window_labels(starts, refs, float(meta.get('segment_s',
                                                     cfg.SEGMENT_S)))
    return lab, bp, zp


def pooled_auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate(labels)
    s = np.concatenate(scores)
    if y.min() == y.max():
        return float('nan')
    return float(roc_auc_score(y, s))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='artifacts/zuna_thesis/compare_first10')
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(REPO, args.dir,
                                          '*.baseline.probs.npz')))
    per_file = [p for p in (load_pair(q) for q in paths) if p is not None]
    if not per_file:
        print('no paired recordings found under {}'.format(args.dir))
        return 1

    labs = [x[0] for x in per_file]
    base = [x[1] for x in per_file]
    zuna = [x[2] for x in per_file]
    n_files = len(per_file)
    n_win = sum(len(l) for l in labs)
    n_pos = int(sum(int(l.sum()) for l in labs))

    a_base = pooled_auc(labs, base)
    a_zuna = pooled_auc(labs, zuna)
    delta = a_zuna - a_base

    print('paired recordings          : {}'.format(n_files))
    print('windows compared           : {} ({} ictal)'.format(n_win, n_pos))
    print()
    print('pooled window AUC, baseline: {:.4f}'.format(a_base))
    print('pooled window AUC, ZUNA    : {:.4f}'.format(a_zuna))
    print('delta (ZUNA - baseline)    : {:+.4f}'.format(delta))
    print()

    rng = np.random.RandomState(args.seed)
    deltas, bases, zunas = [], [], []
    for _ in range(args.boot):
        idx = rng.randint(0, n_files, n_files)      # resample RECORDINGS
        L = [labs[i] for i in idx]
        if sum(int(l.sum()) for l in L) == 0:
            continue                                 # no positives to rank
        ab = pooled_auc(L, [base[i] for i in idx])
        az = pooled_auc(L, [zuna[i] for i in idx])
        if np.isnan(ab) or np.isnan(az):
            continue
        bases.append(ab)
        zunas.append(az)
        deltas.append(az - ab)

    d = np.array(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_worse = float((d < 0).mean())

    print('cluster bootstrap by recording, {} resamples'.format(len(d)))
    print('  baseline AUC  95 % CI    : [{:.4f}, {:.4f}]'.format(
        *np.percentile(bases, [2.5, 97.5])))
    print('  ZUNA AUC      95 % CI    : [{:.4f}, {:.4f}]'.format(
        *np.percentile(zunas, [2.5, 97.5])))
    print('  DELTA         95 % CI    : [{:+.4f}, {:+.4f}]'.format(lo, hi))
    print('  P(ZUNA worse on AUC)     : {:.3f}'.format(p_worse))
    print()
    crosses_zero = lo < 0.0 < hi
    print('interval {} zero'.format('CROSSES' if crosses_zero else
                                    'excludes'))
    if crosses_zero:
        print('  -> the AUC difference is NOT statistically distinguishable')
        print('     from zero at this sample size. "Mildly negative" overstates')
        print('     what 10 recordings can support; say "no detectable')
        print('     difference, point estimate negative" instead.')
    else:
        print('  -> the difference is distinguishable from zero at this')
        print('     sample size, in the direction of the point estimate.')

    out = {
        'n_files': n_files, 'n_windows': n_win, 'n_ictal_windows': n_pos,
        'auc_baseline': a_base, 'auc_zuna': a_zuna, 'delta': delta,
        'delta_ci95': [float(lo), float(hi)],
        'p_zuna_worse_on_auc': p_worse,
        'bootstrap_resamples': int(len(d)),
        'method': 'cluster bootstrap by recording, paired arms',
    }
    dest = os.path.join(REPO, 'artifacts', 'zuna_thesis',
                        'zuna_auc_interval.json')
    with open(dest, 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write('\n')
    print('\nwrote {}'.format(os.path.relpath(dest, REPO)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
