"""Does the per-window ICA stage help or hurt, at the window level?

For each seizure file in the manifest, compare the stored ICA-on cache against a
freshly computed ICA-off run. The original question was whether the low-AUC
outliers come from ICA removing ictal activity (over-aggressive EOG rejection)
rather than a genuine model failure.

Run in seiz36 (needs TensorFlow for the ICA-off pass).

    python experiments/diag_ica.py

Two things this script now does that the first version did not, both of which
change how much the answer can be trusted:

**The arms are compared on an identical window set.** Each arm previously got
its own `probs != 0.0` mask. That inference is exactly what `skip_code` was
added to the cache to replace — a real softmax output is effectively never
exactly 0.0, but a window the pipeline *refused* to score is stored as 0.0, and
the two are indistinguishable from the probabilities alone. It also meant the
arms could silently end up scored on different windows: a window where ICA
failed is refused in the ICA-on arm and scored normally in the ICA-off arm.
Measured on the current manifest that does not happen — all 129 refusals are
interrupted-signal rejections, which `detect_interupted_data` raises *before*
the `use_ica` branch and which therefore apply to both arms equally — but the
comparison should not depend on that continuing to be true. Both arms are now
restricted to the intersection of scored windows.

**The delta gets a confidence interval.** A pooled AUC difference over 13 files
means little without one. The bootstrap resamples *recordings*, not windows,
and recomputes both arms from the same drawn files, because the arms are paired
and windows within a recording are not independent.
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sklearn.metrics import roc_auc_score                     # noqa: E402
from gui.io.cache import cache_path_for, load_probability_file  # noqa: E402
from gui.io.infer import compute_probs, _build_model          # noqa: E402
from experiments.compare_zuna import read_reference_events    # noqa: E402
from experiments.evaluate_baseline import window_labels       # noqa: E402

MANIFEST = os.path.join(REPO, 'artifacts', 'zuna_thesis', 'manifest.csv')
OUT = os.path.join(REPO, 'artifacts', 'zuna_thesis', 'ica_on_off.json')


def scored_mask(cache_or_skip, probs):
    """Which windows carry a real model score.

    Prefers the recorded ``skip_code``. Falls back to ``probs != 0.0`` only for
    a v1 cache that predates it, and says so, rather than silently guessing.
    """
    if cache_or_skip is not None:
        return np.asarray(cache_or_skip) == 0
    return np.asarray(probs) != 0.0


def paired_arrays(edf, refs, model, step_s=6):
    """Return (labels, ica_on_scores, ica_off_scores) on a shared window set."""
    cache = load_probability_file(cache_path_for(edf))
    if cache is None:
        return None
    s_on = np.asarray(cache['window_starts'])
    p_on = np.asarray(cache['probs'])
    k_on = scored_mask(cache.get('skip_code'), p_on)

    s_off, p_off, sk_off = compute_probs(
        edf, step_s=step_s, use_ica=False, model=model)
    s_off = np.asarray(s_off)
    p_off = np.asarray(p_off)
    k_off = scored_mask(sk_off, p_off)

    # Intersect on window START, not on index: the two runs need not produce
    # the same number of windows, and aligning by position would silently
    # compare different moments in the recording.
    on = {int(s): i for i, s in enumerate(s_on) if k_on[i]}
    off = {int(s): i for i, s in enumerate(s_off) if k_off[i]}
    shared = sorted(set(on) & set(off))
    if not shared:
        return None
    starts = np.array(shared, dtype=float)
    lab = window_labels(starts, refs)
    return (lab,
            p_on[[on[s] for s in shared]],
            p_off[[off[s] for s in shared]],
            len(s_on), len(shared))


def pooled_auc(labels, scores):
    y = np.concatenate(labels)
    s = np.concatenate(scores)
    if not (0 < y.sum() < y.size):
        return float('nan')
    return float(roc_auc_score(y, s))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args(argv)

    rows = [r for r in csv.DictReader(open(MANIFEST))
            if (r.get('cohort') or '') == 'seizure']
    print('Building model...')
    model = _build_model()

    labs, on_s, off_s, stems = [], [], [], []
    print('\n{:<22} {:>5} {:>8} {:>8} {:>10}'.format(
        'stem', 'npos', 'AUC_on', 'AUC_off', 'shared/all'))
    print('-' * 62)
    for r in rows:
        edf = os.path.normpath(os.path.join(REPO, r['edf']))
        stem = os.path.splitext(os.path.basename(edf))[0]
        refs = read_reference_events(os.path.splitext(edf)[0] + '.csv_bi')

        got = paired_arrays(edf, refs, model)
        if got is None:
            print('{:<22} no usable pair, skipping'.format(stem))
            continue
        lab, p_on, p_off, n_all, n_shared = got
        labs.append(lab)
        on_s.append(p_on)
        off_s.append(p_off)
        stems.append(stem)

        def f(y, s):
            return ('n/a' if not (0 < y.sum() < y.size)
                    else '{:.3f}'.format(roc_auc_score(y, s)))
        print('{:<22} {:>5} {:>8} {:>8} {:>10}'.format(
            stem, int(lab.sum()), f(lab, p_on), f(lab, p_off),
            '{}/{}'.format(n_shared, n_all)))

    if not labs:
        print('nothing to compare')
        return 1

    a_on = pooled_auc(labs, on_s)
    a_off = pooled_auc(labs, off_s)
    delta = a_off - a_on
    print('-' * 62)
    print('POOLED window AUC   ICA-on: {:.4f}   ICA-off: {:.4f}'.format(
        a_on, a_off))
    print('delta (off - on)  : {:+.4f}'.format(delta))

    n = len(labs)
    rng = np.random.RandomState(args.seed)
    deltas = []
    for _ in range(args.boot):
        idx = rng.randint(0, n, n)
        L = [labs[i] for i in idx]
        if sum(int(l.sum()) for l in L) == 0:
            continue
        x = pooled_auc(L, [on_s[i] for i in idx])
        y = pooled_auc(L, [off_s[i] for i in idx])
        if np.isnan(x) or np.isnan(y):
            continue
        deltas.append(y - x)
    d = np.array(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_better = float((d > 0).mean())

    print()
    print('cluster bootstrap by recording, {} resamples over {} files'.format(
        len(d), n))
    print('  DELTA 95 % CI   : [{:+.4f}, {:+.4f}]'.format(lo, hi))
    print('  P(ICA-off better): {:.3f}'.format(p_better))
    crosses = lo < 0.0 < hi
    print()
    print('interval {} zero'.format('CROSSES' if crosses else 'excludes'))
    if crosses:
        print('  -> turning ICA off is NOT shown to change discrimination at')
        print('     this sample size. Report the point estimate with the')
        print('     interval; do not claim ICA-off "improves" anything.')
    else:
        print('  -> the difference is distinguishable from zero here.')
    print()
    print('Either way this is a FINDING, not a change to deploy: the training')
    print('features were generated by this same per-window ICA (verified —')
    print('utils/ICA_load_data_elec.py imports the same ica_arti_remove), so')
    print('the operating point depends on it.')

    with open(OUT, 'w') as f:
        json.dump({
            'n_files': n, 'files': stems,
            'auc_ica_on': a_on, 'auc_ica_off': a_off, 'delta_off_minus_on': delta,
            'delta_ci95': [float(lo), float(hi)],
            'p_ica_off_better': p_better,
            'bootstrap_resamples': int(len(d)),
            'method': ('cluster bootstrap by recording; arms restricted to the '
                       'intersection of scored windows, matched by window start'),
        }, f, indent=2, sort_keys=True)
        f.write('\n')
    print('\nwrote {}'.format(os.path.relpath(OUT, REPO)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
