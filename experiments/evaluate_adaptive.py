"""Does per-recording adaptive normalisation beat a global threshold?

This is the falsifiable question posed in `docs/RESULTS.md` §3b. Raising the
global threshold already reduces false alarms — 0.80 gives 66.4 FA/24 h at
43.5 % sensitivity against 204.4 at 48.2 % — so any adaptive scheme has to beat
*that curve*, not merely reduce false alarms.

The comparison is therefore **at matched sensitivity**, never at matched
threshold. A method that cuts alarms by detecting less has achieved nothing.

    python experiments/evaluate_adaptive.py --manifest artifacts/zuna_thesis/manifest_full.csv

Reads probability caches only; runs no inference.
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

import eval_config as cfg                                    # noqa: E402
from gui.adaptive import (adaptive_scale, adaptive_reference,  # noqa: E402
                          DEFAULT_PERCENTILE, DEFAULT_FLOOR, MIN_WINDOWS)
from gui.io.cache import cache_path_for, load_probability_file  # noqa: E402
from experiments.evaluate_baseline import (                   # noqa: E402
    iter_manifest_rows, load_file, event_sweep,
)

GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def build_records(manifest, percentile, floor, min_windows=MIN_WINDOWS):
    """Return (raw_records, adaptive_records, per-file reference levels)."""
    raw, adapt, refs = [], [], []
    for edf, cohort in iter_manifest_rows(os.path.abspath(manifest)):
        r = load_file(edf, cohort)
        if r is None or not r['has_reference']:
            continue
        c = load_probability_file(cache_path_for(edf))
        if c is None:
            continue
        starts = np.asarray(c['window_starts'])
        probs = np.asarray(c['probs'], dtype=float)
        skip = np.asarray(c['skip_code'])
        scaled, ref = adaptive_scale(probs, skip, percentile, floor,
                                     min_windows)

        base = {'stem': os.path.splitext(os.path.basename(edf))[0],
                'starts': starts, 'refs': r['refs'],
                'duration_s': r['duration_s'], 'cohort': cohort}
        raw.append(dict(base, probs=probs))
        adapt.append(dict(base, probs=scaled))
        refs.append((base['stem'], ref))
    return raw, adapt, refs


def curve(records, tolerance_s):
    out = []
    for row in event_sweep(records, GRID, tolerance_s, postprocess=True):
        p = row['pooled']
        if p['sensitivity'] is None or p['false_alarms_per_24h'] is None:
            continue
        out.append((row['threshold'], p['sensitivity'],
                    p['false_alarms_per_24h'], p['hits'], p['n_refs']))
    return out


def fa_at_sensitivity(pts, target):
    """Lowest false-alarm rate achieving at least ``target`` sensitivity.

    Comparing at matched sensitivity is the whole point: a method that reduces
    alarms by detecting fewer seizures has not improved anything.
    """
    ok = [p for p in pts if p[1] >= target - 1e-9]
    return min(ok, key=lambda p: p[2]) if ok else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest',
                    default='artifacts/zuna_thesis/manifest_full.csv')
    ap.add_argument('--percentile', type=float,
                    default=DEFAULT_PERCENTILE)
    ap.add_argument('--floor', type=float, default=DEFAULT_FLOOR)
    ap.add_argument('--min-windows', type=int, default=MIN_WINDOWS)
    ap.add_argument('--out', default=os.path.join(
        REPO, 'artifacts', 'zuna_thesis', 'baseline_eval', 'adaptive.json'))
    args = ap.parse_args(argv)

    raw, adapt, refs = build_records(args.manifest, args.percentile,
                                     args.floor, args.min_windows)
    print('recordings scored : {}'.format(len(raw)))
    noisy = [(s, r) for s, r in refs if r > args.floor + 1e-9]
    print('recordings tightened by the adaptive reference : {} ({:.0%})'
          .format(len(noisy), len(noisy) / float(max(1, len(refs)))))
    print('  the rest are left exactly as they are (reference floored)')
    if noisy:
        noisy.sort(key=lambda x: -x[1])
        print('  most-tightened:')
        for s, r in noisy[:5]:
            print('    {:<26} reference {:.3f}  -> scores x{:.3f}'.format(
                s, r, args.floor / r))

    c_raw = curve(raw, cfg.EVENT_TOLERANCE_S)
    c_ad = curve(adapt, cfg.EVENT_TOLERANCE_S)

    print('\n{:<10} {:>10} {:>12}   {:>10} {:>12}'.format(
        '', 'RAW sens', 'RAW FA/24h', 'ADAPT sens', 'ADAPT FA/24h'))
    print('-' * 60)
    for (t, s1, f1, h1, n1), (_t, s2, f2, h2, _n) in zip(c_raw, c_ad):
        print('thr {:<6.2f} {:>9.1%} {:>12.1f}   {:>9.1%} {:>12.1f}'.format(
            t, s1, f1, s2, f2))

    print('\nMATCHED-SENSITIVITY COMPARISON  (the question that matters)')
    print('{:<14} {:>12} {:>14} {:>12}'.format(
        'sensitivity', 'RAW FA/24h', 'ADAPTIVE FA/24h', 'change'))
    print('-' * 56)
    verdict = []
    for target in (0.25, 0.30, 0.35, 0.40, 0.45, 0.482):
        a = fa_at_sensitivity(c_raw, target)
        b = fa_at_sensitivity(c_ad, target)
        if a is None or b is None:
            continue
        delta = (b[2] - a[2]) / a[2] * 100.0 if a[2] else float('nan')
        verdict.append((target, a[2], b[2], delta))
        print('{:<14.1%} {:>12.1f} {:>14.1f} {:>11.1f}%'.format(
            target, a[2], b[2], delta))

    wins = sum(1 for _t, r, ad, _d in verdict if ad < r - 1e-9)
    print('\nadaptive is better at {} of {} matched-sensitivity points'.format(
        wins, len(verdict)))
    if wins == len(verdict) and verdict:
        print('-> per-recording normalisation dominates the global threshold')
    elif wins == 0:
        print('-> NO benefit over simply raising the global threshold.')
        print('   Report the negative result; do not ship it.')
    else:
        print('-> mixed; benefit depends on the operating point.')

    with open(args.out, 'w') as f:
        json.dump({
            'percentile': args.percentile, 'floor': args.floor,
            'n_files': len(raw), 'n_tightened': len(noisy),
            'raw_curve': c_raw, 'adaptive_curve': c_ad,
            'matched_sensitivity': verdict,
        }, f, indent=2, sort_keys=True, default=float)
        f.write('\n')
    print('\nwrote {}'.format(os.path.relpath(args.out, REPO)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
