"""Ablation of the source method's decision stage on the cached probabilities.

Answers a question the thesis has to answer explicitly: how much of the gap
between this project's event-level numbers and the published ones is explained
by the decision stage rather than by the detector?

Four configurations, same probabilities, same reference annotations:

    raw          threshold the raw windows and merge whatever is adjacent
                 (what the GUI and the evaluation scripts did originally)
    shape        + concatenate events <MAX_MERGE_GAP_S apart, discard events
                 shorter than MIN_EVENT_DURATION_S  (the source method's rules)
    avg          + collapse overlapping windows to a per-second mean first
    avg+shape    both

Reads only the cached ``<edf>.probs.npz`` files, so it needs no TensorFlow and
re-runs no inference. Run in seiz36.

    python experiments/ablate_postprocessing.py \
        --manifest artifacts/zuna_thesis/manifest.csv
"""
from __future__ import print_function

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import eval_config as cfg  # noqa: E402
from experiments.evaluate_baseline import (  # noqa: E402
    event_sweep, iter_manifest_rows, load_file,
)

CONFIGS = [
    ('raw',       dict(postprocess=False, average=False)),
    ('shape',     dict(postprocess=True,  average=False)),
    ('avg',       dict(postprocess=False, average=True)),
    ('avg+shape', dict(postprocess=True,  average=True)),
]


def _sweep(records, thresholds, tolerance, postprocess, average):
    """event_sweep with the averaging flag threaded through summarize_run."""
    import experiments.compare_zuna as cz
    original = cz.summarize_run

    def patched(*a, **kw):
        kw['postprocess'] = postprocess
        kw['average'] = average
        return original(*a, **kw)

    # event_sweep imported summarize_run into evaluate_baseline's namespace.
    import experiments.evaluate_baseline as eb
    eb_original = eb.summarize_run
    eb.summarize_run = patched
    try:
        return event_sweep(records, thresholds, tolerance)
    finally:
        eb.summarize_run = eb_original


def _fmt(v, p=3):
    return 'n/a' if v is None else ('{:.' + str(p) + 'f}').format(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=cfg.THRESHOLD_GRID)
    ap.add_argument('--tolerance', type=float, default=cfg.EVENT_TOLERANCE_S)
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    records = []
    for edf_abs, cohort in iter_manifest_rows(os.path.abspath(args.manifest)):
        rec = load_file(edf_abs, cohort)
        if rec is not None:
            records.append(rec)
    if not records:
        raise SystemExit('No probability caches found.')

    print('{} files scored\n'.format(len(records)))
    payload = {'n_files': len(records), 'config': cfg.as_dict(),
               'configurations': {}}

    for name, opts in CONFIGS:
        sweep = _sweep(records, args.thresholds, args.tolerance, **opts)
        payload['configurations'][name] = sweep
        print('--- {} ---'.format(name))
        print('  {:>6}  {:>6}  {:>8}  {:>5}  {:>9}  {:>4}'.format(
            'thr', 'sens', 'hits', 'fp', 'fp/24h', 'dup'))
        for s in sweep:
            p = s['pooled']
            print('  {:>6}  {:>6}  {:>8}  {:>5}  {:>9}  {:>4}'.format(
                _fmt(s['threshold'], 2), _fmt(p['sensitivity'], 3),
                '{}/{}'.format(p['hits'], p['n_refs']),
                p['false_positives'], _fmt(p['false_alarms_per_24h'], 1),
                p['duplicate_detections']))
        print()

    thr = cfg.THRESHOLD
    print('=== summary at the canonical threshold {:.2f} ==='.format(thr))
    print('  {:<10} {:>8} {:>10} {:>8}'.format(
        'config', 'sens', 'fp/24h', 'dup'))
    for name, _ in CONFIGS:
        row = next((s['pooled'] for s in payload['configurations'][name]
                    if abs(s['threshold'] - thr) < 1e-9), None)
        if row:
            print('  {:<10} {:>8} {:>10} {:>8}'.format(
                name, _fmt(row['sensitivity'], 3),
                _fmt(row['false_alarms_per_24h'], 1),
                row['duplicate_detections']))

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
