"""Precompute per-window seizure probabilities for one or more EDFs and
write them as `<edf>.probs.npz` next to each file. Run this in the
seiz36 env; the GUI loads the resulting cache and never imports TF."""
import argparse
import glob
import os
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from gui.io.infer import compute_probs
from gui.io.cache import save_probs, cache_path_for


def run(edf_path, step_s=6, use_ica=True, overwrite=False):
    out = cache_path_for(edf_path)
    if os.path.exists(out) and not overwrite:
        print('  skip (cache exists):', out)
        return
    t0 = time.time()
    print('  computing...')
    starts, probs = compute_probs(edf_path, step_s=step_s, use_ica=use_ica)
    save_probs(edf_path, starts, probs, meta={
        'step_s': step_s,
        'segment_s': 12,
        'use_ica': use_ica,
        'fs': 250,
        'weights': 'convlstm_ICA_12_train.h5',
    })
    print('  wrote {} ({:.1f}s; {} windows, max p={:.3f})'.format(
        out, time.time() - t0, len(probs), float(probs.max() if len(probs) else 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+',
                    help='EDF paths or glob patterns')
    ap.add_argument('--step', type=int, default=6,
                    help='stride in seconds between windows (default 6)')
    ap.add_argument('--no-ica', action='store_true',
                    help='skip ICA artifact removal (faster, weaker)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    targets = []
    for p in args.paths:
        expanded = glob.glob(p, recursive=True)
        targets.extend(expanded if expanded else [p])
    targets = [os.path.abspath(t) for t in targets if t.endswith('.edf')]
    print('Found {} EDF(s)'.format(len(targets)))
    for t in targets:
        print('>>', t)
        try:
            run(t, step_s=args.step,
                use_ica=not args.no_ica, overwrite=args.overwrite)
        except Exception as ex:
            print('  FAILED:', ex)


if __name__ == '__main__':
    main()
