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

import eval_config as cfg  # noqa: E402
from gui.io.infer import compute_probs
from gui.io.cache import save_probs, cache_path_for


def run(edf_path, step_s=cfg.STEP_S, use_ica=cfg.USE_ICA, overwrite=False,
        model=None, workers=0):
    out = cache_path_for(edf_path)
    if os.path.exists(out) and not overwrite:
        print('  skip (cache exists):', out)
        return
    t0 = time.time()
    print('  computing...')
    if workers and use_ica:
        # Fan the per-window ICA across cores; predict() stays serial in this
        # process. Bit-identical to the serial path -- the guarantee, and the
        # test that holds it, live in gui/io/infer_parallel.py.
        from gui.io.edf import load_edf_19ch
        from gui.io.infer_parallel import compute_probs_parallel
        data, fs, _dur = load_edf_19ch(edf_path)
        starts, probs, skip_code = compute_probs_parallel(
            data, fs, step_s=step_s, workers=workers, model=model,
            progress_cb=_progress_line)
        print()
    else:
        starts, probs, skip_code = compute_probs(
            edf_path, step_s=step_s, use_ica=use_ica, model=model)
    # Stamp the ACTUAL settings used, not the canonical ones, so a cache built
    # with --no-ica or a non-default stride is self-describing and the
    # evaluation scripts can detect the mismatch.
    save_probs(edf_path, starts, probs, skip_code=skip_code, meta={
        'step_s': step_s,
        'segment_s': cfg.SEGMENT_S,
        'use_ica': use_ica,
        'fs': cfg.FS,
        'weights': cfg.WEIGHTS,
    })
    print('  wrote {} ({:.1f}s; {} windows, max p={:.3f})'.format(
        out, time.time() - t0, len(probs), float(probs.max() if len(probs) else 0)))


def _progress_line(i, n):
    if i == n or i % 25 == 0:
        sys.stdout.write('\r  window %d/%d' % (i, n))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+',
                    help='EDF paths or glob patterns')
    ap.add_argument('--step', type=int, default=cfg.STEP_S,
                    help='stride in seconds between windows (default: eval_config)')
    ap.add_argument('--no-ica', action='store_true',
                    help='skip ICA artifact removal (faster, weaker)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--workers', type=int, default=0, metavar='N',
                    help='fan the per-window ICA across N processes WITHIN '
                         'each file. Bit-identical to serial scoring (the '
                         'window pipeline is seeded and windows are '
                         'independent; verified in tests/test_infer_parallel'
                         '.py). Use for single long recordings; use --shard '
                         'to spread many files across processes instead. '
                         '0 = serial (default).')
    ap.add_argument('--shard', default=None, metavar='I/N',
                    help='process only shard I of N (e.g. 3/8), so several '
                         'processes can cover disjoint files in parallel. '
                         'Each shard is an independent process on its own '
                         'files, so results are numerically identical to a '
                         'single-process run.')
    args = ap.parse_args()

    targets = []
    for p in args.paths:
        expanded = glob.glob(p, recursive=True)
        targets.extend(expanded if expanded else [p])
    targets = sorted(os.path.abspath(t) for t in targets
                     if t.endswith('.edf'))

    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        targets = [t for k, t in enumerate(targets) if k % n == i]
        print('shard {}/{}: {} EDF(s)'.format(i, n, len(targets)))
    else:
        print('Found {} EDF(s)'.format(len(targets)))

    # Build the model once per process rather than once per file: it costs
    # ~1.5 s and was previously rebuilt for every EDF.
    model = None
    if targets:
        from gui.io.infer import _build_model
        model = _build_model()

    for t in targets:
        print('>>', t)
        try:
            run(t, step_s=args.step, use_ica=not args.no_ica,
                overwrite=args.overwrite, model=model,
                workers=max(0, args.workers))
        except Exception as ex:
            print('  FAILED:', ex)


if __name__ == '__main__':
    main()
