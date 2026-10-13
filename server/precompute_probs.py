"""Precompute per-window seizure probabilities for one or more EDFs and
write them as `<edf>.probs.npz` next to each file. Run this in the
seiz36 env; the GUI loads the resulting cache and never imports TF."""
import argparse
import glob
import os
import sys
import tempfile
import time

# This file lives in server/, so the repository root is one level up.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import eval_config as cfg  # noqa: E402
from gui.io.infer import compute_probs
from gui.io.cache import save_probs, cache_path_for


# A spawned worker re-imports numpy, MNE and scikit-learn. Measured on this
# project's own run: about 172 MB resident each, but 1.7 GB of COMMIT each —
# and commit is what Windows runs out of first. Twelve workers reserved 20 GB
# on a 16 GB machine, which is how a scoring run makes a desktop unusable even
# while its resident footprint looks modest.
WORKER_COMMIT_BYTES = 1700 * 1024 * 1024
# What the parent needs for itself: the mapped signal's touched pages, the
# model, and TensorFlow.
PARENT_RESERVE_BYTES = 3 * 1024 * 1024 * 1024


def _workers_within_memory(requested):
    """Cap the worker count at what the machine can actually carry.

    Scoring is a background batch job. It has no deadline and no right to make
    the machine it runs on unusable — this one froze twice before the limits
    were understood. So the count asked for is an upper bound, not a promise.
    """
    if not requested:
        return requested
    try:
        from gui.memory import available_bytes
        avail = available_bytes()
    except Exception:                                        # noqa: BLE001
        return requested
    if not avail:
        return requested
    room = max(0, avail - PARENT_RESERVE_BYTES)
    allowed = max(1, int(room // WORKER_COMMIT_BYTES))
    if allowed < requested:
        print('  memory: {:.1f} GB free -> using {} worker(s), not {} '
              '(each reserves ~{:.1f} GB)'.format(
                  avail / 2 ** 30, allowed, requested,
                  WORKER_COMMIT_BYTES / 2 ** 30))
        return allowed
    return requested


def run(edf_path, step_s=cfg.STEP_S, use_ica=cfg.USE_ICA, overwrite=False,
        model=None, workers=0, batch_size=1):
    out = cache_path_for(edf_path)
    if os.path.exists(out) and not overwrite:
        print('  skip (cache exists):', out)
        return
    t0 = time.time()
    print('  computing...')
    # Load once, through the streaming loader, and hand the array to whichever
    # scoring path runs. `load_edf_19ch` reads every channel at float64 and
    # resamples the whole 2D array at once, which on a 22-hour recording peaks
    # at 9.0 GB and takes 136 s before any window is scored -- too heavy for
    # the hardware this is meant to run on. The streaming loader returns the
    # SAME BITS (tests/test_edf_stream.py) for a fraction of that.
    from gui.io.edf_stream import load_edf_19ch_streaming
    # Map the signal to a scratch file rather than holding it. On a 24-hour
    # recording the finished array is 1.5 GB and it was the single biggest
    # thing in this process (peak 3.1 GB). Mapped, the operating system pages
    # it and the peak is one channel plus whatever the scoring loop touches.
    scratch = os.path.join(tempfile.gettempdir(),
                           'seizreview_%d.npy' % os.getpid())
    data, fs, _dur = load_edf_19ch_streaming(edf_path, out_path=scratch)
    workers = _workers_within_memory(workers)
    if workers and use_ica:
        # Fan the per-window ICA across cores; predict() stays serial in this
        # process. Bit-identical to the serial path -- the guarantee, and the
        # test that holds it, live in gui/io/infer_parallel.py.
        from gui.io.infer_parallel import compute_probs_parallel
        starts, probs, skip_code = compute_probs_parallel(
            data, fs, step_s=step_s, workers=workers, model=model,
            progress_cb=_progress_line, batch_size=batch_size)
        print()
    else:
        starts, probs, skip_code = compute_probs(
            edf_path, step_s=step_s, use_ica=use_ica, model=model,
            data=data, fs=fs)
    del data
    try:
        os.remove(scratch)
    except OSError:
        pass
    # Stamp the ACTUAL settings used, not the canonical ones, so a cache built
    # with --no-ica or a non-default stride is self-describing and the
    # evaluation scripts can detect the mismatch.
    save_probs(edf_path, starts, probs, skip_code=skip_code, meta={
        'step_s': step_s,
        'segment_s': cfg.SEGMENT_S,
        'use_ica': use_ica,
        'fs': cfg.FS,
        'weights': cfg.WEIGHTS,
        # Recorded because a cache scored in batches is NOT bit-comparable to
        # one scored a window at a time, and a reader has no other way to
        # tell. 1 is the bit-identical default.
        'predict_batch_size': int(batch_size),
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
    ap.add_argument('--batch', type=int, default=1, metavar='N',
                    help='score N windows per predict() call instead of one. '
                         'Default 1, which is bit-identical to serial '
                         'scoring. Larger is faster where it matters most: '
                         'once --workers spreads the ICA, the serial '
                         'predict() is about three quarters of what is left '
                         '(42.3 ms/window at 1, 7.1 at 32, 5.0 at 48). The '
                         'cost is a batched matmul reducing in a different '
                         'order -- measured at most 1.3e-09 on p(seizure), '
                         'no decision changed at threshold 0.5. A cache built '
                         'with --batch is NOT bit-comparable to one without.')
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
                workers=max(0, args.workers),
                batch_size=max(1, args.batch))
        except Exception as ex:
            print('  FAILED:', ex)


if __name__ == '__main__':
    main()
