"""Build the zoom-out level for a recording, ahead of review.

    python precompute_level.py "D:/path/*.edf"

Writes `<stem>.level64.npz` beside each recording — about 26 MB for 24 hours.
The review application picks it up automatically when it opens the file, and
zooming out then slices that array instead of reading the span off disk.

Why this is a separate pass and not done on open
------------------------------------------------
It is one full read of the recording, measured at about a minute for a 24-hour
file. Zooming out without it was measured at 0.3-3.0 s per wheel notch, with
96-103% of that in the strided read — so the level is worth a lot, but paying
for it while a reviewer waits for a recording to appear would be a worse
regression than the lag it removes. Precompute it with the probabilities, on
whatever machine prepares the study, and the hospital PC just reads it.

Nothing here touches the detector or the model path: the level is display
geometry only, and the samples it serves are the recording's own.
"""
import argparse
import glob
import os
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def process(path, args):
    from gui.io import decimated_level as DL
    from gui.io.edf_source import open_source
    from gui.paths import params_path

    out = DL.path_for(path, args.stride)
    if os.path.exists(out) and not args.force:
        if DL.load(out, path) is not None:
            print('  skip     {} (current; --force to rebuild)'
                  .format(os.path.basename(out)))
            return None
        print('  stale    {} — rebuilding'.format(os.path.basename(out)))

    src = open_source(path, params_path())
    if src is None:
        print('  skip     {}: cannot open span-by-span'
              .format(os.path.basename(path)))
        return None
    try:
        fs = int(round(src.fs))
        hours = src.shape[1] / float(fs) / 3600.0
        name = os.path.basename(path)

        def report(frac):
            sys.stdout.write('\r  {:<30} {:5.1f}%'.format(name, 100 * frac))
            sys.stdout.flush()

        t0 = time.time()
        level = DL.build(src, fs, stride=args.stride, progress=report)
        elapsed = time.time() - t0
        sys.stdout.write('\r' + ' ' * 70 + '\r')
        level.save(out, path)
        print('  {:<30} {:5.1f} h  {:6.1f} MB  {:5.0f} s'
              .format(name, hours, level.nbytes / 1e6, elapsed))
        return {'hours': hours, 'bytes': level.nbytes, 'seconds': elapsed}
    finally:
        src.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('paths', nargs='+', help='EDF files, or a glob')
    p.add_argument('--stride', type=int, default=None,
                   help='samples between kept samples (default 64, ~4 Hz '
                        'from a 256 Hz recording)')
    p.add_argument('--force', action='store_true',
                   help='rebuild levels that already exist')
    args = p.parse_args(argv)

    from gui.io.decimated_level import DEFAULT_STRIDE
    if args.stride is None:
        args.stride = DEFAULT_STRIDE

    files = []
    for pattern in args.paths:
        hits = sorted(glob.glob(pattern))
        files.extend(hits if hits else [pattern])
    files = [f for f in files if os.path.exists(f)
             and f.lower().endswith('.edf')]
    if not files:
        print('no EDF files matched')
        return 1

    print('\nzoom-out levels, stride {}\n'.format(args.stride))
    done = [r for r in (process(f, args) for f in files) if r]
    if done:
        print('\n  {} recordings, {:.1f} h, {:.0f} MB total, {:.1f} min'
              .format(len(done), sum(r['hours'] for r in done),
                      sum(r['bytes'] for r in done) / 1e6,
                      sum(r['seconds'] for r in done) / 60.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
