"""Find interictal discharge candidates in a recording, ahead of review.

    python precompute_ieds.py "D:/path/*.edf"
    python precompute_ieds.py recording.edf --threshold 0.9

Writes `<stem>.ied.csv` beside each recording, in the format
`gui.io.ied_marks` already uses for a reviewer's own marks, with every row
marked `source = spikenet1` so the two can never be added together.

Why this is a separate pass
--------------------------
It is cheap -- about 0.2 min of compute per hour of EEG, against roughly 54
min per 24 h for the seizure detector, because SpikeNet needs no ICA. On a
day-long recording that is around five minutes. Doing it ahead of review means
the reviewer opens a file with the candidates already on the strip.

WHAT THESE MARKS ARE NOT
------------------------
They are not findings. SpikeNet1's published performance is for classifying
candidate events that an earlier stage proposed, not for sliding across
continuous EEG, and no threshold has been validated on this project's data.
Every mark is a proposal for a human to confirm or reject. The file says so in
its header and every row names the detector it came from.
"""
import argparse
import glob
import os
import sys
import time
import warnings

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def process(path, model, args):
    import mne
    from gui.io import ied_detect as D
    from gui.io.edf import read_start_datetime
    from gui.io.ied_marks import ied_path_for, summarise, write_ied_marks

    out_path = ied_path_for(path)
    if os.path.exists(out_path) and not args.force:
        print('  skip     {} (exists; --force to redo)'
              .format(os.path.basename(out_path)))
        return None

    raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
    try:
        fs = float(raw.info['sfreq'])
        n = int(raw.n_times)
        try:
            idx = D.channel_index(raw.ch_names)
        except ValueError as exc:
            print('  skip     {}: {}'.format(os.path.basename(path), exc))
            return None

        def read(a, b):
            # Microvolts: the model's output is amplitude-sensitive, so the
            # 1e6 is not cosmetic.
            return raw.get_data(picks=idx, start=a, stop=b) * 1e6

        t0 = time.time()
        marks = D.detect_in_chunks(
            read, n, fs, chunk_s=args.chunk, threshold=args.threshold,
            step=args.step, notch_hz=args.notch, model=model,
            progress=_progress(os.path.basename(path)))
        elapsed = time.time() - t0
        sys.stdout.write('\r' + ' ' * 78 + '\r')

        write_ied_marks(out_path, os.path.splitext(os.path.basename(path))[0],
                        n / fs, marks, start_time=read_start_datetime(path),
                        source_path=path)
        hours = n / fs / 3600.0
        s = summarise(marks)
        print('  {:<28} {:5d} candidates  {:5.1f}/h  {:6.1f} h  '
              '{:5.1f} s compute'
              .format(os.path.basename(path), s['total'],
                      s['total'] / hours if hours else 0.0, hours, elapsed))
        return {'file': path, 'marks': s['total'], 'hours': hours,
                'seconds': elapsed}
    finally:
        raw.close()


def _progress(name):
    last = [0.0]

    def report(frac):
        if frac - last[0] < 0.02:
            return
        last[0] = frac
        sys.stdout.write('\r  {:<28} {:5.1f}%'.format(name, 100.0 * frac))
        sys.stdout.flush()
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('paths', nargs='+', help='EDF files, or a glob')
    p.add_argument('--threshold', type=float, default=None,
                   help='score above which a window is a candidate '
                        '(default: the module default)')
    p.add_argument('--step', type=int, default=None,
                   help='samples between windows at 128 Hz (16 = 8 per second)')
    p.add_argument('--chunk', type=float, default=300.0,
                   help='seconds held in memory at once (default 300)')
    p.add_argument('--notch', type=float, default=None,
                   help='mains frequency; 50 in AU/EU, 60 in North America')
    p.add_argument('--force', action='store_true',
                   help='rewrite files that already have marks')
    args = p.parse_args(argv)

    from gui.io import ied_detect as D
    if args.threshold is None:
        args.threshold = D.DEFAULT_THRESHOLD
    if args.step is None:
        args.step = D.DEFAULT_STEP
    if args.notch is None:
        args.notch = D.DEFAULT_NOTCH_HZ

    files = []
    for pattern in args.paths:
        hits = sorted(glob.glob(pattern))
        files.extend(hits if hits else [pattern])
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print('no files matched')
        return 1

    reason = D.missing_reason()
    if reason:
        print(reason)
        return 1

    print('\nSpikeNet1 interictal discharge candidates')
    print('  threshold {:.2f}, {} windows/s, {:.0f} s chunks, {:.0f} Hz notch'
          .format(args.threshold, D.FS // args.step, args.chunk, args.notch))
    print('  UNVALIDATED on this data: every mark is a candidate for review,\n'
          '  not a finding. See models/spikenet1/NOTICE.txt.\n')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = D.load_model()
        done = [r for r in (process(f, model, args) for f in files) if r]

    if done:
        h = sum(r['hours'] for r in done)
        s = sum(r['seconds'] for r in done)
        print('\n  {} recordings, {:.1f} h, {} candidates, {:.1f} min compute'
              .format(len(done), h, sum(r['marks'] for r in done), s / 60.0))
        if h:
            print('  {:.2f} min of compute per hour of EEG'
                  .format(s / 60.0 / h))
    return 0


if __name__ == '__main__':
    sys.exit(main())
