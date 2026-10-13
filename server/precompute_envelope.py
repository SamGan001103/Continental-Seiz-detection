"""Build the overview envelope for a recording, on the server.

    python server/precompute_envelope.py "D:/studies/*.edf"

The review application draws a zoomed-out view from a min/max envelope of the
whole filtered recording. Computing one costs about 48 seconds on a 24-hour
file, and a filter change costs it again -- measured, and it was the whole of
the lag a reviewer met on the clinical workstation.

It is deterministic given the recording and the display settings, and bounded
whatever the duration, so it belongs in the study bundle with everything else
the server prepares. After this runs, the first zoom out is a file read.

One envelope per display preset. The application falls back to building
locally for any combination the bundle does not carry, so adding presets here
trades disk for responsiveness and never changes what is drawn.
"""
import argparse
import glob
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# The settings the application starts in. A reviewer who never touches the
# filter boxes -- which is most of them, most of the time -- pays nothing.
DEFAULT_PRESETS = [
    {'hp': 1.0, 'lp': 70.0, 'notch': 60.0, 'montage': 'Longitudinal Bipolar'},
]


def build(recording, presets, force=False):
    from gui.display_pipeline import DisplayPipeline
    from gui.io.edf import read_start_datetime
    from gui.io.edf_annotations import read_edf_annotations
    from gui.io.edf_source import open_source
    from gui.io.envelope_cache import load, path_for, save, settings_key
    from gui.paths import params_path

    read_edf_annotations(recording)
    read_start_datetime(recording)
    src = open_source(recording, params_path())
    if src is None:
        print('  skip     {}: cannot open span-by-span'
              .format(os.path.basename(recording)))
        return None
    try:
        fs = int(round(src.fs))
        hours = src.shape[1] / float(fs) / 3600.0
        rows = ['EEG {}'.format(i) for i in range(src.shape[0])]
        made = 0
        for preset in presets:
            key = settings_key(preset['hp'], preset['lp'], preset['notch'],
                               preset['montage'], fs)
            if not force and load(recording, key) is not None:
                print('  have     {}  {}'.format(
                    os.path.basename(recording), _describe(preset)))
                continue
            pipe = DisplayPipeline(src, fs, preset['hp'], preset['lp'],
                                   preset['notch'], preset['montage'], rows)
            t0 = time.time()
            k, step, lo, hi = pipe.build_envelope_for_cache()
            secs = time.time() - t0
            out = save(recording, settings_key(*k), step, lo, hi)
            mb = (lo.nbytes + hi.nbytes) / 1e6
            print('  {:<26} {:5.1f} h  {}  {:6.1f} MB  {:5.0f} s'
                  .format(os.path.basename(recording), hours,
                          _describe(preset), mb, secs))
            made += 1
        return {'hours': hours, 'made': made}
    finally:
        src.close()


def _describe(p):
    return 'hp{:g} lp{:g} notch{:g}'.format(p['hp'], p['lp'], p['notch'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('paths', nargs='+', help='EDF recordings, or a glob')
    ap.add_argument('--force', action='store_true',
                    help='rebuild envelopes that already exist')
    args = ap.parse_args(argv)

    files = []
    for pattern in args.paths:
        hits = sorted(glob.glob(pattern))
        files.extend(hits if hits else [pattern])
    files = [f for f in files
             if os.path.exists(f) and f.lower().endswith('.edf')]
    if not files:
        print('no EDF recordings matched')
        return 1

    print('\nOverview envelopes for {} recording(s)'.format(len(files)))
    print('  presets: ' + ', '.join(_describe(p) for p in DEFAULT_PRESETS))
    print()
    t0 = time.time()
    done = [r for r in (build(f, DEFAULT_PRESETS, args.force) for f in files)
            if r]
    if done:
        print('\n  {} recording(s), {:.1f} h, {} envelope(s), {:.1f} min'
              .format(len(done), sum(r['hours'] for r in done),
                      sum(r['made'] for r in done), (time.time() - t0) / 60.0))
    return 0


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    sys.exit(main())
