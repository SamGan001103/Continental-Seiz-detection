"""Build the overview envelopes for a recording, on the server.

    python server/precompute_envelope.py "D:/studies/*.edf"

The review application draws a zoomed-out view from a min/max envelope of the
whole filtered recording. Computing one costs about 40 seconds on a 24-hour
file, it is paid again whenever the filters or the montage change, and it
happens on the GUI thread -- so what a reviewer sees is the window going white
and Windows marking it Not Responding. Measured, and it was the whole of the
lag on the clinical workstation.

An envelope is deterministic given the recording and the display settings, and
bounded whatever the duration, so it belongs in the study bundle with
everything else the server prepares. After this runs, a zoom out is a file
read.

Which presets
-------------
The application offers four montages and six choices each of high-pass and
low-pass, so 432 combinations exist and shipping them all would be ~15 GB per
recording-day. It ships the ones a reviewer actually reaches for: the default
settings, the same filters in each of the other three montages, and the two
standard filter changes. Six files, about 220 MB per recording-day.

Anything else still builds locally -- slower, never wrong. Adding presets here
trades disk for responsiveness and cannot change what is drawn.
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
# This one is first because it is the envelope the bundle manifest names.
DEFAULT_PRESET = {'hp': 1.0, 'lp': 70.0, 'montage': 'Longitudinal Bipolar'}

# Then: the same filters in the other montages, because re-montaging is the
# commonest thing a reviewer does to a suspicious stretch; and the two filter
# changes that are part of routine reading -- 0.5 Hz to see slow activity,
# 35 Hz to put muscle artefact down.
EXTRA_PRESETS = [
    {'hp': 1.0, 'lp': 70.0, 'montage': 'Referential (raw)'},
    {'hp': 1.0, 'lp': 70.0, 'montage': 'Common Average'},
    {'hp': 1.0, 'lp': 70.0, 'montage': 'Transverse Bipolar'},
    {'hp': 0.5, 'lp': 70.0, 'montage': 'Longitudinal Bipolar'},
    {'hp': 1.0, 'lp': 35.0, 'montage': 'Longitudinal Bipolar'},
]


def presets(notch=60.0, default_only=False):
    """The presets to prepare, at this site's mains frequency.

    The notch is not a reviewer's choice so much as a fact about the building,
    so it is a setting here rather than another axis to multiply out.
    """
    chosen = [DEFAULT_PRESET] if default_only \
        else [DEFAULT_PRESET] + EXTRA_PRESETS
    return [dict(p, notch=notch) for p in chosen]


def build(recording, wanted, force=False):
    from gui.display_pipeline import DisplayPipeline
    from gui.io.edf import read_start_datetime
    from gui.io.edf_annotations import read_edf_annotations
    from gui.io.edf_source import open_source
    from gui.io.envelope_cache import has, save, settings_key
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
        made, bytes_written = 0, 0
        print('  {:<24} {:5.1f} h'.format(
            os.path.basename(recording), hours))
        for preset in wanted:
            key = settings_key(preset['hp'], preset['lp'], preset['notch'],
                               preset['montage'], fs)
            if not force and has(recording, key):
                print('      have  {}'.format(_describe(preset)))
                continue
            pipe = DisplayPipeline(src, fs, preset['hp'], preset['lp'],
                                   preset['notch'], preset['montage'], rows)
            t0 = time.time()
            k, step, lo, hi = pipe.build_envelope_for_cache()
            secs = time.time() - t0
            out = save(recording, settings_key(*k), step, lo, hi)
            mb = os.path.getsize(out) / 1e6
            print('      built {:<34} {:6.1f} MB  {:5.0f} s'
                  .format(_describe(preset), mb, secs))
            made += 1
            bytes_written += os.path.getsize(out)
        return {'hours': hours, 'made': made, 'bytes': bytes_written}
    finally:
        src.close()


def _describe(p):
    return 'hp{:g} lp{:g} notch{:g} {}'.format(
        p['hp'], p['lp'], p['notch'], p['montage'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('paths', nargs='+', help='EDF recordings, or a glob')
    ap.add_argument('--force', action='store_true',
                    help='rebuild envelopes that already exist')
    ap.add_argument('--notch', type=float, default=60.0,
                    help='mains frequency at this site (default 60)')
    ap.add_argument('--default-only', action='store_true',
                    help='prepare only the default display preset')
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

    wanted = presets(args.notch, args.default_only)
    print('\nOverview envelopes for {} recording(s), {} preset(s) each'
          .format(len(files), len(wanted)))
    for p in wanted:
        print('    {}'.format(_describe(p)))
    print()
    t0 = time.time()
    done = [r for r in (build(f, wanted, args.force) for f in files) if r]
    if done:
        print('\n  {} recording(s), {:.1f} h, {} envelope(s), {:.0f} MB, '
              '{:.1f} min'
              .format(len(done), sum(r['hours'] for r in done),
                      sum(r['made'] for r in done),
                      sum(r['bytes'] for r in done) / 1e6,
                      (time.time() - t0) / 60.0))
    return 0


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    sys.exit(main())
