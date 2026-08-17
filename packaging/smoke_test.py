"""Verify a frozen build before it leaves the developer's machine.

A PyInstaller build fails in a way source code never does: it builds cleanly,
then dies on the target machine because a module was imported inside a function
and never got bundled. The only way to find that is to run the executable.

    python packaging/smoke_test.py [--dist dist/SeizureReview]

Checks, in order of how expensive they are to discover in the field:
  1. the executable and its bundled resources exist
  2. the bundled weights are byte-identical to the reviewed ones
  3. the app starts, builds the model, scores a recording, and exits 0

Step 3 runs the executable with --self-test, which exercises the whole
inference path (EDF read, montage, ICA, STFT, TensorFlow) without a display.
"""
from __future__ import print_function

import argparse
import hashlib
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import eval_config as cfg                                    # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def find_sample_edf():
    """The SHORTEST local recording; the point is that inference runs.

    Shortest, not first: the manifest is ordered by subject, and its first row
    is a 3337 s recording — roughly 278 windows and 21 minutes of inference,
    against a 900 s timeout. The gate would then fail for being slow rather
    than for being broken. Duration is already in the manifest, so preferring
    the short end costs nothing and keeps the gate to about a minute.
    """
    manifest = os.path.join(REPO, 'artifacts', 'zuna_thesis',
                            'manifest_full.csv')
    found = []
    if os.path.exists(manifest):
        import csv
        with open(manifest) as f:
            for row in csv.DictReader(f):
                p = os.path.join(REPO, row['edf'])
                if os.path.exists(p):
                    try:
                        dur = float(row.get('duration_s') or 0)
                    except ValueError:
                        dur = 0.0
                    found.append((dur, os.path.abspath(p)))
    if found:
        # A recording still has to be longer than one 12 s window to score
        # anything; below that the self-test reports FAIL for the wrong reason.
        usable = [t for t in found if t[0] >= 30.0] or found
        return min(usable)[1]
    for root, _dirs, files in os.walk(os.path.join(REPO, 'sample_data')):
        for fn in files:
            if fn.lower().endswith('.edf'):
                return os.path.join(root, fn)
    return None


def make_synthetic_edf(path, seconds=60, fs=250):
    """Write a 19-channel EDF of noise, for machines that hold no recordings.

    This exists because the check below must never be skipped. Gate 4 is the
    only step that executes the bundled import graph, and every packaging trap
    in `docs/known_issues.md` — the conda DLLs, matplotlib under MNE 1.x,
    `tensorflow.python.debug` under TF 2 — froze cleanly and died here. A build
    machine with no corpus is the normal case, not the exception, so skipping
    on missing data made the gate report success on exactly the builds it was
    written to catch.

    The scores from noise are meaningless and are not checked. What is checked
    is that the frozen binary reads an EDF, applies the montage, runs ICA and
    the STFT, loads TensorFlow and returns finite numbers — which is the whole
    of what this gate can establish even with a real recording.
    """
    import numpy as np
    import pyedflib

    rng = np.random.RandomState(0)
    n = seconds * fs
    t = np.arange(n) / float(fs)
    # Pure noise trips the "almost DC" guard in the preprocessing and most
    # windows get skipped, which weakens the check. A few microvolt-scale
    # oscillations in the EEG band keep the windows scorable.
    labels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
              'T3', 'C3', 'CZ', 'C4', 'T4',
              'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']
    sigs = []
    for i in range(len(labels)):
        x = 20.0 * np.sin(2 * np.pi * (8.0 + 0.3 * i) * t)
        x += 10.0 * np.sin(2 * np.pi * (1.0 + 0.1 * i) * t)
        x += 15.0 * rng.randn(n)
        sigs.append(x.astype(np.float64))

    # The two stacks want mutually exclusive spellings of the sample rate, and
    # supplying both does not satisfy either: pyedflib 0.1.22 (the Python 3.6
    # environment) reads ch['sample_rate'] and raises KeyError without it,
    # while 0.1.42 (the modern stack) raises FutureWarning if 'sample_rate' is
    # present *at all*, whether or not 'sample_frequency' accompanies it —
    # edfwriter.update_header, "raise exception, can be removed in later
    # release". Feature-detected by attempt rather than by parsing a version
    # string, so a future release that drops the legacy key needs no change
    # here. The modern spelling is tried first; only the 3.6 stack reaches the
    # retry.
    def _write(rate_key):
        w = pyedflib.EdfWriter(path, len(labels),
                               file_type=pyedflib.FILETYPE_EDFPLUS)
        try:
            w.setSignalHeaders([{
                'label': 'EEG {}-REF'.format(lab),  # TUH-style; montage is partial-match
                'dimension': 'uV',
                rate_key: fs,
                'physical_max': 5000.0,
                'physical_min': -5000.0,
                'digital_max': 32767,
                'digital_min': -32768,
                'transducer': '',
                'prefilter': '',
            } for lab in labels])
            w.writeSamples([s for s in sigs])
        finally:
            try:
                w.close()
            except Exception:              # noqa: BLE001 — already failing
                pass

    try:
        _write('sample_frequency')
    except (FutureWarning, KeyError, TypeError, ValueError):
        _write('sample_rate')
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dist', default=os.path.join(REPO, 'dist',
                                                   'SeizureReview'))
    ap.add_argument('--timeout', type=int, default=900)
    args = ap.parse_args(argv)

    dist = os.path.abspath(args.dist)
    # PyInstaller emits SeizureReview.exe on Windows and a bare executable on
    # macOS/Linux, so the smoke test cannot hardcode the extension.
    exe = os.path.join(dist, 'SeizureReview.exe')
    if not os.path.exists(exe):
        plain = os.path.join(dist, 'SeizureReview')
        if os.path.exists(plain) and not os.path.isdir(plain):
            exe = plain
    failures = []

    print('checking {}'.format(dist))

    # 1 -------------------------------------------------------------- layout
    # PyInstaller 6 moved bundled data into an _internal/ subdirectory; 4.x
    # put it beside the executable. Accept either, so the same check works
    # across the versions used on different platforms.
    roots = [dist, os.path.join(dist, '_internal')]

    def bundled(*parts):
        for r in roots:
            p = os.path.join(r, *parts)
            if os.path.exists(p):
                return p
        return os.path.join(dist, *parts)

    required = [
        exe,
        bundled('convlstm_ICA_12_train.h5'),
        bundled('utils', 'params_common_electrodes.txt'),
    ]
    for p in required:
        ok = os.path.exists(p)
        print('  [{}] {}'.format('ok' if ok else 'XX',
                                 os.path.relpath(p, dist)))
        if not ok:
            failures.append('missing: {}'.format(p))

    if not os.path.exists(exe):
        print('\nFAILED: no executable to test.')
        return 1

    # 2 ------------------------------------------------------------- weights
    weights = bundled('convlstm_ICA_12_train.h5')
    if os.path.exists(weights):
        got = sha256(weights)
        ok = (got == cfg.WEIGHTS_SHA256)
        print('  [{}] bundled weights hash {}'.format('ok' if ok else 'XX',
                                                      got[:16]))
        if not ok:
            failures.append('bundled weights do not match the reviewed model')

    # 3 ---------------------------------------------------------- Qt layer
    # Checked before inference because it is fast and because a broken widget
    # import is the more likely packaging failure of the two.
    env = dict(os.environ)
    env['QT_QPA_PLATFORM'] = 'offscreen'
    try:
        proc = subprocess.run([exe, '--gui-self-test'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              timeout=300, env=env, cwd=dist)
        out = proc.stdout.decode('utf-8', 'replace')
        if proc.returncode != 0 or 'gui-self-test: PASS' not in out:
            failures.append('GUI self-test failed')
            print(out[-4000:])
        else:
            for line in out.splitlines():
                if line.startswith('gui-self-test'):
                    print('      {}'.format(line))
            print('  [ok] frozen app builds its main window')
    except subprocess.TimeoutExpired:
        failures.append('GUI self-test timed out')

    # 4 ------------------------------------------------------------ run it
    # This step is never skipped. It used to print "[--] no local EDF found"
    # and pass, which meant a build machine without a corpus got a green smoke
    # test having never once executed the bundled TensorFlow — the single thing
    # this script exists to prove. Real data if there is any, synthetic noise
    # if there is not; only an unwritable temporary directory can stop it, and
    # that is reported as a failure rather than a skip.
    edf = find_sample_edf()
    synthetic = False
    tmpdir = None
    if edf is None:
        import tempfile
        try:
            tmpdir = tempfile.mkdtemp(prefix='seizreview-smoke-')
            edf = make_synthetic_edf(os.path.join(tmpdir, 'synthetic.edf'))
            synthetic = True
            print('  [--] no local recording; generated a synthetic one')
            print('       the import graph IS checked; the scores are not')
        except Exception as ex:                       # noqa: BLE001
            failures.append('could not obtain any EDF to test with: {}'
                            .format(ex))

    if edf is not None:
        print('  ... running inference through the frozen app on {}'.format(
            os.path.basename(edf)))
        print('      this loads TensorFlow and takes a few minutes')
        try:
            proc = subprocess.run(
                [exe, '--self-test', edf],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=args.timeout, env=env, cwd=dist)
            out = proc.stdout.decode('utf-8', 'replace')
            if proc.returncode != 0 or 'self-test: PASS' not in out:
                failures.append('self-test exited {}'.format(proc.returncode))
                print(out[-4000:])
            else:
                for line in out.splitlines():
                    if line.startswith('self-test'):
                        print('      {}'.format(line))
                print('  [ok] frozen app ran the full inference path on {}'
                      .format('synthetic data' if synthetic
                              else 'a real recording'))

                # Which Keras the FROZEN app selected, not which one the build
                # machine has. These differ, and the difference is silent.
                #
                # A Windows build shipped running Keras 3 while every gate here
                # passed and the spec had printed "tf_keras bundled (Keras 2
                # numerics preserved)". Both statements were true: tf_keras was
                # in the bundle, and it was never selected, because the runtime
                # hook imported TensorFlow before anything set
                # TF_USE_LEGACY_KERAS and `tf.keras` binds on first access.
                # The scores stayed plausible -- 0.0141 mean against 0.0113 --
                # so nothing downstream could have caught it either.
                #
                # Only meaningful on the modern stack. Under TF 1.15 there is
                # one Keras and the question does not arise.
                keras_line = ''
                for line in out.splitlines():
                    if line.startswith('self-test: keras'):
                        keras_line = line
                        break
                tf2 = 'self-test: keras' in out and 'unavailable' not in keras_line
                if tf2 and 'Keras 2' not in keras_line:
                    failures.append(
                        'the frozen app is running Keras 3, not Keras 2 -- it '
                        'will load the weights and return different numbers '
                        'with no error ({})'.format(keras_line.strip()))
                elif tf2:
                    print('  [ok] frozen app selected Keras 2 (numerics match '
                          'the published stack)')
        except subprocess.TimeoutExpired:
            failures.append('self-test timed out after {}s'.format(
                args.timeout))
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if failures:
        print('FAILED:')
        for f in failures:
            print('  - {}'.format(f))
        return 1
    print('Smoke test passed. dist/SeizureReview is safe to copy.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
