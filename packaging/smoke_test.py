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
    """Any local recording will do; the point is that inference runs."""
    manifest = os.path.join(REPO, 'artifacts', 'zuna_thesis',
                            'manifest_full.csv')
    if os.path.exists(manifest):
        import csv
        with open(manifest) as f:
            for row in csv.DictReader(f):
                p = os.path.join(REPO, row['edf'])
                if os.path.exists(p):
                    return os.path.abspath(p)
    for root, _dirs, files in os.walk(os.path.join(REPO, 'sample_data')):
        for fn in files:
            if fn.lower().endswith('.edf'):
                return os.path.join(root, fn)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dist', default=os.path.join(REPO, 'dist',
                                                   'SeizureReview'))
    ap.add_argument('--timeout', type=int, default=900)
    args = ap.parse_args(argv)

    dist = os.path.abspath(args.dist)
    exe = os.path.join(dist, 'SeizureReview.exe')
    failures = []

    print('checking {}'.format(dist))

    # 1 -------------------------------------------------------------- layout
    required = [
        exe,
        os.path.join(dist, 'convlstm_ICA_12_train.h5'),
        os.path.join(dist, 'utils', 'params_common_electrodes.txt'),
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
    bundled = os.path.join(dist, 'convlstm_ICA_12_train.h5')
    if os.path.exists(bundled):
        got = sha256(bundled)
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
    edf = find_sample_edf()
    if edf is None:
        print('  [--] no local EDF found; skipping the inference run')
        print('       (layout was checked, but the import graph was not)')
    else:
        print('  ... running inference through the frozen app on {}'.format(
            os.path.basename(edf)))
        print('      this loads TensorFlow and takes a few minutes')
        try:
            proc = subprocess.run(
                [exe, '--self-test', edf],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=args.timeout, env=env, cwd=dist)
            out = proc.stdout.decode('utf-8', 'replace')
            if proc.returncode != 0:
                failures.append('self-test exited {}'.format(proc.returncode))
                print(out[-4000:])
            else:
                for line in out.splitlines():
                    if line.startswith('self-test'):
                        print('      {}'.format(line))
                print('  [ok] frozen app scored a real recording')
        except subprocess.TimeoutExpired:
            failures.append('self-test timed out after {}s'.format(
                args.timeout))

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
