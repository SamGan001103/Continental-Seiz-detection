"""Assemble the USB stick that carries the application to a review machine.

    python packaging/make_usb.py --dest E:/SeizureReview

Run it once per platform, pointing at the same USB. It adds the build for the
platform it is run on and leaves the others alone, so the stick accumulates:

    SeizureReview/
        START_HERE.txt          plain text, readable on any machine
        windows/                built on Windows
        macos/                  built on the Mac
        linux/                  built on Linux
        source/                 the repository, to build any platform still missing
        docs/                   deployment, intended use, results
        CHECKSUMS.txt           SHA-256 of every executable and of the weights

**PyInstaller does not cross-compile.** There is no way to produce the macOS
build from Windows. Each platform's folder has to be made on that platform,
which is why this script is additive rather than all-at-once, and why it writes
a clear placeholder into any slot it cannot fill.
"""
from __future__ import print_function

import argparse
import hashlib
import os
import platform
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PLATFORM_DIRS = ('windows', 'macos', 'linux')

# Copied so the stick is self-sufficient: someone holding it can build a
# missing platform without the network.
SOURCE_ITEMS = [
    'gui', 'models', 'utils', 'experiments', 'tests', 'packaging',
    'eval_config.py', 'precompute_probs.py', 'run_inference.py',
    'convlstm_ICA_12_train.h5',
    'requirements-modern.txt', 'requirements-seiz36.txt',
    'artifacts/zuna_thesis/manifest.csv', 'artifacts/zuna_thesis/manifest_full.csv',
    'environment-seiz36.yml', 'setup.bat', 'launch_gui.bat',
    'README.md', 'INSTALL.md',
]
DOC_ITEMS = [
    'DEPLOYMENT.md', 'INTENDED_USE.md', 'RESULTS.md', 'portability.md',
    'source_verification.md', 'BUILD_ON_MAC.md',
]


def current_platform():
    s = platform.system().lower()
    if s.startswith('win'):
        return 'windows'
    if s == 'darwin':
        return 'macos'
    return 'linux'


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


START_HERE = """SEIZURE REVIEW - START HERE
===========================================================================

Research prototype for human-in-the-loop EEG seizure review.
NOT a medical device. Read docs/INTENDED_USE.md before using it on any
recording. Every proposed event must be checked by a qualified reviewer.

---------------------------------------------------------------------------
WINDOWS
---------------------------------------------------------------------------
1. Copy the whole "windows\\SeizureReview" folder off this USB, onto the PC.
   Somewhere the logged-in user can write - Documents is fine.
   Do NOT run it from the USB: it is slow, and the app needs to write.
2. Open the copied folder and double-click SeizureReview.exe
3. Windows will say "Windows protected your PC" because the app is not
   code-signed. Click "More info" then "Run anyway". This is expected.

---------------------------------------------------------------------------
macOS (Apple Silicon)
---------------------------------------------------------------------------
1. Copy the whole "macos/SeizureReview" folder off this USB, onto the Mac.
2. RIGHT-CLICK SeizureReview and choose "Open" - do not double-click.
   Choose "Open" again in the dialog. The app is not notarised, and
   double-clicking gives a dead end with no "open anyway" button.
   You only need to do this once.
3. If macOS still refuses:
       xattr -dr com.apple.quarantine /path/to/SeizureReview

---------------------------------------------------------------------------
LINUX
---------------------------------------------------------------------------
1. Copy the whole "linux/SeizureReview" folder off this USB.
2. chmod +x SeizureReview
3. ./SeizureReview

---------------------------------------------------------------------------
IF A PLATFORM FOLDER IS MISSING OR SAYS "NOT BUILT"
---------------------------------------------------------------------------
The application cannot be built for one operating system from another.
Each build must be made on that kind of machine. The full source is in
"source/" - see source/docs/portability.md, then:

    conda create -n seizmodern python=3.11
    conda activate seizmodern
    pip install -r requirements-modern.txt pyinstaller
    bash packaging/build_app.sh          (macOS / Linux)
    packaging\\build_app.bat             (Windows)

---------------------------------------------------------------------------
FIRST RUN
---------------------------------------------------------------------------
* You need EEG recordings in EDF format with all 19 channels of the
  standard 10-20 montage. None are included on this USB.
* The first time a recording is opened it is scored, which takes a few
  minutes. After that it is instant, because the result is cached.
* Nothing is sent anywhere. All computation is local. No network is used.

Problems: %LOCALAPPDATA%\\SeizureReview\\logs\\seizure_review.log   (Windows)
          ~/Library/Application Support/SeizureReview/logs/         (macOS)
          ~/.local/share/SeizureReview/logs/                        (Linux)
"""

NOT_BUILT = """This platform has NOT been built yet.

The application cannot be cross-compiled: a {plat} build has to be produced
on a {plat} machine. The full source is in the "source" folder of this USB.

    conda create -n seizmodern python=3.11
    conda activate seizmodern
    pip install -r requirements-modern.txt pyinstaller
    {cmd}

Then re-run packaging/make_usb.py on that machine, pointing at this USB.
"""


def copy_tree(src, dst):
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    elif os.path.exists(src):
        d = os.path.dirname(dst)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        shutil.copy2(src, dst)
    else:
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', required=True, help='USB folder to write into')
    ap.add_argument('--dist', default=os.path.join(REPO, 'dist',
                                                   'SeizureReview'))
    ap.add_argument('--skip-source', action='store_true')
    args = ap.parse_args(argv)

    dest = os.path.abspath(args.dest)
    plat = current_platform()
    print('assembling USB at {}'.format(dest))
    print('this machine is: {}'.format(plat))

    if not os.path.isdir(dest):
        os.makedirs(dest)

    # 1 -- this platform's build ------------------------------------------
    if os.path.isdir(args.dist):
        target = os.path.join(dest, plat, 'SeizureReview')
        print('  copying the {} build (this takes a few minutes, ~1.2 GB)'
              .format(plat))
        copy_tree(args.dist, target)
        print('    done: {}'.format(target))
    else:
        print('  !! no build at {} — run the build script first'
              .format(args.dist))

    # 2 -- placeholders for platforms not yet built ------------------------
    for p in PLATFORM_DIRS:
        d = os.path.join(dest, p)
        if os.path.isdir(os.path.join(d, 'SeizureReview')):
            continue
        if not os.path.isdir(d):
            os.makedirs(d)
        cmd = ('packaging\\build_app.bat' if p == 'windows'
               else 'bash packaging/build_app.sh')
        with open(os.path.join(d, 'NOT_BUILT.txt'), 'w') as f:
            f.write(NOT_BUILT.format(plat=p, cmd=cmd))
        print('  {}: placeholder written (not built yet)'.format(p))

    # 3 -- source, so a missing platform can be built from the stick -------
    if not args.skip_source:
        print('  copying source')
        src_root = os.path.join(dest, 'source')
        for item in SOURCE_ITEMS:
            copy_tree(os.path.join(REPO, item), os.path.join(src_root, item))
        docs_dst = os.path.join(src_root, 'docs')
        if not os.path.isdir(docs_dst):
            os.makedirs(docs_dst)
        for doc in DOC_ITEMS:
            copy_tree(os.path.join(REPO, 'docs', doc),
                      os.path.join(docs_dst, doc))

    # 4 -- the documents a reviewer or installer actually reads ------------
    docs = os.path.join(dest, 'docs')
    if not os.path.isdir(docs):
        os.makedirs(docs)
    for doc in DOC_ITEMS:
        copy_tree(os.path.join(REPO, 'docs', doc), os.path.join(docs, doc))

    with open(os.path.join(dest, 'START_HERE.txt'), 'w') as f:
        f.write(START_HERE)

    # 5 -- checksums -------------------------------------------------------
    # A truncated copy produces an application that starts and then fails in an
    # unobvious place. This is the cheap way to rule that out.
    lines = ['SHA-256 checksums', '=' * 60, '']
    for p in PLATFORM_DIRS:
        for name in ('SeizureReview.exe', 'SeizureReview'):
            exe = os.path.join(dest, p, 'SeizureReview', name)
            if os.path.isfile(exe):
                lines.append('{:<10} {}  {}'.format(p, sha256(exe), name))
        w = os.path.join(dest, p, 'SeizureReview',
                         'convlstm_ICA_12_train.h5')
        if os.path.isfile(w):
            lines.append('{:<10} {}  weights'.format(p, sha256(w)))
    lines.append('')
    lines.append('Verify on Windows : certutil -hashfile <file> SHA256')
    lines.append('Verify on mac/Linux: shasum -a 256 <file>')
    with open(os.path.join(dest, 'CHECKSUMS.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print()
    print('USB ready. Contents:')
    for entry in sorted(os.listdir(dest)):
        print('   {}'.format(entry))
    built = [p for p in PLATFORM_DIRS
             if os.path.isdir(os.path.join(dest, p, 'SeizureReview'))]
    missing = [p for p in PLATFORM_DIRS if p not in built]
    print()
    print('built    : {}'.format(', '.join(built) or 'none'))
    if missing:
        print('still to build: {}'.format(', '.join(missing)))
        print('   run this script again on each of those machines.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
