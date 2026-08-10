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
    'source_verification.md', 'BUILD_ON_MAC.md', 'BUILD_ON_LINUX.md',
    'known_issues.md',
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

YOU DO NOT NEED TO INSTALL ANYTHING.

No Python. No downloads. No internet. No programming. No VS Code.
Everything the program needs is already inside the folder on this USB.

Copy a folder. Double-click it. That is the whole installation.

---------------------------------------------------------------------------
WHAT THIS IS
---------------------------------------------------------------------------
A research tool that reads an EEG recording, marks where it thinks a
seizure may be, and lets you accept or reject each one. The saved result
is YOUR annotation, not the computer's.

It is NOT a medical device and must not be used unsupervised. It misses
about half of all seizures, so a recording with no marks has NOT been
shown to be seizure-free. Please read docs/INTENDED_USE.md.

---------------------------------------------------------------------------
WINDOWS  -  three steps
---------------------------------------------------------------------------
1. Open the "windows" folder on this USB. Copy the "SeizureReview" folder
   into your Documents. (Copy the WHOLE folder, not just one file.)
   Wait for the copy to finish - it is large, about 1.2 GB.

2. Open the copied folder in Documents. Double-click SeizureReview

3. A blue box may say "Windows protected your PC".
   This is normal and expected. Click "More info", then "Run anyway".

   (It appears because the program has no purchased security
   certificate, not because anything is wrong with it.)

---------------------------------------------------------------------------
macOS  -  three steps
---------------------------------------------------------------------------
1. Open the "macos" folder on this USB. Copy the "SeizureReview" folder
   into your Documents. Wait for the copy to finish.

2. Open the copied folder. RIGHT-CLICK on SeizureReview and choose "Open".
   DO NOT double-click it the first time - double-clicking gives a message
   with no way to continue.

3. A box says the developer cannot be verified. Click "Open".
   You only ever have to do this once.

   If macOS still refuses, open the Terminal app and paste this line,
   then drag the SeizureReview folder onto the window and press Enter:
       xattr -dr com.apple.quarantine

---------------------------------------------------------------------------
LINUX
---------------------------------------------------------------------------
1. Copy the "linux/SeizureReview" folder off this USB.
2. In a terminal:   chmod +x SeizureReview
3. Run:             ./SeizureReview

---------------------------------------------------------------------------
USING IT
---------------------------------------------------------------------------
* Open your own EEG file: File > Open, and choose a .edf recording.
  No recordings are included on this USB.
  The recording must have all 19 standard electrodes.

* The FIRST time you open a recording it has to be analysed. This takes
  a few minutes and the window will show progress. Every time after that
  it opens instantly.

* Press F1 at any time for the list of keyboard shortcuts.
* Help > Intended use and limitations explains what it can and cannot do.

* Nothing leaves your computer. There is no internet connection, no
  upload, and no cloud. The recording stays on the machine you opened it on.

---------------------------------------------------------------------------
IF SOMETHING GOES WRONG
---------------------------------------------------------------------------
The program writes a log file. Send this file to whoever gave you the USB:

  Windows  %LOCALAPPDATA%\SeizureReview\logs\seizure_review.log
  macOS    ~/Library/Application Support/SeizureReview/logs/
  Linux    ~/.local/share/SeizureReview/logs/

Nothing happens when you double-click it?
  -> The folder was probably copied incompletely. Copy it again, in full.

---------------------------------------------------------------------------
FOR WHOEVER MAINTAINS THIS  (not needed by reviewers)
---------------------------------------------------------------------------
If a platform folder says NOT_BUILT, that platform has not been built yet.
The application cannot be built for one operating system from another.
Full source and instructions are in "source/" - see docs/BUILD_ON_MAC.md
and docs/portability.md.
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
