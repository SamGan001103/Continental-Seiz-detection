# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the seizure-review GUI.

Produces a ONE-FOLDER build: dist/SeizureReview/ contains SeizureReview.exe and
everything it needs. Copy that folder to a Windows PC and double-click. No
Python, no conda, no installer, no administrator rights, no network.

    python -m PyInstaller packaging/SeizureReview.spec --noconfirm

Deliberate choices, each of which has a reason specific to deploying onto a
locked-down clinical workstation:

one-folder, not one-file
    A one-file build unpacks ~1 GB to %TEMP% on every launch — slow, and the
    first thing a hospital's endpoint protection flags. One-folder starts fast
    and looks like ordinary software.

UPX disabled
    UPX-compressed executables are heuristically detected as packed malware by
    most enterprise antivirus. Saving 200 MB is not worth failing the scan.

console=False, but every failure is written to a log file
    There is no terminal on the target machine to read a traceback from, so
    gui/main.py installs an excepthook that writes to the per-user data
    directory and shows the path in a dialog.
"""
import json
import os
import subprocess
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(SPEC)),
                                    '..'))
sys.path.insert(0, REPO)

block_cipher = None


# --------------------------------------------------------------------------
# Build stamp
# --------------------------------------------------------------------------
# The frozen app has no repository and the target machine has no git, so the
# commit under test is recorded here, at build time. Without it every
# annotation exported from the packaged application would carry a null build
# identifier — on precisely the machine where provenance matters most.
def _git(*argv):
    try:
        return subprocess.check_output(
            ('git',) + argv, cwd=REPO,
            stderr=subprocess.PIPE).decode('utf-8', 'replace').strip()
    except Exception:
        return ''


_commit = _git('rev-parse', '--short', 'HEAD')
if _commit and _git('status', '--porcelain'):
    _commit += '+dirty'

_build_dir = os.path.join(REPO, 'build', 'stamp')
if not os.path.exists(_build_dir):
    os.makedirs(_build_dir)
_stamp = os.path.join(_build_dir, 'build_info.json')
with open(_stamp, 'w') as _f:
    json.dump({
        'commit': _commit or None,
        'built_on': __import__('platform').node(),
        'python': sys.version.split()[0],
    }, _f, indent=2)
print('build stamp: commit={}'.format(_commit or '(unknown)'))

# --------------------------------------------------------------------------
# Data files
# --------------------------------------------------------------------------
# The layout inside the bundle must mirror the repository, because gui/paths.py
# resolves resources as resource('utils', 'params_common_electrodes.txt') etc.
datas = [
    (os.path.join(REPO, 'convlstm_ICA_12_train.h5'), '.'),
    (_stamp, '.'),
]

# Every montage/parameter file, not just the one the GUI loads today: they are
# a few kilobytes and a missing one is a runtime failure on the target machine.
for fn in sorted(os.listdir(os.path.join(REPO, 'utils'))):
    if fn.startswith('params_') and fn.endswith('.txt'):
        datas.append((os.path.join(REPO, 'utils', fn), 'utils'))

# Documentation the reviewer may need without a network connection.
for doc in ('README.md', 'INSTALL.md'):
    p = os.path.join(REPO, doc)
    if os.path.exists(p):
        datas.append((p, 'doc'))
for doc in ('RESULTS.md', 'INTENDED_USE.md', 'DEPLOYMENT.md'):
    p = os.path.join(REPO, 'docs', doc)
    if os.path.exists(p):
        datas.append((p, 'doc'))

# MNE ships montage definitions and layout files as package data; ICA fails at
# run time without them, and only on the packaged build.
datas += collect_data_files('mne')

# --------------------------------------------------------------------------
# Hidden imports
# --------------------------------------------------------------------------
# Modules imported inside functions, or resolved by string, are invisible to
# PyInstaller's static analysis. Each entry below corresponds to a real deferred
# import in this codebase.
hiddenimports = [
    'eval_config',
    'models',
    'models.deep_conv_lstm',
    'models.customCallbacks',
    'utils',
    'utils.pyst',
    'utils.preprocessing',
    'gui.io.infer',
    'gui.io.cache',
    'gui.postprocess',
    'stft',
    'stft.stft',
    'pyedflib',
]

# Keras 2.2.5 loads its backend by name at import time.
hiddenimports += collect_submodules('keras')
hiddenimports += ['keras.backend.tensorflow_backend']

# TensorFlow 1.15's Python layer is heavily lazy-imported.
hiddenimports += [
    'tensorflow',
    'tensorflow.python',
    'tensorflow.python.ops',
    'tensorflow.python.keras',
    'tensorflow.python.platform',
    'tensorflow_core',
]

# scipy/sklearn compiled helpers pulled in dynamically.
hiddenimports += [
    'scipy.special.cython_special',
    'scipy._lib.messagestream',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.tree._utils',
]

hiddenimports += collect_submodules('pyqtgraph')
hiddenimports += collect_submodules('mne')

# --------------------------------------------------------------------------
# Exclusions — pure size reduction, verified unused by the GUI
# --------------------------------------------------------------------------
excludes = [
    'matplotlib',        # experiments/thesis_figures.py only, never the GUI
    'pandas',
    'tkinter',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'nose',
    'sphinx',
    'PyQt5.QtWebEngineWidgets',
    'PyQt5.QtWebEngineCore',
    'PyQt5.QtBluetooth',
    'PyQt5.QtNfc',
    'PyQt5.QtQuick',
    'PyQt5.QtQml',
    'PyQt5.Qt3DCore',
    'PyQt5.QtMultimedia',
    'PyQt5.QtDesigner',
    'tensorflow.python.debug',
    'tensorboard',
]

a = Analysis(
    [os.path.join(REPO, 'gui', 'main.py')],
    pathex=[REPO],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SeizureReview',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # see module docstring: antivirus heuristics
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=(os.path.join(REPO, 'packaging', 'app.ico')
          if os.path.exists(os.path.join(REPO, 'packaging', 'app.ico'))
          else None),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SeizureReview',
)
