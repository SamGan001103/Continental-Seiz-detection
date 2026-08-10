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

# --untracked-files=no matters. Plain `git status --porcelain` also lists
# untracked files, and this repository permanently carries a large untracked
# thesis bundle — so every build was stamped "+dirty" regardless of the code,
# which makes the flag convey nothing. "+dirty" must mean "a TRACKED file
# differs from the commit", i.e. this build is not reproducible from the commit
# alone. The untracked count is recorded separately rather than discarded,
# because an untracked .py inside gui/ would genuinely end up in the bundle.
_dirty = bool(_git('status', '--porcelain', '--untracked-files=no'))
_untracked = len([l for l in _git('status', '--porcelain').splitlines()
                  if l.startswith('??')])
if _commit and _dirty:
    _commit += '+dirty'

_build_dir = os.path.join(REPO, 'build', 'stamp')
if not os.path.exists(_build_dir):
    os.makedirs(_build_dir)
_stamp = os.path.join(_build_dir, 'build_info.json')
with open(_stamp, 'w') as _f:
    json.dump({
        'commit': _commit or None,
        'tracked_tree_clean': not _dirty,
        'untracked_files_present': _untracked,
        'built_on': __import__('platform').node(),
        'python': sys.version.split()[0],
    }, _f, indent=2)
print('build stamp: commit={} tracked_clean={} untracked={}'.format(
    _commit or '(unknown)', not _dirty, _untracked))

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
    'utils',
    'utils.pyst',
    'utils.preprocessing',
    'utils.fastica_pinned',
    'gui.io.infer',
    'gui.io.cache',
    'gui.postprocess',
    'stft',
    'stft.stft',
    'pyedflib',
]

# The graph builder differs by runtime, so both are declared and the one that
# imports wins at run time (see gui/io/infer._build_model).
hiddenimports += ['models.convlstm_tf2']

# Which Keras/TensorFlow to pull in depends on what is installed. TF 1.15 with
# standalone Keras is the Windows build; TF 2 with tf.keras is what runs on
# Apple Silicon, where TF 1.15 has no wheel. Declaring TF 1 hidden imports on a
# TF 2 machine makes PyInstaller fail on modules that do not exist.
try:
    import tensorflow as _tf
    _TF_MAJOR = int(str(_tf.__version__).split('.')[0])
except Exception:
    _TF_MAJOR = 1
print('spec: building against TensorFlow major version {}'.format(_TF_MAJOR))

if _TF_MAJOR >= 2:
    hiddenimports += collect_submodules('tensorflow.python.keras')
    hiddenimports += ['tensorflow', 'tensorflow.python']
    # models/deep_conv_lstm.py imports keras.layers.core, which TF2 removed, so
    # declaring it here would only produce an unresolvable-import warning.
else:
    hiddenimports += ['models.deep_conv_lstm', 'models.customCallbacks']
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
hiddenimports += ['scipy.special.cython_special', 'scipy._lib.messagestream']
# Private sklearn modules that existed in 0.22 and were removed later. Declaring
# a module that does not exist only produces warnings, but a clean build log is
# worth more than the two lines saved.
for _m in ('sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs',
           'sklearn.tree._utils'):
    try:
        __import__(_m)
        hiddenimports.append(_m)
    except Exception:
        pass

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

# --------------------------------------------------------------------------
# conda runtime DLLs (Windows)
# --------------------------------------------------------------------------
# conda puts shared libraries in <env>/Library/bin, which PyInstaller does not
# scan. The one that matters is ffi-8.dll: _ctypes.pyd links against it, and
# without it the frozen application dies on `import ctypes`. scipy then reports
# "The scipy install you are using seems to be broken", which points at
# entirely the wrong library and costs an hour if you believe it.
binaries = []
if sys.platform.startswith('win'):
    _conda_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.isdir(_conda_bin):
        for _dll in ('ffi-8.dll', 'ffi-7.dll', 'ffi.dll',
                     'libcrypto-3-x64.dll', 'libssl-3-x64.dll'):
            _p = os.path.join(_conda_bin, _dll)
            if os.path.exists(_p):
                binaries.append((_p, '.'))
                print('spec: bundling conda DLL {}'.format(_dll))

a = Analysis(
    [os.path.join(REPO, 'gui', 'main.py')],
    pathex=[REPO],
    binaries=binaries,
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
    # .ico is Windows-only; PyInstaller rejects it on macOS/Linux.
    icon=(os.path.join(REPO, 'packaging', 'app.ico')
          if (sys.platform.startswith('win')
              and os.path.exists(os.path.join(REPO, 'packaging', 'app.ico')))
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
