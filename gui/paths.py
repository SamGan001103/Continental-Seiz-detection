"""Where the application's own files live, in source and when frozen.

A PyInstaller build has no repository layout: modules are imported from an
archive and data files are unpacked to a temporary directory named by
``sys._MEIPASS``. Anything that located a resource by walking up from
``__file__``, or by ``os.chdir``-ing into ``utils/`` and opening a bare relative
filename, therefore breaks the moment the app is packaged — and breaks at run
time, on a hospital PC, not at build time.

Every path to a bundled resource should go through :func:`resource`.
"""
import os
import sys


def is_frozen():
    return bool(getattr(sys, 'frozen', False))


def app_root():
    """Directory containing bundled resources.

    Frozen: PyInstaller's extraction directory (``sys._MEIPASS``), falling back
    to the directory holding the executable for a one-folder build that was
    started without onefile extraction.
    Source: the repository root.
    """
    if is_frozen():
        return getattr(sys, '_MEIPASS',
                       os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def resource(*parts):
    """Absolute path to a bundled file, e.g. resource('utils', 'params.txt')."""
    return os.path.join(app_root(), *parts)


def writable_root():
    """Where the app may write caches, exports and logs.

    Never the bundle, for two independent reasons. In a one-file build
    ``sys._MEIPASS`` is a temporary directory deleted when the process exits,
    so anything written there is destroyed on close. In the one-folder build we
    actually ship, it is the application directory — which on a clinical
    workstation is frequently a read-only share or a managed software location.
    Either way the bundle is the wrong place, so this never depends on which
    build shape is in use.
    """
    if is_frozen():
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
        d = os.path.join(base, 'SeizureReview')
        try:
            if not os.path.exists(d):
                os.makedirs(d)
            return d
        except OSError:
            return os.path.expanduser('~')
    return app_root()


def weights_path():
    return resource('convlstm_ICA_12_train.h5')


def params_path(name='params_common_electrodes.txt'):
    return resource('utils', name)
