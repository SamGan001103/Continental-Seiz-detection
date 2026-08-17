"""Load TensorFlow before Qt, on the platforms where the order matters.

Importing this module must be the FIRST thing `gui/main.py` does — before
`from PyQt5 import ...`. That is the whole point of it existing as a separate
module: an import at the top of a file is easy to reorder by accident, and a
module whose docstring says why is not.

The problem
-----------
On Windows with TensorFlow 2.x, importing PyQt5 first makes TensorFlow's native
library fail to initialise:

    ImportError: DLL load failed while importing _pywrap_tensorflow_internal:
                 A dynamic link library (DLL) initialization routine failed.

Reversing the order fixes it. Measured on this machine, modern stack:

    import tensorflow  ->  import PyQt5      works
    import PyQt5       ->  import tensorflow DLL failure
    import PyQt5.QtCore only, then tensorflow DLL failure

The last line matters: it is not QApplication, not the widgets, not the
offscreen platform plugin. Merely loading Qt's DLLs is enough.

Note "initialization routine failed", not "not found" — the library is located
and its DllMain then fails. The usual cause on Windows is exhaustion of the
per-process TLS slots available to implicitly loaded DLLs; PyQt5 ships 159 of
them and TensorFlow's native library is large. That mechanism is consistent with
the evidence but has not been proven here, so treat it as the likely explanation
rather than an established one. The ordering fix is what was actually measured.

Why this matters beyond a tidy import
-------------------------------------
This is `docs/known_issues.md` §1 — the modern stack "not freezing on Windows",
which cost six build attempts and was attributed to PyInstaller, to conda's DLL
layout, and to bundled-DLL shadowing in turn. It is none of those. It reproduces
from source, with no PyInstaller involved, through the application's own entry
point. The frozen build failed because the frozen build imports Qt first, exactly
as the source does.

The legacy Python 3.6 stack (TensorFlow 1.15) does NOT have this problem — Qt
first, then TF 1.15, loads fine — which is why the application that ships today
works and why the fault stayed hidden for so long.

Cost
----
TensorFlow takes a couple of seconds to import, and this moves that cost to
startup instead of to the first recording that needs scoring. That is a real
regression in launch time and a deliberate trade: an application that starts
two seconds later is better than one that cannot score anything.

It is skipped entirely where it is not needed — on macOS and Linux, and on the
legacy stack — so nothing changes for the builds that already work.
"""
import os
import sys

#: Set by :func:`preload` so the smoke test and the About box can report what
#: happened without repeating the platform logic.
STATUS = 'not attempted'


def _should_preload():
    """Only Windows, only on the modern stack.

    Deliberately does NOT import tensorflow to decide. The legacy Python 3.6
    build ships today, defers TensorFlow until a recording actually needs
    scoring, and does not have the conflict — importing TF 1.15 here to discover
    that would add several seconds to every launch of the one build that works.

    The Python version is the cheapest reliable discriminator: TensorFlow 2
    requires >= 3.9, so a 3.6 interpreter cannot be running the stack that has
    the problem. No import, no cost, and it fails safe — a future stack on 3.9+
    with TF 1 would preload needlessly, which is harmless.
    """
    if not sys.platform.startswith('win'):
        return False
    if sys.version_info < (3, 9):
        return False            # legacy 3.6 stack: TF 1.15, no conflict
    if os.environ.get('SEIZ_NO_TF_PRELOAD'):
        return False            # escape hatch for debugging startup time
    try:
        import importlib.util
        spec = importlib.util.find_spec('tensorflow')
    except Exception:                                   # noqa: BLE001
        return False
    return spec is not None


def preload():
    """Import TensorFlow now, if this platform needs it done before Qt.

    Never raises. A machine where TensorFlow cannot be imported at all is a
    machine where inference will fail later with a message the user can act on;
    failing here would only replace that with a crash before the window opens.
    """
    global STATUS
    if not _should_preload():
        STATUS = 'skipped (not needed on this platform/stack)'
        return STATUS
    try:
        import tensorflow as tf
        major = int(str(tf.__version__).split('.')[0])
        if major < 2:
            # TF 1.15 does not have the conflict; nothing was gained, but
            # nothing was harmed either.
            STATUS = 'loaded TensorFlow {} (preload unnecessary)'.format(
                tf.__version__)
        else:
            STATUS = 'loaded TensorFlow {} before Qt'.format(tf.__version__)
    except Exception as ex:                             # noqa: BLE001
        STATUS = 'TensorFlow preload failed: {}'.format(ex)
    return STATUS


# Import-time side effect, on purpose. A caller that has to remember to call
# preload() is a caller that will one day forget, and the failure mode is a
# frozen application that opens a window and cannot score anything.
preload()
