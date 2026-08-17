"""PyInstaller runtime hook: import TensorFlow before Qt.

Runtime hooks execute before the frozen application's entry script, and
PyInstaller ships one of its own — `pyi_rth_pyqt5.py` — which imports PyQt5 to
point Qt at the bundled plugin directory. That is why the fix in `gui/main.py`
does not carry into a frozen build: by the time the entry script runs, Qt's DLLs
are already loaded, and on Windows with TensorFlow 2 that makes TensorFlow's
native library fail to initialise:

    ImportError: DLL load failed while importing _pywrap_tensorflow_internal:
                 A dynamic link library (DLL) initialization routine failed.

That failure is `docs/known_issues.md` §1. It cost six build attempts and was
attributed to PyInstaller, to conda's DLL layout, and to bundled-DLL shadowing
in turn. It is an import order, and the frozen build fails for the same reason
the source did — with the additional wrinkle that the entry script is not early
enough to fix it.

The symptom is deceptive. Gate 3 succeeds, the application starts, the main
window builds, and the GUI self-test passes with every widget present. Only
scoring a recording fails, which is why an audit that stops at "it launches"
reports success.

Registered via `runtime_hooks=` in SeizureReview.spec, which places it ahead of
the automatic hooks.

Scope: Windows, and only on the modern stack. macOS and Linux freeze and run
correctly without it, and a hook that raised on a platform that does not need it
would turn a working build into a broken one.

The Python version gate matters as much as the platform one. The legacy 3.6
build ships today, defers TensorFlow until a recording is actually scored, and
does not have the conflict — importing TF 1.15 here would add several seconds to
every launch of the one build that already works, in exchange for nothing.
TensorFlow 2 requires 3.9, so the interpreter version separates the two stacks
for free, without importing anything to find out.

Nothing here may raise: an application that cannot import TensorFlow should
still open a window and say so, rather than die before the splash screen with no
log to read it from.
"""
import os
import sys

if (sys.platform.startswith('win')
        and sys.version_info >= (3, 9)                 # modern stack only
        and not os.environ.get('SEIZ_NO_TF_PRELOAD')):
    try:
        import tensorflow                                       # noqa: F401
    except Exception:                                           # noqa: BLE001
        # Deliberately silent. The application reports a missing or broken
        # TensorFlow when a recording is scored, in a dialog the reviewer can
        # act on; a traceback here would appear on a clinical workstation with
        # no terminal to read it from.
        pass
