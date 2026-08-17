"""Entry point for the seizure-review GUI.

From a source checkout:
    python -m gui.main                 # empty window
    python -m gui.main <path/to.edf>   # auto-load a file

Frozen build: double-click SeizureReview.exe, or pass an EDF path to it.

There is no terminal on a clinical workstation. A traceback printed to stdout
in a windowed build goes nowhere, so an unhandled exception here is written to
a log file under the per-user data directory and its path is shown in a dialog
the reviewer can read out over the phone.
"""
import os
import sys
import traceback

# BEFORE PyQt5, and that is not a style preference. On Windows with
# TensorFlow 2.x, loading Qt's DLLs first makes TensorFlow's native library fail
# to initialise — "DLL load failed while importing _pywrap_tensorflow_internal:
# A dynamic link library (DLL) initialization routine failed." Reversing the
# order fixes it. This is docs/known_issues.md §1, which was attributed to
# PyInstaller for six build attempts and reproduces from source with no
# PyInstaller involved. See gui/tf_preload.py for the measurements.
#
# Do not move this below the PyQt5 import. Do not merge it into the block below.
import gui.tf_preload                                        # noqa: F401,E402

from PyQt5 import QtCore, QtWidgets                          # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.paths import writable_root                          # noqa: E402


def log_path():
    d = os.path.join(writable_root(), 'logs')
    try:
        if not os.path.exists(d):
            os.makedirs(d)
    except OSError:
        return None
    return os.path.join(d, 'seizure_review.log')


def _write_log(text):
    p = log_path()
    if p is None:
        return None
    try:
        with open(p, 'a', encoding='utf-8') as f:
            f.write(text)
        return p
    except OSError:
        return None


def install_excepthook():
    """Route unhandled exceptions to a log file and a dialog.

    Without this a packaged build simply vanishes on an unexpected error, with
    nothing for the user to report and nothing for us to diagnose.
    """
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        stamp = QtCore.QDateTime.currentDateTime().toString(QtCore.Qt.ISODate)
        body = ''.join(traceback.format_exception(exc_type, exc, tb))
        written = _write_log(
            '\n===== {} =====\n{}'.format(stamp, body))
        # In a windowed frozen build sys.stderr is None. Writing to it here
        # would raise inside the handler whose whole job is to survive a crash,
        # and the user would get no dialog and no log entry.
        if sys.stderr is not None:
            try:
                sys.stderr.write(body)
            except Exception:
                pass
        if QtWidgets.QApplication.instance() is not None:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setWindowTitle('Seizure Review — unexpected error')
            msg.setText('Something went wrong.\n\n{}: {}'.format(
                exc_type.__name__, exc))
            msg.setInformativeText(
                'Any review you have already saved is unaffected.\n\n'
                'Details were written to:\n{}'.format(written or '(no log)'))
            msg.setDetailedText(body)
            msg.exec_()
    sys.excepthook = hook


def self_test(edf_path):
    """Exercise the full inference path headlessly and report.

    This exists for the packaged build. A frozen app can import cleanly and
    still die the first time it reaches a module that was imported inside a
    function — TensorFlow, MNE's ICA, the STFT helper — and the only way to
    find that before a clinician does is to actually run one recording through.
    Kept to a few windows so it finishes in minutes, not hours.
    """
    from gui.io.edf import load_edf_19ch
    from gui.io.infer import compute_probs_from_data, SEGMENT_S
    import numpy as np

    print('self-test: python      {}'.format(sys.version.split()[0]))
    print('self-test: frozen      {}'.format(
        bool(getattr(sys, 'frozen', False))))
    print('self-test: reading     {}'.format(edf_path))
    data, fs, duration = load_edf_19ch(edf_path)
    print('self-test: signal      {} ch, {} Hz, {:.1f} s'.format(
        data.shape[0], fs, duration))

    # Two minutes of signal is enough to prove every stage runs.
    n = min(data.shape[1], int(120 * fs))
    if n < SEGMENT_S * fs:
        print('self-test: FAIL        recording shorter than one window')
        return 1
    starts, probs, skip = compute_probs_from_data(data[:, :n], fs)
    scored = int((skip == 0).sum())
    print('self-test: windows     {} ({} scored, {} skipped)'.format(
        len(starts), scored, len(starts) - scored))
    if scored == 0:
        print('self-test: FAIL        no window was scored')
        return 1
    p = np.asarray(probs)[skip == 0]
    print('self-test: scores      min {:.4f}  max {:.4f}  mean {:.4f}'.format(
        p.min(), p.max(), p.mean()))
    print('self-test: PASS')
    return 0


def gui_self_test():
    """Build the real main window, pump the event loop, and exit.

    The headless inference self-test never touches Qt, so it cannot catch a
    pyqtgraph or PyQt5 submodule that failed to get bundled — which would
    surface as a blank window or an instant exit on the target machine. This
    constructs every widget the application uses and then quits.
    """
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    from gui.app import MainWindow
    w = MainWindow()
    w.show()
    for _ in range(20):
        app.processEvents()
    print('gui-self-test: window   {}x{}'.format(w.width(), w.height()))
    print('gui-self-test: widgets  {}'.format(
        len(w.findChildren(QtWidgets.QWidget))))
    print('gui-self-test: PASS')
    w.close()
    return 0


def main():
    if '--gui-self-test' in sys.argv:
        try:
            return gui_self_test()
        except Exception:
            print(''.join(traceback.format_exception(*sys.exc_info())))
            print('gui-self-test: FAIL     exception above')
            return 1

    if '--self-test' in sys.argv:
        rest = [a for a in sys.argv[1:] if not a.startswith('-')]
        if not rest:
            print('self-test: FAIL        no EDF path given')
            return 2
        try:
            return self_test(rest[0])
        except Exception:
            print(''.join(traceback.format_exception(*sys.exc_info())))
            print('self-test: FAIL        exception above')
            return 1

    # Must be set before the QApplication exists. Clinical workstations are
    # frequently 4K at 150 % scaling, where an unaware app renders unreadably
    # small — and EEG review is a task about small amplitude differences.
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Seizure Review')
    app.setOrganizationName('University of Sydney')
    install_excepthook()

    from gui.app import MainWindow
    w = MainWindow()
    w.show()

    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if args and args[0].lower().endswith('.edf'):
        w.load_edf(os.path.abspath(args[0]))
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
