"""TensorFlow must be imported before Qt on the modern Windows stack.

This is `docs/known_issues.md` §1. Loading Qt's DLLs first makes TensorFlow's
native library fail to initialise on Windows with TF 2.x:

    ImportError: DLL load failed while importing _pywrap_tensorflow_internal:
                 A dynamic link library (DLL) initialization routine failed.

It cost six build attempts and was blamed on PyInstaller, on conda's DLL layout,
and on bundled-DLL shadowing in turn. It is an import-order problem and
reproduces from source with no PyInstaller involved.

The failure has no good symptom: the window opens normally and the application
only dies when a recording needs scoring. So the ordering has to be a tested
property, not a comment.
"""
import os
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class PreloadPolicy(unittest.TestCase):
    """The decision must not itself import TensorFlow."""

    def test_the_legacy_stack_is_left_alone(self):
        """3.6 ships today, has no conflict, and must not pay for the fix.

        The sys.modules half runs in a subprocess on purpose. Asserting it in
        this process would only be true when this module happens to run before
        any test that imports TensorFlow - an order dependence that passes alone
        and fails in the suite, which is worse than no test at all.
        """
        from gui import tf_preload
        if sys.version_info >= (3, 9):
            self.skipTest('this interpreter is the modern stack')
        self.assertFalse(tf_preload._should_preload())

        out = subprocess.run(
            [sys.executable, '-c',
             'import gui.tf_preload, sys; '
             'print("TF_LOADED", "tensorflow" in sys.modules)'],
            cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=300).stdout.decode('utf-8', 'replace')
        self.assertIn('TF_LOADED False', out,
                      'importing gui.tf_preload pulled TensorFlow into the '
                      'legacy build, which defers it on purpose:\n' + out[-800:])

    def test_non_windows_is_left_alone(self):
        from gui import tf_preload
        if sys.platform.startswith('win'):
            self.skipTest('this machine is Windows')
        self.assertFalse(tf_preload._should_preload())

    def test_preload_never_raises(self):
        """A machine without TensorFlow must still get a window."""
        from gui import tf_preload
        self.assertIsInstance(tf_preload.preload(), str)

    def test_status_is_reported(self):
        from gui import tf_preload
        self.assertTrue(tf_preload.STATUS)


class MainImportsTensorFlowFirst(unittest.TestCase):
    """The ordering inside gui/main.py, checked as source, not as behaviour.

    Reading the file is the point: the behavioural consequence only appears on
    one platform with one stack, so a runtime check would pass everywhere the
    bug is invisible and catch nothing on the machine that matters.
    """

    def test_tf_preload_is_imported_before_pyqt5(self):
        path = os.path.join(REPO, 'gui', 'main.py')
        with open(path, encoding='utf-8') as f:
            src = f.read()
        i_pre = src.find('import gui.tf_preload')
        i_qt = src.find('from PyQt5')
        self.assertNotEqual(i_pre, -1, 'gui/main.py no longer preloads '
                                       'TensorFlow; see known_issues.md 1')
        self.assertNotEqual(i_qt, -1)
        self.assertLess(i_pre, i_qt,
                        'gui/main.py imports PyQt5 before gui.tf_preload. On '
                        'Windows with TF 2 that makes the frozen application '
                        'open a window and then fail to score anything.')


@unittest.skipUnless(sys.platform.startswith('win') and
                     sys.version_info >= (3, 9),
                     'the conflict is Windows + TensorFlow 2 only')
class TheConflictIsRealOnThisMachine(unittest.TestCase):
    """Demonstrate the bug still exists, so the fix is not cargo cult.

    Each order runs in a fresh interpreter - once a process has imported either
    library the experiment cannot be repeated in it.
    """

    def _run(self, code):
        return subprocess.run([sys.executable, '-c', code], cwd=REPO,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=600).stdout.decode('utf-8', 'replace')

    def test_tensorflow_first_then_qt_works(self):
        out = self._run('import tensorflow, PyQt5.QtCore; print("OK")')
        self.assertIn('OK', out)

    def test_qt_first_then_tensorflow_still_fails(self):
        """If this ever starts passing, the upstream bug is fixed.

        That would be good news and this test failing is how we would learn it -
        at which point the preload can be reconsidered rather than kept forever
        out of superstition.
        """
        out = self._run('import PyQt5.QtCore, tensorflow; print("OK")')
        if 'OK' in out:
            self.skipTest('Qt-before-TF no longer fails here; the preload may '
                          'no longer be needed - re-measure before removing it')
        self.assertIn('DLL load failed', out)

    def test_the_application_entry_point_can_build_the_model(self):
        """The end-to-end statement: import the app, then score-load the model."""
        out = self._run(
            'import gui.main; from gui.io.infer import _build_model; '
            'print("PARAMS", _build_model().count_params())')
        self.assertIn('PARAMS 384846', out,
                      'the entry point cannot build the model on this stack:\n'
                      + out[-1500:])


if __name__ == '__main__':
    unittest.main()
