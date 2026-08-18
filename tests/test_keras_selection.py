"""Keras 2 must be selected before TensorFlow is imported, on every path.

The detector reproduces the published numbers only under Keras 2. TF 2.16+ ships
Keras 3, which reimplemented `ConvLSTM2D`; it loads the same weights and returns
different probabilities. `tf-keras` is installed for this reason and selected by
`TF_USE_LEGACY_KERAS=1`.

The trap is that `tf.keras` resolves **lazily**, on first attribute access, and
TensorFlow reads the variable at that moment. So setting it after `import
tensorflow` is not reliably too late, and setting it after something has touched
`tf.keras` is silently far too late. There is no error either way.

This shipped once. `packaging/rthook_tf_before_qt.py` -- added to fix a Windows
DLL ordering problem (known_issues.md 1) -- imports TensorFlow before the frozen
application's entry script, and did not set the variable. `gui/io/infer.py` sets
it correctly, but by then the decision was made. The frozen Windows build scored
a recording at

    source, Keras 2   min 0.0001  max 0.0333  mean 0.0113
    frozen            min 0.0002  max 0.0391  mean 0.0141
    source, Keras 3   min 0.0002  max 0.0391  mean 0.0141      <- the match

while every build gate passed and the spec printed "tf_keras bundled (Keras 2
numerics preserved)" -- which was true and irrelevant: bundling is not selecting.

Note the direction, and its limits. Both preloads are Windows-only, so macOS and
Linux never lost *this* race, and for the frozen build Windows was the outlier --
the opposite of what a cross-platform investigation would assume.

"Correct all along" would be too strong, and an earlier version of this docstring
said it. Nothing selected Keras 2 on any platform before 36a56b6 (2026-08-11
10:11), which added both the tf-keras pin and the setdefault in gui/io/infer.py.
Every modern-stack run on every platform before that date was Keras 3, including
both arms of the platform-drift study. What is Windows-specific is the frozen
build continuing to lose the race for a further week, after the source path had
been fixed.

These tests read the source rather than importing TensorFlow: the ordering is a
property of the text, they must pass on the legacy 3.6 stack where tf_keras does
not exist, and a test that imported TF to check would itself have to win the same
race it is policing.
"""
import io
import os
import re
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

VAR = 'TF_USE_LEGACY_KERAS'

# Every file that imports TensorFlow *early* -- before the module that owns the
# setting has necessarily been imported. Each must select Keras 2 itself.
PRELOADERS = (
    os.path.join('packaging', 'rthook_tf_before_qt.py'),
    os.path.join('gui', 'tf_preload.py'),
)


def _source(rel):
    with io.open(os.path.join(REPO, rel), encoding='utf-8') as f:
        return f.read()


def _strip_comments_and_strings(src):
    """Crude, but enough to stop a docstring mention counting as code.

    Both files discuss this variable at length in their comments, so a plain
    substring search would pass on a file that only *talks* about setting it.
    """
    src = re.sub(r'"""(?:.|\n)*?"""', '', src)
    src = re.sub(r"'''(?:.|\n)*?'''", '', src)
    src = re.sub(r'#[^\n]*', '', src)
    return src


class PreloadersSelectKeras2First(unittest.TestCase):
    def test_each_preloader_sets_the_variable(self):
        for rel in PRELOADERS:
            code = _strip_comments_and_strings(_source(rel))
            self.assertIn(
                VAR, code,
                '{} imports TensorFlow but never sets {}. Whatever touches '
                'tf.keras first will pick the Keras version, and Keras 3 '
                'returns different probabilities with no error.'
                .format(rel, VAR))

    def test_the_variable_is_set_before_tensorflow_is_imported(self):
        """The whole point. Order, not presence."""
        for rel in PRELOADERS:
            code = _strip_comments_and_strings(_source(rel))
            set_at = code.find(VAR)
            # Guard the guard: str.find returns -1 when absent, and -1 is less
            # than any import position, so without this the ordering assertion
            # below passes cleanly on a file that never sets the variable at
            # all -- reporting order as correct in exactly the case that
            # shipped Keras 3.
            self.assertNotEqual(set_at, -1,
                                '{}: does not set {} anywhere'.format(rel, VAR))
            import_at = min(
                [m.start() for m in re.finditer(r'^\s*import\s+tensorflow',
                                                code, re.M)] or [-1])
            self.assertNotEqual(import_at, -1,
                                '{}: no `import tensorflow` found; this test '
                                'no longer covers what it claims to'.format(rel))
            self.assertLess(
                set_at, import_at,
                '{}: {} is set AFTER `import tensorflow`. tf.keras binds on '
                'first attribute access, so this is a race the file loses '
                'silently.'.format(rel, VAR))

    def test_it_is_set_with_setdefault_not_assignment(self):
        """An operator must be able to force Keras 3 to reproduce the bug.

        Hard assignment would also stop `SEIZ_NO_TF_PRELOAD`-style debugging and
        would overwrite a deliberate choice made by the caller -- which is how
        the diagnosis was confirmed in the first place, by running the shipped
        frozen binary with the variable forced on and watching the Keras 2
        numbers come back.
        """
        for rel in PRELOADERS:
            code = _strip_comments_and_strings(_source(rel))
            self.assertTrue(
                re.search(r'environ\.setdefault\(\s*[\'"]' + VAR, code),
                '{}: set {} with os.environ.setdefault, not assignment, so it '
                'stays overridable'.format(rel, VAR))


class TheOwningModuleStillSetsIt(unittest.TestCase):
    """The preloads are Windows-only; every other platform relies on this one."""

    def test_infer_sets_the_variable_before_importing_tensorflow(self):
        code = _strip_comments_and_strings(
            _source(os.path.join('gui', 'io', 'infer.py')))
        self.assertIn(VAR, code, 'gui/io/infer.py no longer selects Keras 2 — '
                                 'macOS and Linux have nothing else that does')
        set_at = code.find(VAR)
        imports = [m.start() for m in
                   re.finditer(r'^\s*(?:import\s+tensorflow'
                               r'|from\s+tensorflow)', code, re.M)]
        if imports:
            self.assertLess(set_at, min(imports),
                            'gui/io/infer.py sets {} after importing '
                            'TensorFlow'.format(VAR))


class TheBuildGateChecksTheFrozenApp(unittest.TestCase):
    """Bundling tf_keras is not the same as selecting it, and only one is gated.

    The spec's build-time message says tf_keras was collected. That was printed,
    truthfully, by the build that shipped Keras 3. The check that matters runs
    the frozen binary and asks what it actually selected.
    """

    def test_the_self_test_reports_which_keras_it_selected(self):
        code = _source(os.path.join('gui', 'main.py'))
        self.assertIn('self-test: keras', code,
                      'the self-test no longer reports the Keras flavour, so '
                      'smoke_test.py has nothing to check')

    def test_the_smoke_test_fails_the_build_on_keras_3(self):
        code = _source(os.path.join('packaging', 'smoke_test.py'))
        self.assertIn('self-test: keras', code,
                      'smoke_test.py no longer inspects the Keras flavour')
        self.assertTrue(
            re.search(r'failures\.append\([^)]*Keras 3', code, re.S),
            'smoke_test.py notices the Keras flavour but does not fail the '
            'build on Keras 3 — a silent wrong-numbers build would ship again')


if __name__ == '__main__':
    unittest.main()
