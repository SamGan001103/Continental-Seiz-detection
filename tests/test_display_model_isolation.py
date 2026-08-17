"""The display path must never reach the detector's input.

This is the most defensible piece of architecture in the repository and until
now it was maintained by convention alone. Two signal paths exist:

    DISPLAY   _raw_data -> apply_filters -> apply_montage -> SignalView
    MODEL     _original_data -> ica_arti_remove -> _calc_stft -> ConvLSTM

The released weights `convlstm_ICA_12_train.h5` were fitted to features produced
by the second path exactly as written. Feed the model a 15 Hz-lowpassed array and
the log-STFT it consumes is a different tensor; feed it a bipolar montage and it
is an 18-row array, so `ICA(n_components=19)` either raises or silently re-fits a
different spatial map. Either way the scores stop being the published model's and
the reproduction argument collapses - retroactively, and with nothing on screen
to indicate it.

The hazard is specific and current: `_raw_data` and `_original_data` are bound to
the *same numpy object* in the baseline case (`gui/app.py`, on load). That is safe
only because `apply_filters` allocates a fresh array rather than writing in place.
Nothing enforced that. A future in-place display operation - a detrend, a rescale,
a `-=` - would corrupt the detector's input silently, and every existing test
would still pass.
"""
import inspect
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class DisplayFiltersDoNotMutateTheirInput(unittest.TestCase):
    """apply_filters must copy, not write through."""

    def setUp(self):
        from gui.processing import apply_filters
        self.apply_filters = apply_filters
        rng = np.random.RandomState(0)
        self.data = (rng.randn(19, 2500) * 50.0).astype(np.float32)

    def test_input_array_is_unchanged(self):
        before = self.data.copy()
        self.apply_filters(self.data, 250, hp=1.0, lp=15.0, notch=60.0)
        np.testing.assert_array_equal(
            self.data, before,
            'apply_filters modified its input. _raw_data and _original_data '
            'are the same object on load, so this silently changes what the '
            'detector scores.')

    def test_output_does_not_share_memory_with_input(self):
        out = self.apply_filters(self.data, 250, hp=1.0, lp=70.0, notch=60.0)
        self.assertFalse(
            np.shares_memory(out, self.data),
            'the filtered array is a view onto the input; a later in-place '
            'edit of the display data would reach the detector')

    def test_the_no_op_case_still_copies(self):
        """All-None must not hand back the caller's own array."""
        out = self.apply_filters(self.data, 250)
        self.assertFalse(np.shares_memory(out, self.data))

    def test_montage_does_not_mutate_its_input(self):
        from gui.processing import apply_montage, MONTAGES
        for name in MONTAGES:
            before = self.data.copy()
            apply_montage(self.data, name)
            np.testing.assert_array_equal(
                self.data, before,
                'apply_montage({!r}) modified its input'.format(name))


class TheInferencePathDoesNotImportDisplayCode(unittest.TestCase):
    """A structural guard, not a behavioural one.

    If `gui.io.infer` never imports `gui.processing`, no display control can
    reach the detector however the UI is wired later.
    """

    def test_infer_does_not_import_gui_processing(self):
        src = inspect.getsource(__import__('gui.io.infer', fromlist=['x']))
        for needle in ('from gui.processing', 'import gui.processing',
                       'apply_filters', 'apply_montage'):
            self.assertNotIn(
                needle, src,
                'gui/io/infer.py references {!r}. The detector must not be '
                'able to see the display filter chain.'.format(needle))

    def test_preprocessing_does_not_import_display_code(self):
        from utils import preprocessing
        src = inspect.getsource(preprocessing)
        self.assertNotIn('gui.processing', src)
        self.assertNotIn('apply_filters', src)


class TheModelIsScoredOnTheUnfilteredRecording(unittest.TestCase):
    """End to end: filtering the display must not move p(seizure).

    The cheapest possible statement of the whole contract - score a signal,
    score it again after the display filters have been applied to a *copy*,
    and require the probabilities to be identical.
    """

    # Run in a fresh interpreter, via gui.main so the TensorFlow preload
    # applies. In-process, this test inherits whatever import order the rest of
    # the suite happened to establish: on Windows with TF 2, an earlier module
    # that imported PyQt5 makes TensorFlow's native library fail to load here
    # (docs/known_issues.md 1), and the test then reports a DLL error that has
    # nothing to do with what it is asserting.
    PROBE = '''
import sys
import numpy as np
import gui.main                                    # noqa: F401 - orders the imports
from gui.io.infer import compute_probs_from_data
from gui.processing import apply_filters

rng = np.random.RandomState(7)
fs = 250
t = np.arange(30 * fs) / float(fs)
data = np.stack([(40.0 * np.sin(2 * np.pi * (8.0 + 0.2 * i) * t)
                  + 15.0 * rng.randn(t.size)).astype(np.float32)
                 for i in range(19)])

starts_a, probs_a, _ = compute_probs_from_data(data, fs)
apply_filters(data, fs, hp=1.0, lp=15.0, notch=60.0)   # the display path, same array
starts_b, probs_b, _ = compute_probs_from_data(data, fs)

same = (np.array_equal(np.asarray(starts_a), np.asarray(starts_b))
        and np.array_equal(np.asarray(probs_a), np.asarray(probs_b)))
print("IDENTICAL", same)
'''

    def test_filtering_the_display_copy_leaves_scores_untouched(self):
        import subprocess
        out = subprocess.run(
            [sys.executable, '-c', self.PROBE], cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=900).stdout.decode('utf-8', 'replace')
        if 'IDENTICAL' not in out:
            self.skipTest('inference stack unavailable in a clean '
                          'interpreter:\n' + out[-800:])
        self.assertIn(
            'IDENTICAL True', out,
            'p(seizure) changed after the display filters ran over the same '
            'array. The display path is reaching the detector.')


if __name__ == '__main__':
    unittest.main()
