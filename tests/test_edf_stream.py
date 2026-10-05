"""The streaming loader must return the SAME NUMBERS, not similar ones.

`gui/io/edf_stream.load_edf_19ch_streaming` exists so a day-long recording can
be scored without a 9 GB peak. It earns its place only if its output is bit
identical to `load_edf_19ch`, because everything this project reports is
produced through that function.

The standard is not arbitrary. A parallel scorer that was 8.8x faster was
thrown away here because pinning BLAS threads moved p(seizure) by 0.138, and a
3.3x one that is bitwise identical shipped instead
(`gui/io/infer_parallel.py`). "Numerically equivalent" is not the bar.

Two properties are pinned separately, because they can fail independently:

* reading only the mapped channels equals reading every channel and selecting;
* resampling row by row equals `scipy.signal.resample(X, n, axis=1)`.

The second is the one that looks obviously true and is worth checking anyway:
a batched transform is free to use a different algorithm from a single-row one,
and this project has already been caught out once by a library quietly changing
its arithmetic (`docs/known_issues.md` section 1, Keras 2 versus Keras 3).
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FS_NATIVE = 256
FS_TARGET = 250
CH = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ',
      'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']
# Channels a real recording carries that the detector does not use. Reading
# and discarding these is a third of the memory the streaming loader saves.
EXTRA = ['SP1', 'SP2', 'EOG1', 'EOG2', 'EMG', 'ECG', 'PHO', 'T1', 'T2', 'RS']


def _write_edf(path, seconds, fs, labels, seed=13):
    """A recording shaped like the real ones: extra channels, 16-bit source."""
    import pyedflib
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    # Pink-ish noise rather than white: real EEG is heavily oversampled and
    # smooth, which is exactly what a resample has to preserve.
    sigs = np.cumsum(rng.standard_normal((len(labels), n)), axis=1)
    sigs -= sigs.mean(axis=1, keepdims=True)
    sigs *= 40.0 / np.maximum(sigs.std(axis=1, keepdims=True), 1e-9)

    def _attempt(rate_key):
        w = pyedflib.EdfWriter(path, len(labels),
                               file_type=pyedflib.FILETYPE_EDF)
        try:
            w.setSignalHeaders([{
                'label': 'EEG {}-REF'.format(lab), 'dimension': 'uV',
                rate_key: fs,
                'physical_min': -5000.0, 'physical_max': 5000.0,
                'digital_min': -32768, 'digital_max': 32767,
                'prefilter': '', 'transducer': ''} for lab in labels])
            w.writeSamples([np.ascontiguousarray(s) for s in sigs])
        finally:
            try:
                w.close()
            except Exception:                                # noqa: BLE001
                pass

    try:
        _attempt('sample_frequency')
    except (FutureWarning, KeyError, TypeError, ValueError):
        _attempt('sample_rate')
    return path


class RowWiseResamplingIsBitIdentical(unittest.TestCase):
    """The saving rests on this: resample each row separately rather than
    handing scipy the whole array. Needs no EDF and no Qt."""

    def _check(self, n_rows, n_src, n_out, seed):
        from scipy.signal import resample
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.standard_normal((n_rows, n_src)), axis=1)
        both = resample(x, n_out, axis=1)
        rows = np.empty((n_rows, n_out), dtype=both.dtype)
        for i in range(n_rows):
            rows[i] = resample(x[i], n_out)
        self.assertTrue(
            np.array_equal(both, rows),
            'row-wise resampling differs from the 2D call by up to {} — the '
            'streaming loader would change every reported number'.format(
                np.abs(both - rows).max()))

    def test_at_the_real_ratio(self):
        n_src = 60 * FS_NATIVE
        self._check(19, n_src, int(n_src * FS_TARGET / FS_NATIVE), 1)

    def test_at_an_awkward_length(self):
        """A prime-ish length exercises a different FFT plan."""
        self._check(19, 15361, int(15361 * FS_TARGET / FS_NATIVE), 2)

    def test_when_upsampling(self):
        self._check(19, 4000, 6000, 3)

    def test_one_row_is_the_degenerate_case(self):
        self._check(1, 8000, 7812, 4)


class TheStreamingLoaderMatchesTheCanonicalOne(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix='edfstream')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._shutil = shutil

    def _compare(self, seconds, fs, labels):
        from gui.io.edf import load_edf_19ch
        from gui.io.edf_stream import load_edf_19ch_streaming
        path = _write_edf(os.path.join(self.tmp, 'r.edf'), seconds, fs, labels)
        ref, fs_ref, dur_ref = load_edf_19ch(path)
        got, fs_got, dur_got = load_edf_19ch_streaming(path)
        self.assertEqual(fs_ref, fs_got)
        self.assertEqual(ref.shape, got.shape,
                         'shapes differ: {} vs {}'.format(ref.shape,
                                                          got.shape))
        self.assertEqual(ref.dtype, got.dtype)
        self.assertAlmostEqual(dur_ref, dur_got, places=9)
        self.assertTrue(
            np.array_equal(ref, got),
            'streaming loader differs from load_edf_19ch by up to {}'.format(
                np.abs(ref.astype(np.float64) -
                       got.astype(np.float64)).max()))

    def test_at_the_target_rate_no_resample_happens(self):
        self._compare(20, FS_TARGET, CH)

    def test_at_the_native_rate_of_the_real_recordings(self):
        """256 Hz is what EPILEPSIAE ships, so this is the branch that
        matters and the one the earlier TUH-only checks never exercised."""
        self._compare(20, FS_NATIVE, CH)

    def test_with_the_extra_channels_a_real_recording_carries(self):
        """Ten of 29 channels are read and discarded by the eager loader.
        Not reading them must not change the nineteen that are kept."""
        self._compare(20, FS_NATIVE, CH + EXTRA)

    def test_with_the_channels_out_of_montage_order(self):
        """EPILEPSIAE stores FP1 FP2 F3 F4 ... — not the order the detector
        wants. The row order of the result is load-bearing: the first ConvLSTM
        kernel spans the whole electrode axis."""
        shuffled = ['F3', 'O2', 'FP1', 'T4'] + \
            [c for c in CH if c not in ('F3', 'O2', 'FP1', 'T4')] + EXTRA
        self._compare(20, FS_NATIVE, shuffled)


class TheEstimateIsHonestAboutWhatItCosts(unittest.TestCase):
    def test_streaming_is_predicted_to_cost_far_less_than_eager(self):
        from gui.io.edf_stream import estimate_peak_bytes
        from gui.memory import estimate_bytes
        n_src = 22 * 3600 * FS_NATIVE
        resident, transient = estimate_bytes(29, n_src, FS_NATIVE)
        lean = estimate_peak_bytes(29, n_src, FS_NATIVE)
        self.assertLess(lean, (resident + transient) / 2.0,
                        'the streaming loader is supposed to be the cheap '
                        'path; if it is not, it has no reason to exist')

    def test_it_does_not_claim_to_be_free(self):
        from gui.io.edf_stream import estimate_peak_bytes
        n_src = 22 * 3600 * FS_NATIVE
        n_out = int(n_src * FS_TARGET / FS_NATIVE)
        self.assertGreaterEqual(estimate_peak_bytes(29, n_src, FS_NATIVE),
                                19 * n_out * 4)


if __name__ == '__main__':
    unittest.main()
