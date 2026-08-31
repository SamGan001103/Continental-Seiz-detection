"""Reading a recording span-by-span must give the same samples as loading it.

`gui/io/edf_source.py` lets the display read spans off disk instead of holding
a recording in RAM: a 24-hour file costs 0.15 GB open rather than the 6.5 GB
`gui/memory.py` refuses. The whole value depends on one property — the samples
it serves are the samples `load_edf_19ch` would have loaded, in the same rows.
A display showing different channels, or different samples, from the ones the
detector scored is the worst failure this application has available, so the
central test here assembles the lazy source in full and demands equality with
the ordinary loader.

Two behaviours carry most of the risk and are tested directly:

* **Channel mapping.** Resolved through the detector's own `nedc_get_pos`
  under the parameter file's match mode, so vendor labels like 'EEG FP1-REF'
  land where they do in the model path. Reimplementing that matching would be
  the obvious way to get this subtly wrong.

* **Strided reads.** Overviews ask for one sample in every few thousand. The
  reader batches small strides into contiguous reads and seeks for large ones —
  a whole-day overview read 1.6 GB and took 12 s before that threshold existed.
  Both paths must return exactly what a contiguous read and a stride would, and
  the tests straddle the threshold so neither path goes unexercised.

Not yet wired into the GUI, deliberately: TUH recordings are 256 Hz and are
resampled on load, and an FFT resample of a span is not the corresponding slice
of the resampled whole, so `open_source` declines them. Wiring a second numeric
path into the application for data that would not use it is how the Keras 3
defect happened (docs/known_issues.md §1). It is exercised by these tests and
ready for 250 Hz day-length recordings.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.edf_source import (EdfSource, SEEK_BEATS_READ_STRIDE,   # noqa: E402
                               open_source, resolve_channel_map)
from gui.paths import params_path                                   # noqa: E402

FS = 250
CH = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ',
      'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']


def _write_edf(path, seconds=120, fs=FS, extra_channels=0, seed=5):
    """A small TUH-labelled EDF, optionally with decoy channels interleaved."""
    import pyedflib
    rng = np.random.RandomState(seed)
    names = ['EEG {}-REF'.format(c) for c in CH]
    # Decoys go FIRST, so a mapping that assumed channel i == row i fails.
    labels = ['EKG1-REF', 'PHOTIC-REF'][:extra_channels] + names
    n = seconds * fs
    sigs = (rng.randn(len(labels), n) * 40.0)
    _write_signals(path, labels, sigs, fs)
    return path


def _write_signals(path, labels, sigs, fs):
    """Write an EDF+, coping with either spelling of the rate header key.

    pyedflib renamed `sample_rate` to `sample_frequency`; the legacy 3.6 stack
    ships the old one and the modern stack the new. Feature-detected by
    attempt rather than by version string, the same way
    packaging/smoke_test.py does it — a version check is a second thing to
    keep right, and this suite must pass on both stacks.
    """
    import pyedflib

    def _attempt(rate_key):
        w = pyedflib.EdfWriter(path, len(labels),
                               file_type=pyedflib.FILETYPE_EDFPLUS)
        try:
            w.setSignalHeaders([{
                'label': lab, 'dimension': 'uV', rate_key: fs,
                'physical_min': -3276.8, 'physical_max': 3276.7,
                'digital_min': -32768, 'digital_max': 32767,
                'prefilter': '', 'transducer': ''} for lab in labels])
            w.writeSamples([np.ascontiguousarray(sigs[i])
                            for i in range(len(labels))])
        finally:
            try:
                w.close()
            except Exception:                            # noqa: BLE001
                pass

    try:
        _attempt('sample_frequency')
    except (FutureWarning, KeyError, TypeError, ValueError):
        _attempt('sample_rate')


class LazyReadingEqualsLoading(unittest.TestCase):
    """The property everything else rests on."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix='edfsrc')
        cls.plain = _write_edf(os.path.join(cls.tmp, 'plain.edf'))
        cls.decoy = _write_edf(os.path.join(cls.tmp, 'decoy.edf'),
                               extra_channels=2, seed=6)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _compare(self, path):
        from gui.io.edf import load_edf_19ch
        ref, fs, _dur = load_edf_19ch(path)
        src = open_source(path, params_path())
        self.assertIsNotNone(src, 'a 250 Hz file must get a lazy source')
        try:
            self.assertEqual(src.shape, ref.shape)
            got = np.asarray(src)
            self.assertTrue(
                np.array_equal(got, ref),
                'lazy read differs from load_edf_19ch by up to {}'.format(
                    float(np.abs(got.astype(np.float64)
                                 - ref.astype(np.float64)).max())))
        finally:
            src.close()

    def test_matches_the_loader_exactly(self):
        self._compare(self.plain)

    def test_matches_when_the_file_has_other_channels_first(self):
        """Decoy channels ahead of the EEG catch an index-vs-label mix-up."""
        self._compare(self.decoy)

    def test_the_channel_map_is_not_the_identity_for_that_file(self):
        mapping, fs, n, reader = resolve_channel_map(self.decoy, params_path())
        reader.close()
        self.assertEqual(len(mapping), 19)
        self.assertNotEqual(mapping, list(range(19)),
                            'the decoy file should force a non-trivial '
                            'mapping, or this test proves nothing')
        self.assertEqual(int(round(fs)), FS)


class SlicingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix='edfslice')
        cls.path = _write_edf(os.path.join(cls.tmp, 's.edf'), seconds=300)
        cls.src = open_source(cls.path, params_path())
        cls.full = np.asarray(cls.src)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.src is not None:
            cls.src.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_span_matches(self):
        got = self.src[:, 1000:5000]
        np.testing.assert_array_equal(got, self.full[:, 1000:5000])

    def test_a_single_row_matches_and_is_1d(self):
        got = self.src[7, 2000:2500]
        self.assertEqual(got.ndim, 1)
        np.testing.assert_array_equal(got, self.full[7, 2000:2500])

    def test_small_strides_use_the_batched_path_and_match(self):
        step = 8
        self.assertLess(step, SEEK_BEATS_READ_STRIDE)
        np.testing.assert_array_equal(self.src[:, 0:60000:step],
                                      self.full[:, 0:60000:step])

    def test_large_strides_use_the_seeking_path_and_match(self):
        step = SEEK_BEATS_READ_STRIDE * 2
        np.testing.assert_array_equal(self.src[:, 0:70000:step],
                                      self.full[:, 0:70000:step])

    def test_the_stride_either_side_of_the_threshold_agrees(self):
        for step in (SEEK_BEATS_READ_STRIDE - 1, SEEK_BEATS_READ_STRIDE + 1):
            np.testing.assert_array_equal(
                self.src[:, 0:70000:step], self.full[:, 0:70000:step],
                'stride {} disagrees across the batching threshold'.format(step))

    def test_an_empty_span_is_empty_not_an_error(self):
        self.assertEqual(self.src[:, 500:500].shape, (19, 0))

    def test_a_span_past_the_end_is_clipped(self):
        n = self.src.shape[1]
        got = self.src[:, n - 100:n + 5000]
        self.assertEqual(got.shape, (19, 100))

    def test_reading_after_close_raises_rather_than_returning_junk(self):
        import tempfile
        import shutil
        tmp = tempfile.mkdtemp(prefix='edfclosed')
        try:
            p = _write_edf(os.path.join(tmp, 'c.edf'), seconds=30)
            s = open_source(p, params_path())
            s.close()
            self.assertRaises(RuntimeError, lambda: s[:, 0:10])
            s.close()          # idempotent
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class WhenLazyReadingIsDeclined(unittest.TestCase):
    def test_a_non_250hz_file_is_served_at_its_native_rate(self):
        """It must NOT be resampled, and it must not be refused either.

        This test used to assert the opposite — that a 256 Hz file got no lazy
        source at all, because a span cannot be resampled to match the whole.
        The premise was right and the conclusion was wrong: TUH recordings are
        256 Hz, so declining them left every day-long TUH file refused
        outright, and a tool for ambulatory EEG that cannot open a day of EEG
        is not a working tool. The display does not need 250 Hz; only the
        detector does, and its scores are already cached in seconds.
        """
        import tempfile
        import shutil
        tmp = tempfile.mkdtemp(prefix='edf256')
        try:
            p = _write_edf(os.path.join(tmp, 'f256.edf'), seconds=60, fs=256)
            src = open_source(p, params_path())
            self.assertIsNotNone(
                src, 'a 256 Hz file must still be readable span-by-span — '
                     'refusing it is what made day-long TUH files unopenable')
            try:
                self.assertEqual(int(round(src.fs)), 256,
                                 'the lazy source resampled, which it cannot '
                                 'do correctly span-by-span')
                self.assertTrue(src.is_native_rate,
                                'the source must declare that it is not at '
                                "the detector's rate, so callers needing 250 "
                                'Hz can decline rather than approximate')
            finally:
                src.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_file_without_the_19_channels_gets_no_lazy_source(self):
        import tempfile
        import shutil
        tmp = tempfile.mkdtemp(prefix='edfbad')
        try:
            p = os.path.join(tmp, 'bad.edf')
            _write_signals(p, ['sine', 'ramp'],
                           np.zeros((2, FS * 10)), FS)
            self.assertIsNone(open_source(p, params_path()))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_missing_file_returns_none_rather_than_raising(self):
        self.assertIsNone(open_source(os.path.join(REPO, 'nope.edf'),
                                      params_path()))


class ThePipelineCanDrawFromIt(unittest.TestCase):
    def test_rendering_from_a_lazy_source_matches_rendering_from_memory(self):
        import tempfile
        import shutil
        from gui.display_pipeline import DisplayPipeline
        tmp = tempfile.mkdtemp(prefix='edfpipe')
        try:
            p = _write_edf(os.path.join(tmp, 'p.edf'), seconds=200)
            src = open_source(p, params_path())
            try:
                mem = np.asarray(src)
                a = DisplayPipeline(src, FS, hp=1.0, lp=70.0, notch=60.0,
                                    montage='Longitudinal Bipolar')
                b = DisplayPipeline(mem, FS, hp=1.0, lp=70.0, notch=60.0,
                                    montage='Longitudinal Bipolar')
                for s0, s1, px in ((50 * FS, 80 * FS, 4000),
                                   (0, 200 * FS, 800)):
                    ra, ta = a.render(s0, s1, px)
                    rb, tb = b.render(s0, s1, px)
                    np.testing.assert_array_equal(ta, tb)
                    np.testing.assert_allclose(ra, rb, rtol=0, atol=1e-4)
            finally:
                src.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
