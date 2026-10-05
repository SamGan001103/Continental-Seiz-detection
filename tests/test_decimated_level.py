"""The in-RAM decimated level, and what it is allowed to change.

Zooming out was measured at 0.3-3.0 s per render, and 96-103% of that is the
strided read walking the span off a 1.2 GB file to draw 1600 points. The level
replaces that read with a slice of a 26 MB array.

The risk it carries is that it draws a DIFFERENT PICTURE, quietly, in a tool
whose whole claim is that what the clinician sees is the recording. So these
pin the properties that make it honest:

  * every point it serves is a real sample from the recording, never
    interpolated and never averaged;
  * every point is at its true time, so nothing slides sideways;
  * it refuses requests it cannot serve at full density rather than thinning
    the trace;
  * it never serves a level built from a different recording.

No Qt, no model, no EDF: a level is an array and the arithmetic can be argued
about on its own.
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io import decimated_level as DL              # noqa: E402

FS = 256
N = FS * 600                      # ten minutes


class _Source(object):
    """Stands in for an EdfSource: an array that counts how much is read."""

    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return self.data[key]


def ramp():
    """Each sample's value IS its index, so a drawn point can be checked
    against the time it claims to be at."""
    return np.tile(np.arange(N, dtype=np.float32), (19, 1))


class EveryPointIsARealSampleAtItsTrueTime(unittest.TestCase):
    def setUp(self):
        self.src = _Source(ramp())
        self.level = DL.build(self.src, FS, stride=64)

    def test_the_level_holds_every_stride_th_sample(self):
        self.assertEqual(self.level.stride, 64)
        self.assertEqual(self.level.data.shape[1], (N + 63) // 64)
        np.testing.assert_array_equal(self.level.data[0, :5],
                                      [0, 64, 128, 192, 256])

    def test_a_slice_returns_samples_that_exist_in_the_recording(self):
        rows, first, eff = self.level.slice(0, N, 256)
        # The ramp means value == original index. Every value must be an exact
        # multiple of the level stride, i.e. a sample that was really taken.
        self.assertTrue(np.all(rows[0] % 64 == 0))

    def test_the_reported_times_match_the_samples_returned(self):
        rows, first, eff = self.level.slice(10_000, 60_000, 300)
        claimed = first + np.arange(rows.shape[1]) * eff
        np.testing.assert_array_equal(rows[0], claimed.astype(np.float32))

    def test_a_slice_never_starts_before_the_span_asked_for(self):
        for s0 in (0, 1, 63, 64, 65, 5000):
            rows, first, _ = self.level.slice(s0, s0 + 50_000, 256)
            self.assertGreaterEqual(first, s0)
            self.assertLess(first - s0, 64)

    def test_the_effective_stride_is_a_multiple_of_the_levels_own(self):
        for want in (64, 100, 128, 257, 1000, 4096):
            _, _, eff = self.level.slice(0, N, want)
            self.assertEqual(eff % 64, 0)
            self.assertGreaterEqual(eff, 64)

    def test_it_never_draws_a_thinner_trace_than_was_asked_for(self):
        """The one quality regression this module could cause, and the reason
        the stride rounds DOWN rather than to nearest.

        Rounding to nearest put a 600 s view on stride 128 when 96 was asked
        for — 75% of the points, a visibly sparser trace, in the regime where
        the reviewer is scanning for events. Serving more points than asked is
        harmless: paint is about 1 ms per extra 1000 points, against the 1.7 s
        the level saves.
        """
        for want in range(64, 4096, 17):
            _, _, eff = self.level.slice(0, N, want)
            self.assertLessEqual(
                eff, want,
                'asked for stride {} and got {} — a thinner trace than the '
                'full-rate path would draw'.format(want, eff))

    def test_it_never_interpolates(self):
        """A level that averaged or resampled would show a waveform the
        recording does not contain. Checked by value, not by trusting it."""
        rows, _, _ = self.level.slice(0, N, 512)
        self.assertTrue(np.all(rows[0] == np.round(rows[0])))

    def test_an_empty_span_is_empty_not_an_error(self):
        rows, _, _ = self.level.slice(500, 500, 256)
        self.assertEqual(rows.shape[1], 0)


class ItRefusesWhatItCannotServe(unittest.TestCase):
    def setUp(self):
        self.level = DL.build(_Source(ramp()), FS, stride=64)

    def test_it_declines_strides_finer_than_its_own(self):
        """Below its own stride it holds fewer points than the caller wants.
        Serving them anyway would thin the trace silently, which at a review
        zoom is lost morphology."""
        for stride in (1, 2, 16, 63):
            self.assertFalse(self.level.covers(stride))

    def test_it_accepts_its_own_stride_and_coarser(self):
        for stride in (64, 65, 1000, 100000):
            self.assertTrue(self.level.covers(stride))


class BuildingIsChunkedButSeamless(unittest.TestCase):
    def test_chunking_does_not_shift_the_sample_grid(self):
        """Each chunk starts on the global stride grid. Without that, a chunk
        boundary moves the phase and the trace steps sideways mid-recording."""
        src = _Source(ramp())
        whole = DL.build(src, FS, stride=64, chunk_s=10_000.0)
        for chunk_s in (37.0, 60.0, 123.5):
            part = DL.build(_Source(ramp()), FS, stride=64, chunk_s=chunk_s)
            np.testing.assert_array_equal(part.data, whole.data)

    def test_it_matches_a_single_whole_file_stride(self):
        src = _Source(ramp())
        level = DL.build(src, FS, stride=64)
        np.testing.assert_array_equal(level.data, ramp()[:, ::64])

    def test_the_size_is_what_was_promised(self):
        # 19 channels x 24 h at 4 Hz x 4 bytes ~ 26 MB. Scaled down here.
        level = DL.build(_Source(ramp()), FS, stride=64)
        expected = 19 * ((N + 63) // 64) * 4
        self.assertEqual(level.nbytes, expected)

    def test_progress_is_reported_monotonically(self):
        seen = []
        DL.build(_Source(ramp()), FS, stride=64, chunk_s=60.0,
                 progress=seen.append)
        self.assertTrue(seen)
        self.assertEqual(seen, sorted(seen))
        self.assertAlmostEqual(seen[-1], 1.0, places=6)


class ALevelNeverOutlivesItsRecording(unittest.TestCase):
    """Same guard as the probability cache, for the same reason: a recording
    copied, regenerated or replaced must not be drawn from the old one's
    samples."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rec = os.path.join(self.tmp, 'rec.edf')
        with open(self.rec, 'wb') as f:
            f.write(b'x' * 4096)
        self.path = DL.path_for(self.rec, 64)
        DL.build(_Source(ramp()), FS, stride=64).save(self.path, self.rec)

    def test_it_round_trips(self):
        back = DL.load(self.path, self.rec)
        self.assertIsNotNone(back)
        self.assertEqual(back.stride, 64)
        self.assertEqual(back.fs, float(FS))
        np.testing.assert_array_equal(back.data, ramp()[:, ::64])

    def test_a_changed_recording_invalidates_it(self):
        with open(self.rec, 'wb') as f:
            f.write(b'x' * 8192)
        self.assertIsNone(DL.load(self.path, self.rec))

    def test_a_missing_recording_invalidates_it(self):
        os.remove(self.rec)
        self.assertIsNone(DL.load(self.path, self.rec))

    def test_a_truncated_level_is_ignored_not_fatal(self):
        with open(self.path, 'r+b') as f:
            f.truncate(200)
        self.assertIsNone(DL.load(self.path, self.rec))

    def test_a_missing_level_is_simply_absent(self):
        self.assertIsNone(DL.load(os.path.join(self.tmp, 'nope.npz')))

    def test_the_name_says_the_stride(self):
        self.assertTrue(DL.path_for('/x/y.edf', 64).endswith('.level64.npz'))
        self.assertTrue(DL.path_for('/x/y.edf', 256).endswith('.level256.npz'))


class AttachingIsOptional(unittest.TestCase):
    def test_a_missing_level_does_not_build_on_open_by_default(self):
        """Building is ~60 s on a day-long recording. Doing it while a
        reviewer waits for a file to open would be a worse regression than the
        lag it fixes."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        rec = os.path.join(tmp, 'rec.edf')
        open(rec, 'wb').write(b'x' * 16)
        src = _Source(ramp())
        self.assertIsNone(DL.attach(src, rec, FS))
        self.assertFalse(hasattr(src, 'level'))
        self.assertEqual(src.reads, 0)

    def test_it_attaches_when_asked_to_build(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        rec = os.path.join(tmp, 'rec.edf')
        open(rec, 'wb').write(b'x' * 16)
        src = _Source(ramp())
        level = DL.attach(src, rec, FS, build_if_missing=True)
        self.assertIsNotNone(level)
        self.assertIs(src.level, level)
        self.assertTrue(os.path.exists(DL.path_for(rec, 64)))


class _LazySource(object):
    """A day-long recording that costs nothing to hold.

    Samples are computed from their own index, so a 24-hour 19-channel source
    exists without the 6.5 GB it would really take — which is the only way to
    exercise the wide-zoom regime end to end on a machine this short of RAM.
    The waveform is deliberately not noise: a slow sine plus a fast one, so a
    decimated trace that was silently wrong in phase or amplitude shows up.
    """

    def __init__(self, n_ch, n, fs):
        self.shape = (n_ch, int(n))
        self.fs = float(fs)

    def _values(self, idx):
        t = idx / self.fs
        return (50.0 * np.sin(2 * np.pi * 0.05 * t)
                + 10.0 * np.sin(2 * np.pi * 3.0 * t)).astype(np.float32)

    def __getitem__(self, key):
        rows, cols = key
        idx = np.arange(*cols.indices(self.shape[1]), dtype=np.float64)
        base = self._values(idx)
        n_rows = (len(range(*rows.indices(self.shape[0])))
                  if isinstance(rows, slice) else 1)
        out = np.tile(base, (n_rows, 1))
        # A per-channel offset so the montage subtraction is not identically
        # zero, which would make every comparison below vacuously pass.
        out += np.arange(n_rows, dtype=np.float32)[:, None]
        return out


class ThePipelineDrawsTheSameRecordingEitherWay(unittest.TestCase):
    """End to end through DisplayPipeline, the thing the level actually feeds.

    The level is not expected to be bit-identical to the disk path: it serves
    every 64th sample, so where the requested stride is not a multiple of 64 it
    returns a neighbouring subset of real samples. What must hold is that the
    trace is the same signal — identical at the times both produce, and never
    a different waveform.
    """

    FS = 256
    HOURS = 6

    @classmethod
    def setUpClass(cls):
        from gui.display_pipeline import DisplayPipeline
        cls.DisplayPipeline = DisplayPipeline
        n = cls.FS * 3600 * cls.HOURS
        cls.src = _LazySource(19, n, cls.FS)
        cls.level = DL.build(cls.src, cls.FS, stride=64)
        cls.rows = ['EEG {}'.format(i) for i in range(19)]

    def _pipelines(self):
        args = (self.FS, 1.0, 70.0, 50.0, 'Longitudinal Bipolar', self.rows)
        plain = self.DisplayPipeline(self.src, *args)
        self.src.level = self.level
        fast = self.DisplayPipeline(self.src, *args)
        return plain, fast

    def tearDown(self):
        if hasattr(self.src, 'level'):
            del self.src.level

    def test_the_trace_matches_where_both_produce_a_point(self):
        plain, fast = self._pipelines()
        for span_s in (600.0, 1800.0, 3600.0, 3 * 3600.0):
            s1 = min(self.src.shape[1], int(span_s * self.FS))
            d_rows, d_t = plain.render(0, s1, 1600)
            l_rows, l_t = fast.render(0, s1, 1600)
            common, i_d, i_l = np.intersect1d(
                np.round(d_t, 6), np.round(l_t, 6), return_indices=True)
            self.assertTrue(len(common) > 0,
                            'no shared sample times at {:.0f} s'.format(span_s))
            worst = float(np.abs(d_rows[:, i_d] - l_rows[:, i_l]).max())
            self.assertLess(worst, 1e-3,
                            '{:.0f} s view differs by {:.4f} uV at shared '
                            'times'.format(span_s, worst))

    def test_it_does_not_slide_the_trace_sideways(self):
        """The level starts on its own stride grid. Building the time axis
        from s0 instead of the first sample actually taken would shift the
        whole trace by up to one level step — a quarter second here, and
        invisible without a check like this."""
        plain, fast = self._pipelines()
        s1 = int(3600 * self.FS)
        _, d_t = plain.render(0, s1, 1600)
        _, l_t = fast.render(0, s1, 1600)
        self.assertLess(abs(float(l_t[0]) - float(d_t[0])), 64 / self.FS)
        self.assertLess(abs(float(l_t[-1]) - float(d_t[-1])), 64 / self.FS)

    def test_a_review_zoom_still_goes_to_the_disk_path(self):
        """Below the wide regime the level must not be used at all: there the
        reviewer is reading morphology and every sample matters."""
        plain, fast = self._pipelines()
        s1 = int(30 * self.FS)
        d_rows, d_t = plain.render(0, s1, 1600)
        l_rows, l_t = fast.render(0, s1, 1600)
        np.testing.assert_array_equal(d_t, l_t)
        np.testing.assert_array_equal(d_rows, l_rows)

    def test_the_level_actually_gets_used(self):
        """Guards against the whole thing silently never engaging — which
        would make every comparison above pass for the wrong reason."""
        plain, fast = self._pipelines()
        s1 = int(3 * 3600 * self.FS)
        before = self.level.data.shape
        _, l_t = fast.render(0, s1, 1600)
        self.assertEqual(self.level.data.shape, before)
        # Times served from the level land on its stride grid.
        idx = np.round(np.asarray(l_t) * self.FS).astype(np.int64)
        self.assertTrue(np.all(idx % 64 == 0))


if __name__ == '__main__':
    unittest.main()
