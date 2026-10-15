"""The detector-input view must not read the whole recording to show 12 s of it.

`EdfSource` is array-like on purpose: it has shape, dtype and ndim, and slicing
it reads just those samples off disk. It also implements `__array__`, which
reads the ENTIRE recording -- 1.7 GB and a minute of disk for a day-long
study -- so any `np.asarray()` on the way to a 12-second window undoes the
whole point of streaming.

Two places did exactly that: `_validate` at the top of the display-ICA path,
and `SplicedSource`, which overlays the cleaned span on the raw trace. Neither
needs an array. They need shape and a slice.

This went unnoticed because the path was unreachable: the view declines unless
a recording is served at the detector's 250 Hz, and a 250 Hz recording was
always loaded whole because it fitted in memory. Once the open policy started
streaming anything longer than an hour, ticking the ICA box on a 250 Hz
ambulatory recording became a silent 1.7 GB read on the GUI thread -- the
freeze that policy exists to prevent, moved behind a checkbox.

So the rule these tests hold is: nothing on the display-ICA path converts its
source. A reader that refuses to be converted must still work.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.display_pipeline import SplicedSource              # noqa: E402
from gui.io.ica_display import (SEGMENT_S, clean_span_for_display,  # noqa: E402
                                cache_clear)

FS = 250
N_ROWS = 19


class Streamed(object):
    """An EdfSource's shape: array-like, sliceable, and expensive to convert.

    `__array__` does not return the recording -- it fails the test. That is
    the point: a full conversion is the bug, so it must not be reachable by
    accident, and a spy that merely counted would let the bug pass as long as
    nobody looked at the counter.
    """

    def __init__(self, data):
        self._d = np.asarray(data, dtype=np.float32)
        self.shape = self._d.shape
        self.dtype = self._d.dtype
        self.ndim = 2
        self.reads = []

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        rows, cols = key
        self.reads.append((cols.start, cols.stop)
                          if isinstance(cols, slice) else cols)
        return self._d[key]

    def __array__(self, dtype=None):
        raise AssertionError(
            'the whole recording was converted to an array; on a day-long '
            'study that is a 1.7 GB read on the GUI thread')


def _eeg(seconds, seed=5):
    rng = np.random.RandomState(seed)
    x = rng.randn(N_ROWS, int(seconds * FS)).astype(np.float32) * 20.0
    t = np.arange(x.shape[1]) / float(FS)
    x += (40.0 * np.sin(2 * np.pi * 0.3 * t)).astype(np.float32)
    return x


class TheCleanedSpanComesFromSlices(unittest.TestCase):
    def setUp(self):
        cache_clear()
        self.addCleanup(cache_clear)
        self.data = _eeg(4 * SEGMENT_S)

    def test_a_span_is_cleaned_without_converting_the_source(self):
        src = Streamed(self.data)
        span, start_s, blocks = clean_span_for_display(
            src, FS, 0.0, float(SEGMENT_S), recording_id='streamed-test')
        self.assertEqual(span.shape[0], N_ROWS)
        self.assertEqual(span.shape[1], SEGMENT_S * FS)
        self.assertTrue(blocks)
        self.assertTrue(np.all(np.isfinite(span)))

    def test_it_reads_only_what_the_span_needs(self):
        """A 12 s span out of a 48 s recording must not read 48 s."""
        src = Streamed(self.data)
        clean_span_for_display(src, FS, 0.0, float(SEGMENT_S),
                               recording_id='streamed-reads')
        widest = max((b or 0) - (a or 0) for a, b in src.reads)
        self.assertLessEqual(
            widest, 2 * SEGMENT_S * FS,
            'the display ICA read {} samples to show {}'.format(
                widest, SEGMENT_S * FS))

    def test_the_result_matches_what_an_array_source_gives(self):
        """Laziness must not change the trace, only how it was fetched."""
        a, _s, _b = clean_span_for_display(
            Streamed(self.data), FS, 0.0, float(SEGMENT_S),
            recording_id='same-a')
        cache_clear()
        b, _s, _b = clean_span_for_display(
            self.data, FS, 0.0, float(SEGMENT_S), recording_id='same-b')
        np.testing.assert_array_equal(a, b)

    def test_a_plain_nested_list_is_still_accepted(self):
        """Not every caller passes something array-like; those still convert."""
        small = _eeg(SEGMENT_S)[:, :SEGMENT_S * FS].tolist()
        span, _s, _b = clean_span_for_display(
            small, FS, 0.0, float(SEGMENT_S), recording_id='list-src')
        self.assertEqual(span.shape[0], N_ROWS)

    def test_the_wrong_channel_count_is_still_refused(self):
        bad = Streamed(_eeg(SEGMENT_S)[:4])
        with self.assertRaises(RuntimeError):
            clean_span_for_display(bad, FS, 0.0, float(SEGMENT_S))


class TheOverlayDoesNotConvertEither(unittest.TestCase):
    def setUp(self):
        self.data = _eeg(4 * SEGMENT_S)
        self.block = np.ones((N_ROWS, SEGMENT_S * FS), dtype=np.float32)

    def test_a_streamed_source_can_be_spliced(self):
        src = SplicedSource(Streamed(self.data), 0, self.block)
        self.assertEqual(src.shape, self.data.shape)
        self.assertEqual(src.ndim, 2)

    def test_the_overlay_is_what_is_read_back(self):
        src = SplicedSource(Streamed(self.data), 0, self.block)
        got = src[:, :SEGMENT_S * FS]
        np.testing.assert_array_equal(got, self.block)

    def test_outside_the_overlay_the_recording_is_read_back(self):
        i0 = SEGMENT_S * FS
        src = SplicedSource(Streamed(self.data), i0, self.block)
        got = src[:, 0:i0]
        np.testing.assert_array_equal(got, self.data[:, 0:i0])

    def test_a_span_outside_the_recording_is_still_refused(self):
        with self.assertRaises(ValueError):
            SplicedSource(Streamed(self.data), self.data.shape[1], self.block)

    def test_a_plain_array_still_works(self):
        src = SplicedSource(self.data, 0, self.block)
        np.testing.assert_array_equal(src[:, :SEGMENT_S * FS], self.block)


if __name__ == '__main__':
    unittest.main()
