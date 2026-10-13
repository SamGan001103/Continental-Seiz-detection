"""The on-demand display pipeline must draw what whole-file filtering drew.

`gui/display_pipeline.py` replaced "filter and montage the entire recording on
every control change" with "derive the visible span, now". That is the fix for
a real freeze: on a 24-hour recording the old path allocated five full-length
arrays per refresh (float64 filtering, montage, an inspector copy, a view copy)
and took a 16 GB machine down hard enough to need a power cycle.

The risk it introduces is subtle. Zero-phase filters have edge transients, so a
span filtered in isolation is NOT equal to the same span cut out of a
whole-file filtering — unless enough padding is filtered either side and then
discarded. If the padding is too small the reviewer sees a distorted trace at
the edges of every redraw and has no way to know. So the first test compares
against `apply_filters` over the full array and demands agreement to a
microvolt, and the second measures what the padding is actually worth by
removing it.

The overview regime (views wider than FULLRATE_MAX_SPAN_S) filters after
decimation and does NOT promise equality — it is a navigation view. What it
must do is stay finite, keep the montage right, and drop cutoffs the decimated
rate cannot represent instead of raising.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.display_pipeline import (DisplayPipeline, FILTER_PAD_S,   # noqa: E402
                                  FULLRATE_MAX_SPAN_S, SplicedSource)
from gui.processing import (CH_NAMES_19, apply_filters,            # noqa: E402
                            apply_montage)

FS = 250


def _eeg(seconds=400, seed=11):
    from scipy.ndimage import uniform_filter1d
    rng = np.random.RandomState(seed)
    x = rng.randn(19, seconds * FS).astype(np.float32)
    x = uniform_filter1d(x, size=7, axis=1) * 40.0
    # a slow drift the high-pass must remove, and mains the notch must remove
    t = np.arange(x.shape[1]) / float(FS)
    x += (60.0 * np.sin(2 * np.pi * 0.05 * t)).astype(np.float32)
    x += (8.0 * np.sin(2 * np.pi * 60.0 * t)).astype(np.float32)
    return x


class FullRateMatchesWholeFileFiltering(unittest.TestCase):
    """The regime a reviewer judges morphology in must be exact."""

    def setUp(self):
        self.data = _eeg()
        self.hp, self.lp, self.notch = 1.0, 70.0, 60.0

    def _reference(self, montage):
        filt = apply_filters(self.data, FS, hp=self.hp, lp=self.lp,
                             notch=self.notch)
        rows, labels = apply_montage(filt, montage)
        return rows, labels

    def test_matches_within_a_microvolt_for_every_montage(self):
        from gui.processing import MONTAGES
        for montage in MONTAGES:
            ref, labels = self._reference(montage)
            p = DisplayPipeline(self.data, FS, hp=self.hp, lp=self.lp,
                                notch=self.notch, montage=montage)
            self.assertEqual(list(p.labels), list(labels))
            s0, s1 = 100 * FS, 130 * FS          # a 30 s clinical page
            # target_px large enough that no decimation happens, so this
            # compares the filtering rather than the sampling.
            got, t = p.render(s0, s1, target_px=s1 - s0)
            self.assertEqual(got.shape[0], ref.shape[0])
            self.assertEqual(got.shape[1], s1 - s0)
            err = np.abs(got - ref[:, s0:s1]).max()
            self.assertLess(
                err, 1.0,
                '{}: span-filtered trace differs from whole-file filtering by '
                '{:.3f} uV — the reviewer would be judging a distorted '
                'waveform'.format(montage, err))
            self.assertAlmostEqual(float(t[0]), s0 / float(FS), places=4)

    def test_the_padding_is_what_makes_it_match(self):
        """Strip the padding and the match should break — proving it is earned.

        TWO things pad the filtering now, and both have to be neutralised or
        this test passes for the wrong reason. FILTER_PAD_S is the explicit
        margin; the block cache is the accidental one, because it filters a
        region several screens wide and the requested span sits in the middle
        of it. When the block cache was added this test began passing with
        FILTER_PAD_S = 0 — the widened block was doing the padding's job.

        So the block is shrunk to exactly the requested span as well, which is
        the naive implementation this whole design exists to avoid.
        """
        import gui.display_pipeline as dp
        ref, _ = self._reference('Longitudinal Bipolar')
        s0, s1 = 100 * FS, 130 * FS
        saved = (dp.FILTER_PAD_S, dp.BLOCK_SPAN_FACTOR, dp.BLOCK_MIN_S)
        try:
            dp.FILTER_PAD_S = 0.0
            dp.BLOCK_SPAN_FACTOR = 1
            dp.BLOCK_MIN_S = 0.0
            p = DisplayPipeline(self.data, FS, hp=self.hp, lp=self.lp,
                                notch=self.notch, montage='Longitudinal Bipolar')
            unpadded, _ = p.render(s0, s1, target_px=s1 - s0)
        finally:
            (dp.FILTER_PAD_S, dp.BLOCK_SPAN_FACTOR,
             dp.BLOCK_MIN_S) = saved
        err = np.abs(unpadded - ref[:, s0:s1]).max()
        self.assertGreater(
            err, 1.0,
            'filtering with no padding and no block widening matched anyway, '
            'so this test proves nothing — check FILTER_PAD_S is really used')

    def test_the_block_cache_returns_the_same_samples_as_a_fresh_render(self):
        """Panning inside a cached block must not change what is drawn."""
        p = DisplayPipeline(self.data, FS, hp=self.hp, lp=self.lp,
                            notch=self.notch, montage='Longitudinal Bipolar')
        first, t1 = p.render(100 * FS, 130 * FS, target_px=30 * FS)
        # a later span inside the same block — served from cache
        second, t2 = p.render(140 * FS, 170 * FS, target_px=30 * FS)
        fresh = DisplayPipeline(self.data, FS, hp=self.hp, lp=self.lp,
                                notch=self.notch,
                                montage='Longitudinal Bipolar')
        expect, te = fresh.render(140 * FS, 170 * FS, target_px=30 * FS)
        np.testing.assert_allclose(second, expect, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(t2, te)

    def test_changing_a_filter_invalidates_the_block(self):
        """A stale block would draw the previous filter settings."""
        p = DisplayPipeline(self.data, FS, hp=1.0, lp=70.0, notch=60.0,
                            montage='Longitudinal Bipolar')
        a, _ = p.render(100 * FS, 130 * FS, target_px=30 * FS)
        p.lp = 15.0                      # as MainWindow rebuilds it
        q = DisplayPipeline(self.data, FS, hp=1.0, lp=15.0, notch=60.0,
                            montage='Longitudinal Bipolar')
        b, _ = q.render(100 * FS, 130 * FS, target_px=30 * FS)
        c, _ = p.render(100 * FS, 130 * FS, target_px=30 * FS)
        self.assertGreater(np.abs(a - b).max(), 1.0,
                           'a 15 Hz low-pass should differ from 70 Hz')
        np.testing.assert_allclose(c, b, rtol=0, atol=1e-4)

    def test_a_span_at_the_recording_start_is_still_correct(self):
        """No padding is available before t=0; the result must still be sane."""
        ref, _ = self._reference('Referential (raw)')
        p = DisplayPipeline(self.data, FS, hp=self.hp, lp=self.lp,
                            notch=self.notch, montage='Referential (raw)')
        got, _t = p.render(0, 30 * FS, target_px=30 * FS)
        # The first seconds legitimately differ (neither has more signal to
        # use), but past one pad-length in they must agree.
        k = int(FILTER_PAD_S * FS)
        err = np.abs(got[:, k:] - ref[:, k:30 * FS]).max()
        self.assertLess(err, 1.0, 'start-of-recording span diverges')

    def test_decimation_reduces_points_without_changing_span(self):
        p = DisplayPipeline(self.data, FS, hp=self.hp, lp=self.lp,
                            notch=self.notch, montage='Longitudinal Bipolar')
        got, t = p.render(50 * FS, 110 * FS, target_px=1200)
        self.assertLessEqual(got.shape[1], 1300)
        self.assertGreater(got.shape[1], 600)
        self.assertAlmostEqual(float(t[0]), 50.0, places=3)
        self.assertLess(float(t[-1]), 110.0 + 1e-6)


class OverviewRegimeStaysSafe(unittest.TestCase):
    """Wide views draw the min/max of the FILTERED signal, per column.

    Until 2026-09-01 they decimated first and filtered the decimated series,
    which lost two things at once. A spike between the selected samples was
    never drawn - at a 1818 s view, one sample in 187 - and every cutoff above
    the decimated Nyquist was dropped, so a view the toolbar labelled
    HP 1.0 / LP 70 / Notch 60 was in fact unfiltered and aliased.
    """

    def setUp(self):
        self.data = _eeg(seconds=1200)

    def test_a_very_wide_view_renders_and_is_finite(self):
        p = DisplayPipeline(self.data, FS, hp=1.0, lp=70.0, notch=60.0,
                            montage='Longitudinal Bipolar')
        span = int(1200 * FS)
        self.assertGreater(1200.0, FULLRATE_MAX_SPAN_S)
        got, t = p.render(0, span, target_px=1600)
        self.assertEqual(got.shape[0], 18)
        self.assertTrue(np.all(np.isfinite(got)), 'overview produced non-finite')
        self.assertLessEqual(got.shape[1], 2000)
        self.assertGreater(t[-1], 1000.0)

    def test_a_spike_narrower_than_a_column_is_still_drawn(self):
        """The property the whole change exists for.

        A 40 ms, 400 uV transient at 600 s, on a 1200 s view where one column
        spans several seconds. Stride decimation sampled every 187th sample
        and missed it completely; the envelope cannot, because the column
        carries the extremes of every sample inside it.
        """
        data = _eeg(seconds=1200)
        at = 600 * FS
        data[:, at:at + 10] += 400.0
        p = DisplayPipeline(data, FS, hp=1.0, lp=70.0, notch=60.0,
                            montage='Referential (raw)')
        got, _t = p.render(0, int(1200 * FS), target_px=1600)
        self.assertGreater(
            float(np.abs(got).max()), 200.0,
            'a 400 uV spike vanished from the overview - a reviewer scrolling '
            'the recording would never see it')

    def test_the_filters_the_toolbar_names_are_actually_applied(self):
        """Overview used to drop every cutoff above the decimated Nyquist."""
        n = 1200 * FS
        t = np.arange(n) / float(FS)
        data = (_eeg(seconds=1200)
                + 60.0 * np.sin(2 * np.pi * 60.0 * t)[None, :])
        common = dict(hp=1.0, lp=70.0, montage='Referential (raw)')
        notched, _ = DisplayPipeline(data, FS, notch=60.0, **common).render(
            0, n, target_px=1600)
        plain, _ = DisplayPipeline(data, FS, notch=None, **common).render(
            0, n, target_px=1600)
        self.assertLess(
            float(np.abs(notched).max()), float(np.abs(plain).max()),
            'the 60 Hz notch made no difference to the overview, so the '
            'filters named in the toolbar are not being applied')

    def test_aggregating_the_stored_envelope_matches_filtering_directly(self):
        """Composition must be exact, or the cache is a second answer.

        min/max composes: the minimum of a set of column minima is the
        minimum of their union. So aggregating the stored base envelope has to
        give the same extremes as filtering the span and reducing it in one
        go, otherwise which one you see depends on cache state.
        """
        data = _eeg(seconds=900)
        p = DisplayPipeline(data, FS, hp=1.0, lp=70.0, notch=60.0,
                            montage='Referential (raw)')
        s0, s1, bins = 100 * FS, 800 * FS, 300
        cached, _ = p.render(s0, s1, target_px=2 * bins)
        direct, _ = p._render_envelope_direct(s0, s1, bins)
        span = min(cached.shape[1], direct.shape[1])
        err = np.abs(cached[:, :span].max(axis=1)
                     - direct[:, :span].max(axis=1)).max()
        self.assertLess(err, 1.0,
                        'aggregated envelope disagrees with a direct one by '
                        '{:.3f} uV'.format(err))

    def test_the_regime_boundary_is_continuous_enough_to_navigate(self):
        """Crossing the threshold must not change what the trace looks like."""
        p = DisplayPipeline(self.data, FS, hp=1.0, lp=70.0, notch=60.0,
                            montage='Referential (raw)')
        s0 = 100 * FS
        narrow, _ = p.render(s0, s0 + int((FULLRATE_MAX_SPAN_S - 5) * FS), 1200)
        wide, _ = p.render(s0, s0 + int((FULLRATE_MAX_SPAN_S + 5) * FS), 1200)
        # Compare RMS per row: same signal, same order of magnitude.
        rn = np.sqrt((narrow.astype(np.float64) ** 2).mean(axis=1))
        rw = np.sqrt((wide.astype(np.float64) ** 2).mean(axis=1))
        ratio = rw / np.maximum(rn, 1e-9)
        self.assertTrue(
            np.all((ratio > 0.4) & (ratio < 2.5)),
            'the trace jumps in amplitude when the view crosses the '
            'full-rate/overview boundary: ratios {}'.format(ratio))


class ChannelSlicesForTheInspector(unittest.TestCase):
    def setUp(self):
        self.data = _eeg(seconds=300)

    def test_one_row_matches_the_full_render_of_that_row(self):
        for montage in ('Referential (raw)', 'Longitudinal Bipolar',
                        'Common Average'):
            p = DisplayPipeline(self.data, FS, hp=1.0, lp=70.0, notch=60.0,
                                montage=montage)
            s0, s1 = 60 * FS, 120 * FS
            rows, _ = p.render(s0, s1, target_px=s1 - s0)
            vals, t = p.channel(3, s0, s1)
            err = np.abs(vals - rows[3]).max()
            self.assertLess(err, 1e-3,
                            '{}: inspector slice differs from the drawn '
                            'trace'.format(montage))
            self.assertAlmostEqual(float(t[0]), 60.0, places=4)


class SplicedSourceOverlaysWithoutCopying(unittest.TestCase):
    """The ICA view's overlay must never touch the detector's array."""

    def setUp(self):
        self.raw = _eeg(seconds=120)
        self.block = np.full((19, 30 * FS), 7.0, dtype=np.float32)
        self.i0 = 40 * FS
        self.src = SplicedSource(self.raw, self.i0, self.block)

    def test_the_raw_array_is_never_written(self):
        before = self.raw.copy()
        _ = self.src[:, 0:self.raw.shape[1]]
        np.testing.assert_array_equal(self.raw, before)

    def test_slices_show_the_override_where_it_applies(self):
        inside = self.src[:, self.i0 + 10:self.i0 + 20]
        np.testing.assert_array_equal(inside, np.full((19, 10), 7.0,
                                                      dtype=np.float32))
        before = self.src[:, 0:100]
        np.testing.assert_array_equal(before, self.raw[:, 0:100])

    def test_a_slice_straddling_the_boundary_is_stitched(self):
        s0, s1 = self.i0 - 50, self.i0 + 50
        got = self.src[:, s0:s1]
        np.testing.assert_array_equal(got[:, :50], self.raw[:, s0:self.i0])
        np.testing.assert_array_equal(got[:, 50:], np.full((19, 50), 7.0,
                                                           dtype=np.float32))

    def test_strided_access_lands_on_the_right_samples(self):
        step = 7
        got = self.src[:, 0:self.raw.shape[1]:step]
        ref = np.asarray(self.src)[:, ::step]
        np.testing.assert_array_equal(got, ref)

    def test_single_row_access_works(self):
        row = self.src[4, self.i0 - 10:self.i0 + 10]
        self.assertEqual(row.shape, (20,))
        np.testing.assert_array_equal(row[10:], np.full(10, 7.0,
                                                        dtype=np.float32))

    def test_the_pipeline_reads_it_like_an_array(self):
        p = DisplayPipeline(self.src, FS, hp=1.0, lp=70.0, notch=60.0,
                            montage='Longitudinal Bipolar')
        got, _t = p.render(self.i0 - 5 * FS, self.i0 + 5 * FS, target_px=4000)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertEqual(got.shape[0], 18)


class AskingWhetherADrawWillBeExpensive(unittest.TestCase):
    """The redraw path calls render() on the GUI thread. In the overview
    regime, with no prepared envelope, that call filters the whole recording
    and does not return for ~40 s on a recording-day -- the window goes white
    and Windows marks it Not Responding. So the view asks first, and the
    question itself has to be cheap: no 36 MB read to answer it.
    """

    class Source(object):
        """An array with a path, which is what a real EdfSource is here."""

        def __init__(self, data, path):
            self._d = data
            self.path = path
            self.shape = data.shape
            self.fs = FS

        def __getitem__(self, k):
            return self._d[k]

    def setUp(self):
        import shutil
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rec = os.path.join(self.tmp, 'rec.edf')
        with open(self.rec, 'wb') as fh:
            fh.write(b'x' * 4096)
        self.data = _eeg(seconds=int(FULLRATE_MAX_SPAN_S * 4))
        self.args = dict(hp=1.0, lp=70.0, notch=60.0,
                         montage='Longitudinal Bipolar')

    def _pipe(self, with_path=True):
        src = self.Source(self.data, self.rec) if with_path else self.data
        return DisplayPipeline(src, FS, **self.args)

    def _wide(self, p):
        return 0, min(p.n_samples, int(FULLRATE_MAX_SPAN_S * 3 * FS))

    def test_a_reading_page_is_never_expensive(self):
        p = self._pipe()
        self.assertFalse(p.needs_envelope_build(0, int(10 * FS)))
        self.assertFalse(
            p.needs_envelope_build(0, int(FULLRATE_MAX_SPAN_S * FS)))

    def test_an_unprepared_overview_is_expensive(self):
        p = self._pipe()
        self.assertTrue(p.needs_envelope_build(*self._wide(p)))

    def test_it_is_not_expensive_twice(self):
        p = self._pipe()
        a, b = self._wide(p)
        p.render(a, b, 1600)
        self.assertFalse(p.needs_envelope_build(a, b))

    def test_a_prepared_envelope_makes_it_cheap_from_the_first_draw(self):
        from gui.io.envelope_cache import save, settings_key
        built = self._pipe()
        key, step, lo, hi = built.build_envelope_for_cache()
        save(self.rec, settings_key(*key), step, lo, hi)
        fresh = self._pipe()
        self.assertFalse(fresh.needs_envelope_build(*self._wide(fresh)))

    def test_an_envelope_for_other_settings_does_not_count(self):
        from gui.io.envelope_cache import save, settings_key
        built = DisplayPipeline(self.Source(self.data, self.rec), FS,
                                hp=0.5, lp=35.0, notch=60.0,
                                montage='Transverse Bipolar')
        key, step, lo, hi = built.build_envelope_for_cache()
        save(self.rec, settings_key(*key), step, lo, hi)
        fresh = self._pipe()
        self.assertTrue(fresh.needs_envelope_build(*self._wide(fresh)))

    def test_a_source_with_no_path_is_assumed_expensive(self):
        """Nowhere to have cached it, so the honest answer is yes."""
        p = self._pipe(with_path=False)
        self.assertTrue(p.needs_envelope_build(*self._wide(p)))

    def test_an_empty_span_is_not_expensive(self):
        p = self._pipe()
        self.assertFalse(p.needs_envelope_build(500, 500))
        self.assertFalse(p.needs_envelope_build(800, 300))


if __name__ == '__main__':
    unittest.main()
