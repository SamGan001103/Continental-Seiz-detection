"""The whole-recording map must not hide the things it exists to show.

`ProbStrip` is x-linked to `SignalView`, so it shows the page — ten seconds by
default. Until `OverviewStrip` existed, nothing in the application showed the
recording as a whole: on a 24-hour ambulatory study that is 8,640 pages with no
map, while `docs/INTENDED_USE.md` asks the reviewer to look at signal the
detector said nothing about.

Two properties carry the clinical weight, and both are ways this widget could
quietly lie rather than visibly break:

* **Reduction by maximum, never mean.** At 24 hours one pixel spans ~54 s.
  Averaging a 27 s seizure against quiet background halves it and can drop it
  under the threshold line — erasing exactly the mark the reviewer needs. The
  tests drive a single high second into a long quiet recording and demand it
  survives.

* **Unscored windows stay visible.** A window the pipeline refused carries no
  score, and at overview scale one refused window is a fraction of a pixel.
  "Not assessed" rendering as "nothing here" is the misreading the grey bands
  exist to prevent, so they are widened to a pixel floor.

Blind mode is tested here too: the overview is the one widget that shows the
whole recording at once, so ground truth surviving on it would defeat blinding
everywhere else.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    import PyQt5  # noqa: F401
    HAVE_QT = True
except Exception:                                            # noqa: BLE001
    HAVE_QT = False

if HAVE_QT:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtWidgets

from gui.widgets.overview_strip import (_merge_spans,        # noqa: E402
                                        _reduce_by_max)


class ReductionKeepsBriefEventsVisible(unittest.TestCase):
    """Pure arithmetic, no Qt: the property most likely to be got wrong."""

    def test_a_brief_spike_survives_a_day_long_reduction(self):
        dur = 86400.0
        t = np.arange(0, dur, 1.0)
        y = np.zeros_like(t)
        y[(t >= 43200) & (t < 43227)] = 0.97          # a 27 s seizure
        tx, yx = _reduce_by_max(t, y, dur, 1600)
        self.assertLessEqual(tx.size, 1600)
        self.assertGreaterEqual(
            yx.max(), 0.97 - 1e-9,
            'a 27 s event at 0.97 was flattened by the whole-day reduction — '
            'this is the mark the reviewer most needs to see')
        over = yx > 0.5
        self.assertTrue(over.any(), 'nothing crosses the threshold line')
        # and it must land in the right place
        self.assertAlmostEqual(float(tx[np.argmax(yx)]), 43200.0, delta=120.0)

    def test_the_mean_would_have_hidden_it(self):
        """Proves the choice of max is load-bearing, not stylistic."""
        dur = 86400.0
        t = np.arange(0, dur, 1.0)
        y = np.zeros_like(t)
        y[(t >= 43200) & (t < 43227)] = 0.97
        edges = np.linspace(0.0, dur, 1601)
        idx = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, 1599)
        sums = np.zeros(1600)
        counts = np.zeros(1600)
        np.add.at(sums, idx, y)
        np.add.at(counts, idx, 1.0)
        means = sums / np.maximum(counts, 1)
        self.assertLess(
            means.max(), 0.5,
            'averaging did NOT hide the event, so this test no longer '
            'demonstrates why the reduction takes a maximum')

    def test_short_recordings_are_not_reduced_at_all(self):
        t = np.arange(0, 300, 1.0)
        y = np.linspace(0, 1, t.size)
        tx, yx = _reduce_by_max(t, y, 300.0, 1600)
        np.testing.assert_array_equal(tx, t)
        np.testing.assert_array_equal(yx, y)

    def test_every_bucket_reports_its_own_maximum(self):
        dur = 1000.0
        t = np.arange(0, dur, 1.0)
        y = (t % 10 == 0).astype(float)      # a spike every 10 s
        tx, yx = _reduce_by_max(t, y, dur, 50)
        self.assertTrue(np.all(yx == 1.0),
                        'a bucket containing a spike reported less than it')


class SpanMerging(unittest.TestCase):
    def test_overlapping_spans_merge(self):
        self.assertEqual(_merge_spans([(0, 10), (5, 20), (30, 40)]),
                         [(0, 20), (30, 40)])

    def test_empty_is_empty(self):
        self.assertEqual(_merge_spans([]), [])

    def test_order_does_not_matter(self):
        self.assertEqual(_merge_spans([(30, 40), (0, 10), (5, 20)]),
                         [(0, 20), (30, 40)])


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheWidgetBehaves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(['-platform', 'offscreen']))

    def _strip(self, duration_s=3600.0, with_skips=True):
        from gui.widgets.overview_strip import OverviewStrip
        w = OverviewStrip()
        starts = np.arange(0, int(duration_s) - 12 + 1, 6, dtype=np.int32)
        probs = np.full(starts.size, 0.02, dtype=np.float32)
        probs[100:104] = 0.93
        skip = np.zeros(starts.size, dtype=np.int8)
        if with_skips:
            skip[200] = 1
        w.set_probs(starts, probs, segment_s=12, skip_code=skip,
                    duration_s=duration_s)
        return w

    def test_it_draws_the_whole_recording_not_the_page(self):
        w = self._strip()
        try:
            x0, x1 = w.plot_item().viewRange()[0]
            self.assertLessEqual(x0, 1.0)
            self.assertGreaterEqual(x1, 3599.0,
                                    'the overview is not showing the whole '
                                    'recording, which is its only job')
        finally:
            w.close()

    def test_a_single_unscored_window_is_still_drawn(self):
        w = self._strip(duration_s=86400.0)
        try:
            self.assertTrue(w._unscored_items,
                            'one refused window vanished at day scale — '
                            '"not assessed" must not read as "nothing here"')
        finally:
            w.close()

    def test_the_viewport_marker_tracks_the_page(self):
        w = self._strip()
        try:
            w.set_viewport(100.0, 110.0)
            a, b = w._viewport.getRegion()
            self.assertAlmostEqual(a, 100.0, places=3)
            self.assertAlmostEqual(b, 110.0, places=3)
        finally:
            w.close()

    def test_moving_the_marker_programmatically_does_not_emit_a_seek(self):
        """Otherwise the view moves the marker, which seeks the view, forever."""
        w = self._strip()
        seen = []
        w.seek_requested.connect(seen.append)
        try:
            w.set_viewport(500.0, 510.0)
            self.app.processEvents()
            self.assertEqual(seen, [],
                             'set_viewport echoed back a seek request — the '
                             'view and the marker will fight each other')
        finally:
            w.close()

    def test_dragging_the_marker_asks_for_a_seek(self):
        w = self._strip()
        seen = []
        w.seek_requested.connect(seen.append)
        try:
            w._viewport.setRegion((700.0, 730.0))
            w._on_marker_dragged()
            self.assertTrue(seen, 'dragging the marker requested no seek')
            self.assertAlmostEqual(seen[-1], 715.0, delta=1.0)
        finally:
            w.close()

    def test_blind_mode_hides_the_reference_bands(self):
        w = self._strip()
        try:
            w.set_references([(100.0, 130.0)])
            self.assertTrue(w._ref_items)
            w.set_references_visible(False)
            self.assertFalse(
                w._ref_items,
                'ground truth survived blind mode on the one widget that '
                'shows the whole recording at once')
            w.set_references([(200.0, 230.0)])
            self.assertFalse(w._ref_items,
                             'a later set_references re-drew ground truth '
                             'while blind mode was on')
        finally:
            w.close()

    def test_clear_empties_it(self):
        w = self._strip()
        try:
            w.clear()
            self.assertEqual(w._duration_s, 0.0)
            self.assertFalse(w._unscored_items)
        finally:
            w.close()

    def test_no_probabilities_is_not_a_crash(self):
        from gui.widgets.overview_strip import OverviewStrip
        w = OverviewStrip()
        try:
            w.set_probs([], [], duration_s=0.0)
        finally:
            w.close()


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheCaptionNamesTheDaysItCovers(unittest.TestCase):
    """This widget is always the WHOLE recording, so its caption is the whole
    recording's dates.

    Naming only the start date put "whole recording - 10 Jan 2004" under a map
    whose right-hand half was the 11th, directly below an axis that had the
    same bug. They now share one formatter so they cannot drift apart.
    """

    # The real start of the converted EPILEPSIAE day.
    START = __import__('datetime').datetime(2004, 1, 10, 18, 4, 44)

    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(['-platform', 'offscreen']))

    def _strip(self, duration_s):
        from gui.widgets.overview_strip import OverviewStrip
        w = OverviewStrip()
        starts = np.arange(0, int(duration_s) - 12 + 1, 6, dtype=np.int32)
        probs = np.full(starts.size, 0.02, dtype=np.float32)
        w.set_probs(starts, probs, segment_s=12, duration_s=duration_s)
        return w

    def _caption(self, w):
        return w.plot_item().getAxis('bottom').labelText

    def test_a_recording_inside_one_day_names_that_day(self):
        w = self._strip(3600.0)
        w.set_start_time(self.START)
        self.assertIn('10 Jan 2004', self._caption(w))

    def test_a_day_long_recording_names_both_days(self):
        w = self._strip(86400.0)
        w.set_start_time(self.START)
        self.assertIn('10–11 Jan 2004', self._caption(w))

    def test_the_duration_arriving_later_still_updates_it(self):
        """load_edf sets the start time before any probabilities exist."""
        from gui.widgets.overview_strip import OverviewStrip
        w = OverviewStrip()
        w.set_start_time(self.START)
        self.assertIn('10 Jan 2004', self._caption(w))
        starts = np.arange(0, 86400 - 12 + 1, 6, dtype=np.int32)
        w.set_probs(starts, np.full(starts.size, 0.02, dtype=np.float32),
                    segment_s=12, duration_s=86400.0)
        self.assertIn('10–11 Jan 2004', self._caption(w))

    def test_no_start_time_says_only_what_it_is(self):
        w = self._strip(86400.0)
        w.set_start_time(None)
        self.assertEqual(self._caption(w), 'whole recording')

    def test_it_agrees_with_the_axis_above_it(self):
        """Two captions, one formatter: the map and the trace must not
        disagree about which day the reviewer is looking at."""
        from gui.widgets.time_axis import axis_label
        w = self._strip(86400.0)
        w.set_start_time(self.START)
        above = axis_label(self.START, (0.0, 86400.0))
        self.assertIn(above.split('—')[1].strip(), self._caption(w))


class TheMainWindowWiresIt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(['-platform', 'offscreen']))

    def setUp(self):
        from gui.app import MainWindow
        self.w = MainWindow()

    def tearDown(self):
        self.w.close()

    def test_the_overview_is_not_x_linked_to_the_page(self):
        """If it were linked it would be a second copy of the ProbStrip."""
        sv_vb = self.w.signal_view._pw.getPlotItem().vb
        ov_vb = self.w.overview_strip.plot_item().vb
        linked = sv_vb.linkedView(0)
        self.assertIsNot(
            linked, ov_vb,
            'the overview is x-linked to the signal view, so it shows the '
            'page rather than the recording')

    def test_the_page_navigation_shortcuts_exist(self):
        from PyQt5 import QtGui
        seqs = {sc.key().toString()
                for sc in self.w.findChildren(QtWidgets.QShortcut)}
        for want in ('PgDown', 'PgUp', 'Ctrl+Home', 'Ctrl+End'):
            self.assertIn(
                QtGui.QKeySequence(want).toString(), seqs,
                '{} is not bound; reading a long recording end to end is '
                'what these are for'.format(want))

    def test_help_lists_the_new_keys(self):
        """Help > Keyboard shortcuts renders the module docstring."""
        import gui.app as ga
        for want in ('PgUp', 'Ctrl+Home', 'Ctrl+End'):
            self.assertIn(want, ga.__doc__,
                          '{} is bound but undocumented, so Help lies about '
                          'what the application does'.format(want))


if __name__ == '__main__':
    unittest.main()
