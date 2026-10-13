"""Showing interictal discharges next to the seizure score, honestly.

The clinician's question is whether the model scored highly where the patient
is actually spiking. That comparison only works if both are on one time axis,
which is why the rate lives under the score curve rather than in a strip of
its own.

The harder constraint is that the right representation CHANGES WITH ZOOM. A
discharge lasts about a tenth of a second. Across a day-long view one pixel
spans most of a minute, so an individual tick is not a discharge -- it is a
smear of hundreds -- and drawing one shows density while implying morphology.
So marks are drawn only where they can be read, and the rate curve carries the
information everywhere else.

These pin both halves of that rule.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    from PyQt5 import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.widgets.prob_strip import IED_BAND, ProbStrip, ied_rate
    from gui.widgets.signal_view import (MARKS_LEGIBLE_MAX_SPAN_S, SignalView)
    HAVE_QT = True
except Exception:                                    # pragma: no cover
    HAVE_QT = False

DAY = 86400.0


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheRateIsARateNotACount(unittest.TestCase):
    """Bin width changes with recording length, so a count would make a long
    study look busier than a short one with the same clinical burden."""

    def test_the_same_burden_reads_the_same_at_any_length(self):
        # The MEAN, not the median. With 240 bins across ten minutes a bin is
        # 2.5 s wide, so a discharge every ten seconds leaves three bins in
        # four empty and the median is correctly zero.
        short = ied_rate([i * 10.0 for i in range(60)], 600.0)[1]
        long_ = ied_rate([i * 10.0 for i in range(360)], 3600.0)[1]
        self.assertAlmostEqual(float(short.mean()), float(long_.mean()),
                               delta=0.3)
        self.assertAlmostEqual(float(short.mean()), 6.0, delta=0.3)

    def test_six_per_minute_reads_as_six(self):
        _, rate = ied_rate([i * 10.0 for i in range(60)], 600.0, bins=10)
        self.assertAlmostEqual(float(rate[0]), 6.0, delta=0.2)

    def test_an_empty_recording_is_all_zero(self):
        _, rate = ied_rate([], DAY)
        self.assertTrue((rate == 0).all())

    def test_a_zero_length_recording_does_not_divide_by_zero(self):
        edges, rate = ied_rate([1.0], 0.0)
        self.assertEqual(len(rate), 0)

    def test_marks_outside_the_recording_are_ignored(self):
        # Only the one at 300 s counts; a 60 s bin holding one mark is 1/min.
        _, rate = ied_rate([-50.0, 700.0, 300.0], 600.0, bins=10)
        self.assertAlmostEqual(float(rate.sum()), 1.0, delta=1e-6)
        self.assertEqual(int((rate > 0).sum()), 1)


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheRateSitsBelowTheScore(unittest.TestCase):
    """Two different quantities -- an uncalibrated network output per second,
    and events per minute -- must not share one scale, or a reader will take a
    spike rate off the threshold line."""

    def setUp(self):
        self.strip = ProbStrip()

    def test_the_score_axis_still_runs_zero_to_one(self):
        # viewRange carries pyqtgraph's own padding, so this checks the band
        # below zero was added without disturbing the top of the score axis.
        lo, hi = self.strip.plot_item().viewRange()[1]
        self.assertGreater(hi, 0.95)
        self.assertLess(hi, 1.15)
        self.assertLess(lo, 0.0)
        self.assertGreater(lo, -IED_BAND - 0.15)

    def test_the_rate_is_drawn_below_zero(self):
        self.strip.set_ied_density([10.0, 20.0, 30.0], 600.0)
        xs, ys = self.strip._ied_curve.getData()
        self.assertTrue(len(ys))
        self.assertLessEqual(float(np.nanmax(ys)), 0.0)
        self.assertGreaterEqual(float(np.nanmin(ys)), -IED_BAND - 1e-6)

    def test_it_reports_the_peak_rate_it_drew(self):
        peak = self.strip.set_ied_density([i * 10.0 for i in range(60)],
                                          600.0)
        self.assertGreater(peak, 0.0)

    def test_no_marks_draws_nothing_and_hides_the_label(self):
        self.assertEqual(self.strip.set_ied_density([], DAY), 0.0)
        xs, ys = self.strip._ied_curve.getData()
        self.assertFalse(len(xs or []))
        self.assertFalse(self.strip._ied_label.isVisible())

    def test_a_quiet_recording_is_still_visible(self):
        """Scaled to its own peak: absolute rates differ by an order of
        magnitude between patients, and a fixed scale flattens a quiet one
        into an empty band."""
        self.strip.set_ied_density([100.0, 40000.0], DAY)
        _, ys = self.strip._ied_curve.getData()
        self.assertLess(float(np.nanmin(ys)), -IED_BAND * 0.5)


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class MarksStepAsideWhenTheyCannotBeRead(unittest.TestCase):
    def setUp(self):
        self.sv = SignalView()
        self.sv.resize(1600, 900)
        self.sv._channel_names = ['Fp1-F7', 'F7-T3', 'T3-T5']
        self.sv._spacing = 100.0
        self.marks = [{'time': float(t), 'channel': 'Fp1-F7',
                       'source': 'reviewer'} for t in (10, 20, 30)]

    def _at(self, span):
        self.sv._pw.getPlotItem().setXRange(0, span, padding=0)
        return self.sv.set_ied_marks(self.marks)

    def test_a_review_page_draws_them(self):
        self.assertEqual(self._at(10.0), 3)

    def test_the_boundary_still_draws_them(self):
        self.assertEqual(self._at(MARKS_LEGIBLE_MAX_SPAN_S), 3)

    def test_a_whole_day_does_not(self):
        """At 1600 px a day puts 54 seconds in a pixel. A tick there is a
        smear of several hundred discharges drawn as one."""
        self.assertEqual(self._at(DAY), 0)

    def test_they_are_counted_as_hidden_not_as_absent(self):
        self._at(DAY)
        self.assertEqual(self.sv._ied_hidden, len(self.marks))

    def test_they_come_back_on_zooming_in(self):
        self._at(DAY)
        self.assertEqual(self._at(10.0), 3)


if __name__ == '__main__':
    unittest.main()
