"""Zoomed out, the lanes must not all clip flat.

Past FULLRATE_MAX_SPAN_S each pixel is the minimum and maximum of several
seconds rather than one sample, so the range drawn is far wider than a reading
page's. At a reading sensitivity nearly all of it lands on the lane edge and
every row reads as a solid grey block.

Measured on pat_21902_part02 at a 2.4 hour view, 7 uV/mm: per-pixel range
median 52 uV, p75 121, p90 198, against a lane holding 77 uV peak to peak --
38% of drawn pixels clipped. At 30 uV/mm, 4.8%.

The rules these tests hold:

* the scale is fitted to what is about to be drawn, not to a guess;
* it changes only when the view CROSSES the boundary, so a reviewer who sets a
  scale while zoomed out keeps it;
* the reviewer's reading scale is remembered and restored on the way back in;
* the toolbar always shows the number the trace is being drawn against,
  because a sensitivity that moved silently would be the kind of lie a
  clinical display must not tell.
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
    from gui.app import SENSITIVITIES
    from gui.display_pipeline import FULLRATE_MAX_SPAN_S
    from gui.widgets.signal_view import SignalView
    HAVE_QT = True
except Exception:                                        # pragma: no cover
    HAVE_QT = False

FS = 100
N_ROWS = 6


class _Provider(object):
    """Draws a fixed amplitude, so the fitted scale is predictable."""

    def __init__(self, amplitude_uv, seconds=4 * 3600.0):
        self.amplitude = float(amplitude_uv)
        self.n_rows = N_ROWS
        self.fs = FS
        self.n_samples = int(seconds * FS)
        self.duration_s = float(seconds)

    def needs_envelope_build(self, s0, s1):
        return False

    def render(self, s0, s1, target_px):
        m = max(2, min(256, int(target_px)))
        rows = np.zeros((N_ROWS, m), dtype=np.float32)
        rows[:, ::2] = self.amplitude          # alternating min/max, as the
        rows[:, 1::2] = -self.amplitude        # overview regime really returns
        return rows, np.linspace(s0 / float(FS), s1 / float(FS), m,
                                 dtype=np.float32)


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheScaleFollowsTheZoom(unittest.TestCase):
    def setUp(self):
        self.view = SignalView()
        self.addCleanup(self.view.deleteLater)
        self.scale_events = []
        self.sens_events = []
        self.view.overviewScaleChanged.connect(
            lambda v, out: self.scale_events.append((v, out)))
        self.view.sensitivityChanged.connect(
            lambda v, done: self.sens_events.append(v))

    def _load(self, amplitude_uv=300.0, sensitivity=7.0):
        self.view.set_provider(
            _Provider(amplitude_uv), FS,
            ['r{}'.format(i) for i in range(N_ROWS)],
            sensitivity_uv=sensitivity)
        # The toolbar owns the value; mirror what the window does with it.
        self.view.sensitivityChanged.connect(
            lambda v, done: self.view.set_sensitivity(v))
        del self.scale_events[:], self.sens_events[:]

    def _look_at(self, t0, t1):
        self.view._pw.getPlotItem().setXRange(t0, t1, padding=0)
        self.view._redecimate_visible()

    def test_a_reading_page_is_left_alone(self):
        self._load()
        self._look_at(0.0, 10.0)
        self.assertEqual(self.scale_events, [])
        self.assertEqual(self.view._sensitivity_uv, 7.0)

    def test_zooming_out_widens_the_scale(self):
        self._load(amplitude_uv=300.0)
        self._look_at(0.0, FULLRATE_MAX_SPAN_S * 4)
        self.assertTrue(self.scale_events, 'the scale did not follow the zoom')
        value, zooming_out = self.scale_events[0]
        self.assertTrue(zooming_out)
        self.assertGreater(value, 7.0)
        self.assertIn(value, [float(v) for v in SENSITIVITIES])

    def test_the_new_scale_actually_fits_what_is_drawn(self):
        """300 uV peaks need a lane of at least 600 uV peak to peak."""
        self._load(amplitude_uv=300.0)
        self._look_at(0.0, FULLRATE_MAX_SPAN_S * 4)
        value = self.scale_events[0][0]
        mm = self.view._row_mm(N_ROWS)
        self.assertGreaterEqual(
            0.5 * value * mm, 300.0 * 0.95,
            'the fitted lane still clips the signal it was fitted to')

    def test_a_quiet_recording_is_not_widened_pointlessly(self):
        """The fit is to the recording, not to a fixed overview number."""
        self._load(amplitude_uv=1.0)
        self._look_at(0.0, FULLRATE_MAX_SPAN_S * 4)
        if self.scale_events:
            self.assertLessEqual(self.scale_events[0][0], 7.0)

    def test_zooming_back_in_restores_the_reviewers_scale(self):
        self._load(amplitude_uv=300.0, sensitivity=7.0)
        self._look_at(0.0, FULLRATE_MAX_SPAN_S * 4)
        del self.scale_events[:]
        self._look_at(0.0, 10.0)
        self.assertTrue(self.scale_events, 'the reading scale never came back')
        value, zooming_out = self.scale_events[0]
        self.assertFalse(zooming_out)
        self.assertEqual(value, 7.0)

    def test_it_does_not_fire_again_inside_the_same_regime(self):
        """A reviewer who picks a scale while zoomed out must keep it."""
        self._load(amplitude_uv=300.0)
        self._look_at(0.0, FULLRATE_MAX_SPAN_S * 4)
        del self.scale_events[:]
        self._look_at(0.0, FULLRATE_MAX_SPAN_S * 8)
        self._look_at(100.0, 100.0 + FULLRATE_MAX_SPAN_S * 6)
        self.assertEqual(self.scale_events, [],
                         'the scale was re-fitted while already zoomed out, '
                         'overriding a choice the reviewer had made')

    def test_the_scale_it_asks_for_is_one_the_toolbar_can_show(self):
        """The displayed number must be the number being drawn, so it has to
        be a value the combo box actually holds."""
        self._load(amplitude_uv=900.0)
        self._look_at(0.0, FULLRATE_MAX_SPAN_S * 4)
        for value, _out in self.scale_events:
            self.assertIn(value, [float(v) for v in SENSITIVITIES])

    def test_an_empty_render_does_not_change_anything(self):
        self._load()
        self.assertIsNone(self.view._needed_sensitivity(
            np.zeros((0, 0), dtype=np.float32)))

    def test_all_clipped_data_asks_for_the_widest_scale(self):
        self._load()
        huge = np.full((N_ROWS, 64), 50000.0, dtype=np.float32)
        self.assertEqual(self.view._needed_sensitivity(huge),
                         float(max(SENSITIVITIES)))

    def test_infinities_in_the_render_do_not_break_the_fit(self):
        self._load()
        data = np.full((N_ROWS, 64), 20.0, dtype=np.float32)
        data[0, 0] = np.inf
        data[0, 1] = np.nan
        self.assertIsNotNone(self.view._needed_sensitivity(data))


if __name__ == '__main__':
    unittest.main()
