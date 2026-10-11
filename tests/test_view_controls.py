"""The sweep control must state the span that is actually on screen.

It was write-only: it set the span and was never told when anything else
changed it. Zoom is free and unmodified — a wheel tick, or a trackpad
two-finger scroll — so the view moved constantly while the control did not,
and after a scroll to the whole recording it read `10` over a view of 1818 s.
That is wrong by two orders of magnitude, in the direction that flatters the
view: it under-reported the span, making the display look more trustworthy
than it was, and the span is the number a reviewer uses to judge whether they
are looking at morphology or at a navigation view.

Free zoom on an unmodified wheel is deliberate and is tested here too, because
it was briefly taken away and should not be again by accident: it is how this
view is driven, and the danger that motivated removing it is now handled where
it belongs — the wide view draws the min/max of the filtered signal instead of
one sample in 187, and says so with a lighter pen.
"""
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    from PyQt5 import QtCore, QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    import pyqtgraph as pg
    from gui.app import MainWindow, TIMEBASES
    from gui.widgets.signal_view import _SignalViewBox
    HAVE_QT = True
except Exception:                                        # pragma: no cover
    HAVE_QT = False


class _Wheel(object):
    """Minimal stand-in for QWheelEvent: only what wheelEvent reads."""

    def __init__(self, dy, modifiers=None):
        self._dy = dy
        self._mods = (modifiers if modifiers is not None
                      else QtCore.Qt.NoModifier)
        self.accepted = False

    def angleDelta(self):
        return QtCore.QPoint(0, self._dy)

    def modifiers(self):
        return self._mods

    def pos(self):
        return QtCore.QPointF(0.0, 0.0)

    def accept(self):
        self.accepted = True


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class ScrollZoomsWithoutAModifier(unittest.TestCase):
    """Free zoom is the primary gesture; keep it that way."""

    def setUp(self):
        self.vb = _SignalViewBox()
        self.pw = pg.PlotWidget(viewBox=self.vb)
        self.vb.setXRange(100.0, 110.0, padding=0)

    def _span(self):
        (x0, x1), _ = self.vb.viewRange()
        return x1 - x0

    def test_a_plain_scroll_zooms_out(self):
        before = self._span()
        self.vb.wheelEvent(_Wheel(-120))
        self.assertGreater(
            self._span(), before * 1.02,
            'an unmodified scroll no longer zooms — that is the gesture this '
            'view is driven by, and requiring a modifier for it was reverted '
            'on purpose')

    def test_a_plain_scroll_the_other_way_zooms_in(self):
        before = self._span()
        self.vb.wheelEvent(_Wheel(120))
        self.assertLess(self._span(), before * 0.99)

    def test_zooming_is_gentler_than_pyqtgraph_default(self):
        """~8% per tick, not ~12%, so a zoom does not lurch."""
        before = self._span()
        self.vb.wheelEvent(_Wheel(-120))
        self.assertLess(self._span() / before, 1.10)


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheSweepControlStatesWhatIsOnScreen(unittest.TestCase):
    def setUp(self):
        self.w = MainWindow()

    def _shown(self):
        return self.w.cb_tb.currentText()

    def test_a_preset_span_reads_as_that_preset(self):
        for v in (10, 30, 300):
            self.w._sync_timebase_combo(float(v))
            self.assertEqual(self._shown(), str(v))

    def test_a_span_that_is_not_a_preset_reads_as_the_real_span(self):
        """The 1818 s case that read as `10`."""
        self.w._sync_timebase_combo(1818.1)
        shown = self._shown()
        self.assertIn('1818', shown,
                      'the sweep control does not state the span on screen; '
                      'it shows {!r}'.format(shown))
        self.assertNotIn(shown, [str(v) for v in TIMEBASES],
                         'a non-preset span is being shown as a preset')

    def test_going_back_to_a_preset_removes_the_transient_entry(self):
        self.w._sync_timebase_combo(1818.1)
        n_custom = self.w.cb_tb.count()
        self.w._sync_timebase_combo(10.0)
        self.assertEqual(self._shown(), '10')
        self.assertLess(self.w.cb_tb.count(), n_custom,
                        'the transient view entry outlived the view it '
                        'described')

    def test_the_transient_entry_cannot_be_applied_as_a_timebase(self):
        """Re-selecting it must be refused, not parsed into nonsense."""
        self.w._sync_timebase_combo(1818.1)
        before = self.w._timebase_s
        self.w._on_timebase_change(self._shown())
        self.assertEqual(self.w._timebase_s, before)

    def test_paging_steps_by_what_is_on_screen(self):
        """Page keys move by _timebase_s, so it must track the real span."""
        self.w._sync_timebase_combo(450.0)
        self.assertAlmostEqual(float(self.w._timebase_s), 450.0, places=3)


if __name__ == '__main__':
    unittest.main()
