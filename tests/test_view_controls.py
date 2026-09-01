"""The view must not be changed by accident, nor described wrongly.

Two defects, one cause: the toolbar and the display could disagree, and the
gesture that made them disagree was the ordinary one.

* pyqtgraph zooms on any wheel event. A trackpad two-finger scroll IS a wheel
  event, so scrolling through a recording zoomed out instead, and kept going
  until the whole recording was on screen because the x-limits stop exactly
  there. Past 300 s the renderer switches to a navigation view that is not for
  judging morphology, so a reviewer could arrive there without asking.

* The sweep control was write-only. It set the span and was never told when
  anything else changed it, so after that zoom it read `10` over a view of
  1818 s — wrong by two orders of magnitude, and it is the number a reviewer
  would use to judge what they are looking at.

Neither is caught by a test of the renderer, because the renderer was doing
what it was asked. These test the asking.
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
class ScrollMovesAndOnlyAModifierZooms(unittest.TestCase):
    def setUp(self):
        self.vb = _SignalViewBox()
        self.pw = pg.PlotWidget(viewBox=self.vb)
        self.vb.setXRange(100.0, 110.0, padding=0)

    def _span(self):
        (x0, x1), _ = self.vb.viewRange()
        return x1 - x0

    def _x0(self):
        return self.vb.viewRange()[0][0]

    def test_a_plain_scroll_does_not_change_the_span(self):
        """The regression: this used to zoom, and zoom out to the whole file."""
        before = self._span()
        self.vb.wheelEvent(_Wheel(-120))
        self.assertAlmostEqual(self._span(), before, places=3,
                               msg='an unmodified scroll changed the zoom')

    def test_a_plain_scroll_moves_through_the_recording(self):
        before = self._x0()
        self.vb.wheelEvent(_Wheel(-120))
        self.assertGreater(self._x0(), before,
                           'scrolling down did not advance through the '
                           'recording')
        forward = self._x0()
        self.vb.wheelEvent(_Wheel(120))
        self.assertLess(self._x0(), forward, 'scrolling up did not go back')

    def test_a_modified_scroll_still_zooms(self):
        before = self._span()
        self.vb.wheelEvent(_Wheel(-120, QtCore.Qt.ControlModifier))
        self.assertGreater(self._span(), before * 1.02,
                           'modifier+scroll no longer zooms out')

    def test_command_on_macos_zooms_too(self):
        """Qt reports Command as ControlModifier, but Meta is accepted too."""
        before = self._span()
        self.vb.wheelEvent(_Wheel(-120, QtCore.Qt.MetaModifier))
        self.assertGreater(self._span(), before * 1.02)


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
