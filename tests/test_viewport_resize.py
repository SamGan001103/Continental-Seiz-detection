"""Dragging the ends of the "you are here" marker changes the span on screen.

The marker on the whole-recording map is the page. Dragging its body moves
the page; dragging one end should change how much of the recording the page
covers. Both gestures arrive through one pyqtgraph signal, and the only thing
separating them is the width: a body drag moves both edges by the same
amount, a handle drag moves one.

Before this, every drag was read as "go here". An edge drag therefore
re-centred the page at the span it already had, and the marker snapped
straight back to its old width -- resizing it looked broken because nothing
was listening for it.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtCore, QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.widgets.overview_strip import MIN_VIEWPORT_S, OverviewStrip
    HAVE_QT = True
except Exception:                                        # pragma: no cover
    HAVE_QT = False

DAY = 86400.0


def _drain():
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.processEvents()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class DraggingAnEndResizesThePage(unittest.TestCase):
    def setUp(self):
        self.w = OverviewStrip()
        self.addCleanup(_drain)
        self.addCleanup(self.w.deleteLater)
        starts = np.arange(0, int(DAY) - 12 + 1, 6, dtype=np.int32)
        self.w.set_probs(starts, np.full(starts.size, 0.02, dtype=np.float32),
                         segment_s=12, duration_s=DAY)
        self.seeks, self.ranges = [], []
        self.w.seek_requested.connect(self.seeks.append)
        self.w.range_requested.connect(
            lambda a, b: self.ranges.append((a, b)))

    def _drag(self, frm, to):
        """The window sets the marker, the user drags it, pyqtgraph reports
        the finish. pyqtgraph has no drag-started signal, so the span the
        window last set is what the result is measured against."""
        self.w.set_viewport(*frm)
        # setRegion emits sigRegionChangeFinished itself, which is the same
        # path a real drag takes -- calling the handler as well would report
        # one gesture twice.
        self.w._viewport.setRegion(to)

    def test_moving_the_body_asks_to_go_somewhere(self):
        self._drag((1000.0, 1300.0), (5000.0, 5300.0))
        self.assertEqual(len(self.ranges), 0,
                         'a body drag was read as a resize')
        self.assertEqual(len(self.seeks), 1)
        self.assertAlmostEqual(self.seeks[0], 5150.0, places=3)

    def test_dragging_one_end_asks_for_a_new_span(self):
        self._drag((1000.0, 1300.0), (1000.0, 4000.0))
        self.assertEqual(len(self.seeks), 0,
                         'a resize was read as "go here", which is what made '
                         'the marker snap back to its old width')
        self.assertEqual(len(self.ranges), 1)
        lo, hi = self.ranges[0]
        self.assertAlmostEqual(lo, 1000.0, places=3)
        self.assertAlmostEqual(hi, 4000.0, places=3)

    def test_dragging_the_left_end_works_too(self):
        self._drag((5000.0, 5300.0), (4000.0, 5300.0))
        self.assertEqual(len(self.ranges), 1)
        lo, hi = self.ranges[0]
        self.assertAlmostEqual(hi - lo, 1300.0, places=3)

    def test_a_shaky_hand_on_the_body_is_still_a_move(self):
        """A drag that changes the width by a hair is a move, not a resize."""
        self._drag((1000.0, 1300.0), (2000.0, 2300.2))
        self.assertEqual(len(self.ranges), 0)
        self.assertEqual(len(self.seeks), 1)

    def test_the_page_cannot_be_collapsed_to_nothing(self):
        self._drag((1000.0, 1300.0), (1000.0, 1000.0))
        self.assertEqual(len(self.ranges), 1)
        lo, hi = self.ranges[0]
        self.assertGreaterEqual(hi - lo, MIN_VIEWPORT_S - 1e-6)

    def test_a_resize_stays_inside_the_recording(self):
        self._drag((100.0, 400.0), (-5000.0, 400.0))
        lo, hi = self.ranges[0]
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, DAY)

    def test_it_cannot_ask_for_more_than_the_recording(self):
        self._drag((0.0, 300.0), (-50000.0, DAY + 50000.0))
        lo, hi = self.ranges[0]
        self.assertLessEqual(hi - lo, DAY + 1e-6)
        self.assertGreaterEqual(lo, 0.0)

    def test_the_window_syncing_the_marker_is_not_a_drag(self):
        """set_viewport is called on every redraw; it must not echo back as a
        seek and fight the view that just moved."""
        self.w.set_viewport(2000.0, 2300.0)
        self.assertEqual(self.seeks, [])
        self.assertEqual(self.ranges, [])


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheWindowActsOnARequestedSpan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(['-platform', 'offscreen']))

    def test_a_requested_range_lands_on_the_traces(self):
        from gui.app import MainWindow
        w = MainWindow()
        try:
            self.assertTrue(hasattr(w, '_on_overview_range'))
            w._on_overview_range(100.0, 400.0)   # no recording: must not raise
        finally:
            w.close()
            _drain()


if __name__ == '__main__':
    unittest.main()
