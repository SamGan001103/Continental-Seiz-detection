"""A threshold you can type, not only drag.

It is a number a clinician arrives at and then wants to SET -- from a
protocol, from a previous session, from what a colleague used -- and a
160-pixel slider cannot be asked for 0.62.

The slider stays: dragging is how you explore what a threshold does to the
event list. The box is how you commit to one. They are two ways into a single
value, and the test that matters is that they do not feed each other -- each
updates the other, so without blocked signals the pair ping back and forth
and re-score the whole recording on every bounce.
"""
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtCore, QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    HAVE_QT = True
except Exception:                                        # pragma: no cover
    HAVE_QT = False


def _drain():
    """Run queued events and pending deleteLater destructions to completion.

    A MainWindow closed without this leaves graphics items queued for
    deletion; a later test then paints one whose ViewBox is already gone,
    which takes the whole run down rather than failing a test.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.processEvents()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheThresholdCanBeTyped(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(['-platform', 'offscreen']))

    def setUp(self):
        from gui.app import MainWindow
        self.w = MainWindow()
        self.addCleanup(_drain)
        self.addCleanup(self.w.close)

    def test_typing_a_value_sets_the_threshold(self):
        self.w.thr_box.setValue(0.62)
        self.assertAlmostEqual(self.w._threshold, 0.62, places=6)

    def test_typing_moves_the_slider_with_it(self):
        self.w.thr_box.setValue(0.62)
        self.assertEqual(self.w.thr_slider.value(), 62)

    def test_dragging_moves_the_number_with_it(self):
        self.w.thr_slider.setValue(77)
        self.assertAlmostEqual(self.w.thr_box.value(), 0.77, places=6)
        self.assertAlmostEqual(self.w._threshold, 0.77, places=6)

    def test_the_two_do_not_feed_each_other(self):
        """Each updates the other, so without blocked signals the pair ping
        back and forth and re-score the recording on every bounce."""
        seen = []
        real = self.w._rebuild_events_from_probs
        self.w._rebuild_events_from_probs = (
            lambda *a, **k: seen.append(1) or real(*a, **k))
        self.w.thr_box.setValue(0.33)
        self.assertEqual(len(seen), 1,
                         'setting the threshold once re-scored the recording '
                         '{} times'.format(len(seen)))

    def test_it_is_bounded_to_a_real_score(self):
        self.assertEqual(self.w.thr_box.minimum(), 0.0)
        self.assertEqual(self.w.thr_box.maximum(), 1.0)

    def test_it_carries_the_same_warning_as_the_slider(self):
        """The value is a raw uncalibrated score, not a probability. Whoever
        types one needs that as much as whoever drags one."""
        self.assertEqual(self.w.thr_box.toolTip(), self.w.thr_slider.toolTip())
        self.assertIn('uncalibrated', self.w.thr_box.toolTip())


if __name__ == '__main__':
    unittest.main()
