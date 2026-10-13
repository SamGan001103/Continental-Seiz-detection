"""A draw that cannot be fast must at least look alive.

The review application reads a prepared study bundle and computes nothing --
for the detectors, and for the zoomed-out picture in the presets the bundle
carries. It cannot carry all of them: four montages and six choices each of
high-pass and low-pass are 432 combinations, about 15 GB per recording-day, so
the bundle ships the handful reviewers actually use and anything else is built
on the spot.

That build is ~40 seconds on a recording-day and it happens on the GUI thread,
inside a range-changed handler. With no feedback the window simply goes white
and Windows writes "Not Responding" in the title bar, which is what was
reported: "the app is supposed to be cached, why do I still get lag and
unresponding". It was not the cache failing; it was an uncached combination
with nothing on screen to say so.

So the view asks whether the draw is expensive before making it, and if it is,
puts up a wait cursor and names the reason in the status bar first. These tests
pin that, and pin the thing the pumped repaint makes possible: a range change
arriving mid-build must not start a second one.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    from PyQt5 import QtCore, QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.widgets.signal_view import SignalView
    HAVE_QT = True
except Exception:                                        # pragma: no cover
    HAVE_QT = False

FS = 250
N_ROWS = 18


def _drain():
    """Run queued events and pending deleteLater destructions to completion."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.processEvents()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


class _Provider(object):
    """A render that is expensive, and says so when asked."""

    def __init__(self, heavy=True, seconds=600):
        self.heavy = heavy
        self.n_rows = N_ROWS
        self.n_samples = int(seconds * FS)
        self.duration_s = float(seconds)
        self.fs = FS
        self.asked = 0
        self.renders = 0
        self.cursors_during = []
        self.during = None          # called inside render(), to re-enter
        self.raise_in_render = False

    def needs_envelope_build(self, s0, s1):
        self.asked += 1
        return self.heavy

    def render(self, s0, s1, target_px):
        self.renders += 1
        self.cursors_during.append(
            QtWidgets.QApplication.overrideCursor() is not None)
        if self.during is not None:
            hook, self.during = self.during, None
            hook()
        if self.raise_in_render:
            raise RuntimeError('a damaged recording, mid-draw')
        m = max(2, min(64, target_px))
        return (np.zeros((N_ROWS, m), dtype=np.float32),
                np.linspace(s0 / float(FS), s1 / float(FS), m,
                            dtype=np.float32))

    # the rest of the provider surface the view touches
    def channel_slice(self, *a, **k):
        return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class AnExpensiveDrawSaysSo(unittest.TestCase):
    def setUp(self):
        self.view = SignalView()
        self.started = []
        self.finished = []
        self.view.longBuildStarted.connect(lambda: self.started.append(1))
        self.view.longBuildFinished.connect(lambda: self.finished.append(1))
        self.addCleanup(self._restore)

    def _restore(self):
        # A leaked override cursor would follow every later test in the run.
        while QtWidgets.QApplication.overrideCursor() is not None:
            QtWidgets.QApplication.restoreOverrideCursor()
        # The build pumps the event loop, so a debounced redraw can be sitting
        # in the queue. Let it run while the widget is still alive, then
        # destroy the widget to completion -- out of that order it arrives at
        # a deleted C++ object and prints a traceback from every later test.
        self.view._redec_timer.stop()
        _drain()
        self.view.deleteLater()
        _drain()

    def _load(self, provider):
        self.view.set_provider(provider, FS,
                               ['r{}'.format(i) for i in range(N_ROWS)])
        provider.renders = 0
        provider.cursors_during = []
        del self.started[:], self.finished[:]
        return provider

    def test_the_cursor_is_busy_while_the_build_runs(self):
        p = self._load(_Provider(heavy=True))
        self.view._redecimate_visible()
        self.assertTrue(p.renders >= 1)
        self.assertTrue(all(p.cursors_during),
                        'render ran with no busy cursor, so a 40 s build '
                        'shows as a dead window')

    def test_the_cursor_is_given_back_afterwards(self):
        self._load(_Provider(heavy=True))
        self.view._redecimate_visible()
        self.assertIsNone(QtWidgets.QApplication.overrideCursor())

    def test_it_announces_the_build_and_its_end(self):
        self._load(_Provider(heavy=True))
        self.view._redecimate_visible()
        self.assertEqual((len(self.started), len(self.finished)), (1, 1))

    def test_a_cheap_draw_is_left_alone(self):
        """Every redraw asks. A reading page must pay nothing for that."""
        p = self._load(_Provider(heavy=False))
        self.view._redecimate_visible()
        self.assertTrue(p.renders >= 1)
        self.assertFalse(any(p.cursors_during))
        self.assertEqual((len(self.started), len(self.finished)), (0, 0))
        self.assertIsNone(QtWidgets.QApplication.overrideCursor())

    def test_a_redraw_arriving_mid_build_does_not_start_a_second(self):
        """The pumped repaint is what lets the window live through the build,
        and it is also what can deliver a range change straight back in."""
        p = self._load(_Provider(heavy=True))
        p.during = self.view._redecimate_visible
        self.view._redecimate_visible()
        self.assertEqual(p.renders, 1,
                         'a re-entrant redraw started another whole-recording '
                         'build inside the first one')
        self.assertEqual(len(self.started), 1)

    def test_a_redraw_refused_mid_build_is_asked_again_not_dropped(self):
        """The range that arrived during the build is the one the reviewer is
        now looking at. Dropped, the axis reads the new span while the traces
        still show the old one."""
        p = self._load(_Provider(heavy=True))
        p.during = self.view._redecimate_visible
        self.view._redec_timer.stop()
        self.view._redecimate_visible()
        self.assertTrue(self.view._redec_timer.isActive(),
                        'the refused redraw was dropped, so the traces stay '
                        'stale against the axis until something else moves')

    def test_a_failed_draw_still_gives_the_cursor_back(self):
        """A build that raises must not leave the application wearing an
        hourglass with no way to clear it."""
        p = self._load(_Provider(heavy=True))
        p.raise_in_render = True
        with self.assertRaises(RuntimeError):
            self.view._redecimate_visible()
        self.assertIsNone(QtWidgets.QApplication.overrideCursor())
        self.assertFalse(self.view._building)
        self.assertEqual(len(self.finished), 1)


if __name__ == '__main__':
    unittest.main()
