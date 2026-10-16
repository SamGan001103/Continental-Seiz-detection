"""The spike density band between the traces and the score map.

Interictal marks used to be drawn at the foot of the score axis. A score
curve rests near zero for most of a recording, so the two were competing for
the same rows of pixels and the amber lost under the red every time -- two
different quantities, in different units, sharing one axis.

They now have their own row. A band and not a lane, because the height on
this screen belongs to the traces; colour and not bar heights, because a day
across nineteen hundred pixels is forty-five seconds per pixel and a few
hundred detections drawn as ticks fill the strip evenly and say nothing.

The thing these tests are really protecting is the smoothing. Counted
straight into pixel columns, a few hundred marks land one per column and the
band renders as scattered hairlines -- the same picket fence, in colour.
Spread over a few columns, an isolated discharge becomes a faint blot and a
run of twenty becomes a solid one, so the eye reads the local rate rather
than the accident of which pixel a mark fell in.

And the distinction that must survive any redesign: a detector proposal is
unvalidated on this data, and must never be drawn as the same thing as a
mark a clinician confirmed.
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
    from PyQt5 import QtGui, QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.widgets.ied_band import AXIS_WIDTH, IedBand, density
    HAVE_QT = True
except Exception:                                        # pragma: no cover
    HAVE_QT = False

DAY = 86400.0
COLS = 1000


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheDensityIsALocalRate(unittest.TestCase):
    def test_no_marks_is_an_empty_band(self):
        self.assertEqual(float(density([], DAY, COLS).max()), 0.0)

    def test_a_recording_of_no_length_does_not_divide_by_zero(self):
        self.assertEqual(density([1.0, 2.0], 0.0, COLS).shape, (COLS,))
        self.assertEqual(float(density([1.0], 0.0, COLS).max()), 0.0)

    def test_marks_outside_the_recording_are_ignored(self):
        self.assertEqual(float(density([-5.0, DAY + 90.0], DAY, COLS).max()),
                         0.0)

    def test_the_peak_lands_where_the_marks_are(self):
        times = list(np.linspace(0.25 * DAY, 0.25 * DAY + 600.0, 120))
        d = density(times, DAY, COLS)
        self.assertAlmostEqual(int(np.argmax(d)) / float(COLS), 0.25,
                               delta=0.02)

    def test_a_lone_mark_is_spread_rather_than_left_as_one_column(self):
        """One mark per pixel column is the picket fence, in colour."""
        d = density([0.5 * DAY], DAY, COLS)
        self.assertGreater(int((d > 0).sum()), 3,
                           'a single discharge still occupies one column, so '
                           'the band is hairlines rather than a heat map')

    def test_a_cluster_reads_hotter_than_a_lone_mark(self):
        lone = density([0.2 * DAY], DAY, COLS).max()
        cluster = density(list(np.linspace(0.6 * DAY, 0.6 * DAY + 300.0, 30)),
                          DAY, COLS).max()
        self.assertGreater(cluster, 3 * lone)

    def test_a_cluster_at_the_very_start_is_not_dimmed_by_the_edge(self):
        """Zero-padding the convolution would halve the first and last bins,
        and a run of discharges at either end of a recording is exactly the
        thing a reviewer is looking for."""
        at_edge = density(list(np.linspace(0.0, 300.0, 30)), DAY, COLS).max()
        mid = density(list(np.linspace(0.5 * DAY, 0.5 * DAY + 300.0, 30)),
                      DAY, COLS).max()
        self.assertGreater(at_edge, 0.7 * mid)

    def test_it_returns_one_value_per_column_asked_for(self):
        for cols in (1, 7, 480, 1936):
            self.assertEqual(density([100.0], DAY, cols).shape, (cols,))


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheBandDrawsWhatItWasGiven(unittest.TestCase):
    def setUp(self):
        self.w = IedBand()
        self.addCleanup(self.w.deleteLater)
        self.w.setFixedWidth(1000)

    def _render(self):
        img = QtGui.QImage(self.w.size(), QtGui.QImage.Format_RGB32)
        img.fill(0xFFFFFFFF)
        self.w.render(img)
        return img

    def _amberness(self, img, x):
        """How far from white a column is, 0..1."""
        y = img.height() // 2
        c = QtGui.QColor(img.pixel(int(x), y))
        return 1.0 - (c.blue() / 255.0)

    def test_proposals_and_confirmed_marks_are_counted_apart(self):
        self.w.set_marks([{'time': 10.0, 'source': 'spikenet1'},
                          {'time': 20.0, 'source': 'reviewer'},
                          {'time': 30.0}], DAY)
        proposed, confirmed = self.w.counts()
        self.assertEqual(proposed, 1)
        self.assertEqual(confirmed, 2,
                         'a mark with no source is the reviewer\'s own; '
                         'defaulting it to the detector would credit a human '
                         'judgement to a model')

    def test_a_malformed_mark_is_skipped_not_fatal(self):
        self.w.set_marks([{'time': 'x'}, {}, {'time': 5.0}], DAY)
        self.assertEqual(sum(self.w.counts()), 1)

    def test_where_the_marks_are_is_amber_and_elsewhere_is_white(self):
        self.w.set_marks(
            [{'time': t, 'source': 'spikenet1'}
             for t in np.linspace(0.5 * DAY, 0.5 * DAY + 600.0, 60)], DAY)
        img = self._render()
        rect = self.w.plot_rect()
        hot = rect.left() + rect.width() // 2
        cold = rect.left() + int(rect.width() * 0.1)
        self.assertGreater(self._amberness(img, hot), 0.4)
        self.assertLess(self._amberness(img, cold), 0.05)

    def test_the_label_column_is_left_clear_for_the_axis_below(self):
        """The band must line up with the score strip column for column."""
        self.assertEqual(self.w.plot_rect().left(), AXIS_WIDTH)

    def test_clearing_empties_the_band(self):
        self.w.set_marks([{'time': 100.0, 'source': 'spikenet1'}], DAY)
        self.w.clear()
        self.assertEqual(self.w.counts(), (0, 0))
        img = self._render()
        rect = self.w.plot_rect()
        self.assertLess(self._amberness(img, rect.left() + 5), 0.05)

    def test_it_paints_without_a_recording_loaded(self):
        self.w.set_marks([], None)
        self._render()          # must not raise

    def test_it_stays_short(self):
        """Height here is taken from the traces, which are what is read."""
        self.assertLessEqual(self.w.height(), 40)

    def test_it_says_the_marks_are_unvalidated(self):
        self.assertIn('UNVALIDATED', self.w.toolTip())


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheWindowPutsItBetweenTheTracesAndTheMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(['-platform', 'offscreen']))

    def test_the_band_exists_and_sits_below_the_traces(self):
        from gui.app import MainWindow
        w = MainWindow()
        try:
            self.assertIsInstance(w.ied_band, IedBand)
            lay = w.ied_band.parentWidget().layout()
            order = [lay.itemAt(i).widget() for i in range(lay.count())]
            self.assertLess(order.index(w.signal_view),
                            order.index(w.ied_band))
            self.assertLess(order.index(w.ied_band),
                            order.index(w.overview_strip))
        finally:
            w.close()
            from PyQt5 import QtCore
            self.app.processEvents()
            self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
