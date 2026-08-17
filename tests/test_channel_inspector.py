"""The detail window's two claims about itself, held to account.

The channel inspector exists so a reviewer can decide whether something is
muscle artefact. Both features tested here are ones where the window could look
right and be wrong, silently:

  * the spectrum panel. It must describe the span the reviewer is looking at,
    and it must say which band figures the display filters have already
    decided. A band table computed under a 70 Hz low-pass and a 60 Hz notch
    reports roughly two thirds of the real gamma power, and nothing in the
    number itself says so — so if the annotation ever stops appearing, the panel
    starts quietly lying about the patient. That is the test worth having here.

  * blind mode. SignalView and ProbStrip hide their ground-truth bands on
    request; this window used to draw them unconditionally at construction with
    no way to take them away, so a session could show the answer key in a detail
    window while exporting blind_mode: true. The export would then be evidence
    of something that did not happen.

Headless-safe (QT_QPA_PLATFORM=offscreen) and skipped entirely without PyQt5,
following tests/test_review_guards.py.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.processing import BANDS, band_powers, psd

try:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtWidgets
    from PyQt5.QtTest import QTest
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.widgets.channel_inspector import (MAX_SPECTRUM_SECONDS,
                                               MIN_SPECTRUM_SAMPLES,
                                               ChannelInspector)
    HAVE_QT = True
except Exception:                                    # pragma: no cover
    HAVE_QT = False

FS = 250.0

# Comfortably longer than the 80 ms debounce, short enough that the whole file
# still runs in a couple of seconds.
SETTLE_MS = 250


def _two_tone(first_hz, second_hz, seconds=120.0, fs=FS, amp_uv=40.0):
    """Half a recording at one frequency, half at another.

    Gives the range-following test something unambiguous to detect: which band
    dominates is a property of WHICH HALF is on screen, so a spectrum that
    ignored the visible range would be caught rather than merely suspected.
    """
    n = int(seconds * fs)
    t = (np.arange(n) / fs).astype(np.float32)
    x = np.empty(n, dtype=np.float32)
    half = n // 2
    x[:half] = amp_uv * np.sin(2 * np.pi * first_hz * t[:half])
    x[half:] = amp_uv * np.sin(2 * np.pi * second_hz * t[half:])
    return x, t


def _dominant(powers):
    return max(powers, key=lambda name: float(powers[name]))


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class ChannelInspectorSpectrumTests(unittest.TestCase):
    def setUp(self):
        # 10 Hz (alpha) for the first minute, 40 Hz (gamma) for the second.
        self.data, self.t = _two_tone(10.0, 40.0)
        self.insp = ChannelInspector(
            channel_idx=3, channel_name='F3', fs=FS, data1d=self.data,
            t=self.t, reference_intervals=[(5.0, 15.0)],
            initial_x_range=(0.0, 30.0))

    def tearDown(self):
        self.insp.close()
        self.insp.deleteLater()

    def _settle(self):
        QTest.qWait(SETTLE_MS)

    # ---------------------------------------------------------------- spectrum
    def test_the_spectrum_describes_the_visible_span(self):
        """Zooming to the 40 Hz half must move the power there.

        This is the whole contract of the panel: it is a spectrum OF THE VISIBLE
        SPAN, not of the recording.
        """
        self.insp.set_filters(1.0, 70.0, 60.0)
        self._settle()
        self.assertEqual(_dominant(self.insp.current_band_powers()), 'alpha')

        self.insp.set_x_range(70.0, 100.0)
        self._settle()
        self.assertEqual(_dominant(self.insp.current_band_powers()), 'gamma')

        self.insp.set_x_range(5.0, 35.0)
        self._settle()
        self.assertEqual(_dominant(self.insp.current_band_powers()), 'alpha')

    def test_the_plotted_curve_moves_with_the_range_too(self):
        """Not just the numbers: the drawn peak has to move as well, or the
        reviewer is reading a table that disagrees with the picture above it."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self._settle()
        f, p = self.insp._spec_curve.getData()
        self.assertAlmostEqual(float(f[int(np.argmax(p))]), 10.0, delta=1.5)

        self.insp.set_x_range(70.0, 100.0)
        self._settle()
        f, p = self.insp._spec_curve.getData()
        self.assertAlmostEqual(float(f[int(np.argmax(p))]), 40.0, delta=1.5)

    def test_a_range_change_is_debounced_rather_than_computed_inline(self):
        """Scrubbing fires sigRangeChanged per mouse-move. If each one ran a
        Welch estimate the window would stutter, so the work must be deferred."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self._settle()
        before = self.insp.current_band_powers()

        self.insp.set_x_range(70.0, 100.0)
        self.assertTrue(self.insp._spec_timer.isActive())
        self.assertEqual(self.insp.current_band_powers(), before)

        self._settle()
        self.assertFalse(self.insp._spec_timer.isActive())
        self.assertNotEqual(self.insp.current_band_powers(), before)

    def test_a_y_only_change_does_not_arm_the_spectrum_timer(self):
        """Amplitude changes fire the same signal and cannot alter a spectrum;
        recomputing on them would be work done for nothing on every axis drag."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self._settle()
        self.insp.cb_amp.setCurrentText('500')
        self.assertFalse(self.insp._spec_timer.isActive())

    def test_the_panel_numbers_are_the_processing_helpers_numbers(self):
        """The widget must not do its own quantitative EEG arithmetic — one
        copy of that logic, in gui/processing.py, where it is unit-tested."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self.insp.set_x_range(0.0, 30.0)
        self._settle()
        seg = self.data[:int(30 * FS)]
        expect = band_powers(*psd(seg, FS))
        got = self.insp.current_band_powers()
        for name in BANDS:
            self.assertAlmostEqual(float(got[name]), float(expect[name]),
                                   delta=abs(float(expect[name])) * 1e-6 + 1e-9)

    def test_a_selection_too_short_to_analyse_says_so(self):
        """Better an explicit refusal than a shape with no reading behind it."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self.insp.set_x_range(10.0, 10.0 + (MIN_SPECTRUM_SAMPLES - 8) / FS)
        self._settle()
        self.assertIsNone(self.insp.current_band_powers())
        self.assertIn('too short', self.insp._spec_note.text())
        f, _p = self.insp._spec_curve.getData()
        self.assertTrue(f is None or len(f) == 0)

    def test_a_bounded_sample_of_a_long_span_admits_it_is_a_sample(self):
        """Zoomed out over hours, Welch across every sample would stall the
        window — so it samples. Sampling silently would be the bad version."""
        data, t = _two_tone(10.0, 10.0, seconds=MAX_SPECTRUM_SECONDS * 3)
        insp = ChannelInspector(0, 'C3', FS, data, t, initial_x_range=(0.0, 30.0))
        try:
            insp.set_filters(1.0, 70.0, 60.0)
            insp.set_x_range(0.0, MAX_SPECTRUM_SECONDS * 3)
            QTest.qWait(SETTLE_MS)
            note = insp._spec_note.text()
            self.assertIn('middle', note)
            self.assertIn('not all of it', note)
        finally:
            insp.close()

    # ------------------------------------------------------- filter annotation
    def test_gamma_is_flagged_under_a_seventy_hertz_low_pass(self):
        """The headline case. At the app's defaults the top of gamma is outside
        the low-pass and the mains notch is inside it, so the gamma figure is
        substantially a statement about the filter chain."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self._settle()
        self.assertIn('gamma', self.insp.compromised_bands())

        chip = self.insp._band_chips['gamma']
        self.assertIn('filtered', chip.text())
        tip = chip.toolTip()
        self.assertIn('70', tip)
        self.assertIn('low-pass', tip)
        self.assertIn('60', tip)         # the notch is named too
        self.assertIn('gamma', self.insp._spec_note.text())

    def test_a_band_the_filters_leave_alone_is_not_flagged(self):
        """A warning on every row is a warning on none. Alpha (8-13 Hz) sits
        clear of a 1 Hz high-pass, a 70 Hz low-pass and a 60 Hz notch."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self._settle()
        self.assertNotIn('alpha', self.insp.compromised_bands())
        self.assertNotIn('filtered', self.insp._band_chips['alpha'].text())

    def test_turning_the_filters_off_clears_the_flags(self):
        """The annotation has to track the toolbar, not the first value seen."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self._settle()
        self.assertTrue(self.insp.compromised_bands())
        self.insp.set_filters(None, None, None)
        self._settle()
        self.assertEqual(self.insp.compromised_bands(), [])
        self.assertIn('no display filters', self.insp._spec_note.text())

    def test_a_flagged_band_is_coloured_on_the_plot_not_only_in_the_table(self):
        """The panel's tooltip tells the reviewer an amber column is one the
        filters shaped. Their eye is on the spectrum when they judge the shape,
        so the colour has to be there and not only on the chip underneath."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self._settle()
        flagged = self.insp._band_regions['gamma'].brush.color()
        clean = self.insp._band_regions['alpha'].brush.color()
        self.assertNotEqual(flagged.getRgb(), clean.getRgb())

        self.insp.set_filters(None, None, None)
        self._settle()
        self.assertEqual(
            self.insp._band_regions['gamma'].brush.color().getRgb(),
            self.insp._band_tints['gamma'])

    def test_filters_are_drawn_on_the_spectrum(self):
        """The markers are why the reviewer can see that the dive at 70 Hz is
        the low-pass and not the patient."""
        self.insp.set_filters(1.0, 70.0, 60.0)
        self._settle()
        self.assertTrue(self.insp._filter_items)
        positions = [round(float(it.value()), 3)
                     for it in self.insp._filter_items
                     if hasattr(it, 'value')]
        for hz in (1.0, 70.0, 60.0):
            self.assertIn(hz, positions)

    def test_an_untold_window_does_not_claim_the_bands_are_clean(self):
        """Constructed but never told the filter settings, the panel must not
        present the band table as physiology: the data it holds HAS been
        filtered, and 'nobody told me' is not the same as 'no filters'."""
        self.assertFalse(self.insp.filters_known())
        self._settle()
        self.assertEqual(self.insp.compromised_bands(), [])   # claims nothing
        self.assertIn('not supplied', self.insp._spec_note.text())
        for name in BANDS:
            self.assertIn('unverified',
                          self.insp._band_chips[name].toolTip())

    def test_a_filter_that_could_not_run_is_not_reported_as_running(self):
        """apply_filters drops a low-pass at or above Nyquist and cannot build a
        notch there at all. Echoing the toolbar verbatim would name filters that
        never touched the signal."""
        data, t = _two_tone(9.0, 9.0, seconds=60.0, fs=100.0)
        insp = ChannelInspector(0, 'O1', 100.0, data, t,
                                initial_x_range=(0.0, 20.0))
        try:
            insp.set_filters(1.0, 70.0, 60.0)
            QTest.qWait(SETTLE_MS)
            note = insp._spec_note.text()
            self.assertIn('HP 1 Hz', note)
            self.assertIn('not applied', note)
            self.assertIn('Nyquist', note)
            # No marker may be drawn for a filter that did not run.
            positions = [round(float(it.value()), 3)
                         for it in insp._filter_items if hasattr(it, 'value')]
            self.assertIn(1.0, positions)
            self.assertNotIn(70.0, positions)
            self.assertNotIn(60.0, positions)
        finally:
            insp.close()

    # ------------------------------------------------------------ plot chrome
    def test_both_plots_refuse_the_pyqtgraph_context_menu_and_autorange(self):
        """That menu offers an undisclaimered image export of a clinical trace,
        and a one-click auto-range that destroys the µV calibration the
        amplitude selector exists to guarantee."""
        for pi in (self.insp.plot_item(), self.insp.spectrum_plot_item()):
            self.assertIsNone(pi.vb.menu)
            self.assertTrue(pi.buttonsHidden)


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class ChannelInspectorBlindModeTests(unittest.TestCase):
    REFS = [(5.0, 15.0), (40.0, 55.0)]

    def setUp(self):
        self.data, self.t = _two_tone(10.0, 10.0, seconds=60.0)
        self.insp = ChannelInspector(
            channel_idx=0, channel_name='Fp1', fs=FS, data1d=self.data,
            t=self.t, reference_intervals=self.REFS,
            initial_x_range=(0.0, 30.0))

    def tearDown(self):
        self.insp.close()
        self.insp.deleteLater()

    def _visible(self):
        return [r.isVisible() for r in self.insp._ref_items]

    def test_the_window_offers_the_same_switch_as_the_other_views(self):
        """MainWindow calls this on SignalView and ProbStrip by name; an
        inspector without it is the hole through which the answer key leaks."""
        for name in ('set_references_visible', 'references_visible'):
            self.assertTrue(callable(getattr(self.insp, name, None)),
                            '{} missing'.format(name))

    def test_blind_mode_hides_the_reference_bands_after_construction(self):
        self.assertEqual(self._visible(), [True, True])
        self.insp.set_references_visible(False)
        self.assertEqual(self._visible(), [False, False])
        self.assertFalse(self.insp.references_visible())

    def test_leaving_blind_mode_restores_them(self):
        self.insp.set_references_visible(False)
        self.insp.set_references_visible(True)
        self.assertEqual(self._visible(), [True, True])
        self.assertTrue(self.insp.references_visible())

    def test_blind_mode_survives_a_montage_or_filter_refresh(self):
        """set_data rebuilds the bands. If it rebuilt them visible, changing the
        montage mid-session would silently un-blind the review — and the export
        would still say blind_mode: true."""
        self.insp.set_references_visible(False)
        self.insp.set_data('Fp1-F7', self.data, self.t, self.REFS)
        self.assertEqual(self._visible(), [False, False])

    def test_a_refresh_replaces_the_bands_rather_than_stacking_them(self):
        """They used to be drawn once at construction and never again, so a new
        reference list left the previous recording's seizures on screen."""
        self.insp.set_data('Fp1-F7', self.data, self.t, [(1.0, 2.0)])
        self.assertEqual(len(self.insp._ref_items), 1)
        self.assertEqual(self.insp._ref_items[0].getRegion(), (1.0, 2.0))

    def test_a_window_with_no_reference_still_answers_the_switch(self):
        """A recording without a .csv_bi has nothing to hide; the call must
        still be safe, because MainWindow makes it unconditionally."""
        insp = ChannelInspector(0, 'Cz', FS, self.data, self.t,
                                reference_intervals=None,
                                initial_x_range=(0.0, 10.0))
        try:
            insp.set_references_visible(False)
            insp.set_references_visible(True)
            self.assertEqual(insp._ref_items, [])
        finally:
            insp.close()


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class DegenerateSpansDoNotBreakTheSpectrum(unittest.TestCase):
    """A span the amplifier did not record must not take the panel down.

    `pmax <= 0.0` does not catch NaN — `nan <= 0.0` is False — so a non-finite
    sample reached `setYRange(log10(nan), ...)`, where pyqtgraph raises. The
    exception escaped after the curve had been redrawn but before the band
    chips and the note were updated, leaving an empty plot above a populated
    band table describing a different time range. In the frozen build the
    excepthook turns that into a modal error dialog, once per debounce tick
    while the reviewer scrubs across the region.
    """

    def _inspector(self, data):
        import numpy as np
        t = np.arange(data.size) / 250.0
        insp = ChannelInspector(0, 'Fp1', 250.0, data, t,
                                initial_x_range=(0.0, 30.0))
        insp.set_filters(1.0, 70.0, 60.0)
        return insp

    def test_a_span_containing_nan_reports_it_instead_of_raising(self):
        import numpy as np
        x = (30.0 * np.sin(2 * np.pi * 10.0 *
                           np.arange(30 * 250) / 250.0)).astype(np.float32)
        x[1000:1010] = np.nan
        insp = self._inspector(x)
        try:
            insp.refresh_spectrum()          # must not raise
            note = insp._spec_note.text()
            self.assertIn('finite', note.lower(),
                          'a non-finite span must say so; got: %r' % note)
        finally:
            insp.close()

    def test_an_infinite_sample_is_handled_too(self):
        import numpy as np
        x = np.zeros(30 * 250, dtype=np.float32)
        x[500] = np.inf
        insp = self._inspector(x)
        try:
            insp.refresh_spectrum()
            self.assertTrue(insp._spec_note.text())
        finally:
            insp.close()

    def test_a_flat_channel_still_gets_its_own_message(self):
        """A dead electrode and an unrecorded span are different findings."""
        import numpy as np
        insp = self._inspector(np.zeros(30 * 250, dtype=np.float32))
        try:
            insp.refresh_spectrum()
            note = insp._spec_note.text().lower()
            self.assertIn('flat', note)
            self.assertNotIn('finite', note)
        finally:
            insp.close()

    def test_setting_filters_at_a_low_sample_rate_does_not_raise(self):
        """refresh_spectrum used the raw toolbar values, not the effective ones.

        _butter_sos is unguarded for a cutoff at or above Nyquist, so a high-pass
        of 70 Hz at fs=100 raised out of the setter. Unreachable from the
        toolbar, but nothing structural prevented it.
        """
        import numpy as np
        t = np.arange(3000) / 100.0
        insp = ChannelInspector(0, 'Fp1', 100.0,
                                np.zeros(3000, dtype=np.float32), t,
                                initial_x_range=(0.0, 30.0))
        try:
            insp.set_filters(70.0, None, None)     # must not raise
            insp.refresh_spectrum()
        finally:
            insp.close()
