"""Single-channel detail window.

Opened from SignalView via double-click. Non-modal — the user can open
one per channel and keep them on screen alongside the main view.

Two panels, answering two different questions. The trace answers "what does it
look like". The power spectrum beneath it answers "what is it made of", which
is the question that actually separates muscle artefact from an ictal rhythm:
EMG is broadband and climbs toward high frequency, a seizure rhythm has a peak
and evolves. That is a FREQUENCY judgement, and until this panel existed the
window asked the reviewer to make it from a time-domain trace and two summary
numbers.

The spectrum is drawn with the active display filters marked on it, and any
clinical band those filters have already decided is labelled as such. At this
app's defaults (HP 1.0, LP 70, notch 60) the top of gamma is outside the
low-pass and the mains notch sits inside it, so an unannotated band table would
be reporting the filter and not the brain. gui.processing.filter_caveats
measures that from the very same sos sections apply_filters builds, so the
percentages here are a measurement rather than a hand-set warning.

DISPLAY PATH ONLY. Everything in this window is computed from the montaged,
filtered display copy handed to it. Nothing here touches the arrays feeding
inference, and the spectrum deliberately describes what is ON SCREEN — not
what the model saw, which is a different signal down a different path.
"""
from collections import OrderedDict

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

from gui.processing import BANDS, band_powers, filter_caveats, psd
from gui.widgets.signal_view import (_SignalViewBox, TimeCursor,
                                      _make_time_label, _update_time_label)


AMPLITUDES_UV = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500,
                 1000, 2000, 5000, 10000]

# Recomputing Welch on every sigRangeChanged makes the window stutter while the
# user drags: a range change fires per mouse-move, and a wheel zoom fires a
# burst of them. 80 ms is below the ~100 ms at which a redraw stops feeling
# attached to the gesture, and long enough that a scrub collapses to one
# estimate at the end of it. Same debounce idiom as SignalView._redec_timer.
SPECTRUM_DEBOUNCE_MS = 80

# Cap on how much signal one estimate covers. Zoomed out to a whole ambulatory
# recording the visible span is tens of millions of samples; Welch over all of
# them on every scrub would stall the window for seconds. A bounded, centred
# sample that SAYS it is a sample beats both a frozen GUI and a silently
# different estimate.
MAX_SPECTRUM_SECONDS = 120.0

# Below this the estimate is not worth drawing: at 250 Hz, 64 samples is a
# quarter-second window whose lowest resolvable frequency is above the whole
# delta band, so the plot would be a shape with no physiological reading.
MIN_SPECTRUM_SAMPLES = 64

# Decades of the log axis. EEG power spans several orders of magnitude between
# delta and gamma; five decades below the peak shows the shape of the tail
# without letting a numerically-zero bin drag the axis to -inf.
SPECTRUM_DECADES = 5.0

# Bins wider than this cannot place a rhythm inside its band — 3 Hz bins put
# alpha and beta in the same cell — so the note says so rather than letting the
# reviewer read band edges that are not there.
COARSE_BIN_HZ = 2.0

# Band shading. Alternating tints so neighbouring bands stay distinguishable,
# and deliberately NOT green: green is the ground-truth colour everywhere else
# in this app (see SignalView.STATUS_COLOURS) and must not appear on a panel
# that shows no ground truth. A flagged band is amber, matching the "attention,
# not verdict" amber used for proposed events.
_BAND_TINTS = ((70, 100, 160, 24), (70, 100, 160, 44))
_BAND_FLAGGED_TINT = (210, 130, 20, 55)

# Filters are drawn achromatic on purpose. Grey already means "no information
# here" in this app (ProbStrip's unscored bands), which is exactly what a
# stop-band is: whatever the patient's brain did up there, the display filter
# removed it before this window ever saw it.
_FILTER_PEN = (90, 90, 90)
_STOPBAND_BRUSH = (120, 120, 120, 45)

_CHIP_BASE = 'border-radius:3px; padding:1px 6px; margin-right:2px;'
_CHIP_OK = 'color:#333; background:#f2f4f8; border:1px solid #dde1e8; ' + _CHIP_BASE
_CHIP_FLAGGED = ('color:#7a4400; background:#fdf0d8; border:1px solid #e2b464; '
                 'font-weight:bold; ') + _CHIP_BASE
_CHIP_UNKNOWN = ('color:#555; background:#eee; border:1px dashed #bbb; '
                 ) + _CHIP_BASE

_FILTERS_UNKNOWN_NOTE = (
    'Display filter settings were not supplied to this window, so it cannot '
    'say which of these numbers the filters already decided. Treat every band '
    'as unverified.')

_SPECTRUM_TOOLTIP = (
    'Welch power spectral density of the visible span of THIS channel, after '
    'the montage and the display filters — not the signal the model saw.\n'
    'Shaded columns are the clinical bands. An amber column is one the display '
    'filters have already shaped; its power figure is partly a statement about '
    'the filter, so do not compare it across recordings filtered differently.\n'
    'Grey areas and dashed lines are the filters themselves: the grey is what '
    'the high-pass and low-pass removed, the dashed line at the notch is a hole '
    'punched at mains frequency.\n'
    'Muscle artefact is broadband and rises toward the high end; a rhythm has a '
    'peak. Judge that shape inside the unshaded region only.')


def _fit_amplitude(data_slice, headroom=1.2):
    if data_slice.size == 0:
        return AMPLITUDES_UV[-1]
    p98 = float(np.percentile(np.abs(data_slice), 98))
    target = p98 * headroom
    for a in AMPLITUDES_UV:
        if a >= target:
            return a
    return AMPLITUDES_UV[-1]


def _band_label_html(name, flagged):
    """In-plot band name. Amber and marked when the filters shaped the band."""
    if flagged:
        return ('<span style="color:#8a4b00; font-size:8pt;">'
                '{} !</span>'.format(name))
    return '<span style="color:#666; font-size:8pt;">{}</span>'.format(name)


def _filter_summary(hp, lp, notch):
    parts = []
    if hp:
        parts.append('HP {:g} Hz'.format(hp))
    if lp:
        parts.append('LP {:g} Hz'.format(lp))
    if notch:
        parts.append('notch {:g} Hz'.format(notch))
    if not parts:
        return 'no display filters'
    return 'display filters: ' + ', '.join(parts)


def _hz_ok(hz, nyq):
    """Whether apply_filters would have applied a filter at this frequency."""
    return bool(hz) and 0.0 < float(hz) < nyq


class _DraggableYAxis(pg.AxisItem):
    """Numeric Y-axis whose left-drag rescales the Y range symmetrically.

    ~140 px of drag corresponds to a 2× amplitude change — same feel as
    the main view's channel-axis drag. The callback is fired live during
    the drag and one final time on release so the caller can snap to a
    preset."""

    def __init__(self, orientation='left'):
        super().__init__(orientation=orientation)
        self._start = None           # (start_scene_y, start_half_range)
        self._on_drag = None
        self.setCursor(QtCore.Qt.SizeVerCursor)

    def set_drag_callback(self, cb):
        """cb(new_amplitude_uv: float, finished: bool)"""
        self._on_drag = cb

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            vb = self.linkedView()
            if vb is None:
                return
            y0, y1 = vb.viewRange()[1]
            self._start = (float(ev.buttonDownPos().y()),
                           max(1.0, (y1 - y0) / 2.0))
            return
        if self._start is None:
            return
        start_y, half0 = self._start
        dy = float(ev.pos().y()) - start_y
        # drag DOWN (positive dy) = bigger amplitude range (less sensitive)
        factor = 2.0 ** (dy / 140.0)
        new_amp = max(5.0, min(20000.0, half0 * factor))
        if self._on_drag is not None:
            self._on_drag(new_amp, ev.isFinish())
        if ev.isFinish():
            self._start = None


class ChannelInspector(QtWidgets.QDialog):
    closed = QtCore.pyqtSignal(int)

    def __init__(self, channel_idx, channel_name, fs, data1d, t,
                 reference_intervals=None, initial_x_range=None, parent=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowTitle('Channel — {}'.format(channel_name))
        self.setModal(False)
        self.resize(1000, 620)

        self._channel_idx = channel_idx
        self._channel_name = channel_name
        self._fs = fs
        self._data = data1d.astype(np.float32, copy=False)
        self._t = t

        # Ground truth is drawn by this window too, so it needs the same blind
        # switch as SignalView and ProbStrip. Without it a "blind" session could
        # export blind_mode: true while a detail window sat on screen showing
        # the answer key.
        self._reference_intervals = [(float(s), float(e))
                                     for s, e in (reference_intervals or [])]
        self._refs_visible = True
        self._ref_items = []

        # Filter state starts UNKNOWN rather than "off". The data handed to this
        # window has already been through apply_filters, and defaulting to "no
        # filters" would let the band table quietly claim a clean delta figure
        # that the high-pass had already taken a quarter out of. See set_filters.
        self._filters_known = False
        self._hp = None
        self._lp = None
        self._notch = None

        self._compromised = []
        self._band_powers = None
        self._last_spec_range = None

        # Created before the plots because adding a curve auto-ranges the view,
        # which fires sigRangeChanged — and the handler arms this timer. The
        # timer cannot actually fire until the event loop runs, by which time
        # the panel it refreshes exists.
        self._spec_timer = QtCore.QTimer(self)
        self._spec_timer.setSingleShot(True)
        self._spec_timer.setInterval(SPECTRUM_DEBOUNCE_MS)
        self._spec_timer.timeout.connect(self.refresh_spectrum)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)

        # ---- controls --------------------------------------------------
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(QtWidgets.QLabel('<b>{}</b>'.format(channel_name)))
        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel('Amplitude ±µV'))
        self.cb_amp = QtWidgets.QComboBox()
        self.cb_amp.addItems([str(a) for a in AMPLITUDES_UV])
        self.cb_amp.currentTextChanged.connect(self._on_amp_change)
        ctrl.addWidget(self.cb_amp)

        self.btn_fit = QtWidgets.QToolButton()
        self.btn_fit.setText('Fit')
        self.btn_fit.setToolTip('Auto-pick amplitude for the visible slice')
        self.btn_fit.clicked.connect(self._fit_to_visible)
        ctrl.addWidget(self.btn_fit)

        ctrl.addSpacing(16)
        self.chk_sync = QtWidgets.QCheckBox('Sync time with main view')
        self.chk_sync.setChecked(True)
        ctrl.addWidget(self.chk_sync)

        self.chk_autofit = QtWidgets.QCheckBox('Auto-fit amplitude')
        self.chk_autofit.setChecked(False)   # off by default — too jumpy
        self.chk_autofit.setToolTip(
            'Re-fit amplitude automatically when the visible range changes')
        ctrl.addWidget(self.chk_autofit)

        self.chk_spec = QtWidgets.QCheckBox('Power spectrum')
        self.chk_spec.setChecked(True)
        self.chk_spec.setToolTip(
            'Show the spectrum of the visible span. Turning it off also stops '
            'it being recomputed, which is worth doing on a slow machine.')
        self.chk_spec.toggled.connect(self._on_spec_toggled)
        ctrl.addWidget(self.chk_spec)

        ctrl.addStretch(1)
        self._stats_lbl = QtWidgets.QLabel('')
        self._stats_lbl.setStyleSheet('color:#666;')
        ctrl.addWidget(self._stats_lbl)
        v.addLayout(ctrl)

        # ---- trace plot ------------------------------------------------
        self._y_axis = _DraggableYAxis(orientation='left')
        self._y_axis.set_drag_callback(self._on_axis_drag)
        vb = _SignalViewBox()       # gentle wheel, same as main view
        self._pw = pg.PlotWidget(viewBox=vb,
                                 axisItems={'left': self._y_axis},
                                 background='w')
        pi = self._pw.getPlotItem()
        pi.setLabel('bottom', 'time (s)')
        pi.setLabel('left', 'µV')
        pi.showGrid(x=True, y=True, alpha=0.25)
        pi.setMouseEnabled(x=True, y=False)
        pi.sigRangeChanged.connect(self._on_range_changed)
        self._lock_plot_chrome(pi)
        v.addWidget(self._pw, 3)

        self._time_lbl = _make_time_label()
        v.addWidget(self._time_lbl)

        self._time_cursor = TimeCursor(
            self._pw,
            on_hover=lambda t: _update_time_label(self._time_lbl, t))

        self._rebuild_references()

        self._curve = pi.plot(t, self._data,
                              pen=pg.mkPen((20, 60, 160), width=0.9))
        if len(t):
            pi.vb.setLimits(xMin=0.0, xMax=float(t[-1]))

        # ---- spectrum panel --------------------------------------------
        self._build_spectrum_panel(v)

        # One-shot initial amplitude pick from the initial visible range.
        if initial_x_range is not None:
            pi.setXRange(*initial_x_range, padding=0)
            init_slice = self._slice_between(*initial_x_range)
        else:
            init_slice = self._data
        amp = _fit_amplitude(init_slice)
        self.cb_amp.blockSignals(True)
        self.cb_amp.setCurrentText(str(amp))
        self.cb_amp.blockSignals(False)
        self._set_y_range(amp)
        self._update_stats(init_slice)
        self.refresh_spectrum()

    # ------------------------------------------------------------------
    # Plot chrome
    # ------------------------------------------------------------------
    def _lock_plot_chrome(self, plot_item):
        """Remove pyqtgraph's context menu and auto-range button.

        Two entries on that menu are actively harmful here. "Export…" writes an
        image of a clinical trace with none of the disclaimers the app's own
        export carries — provenance, filter settings, blind-mode state — so the
        picture leaves the building saying nothing about how it was made. And
        the ViewBox's auto-range (menu entry and the little [A] button) rescales
        Y to fit the data, silently destroying the µV calibration the amplitude
        selector exists to guarantee; a reviewer measuring a deflection against
        the axis would then be measuring against nothing.
        """
        plot_item.setMenuEnabled(False)
        plot_item.hideButtons()

    # ------------------------------------------------------------------
    # Spectrum panel
    # ------------------------------------------------------------------
    def _build_spectrum_panel(self, v):
        spec_vb = _SignalViewBox()
        self._spec_pw = pg.PlotWidget(viewBox=spec_vb, background='w')
        spi = self._spec_pw.getPlotItem()
        spi.setLabel('bottom', 'frequency (Hz)')
        spi.setLabel('left', 'µV²/Hz')
        spi.getAxis('left').setWidth(64)
        spi.showGrid(x=True, y=True, alpha=0.2)
        spi.setMouseEnabled(x=True, y=False)
        # Log power before the menu is taken away: PlotItem.setLogMode drives
        # the same state the (now hidden) control menu would have set.
        spi.setLogMode(x=False, y=True)
        self._lock_plot_chrome(spi)
        self._spec_pw.setToolTip(_SPECTRUM_TOOLTIP)
        self._spec_pw.setMinimumHeight(150)
        v.addWidget(self._spec_pw, 2)

        # Band shading and in-plot band names, built once and repositioned.
        self._band_regions = OrderedDict()
        self._band_texts = OrderedDict()
        self._band_tints = OrderedDict()
        for i, name in enumerate(BANDS):
            self._band_tints[name] = _BAND_TINTS[i % 2]
            region = pg.LinearRegionItem(
                values=(0.0, 0.0),
                brush=pg.mkBrush(*_BAND_TINTS[i % 2]),
                pen=pg.mkPen(0, 0, 0, 0),
                movable=False)
            region.setZValue(-20)
            spi.addItem(region)
            self._band_regions[name] = region
            text = pg.TextItem(html=_band_label_html(name, False),
                               anchor=(0.5, 0.0))
            text.setZValue(10)
            spi.addItem(text)
            self._band_texts[name] = text

        self._filter_items = []
        self._spec_curve = spi.plot(
            [], [], pen=pg.mkPen((20, 60, 160), width=1.1))

        self._spec_note = QtWidgets.QLabel('')
        self._spec_note.setWordWrap(True)
        self._spec_note.setStyleSheet('color:#555; font-size:11px;')
        v.addWidget(self._spec_note)

        chips = QtWidgets.QHBoxLayout()
        chips.setSpacing(2)
        self._band_chips = OrderedDict()
        for name in BANDS:
            chip = QtWidgets.QLabel(name)
            chip.setStyleSheet(_CHIP_UNKNOWN)
            self._band_chips[name] = chip
            chips.addWidget(chip)
        chips.addStretch(1)
        v.addLayout(chips)

        self._rebuild_band_regions()
        self._rebuild_filter_markers()

    def _effective_filters(self):
        """The filters apply_filters would actually have applied at this fs.

        It silently drops a low-pass at or above Nyquist, and a notch up there
        cannot be built at all — so a window that echoed the toolbar's settings
        verbatim would be naming filters that never ran, which is the same class
        of error this panel exists to prevent. Mirrors the conditions in
        gui.processing._display_chain_sos, which is what filter_caveats measures.

        Returns (hp, lp, notch, dropped); `dropped` names the requested filters
        that could not apply, so the note can say so out loud.
        """
        nyq = 0.5 * float(self._fs)
        out = []
        dropped = []
        for hz, label in ((self._hp, 'HP'), (self._lp, 'LP'),
                          (self._notch, 'notch')):
            if hz and not _hz_ok(hz, nyq):
                dropped.append('{} {:g} Hz'.format(label, hz))
                out.append(None)
            else:
                out.append(float(hz) if hz else None)
        return out[0], out[1], out[2], dropped

    def _spec_top_hz(self):
        """Top of the frequency axis.

        The band table stops at 80 Hz, but a low-pass or notch above that still
        has to be visible — a filter marker off the edge of the plot is a filter
        the reviewer cannot see. Never above Nyquist: nothing was recorded there.
        """
        nyq = 0.5 * float(self._fs)
        top = min(nyq, 80.0)
        _, lp, notch, _ = self._effective_filters()
        for hz in (lp, notch):
            if hz:
                top = max(top, min(nyq, float(hz) * 1.15))
        return max(1.0, top)

    def _rebuild_band_regions(self):
        """Place the band columns, clipped to what this recording contains."""
        top = self._spec_top_hz()
        for name, (lo, hi) in BANDS.items():
            band_hi = min(float(hi), top)
            band_lo = min(float(lo), band_hi)
            self._band_regions[name].setRegion((band_lo, band_hi))
            self._band_texts[name].setPos(0.5 * (band_lo + band_hi), 0.0)
        self._spec_pw.getPlotItem().setXRange(0.0, top, padding=0)
        self._spec_pw.getPlotItem().vb.setLimits(xMin=0.0, xMax=top)

    def _rebuild_filter_markers(self):
        """Draw the active HP / LP / notch onto the spectrum.

        The markers are the whole point of the panel being honest: a reviewer
        looking at a spectrum that dives at 70 Hz must be able to see at a
        glance that the dive is the low-pass. Only filters that apply_filters
        would actually have applied are drawn (a low-pass at or above Nyquist
        is dropped there, so claiming one here would be its own fiction).
        """
        spi = self._spec_pw.getPlotItem()
        for item in self._filter_items:
            spi.removeItem(item)
        self._filter_items = []
        if not self._filters_known:
            return

        hp, lp, notch, _ = self._effective_filters()
        top = self._spec_top_hz()

        def stopband(lo, hi):
            band = pg.LinearRegionItem(
                values=(float(lo), float(hi)),
                brush=pg.mkBrush(*_STOPBAND_BRUSH),
                pen=pg.mkPen(0, 0, 0, 0),
                movable=False)
            band.setZValue(-15)
            spi.addItem(band)
            self._filter_items.append(band)

        def marker(hz, text):
            line = pg.InfiniteLine(
                pos=float(hz), angle=90, movable=False,
                pen=pg.mkPen(_FILTER_PEN[0], _FILTER_PEN[1], _FILTER_PEN[2],
                             width=1, style=QtCore.Qt.DashLine),
                label=text,
                labelOpts={'position': 0.93, 'color': (70, 70, 70),
                           'fill': (255, 255, 255, 190), 'movable': False})
            line.setZValue(5)
            spi.addItem(line)
            self._filter_items.append(line)

        if hp:
            stopband(0.0, min(hp, top))
            if hp < top:
                marker(hp, 'HP {:g}'.format(hp))
        if lp and lp < top:
            stopband(lp, top)
            marker(lp, 'LP {:g}'.format(lp))
        if notch and notch < top:
            marker(notch, 'notch {:g}'.format(notch))

    def _spectrum_slice(self):
        """Samples the spectrum is computed from, and the span they cover.

        Returns (segment, t0, t1, truncated); segment is None when the visible
        span holds too few samples for an estimate worth drawing. The slice is
        capped at MAX_SPECTRUM_SECONDS and taken from the CENTRE of the view —
        the centre is where the reviewer put the thing they are looking at.
        """
        n = int(self._data.shape[0])
        if n == 0:
            return None, 0.0, 0.0, False
        fs = float(self._fs)
        x0, x1 = self._pw.getPlotItem().viewRange()[0]
        s0 = max(0, int(x0 * fs))
        s1 = min(n, int(np.ceil(x1 * fs)))
        if s1 - s0 < MIN_SPECTRUM_SAMPLES:
            return None, float(x0), float(x1), False
        cap = int(MAX_SPECTRUM_SECONDS * fs)
        truncated = (s1 - s0) > cap
        if truncated:
            mid = (s0 + s1) // 2
            s0 = max(0, mid - cap // 2)
            s1 = min(n, s0 + cap)
        return self._data[s0:s1], s0 / fs, s1 / fs, truncated

    def refresh_spectrum(self):
        """Recompute the spectrum and band table for the visible span, now.

        Fired by the debounce timer, and called directly whenever the inputs
        change underneath it (new data, new filter settings, panel re-enabled).
        """
        if not self.chk_spec.isChecked():
            return
        seg, t0, t1, truncated = self._spectrum_slice()
        if seg is None:
            self._blank_spectrum(
                'Selection is shorter than {} samples — too short to estimate '
                'a spectrum. Zoom out.'.format(MIN_SPECTRUM_SAMPLES))
            return

        freqs, power = psd(seg, self._fs)
        powers = band_powers(freqs, power)
        # The EFFECTIVE filters, not the raw toolbar values. Every other
        # consumer goes through _effective_filters(); this one did not, which
        # made it the only path that could hand _butter_sos a cutoff at or above
        # Nyquist. set_filters(70.0, None, None) at fs=100 then raised
        # "Digital filter critical frequencies must be 0 < Wn < 1" out of a
        # setter. Unreachable through the toolbar today - the HP combo stops at
        # 5 Hz and every recording is resampled to 250 Hz - but the widget's own
        # tests construct it at fs=100, so nothing structural prevents it.
        hp_e, lp_e, notch_e, _dropped = self._effective_filters()
        caveats = (filter_caveats(self._fs, hp=hp_e, lp=lp_e, notch=notch_e)
                   if self._filters_known else None)

        top = self._spec_top_hz()
        vis = np.asarray(freqs) <= top
        f_vis = np.asarray(freqs)[vis]
        p_vis = np.asarray(power)[vis]
        pmax = float(p_vis.max()) if p_vis.size else 0.0
        if not np.isfinite(pmax):
            # NaN slips past `pmax <= 0.0` — nan <= 0.0 is False — and reaches
            # setYRange as log10(nan), where pyqtgraph raises. The exception
            # escapes after the curve has been redrawn but before the band
            # chips and the note are updated, leaving an empty plot above a
            # fully populated band table describing a DIFFERENT time range,
            # with nothing on screen saying so. In the frozen build the
            # excepthook then shows a modal error dialog once per debounce tick
            # while the reviewer scrubs.
            #
            # A span the amplifier did not record is not a span with a quiet
            # spectrum, so it gets its own message rather than the flat-channel
            # one below.
            self._blank_spectrum(
                'This span contains samples that are not finite, so no '
                'spectrum can be estimated for it.')
            return
        if pmax <= 0.0:
            self._blank_spectrum(
                'No power in this span — the channel is flat here, which is '
                'itself worth checking (a disconnected electrode looks like '
                'this).')
            return

        # The axis is logarithmic, so a bin that came out as exactly zero would
        # be -inf. Clamp to the bottom of the plotted range instead: the curve
        # then rests on the axis floor, which reads as "below this range" rather
        # than taking the whole plot with it.
        floor = pmax * (10.0 ** -SPECTRUM_DECADES)
        self._spec_curve.setData(f_vis, np.maximum(p_vis, floor))
        spi = self._spec_pw.getPlotItem()
        spi.setYRange(np.log10(floor), np.log10(pmax) + 0.3, padding=0)
        # Band names ride at the top of the current range.
        for name in BANDS:
            pos = self._band_texts[name].pos()
            self._band_texts[name].setPos(pos.x(), np.log10(pmax) + 0.3)

        self._update_band_chips(powers, caveats)
        self._update_spectrum_note(freqs, t0, t1, truncated)
        self._band_powers = powers

    def _blank_spectrum(self, message):
        self._spec_curve.setData([], [])
        self._spec_note.setText(message)
        self._spec_note.setToolTip('')
        for name, chip in self._band_chips.items():
            chip.setText(name)
            chip.setStyleSheet(_CHIP_UNKNOWN)
            chip.setToolTip('No spectrum for this selection.')
            self._band_texts[name].setHtml(_band_label_html(name, False))
            self._band_regions[name].setBrush(
                pg.mkBrush(*self._band_tints[name]))
        self._compromised = []
        self._band_powers = None

    def _update_band_chips(self, powers, caveats):
        """One chip per clinical band: absolute power, share, and the caveat.

        The caveat is not decoration. A gamma figure taken under a 70 Hz
        low-pass and a 60 Hz notch is roughly two thirds of the real one, and
        nothing in the number itself says so — so the chip says it, and the
        tooltip quotes the measured surviving fraction.
        """
        total = 0.0
        for p in powers.values():
            total += float(p)
        flagged = []
        for name, (lo, hi) in BANDS.items():
            chip = self._band_chips[name]
            p = float(powers[name])
            pct = (100.0 * p / total) if total > 0 else 0.0
            head = '{} {:.3g} µV² · {:.0f}%'.format(name, p, pct)
            where = '{} {:g}–{:g} Hz'.format(name, lo, hi)
            cav = None if caveats is None else caveats.get(name)
            if cav is None:
                chip.setText(head + ' · ?')
                chip.setStyleSheet(_CHIP_UNKNOWN)
                chip.setToolTip('{}\n{}'.format(where, _FILTERS_UNKNOWN_NOTE))
            elif cav.compromised:
                flagged.append(name)
                chip.setText(head + ' · filtered')
                chip.setStyleSheet(_CHIP_FLAGGED)
                chip.setToolTip(
                    '{}\nShaped by the DISPLAY filters, {}.\nThis figure is '
                    'partly a statement about the filter — it is not a '
                    'correction factor, the power is gone.'.format(
                        where, cav.note or 'see the filter markers'))
            else:
                chip.setText(head)
                chip.setStyleSheet(_CHIP_OK)
                chip.setToolTip(
                    '{}\nThe display filters leave {:.0f}% of this band.'.format(
                        where, 100.0 * cav.retained))
            # Colour the column itself, not just the chip below it. The
            # reviewer's eye is on the spectrum when they judge the shape, and
            # the shape inside an amber column is the filter's.
            is_flagged = cav is not None and cav.compromised
            self._band_texts[name].setHtml(_band_label_html(name, is_flagged))
            self._band_regions[name].setBrush(pg.mkBrush(
                *(_BAND_FLAGGED_TINT if is_flagged else self._band_tints[name])))
        self._compromised = flagged

    def _update_spectrum_note(self, freqs, t0, t1, truncated):
        res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
        x0, x1 = self._pw.getPlotItem().viewRange()[0]
        hp, lp, notch, dropped = self._effective_filters()
        if truncated:
            head = ('Welch spectrum of the middle {:.1f} s (t {:.1f}–{:.1f} s) '
                    'of the {:.1f} s on screen'.format(t1 - t0, t0, t1,
                                                       max(0.0, x1 - x0)))
        else:
            head = 'Welch spectrum of {:.1f} s (t {:.1f}–{:.1f} s)'.format(
                t1 - t0, t0, t1)
        bits = [head]
        if res > 0:
            bits.append('{:.2f} Hz bins'.format(res))
        bits.append(_filter_summary(hp, lp, notch) if self._filters_known
                    else 'display filter settings not supplied')
        text = ' · '.join(bits) + '.'
        if truncated:
            text += (' It is a sample of what is on screen, not all of it.')
        if dropped:
            text += (' {} {} not applied to this {:g} Hz recording: nothing '
                     'above the {:g} Hz Nyquist limit exists to filter.'.format(
                         ' and '.join(dropped),
                         'was' if len(dropped) == 1 else 'were',
                         self._fs, 0.5 * float(self._fs)))
        if res > COARSE_BIN_HZ:
            text += (' Bins are wider than {:g} Hz: band edges are approximate '
                     'at this zoom.'.format(COARSE_BIN_HZ))
        if not self._filters_known:
            text += ' ' + _FILTERS_UNKNOWN_NOTE
        elif self._compromised:
            text += (' Bands shaped by the display filters: {} — those figures '
                     'describe the filter as much as the patient.'.format(
                         ', '.join(self._compromised)))
        self._spec_note.setText(text)

    def _on_spec_toggled(self, on):
        self._spec_pw.setVisible(bool(on))
        self._spec_note.setVisible(bool(on))
        for chip in self._band_chips.values():
            chip.setVisible(bool(on))
        if on:
            self.refresh_spectrum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_item(self):
        return self._pw.getPlotItem()

    def spectrum_plot_item(self):
        return self._spec_pw.getPlotItem()

    def link_x_to(self, other_plot_item):
        self._pw.getPlotItem().setXLink(other_plot_item)

    def unlink_x(self):
        self._pw.getPlotItem().setXLink(None)

    def set_x_range(self, x0, x1):
        self._pw.getPlotItem().setXRange(x0, x1, padding=0)

    def channel_index(self):
        return self._channel_idx

    def sync_checked(self):
        return self.chk_sync.isChecked()

    def set_filters(self, hp, lp, notch):
        """Tell the window which DISPLAY filters produced the trace it holds.

        All three are required, because "not told" and "switched off" are
        different states and only one of them is safe to report: an inspector
        that has not been told assumes nothing and marks every band unverified,
        which is why this is not a set of optional keyword arguments.

        Call it whenever the toolbar filters change, not only at construction —
        the window keeps its data across a filter change (set_data) and the band
        caveats would otherwise describe the previous settings.
        """
        self._hp = float(hp) if hp else None
        self._lp = float(lp) if lp else None
        self._notch = float(notch) if notch else None
        self._filters_known = True
        self._rebuild_band_regions()
        self._rebuild_filter_markers()
        self.refresh_spectrum()

    def filters_known(self):
        return self._filters_known

    def compromised_bands(self):
        """Names of the bands the display filters have already shaped."""
        return list(self._compromised)

    def current_band_powers(self):
        """Band powers behind the chips, in µV², or None if there is no
        spectrum for the current selection."""
        if self._band_powers is None:
            return None
        return OrderedDict(self._band_powers)

    def set_data(self, channel_name, data1d, t, reference_intervals=None):
        """Refresh in place after a filter or montage change.

        Previously MainWindow closed every open inspector on any such change and
        never reopened them, so the detail window vanished exactly when a
        reviewer was using it to decide whether something was muscle artefact.

        Note that this does NOT update the filter settings — the caller must
        also call set_filters, because only the caller knows what produced the
        array it is passing in.
        """
        self._channel_name = channel_name
        self._data = np.asarray(data1d, dtype=np.float32)
        self._t = np.asarray(t, dtype=np.float32)
        self.setWindowTitle('Channel — {}'.format(channel_name))
        self._curve.setData(self._t, self._data)
        if reference_intervals is not None:
            self._reference_intervals = [(float(s), float(e))
                                         for s, e in reference_intervals]
            # Rebuilt rather than left alone: the bands used to be drawn once at
            # construction, so a montage change left the previous file's — or a
            # blind session's hidden — reference state on screen.
            self._rebuild_references()
        self._update_stats(self._data)
        self.refresh_spectrum()

    def sync_toggle_signal(self):
        return self.chk_sync.stateChanged

    # ---- reference annotations ---------------------------------------
    def _rebuild_references(self):
        pi = self._pw.getPlotItem()
        for r in self._ref_items:
            pi.removeItem(r)
        self._ref_items = []
        for s, e in self._reference_intervals:
            r = pg.LinearRegionItem(
                values=(s, e),
                brush=pg.mkBrush(50, 160, 70, 40),
                pen=pg.mkPen(50, 160, 70, 140, width=1),
                movable=False)
            r.setZValue(-10)
            r.setVisible(self._refs_visible)
            pi.addItem(r)
            self._ref_items.append(r)

    def set_references_visible(self, visible):
        """Show or hide the ground-truth bands — same contract as SignalView
        and ProbStrip.

        This window drew them unconditionally at construction and had no way to
        take them away, so a blind session could have the answer key open in a
        detail window while the export recorded blind_mode: true. The state is
        remembered, so bands rebuilt later (set_data) stay hidden.
        """
        self._refs_visible = bool(visible)
        for r in self._ref_items:
            r.setVisible(self._refs_visible)

    def references_visible(self):
        return self._refs_visible

    # ------------------------------------------------------------------
    def _slice_between(self, x0, x1):
        s0 = max(0, int(x0 * self._fs))
        s1 = min(self._data.shape[0], int(x1 * self._fs))
        if s1 <= s0:
            return self._data[:0]
        return self._data[s0:s1]

    def _on_range_changed(self, *_):
        x0, x1 = self._pw.getPlotItem().viewRange()[0]
        sl = self._slice_between(x0, x1)
        self._update_stats(sl)
        if self.chk_autofit.isChecked():
            amp = _fit_amplitude(sl)
            if int(self.cb_amp.currentText()) != amp:
                self.cb_amp.blockSignals(True)
                self.cb_amp.setCurrentText(str(amp))
                self.cb_amp.blockSignals(False)
                self._set_y_range(amp)
        # sigRangeChanged also fires for Y-only changes (every amplitude change,
        # and every drag of the Y axis) which cannot alter the spectrum, so the
        # X range is compared before arming the timer.
        key = (round(float(x0), 4), round(float(x1), 4))
        if key != self._last_spec_range:
            self._last_spec_range = key
            self._spec_timer.start()

    def _fit_to_visible(self):
        x0, x1 = self._pw.getPlotItem().viewRange()[0]
        sl = self._slice_between(x0, x1)
        amp = _fit_amplitude(sl)
        self.cb_amp.setCurrentText(str(amp))

    def _on_amp_change(self, txt):
        try:
            self._set_y_range(float(txt))
        except ValueError:
            pass

    def _set_y_range(self, amp_uv):
        self._pw.getPlotItem().setYRange(-amp_uv, amp_uv, padding=0)

    def _update_stats(self, data_slice):
        if data_slice.size == 0:
            self._stats_lbl.setText('')
            return
        rms = float(np.sqrt((data_slice.astype('float64') ** 2).mean()))
        pk = float(np.max(np.abs(data_slice)))
        self._stats_lbl.setText(
            'RMS {:.1f} µV · peak {:.1f} µV'.format(rms, pk))

    def _on_axis_drag(self, new_amp, finished):
        """Live-rescale while the user drags the Y axis. On release we
        snap the combobox to the closest preset so the toolbar reads a
        clean number."""
        self._set_y_range(new_amp)
        if finished:
            nearest = min(AMPLITUDES_UV, key=lambda v: abs(v - new_amp))
            self.cb_amp.blockSignals(True)
            self.cb_amp.setCurrentText(str(nearest))
            self.cb_amp.blockSignals(False)
            self._set_y_range(nearest)

    # ------------------------------------------------------------------
    def closeEvent(self, ev):
        self._spec_timer.stop()
        self.closed.emit(self._channel_idx)
        super().closeEvent(ev)
