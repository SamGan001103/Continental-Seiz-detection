"""Multi-channel EEG trace viewer.

One pyqtgraph PlotWidget holds all traces at vertical offsets. The
y-axis is a custom `_ChannelAxis` that labels each row and lets the
user drag it to rescale sensitivity (like commercial EEG viewers).
The viewbox uses a gentler wheel-zoom factor for smooth scrolling."""
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np

from gui.processing import decimate_for_display
from gui.widgets.time_axis import axis_label, format_cursor, make_time_axis

# Widest view at which an individual interictal mark still means something.
# At 1600 px a 180 s span puts about nine pixels on a tenth-of-a-second
# discharge's neighbourhood; much beyond that and marks merge into a band.
MARKS_LEGIBLE_MAX_SPAN_S = 180.0

pg.setConfigOptions(antialias=False, useOpenGL=False, background='w',
                    foreground='k')


# ---------------------------------------------------------------------- helpers

class TimeCursor:
    """Dashed vertical cursor that follows the mouse X across a pyqtgraph
    PlotWidget. The time readout is rendered by the owning widget via the
    `on_hover` callback (called with the hover time in seconds, or None
    when the mouse leaves the ViewBox) — typically into a QLabel below
    the time axis so it never covers the traces.

    Construct after the PlotWidget has its ViewBox and ranges set."""

    def __init__(self, plot_widget, on_hover=None):
        self._pw = plot_widget
        self._on_hover = on_hover
        pi = plot_widget.getPlotItem()
        self.vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(60, 60, 60, 180, width=1))
        self.vline.setZValue(50)
        self.vline.hide()
        pi.addItem(self.vline, ignoreBounds=True)

        plot_widget.scene().sigMouseMoved.connect(self._on_move)

    def reattach(self):
        """Re-add the vline to the plot after a PlotItem.clear() removed
        it. The scene-level sigMouseMoved connection survives clearing,
        so only the item needs to be re-added."""
        pi = self._pw.getPlotItem()
        pi.addItem(self.vline, ignoreBounds=True)
        self.vline.hide()

    def _on_move(self, scene_pos):
        vb = self._pw.getPlotItem().vb
        if vb is None or not vb.sceneBoundingRect().contains(scene_pos):
            self.vline.hide()
            if self._on_hover is not None:
                self._on_hover(None)
            return
        view_pos = vb.mapSceneToView(scene_pos)
        t = float(view_pos.x())
        self.vline.setPos(t)
        self.vline.show()
        if self._on_hover is not None:
            self._on_hover(t)


_TIME_LBL_STYLE = (
    'color:#333; padding:2px 8px; '
    'font-family: Consolas, "Courier New", monospace; font-size: 11px;')


def _make_time_label():
    """Small monospace QLabel, right-aligned, sits below the plot's time
    axis and shows the hover time."""
    lbl = QtWidgets.QLabel(format_cursor(None))
    lbl.setStyleSheet(_TIME_LBL_STYLE)
    lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    return lbl


def _update_time_label(lbl, t, start=None):
    """`start` is the recording's start datetime, or None if the header had
    none — then the readout is elapsed time only, as before."""
    lbl.setText(format_cursor(t, start))


# ---------------------------------------------------------------------- axis

class _ChannelAxis(pg.AxisItem):
    def __init__(self, ch_names, spacing=200.0, orientation='left'):
        super().__init__(orientation=orientation)
        self._ch_names = list(ch_names)
        self._spacing = spacing
        self.setWidth(64)
        self._drag_cb = None         # callable(new_sensitivity_uv, finished=False)
        self._drag_start = None
        self._sens_of_spacing = None  # set by the owner: spacing -> sensitivity
        # Hint to users that the axis is interactive.
        self.setCursor(QtCore.Qt.SizeVerCursor)

    def set_channels(self, names):
        self._ch_names = list(names)
        self.picture = None
        self.update()

    def set_spacing(self, sp):
        self._spacing = float(sp)
        self.picture = None
        self.update()

    def set_drag_callback(self, cb):
        """cb(new_sensitivity_uv: float, finished: bool)"""
        self._drag_cb = cb

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            idx = int(round(-v / self._spacing))
            out.append(self._ch_names[idx] if 0 <= idx < len(self._ch_names) else '')
        return out

    def tickValues(self, minVal, maxVal, size):
        ticks = [(-i * self._spacing) for i in range(len(self._ch_names))]
        visible = [t for t in ticks if minVal <= t <= maxVal]
        return [(self._spacing, visible)]

    # Vertical drag on the axis rescales amplitude. Use left-button drag;
    # pyqtgraph routes MouseDragEvent here when the axis is hit.
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            # Recover the current sensitivity from the owner rather than
            # assuming a fixed spacing/sensitivity ratio — that ratio now
            # depends on display DPI and channel count.
            sens0 = (self._sens_of_spacing() if self._sens_of_spacing
                     else self._spacing / 3.0)
            self._drag_start = (float(ev.buttonDownPos().y()), sens0)
            return
        if self._drag_start is None:
            return
        _, sens0 = self._drag_start
        dy = float(ev.pos().y()) - float(ev.buttonDownPos().y())
        # Drag down (positive dy in scene) = larger amplitudes per division
        # (lower sensitivity i.e. bigger µV/div). 140 px ≈ 2× change.
        factor = 2.0 ** (dy / 140.0)
        new_sens = max(5.0, min(2000.0, sens0 * factor))
        finished = ev.isFinish()
        if self._drag_cb is not None:
            self._drag_cb(new_sens, finished)
        if finished:
            self._drag_start = None


# ---------------------------------------------------------------------- ViewBox

class _SignalViewBox(pg.ViewBox):
    """ViewBox with a softened wheel-zoom factor. ~8% per wheel tick —
    roughly half of pyqtgraph's default (~12%), so scrolling feels smooth
    without slogging through increments. Reused by the channel inspector so
    wheel behaviour is consistent across windows.

    Wheel zooms with no modifier, deliberately. It was briefly changed so that
    scroll panned and modifier+scroll zoomed, because a trackpad two-finger
    scroll is a wheel event and could zoom a reviewer out to the whole
    recording without their asking. That was reverted: free zoom is how this
    view is actually driven, and requiring a modifier for the most-used
    gesture costs more in a review than it buys.

    What made the accidental zoom dangerous is fixed elsewhere, and properly:
    the wide view now draws the min/max of the *filtered* signal rather than
    one sample in 187 with the filters silently dropped
    (gui/display_pipeline.py), and the sweep control states the span actually
    on screen instead of a preset the view is not showing (gui/app.py,
    _sync_timebase_combo). A reviewer who zooms out now lands somewhere
    honest that says what it is, so there is no longer a state worth
    protecting them from reaching quickly.
    """

    WHEEL_STEP = 1.08            # ~8% per 120-unit wheel tick

    def wheelEvent(self, ev, axis=None):
        mask = self.state['mouseEnabled']
        delta = ev.delta() if hasattr(ev, 'delta') else ev.angleDelta().y()
        # Negative delta (wheel down) zooms OUT → factor > 1
        s = self.WHEEL_STEP ** (-delta / 120.0)
        sx = s if mask[0] else 1.0
        sy = s if mask[1] else 1.0
        center = self.mapToView(ev.pos())
        self.scaleBy((sx, sy), center=center)
        # Measured dead end, recorded so it is not tried again: calling
        # self.updateMatrix() here was supposed to collapse the two scene
        # repaints a range change produces into one, by doing eagerly what
        # pyqtgraph defers to `prepareForPaint` inside the paint event. It
        # changes nothing — 2.00 repaints per tick either way over six
        # interleaved rounds, rendered frame byte-identical. The repaints come
        # from somewhere else.
        ev.accept()


# ---------------------------------------------------------------------- SignalView

# Trace rendered beyond the viewport so a pan has signal ready before the
# coalescing timer fires. Bounded: it used to be a whole extra span either
# side, which at a 10-minute view meant deriving 30 minutes of signal per
# redraw for the sake of the next scroll.
from gui.display_pipeline import FULLRATE_MAX_SPAN_S

PAN_PAD_S = 60.0

# Minimum horizontal room, in pixels, before a second annotation label is drawn.
# Sized for the longest text these files actually carry — 'REC START IIB CAL',
# about 17 characters — plus a gap, so two shown labels never touch.
MIN_LABEL_GAP_PX = 110.0

# Height of the annotation lane above the top trace, in channel-spacings.
# 0.5 is the ordinary top margin; the extra 0.9 is the empty band the labels
# sit in so they never cover a trace.
ANNOT_LANE_FRACTION = 1.4

# Where along the vertical marker line its text sits, 0 = bottom, 1 = top.
# Just under the top so the text lands inside the lane above the first channel.
ANNOT_LABEL_POSITION = 0.985


def _set_label_shown(label, shown, state, key):
    """As below, but skips the write when the label is already in that state.

    `setOpacity` repaints the item, so rewriting all 41 labels on every redraw
    cost 12.2 ms of a 32.6 ms scroll step — more than a third of it — to set
    most of them to the value they already had. Which labels are shown barely
    changes while panning at a fixed zoom, so almost every one of those writes
    was wasted.
    """
    if state.get(key) is shown:
        return
    state[key] = shown
    _apply_label_shown(label, shown)


def _apply_label_shown(label, shown):
    """Show or hide an InfiniteLine's text label, on either pyqtgraph.

    Opacity rather than setVisible, and not by preference: pyqtgraph 0.11.1 —
    the version on the legacy 3.6 stack — has a broken
    `InfLineLabel.setVisible`. Making a label visible calls `self.updateText()`,
    a method that does not exist in that release, so it raises AttributeError.
    Only the True direction is affected, which is exactly the kind of asymmetry
    that passes a quick check and fails in use.

    Opacity works on both versions, does the same job, and needs no version
    test — a runtime probe would be one more thing to keep right.
    """
    try:
        label.setOpacity(1.0 if shown else 0.0)
    except Exception:                                       # noqa: BLE001
        # Some other pyqtgraph. A visible label is the safe failure: crowded
        # text is a legibility problem, a raised exception during a redraw is
        # a broken application.
        pass


class SignalView(QtWidgets.QWidget):
    regionChanged = QtCore.pyqtSignal(int, float, float)
    viewRangeChanged = QtCore.pyqtSignal(float, float)
    channelDoubleClicked = QtCore.pyqtSignal(int)       # channel index
    sensitivityChanged = QtCore.pyqtSignal(float, bool) # (new_µV/div, finished)
    longBuildStarted = QtCore.pyqtSignal()   # an uncached overview is building
    longBuildFinished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        self._sensitivity_uv = 30.0
        self._channel_names = []
        # Placeholder: the real value needs _pw's height and DPI, so it is
        # computed in set_signal() / set_sensitivity() once those exist.
        self._spacing = self._sensitivity_uv * 3.0
        self._ch_axis = _ChannelAxis([], spacing=self._spacing)
        self._ch_axis.set_drag_callback(self._on_axis_drag)
        self._ch_axis._sens_of_spacing = lambda: self._sensitivity_uv

        vb = _SignalViewBox()
        # Clock time, not elapsed seconds. See gui/widgets/time_axis.py: on a
        # day-long recording "48,213 seconds" cannot be matched against a
        # nursing note or a video review, and 03:22 can.
        self._t_axis = make_time_axis('bottom')
        self._pw = pg.PlotWidget(viewBox=vb,
                                 axisItems={'left': self._ch_axis,
                                            'bottom': self._t_axis})
        pi = self._pw.getPlotItem()
        pi.setMouseEnabled(x=True, y=False)
        pi.setLabel('bottom', axis_label(None))
        pi.showGrid(x=True, y=False, alpha=0.2)
        # pyqtgraph's right-click menu is removed, not merely tidied. It
        # offers "Transforms > Power Spectrum (FFT)" and "Log X/Y", each one
        # click away from replacing the EEG trace with something that is no
        # longer an EEG trace - and the reviewer has no obvious way back. A
        # clinician who right-clicks looking for a panel they closed can
        # silently destroy the uV/mm calibration this widget exists to
        # guarantee. It also exposes an undisclaimered image export, which
        # would put a screenshot of a research prototype into a report with no
        # "not for diagnostic use" on it.
        #
        # Everything here that a reviewer legitimately needs - amplitude, sweep
        # speed, montage, filters - is on the toolbar, where it is labelled.
        pi.setMenuEnabled(False)
        pi.hideButtons()
        self._pw.setMenuEnabled(False)
        pi.sigRangeChanged.connect(self._emit_range)
        self._pw.scene().sigMouseClicked.connect(self._on_scene_click)
        v.addWidget(self._pw)

        self._time_lbl = _make_time_label()
        v.addWidget(self._time_lbl)

        # The start time is read back off the axis rather than kept here too,
        # so there is one source of truth for when the recording began.
        self._time_cursor = TimeCursor(
            self._pw,
            on_hover=lambda t: _update_time_label(
                self._time_lbl, t, self._t_axis.start()))

        self._curves = []
        self._ref_items = []
        self._regions = {}
        self._grid_lines = []
        self._channel_names = []
        self._fs = 250
        self._full_unclipped = None    # retired: nothing full-length is held here
        self._provider = None          # DisplayPipeline; renders spans on demand
        self._building = False         # inside a long, uncached overview build
        self._full_offsets = None
        self._ref_intervals = []       # cached (start, stop) tuples for refresh
        self._annot_items = []         # EDF+ technician markers (read-only)
        self._annot_label_state = {}   # index -> shown, to skip no-op repaints
        self._source_annotations = []
        self._event_cache = []         # cached event dicts for refresh
        self._refs_visible = True      # toolbar toggle state

        self._redec_timer = QtCore.QTimer(self)
        self._redec_timer.setSingleShot(True)
        # Coalescing delay before a redraw. Every range change restarts it, so
        # this is dead time after the reviewer stops moving — and at 30 ms it
        # was the largest single component of scroll latency once the redraw
        # itself came down to 9 ms. A reviewer scrolling a recording reported
        # it as the trace "taking time to load", which is exactly what a
        # do-nothing delay looks like.
        self._redec_timer.setInterval(12)
        self._redec_timer.timeout.connect(self._redecimate_visible)
        pi.sigRangeChanged.connect(lambda *_: self._redec_timer.start())
        # Marks appear and disappear with the zoom, so a range change has to
        # re-run them as well as the traces.
        pi.sigRangeChanged.connect(self._on_range_marks)

    # ------------------------------------------------------------------
    def _emit_range(self, *_):
        x0, x1 = self._pw.getPlotItem().viewRange()[0]
        self.viewRangeChanged.emit(float(x0), float(x1))

    def _on_scene_click(self, event):
        try:
            is_double = bool(event.double())
        except Exception:
            return
        if not is_double:
            return
        vb = self._pw.getPlotItem().vb
        pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        view_pos = vb.mapSceneToView(pos)
        idx = int(round(-view_pos.y() / self._spacing))
        if 0 <= idx < len(self._channel_names):
            self.channelDoubleClicked.emit(idx)

    def _on_axis_drag(self, new_sens, finished):
        self.set_sensitivity(new_sens, rebuild_layout=True)
        self.sensitivityChanged.emit(new_sens, finished)

    # ---- data --------------------------------------------------------
    def set_signal(self, data_full, fs, channel_names, sensitivity_uv=30.0,
                   initial_span_s=30.0, preserve_view=False):
        """Draw an already-derived [K, N] array as given: no filters, no montage.

        Kept for callers and tests that hand the view a finished array. The
        application itself uses set_provider, so nothing full-length is ever
        derived on its behalf — see gui/display_pipeline.py for why.
        """
        from gui.display_pipeline import DisplayPipeline
        arr = np.ascontiguousarray(data_full, dtype=np.float32)
        provider = DisplayPipeline(arr, fs, montage=None,
                                   row_labels=list(channel_names))
        self.set_provider(provider, fs, channel_names,
                          sensitivity_uv=sensitivity_uv,
                          initial_span_s=initial_span_s,
                          preserve_view=preserve_view)

    def set_provider(self, provider, fs, channel_names, sensitivity_uv=30.0,
                     initial_span_s=30.0, preserve_view=False):
        """Point the view at a DisplayPipeline and draw the current span.

        The view holds no signal of its own any more. Every redraw asks the
        provider for exactly the samples on screen (plus a pan margin), already
        filtered and montaged; SignalView still owns clipping, so sensitivity
        changes stay cheap. This is what lets a 24-hour recording redraw in the
        same time as a 10-minute one — the previous design cached a filtered,
        montaged copy of the entire recording here and it froze a 16 GB machine.

        preserve_view : if True and the viewport was already populated,
            keep the current x-range. Used when the caller is refreshing
            the same file with a new montage / filter / sensitivity, so
            the reviewer doesn't get yanked back to t=0.
        """
        pi = self._pw.getPlotItem()
        # Capture current x-range before tearing down so we can restore it.
        had_signal = self._provider is not None
        keep_x = preserve_view and had_signal
        if keep_x:
            prev_x0, prev_x1 = pi.viewRange()[0]

        pi.clear()
        self._curves = []
        # Curves are rebuilt with the trace pen below, so any envelope
        # pen from the previous file is gone; say so or the next redraw
        # at the same zoom will not re-apply it.
        self._pen_is_envelope = False
        self._regions = {}
        self._ref_items = []
        self._grid_lines = []
        # pi.clear() removed the annotation markers too. Forgetting the list
        # would leave stale item objects that a later set_source_annotations
        # tries to remove from a plot they are no longer in — and, worse, the
        # previous recording's technician notes still listed as this one's.
        self._annot_items = []
        self._annot_label_state = {}
        # pi.clear() nuked the hover-cursor overlay; put it back before
        # the user's next mouse move needs it.
        self._time_cursor.reattach()
        self._channel_names = list(channel_names)
        self._fs = fs
        self._provider = provider
        self._full_unclipped = None
        self._sensitivity_uv = float(sensitivity_uv)
        n_ch = provider.n_rows
        self._spacing = self._compute_spacing(n_ch)
        self._full_offsets = -np.arange(n_ch, dtype=np.float32) * self._spacing
        self._ch_axis.set_channels(self._channel_names)
        self._ch_axis.set_spacing(self._spacing)

        for i in range(n_ch):
            line = pg.InfiniteLine(pos=self._full_offsets[i], angle=0,
                                   pen=pg.mkPen((230, 230, 230), width=0.5))
            pi.addItem(line)
            self._grid_lines.append(line)

        for i in range(n_ch):
            curve = pi.plot(
                [], [],
                pen=pg.mkPen((40, 40, 40), width=0.7),
            )
            self._curves.append(curve)

        self._update_y_range()
        duration_s = provider.duration_s
        pi.vb.setLimits(xMin=0.0, xMax=float(duration_s))
        if keep_x:
            pi.setXRange(prev_x0, prev_x1, padding=0)
        else:
            pi.setXRange(0.0, min(initial_span_s, duration_s), padding=0)
        self._redecimate_visible()

    def set_sensitivity(self, sensitivity_uv, rebuild_layout=False):
        """Change µV/div without re-running filter/montage. Rebuilds the
        y-axis spacing and re-clips the visible slice."""
        self._sensitivity_uv = float(sensitivity_uv)
        self._spacing = self._compute_spacing(
            len(self._channel_names) or 1)
        if self._provider is None:
            return
        n_ch = self._provider.n_rows
        self._full_offsets = -np.arange(n_ch, dtype=np.float32) * self._spacing
        self._ch_axis.set_spacing(self._spacing)
        if rebuild_layout:
            pi = self._pw.getPlotItem()
            for i, line in enumerate(self._grid_lines):
                line.setPos(self._full_offsets[i])
            self._update_y_range()
            # pg.LinearRegionItem caches its render on first addItem and
            # doesn't always repaint when the view's Y-range changes.
            # Force a rebuild so ref/event bands track the new layout.
            self._rebuild_overlays()
        self._redecimate_visible()

    def _rebuild_overlays(self):
        """Recreate reference and AI-event regions from their caches so
        they span the freshly-resized view cleanly."""
        refs = list(self._ref_intervals)
        events = list(self._event_cache)
        self.set_references(refs)
        self.set_events(events)

    def _update_y_range(self):
        if self._full_offsets is None:
            return
        n_ch = len(self._full_offsets)
        pi = self._pw.getPlotItem()
        # A lane above the top trace for annotation text, but only when there
        # is any. Seen on a real display, the labels were drawn over the
        # bottom channel: legible in themselves — they carry a white
        # background — while hiding the EEG underneath, which is the one thing
        # on this widget that must never be covered. A recording with no
        # annotations keeps the tighter range and loses no screen height.
        head = (ANNOT_LANE_FRACTION if getattr(self, '_source_annotations', None)
                else 0.5)
        pi.setYRange(-(n_ch - 0.5) * self._spacing,
                     head * self._spacing, padding=0)

    def _begin_long_build(self):
        """Say, before a forty-second call, that the application is alive.

        The cursor and the signal go up first and one repaint is pumped, so
        the reviewer sees the busy state rather than a white rectangle. The
        pumped events are why _building exists: a range change arriving here
        mid-build would start a second one.
        """
        self._building = True
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.longBuildStarted.emit()
        QtWidgets.QApplication.processEvents()

    def _end_long_build(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self._building = False
        self.longBuildFinished.emit()

    def _redecimate_visible(self):
        if self._provider is None or not self._curves:
            return
        if self._building:
            # A long build pumps the event loop so the window keeps painting,
            # which can deliver another range change straight back here. One
            # build at a time -- but the range that arrived is the one the
            # reviewer is now looking at, so ask again shortly rather than
            # dropping it. Dropped, the axis would read the new span while
            # the traces still showed the old one.
            self._redec_timer.start()
            return
        pi = self._pw.getPlotItem()
        x0, x1 = pi.viewRange()[0]
        px_w = max(200, int(self._pw.viewport().width() or 1600))
        n = self._provider.n_samples
        span = max(1e-6, x1 - x0)
        pad = min(span, PAN_PAD_S)
        s0 = max(0, int((x0 - pad) * self._fs))
        s1 = min(n, int((x1 + pad) * self._fs))
        if s1 <= s0:
            return
        target = int(px_w * (s1 - s0) / (span * self._fs)) if span > 0 else px_w
        target = max(512, min(target, 20000))
        # The provider filters, montages and decimates just this span. Work
        # per redraw is therefore bounded by the viewport, never by the file --
        # EXCEPT on the first zoom out into a filter and montage combination
        # the study bundle does not carry, where the whole recording has to be
        # filtered into an envelope first. That is ~40 s on a recording-day,
        # it happens on this thread, and with no cursor change the window
        # simply goes white and Windows marks it Not Responding. A busy cursor
        # and one pumped repaint turn a dead application into a working one.
        heavy = self._provider.needs_envelope_build(s0, s1)
        if heavy:
            self._begin_long_build()
        try:
            data_ds, t_ds = self._provider.render(s0, s1, target)
        finally:
            if heavy:
                self._end_long_build()
        # Clip to the current sensitivity so a spike that exceeds the row is
        # drawn flat at the lane edge rather than across the neighbour. Stride
        # decimation selects samples, so clipping after it is pointwise
        # identical to the old clip-then-decimate order.
        lvl = self._clip_level()
        clipped = np.clip(data_ds, -lvl, lvl)
        # Past the full-rate boundary each column is a min/max RANGE, not a
        # sample, so it must not be drawn as though it were a waveform. A
        # lighter, thinner pen reads as a band and tells the reviewer, without
        # a dialog, that morphology is not judged at this zoom. It also
        # answers the objection that sent this view to stride decimation in
        # the first place: at a trace-weight pen, sustained peaks at the clip
        # level fill the lane and look like black boxes.
        envelope = (x1 - x0) > FULLRATE_MAX_SPAN_S
        if envelope != self._pen_is_envelope:
            pen = (pg.mkPen((120, 120, 120), width=0.5) if envelope
                   else pg.mkPen((40, 40, 40), width=0.7))
            for curve in self._curves:
                curve.setPen(pen)
            self._pen_is_envelope = envelope
        for i, curve in enumerate(self._curves):
            # Negative-up: the universal EEG display convention. Negating at
            # render only — the arrays feeding inference are untouched.
            curve.setData(t_ds, -clipped[i] + self._full_offsets[i])
        # Label density depends on the zoom, so it is re-decided here rather
        # than once when the annotations are set.
        self._update_annotation_labels()


    # ------------------------------------------------------------------
    # Amplitude geometry
    #
    # The toolbar quotes sensitivity in µV/mm, the clinical convention, so the
    # renderer has to honour it in real millimetres rather than in arbitrary
    # "divisions". row_mm is the physical height of one channel row on THIS
    # display; spacing is therefore the microvolts that row spans.
    #
    # Traces are clipped at half the row so a large deflection fills its own
    # lane and stops before the neighbour's. Clipping at ±sensitivity — one
    # third of the row — was the old behaviour and it flat-topped 34 % of
    # samples at a clinical 7 µV/mm, destroying exactly the morphology a
    # reviewer judges on.
    # ------------------------------------------------------------------
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Row height in millimetres changed, so µV/mm implies a new spacing.
        if getattr(self, '_channel_names', None):
            self.set_sensitivity(self._sensitivity_uv, rebuild_layout=True)

    def _row_mm(self, n_ch):
        try:
            dpi_y = float(self._pw.physicalDpiY())
        except Exception:
            dpi_y = 96.0
        if not dpi_y or dpi_y <= 0:
            dpi_y = 96.0
        px_per_mm = dpi_y / 25.4
        h_px = max(1, int(self._pw.height()))
        return max(1.0, (h_px / float(max(1, n_ch))) / px_per_mm)

    def _compute_spacing(self, n_ch):
        return float(self._sensitivity_uv) * self._row_mm(n_ch)

    def _clip_level(self):
        return 0.5 * float(self._spacing)

    def plot_item(self):
        return self._pw.getPlotItem()

    def link_x_to(self, other_plot_item):
        other_plot_item.setXLink(self._pw.getPlotItem())

    # ---- reference annotations --------------------------------------
    def set_references(self, ref_intervals):
        pi = self._pw.getPlotItem()
        for r in self._ref_items:
            pi.removeItem(r)
        self._ref_items = []
        self._ref_intervals = [(float(s), float(e)) for s, e in ref_intervals]
        for s, e in self._ref_intervals:
            r = pg.LinearRegionItem(
                values=(s, e),
                brush=pg.mkBrush(50, 160, 70, 40),
                pen=pg.mkPen(50, 160, 70, 120, width=1),
                movable=False,
            )
            r.setZValue(-20)
            r.setVisible(self._refs_visible)
            pi.addItem(r)
            self._ref_items.append(r)

    def set_references_visible(self, visible):
        self._refs_visible = bool(visible)
        for r in self._ref_items:
            r.setVisible(self._refs_visible)

    # ---- annotations the recording itself carries --------------------
    def channel_names(self):
        """The derivations currently displayed, in row order."""
        return list(getattr(self, '_channel_names', []) or [])

    def cursor_time_and_row(self):
        """(seconds, channel index) under the pointer, or None.

        Marking an interictal discharge is a hover-and-press gesture: the
        reviewer sees a spike on one derivation and marks it there. That makes
        the pointer position the input, and a mark without a channel would be
        useless — where the spikes are is most of the clinical information.

        Read from the live cursor rather than from tracked mouse-move state, so
        it cannot go stale when the view scrolls under a stationary pointer.
        """
        pi = self._pw.getPlotItem()
        vb = pi.vb if pi is not None else None
        if vb is None or not getattr(self, '_channel_names', None):
            return None
        scene_pos = self._pw.mapToScene(
            self._pw.mapFromGlobal(QtGui.QCursor.pos()))
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        view_pos = vb.mapSceneToView(scene_pos)
        row = int(round(-view_pos.y() / self._spacing))
        if not (0 <= row < len(self._channel_names)):
            return None
        return float(view_pos.x()), row

    def set_ied_marks(self, marks):
        """Draw reviewer-marked interictal discharges on their own rows.

        A short tick on the derivation it was seen on, not a band across every
        channel: an interictal discharge is a per-channel observation, unlike
        a seizure, which is why it also exports to a different file.
        """
        self._ied_all_marks = marks
        pi = self._pw.getPlotItem()
        for it in getattr(self, '_ied_items', []):
            pi.removeItem(it)
        self._ied_items = []
        names = {n: i for i, n in enumerate(
            getattr(self, '_channel_names', []) or [])}
        half = 0.35 * self._spacing

        # Grouped by row and drawn as ONE item per row, joined by NaN, the way
        # overview_strip already draws its marks. One PlotDataItem per mark was
        # fine for the handful a reviewer places by hand; a detector pass over
        # a day-long recording produces hundreds, and pyqtgraph paints every
        # item on every frame whether or not it is on screen, so the old shape
        # taxed every zoom and pan by the size of the whole recording's mark
        # set. Measured at 29 ms per zoom with 400 marks.
        if not self._marks_are_legible():
            # Too wide to draw them honestly. Reported as hidden, not as
            # absent, so the status tally still tells the reviewer they exist.
            self._ied_hidden = len(marks or [])
            self._ied_row_items = {}
            return 0

        per_row = {}
        proposed = []
        hidden = 0
        for m in marks or []:
            try:
                t = float(m['time'])
            except (KeyError, TypeError, ValueError):
                continue
            channel = m.get('channel')
            if not channel:
                # The detector scores the whole montage and localises nothing,
                # so its marks have no derivation to sit on. They go in the
                # lane above the traces rather than being dropped — which is
                # what the per-row code did to them, silently.
                proposed.append(t)
                continue
            # A montage change renames every derivation: 'Fp1-F7' has no
            # counterpart in a referential view. Such a mark cannot be drawn,
            # and the count is reported rather than swallowed so a reviewer
            # never thinks their marks vanished.
            row = names.get(channel)
            if row is None:
                hidden += 1
                continue
            per_row.setdefault(row, []).append(t)

        drawn = 0
        self._ied_row_items = {}
        for row, times in per_row.items():
            y = -row * self._spacing
            self._ied_row_items[row] = self._add_tick_item(
                pi, times, y - half, y + half, (200, 120, 0), 2)
            drawn += len(times)
        if proposed:
            # Thinner and paler than a reviewer's own mark, in the head lane:
            # these are proposals, and must not read as a clinician's
            # assertion. See gui/io/ied_detect.py on why they are unvalidated.
            top = 0.30 * self._spacing
            self._add_tick_item(pi, proposed, top, top + 0.25 * self._spacing,
                                (230, 170, 60), 1)
            drawn += len(proposed)
        self._ied_hidden = hidden
        return drawn

    def add_ied_mark(self, mark):
        """Draw ONE new mark without rebuilding the rest.

        Pressing D rebuilt every mark on the recording, so the cost of placing
        a mark grew with how many you had already placed — the reviewer who
        had done the most work waited the longest, which is exactly backwards.
        With a detector's output loaded as well it is hundreds of items rebuilt
        for one keypress.

        Returns True if it was drawn. False means the mark's derivation is not
        in the current montage, and the caller should fall back to a full
        rebuild so the hidden count stays right.
        """
        try:
            t = float(mark['time'])
        except (KeyError, TypeError, ValueError):
            return False
        channel = mark.get('channel')
        if not channel:
            return False          # proposals arrive in batches, not one by one
        names = {n: i for i, n in enumerate(
            getattr(self, '_channel_names', []) or [])}
        row = names.get(channel)
        if row is None:
            return False
        item = getattr(self, '_ied_row_items', {}).get(row)
        if item is None:
            return False
        xs, ys = item.getData()
        half = 0.35 * self._spacing
        y = -row * self._spacing
        item.setData(
            np.concatenate([xs, [t, t, np.nan]]),
            np.concatenate([ys, [y - half, y + half, np.nan]]),
            connect='finite')
        return True

    def _on_range_marks(self, *_):
        """Redraw the marks when the view crosses the legibility threshold."""
        legible = self._marks_are_legible()
        if legible == getattr(self, '_marks_were_legible', None):
            return
        self._marks_were_legible = legible
        if getattr(self, '_ied_all_marks', None) is not None:
            self.set_ied_marks(self._ied_all_marks)

    def _marks_are_legible(self):
        """Whether the current view is narrow enough to draw marks at all.

        An interictal discharge lasts about a tenth of a second. Across a
        whole-day view one pixel spans the better part of a minute, so a tick
        is not a discharge, it is a smear of several hundred of them -- it
        shows density while implying morphology, which is the one thing the
        reader must not take from it. Above this span the rate curve under the
        score strip carries the same information honestly, so the marks step
        aside rather than mislead.
        """
        try:
            x0, x1 = self._pw.getPlotItem().viewRange()[0]
        except Exception:                                     # noqa: BLE001
            return True
        return (x1 - x0) <= MARKS_LEGIBLE_MAX_SPAN_S

    def _add_tick_item(self, pi, times, y0, y1, colour, width):
        """One PlotDataItem carrying every tick in a row, NaN-separated."""
        xs = np.repeat(np.asarray(times, dtype=np.float64), 3)
        ys = np.tile(np.array([y0, y1, np.nan]), len(times))
        it = pg.PlotDataItem(xs, ys, connect='finite',
                             pen=pg.mkPen(colour, width=width))
        it.setZValue(30)
        pi.addItem(it)
        self._ied_items.append(it)
        return it

    def set_start_time(self, start):
        """When the recording began, so the x axis can show clock time.

        `start` is a datetime from the EDF header, or None when the header
        carried no usable date — in which case the axis keeps counting from
        the beginning of the file, formatted as h:mm:ss rather than raw
        seconds.
        """
        self._t_axis.set_start(start)
        self._pw.getPlotItem().setLabel('bottom', axis_label(start))

    def set_source_annotations(self, items):
        """Markers the technician wrote during the study (EDF+ annotations).

        Drawn in violet, deliberately unlike anything else on this widget:
        green is the seizure ground truth and the five event colours belong to
        the detector's proposals and the reviewer's decisions. A note saying
        "hoest" (cough) must not be mistakable for either — it is context, not
        a label, and it is never exported.

        Dashed and thin, because these mark instants in a trace the reviewer is
        reading; a solid band would compete with the signal it exists to
        explain. Hyperventilation and photic stimulation are seizure-activation
        procedures, so knowing one was running is often what decides whether a
        run of rhythmic activity is a finding or a provoked response.
        """
        pi = self._pw.getPlotItem()
        for it in getattr(self, '_annot_items', []):
            pi.removeItem(it)
        self._annot_items = []
        self._annot_label_state = {}
        self._source_annotations = list(items or [])
        if not self._source_annotations:
            return
        pen = pg.mkPen((120, 60, 170), width=1, style=QtCore.Qt.DashLine)
        for a in self._source_annotations:
            onset = float(a.get('onset', 0.0))
            text = str(a.get('text', ''))
            if a.get('instant', True):
                item = pg.InfiniteLine(
                    pos=onset, angle=90, pen=pen, movable=False,
                    label=text,
                    labelOpts={'position': ANNOT_LABEL_POSITION,
                               'color': (120, 60, 170), 'movable': False,
                               'fill': (255, 255, 255, 190)})
                item.setZValue(-15)
                pi.addItem(item, ignoreBounds=True)
            else:
                stop = onset + max(0.0, float(a.get('duration', 0.0)))
                item = pg.LinearRegionItem(
                    values=(onset, stop), movable=False,
                    brush=pg.mkBrush(120, 60, 170, 30),
                    pen=pg.mkPen(120, 60, 170, 110, width=1))
                item.setZValue(-18)
                pi.addItem(item)
            self._annot_items.append(item)
        # The lane above the traces only exists when there are annotations, so
        # the y range is recomputed now that we know.
        self._update_y_range()
        self._update_annotation_labels()

    def set_source_annotations_visible(self, visible):
        for it in getattr(self, '_annot_items', []):
            it.setVisible(bool(visible))

    def _update_annotation_labels(self):
        """Show a marker's text only where there is room to read it.

        The lines always draw — they are one pixel wide and each one is a fact
        about the recording. The TEXT is what collides. Measured on the Nihon
        Kohden recording's 41 annotations at a 1500 px width:

            30 s view      3 markers    2 of 2 label pairs overlapping
            300 s view    19 markers   16 of 18 overlapping
            whole record  41 markers   35 of 40 overlapping

        Two of them even share t=0 (`REC START IIB CAL` and `PAT IIB EEG`), so
        this is not only a zoomed-out problem. Past a couple of minutes the
        labels became an unreadable smear laid over the EEG — worse than no
        labels, because it obscures the trace the reviewer is reading in order
        to tell them something they cannot make out anyway.

        So labels are thinned by pixel distance, keeping the earliest of any
        crowded group. Nothing is lost: the line is still there, and zooming in
        brings the text back. Thinning by pixels rather than by seconds is what
        makes it stable across zoom levels and window widths.
        """
        items = getattr(self, '_annot_items', [])
        annots = getattr(self, '_source_annotations', [])
        if not items or len(items) != len(annots):
            return
        pi = self._pw.getPlotItem()
        x0, x1 = pi.viewRange()[0]
        span = max(1e-9, x1 - x0)
        width_px = max(200.0, float(self._pw.viewport().width() or 1500))
        last_px = None
        shown_state = self._annot_label_state
        for i, (item, a) in enumerate(zip(items, annots)):
            label = getattr(item, 'label', None)
            if label is None:                 # a spanning region has no label
                continue
            onset = float(a.get('onset', 0.0))
            if not (x0 <= onset <= x1):
                _set_label_shown(label, False, shown_state, i)
                continue
            px = (onset - x0) / span * width_px
            show = last_px is None or (px - last_px) >= MIN_LABEL_GAP_PX
            _set_label_shown(label, show, shown_state, i)
            if show:
                last_px = px

    # ---- AI-proposed events -----------------------------------------
    def set_selected_event(self, event_id):
        """Only the selected event may be dragged.

        Every region used to be movable, so a left-drag beginning inside one
        moved the event instead of panning the view — promoting it to 'edited',
        which IS exported, with no undo. Restricting mobility to the current
        selection makes an extent change deliberate.
        """
        self._selected_event_id = event_id
        for eid, item in getattr(self, '_regions', {}).items():
            try:
                item.setMovable(eid == event_id)
            except Exception:
                pass

    def set_events(self, events, movable=True):
        pi = self._pw.getPlotItem()
        for item in list(self._regions.values()):
            pi.removeItem(item)
        self._regions = {}
        self._event_cache = list(events)
        for ev in events:
            self._add_region(
                ev,
                movable=movable and (
                    ev.get('id') == getattr(self, '_selected_event_id', None)))

    # Green is reserved for GROUND TRUTH. The reference band is (50,160,70) and
    # accepted events used to be (40,170,90) — the same hue at a different
    # alpha, so a reviewer's own confirmation looked like the answer key. See
    # docs/usability/cognitive_walkthrough_results.md, U-08.
    STATUS_COLOURS = {
        'proposed': (255, 170, 40, 80),     # amber  — awaiting a decision
        'accepted': (35, 110, 200, 95),     # blue   — reviewer confirmed
        'rejected': (180, 180, 180, 40),    # grey   — reviewer dismissed
        'edited':   (150, 80, 200, 90),     # purple — extent adjusted
        'added':    (0, 150, 160, 110),     # teal   — human found it, AI missed
    }

    def _brush_for(self, status):
        return self.STATUS_COLOURS.get(status, self.STATUS_COLOURS['proposed'])

    def _add_region(self, ev, movable=True):
        r = pg.LinearRegionItem(
            values=(ev['start'], ev['stop']),
            brush=pg.mkBrush(self._brush_for(ev['status'])),
            movable=movable,
        )
        r.setZValue(-10)
        r.event_id = ev['id']
        r.sigRegionChangeFinished.connect(
            lambda rr=r, eid=ev['id']: self._on_region_changed(eid, rr))
        self._pw.getPlotItem().addItem(r)
        self._regions[ev['id']] = r

    def _on_region_changed(self, event_id, region):
        s, e = region.getRegion()
        self.regionChanged.emit(event_id, float(s), float(e))

    def update_event_status(self, event_id, new_status):
        r = self._regions.get(event_id)
        if r is not None:
            r.setBrush(pg.mkBrush(self._brush_for(new_status)))

    def update_event_region(self, event_id, start, stop):
        r = self._regions.get(event_id)
        if r is None:
            return
        blocked = False
        try:
            r.blockSignals(True)
            blocked = True
        except Exception:
            pass
        try:
            r.setRegion((float(start), float(stop)))
        finally:
            if blocked:
                try:
                    r.blockSignals(False)
                except Exception:
                    pass

    # ---- viewport controls ------------------------------------------
    def set_timebase(self, seconds_on_screen):
        """Change seconds-on-screen while keeping the centre of the view
        anchored — so bumping timebase doesn't yank the reviewer off the
        event they're focused on."""
        pi = self._pw.getPlotItem()
        x0, x1 = pi.viewRange()[0]
        centre = 0.5 * (x0 + x1)
        half = float(seconds_on_screen) / 2.0
        pi.setXRange(centre - half, centre + half, padding=0)

    def center_on(self, t_sec, half_span=None):
        pi = self._pw.getPlotItem()
        if half_span is None:
            x0, x1 = pi.viewRange()[0]
            half_span = 0.5 * (x1 - x0)
        pi.setXRange(max(0.0, t_sec - half_span), t_sec + half_span, padding=0)

    def shift_view(self, delta_s):
        pi = self._pw.getPlotItem()
        x0, x1 = pi.viewRange()[0]
        span = x1 - x0
        new0 = max(0.0, x0 + delta_s)
        pi.setXRange(new0, new0 + span, padding=0)

    def zoom_x(self, factor):
        pi = self._pw.getPlotItem()
        x0, x1 = pi.viewRange()[0]
        mid = 0.5 * (x0 + x1)
        span = (x1 - x0) * factor
        pi.setXRange(max(0.0, mid - span / 2), mid + span / 2, padding=0)

    def current_span(self):
        x0, x1 = self._pw.getPlotItem().viewRange()[0]
        return float(x0), float(x1)
