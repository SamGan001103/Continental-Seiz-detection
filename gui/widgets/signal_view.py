"""Multi-channel EEG trace viewer.

One pyqtgraph PlotWidget holds all traces at vertical offsets. The
y-axis is a custom `_ChannelAxis` that labels each row and lets the
user drag it to rescale sensitivity (like commercial EEG viewers).
The viewbox uses a gentler wheel-zoom factor for smooth scrolling."""
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

from gui.processing import decimate_for_display

pg.setConfigOptions(antialias=False, useOpenGL=False, background='w',
                    foreground='k')


# ---------------------------------------------------------------------- helpers

def fmt_time(t_sec):
    """Format a time in seconds as [h:]mm:ss.sss for cursor readouts."""
    if t_sec < 0:
        t_sec = 0.0
    h = int(t_sec // 3600)
    rem = t_sec - h * 3600
    m = int(rem // 60)
    s = rem - m * 60
    if h:
        return '{:d}:{:02d}:{:06.3f}'.format(h, m, s)
    return '{:02d}:{:06.3f}'.format(m, s)


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
    lbl = QtWidgets.QLabel('t = --:--.---')
    lbl.setStyleSheet(_TIME_LBL_STYLE)
    lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    return lbl


def _update_time_label(lbl, t):
    if t is None:
        lbl.setText('t = --:--.---')
    else:
        lbl.setText('t = {}'.format(fmt_time(t)))


# ---------------------------------------------------------------------- axis

class _ChannelAxis(pg.AxisItem):
    def __init__(self, ch_names, spacing=200.0, orientation='left'):
        super().__init__(orientation=orientation)
        self._ch_names = list(ch_names)
        self._spacing = spacing
        self.setWidth(64)
        self._drag_cb = None         # callable(new_sensitivity_uv, finished=False)
        self._drag_start = None
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
            # sensitivity = spacing / 3 (set by MainWindow). Remember start state.
            self._drag_start = (float(ev.buttonDownPos().y()), self._spacing / 3.0)
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
    roughly half of pyqtgraph's default (~12%), so scrolling feels
    smooth without slogging through increments. Reused by the channel
    inspector so wheel behaviour is consistent across windows."""

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
        ev.accept()


# ---------------------------------------------------------------------- SignalView

class SignalView(QtWidgets.QWidget):
    regionChanged = QtCore.pyqtSignal(int, float, float)
    viewRangeChanged = QtCore.pyqtSignal(float, float)
    channelDoubleClicked = QtCore.pyqtSignal(int)       # channel index
    sensitivityChanged = QtCore.pyqtSignal(float, bool) # (new_µV/div, finished)

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        self._sensitivity_uv = 30.0
        self._spacing = self._sensitivity_uv * 3.0
        self._ch_axis = _ChannelAxis([], spacing=self._spacing)
        self._ch_axis.set_drag_callback(self._on_axis_drag)

        vb = _SignalViewBox()
        self._pw = pg.PlotWidget(viewBox=vb,
                                 axisItems={'left': self._ch_axis})
        pi = self._pw.getPlotItem()
        pi.setMouseEnabled(x=True, y=False)
        pi.setLabel('bottom', 'time (s)')
        pi.showGrid(x=True, y=False, alpha=0.2)
        pi.sigRangeChanged.connect(self._emit_range)
        self._pw.scene().sigMouseClicked.connect(self._on_scene_click)
        v.addWidget(self._pw)

        self._time_lbl = _make_time_label()
        v.addWidget(self._time_lbl)

        self._time_cursor = TimeCursor(
            self._pw,
            on_hover=lambda t: _update_time_label(self._time_lbl, t))

        self._curves = []
        self._ref_items = []
        self._regions = {}
        self._grid_lines = []
        self._channel_names = []
        self._fs = 250
        self._full_unclipped = None    # [K, N] float32 (unclipped, post-montage)
        self._full_offsets = None
        self._ref_intervals = []       # cached (start, stop) tuples for refresh
        self._event_cache = []         # cached event dicts for refresh
        self._refs_visible = True      # toolbar toggle state

        self._redec_timer = QtCore.QTimer(self)
        self._redec_timer.setSingleShot(True)
        self._redec_timer.setInterval(30)
        self._redec_timer.timeout.connect(self._redecimate_visible)
        pi.sigRangeChanged.connect(lambda *_: self._redec_timer.start())

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
        """Cache full-resolution unclipped data. SignalView owns clipping
        so sensitivity changes are fast (no filter/montage re-run).

        preserve_view : if True and the viewport was already populated,
            keep the current x-range. Used when the caller is refreshing
            the same file with a new montage / filter / sensitivity, so
            the reviewer doesn't get yanked back to t=0.
        """
        pi = self._pw.getPlotItem()
        # Capture current x-range before tearing down so we can restore it.
        had_signal = self._full_unclipped is not None
        keep_x = preserve_view and had_signal
        if keep_x:
            prev_x0, prev_x1 = pi.viewRange()[0]

        pi.clear()
        self._curves = []
        self._regions = {}
        self._ref_items = []
        self._grid_lines = []
        # pi.clear() nuked the hover-cursor overlay; put it back before
        # the user's next mouse move needs it.
        self._time_cursor.reattach()
        self._channel_names = list(channel_names)
        self._fs = fs
        self._full_unclipped = np.ascontiguousarray(data_full, dtype=np.float32)
        self._sensitivity_uv = float(sensitivity_uv)
        self._spacing = self._sensitivity_uv * 3.0
        n_ch = data_full.shape[0]
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
        duration_s = data_full.shape[1] / fs
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
        self._spacing = self._sensitivity_uv * 3.0
        if self._full_unclipped is None:
            return
        n_ch = self._full_unclipped.shape[0]
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
        pi.setYRange(-(n_ch - 0.5) * self._spacing,
                     0.5 * self._spacing, padding=0)

    def _redecimate_visible(self):
        if self._full_unclipped is None or not self._curves:
            return
        pi = self._pw.getPlotItem()
        x0, x1 = pi.viewRange()[0]
        px_w = max(200, int(self._pw.viewport().width() or 1600))
        n = self._full_unclipped.shape[1]
        span = max(1e-6, x1 - x0)
        pad = span
        s0 = max(0, int((x0 - pad) * self._fs))
        s1 = min(n, int((x1 + pad) * self._fs))
        if s1 <= s0:
            return
        slice_ = self._full_unclipped[:, s0:s1]
        # Clip to current sensitivity BEFORE decimation so spikes that
        # exceed the row are rendered flat at ±sens rather than wiping
        # across the neighboring channel.
        clipped = np.clip(slice_, -self._sensitivity_uv, self._sensitivity_uv)
        target = int(px_w * (s1 - s0) / (span * self._fs)) if span > 0 else px_w
        target = max(512, min(target, 20000))
        data_ds, t_local = decimate_for_display(clipped, self._fs, target_px=target)
        t_ds = t_local + (s0 / self._fs)
        for i, curve in enumerate(self._curves):
            curve.setData(t_ds, data_ds[i] + self._full_offsets[i])

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

    # ---- AI-proposed events -----------------------------------------
    def set_events(self, events, movable=True):
        pi = self._pw.getPlotItem()
        for item in list(self._regions.values()):
            pi.removeItem(item)
        self._regions = {}
        self._event_cache = list(events)
        for ev in events:
            self._add_region(ev, movable=movable)

    # Green is reserved for GROUND TRUTH. The reference band is (50,160,70) and
    # accepted events used to be (40,170,90) — the same hue at a different
    # alpha, so a reviewer's own confirmation looked like the answer key. See
    # docs/usability/cognitive_walkthrough_results.md, U-08.
    STATUS_COLOURS = {
        'proposed': (255, 170, 40, 80),     # amber  — awaiting a decision
        'accepted': (35, 110, 200, 95),     # blue   — reviewer confirmed
        'rejected': (180, 180, 180, 40),    # grey   — reviewer dismissed
        'edited':   (150, 80, 200, 90),     # purple — extent adjusted
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
