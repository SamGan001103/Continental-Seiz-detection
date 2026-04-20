"""Single-channel detail window.

Opened from SignalView via double-click. Non-modal — the user can open
one per channel and keep them on screen alongside the main view.
"""
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

from gui.widgets.signal_view import _SignalViewBox


AMPLITUDES_UV = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500,
                 1000, 2000, 5000, 10000]


def _fit_amplitude(data_slice, headroom=1.2):
    if data_slice.size == 0:
        return AMPLITUDES_UV[-1]
    p98 = float(np.percentile(np.abs(data_slice), 98))
    target = p98 * headroom
    for a in AMPLITUDES_UV:
        if a >= target:
            return a
    return AMPLITUDES_UV[-1]


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
        self.resize(1000, 360)

        self._channel_idx = channel_idx
        self._channel_name = channel_name
        self._fs = fs
        self._data = data1d.astype(np.float32, copy=False)
        self._t = t

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

        ctrl.addStretch(1)
        self._stats_lbl = QtWidgets.QLabel('')
        self._stats_lbl.setStyleSheet('color:#666;')
        ctrl.addWidget(self._stats_lbl)
        v.addLayout(ctrl)

        # ---- plot ------------------------------------------------------
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
        v.addWidget(self._pw)

        if reference_intervals:
            for s, e in reference_intervals:
                r = pg.LinearRegionItem(
                    values=(float(s), float(e)),
                    brush=pg.mkBrush(50, 160, 70, 40),
                    pen=pg.mkPen(50, 160, 70, 140, width=1),
                    movable=False)
                r.setZValue(-10)
                pi.addItem(r)

        self._curve = pi.plot(t, self._data,
                              pen=pg.mkPen((20, 60, 160), width=0.9))

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

    # ------------------------------------------------------------------
    def plot_item(self):
        return self._pw.getPlotItem()

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

    def sync_toggle_signal(self):
        return self.chk_sync.stateChanged

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
        self.closed.emit(self._channel_idx)
        super().closeEvent(ev)
