"""MainWindow for the seizure-review GUI.

Toolbar groups (left to right):
    Open | Montage | Filters (HP / LP / notch) | Sensitivity | Timebase |
    Threshold | Export

Keyboard shortcuts:
    Space     accept selected event
    X         reject selected event
    Enter     jump to selected event and frame it
    J / K     next / prev event
    ← / →     pan half a screen
    + / -     zoom in / out on time axis
    [ / ]     amplitude sensitivity down / up
    , / .     timebase shorter / longer
"""
import os
import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.edf import load_edf_19ch
from gui.io.cache import load_probs, cache_path_for
from gui.io.csv_bi import read_csv_bi, write_csv_bi
from gui.processing import (apply_montage, apply_filters, MONTAGES)
from gui.widgets.signal_view import SignalView
from gui.widgets.prob_strip import ProbStrip
from gui.widgets.event_list import EventList
from gui.widgets.channel_inspector import ChannelInspector

SENSITIVITIES = [7, 10, 15, 20, 30, 50, 70, 100, 150]   # µV / div
TIMEBASES = [5, 10, 15, 20, 30, 60, 120, 300]           # s / screen


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Seizure Review — Continental Human-AI')
        self.resize(1600, 960)

        # ------------------ state
        self._edf_path = None
        self._raw_data = None          # [19, N] float32, original order
        self._fs = 250
        self._duration_s = 0.0
        self._refs = []                # list[(start, stop)]
        self._probs = None             # (window_starts, probs) | None
        self._segment_s = 12
        self._events = []
        self._threshold = 0.5

        self._montage = 'Longitudinal Bipolar'
        self._hp = 1.0
        self._lp = 70.0
        self._notch = 50.0
        self._sensitivity_uv = 30.0
        self._timebase_s = 30

        # cache of the most-recent post-filter, post-montage display data so
        # channel-inspector popups can be opened without recomputing.
        self._display_data = None      # [K, N] float32, before clipping
        self._display_labels = []      # list[str]
        self._display_t = None         # [N] seconds
        self._inspectors = {}          # channel_idx -> ChannelInspector

        self._build_ui()
        self._wire_shortcuts()
        self.statusBar().showMessage('Open an EDF to start.')

    # ==================================================================
    def _build_ui(self):
        self.signal_view = SignalView()
        self.prob_strip = ProbStrip()
        self.event_list = EventList()
        self.signal_view.link_x_to(self.prob_strip.plot_item())

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.signal_view, 5)
        lv.addWidget(self.prob_strip, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.event_list)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1200, 380])
        self.setCentralWidget(splitter)

        self._build_toolbar()

        # hover time readout in status bar
        self._hover_lbl = QtWidgets.QLabel('')
        self.statusBar().addPermanentWidget(self._hover_lbl)
        self.signal_view.viewRangeChanged.connect(self._on_view_range)

        # event list wiring
        self.event_list.accepted.connect(self._on_accept)
        self.event_list.rejected.connect(self._on_reject)
        self.event_list.jumpRequested.connect(self._on_jump)
        self.event_list.selectionChanged.connect(self._on_selection_change)
        self.signal_view.regionChanged.connect(self._on_region_edited)
        self.signal_view.channelDoubleClicked.connect(self._open_channel_inspector)
        self.signal_view.viewRangeChanged.connect(self._sync_inspector_ranges)
        self.signal_view.sensitivityChanged.connect(self._on_axis_drag_sens)

    def _build_toolbar(self):
        tb = self.addToolBar('main')
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(16, 16))
        tb.setStyleSheet(
            'QToolBar { spacing: 6px; padding: 2px 6px; }'
            'QLabel { color: #555; }')

        a_open = tb.addAction('Open EDF…')
        a_open.setShortcut(QtGui.QKeySequence.Open)
        a_open.triggered.connect(self._open_edf_dialog)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel('Montage'))
        self.cb_montage = QtWidgets.QComboBox()
        self.cb_montage.addItems(list(MONTAGES.keys()))
        self.cb_montage.setCurrentText(self._montage)
        self.cb_montage.currentTextChanged.connect(self._on_montage_change)
        tb.addWidget(self.cb_montage)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel('HP'))
        self.cb_hp = QtWidgets.QComboBox()
        self.cb_hp.addItems(['off', '0.5', '1.0', '3.0', '5.0'])
        self.cb_hp.setCurrentText('1.0')
        self.cb_hp.currentTextChanged.connect(self._on_filters_change)
        tb.addWidget(self.cb_hp)

        tb.addWidget(QtWidgets.QLabel('LP'))
        self.cb_lp = QtWidgets.QComboBox()
        self.cb_lp.addItems(['off', '15', '30', '35', '50', '70'])
        self.cb_lp.setCurrentText('70')
        self.cb_lp.currentTextChanged.connect(self._on_filters_change)
        tb.addWidget(self.cb_lp)

        tb.addWidget(QtWidgets.QLabel('Notch'))
        self.cb_notch = QtWidgets.QComboBox()
        self.cb_notch.addItems(['off', '50', '60'])
        self.cb_notch.setCurrentText('50')
        self.cb_notch.currentTextChanged.connect(self._on_filters_change)
        tb.addWidget(self.cb_notch)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel('Sens µV/div'))
        self.cb_sens = QtWidgets.QComboBox()
        self.cb_sens.addItems([str(v) for v in SENSITIVITIES])
        self.cb_sens.setCurrentText(str(int(self._sensitivity_uv)))
        self.cb_sens.currentTextChanged.connect(self._on_sens_change)
        tb.addWidget(self.cb_sens)

        tb.addWidget(QtWidgets.QLabel('Timebase s'))
        self.cb_tb = QtWidgets.QComboBox()
        self.cb_tb.addItems([str(v) for v in TIMEBASES])
        self.cb_tb.setCurrentText(str(self._timebase_s))
        self.cb_tb.currentTextChanged.connect(self._on_timebase_change)
        tb.addWidget(self.cb_tb)
        tb.addSeparator()

        self.chk_refs = QtWidgets.QCheckBox('Show reference')
        self.chk_refs.setChecked(True)
        self.chk_refs.setToolTip('Show the green ground-truth seizure bands '
                                 'from the .csv_bi file')
        self.chk_refs.toggled.connect(self._on_refs_toggled)
        tb.addWidget(self.chk_refs)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel('Threshold'))
        self.thr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.thr_slider.setRange(0, 100)
        self.thr_slider.setValue(int(self._threshold * 100))
        self.thr_slider.setFixedWidth(160)
        self.thr_slider.valueChanged.connect(self._on_thr_changed)
        tb.addWidget(self.thr_slider)
        self.thr_lbl = QtWidgets.QLabel(' {:.2f} '.format(self._threshold))
        tb.addWidget(self.thr_lbl)
        tb.addSeparator()

        a_export = tb.addAction('Export reviewed…')
        a_export.triggered.connect(self._export_reviewed)

    def _wire_shortcuts(self):
        def add(seq, fn):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(fn)

        add('Space', self._shortcut_accept)
        add('X', self._shortcut_reject)
        add('Return', self._shortcut_jump)
        add('Enter', self._shortcut_jump)
        add('J', lambda: self._cycle_selection(+1))
        add('K', lambda: self._cycle_selection(-1))
        add('Right', lambda: self.signal_view.shift_view(self._timebase_s * 0.5))
        add('Left', lambda: self.signal_view.shift_view(-self._timebase_s * 0.5))
        add('+', lambda: self.signal_view.zoom_x(0.75))
        add('=', lambda: self.signal_view.zoom_x(0.75))
        add('-', lambda: self.signal_view.zoom_x(1.33))
        add(']', lambda: self._bump_sens(+1))
        add('[', lambda: self._bump_sens(-1))
        add('.', lambda: self._bump_timebase(+1))
        add(',', lambda: self._bump_timebase(-1))

    # ==================================================================
    # File loading
    # ==================================================================
    def _open_edf_dialog(self):
        start_dir = os.path.join(REPO, 'sample_data')
        if not os.path.isdir(start_dir):
            start_dir = REPO
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open EDF', start_dir, 'EDF files (*.edf)')
        if path:
            self.load_edf(path)

    def load_edf(self, path):
        self.statusBar().showMessage('Loading {}…'.format(os.path.basename(path)))
        QtWidgets.QApplication.processEvents()
        data, fs, dur = load_edf_19ch(path)
        self._edf_path = path
        self._raw_data = data
        self._fs = fs
        self._duration_s = dur
        self.setWindowTitle('Seizure Review — {}'.format(os.path.basename(path)))

        # references (ground truth) — optional
        ref_path = os.path.splitext(path)[0] + '.csv_bi'
        self._refs = [(e['start'], e['stop'])
                      for e in read_csv_bi(ref_path) if e['label'] == 'seiz']
        self.signal_view.set_references(self._refs)
        self.prob_strip.set_references(self._refs)

        # probability cache
        cached = load_probs(path)
        if cached is None:
            self._probs = None
            self.prob_strip.set_probs([], [])
            msg = 'No prob cache — run: python precompute_probs.py "{}"'.format(path)
        else:
            self._probs = (cached['window_starts'], cached['probs'])
            self._segment_s = cached['meta'].get('segment_s', 12)
            self.prob_strip.set_probs(cached['window_starts'],
                                      cached['probs'],
                                      segment_s=self._segment_s)
            msg = '{} — {:.1f}s, {} prob windows, {} reference seizures'.format(
                os.path.basename(path), dur, len(cached['probs']), len(self._refs))

        # Fresh file: do reset the viewport to the start.
        self._refresh_signal_view(preserve_view=False)
        self._rebuild_events_from_probs()
        self.statusBar().showMessage(msg)

    # ==================================================================
    # Display pipeline
    # ==================================================================
    def _refresh_signal_view(self, preserve_view=True):
        """Rebuild filtered + montaged signal and push into SignalView.

        preserve_view : default True so changes to montage / filter /
            sensitivity don't scroll the reviewer back to t=0. Callers
            that open a fresh file should pass False."""
        if self._raw_data is None:
            return
        filtered = apply_filters(
            self._raw_data, self._fs,
            hp=self._hp, lp=self._lp, notch=self._notch)
        montaged, labels = apply_montage(filtered, self._montage)
        # Cache unclipped version for channel-inspector popups.
        self._display_data = montaged.copy()
        self._display_labels = list(labels)
        self._display_t = np.arange(montaged.shape[1], dtype=np.float32) / self._fs
        # Hand the unclipped, post-montage signal to the view. SignalView
        # owns clipping + decimation, so sensitivity can change (via
        # toolbar or axis-drag) without re-running filter/montage.
        self.signal_view.set_signal(
            montaged.astype(np.float32), self._fs, labels,
            sensitivity_uv=float(self._sensitivity_uv),
            initial_span_s=float(self._timebase_s),
            preserve_view=preserve_view,
        )
        self.signal_view.set_references(self._refs)
        self.signal_view.set_events(self._events)
        self._refresh_open_inspectors()

    # ==================================================================
    # Events from probs
    # ==================================================================
    def _rebuild_events_from_probs(self):
        self._events = []
        self.prob_strip.set_threshold(self._threshold)
        if self._probs is None:
            self.signal_view.set_events([])
            self.event_list.set_events([])
            return
        starts, probs = self._probs
        in_ev, s0, peak = False, 0, 0.0
        eid = 1
        for t, p in zip(starts, probs):
            if not in_ev and p >= self._threshold:
                in_ev, s0, peak = True, int(t), float(p)
            elif in_ev and p >= self._threshold:
                peak = max(peak, float(p))
            elif in_ev and p < self._threshold:
                self._events.append({
                    'id': eid, 'start': float(s0), 'stop': float(t),
                    'prob': peak, 'status': 'proposed',
                })
                eid += 1
                in_ev = False
        if in_ev:
            self._events.append({
                'id': eid, 'start': float(s0),
                'stop': float(starts[-1] + self._segment_s),
                'prob': peak, 'status': 'proposed',
            })
        self.signal_view.set_events(self._events)
        self.event_list.set_events(self._events)

    # ==================================================================
    # Reviewer actions
    # ==================================================================
    def _find(self, event_id):
        for ev in self._events:
            if ev['id'] == event_id:
                return ev
        return None

    def _on_accept(self, event_id):
        ev = self._find(event_id)
        if ev:
            ev['status'] = 'accepted'
            self.signal_view.update_event_status(event_id, 'accepted')
            self.event_list.update_row(event_id, ev)
            self.event_list._refresh_summary(self._events)

    def _on_reject(self, event_id):
        ev = self._find(event_id)
        if ev:
            ev['status'] = 'rejected'
            self.signal_view.update_event_status(event_id, 'rejected')
            self.event_list.update_row(event_id, ev)
            self.event_list._refresh_summary(self._events)

    def _on_jump(self, event_id):
        ev = self._find(event_id)
        if ev:
            centre = 0.5 * (ev['start'] + ev['stop'])
            # frame event with ±10s context at current timebase
            self.signal_view.center_on(centre, half_span=max(self._timebase_s / 2,
                                                             (ev['stop'] - ev['start']) / 2 + 10))

    def _on_region_edited(self, event_id, s, e):
        ev = self._find(event_id)
        if ev:
            ev['start'] = float(s)
            ev['stop'] = float(e)
            if ev['status'] == 'proposed':
                ev['status'] = 'edited'
                self.signal_view.update_event_status(event_id, 'edited')
            self.event_list.update_row(event_id, ev)
            self.event_list._refresh_summary(self._events)

    def _on_selection_change(self, event_id):
        # no-op for now; shortcuts act on the current selection
        pass

    # ==================================================================
    # Toolbar handlers
    # ==================================================================
    def _on_montage_change(self, name):
        self._montage = name
        self._refresh_signal_view()

    def _on_filters_change(self, _=None):
        def v(cb):
            t = cb.currentText()
            return None if t == 'off' else float(t)
        self._hp = v(self.cb_hp)
        self._lp = v(self.cb_lp)
        self._notch = v(self.cb_notch)
        self._refresh_signal_view()

    def _on_sens_change(self, text):
        try:
            new_sens = float(text)
        except ValueError:
            return
        self._sensitivity_uv = new_sens
        # SignalView owns clipping; just update it. No filter/montage rerun.
        self.signal_view.set_sensitivity(new_sens, rebuild_layout=True)

    def _on_axis_drag_sens(self, new_sens, finished):
        """Mirror the live y-axis drag into toolbar state without triggering
        a full pipeline refresh (SignalView already updated itself)."""
        self._sensitivity_uv = float(new_sens)
        if finished:
            # Snap the combo to the closest preset so the toolbar reads
            # a clean value after the drag.
            from gui.app import SENSITIVITIES as _S  # local ref
            nearest = min(_S, key=lambda v: abs(v - new_sens))
            self.cb_sens.blockSignals(True)
            self.cb_sens.setCurrentText(str(nearest))
            self.cb_sens.blockSignals(False)
            self._sensitivity_uv = float(nearest)
            self.signal_view.set_sensitivity(float(nearest), rebuild_layout=True)

    def _on_timebase_change(self, text):
        try:
            self._timebase_s = int(float(text))
        except ValueError:
            return
        self.signal_view.set_timebase(self._timebase_s)

    def _on_thr_changed(self, v):
        self._threshold = v / 100.0
        self.thr_lbl.setText(' {:.2f} '.format(self._threshold))
        self._rebuild_events_from_probs()

    def _on_refs_toggled(self, checked):
        self.signal_view.set_references_visible(checked)
        self.prob_strip.set_references_visible(checked)

    def _on_view_range(self, x0, x1):
        self._hover_lbl.setText('View: {:.1f}s – {:.1f}s  ({:.1f}s)'
                                .format(x0, x1, x1 - x0))

    # ==================================================================
    # Channel inspector popups
    # ==================================================================
    def _open_channel_inspector(self, idx):
        if self._display_data is None or idx >= len(self._display_labels):
            return
        # Reuse existing inspector if already open for this channel
        if idx in self._inspectors:
            w = self._inspectors[idx]
            w.raise_(); w.activateWindow()
            return
        label = self._display_labels[idx]
        data1d = self._display_data[idx]
        x0, x1 = self.signal_view.current_span()
        insp = ChannelInspector(
            channel_idx=idx,
            channel_name=label,
            fs=self._fs,
            data1d=data1d,
            t=self._display_t,
            reference_intervals=self._refs,
            initial_x_range=(x0, x1),
            parent=self,
        )
        insp.closed.connect(self._on_inspector_closed)
        insp.sync_toggle_signal().connect(lambda _s, i=idx: self._apply_inspector_sync(i))
        self._inspectors[idx] = insp
        insp.show()

    def _on_inspector_closed(self, idx):
        self._inspectors.pop(idx, None)

    def _apply_inspector_sync(self, idx):
        insp = self._inspectors.get(idx)
        if insp is None:
            return
        if insp.sync_checked():
            x0, x1 = self.signal_view.current_span()
            insp.set_x_range(x0, x1)

    def _sync_inspector_ranges(self, x0, x1):
        for insp in self._inspectors.values():
            if insp.sync_checked():
                insp.set_x_range(x0, x1)

    def _refresh_open_inspectors(self):
        """Feed new data into any currently-open inspectors after a
        montage / filter change."""
        if self._display_data is None:
            return
        for idx, insp in list(self._inspectors.items()):
            if idx >= len(self._display_labels):
                insp.close()
                continue
            # Simplest: close + reopen so the label/data are correct.
            x0, x1 = self.signal_view.current_span()
            insp.close()
        # Re-open closed ones? Keep it simple: user reopens manually if needed.

    # ==================================================================
    # Keyboard shortcut plumbing
    # ==================================================================
    def _shortcut_accept(self):
        eid = self.event_list.selected_event_id()
        if eid > 0:
            self._on_accept(eid)

    def _shortcut_reject(self):
        eid = self.event_list.selected_event_id()
        if eid > 0:
            self._on_reject(eid)

    def _shortcut_jump(self):
        eid = self.event_list.selected_event_id()
        if eid > 0:
            self._on_jump(eid)

    def _cycle_selection(self, delta):
        eid = self.event_list.cycle_selection(delta)
        if eid > 0:
            self._on_jump(eid)

    def _bump_sens(self, delta):
        try:
            i = SENSITIVITIES.index(int(self._sensitivity_uv))
        except ValueError:
            i = SENSITIVITIES.index(100)
        i = max(0, min(len(SENSITIVITIES) - 1, i + delta))
        self.cb_sens.setCurrentText(str(SENSITIVITIES[i]))

    def _bump_timebase(self, delta):
        try:
            i = TIMEBASES.index(int(self._timebase_s))
        except ValueError:
            i = TIMEBASES.index(30)
        i = max(0, min(len(TIMEBASES) - 1, i + delta))
        self.cb_tb.setCurrentText(str(TIMEBASES[i]))

    # ==================================================================
    # Export
    # ==================================================================
    def _export_reviewed(self):
        if not self._edf_path:
            return
        default = os.path.splitext(self._edf_path)[0] + '.reviewed.csv_bi'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export reviewed annotations', default,
            'TUSZ csv_bi (*.csv_bi)')
        if not path:
            return
        events = [
            {'start': ev['start'], 'stop': ev['stop'],
             'label': 'seiz', 'confidence': ev['prob']}
            for ev in self._events if ev['status'] in ('accepted', 'edited')
        ]
        write_csv_bi(path,
                     bname=os.path.splitext(os.path.basename(self._edf_path))[0],
                     duration_s=self._duration_s,
                     events=events)
        self.statusBar().showMessage(
            'Exported {} reviewed events → {}'.format(
                len(events), os.path.basename(path)), 5000)
