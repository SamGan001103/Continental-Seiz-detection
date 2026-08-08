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
import json
import time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.edf import load_edf_19ch
from gui.io.cache import (load_probs, load_probability_file,
                          save_probability_file)
from gui.io import csv_bi as csv_bi_module
from gui.io.csv_bi import read_csv_bi, write_csv_bi
from gui.io.zuna import ZunaFullRunner, zuna_session_paths
from gui.events import clamp_interval, rebuild_events
from gui.processing import (apply_montage, apply_filters, MONTAGES)
from gui.widgets.signal_view import SignalView
from gui.widgets.prob_strip import ProbStrip
from gui.widgets.event_list import EventList
from gui.widgets.channel_inspector import ChannelInspector

SENSITIVITIES = [7, 10, 15, 20, 30, 50, 70, 100, 150]   # µV / div
TIMEBASES = [5, 10, 15, 20, 30, 60, 120, 300]           # s / screen

APP_NAME = 'Seizure Review'
# Shown permanently in the title bar and the status bar, and written into every
# export. The detector runs at roughly 48% event sensitivity, so anyone sitting
# in front of this — supervisor, peer reviewer, clinician — must be able to see
# what it is without being told.
PROTOTYPE_BANNER = 'RESEARCH PROTOTYPE — NOT FOR DIAGNOSTIC USE'
TITLE_SUFFIX = ' — RESEARCH PROTOTYPE, NOT FOR DIAGNOSTIC USE'


def _weights_sha256():
    """Hash of the weights file actually on disk, for export provenance.

    Compared against eval_config.WEIGHTS_SHA256 so a reviewed annotation can be
    tied to the exact model that produced it, and so a swapped or corrupted
    weights file is detectable after the fact rather than silently accepted.
    """
    try:
        import eval_config as cfg
        from gui.io.cache import sha256_file
        return sha256_file(os.path.join(REPO, cfg.WEIGHTS))
    except Exception:
        return None


class ZunaRunThread(QtCore.QThread):
    statusChanged = QtCore.pyqtSignal(str)
    finishedOk = QtCore.pyqtSignal(str, dict)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, edf_path, parent=None):
        super().__init__(parent)
        self.edf_path = os.path.abspath(edf_path)
        self._runner = ZunaFullRunner(self.edf_path)

    def cancel(self):
        self._runner.cancel()

    def run(self):
        try:
            paths = self._runner.run(status_cb=self.statusChanged.emit)
        except Exception as ex:
            self.failed.emit(self.edf_path, str(ex))
            return
        self.finishedOk.emit(self.edf_path, paths)

    def settings(self):
        return {
            'diffusion_steps': self._runner.diffusion_steps,
            'tokens_per_batch': self._runner.tokens_per_batch,
            'gpu_device': self._runner.gpu_device,
            'zuna_python': self._runner.zuna_python,
            'paths': self._runner.paths,
        }


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME + TITLE_SUFFIX)
        self.resize(1600, 960)

        # ------------------ state
        self._edf_path = None
        self._original_data = None     # [19, N] float32, original EDF order
        self._zuna_data = None         # [19, N] float32, reconstructed signal
        self._raw_data = None          # active display signal source
        self._fs = 250
        self._duration_s = 0.0
        self._refs = []                # list[(start, stop)]
        self._probs = None             # (window_starts, probs) | None
        self._prob_sources = {}         # source key -> cache dict
        self._events_by_source = {}     # source key -> reviewed event list
        self._prob_source = 'baseline'
        self._zuna_paths = None
        self._edf_sha256 = None
        self._zuna_thread = None
        self._zuna_progress = None
        self._zuna_progress_timer = QtCore.QTimer(self)
        self._zuna_progress_timer.setInterval(1000)
        self._zuna_progress_timer.timeout.connect(self._update_zuna_progress)
        self._zuna_progress_started = None
        self._zuna_stage_started = None
        self._zuna_stage = 'Idle'
        self._zuna_run_settings = {}
        self._segment_s = 12
        self._nothing_assessed_warned = set()
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

        # Permanent prototype banner, left of the hover readout so it is never
        # scrolled away or overwritten by a transient showMessage().
        self._proto_lbl = QtWidgets.QLabel(PROTOTYPE_BANNER)
        self._proto_lbl.setStyleSheet(
            'QLabel { color: #fff; background: #a11; padding: 1px 8px; '
            'font-weight: bold; border-radius: 3px; }')
        self._proto_lbl.setToolTip(
            'This is a research prototype built on public TUH/TUSZ data. It is '
            'not a medical device, has not been clinically validated, and must '
            'not be used for diagnosis or patient care.')
        self.statusBar().addPermanentWidget(self._proto_lbl)

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

        tb.addWidget(QtWidgets.QLabel('AI source'))
        self.cb_source = QtWidgets.QComboBox()
        self.cb_source.addItem('Baseline', 'baseline')
        self.cb_source.setToolTip(
            'Switch between original-EDF baseline probabilities and full '
            'ZUNA probabilities after ZUNA has finished.')
        self.cb_source.currentIndexChanged.connect(self._on_source_change)
        tb.addWidget(self.cb_source)

        self.a_run_zuna = tb.addAction('Run full ZUNA')
        self.a_run_zuna.setToolTip(
            'Run full-file ZUNA reconstruction for this EDF. This is optional '
            'and computationally heavy.')
        self.a_run_zuna.setEnabled(False)
        self.a_run_zuna.triggered.connect(self._run_full_zuna_for_session)
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

    def closeEvent(self, event):
        if self._zuna_thread is not None and self._zuna_thread.isRunning():
            answer = QtWidgets.QMessageBox.question(
                self, 'Cancel full ZUNA?',
                'Full ZUNA is still running. Cancel it and close the GUI?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes)
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self._zuna_thread.cancel()
            self._zuna_thread.wait(5000)
        super().closeEvent(event)

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
        self._cancel_zuna_for_file_change()
        self.statusBar().showMessage('Loading {}…'.format(os.path.basename(path)))
        QtWidgets.QApplication.processEvents()
        try:
            data, fs, dur = load_edf_19ch(path)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(
                self, 'Could not open EDF',
                'Could not load this EDF:\n\n{}'.format(ex))
            self.statusBar().showMessage('EDF load failed.', 5000)
            return
        self._edf_path = path
        try:
            from gui.io.cache import sha256_file
            self._edf_sha256 = sha256_file(path)
        except Exception:
            self._edf_sha256 = None
        self._original_data = data
        self._zuna_data = None
        self._raw_data = data
        self._fs = fs
        self._duration_s = dur
        self._events = []
        self._events_by_source = {}
        self._prob_sources = {}
        self._prob_source = 'baseline'
        self._zuna_paths = zuna_session_paths(path)
        self._reset_source_combo()
        self.a_run_zuna.setEnabled(True)
        self.setWindowTitle('{} — {}{}'.format(
            APP_NAME, os.path.basename(path), TITLE_SUFFIX))

        # references (ground truth) — optional
        ref_path = os.path.splitext(path)[0] + '.csv_bi'
        self._refs = [(e['start'], e['stop'])
                      for e in read_csv_bi(ref_path) if e['label'] == 'seiz']
        self.signal_view.set_references(self._refs)
        self.prob_strip.set_references(self._refs)

        # probability cache — run inference on-demand if missing
        cached = load_probs(path)
        if cached is None:
            cached = self._ensure_probs_with_progress(path)
        if cached is None:
            self._probs = None
            self.prob_strip.set_probs([], [])
            msg = '{} — inference cancelled, no probabilities loaded.'.format(
                os.path.basename(path))
        else:
            self._set_source_cache('baseline', cached)
            self._activate_prob_source('baseline', preserve_view=True)
            unscored = int(np.count_nonzero(
                np.asarray(cached.get('skip_code', []))))
            msg = '{} — {:.1f}s, {} score windows, {} reference seizures'.format(
                os.path.basename(path), dur, len(cached['probs']),
                len(self._refs))
            if unscored:
                msg += '  |  {} NOT ASSESSED'.format(unscored)

        if self._load_cached_zuna_source():
            msg += ' — cached ZUNA available'

        # Fresh file: do reset the viewport to the start.
        self._refresh_signal_view(preserve_view=False)
        self._rebuild_events_from_probs(preserve_review=False)
        self.statusBar().showMessage(msg)

    def _ensure_probs_with_progress(self, edf_path):
        """Run inference on an EDF that has no cached probs, with a
        cancellable progress dialog. Caches the result next to the EDF.
        Returns a dict matching load_probs()'s shape, or None if the user
        cancelled or inference failed."""
        from gui.io.infer import compute_probs
        from gui.io.cache import save_probs

        dlg = QtWidgets.QProgressDialog(
            'Loading model (first open only)…',
            'Cancel', 0, 100, self)
        dlg.setWindowTitle('Detecting seizures')
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()
        QtWidgets.QApplication.processEvents()

        class _Cancelled(Exception):
            pass

        def progress_cb(i, n):
            if dlg.maximum() != n:
                dlg.setMaximum(n)
            dlg.setValue(i)
            dlg.setLabelText('Scoring window {}/{}…'.format(i, n))
            QtWidgets.QApplication.processEvents()
            if dlg.wasCanceled():
                raise _Cancelled()

        try:
            starts, probs, skip_code = compute_probs(
                edf_path, progress_cb=progress_cb,
                data=self._original_data, fs=self._fs)
        except _Cancelled:
            return None
        except Exception as ex:
            QtWidgets.QMessageBox.critical(
                self, 'Inference failed',
                'Could not run seizure detection on this file:\n\n{}'.format(ex))
            return None
        finally:
            dlg.close()

        meta = {'step_s': 6, 'segment_s': 12, 'use_ica': True,
                'fs': 250, 'weights': 'convlstm_ICA_12_train.h5'}
        try:
            save_probs(edf_path, starts, probs, meta=meta,
                       skip_code=skip_code)
        except Exception as ex:
            self.statusBar().showMessage(
                'Inference done but failed to write cache: {}'.format(ex), 5000)
        return {'window_starts': starts, 'probs': probs, 'meta': meta,
                'skip_code': skip_code, 'skip_code_is_exact': True}

    def _reset_source_combo(self):
        self.cb_source.blockSignals(True)
        self.cb_source.clear()
        self.cb_source.addItem('Baseline', 'baseline')
        self.cb_source.setCurrentIndex(0)
        self.cb_source.blockSignals(False)

    def _set_source_cache(self, key, cache):
        self._prob_sources[key] = cache
        if key == 'zuna':
            self._ensure_zuna_combo_item()

    def _ensure_zuna_combo_item(self):
        for i in range(self.cb_source.count()):
            if self.cb_source.itemData(i) == 'zuna':
                return
        self.cb_source.blockSignals(True)
        self.cb_source.addItem('ZUNA full', 'zuna')
        self.cb_source.blockSignals(False)

    def _on_source_change(self, _idx):
        key = self.cb_source.currentData()
        if not key:
            return
        key = str(key)
        if key not in self._prob_sources:
            self._sync_source_combo_to_key(self._prob_source)
            self.statusBar().showMessage(
                '{} source is not available for this EDF yet.'.format(
                    'ZUNA' if key == 'zuna' else 'Baseline'), 5000)
            return
        self._activate_prob_source(key, preserve_view=True)

    def _activate_prob_source(self, key, preserve_view=True):
        cache = self._prob_sources.get(key)
        if cache is None:
            return
        if self._prob_source:
            self._events_by_source[self._prob_source] = list(self._events)
        self._prob_source = key
        if key == 'zuna' and self._zuna_data is not None:
            self._raw_data = self._zuna_data
        else:
            self._raw_data = self._original_data
        self._probs = (cache['window_starts'], cache['probs'])
        self._segment_s = int(cache.get('meta', {}).get('segment_s', 12))
        self.prob_strip.set_probs(
            cache['window_starts'], cache['probs'],
            segment_s=self._segment_s,
            skip_code=cache.get('skip_code'))
        self._events = list(self._events_by_source.get(key, []))
        self._refresh_signal_view(preserve_view=preserve_view)
        self._rebuild_events_from_probs(
            preserve_review=bool(self._events))
        self._sync_source_combo_to_key(key)
        self._update_source_status()

    def _sync_source_combo_to_key(self, key):
        self.cb_source.blockSignals(True)
        for i in range(self.cb_source.count()):
            if self.cb_source.itemData(i) == key:
                self.cb_source.setCurrentIndex(i)
                break
        self.cb_source.blockSignals(False)

    def _unscored_count(self):
        """How many windows of the active source carry no model score."""
        cache = self._prob_sources.get(self._prob_source) or {}
        skip = cache.get('skip_code')
        if skip is None:
            return 0
        return int(np.count_nonzero(np.asarray(skip)))

    def _update_source_status(self):
        if not self._edf_path:
            return
        label = 'ZUNA full' if self._prob_source == 'zuna' else 'Baseline'
        n = len(self._probs[1]) if self._probs is not None else 0
        extra = ''
        if self._prob_source == 'zuna':
            extra = ' — reconstructed signal displayed'
        # Surface the not-assessed count: a file where most windows were refused
        # looks identical to a quiet recording unless the reviewer is told.
        unscored = self._unscored_count()
        if unscored:
            extra += (' — {} of {} windows NOT ASSESSED (grey bands)'
                      .format(unscored, n))
        self.statusBar().showMessage(
            '{} source active — {} score windows{}'.format(label, n, extra))
        self._warn_if_nothing_assessed(unscored, n)

    def _warn_if_nothing_assessed(self, unscored, n):
        """Warn once per file+source when the detector assessed nothing.

        Guarded so switching between baseline and ZUNA on such a file does not
        re-prompt, which would train the reviewer to dismiss it unread.
        """
        if not (n and unscored == n):
            return
        key = (self._edf_path, self._prob_source)
        if key in self._nothing_assessed_warned:
            return
        self._nothing_assessed_warned.add(key)
        QtWidgets.QMessageBox.warning(
            self, 'No windows could be assessed',
            'The detector could not assess ANY window in this recording '
            '({} of {} were rejected by the artifact check or failed ICA).\n\n'
            'The score strip is empty because nothing was scored. This is NOT '
            'evidence that the recording is free of seizures.'
            .format(unscored, n))

    def _load_cached_zuna_source(self):
        if not self._edf_path:
            return False
        paths = self._zuna_paths or zuna_session_paths(self._edf_path)
        if not (os.path.exists(paths['npz']) and os.path.exists(paths['probs'])):
            return False
        try:
            from gui.io.infer import load_signal_npz
            data, fs, duration_s = load_signal_npz(paths['npz'])
            cache = load_probability_file(paths['probs'])
            if cache is None:
                return False
            if not self._zuna_cache_matches_current_file(cache, paths):
                return False
        except Exception as ex:
            self.statusBar().showMessage(
                'Cached ZUNA output could not be loaded: {}'.format(ex), 7000)
            return False
        if abs(float(duration_s) - float(self._duration_s)) > 2.0:
            self.statusBar().showMessage(
                'Ignoring cached ZUNA output: duration differs from EDF.',
                7000)
            return False
        if int(fs) != int(self._fs):
            self.statusBar().showMessage(
                'Ignoring cached ZUNA output: sampling rate differs from EDF.',
                7000)
            return False
        self._zuna_data = data
        self._set_source_cache('zuna', cache)
        self._zuna_paths = paths
        return True

    def _zuna_cache_matches_current_file(self, cache, paths):
        meta = cache.get('meta', {})
        if meta.get('source') != 'full_zuna':
            return False
        if os.path.abspath(meta.get('source_edf', '')) != os.path.abspath(self._edf_path):
            return False
        if os.path.abspath(meta.get('source_npz', '')) != os.path.abspath(paths['npz']):
            return False
        try:
            edf_st = os.stat(self._edf_path)
            npz_st = os.stat(paths['npz'])
            if int(meta.get('source_edf_size', -1)) != int(edf_st.st_size):
                return False
            if abs(float(meta.get('source_edf_mtime', -1.0)) -
                   float(edf_st.st_mtime)) > 1.0:
                return False
            if int(meta.get('source_npz_size', -1)) != int(npz_st.st_size):
                return False
            if abs(float(meta.get('source_npz_mtime', -1.0)) -
                   float(npz_st.st_mtime)) > 1.0:
                return False
        except Exception:
            return False
        return True

    def _cancel_zuna_for_file_change(self):
        if self._zuna_thread is None or not self._zuna_thread.isRunning():
            return
        self._zuna_thread.cancel()
        if self._zuna_progress is not None:
            self._zuna_progress.close()
            self._zuna_progress = None
        self.statusBar().showMessage(
            'Cancelled previous full-ZUNA run because a new EDF was opened.',
            7000)

    def _run_full_zuna_for_session(self):
        if not self._edf_path:
            return
        if self._zuna_thread is not None and self._zuna_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self, 'ZUNA already running',
                'A full ZUNA reconstruction is already running for this EDF.')
            return
        if self._load_cached_zuna_source():
            self._activate_prob_source('zuna', preserve_view=True)
            return

        paths = self._zuna_paths or zuna_session_paths(self._edf_path)
        if os.path.exists(paths['npz']) and not os.path.exists(paths['probs']):
            self._score_zuna_output(paths)
            return

        answer = QtWidgets.QMessageBox.question(
            self, 'Run full ZUNA?',
            'Full ZUNA reconstructs the whole EDF and can take a long time. '
            'The baseline detector will remain available while it runs.\n\n'
            'Run full ZUNA for this file now?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return

        try:
            self._zuna_thread = ZunaRunThread(self._edf_path, self)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(
                self, 'ZUNA setup failed',
                'Could not start full ZUNA:\n\n{}'.format(ex))
            return

        self._zuna_run_settings = self._zuna_thread.settings()
        self._zuna_stage = 'Starting full ZUNA'
        self._zuna_progress_started = time.time()
        self._zuna_stage_started = self._zuna_progress_started

        self._zuna_progress = QtWidgets.QProgressDialog(
            'Starting full ZUNA...', 'Cancel ZUNA', 0, 100, self)
        self._zuna_progress.setWindowTitle('Full ZUNA reconstruction')
        self._zuna_progress.setWindowModality(QtCore.Qt.NonModal)
        self._zuna_progress.setMinimumDuration(0)
        self._zuna_progress.setAutoClose(False)
        self._zuna_progress.setAutoReset(False)
        self._zuna_progress.setValue(0)
        self._zuna_progress.show()
        self._update_zuna_progress()

        self._zuna_thread.statusChanged.connect(self._on_zuna_status)
        self._zuna_thread.finishedOk.connect(self._on_zuna_finished)
        self._zuna_thread.failed.connect(self._on_zuna_failed)
        self._zuna_progress.canceled.connect(self._zuna_thread.cancel)
        self.a_run_zuna.setEnabled(False)
        self._zuna_progress_timer.start()
        self._zuna_thread.start()
        self.statusBar().showMessage('Full ZUNA started in background.')

    def _on_zuna_status(self, text):
        self._zuna_stage = text
        self._zuna_stage_started = time.time()
        self._update_zuna_progress()
        self.statusBar().showMessage(text)

    def _format_seconds(self, seconds):
        seconds = int(max(0, round(float(seconds))))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return '{}h {:02d}m {:02d}s'.format(h, m, s)
        return '{}m {:02d}s'.format(m, s)

    def _count_files(self, path, suffix):
        if not path or not os.path.isdir(path):
            return 0
        count = 0
        for _root, _dirs, files in os.walk(path):
            for name in files:
                if name.endswith(suffix):
                    count += 1
        return count

    def _zuna_progress_value(self, paths):
        stage = self._zuna_stage.lower()
        if paths.get('npz') and os.path.exists(paths['npz']):
            return 100
        if paths.get('out_fif') and os.path.exists(paths['out_fif']):
            return 90
        pt_input_count = self._count_files(
            os.path.join(paths.get('work_dir', ''), '2_pt_input'), '.pt')
        pt_output_count = self._count_files(
            os.path.join(paths.get('work_dir', ''), '3_pt_output'), '.pt')
        if pt_input_count:
            if pt_output_count:
                ratio = min(float(pt_output_count) / float(pt_input_count), 1.0)
                return int(35 + 45 * ratio)
            return 30
        if paths.get('fif') and os.path.exists(paths['fif']):
            return 20
        if 'preparing' in stage:
            return 10
        if 'running full zuna' in stage:
            return 25
        if 'exporting' in stage:
            return 85
        return 0

    def _update_zuna_progress(self):
        if self._zuna_progress is None or self._zuna_progress_started is None:
            return
        now = time.time()
        elapsed = now - self._zuna_progress_started
        settings = self._zuna_run_settings or {}
        paths = settings.get('paths') or {}
        value = self._zuna_progress_value(paths)
        pt_input_count = self._count_files(
            os.path.join(paths.get('work_dir', ''), '2_pt_input'), '.pt')
        pt_output_count = self._count_files(
            os.path.join(paths.get('work_dir', ''), '3_pt_output'), '.pt')
        artifact_note = 'artifact-based'
        if pt_input_count:
            artifact_note = '{} / {} ZUNA output tensors'.format(
                pt_output_count, pt_input_count)
        label = (
            '{}\n\n'
            'Progress: {}% ({})\n'
            'Elapsed: {}\n'
            'EDF duration: {}\n'
            'Settings: {} diffusion steps, tokens/batch {}, GPU {}\n'
            'Log: {}\n'
            'Output: {}'
        ).format(
            self._zuna_stage,
            value,
            artifact_note,
            self._format_seconds(elapsed),
            self._format_seconds(self._duration_s),
            settings.get('diffusion_steps', '?'),
            settings.get('tokens_per_batch', '?'),
            settings.get('gpu_device', '?'),
            paths.get('log', 'pending'),
            paths.get('root', 'pending'),
        )
        self._zuna_progress.setValue(value)
        self._zuna_progress.setLabelText(label)

    def _close_zuna_progress(self):
        self._zuna_progress_timer.stop()
        if self._zuna_progress is not None:
            self._zuna_progress.close()
            self._zuna_progress = None
        self._zuna_progress_started = None
        self._zuna_stage_started = None
        self._zuna_stage = 'Idle'
        self._zuna_run_settings = {}
        self.a_run_zuna.setEnabled(bool(self._edf_path))

    def _on_zuna_failed(self, edf_path, message):
        if os.path.abspath(edf_path) != os.path.abspath(self._edf_path or ''):
            return
        self._close_zuna_progress()
        if 'cancelled' in message.lower():
            self.statusBar().showMessage('Full ZUNA cancelled.', 7000)
            return
        QtWidgets.QMessageBox.critical(
            self, 'ZUNA failed',
            'Full ZUNA did not complete:\n\n{}\n\nCheck the ZUNA log under '
            'artifacts/gui_zuna for details if a log was created.'
            .format(message))
        self.statusBar().showMessage('Full ZUNA failed.', 7000)

    def _on_zuna_finished(self, edf_path, paths):
        if os.path.abspath(edf_path) != os.path.abspath(self._edf_path or ''):
            return
        self._close_zuna_progress()
        self._score_zuna_output(paths, source_edf=edf_path)

    def _score_zuna_output(self, paths, source_edf=None):
        from gui.io.infer import compute_probs_from_data, load_signal_npz

        source_edf = os.path.abspath(source_edf or self._edf_path or '')
        if source_edf != os.path.abspath(self._edf_path or ''):
            self.statusBar().showMessage(
                'Ignoring ZUNA output from a different EDF session.', 7000)
            return
        try:
            data, fs, duration_s = load_signal_npz(paths['npz'])
        except Exception as ex:
            QtWidgets.QMessageBox.critical(
                self, 'Could not load ZUNA output',
                'ZUNA finished but the exported NPZ could not be loaded:\n\n{}'
                .format(ex))
            return
        if abs(float(duration_s) - float(self._duration_s)) > 2.0:
            QtWidgets.QMessageBox.critical(
                self, 'ZUNA output mismatch',
                'The ZUNA output duration does not match the open EDF.')
            return
        if int(fs) != int(self._fs):
            QtWidgets.QMessageBox.critical(
                self, 'ZUNA output mismatch',
                'The ZUNA output sampling rate does not match the open EDF.')
            return

        dlg = QtWidgets.QProgressDialog(
            'Scoring ZUNA output…', 'Cancel', 0, 100, self)
        dlg.setWindowTitle('Detecting seizures on ZUNA output')
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()
        QtWidgets.QApplication.processEvents()

        class _Cancelled(Exception):
            pass

        def progress_cb(i, n):
            if dlg.maximum() != n:
                dlg.setMaximum(n)
            dlg.setValue(i)
            dlg.setLabelText('Scoring ZUNA window {}/{}…'.format(i, n))
            QtWidgets.QApplication.processEvents()
            if dlg.wasCanceled():
                raise _Cancelled()

        try:
            starts, probs, skip_code = compute_probs_from_data(
                data, fs, progress_cb=progress_cb)
        except _Cancelled:
            self.statusBar().showMessage('ZUNA scoring cancelled.', 5000)
            return
        except Exception as ex:
            QtWidgets.QMessageBox.critical(
                self, 'ZUNA scoring failed',
                'Could not run seizure detection on ZUNA output:\n\n{}'
                .format(ex))
            return
        finally:
            dlg.close()

        meta = {
            'step_s': 6,
            'segment_s': 12,
            'use_ica': True,
            'fs': int(fs),
            'weights': 'convlstm_ICA_12_train.h5',
            'source': 'full_zuna',
            'source_npz': paths['npz'],
            'source_edf': source_edf,
        }
        try:
            edf_st = os.stat(source_edf)
            meta['source_edf_size'] = int(edf_st.st_size)
            meta['source_edf_mtime'] = float(edf_st.st_mtime)
        except OSError:
            pass
        try:
            npz_st = os.stat(paths['npz'])
            meta['source_npz_size'] = int(npz_st.st_size)
            meta['source_npz_mtime'] = float(npz_st.st_mtime)
        except OSError:
            pass
        try:
            save_probability_file(paths['probs'], starts, probs, meta,
                                  skip_code=skip_code)
        except Exception as ex:
            self.statusBar().showMessage(
                'ZUNA probabilities computed but cache write failed: {}'
                .format(ex), 7000)

        self._zuna_data = data
        self._set_source_cache(
            'zuna',
            {'window_starts': starts, 'probs': probs, 'meta': meta,
             'skip_code': skip_code, 'skip_code_is_exact': True})
        self._activate_prob_source('zuna', preserve_view=True)
        self.statusBar().showMessage(
            'Full ZUNA ready — switched to ZUNA source. '
            'Output: {}'.format(os.path.basename(paths['npz'])), 9000)

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
        if self._prob_source == 'zuna':
            labels = ['{} (ZUNA)'.format(label) for label in labels]
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
    def _rebuild_events_from_probs(self, preserve_review=True):
        self.prob_strip.set_threshold(self._threshold)
        if self._probs is None:
            self._events = []
            self.signal_view.set_events([])
            self.event_list.set_events([])
            return
        starts, probs = self._probs
        self._events = rebuild_events(
            starts, probs, self._threshold, self._segment_s,
            duration_s=self._duration_s,
            previous_events=self._events,
            preserve_review=preserve_review)
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
            s, e = clamp_interval(s, e, self._duration_s)
            ev['start'] = float(s)
            ev['stop'] = float(e)
            self.signal_view.update_event_region(event_id, s, e)
            if ev['status'] == 'proposed':
                ev['status'] = 'edited'
                self.signal_view.update_event_status(event_id, 'edited')
            self.event_list.update_row(event_id, ev)
            self.event_list._refresh_summary(self._events)

    def _on_selection_change(self, event_id):
        if event_id > 0:
            self._on_jump(event_id)

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
        self._rebuild_events_from_probs(preserve_review=True)

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
        self.event_list.cycle_selection(delta)

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
        suffix = (
            '.zuna.reviewed.csv_bi'
            if self._prob_source == 'zuna' else '.reviewed.csv_bi')
        default = os.path.splitext(self._edf_path)[0] + suffix
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export reviewed annotations', default,
            'TUSZ csv_bi (*.csv_bi)')
        if not path:
            return
        events = []
        exported_scores = []
        for ev in self._events:
            if ev['status'] not in ('accepted', 'edited'):
                continue
            s, e = clamp_interval(ev['start'], ev['stop'], self._duration_s)
            # confidence = 1.0, not the model score. Only human-confirmed events
            # are exported, so this column describes the REVIEWER's confidence,
            # and TUSZ references carry 1.0000 in it. Writing the network's
            # uncalibrated score here made a human-confirmed annotation look
            # like a low-confidence one. The score is kept in the provenance
            # sidecar instead, where it is unambiguously the model's.
            events.append({
                'start': s, 'stop': e,
                'label': 'seiz', 'confidence': 1.0,
            })
            exported_scores.append({
                'start': round(float(s), 4), 'stop': round(float(e), 4),
                'status': ev['status'],
                'model_score_uncalibrated': round(float(ev['prob']), 6),
            })
        write_csv_bi(path,
                     bname=os.path.splitext(os.path.basename(self._edf_path))[0],
                     duration_s=self._duration_s,
                     events=events,
                     ai_source=self._prob_source)
        self._write_export_provenance(path, len(events),
                                      exported_scores=exported_scores)
        self.statusBar().showMessage(
            'Exported {} reviewed events → {}'.format(
                len(events), os.path.basename(path)), 5000)

    def _write_export_provenance(self, csv_bi_path, n_events,
                                 exported_scores=None):
        """Write a provenance sidecar for ANY reviewed export (baseline or
        ZUNA), so a reviewed .csv_bi is always traceable to the exact source,
        detector settings, and display state that produced it."""
        meta_path = csv_bi_path + '.provenance.json'
        src_meta = (self._prob_sources.get(self._prob_source) or {}).get(
            'meta') or {}
        payload = {
            'annotation_file': csv_bi_path,
            'tool': csv_bi_module.TOOL_NAME,
            'not_for_clinical_use': True,
            'status': csv_bi_module.NOT_FOR_CLINICAL_USE,
            'active_ai_source': self._prob_source,
            'n_exported_events': int(n_events),
            'threshold': float(self._threshold),
            'edf': self._edf_path,
            'edf_sha256': self._edf_sha256,
            'detector': {
                'weights': src_meta.get('weights'),
                'weights_sha256': _weights_sha256(),
                'step_s': src_meta.get('step_s'),
                'segment_s': src_meta.get('segment_s'),
                'use_ica': src_meta.get('use_ica'),
                'fs': src_meta.get('fs'),
            },
            'display': {
                'montage': self._montage,
                'highpass_hz': self._hp,
                'lowpass_hz': self._lp,
                'notch_hz': self._notch,
            },
            'scores_are_calibrated': False,
            'score_note': (
                'model_score_uncalibrated values are raw softmax outputs with '
                'no post-hoc calibration. The csv_bi confidence column is 1.0 '
                'for every exported event because each was human-confirmed.'),
            'exported_events': exported_scores or [],
            'windows_not_assessed': self._unscored_count(),
        }
        if self._prob_source == 'zuna':
            payload['zuna_npz'] = (
                self._zuna_paths.get('npz') if self._zuna_paths else None)
            payload['zuna_probs'] = (
                self._zuna_paths.get('probs') if self._zuna_paths else None)
            payload['note'] = (
                'Events were reviewed while full-ZUNA reconstructed signal and '
                'ZUNA-derived ConvLSTM probabilities were active. ZUNA signal '
                'is reconstructed/imputed data, not original clinical EEG.')
        with open(meta_path, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')
