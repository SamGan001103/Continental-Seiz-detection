"""MainWindow for the seizure-review GUI.

Toolbar groups (left to right):
    Open | Montage | Filters (HP / LP / notch) | Sensitivity | Timebase |
    Threshold | Export

Keyboard shortcuts:
    Space     accept selected event
    X         reject selected event
    Shift+X   reject and record a reason
    N         add an event the detector missed
    Ctrl+Z    undo the last review action
    Ctrl+R    revert an edited event to the detector's extent
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
import hashlib
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
from gui.io.zuna import (ZunaFullRunner, zuna_session_paths,
                         zuna_available)
from gui.events import (clamp_interval, rebuild_events, assign_event_ids,
                        EXPORTED_STATUSES)
from gui.processing import (apply_montage, apply_filters, MONTAGES)
from gui.widgets.head_view import HeadView, derivations_for
from gui.io.ica_display import (SEGMENT_S as ICA_SEGMENT_S,
                                cache_clear as ica_cache_clear,
                                clean_span_for_display,
                                recording_key as ica_recording_key,
                                summarise as ica_summarise)
from gui.widgets.signal_view import SignalView
from gui.widgets.prob_strip import ProbStrip
from gui.widgets.event_list import EventList
from gui.widgets.channel_inspector import ChannelInspector

# Clinical review conventions, not arbitrary numbers. Amplitude is quoted in
# µV/mm and sweep in sec/page, which is what an EEG reader expects; "µV/div"
# and a bare "timebase s" are neither standard nor unambiguous, and the
# heuristic-evaluation instrument flagged them. 7 µV/mm and 10 s/page are the
# usual defaults for reading seizures.
SENSITIVITIES = [3, 5, 7, 10, 15, 20, 30, 50, 70, 100]   # µV / mm
TIMEBASES = [5, 10, 15, 20, 30, 60, 120, 300]            # seconds per page

APP_NAME = 'Seizure Review'
# Shown permanently in the title bar and the status bar, and written into every
# export. The detector runs at roughly 48% event sensitivity, so anyone sitting
# in front of this — supervisor, peer reviewer, clinician — must be able to see
# what it is without being told.
PROTOTYPE_BANNER = 'RESEARCH PROTOTYPE — NOT FOR DIAGNOSTIC USE'
TITLE_SUFFIX = ' — RESEARCH PROTOTYPE, NOT FOR DIAGNOSTIC USE'

# Resolved once, at import. The ZUNA controls are hidden entirely when a run is
# impossible — chiefly in the packaged application, which has no second
# interpreter and no utils/zuna_bridge.py. Leaving a dead button on the toolbar
# of a clinical build invites a press that can only end in a subprocess error.
ZUNA_AVAILABLE = zuna_available()


def _git_commit():
    """Short commit of the working tree, for export provenance.

    peer_review_protocol.md asks the facilitator to log the build under test,
    which the system previously gave them no way to obtain. Suffixed '+dirty'
    when the tree has uncommitted changes, because a session run against
    modified code is not reproducible from the commit alone.
    """
    # A frozen build has no repository and the target machine has no git, so
    # the commit is stamped into the bundle at build time instead. Without this
    # every exported annotation from the packaged app would have a null build
    # identifier — exactly the machine where provenance matters most.
    from gui.paths import is_frozen, resource
    if is_frozen():
        try:
            import json
            with open(resource('build_info.json'), encoding='utf-8') as f:
                return json.load(f).get('commit')
        except Exception:
            return None
    import subprocess
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO,
            stderr=subprocess.PIPE).decode('ascii').strip()
        # --untracked-files=no: plain --porcelain also lists untracked files,
        # and this repository permanently carries a large untracked thesis
        # bundle, so every export was stamped "+dirty" whatever the code was
        # doing. A flag that is always set carries no information. "+dirty"
        # means a TRACKED file differs from the commit, i.e. the session cannot
        # be reproduced from the commit alone — which is the thing a reviewer
        # actually needs to know.
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO,
            stderr=subprocess.PIPE).decode('utf-8', 'replace').strip()
        return sha + ('+dirty' if dirty else '')
    except Exception:
        return None


def _cfg_weights():
    try:
        import eval_config as cfg
        return cfg.WEIGHTS
    except Exception:
        return 'convlstm_ICA_12_train.h5'


def _weights_sha256():
    """Hash of the weights file actually on disk, for export provenance.

    Compared against eval_config.WEIGHTS_SHA256 so a reviewed annotation can be
    tied to the exact model that produced it, and so a swapped or corrupted
    weights file is detectable after the fact rather than silently accepted.
    """
    try:
        from gui.io.cache import sha256_file
        from gui.paths import weights_path
        return sha256_file(weights_path())
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
        # Geometry is restored if this reviewer has run the app before, and
        # otherwise the window opens maximised. A review session wants every
        # pixel of vertical space it can get for traces, and 1600x960 on a
        # 2560x1440 screen wastes half of it. saveState() carries the dock
        # layout too, so a reviewer who moved or closed the electrode map keeps
        # that arrangement.
        self.resize(1600, 960)
        self._settings = QtCore.QSettings('NeuroSyd', 'SeizureReview')
        geo = self._settings.value('geometry')
        state = self._settings.value('windowState')
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.setWindowState(self.windowState() |
                                QtCore.Qt.WindowMaximized)
        self._restore_dock_state = state

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
        self._dirty = False            # reviewer decisions not yet exported
        self._reviewer_id = None       # prompted once per session
        self._last_open_dir = None     # reviewers work out of one folder
        self._autosave_written_to = None   # where crash recovery actually went
        self._session_started = time.time()
        self._file_opened_at = None
        self._events = []
        self._threshold = 0.5

        self._montage = 'Longitudinal Bipolar'
        self._hp = 1.0
        self._lp = 70.0
        self._notch = 60.0   # TUH is US data — 60 Hz mains
        self._sensitivity_uv = 7.0     # µV/mm — clinical default
        self._timebase_s = 10          # sec/page — clinical default

        # cache of the most-recent post-filter, post-montage display data so
        # channel-inspector popups can be opened without recomputing.
        self._display_data = None      # [K, N] float32, before clipping
        self._display_labels = []      # list[str]
        self._display_t = None         # [N] seconds
        self._inspectors = {}          # channel_idx -> ChannelInspector

        # Display-side ICA. Off by default: the raw trace is what a reviewer
        # expects to open on, and the cleaned one answers a question they have
        # to ask first. It re-runs the DETECTOR'S OWN ica_arti_remove, read
        # only, on a copy — so what it draws is literally what the model was
        # fed, not an approximation of it.
        self._ica_display = False
        self._ica_view_span = None     # (x0, x1) the span last cleaned
        self._ica_anchor_s = 0         # block grid alignment
        self._ica_recording_id = None  # cache identity for this file
        self._ica_blocks = []          # per-block IcaDisplayInfo, for the UI
        # Scrolling fires viewRangeChanged continuously and a ~1 s ICA per
        # event would make the scrollbar feel broken. Coalesce.
        self._ica_timer = QtCore.QTimer(self)
        self._ica_timer.setSingleShot(True)
        self._ica_timer.setInterval(250)
        self._ica_timer.timeout.connect(
            lambda: self._refresh_signal_view(preserve_view=True))

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
        # 4:1 rather than 5:1. The strip is the detector's entire output and
        # the thing a reviewer scans to decide where to look; at one sixth of
        # the height a shallow rise near threshold was hard to read at all.
        lv.addWidget(self.signal_view, 4)
        lv.addWidget(self.prob_strip, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.event_list)
        # The worklist holds five narrow columns and was given 380 px of a
        # 1600 px window - width taken directly from the traces, which are the
        # thing being read. 300 is enough for the widest row.
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1300, 300])
        self.event_list.setMaximumWidth(420)
        self.setCentralWidget(splitter)

        # Head map. Shows where a derivation actually sits on the scalp, which
        # a row label like "Fp1-F7" only tells a reader who already knows the
        # montage. Display only: it never changes what was scored, and it says
        # so in a caption that is drawn into reserved space so it cannot be
        # clipped away by a narrow dock.
        self.head_view = HeadView()
        self.head_view.set_montage(self._montage)
        self.head_view.electrodeClicked.connect(self._on_electrode_clicked)
        # Title matches the View menu entry verbatim. They differed - 'Electrodes'
        # against 'Electrode map' - and a reviewer hunting for a panel they
        # closed should not have to infer that two names mean one thing.
        self._head_dock = QtWidgets.QDockWidget('Electrode map (10-20)', self)
        self._head_dock.setObjectName('headDock')   # saveState() needs a name
        self._head_dock.setWidget(self.head_view)
        self._head_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea |
                                        QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._head_dock)

        self._build_toolbar()
        self._build_menu()
        # After the docks and the menu exist, or restoreState has nothing to
        # attach to and silently drops the arrangement.
        if getattr(self, '_restore_dock_state', None) is not None:
            self.restoreState(self._restore_dock_state)

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

        # No shortcut here: File > Open EDF carries Ctrl+O. Setting it on both
        # the toolbar button and the menu item is the same ambiguity that
        # silently disabled N, Ctrl+Z and Ctrl+R - Qt refuses both and neither
        # fires. The toolbar button stays clickable; the key lives in the menu,
        # where it is also visible to the reviewer.
        a_open = tb.addAction('Open EDF…')
        a_open.setToolTip('Open an EDF recording (Ctrl+O)')
        a_open.triggered.connect(self._open_edf_dialog)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel('Montage'))
        self.cb_montage = QtWidgets.QComboBox()
        self.cb_montage.addItems(list(MONTAGES.keys()))
        self.cb_montage.setToolTip(
            'Display montage. Display only. The detector always scores the unfiltered, referential recording — changing this changes what you see, never what the AI scored.')
        self.cb_montage.setCurrentText(self._montage)
        self.cb_montage.currentTextChanged.connect(self._on_montage_change)
        tb.addWidget(self.cb_montage)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel('HP'))
        self.cb_hp = QtWidgets.QComboBox()
        self.cb_hp.addItems(['off', '0.3', '0.5', '1.0', '3.0', '5.0'])
        self.cb_hp.setToolTip(
            'Display high-pass filter. Display only. The detector always scores the unfiltered, referential recording — changing this changes what you see, never what the AI scored.')
        self.cb_hp.setCurrentText('1.0')
        self.cb_hp.currentTextChanged.connect(self._on_filters_change)
        tb.addWidget(self.cb_hp)

        tb.addWidget(QtWidgets.QLabel('LP'))
        self.cb_lp = QtWidgets.QComboBox()
        self.cb_lp.addItems(['off', '15', '30', '35', '50', '70'])
        self.cb_lp.setToolTip(
            'Display low-pass filter. Display only. The detector always scores the unfiltered, referential recording — changing this changes what you see, never what the AI scored.')
        self.cb_lp.setCurrentText('70')
        self.cb_lp.currentTextChanged.connect(self._on_filters_change)
        tb.addWidget(self.cb_lp)

        tb.addWidget(QtWidgets.QLabel('Notch'))
        self.cb_notch = QtWidgets.QComboBox()
        self.cb_notch.addItems(['off', '50', '60'])
        self.cb_notch.setToolTip(
            'Display notch filter (mains). Display only. The detector always scores the unfiltered, referential recording — changing this changes what you see, never what the AI scored.')
        self.cb_notch.setCurrentText('60')
        self.cb_notch.currentTextChanged.connect(self._on_filters_change)
        tb.addWidget(self.cb_notch)
        tb.addSeparator()

        lbl_sens = QtWidgets.QLabel('Sensitivity µV/mm')
        lbl_sens.setToolTip('Vertical scale, in the clinical convention. '
                            'Lower = larger deflections. [ and ] step it.')
        tb.addWidget(lbl_sens)
        self.cb_sens = QtWidgets.QComboBox()
        self.cb_sens.addItems([str(v) for v in SENSITIVITIES])
        self.cb_sens.setCurrentText(str(int(self._sensitivity_uv)))
        self.cb_sens.currentTextChanged.connect(self._on_sens_change)
        tb.addWidget(self.cb_sens)

        lbl_tb = QtWidgets.QLabel('Sweep sec/page')
        lbl_tb.setToolTip('Seconds of EEG per screen width. 10 s is the usual '
                          'clinical default. , and . step it.')
        tb.addWidget(lbl_tb)
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

        self.chk_ica = QtWidgets.QCheckBox('Detector input (ICA)')
        self.chk_ica.setToolTip(
            "Re-run the detector's own ICA artifact removal on the visible "
            'span and draw that instead of the raw trace.\n\n'
            'Display only — the scores never change. Windows where no eye-'
            'movement component was found reached the detector uncleaned; '
            'the status bar says how many.')
        self.chk_ica.toggled.connect(self._on_ica_display_toggled)
        tb.addWidget(self.chk_ica)
        tb.addSeparator()

        # ZUNA is a research side-study and cannot run in the packaged
        # application (see gui.io.zuna.zuna_available). The widgets are still
        # constructed so the code paths that reference them stay simple, but
        # they are not put on the toolbar where a clinician could press them.
        self.cb_source = QtWidgets.QComboBox()
        self.cb_source.addItem('Baseline', 'baseline')
        self.cb_source.setToolTip(
            'Switch between original-EDF baseline probabilities and full '
            'ZUNA probabilities after ZUNA has finished.')
        self.cb_source.currentIndexChanged.connect(self._on_source_change)

        self.a_run_zuna = QtWidgets.QAction('Run full ZUNA', self)
        self.a_run_zuna.setToolTip(
            'Run full-file ZUNA reconstruction for this EDF. This is optional '
            'and computationally heavy.')
        self.a_run_zuna.setEnabled(False)
        self.a_run_zuna.triggered.connect(self._run_full_zuna_for_session)

        if ZUNA_AVAILABLE:
            tb.addWidget(QtWidgets.QLabel('AI source'))
            tb.addWidget(self.cb_source)
            tb.addAction(self.a_run_zuna)
            tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel('Threshold'))
        self.thr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.thr_slider.setRange(0, 100)
        self.thr_slider.setValue(int(self._threshold * 100))
        self.thr_slider.setFixedWidth(160)
        self.thr_slider.setToolTip(
            'Detection threshold. LOWER = more candidates proposed (higher '
            'sensitivity, more false alarms). HIGHER = fewer, more confident '
            'candidates.\n\nThe value is a raw uncalibrated model score, not a '
            'probability: 0.50 corresponds to roughly a 29% chance of seizure '
            'on this corpus.')
        self.thr_slider.valueChanged.connect(self._on_thr_changed)
        tb.addWidget(self.thr_slider)
        self.thr_lbl = QtWidgets.QLabel(' {:.2f} '.format(self._threshold))
        tb.addWidget(self.thr_lbl)
        tb.addSeparator()

        a_export = tb.addAction('Export reviewed…')
        a_export.triggered.connect(self._export_reviewed)

    # ==================================================================
    # Menu bar — the walkthrough's dominant failure mode was Q2/Q3:
    # controls that work correctly but cannot be found. There was no menu bar
    # at all, so J/K existed only in this module's docstring and the channel
    # inspector could be reached only by double-clicking a channel label.
    # ==================================================================
    def _build_menu(self):
        mb = self.menuBar()

        m_file = mb.addMenu('&File')
        a_open = m_file.addAction('&Open EDF…')
        a_open.setShortcut(QtGui.QKeySequence.Open)
        a_open.triggered.connect(self._open_edf_dialog)
        m_file.addSeparator()
        a_exp = m_file.addAction('&Export reviewed…')
        a_exp.setShortcut('Ctrl+E')
        a_exp.triggered.connect(self._export_reviewed)
        m_file.addSeparator()
        a_quit = m_file.addAction('&Quit')
        a_quit.setShortcut(QtGui.QKeySequence.Quit)
        a_quit.triggered.connect(self.close)

        m_edit = mb.addMenu('&Edit')
        a_undo = m_edit.addAction('&Undo')
        a_undo.setShortcut('Ctrl+Z')
        a_undo.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        a_undo.triggered.connect(self._undo)
        m_edit.addSeparator()
        a_add = m_edit.addAction('&Add event the detector missed')
        a_add.setShortcut('N')
        a_add.setToolTip('Record a seizure the detector did not propose. '
                         'Without this the reviewer can only remove events, '
                         'never add one.')
        a_add.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        a_add.triggered.connect(self._add_event_at_view)
        a_rev = m_edit.addAction('&Revert extent to detector')
        a_rev.setShortcut('Ctrl+R')
        a_rev.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        a_rev.triggered.connect(self._revert_extent)

        m_view = mb.addMenu('&View')
        # A closed dock is otherwise unrecoverable except through Qt's own
        # right-click menu on the toolbar, which nobody discovers. toggleViewAction
        # ticks and unticks itself, so it also shows whether the panel is open.
        # No shortcut on it: N, Ctrl+Z, Ctrl+R and Ctrl+O were all silently dead
        # because a sequence was registered twice, and tests/test_review_guards.py
        # now fails on any duplicate.
        act_head = self._head_dock.toggleViewAction()
        act_head.setText('&Electrode map (10-20)')
        act_head.setToolTip('Show where each displayed derivation sits on the '
                            'scalp. Display only.')
        m_view.addAction(act_head)
        m_view.addSeparator()
        self.a_inspector = m_view.addAction('&Channel inspector…')
        self.a_inspector.setShortcut('Ctrl+I')
        self.a_inspector.setToolTip(
            'Open a detail window for one channel. You can also double-click '
            'a channel label in the trace view.')
        self.a_inspector.triggered.connect(self._choose_channel_inspector)
        m_view.addSeparator()
        self.a_blind = m_view.addAction('&Blind mode (hide ground truth)')
        self.a_blind.setCheckable(True)
        self.a_blind.setShortcut('Ctrl+B')
        self.a_blind.setToolTip(
            'Hide the green reference bands. Any review session used as '
            'evidence should run in blind mode — the reference is recorded in '
            'the export provenance either way.')
        self.a_blind.toggled.connect(self._on_blind_toggled)

        m_help = mb.addMenu('&Help')
        a_keys = m_help.addAction('&Keyboard shortcuts')
        a_keys.setShortcut('F1')
        a_keys.triggered.connect(self._show_shortcuts)
        a_use = m_help.addAction('&Intended use and limitations…')
        a_use.triggered.connect(self._show_intended_use)
        a_about = m_help.addAction('&About')
        a_about.triggered.connect(self._show_about)

    def _show_intended_use(self):
        """Show the bundled INTENDED_USE.md inside the application.

        It ships inside the application so a reviewer on an offline clinical PC
        can reach it — a limitations document that exists only in the
        repository is not available to the person who needs it.

        Rendered in-window rather than handed to the OS. A fresh Windows
        install has no default handler for `.md`, so opening it externally
        would either do nothing or raise a "how do you want to open this file?"
        prompt — on the one machine where this document most needs to be
        readable.
        """
        from gui.paths import resource
        text = None
        for cand in (resource('doc', 'INTENDED_USE.md'),
                     os.path.join(REPO, 'docs', 'INTENDED_USE.md')):
            if os.path.exists(cand):
                try:
                    with open(cand, encoding='utf-8') as f:
                        text = f.read()
                    break
                except OSError:
                    continue
        if text is None:
            QtWidgets.QMessageBox.warning(
                self, 'Not found',
                'INTENDED_USE.md is missing from this installation.\n\n'
                'The application folder may have been copied incompletely.')
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Intended use and limitations')
        dlg.resize(820, 720)
        view = QtWidgets.QTextBrowser(dlg)
        view.setOpenExternalLinks(False)
        try:
            view.setMarkdown(text)
        except AttributeError:
            # Qt < 5.14 has no markdown renderer. Plain text keeps the content
            # readable rather than showing raw markup as broken HTML.
            view.setPlainText(text)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close, parent=dlg)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(view)
        lay.addWidget(buttons)
        dlg.exec_()

    def _show_shortcuts(self):
        # Reuses this module's docstring verbatim so the dialog cannot drift
        # from the shortcuts actually wired in _wire_shortcuts().
        body = (__doc__ or '').split('Keyboard shortcuts:', 1)[-1].strip()
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle('Keyboard shortcuts')
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText('<b>Keyboard shortcuts</b>')
        box.setInformativeText(
            '<pre style="font-size:12px">{}</pre>'
            '<p>Also: <b>Ctrl+O</b> open · <b>Ctrl+E</b> export · '
            '<b>Ctrl+I</b> channel inspector · <b>Ctrl+B</b> blind mode · '
            '<b>F1</b> this dialog.<br>'
            'Double-clicking a channel label also opens its inspector.</p>'
            .format(body))
        box.exec_()

    def _show_about(self):
        QtWidgets.QMessageBox.about(
            self, 'About {}'.format(APP_NAME),
            '<b>{}</b><br>{}<br><br>'
            'Human-in-the-loop EEG seizure review. BMET4111 thesis, '
            'University of Sydney.<br><br>'
            'Detector: 19-channel ConvLSTM (<code>{}</code>), 12 s windows at '
            '6 s stride, per-window ICA.<br>'
            'Scores shown are <b>raw, uncalibrated</b> network outputs, not '
            'calibrated probabilities of seizure.<br><br>'
            'Validated on public TUH/TUSZ data only. No patient data.'
            .format(APP_NAME, PROTOTYPE_BANNER, _cfg_weights()))

    def _choose_channel_inspector(self):
        """Menu route to the inspector, which was double-click-only before."""
        if not self._display_labels:
            self.statusBar().showMessage('Open an EDF first.', 4000)
            return
        name, ok = QtWidgets.QInputDialog.getItem(
            self, 'Channel inspector', 'Channel:', self._display_labels, 0,
            False)
        if ok and name in self._display_labels:
            self._open_channel_inspector(self._display_labels.index(name))

    def _on_blind_toggled(self, blind):
        """Blind mode hides ground truth so a session can be used as evidence."""
        self.chk_refs.blockSignals(True)
        self.chk_refs.setChecked(not blind)
        self.chk_refs.blockSignals(False)
        self._on_refs_toggled(not blind)
        self.statusBar().showMessage(
            'Blind mode ON — ground-truth bands hidden.' if blind
            else 'Blind mode OFF — ground truth is visible; this session '
                 'should not be used as unbiased evidence.', 6000)

    def _wire_shortcuts(self):
        def add(seq, fn):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(fn)

        # N, Ctrl+Z, Ctrl+R and Ctrl+O are deliberately ABSENT here. Each is
        # already a menu QAction, and registering the same sequence twice makes
        # Qt refuse both: "QAction::event: Ambiguous shortcut overload: N", and
        # nothing fires. That silently disabled the four keys - including N, the
        # only way to record a seizure the detector missed. The menu actions now
        # carry ApplicationShortcut context themselves (see _build_menu), which
        # is what these duplicates were for.
        add('Space', self._shortcut_accept)
        add('X', self._shortcut_reject)
        add('Shift+X', self._reject_with_reason)
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

    @staticmethod
    def _may_persist_layout():
        """False under the test suite, which must not write a real user's settings.

        Every headless test that constructs a MainWindow and closes it runs
        closeEvent, so the suite was saving geometry and dock state into
        the registry under NeuroSyd/SeizureReview. One of those tests closes the
        electrode map on purpose - to prove the View menu can bring it back -
        and that state persisted, so the next real launch restored a window
        arrangement a test had chosen. The panel was simply missing, with no
        indication why.

        `unittest` in sys.modules is a heuristic, and an honest one: the
        application never imports it, directly or transitively, and the frozen
        build does not ship it. The alternative - threading an opt-out through
        every test that builds a window - would be forgotten by the next test
        added, which is exactly how this happened.
        """
        return 'unittest' not in sys.modules

    def closeEvent(self, event):
        if not self._confirm_discard_review('close the application'):
            event.ignore()
            return
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
        # Only once the close is certain - both guards above can still abort it,
        # and saving geometry for a window that stays open would record a state
        # the reviewer never chose.
        try:
            if self._may_persist_layout():
                self._settings.setValue('geometry', self.saveGeometry())
                self._settings.setValue('windowState', self.saveState())
        except Exception:                                   # noqa: BLE001
            pass          # a read-only registry must not block closing
        super().closeEvent(event)

    # ==================================================================
    # File loading
    # ==================================================================
    def _open_edf_dialog(self):
        # Frozen, REPO is PyInstaller's temporary extraction directory — opening
        # the dialog there shows the user the app's own internals. Start
        # somewhere a clinician would actually keep recordings.
        from gui.paths import is_frozen
        start_dir = self._last_open_dir or ''
        if not os.path.isdir(start_dir):
            start_dir = os.path.join(REPO, 'sample_data')
        if not os.path.isdir(start_dir):
            start_dir = (os.path.expanduser('~/Documents') if is_frozen()
                         else REPO)
        if not os.path.isdir(start_dir):
            start_dir = os.path.expanduser('~')
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open EDF', start_dir, 'EDF files (*.edf)')
        if path:
            self._last_open_dir = os.path.dirname(path)
            self.load_edf(path)

    def load_edf(self, path):
        if not self._confirm_discard_review('open another recording'):
            return
        self._cancel_zuna_for_file_change()
        self.statusBar().showMessage('Loading {}…'.format(os.path.basename(path)))
        QtWidgets.QApplication.processEvents()
        try:
            data, fs, dur = load_edf_19ch(path)
        except Exception as ex:
            # Name BOTH recordings. On failure the previous one is still on
            # screen, in the title bar and in the worklist — deliberately, so a
            # reviewer's unexported work is not destroyed by a mistyped filename
            # — but "EDF load failed" alone let them believe the new file had
            # opened. They would then be reading the previous patient's EEG
            # under the impression it was this one, with the only warning a
            # status line that the next message overwrites.
            still = (os.path.basename(self._edf_path) if self._edf_path
                     else None)
            text = 'Could not load this recording:\n\n{}\n\n{}'.format(
                os.path.basename(path), ex)
            if still:
                text += ('\n\nThe previous recording is still open and is what '
                         'you are looking at:\n\n{}'.format(still))
            QtWidgets.QMessageBox.critical(self, 'Could not open EDF', text)
            if still:
                self.statusBar().showMessage(
                    'Could not open {} — still showing {}.'.format(
                        os.path.basename(path), still))
            else:
                self.statusBar().showMessage(
                    'Could not open {}. No recording is open.'.format(
                        os.path.basename(path)))
            return
        self._edf_path = path
        self._file_opened_at = time.time()
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
        self._ica_recording_id = ica_recording_key(path)
        self._ica_view_span = None
        self._ica_anchor_s = 0
        ica_cache_clear()      # the previous recording's windows are dead
        self._events = []
        self._events_by_source = {}
        self._undo_stack = []
        self._dirty = False
        self._prob_sources = {}
        self._prob_source = 'baseline'
        self._zuna_paths = zuna_session_paths(path)
        self._reset_source_combo()
        self.a_run_zuna.setEnabled(ZUNA_AVAILABLE)
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
        # ...then offer back any crash-recovery record for this recording.
        # Must run AFTER _rebuild_events_from_probs, which replaces self._events
        # wholesale with freshly proposed candidates.
        self._offer_autosave_restore()

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
        if not ZUNA_AVAILABLE:
            return
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
            skip_code=cache.get('skip_code'),
            duration_s=self._duration_s)
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
        self.a_run_zuna.setEnabled(ZUNA_AVAILABLE and bool(self._edf_path))

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
    def _display_source(self):
        """The array the display chain filters: raw, or the detector's own ICA.

        Returns a FULL-LENGTH array either way, so SignalView, the x-linked
        probability strip, the event overlays and the channel inspectors need no
        changes — only the visible span is actually cleaned, and it is spliced
        into a copy.

        `self._raw_data` is never written. It is the same numpy object as
        `self._original_data`, which is the detector's input, so writing through
        it would silently change what gets scored. `.copy()` below is the entire
        reason this is safe, and tests/test_display_model_isolation.py asserts
        the property it depends on.
        """
        if not self._ica_display or self._raw_data is None:
            return self._raw_data
        if self._ica_view_span is None:
            self._ica_view_span = (0.0, min(self._duration_s,
                                            float(self._timebase_s)))
        x0, x1 = self._ica_view_span
        # A window of padding either side, so the seam between cleaned and raw
        # signal sits outside what the reviewer is looking at.
        t0 = max(0.0, x0 - ICA_SEGMENT_S)
        t1 = min(self._duration_s, x1 + ICA_SEGMENT_S)

        # Measured at 0.94 s for a first toggle at the default 10 s sweep, and
        # 0.66 s after a scroll; a cache hit is 0.04 s. That is real work — the
        # detector's own ICA, re-run per 12 s window — but a second of frozen
        # window with nothing on screen reads as a hang, and the reviewer's next
        # move is to click the checkbox again. Say what is happening, and paint
        # it BEFORE the work starts rather than after.
        self.statusBar().showMessage(
            'Running the detector\'s ICA over the visible span…')
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        # try/finally, not a restore beside each return. The copy and the splice
        # below allocate a second full-recording array and can raise on a long
        # recording; an override cursor that is never restored leaves the whole
        # application showing a spinner forever, which is indistinguishable from
        # the hang this code exists to avoid looking like.
        try:
            # The ZUNA reconstruction is a different signal under the same
            # filename, so it must not share cache entries with the baseline.
            rec_id = '{}|{}'.format(self._ica_recording_id, self._prob_source)
            try:
                span, t_start, blocks = clean_span_for_display(
                    self._raw_data, self._fs, t0, t1, recording_id=rec_id,
                    anchor_s=self._ica_anchor_s)
            except Exception:                               # noqa: BLE001
                # Never let the display-side ICA take the trace down: falling
                # back to the raw signal is always a legitimate thing to draw,
                # and the checkbox is a question the reviewer asked, not a
                # promise.
                self.chk_ica.blockSignals(True)
                self.chk_ica.setChecked(False)
                self.chk_ica.blockSignals(False)
                self._ica_display = False
                self.statusBar().showMessage(
                    'ICA view failed on this span; showing the raw trace.',
                    8000)
                return self._raw_data
            source = self._raw_data.copy()
            i0 = int(round(t_start * self._fs))
            source[:, i0:i0 + span.shape[1]] = span
            self._ica_blocks = blocks
            self.statusBar().showMessage(ica_summarise(blocks), 8000)
            return source
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_ica_display_toggled(self, on):
        """Toggle between the raw trace and what the detector actually scored.

        The comparison is the point. A reviewer judging whether a candidate is
        an eye-blink artifact is judging the cleaned signal the model saw, and
        until now had no way to see it — or to see the roughly one window in ten
        where the frozen preprocessing finds no EOG component and hands the
        model the raw signal instead.
        """
        self._ica_display = bool(on)
        self._refresh_signal_view(preserve_view=True)

    def _refresh_signal_view(self, preserve_view=True):
        """Rebuild filtered + montaged signal and push into SignalView.

        preserve_view : default True so changes to montage / filter /
            sensitivity don't scroll the reviewer back to t=0. Callers
            that open a fresh file should pass False."""
        if self._raw_data is None:
            return
        filtered = apply_filters(
            self._display_source(), self._fs,
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
        self._mark_discarded_runs()

    def _mark_discarded_runs(self):
        """Show threshold crossings that event shaping rejected as too short."""
        cache = self._prob_sources.get(self._prob_source) or {}
        if self._probs is None or not len(self._probs[0]):
            self.prob_strip.set_discarded_runs([])
            return
        try:
            import eval_config as cfg
            from gui.postprocess import runs_discarded_by_shaping
            spans = runs_discarded_by_shaping(
                self._probs[0], self._probs[1], self._threshold,
                self._segment_s, duration_s=self._duration_s,
                min_duration_s=cfg.MIN_EVENT_DURATION_S,
                max_gap_s=cfg.MAX_MERGE_GAP_S,
                average=cfg.USE_PER_SECOND_AVERAGING,
                skip_code=cache.get('skip_code'))
        except Exception:
            spans = []
        self.prob_strip.set_discarded_runs(spans)

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
            self._push_undo('accept')
            ev['status'] = 'accepted'
            self.signal_view.update_event_status(event_id, 'accepted')
            self.event_list.update_row(event_id, ev)
            self.event_list._refresh_summary(self._events)
            self._mark_dirty()

    # Why a candidate was dismissed. gui/events.py has carried a review_note
    # through threshold rebuilds since the beginning but nothing ever wrote one,
    # so a reviewed session was a binary vector. A reason turns it into evidence
    # a thesis can quote and a clinician can be asked about.
    REJECT_REASONS = ['Muscle / EMG', 'Movement or electrode artefact',
                      'Eye blink', 'Normal variant', 'Not a seizure — other',
                      'Unsure']

    def _on_reject(self, event_id, reason=None):
        """Reject a candidate. Deliberately NOT modal.

        X must stay a single instant keystroke — a reviewer works through
        dozens of candidates and a dialog on every one would make the tool
        exhausting and push them toward the mouse. Use Shift+X (or the menu)
        when a reason is worth recording.
        """
        ev = self._find(event_id)
        if not ev:
            return
        self._push_undo('reject')
        ev['status'] = 'rejected'
        if reason:
            ev['review_note'] = reason
        self.signal_view.update_event_status(event_id, 'rejected')
        self.event_list.update_row(event_id, ev)
        self.event_list._refresh_summary(self._events)
        self._mark_dirty()

    def _reject_with_reason(self, event_id=None):
        """Reject and record why. Optional, opt-in, bound to Shift+X."""
        if event_id is None:
            event_id = self.event_list.selected_event_id()
        if event_id <= 0 or not self._find(event_id):
            return
        reason, ok = QtWidgets.QInputDialog.getItem(
            self, 'Reject candidate',
            'Reason (recorded in the export provenance):',
            self.REJECT_REASONS, len(self.REJECT_REASONS) - 1, True)
        if ok:
            self._on_reject(event_id, reason=reason or None)

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
            self._push_undo('edit extent')
            s, e = clamp_interval(s, e, self._duration_s)
            ev['start'] = float(s)
            ev['stop'] = float(e)
            self.signal_view.update_event_region(event_id, s, e)
            if ev['status'] == 'proposed':
                ev['status'] = 'edited'
                self.signal_view.update_event_status(event_id, 'edited')
            self.event_list.update_row(event_id, ev)
            self.event_list._refresh_summary(self._events)
            self._mark_dirty()

    def _on_selection_change(self, event_id):
        # Only the selected event is draggable, so selection has to reach
        # the trace view as well as the viewport.
        try:
            self.signal_view.set_selected_event(event_id)
        except Exception:
            pass
        if event_id > 0:
            self._on_jump(event_id)

    # ==================================================================
    # Human-originated events, and undo
    #
    # Until this existed the worklist derived entirely from suprathreshold runs
    # and export filtered to accepted/edited, so a reviewer could only ever
    # SUBTRACT from the detector's list. With 49 % event sensitivity and 22 % of
    # seizures scoring essentially zero, a reader who saw a clear seizure the
    # model ignored had no way to record it — and exported a file asserting it
    # did not happen, under their own reviewer id. That made the tool a filter
    # rather than human-in-the-loop review, capped it at the model's
    # sensitivity, and left the thesis's primary figure (reviewer-in-the-loop
    # event recovery) unmeasurable, since "recovered" could only ever mean an
    # extent correction on something the AI had already found.
    # ==================================================================
    DEFAULT_ADDED_EVENT_S = 10.0

    def _add_event_at_view(self):
        """Create a seizure the detector did not propose. Bound to N."""
        # The precondition is an open recording, NOT the presence of
        # probabilities: the whole point is to record a seizure on a file where
        # the detector proposed nothing, which is exactly when _probs may be
        # empty or the strip flat.
        if not self._edf_path:
            self.statusBar().showMessage('Open a recording first.', 4000)
            return
        x0, x1 = self.signal_view.current_span()
        centre = 0.5 * (x0 + x1)
        half = min(self.DEFAULT_ADDED_EVENT_S, max(1.0, (x1 - x0) * 0.4)) / 2.0
        s, e = clamp_interval(centre - half, centre + half, self._duration_s)

        self._push_undo('add event')
        ev = {
            'start': float(s), 'stop': float(e),
            'source_start': float(s), 'source_stop': float(e),
            'prob': None,               # the detector never scored this
            'status': 'added',
            'review_note': None,
        }
        self._events = assign_event_ids(list(self._events) + [ev])
        self.signal_view.set_events(self._events)
        self.event_list.set_events(self._events)
        new_id = next((x['id'] for x in self._events
                       if x['status'] == 'added'
                       and abs(x['start'] - s) < 1e-6), None)
        if new_id:
            self.event_list.select_by_id(new_id)
        self._mark_dirty()
        self.statusBar().showMessage(
            'Added an event at {:.1f}–{:.1f}s. Drag its edges to adjust, or '
            'Ctrl+Z to undo.'.format(s, e), 8000)

    # ---- undo -------------------------------------------------------
    UNDO_DEPTH = 50

    def _push_undo(self, label):
        """Snapshot the event list before a change. Cheap: tens of dicts."""
        if not hasattr(self, '_undo_stack'):
            self._undo_stack = []
        self._undo_stack.append((label, [dict(e) for e in self._events]))
        del self._undo_stack[:-self.UNDO_DEPTH]

    def _undo(self):
        stack = getattr(self, '_undo_stack', None)
        if not stack:
            self.statusBar().showMessage('Nothing to undo.', 3000)
            return
        label, events = stack.pop()
        self._events = events
        self.signal_view.set_events(self._events)
        self.event_list.set_events(self._events)
        self._mark_discarded_runs()
        self._autosave_review()
        self._update_title()
        self.statusBar().showMessage('Undid: {}'.format(label), 5000)

    def _revert_extent(self, event_id=None):
        """Restore an event to the extent the detector proposed.

        source_start/source_stop have been preserved through every rebuild
        since the beginning (gui/events.py) and were never exposed, so an
        accidental drag was irreversible.
        """
        if event_id is None:
            event_id = self.event_list.selected_event_id()
        ev = self._find(event_id) if event_id > 0 else None
        if not ev:
            return
        if ev.get('source_start') is None or ev['status'] == 'added':
            self.statusBar().showMessage(
                'No detector extent to revert to.', 4000)
            return
        self._push_undo('revert extent')
        ev['start'], ev['stop'] = clamp_interval(
            ev['source_start'], ev['source_stop'], self._duration_s)
        if ev['status'] == 'edited':
            ev['status'] = 'proposed'
        self.signal_view.set_events(self._events)
        self.event_list.set_events(self._events)
        self._mark_dirty()
        self.statusBar().showMessage('Reverted to the detector extent.', 5000)

    # ==================================================================
    # Unsaved-review protection (B3)
    #
    # Nothing a reviewer does reaches disk until they choose Export, and both
    # opening another EDF and closing the window used to discard the session
    # without asking. A reviewer who has worked through 40 candidates and then
    # opens the next recording should not lose that silently.
    # ==================================================================
    def _reviewed_events(self):
        # 'added' belongs here. It was omitted, and three things depended on
        # this list: the unsaved-work prompt, the autosave payload, and the
        # session summary. A reviewer who only added seizures the detector
        # missed - the exact case this tool exists to demonstrate - had an
        # empty list, so closing the window or opening the next recording
        # discarded the work with no prompt and no autosave to recover from.
        return [ev for ev in self._events
                if ev.get('status') in EXPORTED_STATUSES + ('rejected',)]

    def _mark_dirty(self):
        self._dirty = True
        self._autosave_review()
        self._update_title()

    def _update_title(self):
        if not self._edf_path:
            self.setWindowTitle(APP_NAME + TITLE_SUFFIX)
            return
        self.setWindowTitle('{}{} — {}{}'.format(
            '*' if self._dirty else '', APP_NAME,
            os.path.basename(self._edf_path), TITLE_SUFFIX))

    def _autosave_path(self):
        if not self._edf_path:
            return None
        return os.path.splitext(self._edf_path)[0] + '.review.autosave.json'

    def _autosave_fallback_path(self):
        """Per-user autosave location for a read-only recording folder.

        Clinical recordings usually sit on a read-only share. Without this the
        autosave silently does nothing there — leaving the reviewer with no
        crash recovery on precisely the machines where it matters, and no
        indication that they are unprotected.
        """
        if not self._edf_path:
            return None
        from gui.paths import writable_root
        p = os.path.abspath(self._edf_path)
        key = hashlib.sha256(p.encode('utf-8', 'replace')).hexdigest()[:16]
        stem = os.path.splitext(os.path.basename(p))[0]
        return os.path.join(writable_root(), 'autosave',
                            '{}.{}.review.autosave.json'.format(stem, key))

    def _autosave_locations(self):
        """Sidecar first, then the per-user fallback."""
        return [p for p in (self._autosave_path(),
                            self._autosave_fallback_path()) if p]

    def _autosave_review(self):
        """Write reviewer decisions after every change.

        A crash-recovery record, not an export: it is deliberately not a
        `.csv_bi`, so it can never be mistaken for an annotation file.
        """
        if not self._edf_path:
            return
        try:
            payload = {
                'edf': self._edf_path,
                'ai_source': self._prob_source,
                'threshold': float(self._threshold),
                'events': [
                    {'start': float(ev['start']), 'stop': float(ev['stop']),
                     'status': ev['status'],
                     'review_note': ev.get('review_note'),
                     'model_score_uncalibrated': (
                         None if ev.get('prob') is None
                         else float(ev['prob']))}
                    for ev in self._reviewed_events()],
                'note': 'Autosaved reviewer decisions. Not an annotation file.',
            }
            blob = json.dumps(payload, indent=2, sort_keys=True) + '\n'
        except Exception:
            return        # autosave is best-effort; never block the reviewer

        for path in self._autosave_locations():
            try:
                parent = os.path.dirname(path)
                if parent and not os.path.isdir(parent):
                    os.makedirs(parent)
                with open(path, 'w') as f:
                    f.write(blob)
                self._autosave_written_to = path
                return
            except OSError:
                continue  # read-only location; try the next one
        self._autosave_written_to = None

    def _read_autosave(self):
        """The most recent crash-recovery record for this recording, or None.

        Both locations are considered because _autosave_review falls back to a
        per-user directory when the recording sits on a read-only share, which
        is the normal case in a hospital. The newer of the two wins: a reviewer
        whose folder permissions changed mid-session could have one of each.
        """
        best, best_mtime = None, -1.0
        for path in self._autosave_locations():
            if not path or not os.path.exists(path):
                continue
            try:
                mtime = os.path.getmtime(path)
                with open(path) as f:
                    blob = json.load(f)
            except (OSError, ValueError):
                continue        # unreadable or truncated by the crash itself
            if not isinstance(blob, dict) or not blob.get('events'):
                continue
            if mtime > best_mtime:
                best, best_mtime = blob, mtime
        return best

    def _offer_autosave_restore(self):
        """Offer to reload decisions the reviewer never exported.

        The application has always written this file, and _confirm_discard_review
        tells the reviewer where it is when they abandon a session — but nothing
        ever read it back, so the promise could not be kept. A reviewer whose
        machine died forty candidates into a recording lost the lot.

        Restoring is opt-in and non-destructive: declining leaves the file on
        disk, so a mis-click costs nothing.
        """
        blob = self._read_autosave()
        if not blob:
            return
        saved = blob.get('events') or []

        # Match on the interval, not on identity — the proposals were rebuilt
        # from the probabilities a moment ago and are different dict objects.
        # A tenth of a second is far below the 6 s window step, so it cannot
        # collide two distinct candidates, and it absorbs the float round-trip
        # through JSON.
        by_span = {}
        for i, ev in enumerate(self._events):
            by_span[(round(float(ev['start']), 1),
                     round(float(ev['stop']), 1))] = i

        restorable, added_back = [], []
        for rec in saved:
            try:
                span = (round(float(rec['start']), 1),
                        round(float(rec['stop']), 1))
            except (KeyError, TypeError, ValueError):
                continue
            if span in by_span:
                restorable.append((by_span[span], rec))
            elif rec.get('status') == 'added':
                added_back.append(rec)      # a seizure the detector never proposed
        if not restorable and not added_back:
            return

        n = len(restorable) + len(added_back)
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle('Unfinished review found')
        box.setText('{} decision(s) for this recording were autosaved and '
                    'never exported.'.format(n))
        detail = ('Restore them and continue where you left off?\n\n'
                  'Declining leaves the autosave file untouched — nothing is '
                  'lost either way.')
        if added_back:
            detail += ('\n\n{} of them are seizures you marked that the '
                       'detector did not propose.'.format(len(added_back)))
        box.setInformativeText(detail)
        box.setStandardButtons(QtWidgets.QMessageBox.Yes |
                               QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.Yes)
        if box.exec_() != QtWidgets.QMessageBox.Yes:
            return

        for idx, rec in restorable:
            ev = self._events[idx]
            ev['status'] = rec.get('status', ev['status'])
            if rec.get('review_note'):
                ev['review_note'] = rec['review_note']
        for rec in added_back:
            self._events.append({
                'start': float(rec['start']), 'stop': float(rec['stop']),
                'source_start': float(rec['start']),
                'source_stop': float(rec['stop']),
                'prob': rec.get('model_score_uncalibrated'),
                'status': 'added',
                'review_note': rec.get('review_note')})
        self._events.sort(key=lambda e: e['start'])
        # Re-issue ids: the restored events have none, and both the worklist and
        # the signal view key their selection off 'id'. assign_event_ids is the
        # same helper _add_event_at_view uses, so a restored event is
        # indistinguishable from one added by hand.
        self._events = assign_event_ids(self._events)

        self._dirty = True
        self._update_title()
        self.event_list.set_events(self._events)
        self._refresh_signal_view(preserve_view=True)
        self.statusBar().showMessage(
            'Restored {} unexported decision(s) from the autosave. Export when '
            'you are finished.'.format(n), 8000)

    def _clear_autosave(self):
        for path in self._autosave_locations():
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        self._autosave_written_to = None

    def _confirm_discard_review(self, action):
        """Ask before losing unexported decisions. True = proceed."""
        n = len(self._reviewed_events())
        if not self._dirty or not n:
            return True
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle('Unexported review')
        box.setText('{} reviewed event(s) have not been exported.'.format(n))
        # Name the file that actually exists. Telling the reviewer their work is
        # safe in a location nothing could be written to would be worse than
        # saying nothing.
        where = getattr(self, '_autosave_written_to', None)
        recovery = ('Autosaved decisions are kept at\n{}'.format(where)
                    if where else
                    'WARNING: decisions could not be autosaved anywhere. '
                    'They exist only in this window.')
        box.setInformativeText(
            'Export them before you {}?\n\n{}'.format(action, recovery))
        export = box.addButton('Export now…', QtWidgets.QMessageBox.AcceptRole)
        discard = box.addButton('Discard', QtWidgets.QMessageBox.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(export)
        box.exec_()

        clicked = box.clickedButton()
        if clicked is export:
            self._export_reviewed()
            return not self._dirty      # only proceed if the export happened
        return clicked is discard

    # ==================================================================
    # Toolbar handlers
    # ==================================================================
    def _on_montage_change(self, name):
        self._montage = name
        self.head_view.set_montage(name)
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
        # Inspectors already open when blind mode is toggled. Missing this, a
        # reviewer could turn blind mode on with a detail window showing and
        # keep seeing the reference bands in it.
        for insp in self._inspectors.values():
            insp.set_references_visible(checked)
        if hasattr(self, 'a_blind') and self.a_blind.isChecked() == checked:
            self.a_blind.blockSignals(True)
            self.a_blind.setChecked(not checked)
            self.a_blind.blockSignals(False)

    def _on_view_range(self, x0, x1):
        self._hover_lbl.setText('View: {:.1f}s – {:.1f}s  ({:.1f}s)'
                                .format(x0, x1, x1 - x0))
        # The ICA view follows the scroll, but only after the reviewer stops
        # moving: cleaning is ~1 s per 12 s window, and doing it per scroll
        # event would make the scrollbar feel broken.
        self._ica_view_span = (float(x0), float(x1))
        if self._ica_display:
            self._ica_timer.start()

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
        # The band table has to know which display filters produced this trace,
        # or it cannot say which of its numbers the filters already decided.
        # Untold, it marks every band unverified rather than guessing — safe,
        # but it withholds the one thing the panel exists to say.
        insp.set_filters(self._hp, self._lp, self._notch)
        # A detail window opened during a blind session must not show the answer
        # key the main view is hiding. Without this a session could display the
        # ground truth in the inspector and still export blind_mode: true.
        insp.set_references_visible(self.chk_refs.isChecked())
        insp.closed.connect(self._on_inspector_closed)
        insp.sync_toggle_signal().connect(lambda _s, i=idx: self._apply_inspector_sync(i))
        self._inspectors[idx] = insp
        insp.show()

    def _on_electrode_clicked(self, name):
        """Head map -> traces: open the first displayed row using this electrode.

        In a bipolar montage one electrode feeds several rows and in a
        referential montage the row is the electrode itself, so the mapping goes
        through the same derivation table the map draws from rather than
        guessing at the label.

        The status message states that the detector scored all 19 regardless.
        Clicking an electrode looks like selecting it, and the one belief this
        panel must not create is that a channel can be deselected from the
        model — the network's first kernel spans the whole electrode axis, so
        that is not offerable even in principle.
        """
        if not self._display_labels:
            return
        wanted = derivations_for(name, self._montage) or [name]
        for label in wanted:
            if label in self._display_labels:
                self._open_channel_inspector(self._display_labels.index(label))
                self.statusBar().showMessage(
                    '{} — showing {}. Display only: the detector scored all '
                    '19 electrodes.'.format(name, label), 6000)
                return
        self.statusBar().showMessage(
            '{} is not a displayed row in the {} montage.'.format(
                name, self._montage), 4000)

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
        """Push new data into open inspectors after a montage / filter change.

        This used to close every inspector and never reopen it, so the detail
        window disappeared precisely when a reviewer was changing the filter to
        decide whether something was muscle artefact. Now it updates in place;
        only an inspector whose channel no longer exists in the new montage is
        closed, because there is nothing left to show it.
        """
        if self._display_data is None:
            return
        for idx, insp in list(self._inspectors.items()):
            if idx >= len(self._display_labels):
                insp.close()                      # channel gone in this montage
                continue
            try:
                insp.set_data(self._display_labels[idx], self._display_data[idx],
                              self._display_t, self._refs)
            except Exception:
                insp.close()

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
    def _reference_path(self):
        if not self._edf_path:
            return None
        return os.path.splitext(self._edf_path)[0] + '.csv_bi'

    def _export_preflight(self):
        """Confirm what is about to be written. True = proceed.

        Export emits only accepted and edited events, so a session where the
        reviewer rejected everything writes a valid header-only file asserting
        "no seizures" — downstream indistinguishable from a careful negative
        read. The counts have to be shown before the write, not after.
        """
        n_acc = sum(1 for e in self._events if e['status'] == 'accepted')
        n_edit = sum(1 for e in self._events if e['status'] == 'edited')
        n_rej = sum(1 for e in self._events if e['status'] == 'rejected')
        n_prop = sum(1 for e in self._events if e['status'] == 'proposed')
        # Count what the writer actually writes. This was n_acc + n_edit while
        # the write filters on EXPORTED_STATUSES, which includes 'added' - so a
        # reviewer who added three missed seizures was shown "0 event(s) will be
        # written", warned that an empty file asserts no seizures, and defaulted
        # to Cancel. Three events were then written anyway.
        n_add = sum(1 for e in self._events if e['status'] == 'added')
        n_out = n_acc + n_edit + n_add

        lines = ['{} event(s) will be written: {} accepted, {} edited, '
                 '{} added by the reviewer.'.format(n_out, n_acc, n_edit, n_add)]
        if n_rej or n_prop:
            lines.append('{} rejected and {} still-unreviewed candidate(s) '
                         'will be omitted.'.format(n_rej, n_prop))
        if n_out == 0:
            lines.append('\nThis writes an EMPTY annotation file, which asserts '
                         'that the recording contains no seizures.')
        if self._unscored_count():
            lines.append('\n{} window(s) were never assessed by the detector, '
                         'so this review does not cover the whole recording.'
                         .format(self._unscored_count()))

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning if n_out == 0
                    else QtWidgets.QMessageBox.Question)
        box.setWindowTitle('Export reviewed annotations')
        box.setText('\n'.join(lines))
        box.setStandardButtons(QtWidgets.QMessageBox.Ok |
                               QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel if n_out == 0
                             else QtWidgets.QMessageBox.Ok)
        return box.exec_() == QtWidgets.QMessageBox.Ok

    def _ensure_reviewer_id(self):
        """Ask once per session who is reviewing.

        Without it an exported annotation cannot be attributed, which
        peer_review_protocol.md requires and which any session used as thesis
        evidence needs. Free text and skippable — this is a research prototype
        on public data, not an identity check.
        """
        if self._reviewer_id is not None:
            return
        try:
            name, ok = QtWidgets.QInputDialog.getText(
                self, 'Reviewer', 'Reviewer name or initials (recorded in '
                                  'the export provenance):')
        except Exception:
            name, ok = '', False
        self._reviewer_id = (name.strip() or 'anonymous') if ok else 'anonymous'

    def _export_reviewed(self):
        if not self._edf_path:
            return
        if not self._export_preflight():
            return
        self._ensure_reviewer_id()
        suffix = (
            '.zuna.reviewed.csv_bi'
            if self._prob_source == 'zuna' else '.reviewed.csv_bi')
        default = os.path.splitext(self._edf_path)[0] + suffix
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export reviewed annotations', default,
            'TUSZ csv_bi (*.csv_bi)')
        if not path:
            return

        # Hard refusal, not a warning. The reference file sits in the same
        # directory, matches the dialog's *.csv_bi filter, and is one click
        # away — and sample_data/ is gitignored, so overwriting it destroys the
        # ground truth with no recovery.
        ref = self._reference_path()
        if ref and os.path.abspath(path) == os.path.abspath(ref):
            QtWidgets.QMessageBox.critical(
                self, 'Refusing to overwrite the reference',
                'That path is the ground-truth annotation file for this '
                'recording:\n\n{}\n\nExporting over it would destroy the '
                'reference this review is scored against. Choose a different '
                'filename — the default {} is safe.'
                .format(ref, os.path.basename(default)))
            return

        events = []
        exported_scores = []
        for ev in self._events:
            if ev['status'] not in EXPORTED_STATUSES:
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
                'model_score_uncalibrated': (
                    None if ev.get('prob') is None
                    else round(float(ev['prob']), 6)),
                'human_originated': ev['status'] == 'added',
                'review_note': ev.get('review_note'),
            })
        write_csv_bi(path,
                     bname=os.path.splitext(os.path.basename(self._edf_path))[0],
                     duration_s=self._duration_s,
                     events=events,
                     ai_source=self._prob_source)
        self._write_export_provenance(path, len(events),
                                      exported_scores=exported_scores)
        # The review is now on disk, so the session is no longer at risk.
        self._dirty = False
        self._clear_autosave()
        self._update_title()
        self.statusBar().showMessage(
            'Exported {} reviewed event(s) → {}'.format(len(events), path),
            10000)

    def _write_export_provenance(self, csv_bi_path, n_events,
                                 exported_scores=None):
        """Write a provenance sidecar for ANY reviewed export (baseline or
        ZUNA), so a reviewed .csv_bi is always traceable to the exact source,
        detector settings, and display state that produced it."""
        meta_path = csv_bi_path + '.provenance.json'
        src_meta = (self._prob_sources.get(self._prob_source) or {}).get(
            'meta') or {}
        now = time.time()
        payload = {
            'annotation_file': csv_bi_path,
            'tool': csv_bi_module.TOOL_NAME,
            'tool_commit': _git_commit(),
            'exported_utc': __import__('datetime').datetime.utcfromtimestamp(
                now).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'reviewer_id': self._reviewer_id,
            'review_seconds_on_this_file': (
                round(now - self._file_opened_at, 1)
                if self._file_opened_at else None),
            'session_seconds': round(now - self._session_started, 1),
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
            'references_visible': bool(self.chk_refs.isChecked()),
            'blind_mode': bool(getattr(self, 'a_blind', None)
                               and self.a_blind.isChecked()),
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
            'all_decisions': [
                {'start': round(float(e['start']), 4),
                 'stop': round(float(e['stop']), 4),
                 'status': e['status'],
                 'model_score_uncalibrated': (
                     None if e.get('prob') is None
                     else round(float(e['prob']), 6)),
                 'review_note': e.get('review_note')}
                for e in self._events],
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
