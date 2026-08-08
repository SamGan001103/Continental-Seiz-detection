"""Seizure-probability strip below the signal traces.

Shows an instantaneous max p(seizure) as a filled area, a draggable
threshold line, and tick markers for reference (ground-truth) seizures
so the reviewer sees AI output and truth side-by-side on the same time
axis."""
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np


def _unscored_spans(window_starts, skip_code, segment_s):
    """Merge windows the pipeline declined to score into [start, stop] spans.

    These must be drawn as absent rather than as zero. A window carrying no
    model score is not evidence of "no seizure" — one recording in the sample
    corpus has every window rejected by the artifact check and contains a real
    27-second seizure, which previously rendered as a flat, confident-looking
    zero line.
    """
    if skip_code is None or len(window_starts) == 0:
        return []
    starts = np.asarray(window_starts, dtype=np.float64)
    codes = np.asarray(skip_code)
    if codes.shape != starts.shape:
        return []
    spans = []
    for t, c in zip(starts, codes):
        if not c:
            continue
        s0, s1 = float(t), float(t) + float(segment_s)
        if spans and s0 <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], s1)
        else:
            spans.append([s0, s1])
    return [(a, b) for a, b in spans]


def _instant_max_probs(window_starts, probs, segment_s):
    """Collapse overlapping windows to a per-second max p(seiz) curve.

    Inference windows overlap (stride < segment_s), so feeding the raw
    per-window values into a line plot produces zigzags. At each second
    we take the max probability of any window that covers that second.
    Returns (xs, ys) for a stepped plot — two points per bin, the second
    pair one second later."""
    if len(window_starts) == 0:
        return [], []
    starts = np.asarray(window_starts, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float32)
    t_min = int(starts.min())
    t_max = int(starts.max() + segment_s)
    per_sec = np.zeros(t_max - t_min, dtype=np.float32)
    for t, p in zip(starts, probs):
        s0 = int(t - t_min)
        s1 = min(len(per_sec), s0 + segment_s)
        np.maximum(per_sec[s0:s1], float(p), out=per_sec[s0:s1])
    # Stepped: (t, p), (t+1, p) for each bin.
    n = len(per_sec)
    xs = np.empty(2 * n, dtype=np.float32)
    ys = np.empty(2 * n, dtype=np.float32)
    xs[0::2] = np.arange(n, dtype=np.float32) + t_min
    xs[1::2] = xs[0::2] + 1
    ys[0::2] = per_sec
    ys[1::2] = per_sec
    return xs, ys


class ProbStrip(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        self._pw = pg.PlotWidget(background='w')
        pi = self._pw.getPlotItem()
        pi.setLabel('bottom', 'time (s)')
        # Not "p(seiz)": this is a raw softmax output with no post-hoc
        # calibration, so it does not carry the frequentist meaning a
        # probability label implies. Modern networks are systematically
        # over-confident, and an uncalibrated strip labelled as a probability
        # misleads exactly the reviewer it is meant to help.
        pi.setLabel('left', 'model score')
        self.setToolTip(
            'Model score per 12 s window — a raw, UNCALIBRATED network output, '
            'not a calibrated probability of seizure. Grey bands are windows '
            'the pipeline could not assess (interrupted signal or failed ICA); '
            'they carry no score and must not be read as "no seizure".')
        pi.getAxis('left').setWidth(64)
        pi.setYRange(0, 1)
        pi.setMouseEnabled(x=True, y=False)
        pi.showGrid(x=True, y=True, alpha=0.2)
        self._pw.setMaximumHeight(140)
        v.addWidget(self._pw)

        # Filled area under the probability curve → reads at a glance.
        self._curve = pi.plot(
            [], [],
            pen=pg.mkPen((180, 40, 40), width=1.2),
            fillLevel=0.0,
            brush=pg.mkBrush(180, 40, 40, 70),
        )
        self._thr_line = pg.InfiniteLine(
            pos=0.5, angle=0,
            pen=pg.mkPen((30, 120, 30), width=1, style=QtCore.Qt.DashLine))
        pi.addItem(self._thr_line)
        self._ref_items = []
        self._refs_visible = True
        self._unscored_items = []

    def plot_item(self):
        return self._pw.getPlotItem()

    def set_probs(self, window_starts, probs, segment_s=12, skip_code=None):
        xs, ys = _instant_max_probs(window_starts, probs, segment_s)
        self._curve.setData(xs, ys)
        self._set_unscored(
            _unscored_spans(window_starts, skip_code, segment_s))

    def _set_unscored(self, spans):
        """Draw windows with no model score as grey 'not assessed' bands."""
        pi = self._pw.getPlotItem()
        for item in self._unscored_items:
            pi.removeItem(item)
        self._unscored_items = []
        for s, e in spans:
            band = pg.LinearRegionItem(
                values=(float(s), float(e)),
                brush=pg.mkBrush(130, 130, 130, 110),
                pen=pg.mkPen(90, 90, 90, 160, width=1),
                movable=False,
            )
            band.setZValue(-5)
            band.setToolTip('Not assessed — no model score for this window')
            pi.addItem(band)
            self._unscored_items.append(band)

    def set_threshold(self, thr):
        self._thr_line.setValue(float(thr))

    def set_references(self, ref_intervals):
        pi = self._pw.getPlotItem()
        for r in self._ref_items:
            pi.removeItem(r)
        self._ref_items = []
        for s, e in ref_intervals:
            r = pg.LinearRegionItem(
                values=(float(s), float(e)),
                brush=pg.mkBrush(50, 160, 70, 50),
                pen=pg.mkPen(50, 160, 70, 140, width=1),
                movable=False,
            )
            r.setZValue(-10)
            r.setVisible(self._refs_visible)
            pi.addItem(r)
            self._ref_items.append(r)

    def set_references_visible(self, visible):
        self._refs_visible = bool(visible)
        for r in self._ref_items:
            r.setVisible(self._refs_visible)
