"""Seizure-probability strip below the signal traces.

Shows an instantaneous max p(seizure) as a filled area, a draggable
threshold line, and tick markers for reference (ground-truth) seizures
so the reviewer sees AI output and truth side-by-side on the same time
axis."""
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np


def _as_steps(seconds, values):
    """Render a per-second series as a stepped polyline: (t, v), (t+1, v)."""
    seconds = np.asarray(seconds, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    n = min(seconds.size, values.size)
    if n == 0:
        return [], []
    xs = np.empty(2 * n, dtype=np.float32)
    ys = np.empty(2 * n, dtype=np.float32)
    xs[0::2] = seconds[:n]
    xs[1::2] = seconds[:n] + 1.0
    ys[0::2] = values[:n]
    ys[1::2] = values[:n]
    return xs, ys


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
            'Solid red = the DECISION curve: the per-second series the '
            'threshold line is compared against, so a crossing always '
            'corresponds to an event in the list.\n'
            'Dotted blue = peak evidence (strongest single window covering '
            'each second). Informative, but NOT what the threshold decides on.\n'
            'Grey bands = windows the pipeline could not assess (interrupted '
            'signal or failed ICA). They carry no score and must not be read '
            'as "no seizure".\n'
            'All values are raw, UNCALIBRATED network outputs, not calibrated '
            'probabilities of seizure.')
        pi.getAxis('left').setWidth(64)
        pi.setYRange(0, 1)
        pi.setMouseEnabled(x=True, y=False)
        pi.showGrid(x=True, y=True, alpha=0.2)
        self._pw.setMaximumHeight(140)
        v.addWidget(self._pw)

        # The FILLED curve is the decision series — the one the threshold line
        # below is actually compared against, so that a visible crossing always
        # corresponds to an event in the worklist.
        self._curve = pi.plot(
            [], [],
            pen=pg.mkPen((180, 40, 40), width=1.4),
            fillLevel=0.0,
            brush=pg.mkBrush(180, 40, 40, 70),
        )
        # Peak per-second evidence, drawn faintly behind it. Informative — it is
        # the strongest single window covering each second — but it is NOT what
        # the threshold decides on, so it must never be the curve under the line.
        self._peak_curve = pi.plot(
            [], [],
            pen=pg.mkPen((150, 150, 190), width=1.0,
                         style=QtCore.Qt.DotLine))
        self._peak_curve.setZValue(-1)
        self._thr_line = pg.InfiniteLine(
            pos=0.5, angle=0,
            pen=pg.mkPen((30, 120, 30), width=1, style=QtCore.Qt.DashLine))
        pi.addItem(self._thr_line)
        self._ref_items = []
        self._refs_visible = True
        self._unscored_items = []
        self._discarded_items = []
        try:
            import eval_config as cfg
            self._min_duration_s = float(cfg.MIN_EVENT_DURATION_S)
        except Exception:
            self._min_duration_s = 5.0

    def plot_item(self):
        return self._pw.getPlotItem()

    def set_probs(self, window_starts, probs, segment_s=12, skip_code=None,
                  duration_s=None, average=None):
        """Draw the decision curve, with peak evidence behind it.

        average : whether events are built from per-second averaging. Defaults
            to eval_config.USE_PER_SECOND_AVERAGING so the drawn curve tracks
            the configured decision stage rather than a fixed choice.
        """
        from gui.postprocess import decision_curve
        if average is None:
            try:
                import eval_config as cfg
                average = bool(cfg.USE_PER_SECOND_AVERAGING)
            except Exception:
                average = True

        secs, values = decision_curve(
            window_starts, probs, segment_s, duration_s=duration_s,
            average=average, skip_code=skip_code)
        self._curve.setData(*_as_steps(secs, values))

        # Peak trace only adds information when it differs from the decision.
        px, py = _instant_max_probs(window_starts, probs, segment_s)
        self._peak_curve.setData(px if average else [], py if average else [])

        self._set_unscored(
            _unscored_spans(window_starts, skip_code, segment_s))

    def set_discarded_runs(self, spans):
        """Mark threshold crossings that event shaping rejected as too short.

        Without this the reviewer sees the curve cross the line with nothing in
        the worklist and no reason given.
        """
        pi = self._pw.getPlotItem()
        for item in self._discarded_items:
            pi.removeItem(item)
        self._discarded_items = []
        for s, e in spans or []:
            band = pg.LinearRegionItem(
                values=(float(s), float(e)),
                brush=pg.mkBrush(230, 160, 30, 70),
                pen=pg.mkPen(200, 130, 0, 170, width=1,
                             style=QtCore.Qt.DashLine),
                movable=False,
            )
            band.setZValue(-6)
            band.setToolTip(
                'Above threshold but shorter than the {:g} s minimum event '
                'duration — not proposed as an event.'.format(
                    self._min_duration_s))
            pi.addItem(band)
            self._discarded_items.append(band)

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
