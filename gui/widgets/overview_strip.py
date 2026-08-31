"""The whole recording at a glance, with a draggable marker for where you are.

Why this exists
---------------
`ProbStrip` is x-linked to `SignalView` (gui/app.py: `link_x_to`), so it shows
the same span as the traces — ten seconds, by default. That is the right
behaviour for it: a strip aligned under the page lets a reviewer read the score
against the signal they are looking at.

The consequence is that nothing in the application ever showed the recording as
a whole. On a 30-minute file that is survivable. On the 24-hour ambulatory
recordings this tool is aimed at, a reviewer at 10 s a page has 8,640 pages, no
map, and no way to see that the detector fired six times in the small hours —
and `docs/INTENDED_USE.md` is explicit that "a recording with no proposed event
has NOT been shown to be seizure-free", which asks the reviewer to look at
signal the detector said nothing about. Without an overview that instruction
cannot be followed except by scrolling blind.

So this widget is the map: the full duration, always, whatever the page shows.

Design decisions that matter
----------------------------
**Per-pixel maximum, never a mean.** At 24 hours a 1600-pixel strip covers 54
seconds per pixel. A 27-second seizure averaged against 27 seconds of quiet
background halves its height and can drop under the threshold line, so the one
mark the reviewer most needs to see is the one the arithmetic erases. Taking
the maximum keeps a single high second visible at any zoom. It also means the
overview OVERSTATES: a pixel at 0.9 says "something in these 54 seconds reached
0.9", not "these 54 seconds were seizure". The axis label says so.

**Unscored windows are drawn by the same rule.** A window the pipeline refused
(interrupted signal, failed ICA) carries no score, and at overview scale a
single refused window is a fraction of a pixel. Painting it at least one pixel
wide keeps "not assessed" from disappearing into "nothing here" — the exact
confusion `ProbStrip`'s grey bands exist to prevent.

**The viewport marker is draggable and the strip is clickable.** An overview a
reviewer can only look at makes them navigate somewhere else; one they can
drive is the navigation. Clicking seeks, dragging scrubs.

Display-only, like everything else on this side of the application: it reads
the same scores `ProbStrip` reads and cannot affect what the detector produced.
"""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets


class OverviewStrip(QtWidgets.QWidget):
    """Whole-recording score map. Emits `seek_requested(centre_seconds)`."""

    seek_requested = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self._pw = pg.PlotWidget(background='w')
        pi = self._pw.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        self._pw.setMenuEnabled(False)
        # Fixed to the whole recording, so mouse interaction must not pan or
        # zoom it — the whole point is that this one does not move.
        pi.setMouseEnabled(x=False, y=False)
        pi.setYRange(0, 1)
        # Explicit ticks: pyqtgraph picks a density from the widget height, so
        # the strip's scale changed as it was resized. The threshold sits at
        # 0.5 and a reviewer reads heights against it, so those gradations are
        # fixed rather than left to the layout.
        pi.getAxis('left').setTicks(
            [[(level, '{:g}'.format(level))
              for level in (0.0, 0.25, 0.5, 0.75, 1.0)]])
        pi.getAxis('left').setWidth(64)
        pi.setLabel('left', 'max score')
        pi.setLabel('bottom', 'whole recording (s)')
        pi.showGrid(x=True, y=True, alpha=0.15)
        # Taller than it was: the page-linked strip above it is hidden by
        # default now, and at 96 px a decision curve peaking around 0.3 was
        # only ~30 px of ink — legible, but not something to scan a recording
        # with. This is the widget a reviewer uses to decide where to look.
        self._pw.setMaximumHeight(260)
        self._pw.setMinimumHeight(120)
        self.setToolTip(
            'The whole recording, however far the page above is zoomed in.\n'
            'Each pixel shows the HIGHEST score in the seconds it covers, not '
            'the average — so a brief event stays visible at this scale, and a '
            'tall pixel means "something in here reached that", not "all of '
            'this did".\n'
            'Grey = windows the pipeline could not assess. Green = reference '
            'seizures. The blue box is the span on screen above: drag it, or '
            'click anywhere, to move there.')
        v.addWidget(self._pw)

        self._curve = pi.plot(
            [], [], pen=pg.mkPen((180, 40, 40), width=1.0),
            fillLevel=0.0, brush=pg.mkBrush(180, 40, 40, 60))
        self._thr_line = pg.InfiniteLine(
            pos=0.5, angle=0,
            pen=pg.mkPen((30, 120, 30), width=1, style=QtCore.Qt.DashLine))
        pi.addItem(self._thr_line)

        # "You are here". movable, so the reviewer can scrub the day.
        self._viewport = pg.LinearRegionItem(
            values=(0.0, 0.0), brush=pg.mkBrush(40, 90, 200, 55),
            pen=pg.mkPen((40, 90, 200), width=1), movable=True)
        self._viewport.setZValue(20)
        pi.addItem(self._viewport)
        self._viewport.sigRegionChangeFinished.connect(self._on_marker_dragged)

        self._ref_items = []
        self._unscored_items = []
        self._refs_visible = True
        self._duration_s = 0.0
        self._suppress = False
        self._pw.scene().sigMouseClicked.connect(self._on_click)

    # -- data ------------------------------------------------------------
    def plot_item(self):
        return self._pw.getPlotItem()

    def set_probs(self, window_starts, probs, segment_s=12, skip_code=None,
                  duration_s=None, average=None, target_px=1600):
        """Draw the decision curve reduced to one point per pixel by MAX."""
        from gui.postprocess import decision_curve
        if average is None:
            try:
                import eval_config as cfg
                average = bool(cfg.USE_PER_SECOND_AVERAGING)
            except Exception:                                # noqa: BLE001
                average = False
        starts = np.asarray(window_starts)
        pr = np.asarray(probs, dtype=np.float64)
        if starts.size == 0:
            self._curve.setData([], [])
            self._set_unscored([])
            return
        dur = float(duration_s if duration_s is not None
                    else starts.max() + segment_s)
        self._duration_s = dur
        try:
            t_s, y_s = decision_curve(starts, pr, segment_s,
                                      duration_s=dur, average=average)
            t_s = np.asarray(t_s, dtype=np.float64)
            y_s = np.asarray(y_s, dtype=np.float64)
        except Exception:                                    # noqa: BLE001
            t_s = starts.astype(np.float64)
            y_s = pr
        if t_s.size == 0:
            self._curve.setData([], [])
        else:
            tx, yx = _reduce_by_max(t_s, y_s, dur, target_px)
            self._curve.setData(tx, yx)
        self._pw.getPlotItem().setXRange(0.0, max(dur, 1e-6), padding=0)

        spans = []
        if skip_code is not None:
            sk = np.asarray(skip_code)
            for i, code in enumerate(sk):
                if code != 0 and i < starts.size:
                    spans.append((float(starts[i]),
                                  float(starts[i]) + float(segment_s)))
        self._set_unscored(_merge_spans(spans))

    def _set_unscored(self, spans):
        pi = self._pw.getPlotItem()
        for it in self._unscored_items:
            pi.removeItem(it)
        self._unscored_items = []
        if not spans or self._duration_s <= 0:
            return
        # One pixel is the floor: a refused window narrower than a pixel would
        # otherwise vanish, and "not assessed" reading as "nothing here" is the
        # misreading this band exists to prevent.
        min_w = self._duration_s / 1600.0
        for s, e in spans:
            if (e - s) < min_w:
                mid = 0.5 * (s + e)
                s, e = mid - min_w / 2.0, mid + min_w / 2.0
            it = pg.LinearRegionItem(values=(s, e), movable=False,
                                     brush=pg.mkBrush(150, 150, 150, 90))
            it.setZValue(-12)
            for line in it.lines:
                line.setPen(pg.mkPen(None))
            pi.addItem(it)
            self._unscored_items.append(it)

    def set_references(self, ref_intervals):
        pi = self._pw.getPlotItem()
        for it in self._ref_items:
            pi.removeItem(it)
        self._ref_items = []
        if not self._refs_visible or self._duration_s <= 0:
            return
        min_w = self._duration_s / 1600.0
        for s, e in (ref_intervals or []):
            s, e = float(s), float(e)
            if (e - s) < min_w:
                mid = 0.5 * (s + e)
                s, e = mid - min_w / 2.0, mid + min_w / 2.0
            it = pg.LinearRegionItem(values=(s, e), movable=False,
                                     brush=pg.mkBrush(40, 160, 60, 70))
            it.setZValue(-10)
            for line in it.lines:
                line.setPen(pg.mkPen(None))
            pi.addItem(it)
            self._ref_items.append(it)

    def set_references_visible(self, visible):
        """Blind mode must reach this strip too, or the answer key survives
        here after being hidden everywhere else."""
        self._refs_visible = bool(visible)
        if not self._refs_visible:
            pi = self._pw.getPlotItem()
            for it in self._ref_items:
                pi.removeItem(it)
            self._ref_items = []

    def set_threshold(self, thr):
        self._thr_line.setValue(float(thr))

    def clear(self):
        self._curve.setData([], [])
        self._set_unscored([])
        self.set_references([])
        self._duration_s = 0.0
        self._set_marker(0.0, 0.0)

    # -- the viewport marker --------------------------------------------
    def set_viewport(self, x0, x1):
        """Show where the page above is. Called from the main window."""
        self._set_marker(float(x0), float(x1))

    def _set_marker(self, x0, x1):
        # Guarded: setRegion emits sigRegionChangeFinished, which would echo
        # back as a seek and fight the view that just moved.
        self._suppress = True
        try:
            self._viewport.setRegion((x0, x1))
        finally:
            self._suppress = False

    def _on_marker_dragged(self):
        if self._suppress:
            return
        x0, x1 = self._viewport.getRegion()
        self.seek_requested.emit(0.5 * (float(x0) + float(x1)))

    def _on_click(self, ev):
        if self._duration_s <= 0:
            return
        try:
            if ev.button() != QtCore.Qt.LeftButton:
                return
            pi = self._pw.getPlotItem()
            pt = pi.vb.mapSceneToView(ev.scenePos())
            x = float(pt.x())
        except Exception:                                    # noqa: BLE001
            return
        if 0.0 <= x <= self._duration_s:
            self.seek_requested.emit(x)


def _reduce_by_max(t, y, duration_s, target_px):
    """One point per pixel, taking the maximum in each bucket.

    Max rather than mean, and the reason is clinical: at 24 hours a pixel spans
    ~54 s, and averaging a 27 s seizure against quiet background halves it —
    potentially below the threshold line, hiding the one mark that matters.
    """
    n = int(t.size)
    px = max(1, int(target_px))
    if n <= px * 2 or duration_s <= 0:
        return t, y
    edges = np.linspace(0.0, duration_s, px + 1)
    idx = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, px - 1)
    out = np.zeros(px, dtype=np.float64)
    np.maximum.at(out, idx, y)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, out


def _merge_spans(spans, gap=0.0):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]
