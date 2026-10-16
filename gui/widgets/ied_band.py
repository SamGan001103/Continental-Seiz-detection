"""A thin heat band for interictal discharge density, across the recording.

Why it is its own widget
------------------------
The marks used to be drawn inside the score strip, at the foot of the
probability axis. A score curve rests near zero for most of a recording, so
the two were competing for the same few rows of pixels and the amber lost
under the red every time. They are different quantities in different units
and they were being asked to share one axis.

Giving the spikes a lane above the score fixed the collision but spent height
on it, and height on this screen belongs to the traces: the EEG is what is
actually being read, and everything else exists to help someone decide where
to look at it. A band costs about thirty pixels and no axis at all.

Why a heat band and not a bar chart
-----------------------------------
A day across nineteen hundred pixels is forty-five seconds per pixel, so a
few hundred detections drawn as ticks fill the strip evenly and say nothing.
What a reviewer wants from this is where the discharges CONCENTRATE --
whether they cluster in the small hours, whether they stop after a medication
change -- and colour carries that in one row of pixels where bar heights need
twenty.

White to amber, scaled to the busiest bin in this recording. Scaled to itself
rather than to an absolute rate because the question is where the spikes are
within this study; a fixed scale would render a quiet recording as an empty
band and tell the reviewer nothing at all.

Confirmed marks are notched on top in solid amber. A proposal is unvalidated
on this data and must never be merged into the same row as something a
clinician asserted.
"""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

#: Matches the score strip's left axis so the two line up column for column.
AXIS_WIDTH = 64
BAND_HEIGHT = 30

#: The amber the rest of the system uses for detector output.
AMBER = (178, 106, 0)
#: Solid, for marks a reviewer confirmed.
CONFIRMED = (150, 85, 0)


def density(times, duration_s, columns, smooth=True):
    """Local discharge rate per column of pixels.

    Binned to the pixels actually being painted, then SMOOTHED, and the
    smoothing is the point. Counted straight into pixel columns, a few
    hundred detections across a day land one per column and the band renders
    as scattered hairlines -- which is the picket fence this widget replaced,
    in colour. Spreading each mark over a few columns turns an isolated
    discharge into a faint blot and a run of twenty into a solid one, so what
    the eye reads is the local RATE rather than the accident of which pixel a
    mark happened to fall in.

    The kernel scales with the width, so the band looks the same on any
    monitor: about one part in two hundred and fifty of the recording, which
    on a twenty-four hour study is a few minutes.
    """
    columns = max(1, int(columns))
    duration_s = float(duration_s or 0.0)
    if duration_s <= 0:
        return np.zeros(columns)
    t = np.asarray([x for x in (times or []) if 0.0 <= x <= duration_s],
                   dtype=float)
    if t.size == 0:
        return np.zeros(columns)
    edges = np.linspace(0.0, duration_s, columns + 1)
    counts = np.histogram(t, bins=edges)[0].astype(float)
    if not smooth:
        return counts
    sigma = max(1.5, columns / 250.0)
    half = int(np.ceil(3.0 * sigma))
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # Padded by reflection so a cluster at either end of the recording is not
    # dimmed by the edge of the array.
    padded = np.pad(counts, half, mode='reflect')
    return np.convolve(padded, kernel, mode='same')[half:half + columns]


class IedBand(QtWidgets.QWidget):
    """Spike density for the whole recording, as one row of colour."""

    def __init__(self, parent=None):
        super(IedBand, self).__init__(parent)
        self._duration_s = 0.0
        self._proposed = []
        self._confirmed = []
        self.setFixedHeight(BAND_HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Fixed)
        self.setToolTip(
            'Interictal discharge density across the whole recording. '
            'Darker amber means more discharges in that span.\n\nThese are '
            'detector proposals and are UNVALIDATED on this data — '
            'candidates for a clinician to confirm or reject, not findings. '
            'Notches are marks a reviewer has confirmed.')

    # -- data ------------------------------------------------------------
    def set_marks(self, marks, duration_s=None):
        """Detector proposals and reviewer-confirmed marks, kept apart."""
        if duration_s is not None:
            self._duration_s = float(duration_s or 0.0)
        proposed, confirmed = [], []
        for m in (marks or []):
            try:
                t = float(m['time'])
            except (KeyError, TypeError, ValueError):
                continue
            (confirmed if (m.get('source') or 'reviewer') == 'reviewer'
             else proposed).append(t)
        self._proposed = proposed
        self._confirmed = confirmed
        self.update()

    def set_duration(self, duration_s):
        self._duration_s = float(duration_s or 0.0)
        self.update()

    def clear(self):
        self._proposed = []
        self._confirmed = []
        self._duration_s = 0.0
        self.update()

    def counts(self):
        return len(self._proposed), len(self._confirmed)

    # -- painting --------------------------------------------------------
    def plot_rect(self):
        """The area the heat occupies, inset to match the strip below."""
        return QtCore.QRect(AXIS_WIDTH, 0,
                            max(1, self.width() - AXIS_WIDTH), self.height())

    def paintEvent(self, event):                           # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.fillRect(self.rect(), QtGui.QColor(255, 255, 255))
            rect = self.plot_rect()
            self._paint_label(p)
            if self._duration_s <= 0:
                self._paint_frame(p, rect)
                return
            self._paint_heat(p, rect)
            self._paint_confirmed(p, rect)
            self._paint_frame(p, rect)
        finally:
            p.end()

    def _paint_label(self, p):
        p.setPen(QtGui.QColor(93, 107, 122))
        f = p.font()
        f.setPointSizeF(max(6.5, f.pointSizeF() - 1.0))
        p.setFont(f)
        p.drawText(QtCore.QRect(0, 0, AXIS_WIDTH - 6, self.height()),
                   int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter),
                   'spikes')

    def _paint_heat(self, p, rect):
        counts = density(self._proposed, self._duration_s, rect.width())
        peak = float(counts.max()) if counts.size else 0.0
        if peak <= 0:
            return
        # One column of pixels per bin, so nothing is resampled on the way to
        # the screen and a single isolated discharge is still a visible line.
        for i, c in enumerate(counts):
            if c <= 0:
                continue
            # Square-rooted so a lone discharge still registers as colour
            # rather than as a tint indistinguishable from white. The band
            # answers "where are they", not "exactly how many".
            f = float(np.sqrt(min(1.0, c / peak)))
            colour = QtGui.QColor(
                int(255 + (AMBER[0] - 255) * f),
                int(255 + (AMBER[1] - 255) * f),
                int(255 + (AMBER[2] - 255) * f))
            p.fillRect(rect.left() + i, rect.top(), 1, rect.height(), colour)

    def _paint_confirmed(self, p, rect):
        if not self._confirmed:
            return
        p.setPen(QtGui.QPen(QtGui.QColor(*CONFIRMED), 2))
        top = rect.top() + 1
        bottom = rect.top() + max(6, int(rect.height() * 0.42))
        for t in self._confirmed:
            x = rect.left() + int(rect.width() * (t / self._duration_s))
            x = min(max(x, rect.left()), rect.right())
            p.drawLine(x, top, x, bottom)

    def _paint_frame(self, p, rect):
        p.setPen(QtGui.QPen(QtGui.QColor(215, 222, 229), 1))
        p.drawRect(rect.adjusted(0, 0, -1, -1))
