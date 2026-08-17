"""Top-down 10-20 head schematic showing where the electrodes are.

A reviewer reading a bipolar montage is reading spatial information encoded as
row labels: 'F7-T3' is a left anterior-temporal derivation, but only if you
already carry the 10-20 map in your head. Registrars and non-neurophysiologist
clinicians frequently do not, and the thesis GUI is aimed at exactly that
reader. This widget makes the map explicit — it draws the head, the electrodes,
and the derivation the active montage is actually forming — so a suspicious
trace can be located on the scalp without translating labels mentally.

Two things this widget must never imply:

  * that the detector reads the montage. It does not. The ConvLSTM is fed the
    referential 19-channel signal via the frozen model path; montage and
    filters live entirely on the display path. Hence the permanent caption.
  * that selecting an electrode restricts the detector's input. It restricts
    only what is drawn. Same reason.

Geometry note: HEAD_POS_1020 was generated ONCE, offline, from
mne.channels.make_standard_montage('standard_1020') and pasted in as a
constant. It is not recomputed at runtime and mne is deliberately not imported
here — Montage.get_positions() exists in mne 1.12 (modern stack) but not in
mne 0.19.2 (legacy seiz36 stack), so a runtime call would break one of the two
interpreters this app has to run on, and would drag mne into the PyInstaller
bundle for the sake of 19 constants.
"""
import math

from PyQt5 import QtCore, QtGui, QtWidgets

from gui.processing import MONTAGES


# ---------------------------------------------------------------- geometry
#
# Azimuthal-equidistant projection of the standard_1020 template: 2-D radius
# proportional to the polar angle from the vertex, azimuth preserved. That is
# the same projection clinical topographic maps use, so the picture matches
# what a neurophysiologist expects rather than being a stylised approximation.
#
# The sphere centre is a least-squares fit over the whole template, not the
# head-coordinate origin: the origin sits below and behind the electrode cap,
# which pushes the temporal chain outside the unit circle and shifts Cz.
# Scaled so the outermost electrode (F8) sits at r = 0.92, leaving the visible
# margin between the last electrode ring and the scalp outline that a drawn
# 10-20 diagram has.
#
# Axes are screen-oriented for a view from ABOVE with the nose UP:
# +x = the subject's right, +y = anterior. Fp1 therefore has x < 0 and is drawn
# on the left of the figure, which is the radiological-free convention every
# EEG head diagram uses.
#
# Names are the legacy TUH set (T3/T4/T5/T6) used by gui/io/edf.py CHANNELS_19
# and gui/processing.py CH_NAMES_19, mapped from the modern 10-20 names
# (T7/T8/P7/P8) that mne's template actually carries. The order matches
# CH_NAMES_19 so an index into either is an index into the other.
HEAD_POS_1020 = (
    ('Fp1', -0.2555, +0.8513),
    ('Fp2', +0.2457, +0.8545),
    ('F7',  -0.7057, +0.5867),
    ('F3',  -0.3546, +0.4846),
    ('Fz',  -0.0024, +0.4545),
    ('F4',  +0.3571, +0.4948),
    ('F8',  +0.7035, +0.5929),
    ('T3',  -0.9084, +0.0055),
    ('C3',  -0.4273, +0.0317),
    ('Cz',  -0.0017, +0.0393),
    ('C4',  +0.4317, +0.0366),
    ('T4',  +0.9094, +0.0163),
    ('T5',  -0.6821, -0.5307),
    ('P3',  -0.3420, -0.3963),
    ('Pz',  -0.0021, -0.3565),
    ('P4',  +0.3467, -0.3913),
    ('T6',  +0.6812, -0.5323),
    ('O1',  -0.2405, -0.7658),
    ('O2',  +0.2338, -0.7677),
)

#: Channel names in canonical order — identical to gui.processing.CH_NAMES_19.
CHANNEL_NAMES = tuple(n for n, _x, _y in HEAD_POS_1020)

#: name -> (x, y) in head-model units, scalp outline at radius 1.0.
HEAD_POS = dict((n, (x, y)) for n, x, y in HEAD_POS_1020)

# Modern 10-20 label -> the legacy TUH label this app uses. Kept as a public
# constant because the same rename bites anywhere a modern montage file meets
# this codebase, and because it documents that T3 and T7 are one electrode.
LEGACY_ALIASES = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}

# Drawing extents in head-model units, used to size the figure so the nose and
# ears are never clipped: they stick out past the scalp circle.
_EXT_X = 1.20
_EXT_TOP = 1.20
_EXT_BOT = 1.06

# Breathing room reserved below the caption block, in pixels.
_CAP_PAD = 5


def montage_pairs(montage_name):
    """Return the derivations of a montage as [(label, anode, cathode), ...].

    Common Average and Referential have no pairwise derivation to draw, so they
    return an empty list rather than raising — the widget asks for this on
    every repaint and an unknown montage name must degrade to "just show the
    electrodes", not to a traceback in paintEvent.
    """
    spec = MONTAGES.get(montage_name)
    if not isinstance(spec, (list, tuple)):
        return []
    return list(spec)


def derivations_for(channel, montage_name):
    """Labels in `montage_name` that this electrode contributes to.

    Used to answer "the reviewer clicked T3 — which displayed rows involve
    T3?", which in a bipolar montage is several rows, not one.
    """
    out = []
    for label, a, b in montage_pairs(montage_name):
        if channel == a or channel == b:
            out.append(label)
    return out


# ---------------------------------------------------------------- colour


# Fixed hue for the optional value ramp. Deliberately NOT taken from the
# palette: a data encoding has to mean the same thing in a screenshot pasted
# into a report as it does on screen, so it cannot drift with the user's theme.
# Only the ramp's *direction* follows the theme (see _value_color), so that
# lightness always moves away from the page background and low values never
# vanish into it.
_RAMP_HUE = 205.0 / 360.0


def _mix(a, b, t):
    """Linear blend of two QColors, t=0 -> a, t=1 -> b."""
    t = min(1.0, max(0.0, float(t)))
    return QtGui.QColor(
        int(round(a.red() * (1 - t) + b.red() * t)),
        int(round(a.green() * (1 - t) + b.green() * t)),
        int(round(a.blue() * (1 - t) + b.blue() * t)))


def _value_color(t, dark):
    """Single-hue sequential ramp for the optional per-electrode shading.

    Saturation and value both move monotonically, so the ramp stays ordered
    when printed in greyscale and for a red-green colour-blind reader — a
    rainbow ramp would satisfy neither. Direction flips with the theme because
    on a dark canvas a pale-to-deep ramp puts its low end at maximum contrast,
    which reads as "high" to the eye.
    """
    t = min(1.0, max(0.0, float(t)))
    if dark:
        # The floor is 0.36, not 0.0: a typical dark Qt canvas sits near 0.13,
        # so a ramp that started at black would make the bottom of the scale
        # indistinguishable from an unshaded electrode.
        return QtGui.QColor.fromHsvF(_RAMP_HUE, 0.78 - 0.42 * t,
                                     0.36 + 0.62 * t)
    return QtGui.QColor.fromHsvF(_RAMP_HUE, 0.10 + 0.82 * t,
                                 0.99 - 0.42 * t)


def _readable_on(color):
    """Near-black or near-white, whichever is legible on `color`."""
    if color.lightness() > 140:
        return QtGui.QColor(25, 25, 25)
    return QtGui.QColor(245, 245, 245)


class HeadView(QtWidgets.QWidget):
    """Clickable 10-20 head map of the active display montage.

    Emits electrodeClicked(name) with the legacy TUH channel name. The caller
    decides what that means; this widget never changes what is scored.
    """

    electrodeClicked = QtCore.pyqtSignal(str)

    #: Permanent, non-dismissable. The whole safety argument for letting a
    #: reviewer poke at individual electrodes rests on it being visible.
    CAPTION = ('All 19 electrodes were scored by the detector, always. '
               'Selecting one changes only what is displayed.')

    def __init__(self, parent=None):
        super().__init__(parent)
        self._montage = 'Referential (raw)'
        self._values = None            # name -> float, or None for no shading
        self._vmin = None
        self._vmax = None
        self._unit = ''
        self._hover = None             # channel name under the cursor
        self._selected = None          # channel name last clicked
        self.setMouseTracking(True)    # hover highlight needs move events
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setMinimumSize(180, 200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.setToolTip(
            'Electrode positions in the international 10-20 system, viewed '
            'from above with the nose at the top and the subject\'s left on '
            'the left.\nArrows show the derivation the display montage forms: '
            'the trace is (tail electrode) minus (head electrode).\n'
            'Click an electrode to select it. This changes the display only — '
            'the detector always used all 19.')

    # -------------------------------------------------------------- API
    def set_montage(self, montage_name):
        """Draw the derivations of `montage_name`, a key of MONTAGES."""
        if montage_name == self._montage:
            return
        self._montage = montage_name
        self.update()

    def montage(self):
        return self._montage

    def set_values(self, values, vmin=None, vmax=None, unit=''):
        """Shade electrodes by a per-channel scalar, or pass None to clear.

        `values` maps channel name -> float (band power, per-channel model
        attribution, whatever a later feature needs). Leave it None and the
        widget is a plain montage diagram, which is its default job. Channels
        absent from the dict, or carrying a non-finite value, are drawn
        unshaded rather than as zero — "not measured" and "measured as zero"
        must not look the same.
        """
        if values is None:
            self._values = None
        else:
            clean = {}
            for name, v in values.items():
                name = LEGACY_ALIASES.get(name, name)
                if name not in HEAD_POS:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    continue
                clean[name] = fv
            self._values = clean if clean else None
        self._vmin = vmin
        self._vmax = vmax
        self._unit = unit or ''
        self.update()

    def values_range(self):
        """(vmin, vmax) actually used for shading, or None when unshaded."""
        if not self._values:
            return None
        vmin = self._vmin
        vmax = self._vmax
        if vmin is None:
            vmin = min(self._values.values())
        if vmax is None:
            vmax = max(self._values.values())
        return float(vmin), float(vmax)

    def set_selected(self, channel):
        """Mark an electrode as selected, or None to clear. Does not emit."""
        if channel is not None:
            channel = LEGACY_ALIASES.get(channel, channel)
            if channel not in HEAD_POS:
                channel = None
        if channel != self._selected:
            self._selected = channel
            self.update()

    def selected(self):
        return self._selected

    def caption_text(self):
        """The permanent caption, exposed so a test can pin that it exists."""
        return self.CAPTION

    def electrode_at(self, point):
        """Channel name under a widget-coordinate QPoint/QPointF, or None.

        Works without a prior paint — the layout is derived from the current
        size, not cached during paintEvent — so hit-testing is testable
        headless and cannot go stale after a resize.
        """
        cx, cy, radius = self._layout()
        if radius <= 0:
            return None
        hit = self._marker_radius(radius) + 4.0
        px, py = float(point.x()), float(point.y())
        best, best_d2 = None, hit * hit
        for name, mx, my in HEAD_POS_1020:
            dx = (cx + mx * radius) - px
            dy = (cy - my * radius) - py
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best, best_d2 = name, d2
        return best

    def sizeHint(self):
        return QtCore.QSize(280, 320)

    # -------------------------------------------------------------- events
    def mouseMoveEvent(self, ev):
        name = self.electrode_at(ev.pos())
        if name != self._hover:
            self._hover = name
            if name is None:
                self.unsetCursor()
            else:
                self.setCursor(QtCore.Qt.PointingHandCursor)
            self.update()
        super().mouseMoveEvent(ev)

    def leaveEvent(self, ev):
        if self._hover is not None:
            self._hover = None
            self.unsetCursor()
            self.update()
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            name = self.electrode_at(ev.pos())
            if name is not None:
                self._selected = name
                self.update()
                self.electrodeClicked.emit(name)
                ev.accept()
                return
        super().mousePressEvent(ev)

    # -------------------------------------------------------------- layout
    def _caption_height(self, width):
        fm = QtGui.QFontMetrics(self.font())
        flags = QtCore.Qt.TextWordWrap | QtCore.Qt.AlignHCenter
        box = fm.boundingRect(QtCore.QRect(0, 0, max(40, width - 8), 1000),
                              flags, self.CAPTION)
        # +6, not +0: QFontMetrics under-reports a word-wrapped block by a
        # pixel or two per line and antialiased descenders spill past that
        # again, so a tight reservation clips the tail of the last line — on
        # the one string in this widget that is not allowed to be unreadable.
        # _CAP_PAD on top of that is margin BELOW the block, kept out of the
        # draw rect, so the caption never renders flush to the widget edge.
        h = box.height() + 6 + _CAP_PAD
        if self._values:
            h += fm.height() + 12          # legend bar + its labels
        return h

    def _reserved_caption_height(self):
        """Pixels the caption strip gets, whatever else has to give way.

        The caption is reserved space, not leftover space: when the widget is
        squeezed the head shrinks and the caption survives. It is a safety
        statement, so it must not be the thing that gets clipped. The 45 % cap
        stops the reverse failure — a widget dragged so narrow that the caption
        wraps to eight lines and leaves no head at all.

        _layout() and _draw_caption() must agree on this to the pixel, so they
        share one implementation rather than each recomputing it.
        """
        r = self.contentsRect()
        return min(self._caption_height(r.width()),
                   max(0.0, r.height() * 0.45))

    def _layout(self):
        """(cx, cy, radius_px) of the scalp circle in widget coordinates."""
        r = self.contentsRect()
        fm = QtGui.QFontMetrics(self.font())
        head_top = r.top() + fm.height() + 6           # header line
        cap_h = self._reserved_caption_height()
        head_h = max(10.0, (r.bottom() - cap_h - 4) - head_top)
        head_w = max(10.0, float(r.width()))
        radius = min(head_w / (2.0 * _EXT_X), head_h / (_EXT_TOP + _EXT_BOT))
        cx = r.left() + head_w / 2.0
        # Centre the figure in whichever direction is not the binding one. A
        # dock is usually taller than it is wide, so the head is width-limited
        # and would otherwise sit jammed against the header with a dead band
        # above the caption.
        slack = head_h - radius * (_EXT_TOP + _EXT_BOT)
        cy = head_top + max(0.0, slack) / 2.0 + radius * _EXT_TOP
        return cx, cy, radius

    @staticmethod
    def _marker_radius(radius):
        return max(5.0, 0.088 * radius)

    @staticmethod
    def _pt(cx, cy, radius, x, y):
        """Head-model (x, y) -> widget QPointF. Model +y is anterior = up."""
        return QtCore.QPointF(cx + x * radius, cy - y * radius)

    # -------------------------------------------------------------- painting
    def paintEvent(self, _ev):
        pal = self.palette()
        canvas = pal.color(QtGui.QPalette.Base)
        ink = pal.color(QtGui.QPalette.Text)
        accent = pal.color(QtGui.QPalette.Highlight)
        dark = canvas.lightness() < 128
        outline = _mix(ink, canvas, 0.35)
        muted = _mix(ink, canvas, 0.55)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        p.fillRect(self.rect(), canvas)

        cx, cy, radius = self._layout()
        self._draw_header(p, muted, ink)
        self._draw_head(p, cx, cy, radius, outline)
        self._draw_derivations(p, cx, cy, radius, muted, accent)
        self._draw_electrodes(p, cx, cy, radius, canvas, ink, accent, dark)
        self._draw_hover_badge(p, cx, cy, radius, canvas, ink, accent)
        self._draw_caption(p, muted, canvas, dark)
        p.end()

    def _draw_header(self, p, muted, ink):
        """Montage name, so the diagram can never be read out of context."""
        r = self.contentsRect()
        fm = QtGui.QFontMetrics(self.font())
        pairs = montage_pairs(self._montage)
        if pairs:
            text = '{} — {} derivations'.format(self._montage, len(pairs))
        elif self._montage == 'Common Average':
            text = 'Common Average — reference is the mean of all 19'
        else:
            text = '{} — electrodes as recorded'.format(self._montage)
        p.setPen(QtGui.QPen(muted))
        p.drawText(QtCore.QRect(r.left() + 4, r.top(), r.width() - 8,
                                fm.height() + 4),
                   QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, text)

    def _draw_head(self, p, cx, cy, radius, outline):
        """Scalp circle, nose and ears — the orientation cues."""
        pen = QtGui.QPen(outline)
        pen.setWidthF(max(1.2, radius * 0.014))
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)

        p.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

        # Nose: an open notch at the top, drawn from where it meets the circle
        # so it does not float. Nose up is the whole reason left/right is
        # unambiguous in this figure.
        ny = math.sqrt(1.0 - 0.14 ** 2)
        nose = QtGui.QPainterPath()
        nose.moveTo(self._pt(cx, cy, radius, -0.14, ny))
        nose.lineTo(self._pt(cx, cy, radius, 0.0, 1.17))
        nose.lineTo(self._pt(cx, cy, radius, +0.14, ny))
        p.drawPath(nose)

        for side in (-1.0, +1.0):
            ear = QtGui.QPainterPath()
            ear.moveTo(self._pt(cx, cy, radius, side * 0.985, +0.16))
            ear.cubicTo(self._pt(cx, cy, radius, side * 1.10, +0.16),
                        self._pt(cx, cy, radius, side * 1.16, +0.05),
                        self._pt(cx, cy, radius, side * 1.15, -0.02))
            ear.cubicTo(self._pt(cx, cy, radius, side * 1.14, -0.10),
                        self._pt(cx, cy, radius, side * 1.08, -0.17),
                        self._pt(cx, cy, radius, side * 0.985, -0.17))
            p.drawPath(ear)

    def _draw_derivations(self, p, cx, cy, radius, muted, accent):
        """One arrow per bipolar pair, tail = anode, head = cathode.

        The arrow direction encodes the subtraction the montage performs
        (trace = tail - head), which is what decides whether a phase reversal
        points at the tail electrode or the head one. Drawing an undirected
        line would lose exactly the information a reviewer localising a spike
        needs.
        """
        pairs = montage_pairs(self._montage)
        if not pairs:
            return
        mr = self._marker_radius(radius)
        focus = self._hover or self._selected
        base_pen = QtGui.QPen(muted)
        base_pen.setWidthF(max(1.0, radius * 0.011))
        base_pen.setCapStyle(QtCore.Qt.RoundCap)
        hot_pen = QtGui.QPen(accent)
        hot_pen.setWidthF(max(1.8, radius * 0.020))
        hot_pen.setCapStyle(QtCore.Qt.RoundCap)

        for _label, a, b in pairs:
            if a not in HEAD_POS or b not in HEAD_POS:
                continue           # montage names an electrode we cannot draw
            hot = focus is not None and (a == focus or b == focus)
            p.setPen(hot_pen if hot else base_pen)
            ax, ay = HEAD_POS[a]
            bx, by = HEAD_POS[b]
            pa = self._pt(cx, cy, radius, ax, ay)
            pb = self._pt(cx, cy, radius, bx, by)
            dx, dy = pb.x() - pa.x(), pb.y() - pa.y()
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            ux, uy = dx / length, dy / length
            # Trim to the electrode rims so arrows do not disappear under the
            # markers, and so a chain like Fp1-F7-T3 reads as separate hops.
            trim = mr + 2.0
            if length <= 2.0 * trim + 4.0:
                continue
            sx, sy = pa.x() + ux * trim, pa.y() + uy * trim
            ex, ey = pb.x() - ux * trim, pb.y() - uy * trim
            p.drawLine(QtCore.QPointF(sx, sy), QtCore.QPointF(ex, ey))

            head_len = max(5.0, radius * 0.055)
            spread = math.radians(24.0)
            ang = math.atan2(uy, ux)
            for sign in (-1.0, +1.0):
                a2 = ang + math.pi + sign * spread
                p.drawLine(QtCore.QPointF(ex, ey),
                           QtCore.QPointF(ex + head_len * math.cos(a2),
                                          ey + head_len * math.sin(a2)))

    def _draw_electrodes(self, p, cx, cy, radius, canvas, ink, accent, dark):
        mr = self._marker_radius(radius)
        rng = self.values_range()
        font = QtGui.QFont(self.font())
        font.setPointSizeF(max(6.0, mr * 0.78))
        fm = QtGui.QFontMetrics(font)
        # Below ~8.5 px of marker the label would be set at 6 pt inside a 17 px
        # disc: present, but not actually readable, which is worse than absent
        # because it looks like information. At that size the hover badge is
        # the way to identify an electrode, and it works at every size.
        label_fits = mr >= 8.5 and fm.height() <= 2.0 * mr - 2

        for name, mx, my in HEAD_POS_1020:
            centre = self._pt(cx, cy, radius, mx, my)
            fill = _mix(canvas, ink, 0.10)
            if rng is not None and name in self._values:
                vmin, vmax = rng
                span = vmax - vmin
                if abs(span) < 1e-12:
                    t = 0.5          # a constant field has no ordering to show
                else:
                    t = (self._values[name] - vmin) / span
                fill = _value_color(t, dark)
            elif rng is not None:
                # Shading requested but this channel has no value: hatch it so
                # "no data" cannot be misread as the bottom of the ramp.
                fill = _mix(canvas, ink, 0.18)

            is_sel = name == self._selected
            is_hot = name == self._hover
            rr = mr + (2.0 if (is_sel or is_hot) else 0.0)

            if is_hot:
                pen = QtGui.QPen(accent)
                pen.setWidthF(max(2.0, radius * 0.020))
            elif is_sel:
                pen = QtGui.QPen(accent)
                pen.setWidthF(max(1.6, radius * 0.016))
            else:
                pen = QtGui.QPen(_mix(ink, canvas, 0.30))
                pen.setWidthF(max(1.0, radius * 0.008))
            p.setPen(pen)
            p.setBrush(QtGui.QBrush(fill))
            p.drawEllipse(centre, rr, rr)

            if not label_fits:
                continue
            p.setFont(font)
            p.setPen(QtGui.QPen(_readable_on(fill)))
            box = QtCore.QRectF(centre.x() - mr * 1.6, centre.y() - mr,
                                mr * 3.2, mr * 2.0)
            p.drawText(box, QtCore.Qt.AlignCenter, name)

    def _draw_hover_badge(self, p, cx, cy, radius, canvas, ink, accent):
        """Name (and value) of the hovered electrode, always legible.

        Separate from the in-marker label because at small widget sizes that
        label is not drawn at all, and hover has to work at every size.
        """
        name = self._hover
        if name is None or name not in HEAD_POS:
            return
        text = name
        if self._values and name in self._values:
            text = '{}  {:.3g}{}'.format(
                name, self._values[name],
                (' ' + self._unit) if self._unit else '')
        derivs = derivations_for(name, self._montage)
        if derivs:
            text += '  ·  ' + ', '.join(derivs)

        font = QtGui.QFont(self.font())
        font.setBold(True)
        fm = QtGui.QFontMetrics(font)
        pad = 5
        w = fm.boundingRect(text).width() + 2 * pad + 4
        h = fm.height() + 2 * pad
        mx, my = HEAD_POS[name]
        anchor = self._pt(cx, cy, radius, mx, my)
        # Place the badge on the side of the head the electrode is NOT on, so
        # it never covers the electrode it describes.
        bx = anchor.x() - w / 2.0
        by = anchor.y() - self._marker_radius(radius) - h - 4
        if by < self.contentsRect().top():
            by = anchor.y() + self._marker_radius(radius) + 4
        bx = max(float(self.contentsRect().left() + 2),
                 min(bx, float(self.contentsRect().right() - w - 2)))

        rect = QtCore.QRectF(bx, by, w, h)
        p.setPen(QtGui.QPen(accent))
        p.setBrush(QtGui.QBrush(_mix(canvas, accent, 0.14)))
        p.drawRoundedRect(rect, 4, 4)
        p.setFont(font)
        p.setPen(QtGui.QPen(ink))
        p.drawText(rect, QtCore.Qt.AlignCenter, text)

    def _draw_caption(self, p, muted, canvas, dark):
        r = self.contentsRect()
        fm = QtGui.QFontMetrics(self.font())
        top = r.bottom() - self._reserved_caption_height()

        if self._values:
            rng = self.values_range()
            bar = QtCore.QRectF(r.left() + 8, top, r.width() - 16,
                                max(6.0, fm.height() * 0.5))
            grad = QtGui.QLinearGradient(bar.left(), 0.0, bar.right(), 0.0)
            for i in range(9):
                t = i / 8.0
                grad.setColorAt(t, _value_color(t, dark))
            p.setPen(QtGui.QPen(_mix(muted, canvas, 0.4)))
            p.setBrush(QtGui.QBrush(grad))
            p.drawRect(bar)
            p.setPen(QtGui.QPen(muted))
            lbl = QtCore.QRectF(bar.left(), bar.bottom() + 1, bar.width(),
                                fm.height())
            unit = (' ' + self._unit) if self._unit else ''
            p.drawText(lbl, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                       '{:.3g}{}'.format(rng[0], unit))
            p.drawText(lbl, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                       '{:.3g}{}'.format(rng[1], unit))
            top = lbl.bottom() + 3

        p.setPen(QtGui.QPen(muted))
        p.drawText(QtCore.QRectF(r.left() + 4, top, r.width() - 8,
                                 max(0.0, r.bottom() - _CAP_PAD - top)),
                   QtCore.Qt.TextWordWrap | QtCore.Qt.AlignHCenter
                   | QtCore.Qt.AlignTop,
                   self.CAPTION)
