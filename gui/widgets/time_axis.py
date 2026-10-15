"""Clock time on the x axis, instead of seconds from the start of the file.

Why
---
The reviewing clinician asked for "time and date on the x axis, easier to
navigate", and on the recordings this tool is now aimed at the request is not
cosmetic. A 24-hour ambulatory recording at ten seconds a page is 8,640 pages.
"48,213 seconds" tells a reviewer nothing; "03:22" tells them it is the middle
of the night, which is when they expect to find things. Every note a clinician
writes, every nursing observation, every video review is timestamped in clock
time, so an axis in elapsed seconds cannot be cross-referenced with anything
else in the record.

Every EDF carries the recording's start date and time in its header and the
application was reading neither. The converted EPILEPSIAE recordings carry real
ones (2002-09-30 10:58:46 and so on), and TUH files carry a surrogate date with
a plausible clock time.

What is shown
-------------
The format follows the zoom, because the useful precision changes with it:

    tick spacing     shown as            typical view
    under 1 s        11:02:46.5          a few seconds on screen
    under 1 min      11:02:46            a ten-second page
    under 1 hour     11:02               a five-minute view
    an hour or more  30 Sep 11:00        the whole recording

With no start time in the header the axis falls back to elapsed time, but
formatted as `1:23:45` rather than `5025`, which is still easier to read than
what it replaces.

The formatting is a plain function so it can be tested without a display; the
`AxisItem` is a thin wrapper over it.
"""
import datetime as _dt

SPACING_SUBSECOND = 1.0
SPACING_SECONDS = 60.0
SPACING_MINUTES = 3600.0


def format_tick(seconds, spacing, start=None):
    """One tick label.

    seconds : offset from the start of the recording
    spacing : the axis's tick spacing, which is what sets useful precision
    start   : recording start as a datetime, or None if the header had none
    """
    spacing = abs(float(spacing)) or 1.0
    if start is None:
        return _elapsed(seconds, spacing)
    try:
        t = start + _dt.timedelta(seconds=float(seconds))
    except (OverflowError, ValueError, TypeError):
        return _elapsed(seconds, spacing)
    if spacing < SPACING_SUBSECOND:
        return '{}.{}'.format(t.strftime('%H:%M:%S'), t.microsecond // 100000)
    if spacing < SPACING_SECONDS:
        return t.strftime('%H:%M:%S')
    if spacing < SPACING_MINUTES:
        return t.strftime('%H:%M')
    # An hour or more between ticks means the view spans most of a day or
    # more, and the date stops being obvious from context.
    return '{} {}'.format(t.strftime('%d %b'), t.strftime('%H:%M'))


def _elapsed(seconds, spacing):
    """Elapsed time when the file does not say when it started."""
    try:
        v = float(seconds)
    except (TypeError, ValueError):
        return ''
    sign = '-' if v < 0 else ''
    v = abs(v)
    if spacing < SPACING_SUBSECOND:
        return '{}{:.1f}'.format(sign, v)
    total = int(round(v))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    # Whether to show an hours field is decided by the SPACING, not by this
    # one value, so every tick on an axis has the same shape. Deciding per
    # value put "0:00" next to "3:00:00" on the same row.
    if h or spacing >= SPACING_MINUTES:
        return '{}{}:{:02d}:{:02d}'.format(sign, h, m, s)
    return '{}{}:{:02d}'.format(sign, m, s)


def format_cursor(seconds, start=None):
    """The hover readout under the plot.

    The axis labels are sparse and rounded to the tick spacing, so they answer
    "roughly where am I"; the cursor has to answer "exactly when is this
    sample", which is what gets written into a report or matched against a
    nursing note. So it shows both clocks at full precision: the offset into
    the file, which is what the detector's scores and the `.csv_bi` are
    indexed by, and the wall-clock date and time, which is what every other
    record in the hospital is indexed by. Having to convert between those two
    in your head is exactly the friction the clinician asked us to remove.

    The weekday is included because these recordings run for days and "is this
    the Saturday night one" is a question reviewers actually ask.
    """
    elapsed = _cursor_elapsed(seconds)
    if elapsed is None:
        return 't = --:--:--.---'
    if start is None:
        return 't = {}'.format(elapsed)
    try:
        t = start + _dt.timedelta(seconds=float(seconds))
    except (OverflowError, ValueError, TypeError):
        return 't = {}'.format(elapsed)
    return 't = {}   ·   {} {}.{:03d}'.format(
        elapsed, t.strftime('%a %d %b %Y'), t.strftime('%H:%M:%S'),
        t.microsecond // 1000)


def _cursor_elapsed(seconds):
    """h:mm:ss.sss from the start of the file, or None if unformattable.

    Always carries the hours field, unlike the axis, because the cursor is
    read on its own rather than in a column of sibling labels.
    """
    try:
        v = float(seconds)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float('inf'), float('-inf')):     # NaN or infinity
        return None
    sign = '-' if v < 0 else ''
    v = abs(v)
    h = int(v // 3600)
    rem = v - h * 3600
    m = int(rem // 60)
    s = rem - m * 60
    return '{}{:d}:{:02d}:{:06.3f}'.format(sign, h, m, s)


def axis_label(start=None, span=None):
    """What to call the axis, given whether a start time is known.

    `span` is the visible range in seconds from the start of the file. Given
    one, the caption names the date ON SCREEN rather than the date the
    recording began.

    That distinction is the whole reason this takes a span. An ambulatory
    study runs through the night: pat_21902_part02 starts at 18:04 on 10 Jan
    and three quarters of it is the 11th. A caption fixed to the start date
    read "10 Jan 2004" over traces from the 11th, and it is wrong in the worst
    direction -- it is the date a reviewer copies down for an event they have
    just marked, while the cursor readout beside it says the 11th. One of the
    two had to go, and the axis is the one that can be made to follow the view.
    """
    if start is None:
        return 'time from start of recording (h:mm:ss)'
    if span is None:
        return 'clock time — {}'.format(start.strftime('%d %b %Y'))
    try:
        t0, t1 = float(span[0]), float(span[1])
    except (TypeError, ValueError, IndexError):
        return 'clock time — {}'.format(start.strftime('%d %b %Y'))
    if t1 < t0:
        t0, t1 = t1, t0
    if t0 != t0 or t1 != t1:                         # NaN
        return 'clock time — {}'.format(start.strftime('%d %b %Y'))
    d0 = (start + _dt.timedelta(seconds=t0)).date()
    d1 = (start + _dt.timedelta(seconds=t1)).date()
    if d0 == d1:
        return 'clock time — {}'.format(d0.strftime('%d %b %Y'))
    # Crossing midnight is the normal case on a day-long recording, so it gets
    # a compact form rather than two full dates: the month and year repeat.
    if (d0.year, d0.month) == (d1.year, d1.month):
        return 'clock time — {}–{}'.format(d0.strftime('%d'),
                                             d1.strftime('%d %b %Y'))
    if d0.year == d1.year:
        return 'clock time — {} – {}'.format(d0.strftime('%d %b'),
                                               d1.strftime('%d %b %Y'))
    return 'clock time — {} – {}'.format(d0.strftime('%d %b %Y'),
                                           d1.strftime('%d %b %Y'))


def make_time_axis(orientation='bottom'):
    """A pyqtgraph AxisItem that shows clock time. Imported lazily so this
    module can be tested, and its formatting reasoned about, without Qt."""
    import pyqtgraph as pg

    class _TimeAxis(pg.AxisItem):
        def __init__(self, orientation='bottom'):
            super().__init__(orientation=orientation)
            self._start = None

        def set_start(self, start):
            """start: datetime from the EDF header, or None."""
            self._start = start
            self.picture = None
            self.update()

        def start(self):
            return self._start

        def tickStrings(self, values, scale, spacing):
            return [format_tick(v, spacing, self._start) for v in values]

    return _TimeAxis(orientation=orientation)
