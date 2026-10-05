"""The x axis has to say when, not how many seconds in.

The reviewing clinician's request was "time and date for the x axis, easier to
navigate". On a 24-hour recording at ten seconds a page that is 8,640 pages, and
"48,213 seconds" cannot be matched against a nursing note, a video review or a
clinician's own recollection — all of which are in clock time.

These tests pin the formatting only. They import no Qt, so they run anywhere.
"""
import datetime as dt
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.widgets.time_axis import (             # noqa: E402
    axis_label, format_cursor, format_tick)

# The real start of pat_102's first converted day.
START = dt.datetime(2002, 9, 30, 10, 58, 46)


class ClockTimeWhenTheFileSaysWhenItStarted(unittest.TestCase):
    def test_a_ten_second_page_shows_seconds(self):
        self.assertEqual(format_tick(0, 2.0, START), '10:58:46')
        self.assertEqual(format_tick(14, 2.0, START), '10:59:00')

    def test_a_five_minute_view_shows_minutes(self):
        self.assertEqual(format_tick(74, 60.0, START), '11:00')

    def test_a_whole_day_view_shows_the_date_too(self):
        """At an hour between ticks the view spans most of a day, and which
        day it is stops being obvious."""
        self.assertEqual(format_tick(0, 10800.0, START), '30 Sep 10:58')
        self.assertEqual(format_tick(24 * 3600, 10800.0, START),
                         '01 Oct 10:58')

    def test_crossing_midnight_rolls_the_date(self):
        s = format_tick(13 * 3600 + 3600, 10800.0, START)   # 14 h in
        self.assertTrue(s.startswith('01 Oct'), s)

    def test_sub_second_spacing_keeps_a_decimal(self):
        self.assertEqual(format_tick(0.5, 0.2, START), '10:58:46.5')

    def test_the_label_names_the_date(self):
        self.assertEqual(axis_label(START), 'clock time — 30 Sep 2002')


class ElapsedTimeWhenItDoesNot(unittest.TestCase):
    """Some EDFs carry no usable start. The axis must still be readable, and
    '5025' is not."""

    def test_seconds_become_minutes_and_seconds(self):
        self.assertEqual(format_tick(125, 10.0, None), '2:05')

    def test_past_an_hour_it_grows_an_hours_field(self):
        self.assertEqual(format_tick(5025, 60.0, None), '1:23:45')

    def test_sub_second_spacing_keeps_a_decimal(self):
        self.assertEqual(format_tick(2.25, 0.5, None), '2.2')

    def test_negative_offsets_do_not_lose_their_sign(self):
        self.assertTrue(format_tick(-65, 10.0, None).startswith('-'))

    def test_every_tick_on_one_axis_has_the_same_shape(self):
        """Deciding the hours field per VALUE put '0:00' next to '3:00:00' on
        the same row. The spacing decides it, so the whole axis agrees."""
        labels = [format_tick(v, 10800.0, None)
                  for v in (0, 10800, 21600)]
        self.assertEqual(labels, ['0:00:00', '3:00:00', '6:00:00'])
        page = [format_tick(v, 2.0, None) for v in (0, 2, 4)]
        self.assertEqual(page, ['0:00', '0:02', '0:04'])

    def test_the_label_says_it_is_elapsed(self):
        self.assertIn('from start', axis_label(None))


class BadInputDoesNotBreakTheAxis(unittest.TestCase):
    """An axis that raises takes the whole plot down, so every branch has to
    survive whatever pyqtgraph hands it."""

    def test_zero_spacing_is_survivable(self):
        self.assertTrue(format_tick(10, 0.0, START))

    def test_an_absurd_offset_falls_back_instead_of_raising(self):
        self.assertTrue(format_tick(1e18, 3600.0, START))

    def test_a_non_numeric_offset_returns_something(self):
        self.assertEqual(format_tick('x', 10.0, None), '')


class TheCursorReadoutCarriesBothClocks(unittest.TestCase):
    """Hovering has to answer "exactly when is this sample" in both the
    frame the detector uses (offset into the file) and the frame the rest of
    the hospital record uses (wall clock). The axis labels round to the tick
    spacing, so they cannot answer either precisely."""

    def test_it_shows_elapsed_and_wall_clock_together(self):
        s = format_cursor(59261.796, START)
        self.assertIn('16:27:41.796', s)        # offset into the file
        self.assertIn('03:26:27.796', s)        # the following morning
        self.assertIn('01 Oct 2002', s)

    def test_the_weekday_is_there_because_recordings_span_days(self):
        self.assertIn('Tue', format_cursor(59261.796, START))

    def test_without_a_header_date_it_falls_back_to_elapsed_only(self):
        s = format_cursor(59261.796, None)
        self.assertEqual(s, 't = 16:27:41.796')
        self.assertNotIn('·', s)

    def test_the_hours_field_is_always_present(self):
        # Unlike the axis, the cursor is read alone, so "02.500" would be
        # ambiguous where "0:00:02.500" is not.
        self.assertEqual(format_cursor(2.5, None), 't = 0:00:02.500')

    def test_millisecond_precision_survives_a_long_offset(self):
        # Float seconds 16 h in still have to round-trip to the millisecond,
        # which is the precision a reviewer quotes a spike at.
        self.assertIn('.796', format_cursor(59261.796, START))

    def test_no_hover_is_a_placeholder_not_a_crash(self):
        self.assertEqual(format_cursor(None), 't = --:--:--.---')

    def test_a_non_numeric_offset_is_survivable(self):
        self.assertEqual(format_cursor('x', START), 't = --:--:--.---')

    def test_nan_does_not_reach_the_label(self):
        self.assertEqual(format_cursor(float('nan'), START),
                         't = --:--:--.---')

    def test_an_absurd_offset_falls_back_instead_of_raising(self):
        self.assertTrue(format_cursor(1e18, START).startswith('t = '))

    def test_a_negative_offset_keeps_its_sign(self):
        self.assertTrue(format_cursor(-65, None).startswith('t = -'))


if __name__ == '__main__':
    unittest.main()
