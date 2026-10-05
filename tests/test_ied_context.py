"""Interictal discharges as context for judging a seizure detection.

The point of showing discharges is not to annotate them. It is that a
detection in a patient who spikes all day reads differently from one in a
recording with no interictal activity at all. These pin the arithmetic behind
that comparison, and in particular the places where it must REFUSE to make a
claim: the detector's absolute threshold is unvalidated, so the only honest
comparison is against the patient's own baseline, and that needs enough
recording to be a baseline.

No Qt, no model.
"""
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.ied_context import (                        # noqa: E402
    DEFAULT_HALO_S, context_for_event, density, describe)

DAY = 24 * 3600.0


def marks_at(*times):
    return [{'time': float(t), 'source': 'spikenet1'} for t in times]


def spread(n, span, t0=0.0):
    """n marks evenly across `span` seconds — a patient's steady baseline."""
    return marks_at(*[t0 + span * i / float(n) for i in range(n)])


class TheCandidateIsComparedWithThePatientsOwnBaseline(unittest.TestCase):
    def test_a_cluster_around_the_event_beats_the_baseline(self):
        base = spread(24, DAY)                       # 1/h all day
        # AROUND, not inside: the discharges sit in the two minutes before
        # onset. An earlier version of this test put all twenty INSIDE the
        # seizure and still called the result corroboration, which is the
        # circularity the ictal exclusion exists to prevent.
        burst = marks_at(*[8880.0 + i for i in range(20)])
        ctx = context_for_event(base + burst, 9000.0, 9060.0, DAY)
        self.assertGreater(ctx['ratio'], 2.0)
        self.assertIn('baseline', describe(ctx))

    def test_a_quiet_neighbourhood_is_reported_as_quiet(self):
        # Dense everywhere except a clear hour around the candidate.
        far = spread(240, DAY * 0.4, t0=DAY * 0.5)
        ctx = context_for_event(far, 3600.0, 3660.0, DAY)
        self.assertEqual(ctx['near'], 0)
        self.assertLess(ctx['ratio'], 0.5)
        self.assertIn('quieter', describe(ctx))

    def test_discharges_inside_the_event_are_counted_separately(self):
        ctx = context_for_event(marks_at(100.0, 105.0, 400.0), 99.0, 110.0, DAY)
        self.assertEqual(ctx['during'], 2)
        # `near` is the INTERICTAL rate around the event, so the two inside it
        # are not in it — only the one at 400 s is.
        self.assertEqual(ctx['near'], 1)

    def test_a_recording_with_no_discharges_says_so(self):
        ctx = context_for_event([], 100.0, 120.0, DAY)
        self.assertEqual(ctx['total'], 0)
        self.assertIsNone(ctx['ratio'])
        self.assertIn('No interictal discharges', describe(ctx))


class ItRefusesToQuoteABaselineItCannotSupport(unittest.TestCase):
    """Quoting "12x baseline" off four minutes of EEG would be worse than
    saying nothing, because it reads as evidence."""

    def test_a_recording_barely_longer_than_the_halo_has_no_baseline(self):
        dur = 2 * DEFAULT_HALO_S + 60.0               # ~11 min total
        ctx = context_for_event(marks_at(300.0, 320.0), 500.0, 560.0, dur)
        self.assertIsNone(ctx['ratio'])
        self.assertNotIn('baseline', describe(ctx))

    def test_a_silent_baseline_does_not_divide_by_zero(self):
        near = marks_at(*[9000.0 + i for i in range(10)])
        ctx = context_for_event(near, 9000.0, 9060.0, DAY)
        self.assertEqual(ctx['baseline_rate_per_h'], 0.0)
        self.assertIsNone(ctx['ratio'])
        self.assertTrue(describe(ctx))

    def test_a_full_day_does_support_a_baseline(self):
        ctx = context_for_event(spread(48, DAY), 9000.0, 9060.0, DAY)
        self.assertIsNotNone(ctx['ratio'])
        self.assertGreater(ctx['baseline_hours'], 20.0)


class RatesUseOnlyTheRecordingThatExists(unittest.TestCase):
    """A candidate near either end of the file has less context around it than
    one in the middle. The rate must be taken over the span actually observed,
    not over the nominal halo — dividing by five minutes of EEG that is not
    there under-reports the rate and makes an early candidate look quieter
    than an identical one mid-recording."""

    NAIVE_SPAN_S = 2 * DEFAULT_HALO_S            # halo either side, minus the
                                                 # 10 s event itself

    def _naive(self, n):
        return n / (self.NAIVE_SPAN_S / 3600.0)

    def test_an_event_at_the_very_start_is_not_under_reported(self):
        # The event sits clear of the marks: seizure time is excluded from
        # both the counts and the span, so an event overlapping them would be
        # measuring something else.
        early = context_for_event(marks_at(10.0, 20.0, 30.0, 40.0),
                                  60.0, 70.0, DAY)
        # The halo runs off the front of the recording, so only 370 s of it
        # exists; take off the 10 s event and 360 s remain. Four discharges
        # over 360 s is a HIGHER rate than the same four over the full 600 s,
        # which is the whole point — an early candidate must not look quieter
        # than an identical one mid-recording.
        self.assertGreater(early['near_rate_per_h'], self._naive(4))
        self.assertAlmostEqual(early['near_rate_per_h'],
                               4 / (360.0 / 3600.0), places=6)

    def test_an_event_at_the_very_end_is_not_under_reported(self):
        late = context_for_event(
            marks_at(DAY - 40, DAY - 30, DAY - 20, DAY - 10),
            DAY - 35, DAY - 25, DAY)
        self.assertGreater(late['near_rate_per_h'], self._naive(4))

    def test_an_event_in_the_middle_uses_the_whole_halo(self):
        mid = context_for_event(marks_at(50010.0, 50020.0, 50030.0, 50040.0),
                                50060.0, 50070.0, DAY)
        self.assertAlmostEqual(mid['near_rate_per_h'], self._naive(4),
                               places=6)

    def test_the_two_ends_agree_with_each_other(self):
        early = context_for_event(marks_at(10.0, 20.0, 30.0, 40.0),
                                  25.0, 35.0, DAY)
        late = context_for_event(
            marks_at(DAY - 40, DAY - 30, DAY - 20, DAY - 10),
            DAY - 35, DAY - 25, DAY)
        self.assertAlmostEqual(early['near_rate_per_h'],
                               late['near_rate_per_h'], places=6)


class BadInputIsSurvivable(unittest.TestCase):
    """This runs on every event in a worklist; one malformed mark must not
    take the review window down."""

    def test_marks_without_a_usable_time_are_skipped(self):
        bad = [{'time': 'x'}, {'nope': 1}, {'time': None}, {'time': 50.0}]
        self.assertEqual(context_for_event(bad, 40.0, 60.0, DAY)['total'], 1)

    def test_a_reversed_extent_is_tolerated(self):
        ctx = context_for_event(marks_at(100.0), 120.0, 80.0, DAY)
        self.assertEqual(ctx['during'], 1)

    def test_a_zero_length_recording_does_not_raise(self):
        self.assertTrue(describe(context_for_event(marks_at(1.0), 0.0, 0.0,
                                                   0.0)))


class DensityIsWhatAWholeRecordingCanShow(unittest.TestCase):
    """Individual marks over 24 h are sub-pixel; a rate curve is not."""

    def test_it_returns_one_rate_per_bin(self):
        self.assertEqual(len(density(spread(100, DAY), DAY, 64)), 64)

    def test_an_even_spread_is_flat(self):
        d = density(spread(240, DAY), DAY, 24)
        self.assertAlmostEqual(min(d), max(d), delta=1.0)

    def test_a_cluster_shows_up_in_its_own_bin(self):
        d = density(marks_at(*[3600.0 + i for i in range(60)]), DAY, 24)
        self.assertEqual(d.index(max(d)), 1)
        self.assertEqual(d[0], 0.0)

    def test_the_rate_is_per_hour_not_a_count(self):
        # 10 marks in a 10-minute bin is 60/h, not 10.
        d = density(spread(10, 600.0), 600.0, 1)
        self.assertAlmostEqual(d[0], 60.0, places=6)

    def test_no_marks_is_a_flat_zero_curve(self):
        self.assertEqual(density([], DAY, 8), [0.0] * 8)

    def test_a_mark_exactly_at_the_end_is_not_dropped(self):
        self.assertGreater(sum(density(marks_at(DAY), DAY, 8)), 0.0)


class SeizureTimeIsNotInterictalTime(unittest.TestCase):
    """Interictal means BETWEEN seizures, and the word is doing real work.

    A spike detector slid across a seizure fires on the seizure. Measured on
    pat_102_part02, 69% of a day's 992 candidates fell in 4.6% of the
    recording, around and inside one event. Counting those as interictal
    activity makes the "independent evidence" for a candidate partly the
    candidate itself — it would read as corroboration while being circular.
    """

    def test_discharges_inside_the_event_do_not_inflate_its_own_context(self):
        inside = marks_at(*[9010.0 + i for i in range(30)])
        ctx = context_for_event(inside, 9000.0, 9060.0, DAY)
        self.assertEqual(ctx['during'], 30)
        self.assertEqual(ctx['near'], 0)
        self.assertEqual(ctx['near_rate_per_h'], 0.0)

    def test_they_are_reported_but_marked_as_not_interictal(self):
        ctx = context_for_event(marks_at(9010.0, 9020.0) + spread(48, DAY),
                                9000.0, 9060.0, DAY)
        self.assertIn('inside the event', describe(ctx))
        self.assertIn('not counted as interictal', describe(ctx))

    def test_another_seizure_is_excluded_from_the_baseline(self):
        """Otherwise a dense burst inside seizure B raises the baseline that
        seizure A is being judged against, and A looks quieter than it is."""
        other = marks_at(*[40000.0 + i for i in range(100)])
        base = spread(24, DAY)
        without = context_for_event(base + other, 9000.0, 9060.0, DAY)
        with_ex = context_for_event(base + other, 9000.0, 9060.0, DAY,
                                    ictal=[(39990.0, 40110.0)])
        self.assertGreater(without['baseline_rate_per_h'],
                           with_ex['baseline_rate_per_h'])
        self.assertEqual(with_ex['ictal'], 100)

    def test_overlapping_seizure_spans_do_not_subtract_time_twice(self):
        """Merged before subtracting; otherwise a span could go negative and
        the rate would come out nonsense or raise."""
        ctx = context_for_event(spread(48, DAY), 9000.0, 9060.0, DAY,
                                ictal=[(9030.0, 9100.0), (9050.0, 9090.0)])
        self.assertGreaterEqual(ctx['baseline_rate_per_h'], 0.0)
        self.assertGreaterEqual(ctx['baseline_hours'], 0.0)

    def test_a_malformed_ictal_span_is_skipped_not_fatal(self):
        ctx = context_for_event(spread(48, DAY), 9000.0, 9060.0, DAY,
                                ictal=[('x', 1), None, (100.0,), (200.0, 250.0)])
        self.assertIsNotNone(ctx['baseline_rate_per_h'])

    def test_a_reversed_ictal_span_is_tolerated(self):
        a = context_for_event(spread(48, DAY), 9000.0, 9060.0, DAY,
                              ictal=[(40110.0, 39990.0)])
        b = context_for_event(spread(48, DAY), 9000.0, 9060.0, DAY,
                              ictal=[(39990.0, 40110.0)])
        self.assertAlmostEqual(a['baseline_rate_per_h'],
                               b['baseline_rate_per_h'], places=9)

    def test_the_total_still_counts_every_mark(self):
        """Excluding ictal marks from the RATES must not hide them from the
        count — a reviewer asking how many discharges are on this recording
        wants all of them."""
        marks = marks_at(9010.0, 9020.0) + spread(24, DAY)
        ctx = context_for_event(marks, 9000.0, 9060.0, DAY)
        self.assertEqual(ctx['total'], len(marks))


if __name__ == '__main__':
    unittest.main()
