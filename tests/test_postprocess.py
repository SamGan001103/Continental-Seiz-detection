"""Tests for the source method's decision stage (gui/postprocess.py).

These pin the three rules the reproduction depends on: per-second averaging of
overlapping windows, concatenation of nearby events, and removal of short ones.
"""
import unittest

import numpy as np

from gui.postprocess import (concatenate_nearby, discard_short,
                             events_from_probs, per_second_probability,
                             shape_events)


class PerSecondProbabilityTests(unittest.TestCase):
    def test_single_window_covers_its_whole_span(self):
        secs, p = per_second_probability([0], [0.8], segment_s=12)
        self.assertEqual(len(p), 12)
        np.testing.assert_allclose(p, [0.8] * 12, rtol=1e-6)
        self.assertEqual(secs[0], 0)

    def test_overlapping_windows_are_averaged_not_maxed(self):
        # Windows at 0 and 6 (12 s each) both cover seconds 6-11.
        _, p = per_second_probability([0, 6], [1.0, 0.0], segment_s=12)
        np.testing.assert_allclose(p[0:6], 1.0, rtol=1e-6)   # window 0 only
        np.testing.assert_allclose(p[6:12], 0.5, rtol=1e-6)  # mean, not max
        np.testing.assert_allclose(p[12:18], 0.0, rtol=1e-6)  # window 1 only

    def test_uncovered_seconds_are_zero(self):
        _, p = per_second_probability([0], [1.0], segment_s=12,
                                      duration_s=30.0)
        self.assertEqual(len(p), 30)
        np.testing.assert_allclose(p[12:], 0.0)

    def test_empty_input(self):
        secs, p = per_second_probability([], [], segment_s=12)
        self.assertEqual(len(secs), 0)
        self.assertEqual(len(p), 0)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            per_second_probability([0, 6], [0.5], segment_s=12)


class EventShapingTests(unittest.TestCase):
    def test_concatenate_joins_events_closer_than_the_gap(self):
        merged = concatenate_nearby([(0, 10), (15, 25)], max_gap_s=10.0)
        self.assertEqual(merged, [(0.0, 25.0)])

    def test_concatenate_leaves_distant_events_alone(self):
        merged = concatenate_nearby([(0, 10), (30, 40)], max_gap_s=10.0)
        self.assertEqual(merged, [(0.0, 10.0), (30.0, 40.0)])

    def test_gap_exactly_at_the_limit_is_not_merged(self):
        merged = concatenate_nearby([(0, 10), (20, 30)], max_gap_s=10.0)
        self.assertEqual(len(merged), 2)

    def test_discard_removes_short_events(self):
        kept = discard_short([(0, 3), (10, 20)], min_duration_s=5.0)
        self.assertEqual(kept, [(10.0, 20.0)])

    def test_duration_exactly_at_the_limit_is_kept(self):
        kept = discard_short([(0, 5)], min_duration_s=5.0)
        self.assertEqual(kept, [(0.0, 5.0)])

    def test_merge_before_discard_rescues_a_fragmented_seizure(self):
        """Three 3 s fragments of one seizure survive as a single 21 s event.
        Discarding first would delete all three."""
        frags = [(0, 3), (6, 9), (18, 21)]
        self.assertEqual(shape_events(frags, 5.0, 10.0), [(0.0, 21.0)])
        self.assertEqual(shape_events(frags, 5.0, 10.0,
                                      strict_discard_first=True), [])


class EventsFromProbsTests(unittest.TestCase):
    def test_isolated_spike_is_averaged_away(self):
        starts = list(range(0, 120, 6))
        probs = [0.02] * len(starts)
        probs[10] = 0.95
        events = events_from_probs(starts, probs, threshold=0.5, segment_s=12,
                                   duration_s=132.0)
        self.assertEqual(events, [])

    def test_sustained_run_survives(self):
        starts = list(range(0, 120, 6))
        probs = [0.02] * len(starts)
        for i in range(5, 12):            # ~42 s of sustained high probability
            probs[i] = 0.95
        events = events_from_probs(starts, probs, threshold=0.5, segment_s=12,
                                   duration_s=132.0)
        self.assertEqual(len(events), 1)
        start, stop, peak = events[0]
        self.assertGreaterEqual(stop - start, 5.0)
        self.assertGreater(peak, 0.5)

    def test_events_are_clamped_to_the_recording(self):
        starts = list(range(0, 60, 6))
        probs = [0.99] * len(starts)
        events = events_from_probs(starts, probs, threshold=0.5, segment_s=12,
                                   duration_s=40.0)
        self.assertTrue(all(0.0 <= s and e <= 40.0 for s, e, _ in events))

    def test_short_event_is_dropped_only_when_shaping_is_enabled(self):
        # 1 s windows at 1 s stride, so the per-second series equals probs and
        # the run length is controlled exactly: 3 s, below the 5 s minimum.
        starts = list(range(30))
        probs = [0.02] * 30
        for i in (10, 11, 12):
            probs[i] = 0.9

        shaped = events_from_probs(starts, probs, 0.5, segment_s=1,
                                   duration_s=30.0)
        unshaped = events_from_probs(starts, probs, 0.5, segment_s=1,
                                     duration_s=30.0, min_duration_s=0.0,
                                     max_gap_s=0.0)

        self.assertEqual(shaped, [])
        self.assertEqual(len(unshaped), 1)
        self.assertAlmostEqual(unshaped[0][1] - unshaped[0][0], 3.0)

    def test_twelve_second_run_survives_the_five_second_minimum(self):
        # Overlapping 12 s windows at 6 s stride cannot produce a run shorter
        # than the window, so a lone hot window still yields a 12 s event.
        events = events_from_probs([0, 6, 12], [0.02, 0.99, 0.02], 0.5, 12,
                                   duration_s=30.0)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0][1] - events[0][0], 12.0)

    def test_no_average_mode_thresholds_raw_windows(self):
        starts = list(range(0, 120, 6))
        probs = [0.02] * len(starts)
        probs[10] = 0.95
        raw = events_from_probs(starts, probs, 0.5, 12, duration_s=132.0,
                                average=False, min_duration_s=0.0,
                                max_gap_s=0.0)
        self.assertEqual(len(raw), 1)
        self.assertAlmostEqual(raw[0][1] - raw[0][0], 12.0)


if __name__ == '__main__':
    unittest.main()
