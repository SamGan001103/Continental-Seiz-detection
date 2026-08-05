import unittest

from experiments.compare_zuna import match_events, summarize_run


class CompareZunaTests(unittest.TestCase):
    def test_match_events_one_prediction_per_reference(self):
        refs = [
            {'start': 10.0, 'stop': 20.0},
            {'start': 50.0, 'stop': 60.0},
        ]
        preds = [
            {'start': 9.0, 'stop': 21.0},
            {'start': 51.0, 'stop': 58.0},
            {'start': 90.0, 'stop': 96.0},
        ]

        matches, fp_indexes, miss_indexes, dup_indexes = match_events(
            preds, refs, tolerance_s=5.0)

        self.assertEqual(len(matches), 2)
        self.assertEqual(fp_indexes, [2])
        self.assertEqual(miss_indexes, [])
        self.assertEqual(dup_indexes, [])
        self.assertEqual(matches[0]['pred_index'], 0)
        self.assertEqual(matches[0]['ref_index'], 0)

    def test_fragment_of_detected_seizure_is_not_a_false_positive(self):
        """A second prediction inside an already-matched reference is a
        fragment of a correctly detected seizure, not a false alarm."""
        refs = [{'start': 10.0, 'stop': 60.0}]
        preds = [
            {'start': 10.0, 'stop': 30.0},   # matches the reference
            {'start': 36.0, 'stop': 58.0},   # same seizure, after a dropout
        ]

        matches, fp_indexes, miss_indexes, dup_indexes = match_events(
            preds, refs, tolerance_s=5.0)

        self.assertEqual(len(matches), 1)
        self.assertEqual(fp_indexes, [])       # previously reported [1]
        self.assertEqual(dup_indexes, [1])
        self.assertEqual(miss_indexes, [])

    def test_prediction_outside_every_reference_is_a_false_positive(self):
        refs = [{'start': 10.0, 'stop': 20.0}]
        preds = [
            {'start': 10.0, 'stop': 20.0},
            {'start': 200.0, 'stop': 210.0},
        ]

        matches, fp_indexes, _miss, dup_indexes = match_events(
            preds, refs, tolerance_s=5.0)

        self.assertEqual(len(matches), 1)
        self.assertEqual(fp_indexes, [1])
        self.assertEqual(dup_indexes, [])

    def test_summarize_run_counts_hits_misses_and_false_alarm_rate(self):
        refs = [{'start': 5.0, 'stop': 20.0}]

        row = summarize_run(
            'baseline',
            window_starts=[0, 6, 18, 40],
            probs=[0.7, 0.6, 0.1, 0.9],
            ref_events=refs,
            threshold=0.5,
            duration_s=60.0,
            tolerance_s=5.0,
            segment_s=12,
            postprocess=False)

        self.assertEqual(row['n_refs'], 1)
        self.assertEqual(row['n_pred'], 2)
        self.assertEqual(row['hits'], 1)
        self.assertEqual(row['misses'], 0)
        self.assertEqual(row['false_positives'], 1)
        self.assertAlmostEqual(row['sensitivity'], 1.0)
        self.assertAlmostEqual(row['false_alarms_per_24h'], 1440.0)
        self.assertAlmostEqual(row['mean_onset_abs_error_s'], 5.0)

    def test_summarize_run_reports_missed_reference(self):
        row = summarize_run(
            'baseline',
            window_starts=[0, 6],
            probs=[0.1, 0.2],
            ref_events=[{'start': 100.0, 'stop': 110.0}],
            threshold=0.5,
            duration_s=120.0,
            tolerance_s=5.0,
            segment_s=12,
            postprocess=False)

        self.assertEqual(row['hits'], 0)
        self.assertEqual(row['misses'], 1)
        self.assertEqual(row['false_positives'], 0)
        self.assertEqual(row['missed_reference_indexes'], [0])

    def test_averaging_not_shaping_suppresses_an_isolated_spike(self):
        """An isolated hot window survives event shaping — at 6 s stride it is
        a 12 s event, comfortably over the 5 s minimum. It is per-second
        AVERAGING against its quiet neighbours that pulls it under threshold.

        This pins which of the two rules does the false-alarm suppression, so a
        later change to either default cannot silently alter the reported
        false-alarm rate."""
        starts = list(range(0, 120, 6))
        probs = [0.02] * len(starts)
        probs[10] = 0.95                      # one spike at t=60

        def fp(postprocess, average):
            return summarize_run(
                'baseline', starts, probs, ref_events=[], threshold=0.5,
                duration_s=132.0, segment_s=12,
                postprocess=postprocess, average=average)['false_positives']

        self.assertEqual(fp(False, False), 1)   # raw
        self.assertEqual(fp(True, False), 1)    # shaping alone does not help
        self.assertEqual(fp(False, True), 0)    # averaging is what removes it
        self.assertEqual(fp(True, True), 0)

    def test_shaping_joins_a_fragmented_detection(self):
        """Two runs split by one sub-threshold window are one seizure, and
        shaping is what rejoins them."""
        starts = list(range(0, 120, 6))
        probs = [0.02] * len(starts)
        for i in (2, 3, 5, 6):                # gap at index 4
            probs[i] = 0.95

        raw = summarize_run(
            'baseline', starts, probs, ref_events=[], threshold=0.5,
            duration_s=132.0, segment_s=12, postprocess=False, average=False)
        shaped = summarize_run(
            'baseline', starts, probs, ref_events=[], threshold=0.5,
            duration_s=132.0, segment_s=12, postprocess=True, average=False)

        self.assertEqual(raw['n_pred'], 2)
        self.assertEqual(shaped['n_pred'], 1)
        self.assertTrue(shaped['postprocessed'])


if __name__ == '__main__':
    unittest.main()
