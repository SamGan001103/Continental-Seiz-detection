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

        matches, fp_indexes, miss_indexes = match_events(
            preds, refs, tolerance_s=5.0)

        self.assertEqual(len(matches), 2)
        self.assertEqual(fp_indexes, [2])
        self.assertEqual(miss_indexes, [])
        self.assertEqual(matches[0]['pred_index'], 0)
        self.assertEqual(matches[0]['ref_index'], 0)

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
            segment_s=12)

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
            segment_s=12)

        self.assertEqual(row['hits'], 0)
        self.assertEqual(row['misses'], 1)
        self.assertEqual(row['false_positives'], 0)
        self.assertEqual(row['missed_reference_indexes'], [0])


if __name__ == '__main__':
    unittest.main()
