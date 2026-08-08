"""A window the pipeline refused to score must never look like a confident 'no'.

The failure this guards against is concrete: aaaaaqtw_s002_t012 has all 49 of its
windows rejected by the artifact check and contains a real 27-second reference
seizure, but rendered as a flat zero line with an empty event list — visually
identical to a clean recording the model had assessed and cleared.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from gui.io.cache import (
    load_probability_file, save_probability_file,
)
from gui.io.infer import (
    SKIP_ICA_FAILED, SKIP_INTERRUPTED, SKIP_LABELS, SKIP_NONE,
)
from gui.widgets.prob_strip import _unscored_spans


class SkipCodeRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, 'p.npz')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skip_code_survives_the_round_trip_and_is_marked_exact(self):
        starts = np.array([0, 6, 12], dtype=np.int32)
        probs = np.array([0.9, 0.0, 0.1], dtype=np.float32)
        skip = np.array([SKIP_NONE, SKIP_ICA_FAILED, SKIP_NONE], dtype=np.int8)
        save_probability_file(self.path, starts, probs, {}, skip_code=skip)

        loaded = load_probability_file(self.path)
        np.testing.assert_array_equal(loaded['skip_code'], skip)
        self.assertTrue(loaded['skip_code_is_exact'])
        self.assertTrue(loaded['meta']['has_skip_code'])
        self.assertEqual(loaded['meta']['n_unscored_windows'], 1)

    def test_legacy_cache_infers_the_mask_and_admits_it_is_inferred(self):
        """A v1 cache has no mask; the 0.0 sentinel is the only signal left."""
        starts = np.array([0, 6, 12], dtype=np.int32)
        probs = np.array([0.9, 0.0, 0.1], dtype=np.float32)
        save_probability_file(self.path, starts, probs, {})   # no skip_code

        loaded = load_probability_file(self.path)
        self.assertFalse(loaded['skip_code_is_exact'])
        self.assertFalse(loaded['meta']['has_skip_code'])
        # The zero-probability window is treated as unscored.
        self.assertEqual(list(loaded['skip_code']),
                         [SKIP_NONE, SKIP_INTERRUPTED, SKIP_NONE])

    def test_a_genuine_low_score_is_not_mistaken_for_a_skip(self):
        """An exact mask distinguishes a real 0.0-ish score from a refusal."""
        starts = np.array([0, 6], dtype=np.int32)
        probs = np.array([0.0004, 0.0], dtype=np.float32)
        skip = np.array([SKIP_NONE, SKIP_INTERRUPTED], dtype=np.int8)
        save_probability_file(self.path, starts, probs, {}, skip_code=skip)

        loaded = load_probability_file(self.path)
        self.assertEqual(loaded['skip_code'][0], SKIP_NONE)
        self.assertEqual(loaded['skip_code'][1], SKIP_INTERRUPTED)


class UnscoredSpanTests(unittest.TestCase):
    def test_adjacent_skipped_windows_merge_into_one_span(self):
        spans = _unscored_spans([0, 6, 12], [SKIP_INTERRUPTED] * 3, 12)
        self.assertEqual(spans, [(0.0, 24.0)])

    def test_isolated_skips_stay_separate(self):
        spans = _unscored_spans(
            [0, 6, 12, 60], [SKIP_INTERRUPTED, SKIP_NONE, SKIP_NONE,
                             SKIP_ICA_FAILED], 12)
        self.assertEqual(spans, [(0.0, 12.0), (60.0, 72.0)])

    def test_all_scored_gives_no_bands(self):
        self.assertEqual(_unscored_spans([0, 6], [SKIP_NONE, SKIP_NONE], 12), [])

    def test_missing_mask_is_tolerated(self):
        self.assertEqual(_unscored_spans([0, 6], None, 12), [])

    def test_mismatched_mask_length_is_ignored_rather_than_crashing(self):
        self.assertEqual(_unscored_spans([0, 6, 12], [SKIP_NONE], 12), [])

    def test_the_all_skipped_recording_becomes_one_full_span(self):
        """The aaaaaqtw_s002_t012 case: 49 windows, none assessed."""
        starts = list(range(0, 49 * 6, 6))
        spans = _unscored_spans(starts, [SKIP_INTERRUPTED] * 49, 12)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][0], 0.0)
        self.assertEqual(spans[0][1], starts[-1] + 12.0)


class SkipLabelTests(unittest.TestCase):
    def test_every_code_has_a_reviewer_facing_label(self):
        for code in (SKIP_NONE, SKIP_INTERRUPTED, SKIP_ICA_FAILED):
            self.assertIn(code, SKIP_LABELS)
        for code, text in SKIP_LABELS.items():
            if code != SKIP_NONE:
                self.assertIn('not assessed', text.lower())


if __name__ == '__main__':
    unittest.main()
