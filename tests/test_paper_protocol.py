"""Tests for the source paper's window-labelling protocol.

The distinction these pin down is what separates a 0.72 AUC from a 0.82 AUC on
identical probabilities, so it must not drift silently.
"""
import unittest

import numpy as np

from experiments.evaluate_baseline import window_labels
from experiments.replicate_paper_auc import paper_protocol_labels

REFS = [{'start': 30.0, 'stop': 90.0}]


class PaperProtocolLabelTests(unittest.TestCase):
    def test_window_fully_inside_a_seizure_is_positive_and_kept(self):
        labels, keep = paper_protocol_labels([36], REFS, segment_s=12)
        self.assertEqual(list(labels), [1])
        self.assertEqual(list(keep), [True])

    def test_window_touching_no_seizure_is_negative_and_kept(self):
        labels, keep = paper_protocol_labels([0], REFS, segment_s=12)
        self.assertEqual(list(labels), [0])
        self.assertEqual(list(keep), [True])

    def test_window_straddling_the_onset_is_excluded(self):
        # [24, 36) overlaps the seizure starting at 30 but is not inside it.
        labels, keep = paper_protocol_labels([24], REFS, segment_s=12)
        self.assertEqual(list(keep), [False])

    def test_window_straddling_the_offset_is_excluded(self):
        # [84, 96) overlaps the seizure ending at 90 but is not inside it.
        _labels, keep = paper_protocol_labels([84], REFS, segment_s=12)
        self.assertEqual(list(keep), [False])

    def test_window_exactly_filling_the_seizure_is_positive(self):
        labels, keep = paper_protocol_labels(
            [0], [{'start': 0.0, 'stop': 12.0}], segment_s=12)
        self.assertEqual(list(labels), [1])
        self.assertEqual(list(keep), [True])

    def test_boundary_windows_are_the_only_difference_from_any_overlap(self):
        """Both protocols agree everywhere except the straddling windows,
        which the project labels positive and the paper never samples."""
        starts = list(range(0, 120, 6))
        proj = window_labels(starts, REFS, segment_s=12)
        paper, keep = paper_protocol_labels(starts, REFS, segment_s=12)

        np.testing.assert_array_equal(proj[keep], paper[keep])
        self.assertTrue((~keep).any())
        # Every excluded window is one the project protocol calls a seizure.
        self.assertTrue(all(proj[i] == 1 for i in np.flatnonzero(~keep)))

    def test_no_references_gives_all_negative_all_kept(self):
        labels, keep = paper_protocol_labels([0, 6, 12], [], segment_s=12)
        self.assertEqual(list(labels), [0, 0, 0])
        self.assertTrue(keep.all())


if __name__ == '__main__':
    unittest.main()
