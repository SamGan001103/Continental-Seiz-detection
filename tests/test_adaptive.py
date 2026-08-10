"""Safety properties of per-recording adaptive normalisation.

`gui/adaptive.py` rescales a recording's scores against its own distribution so
that one global threshold means the same thing across recordings. Everything
here guards a property that, if it broke, would make the mechanism *unsafe*
rather than merely ineffective — this is a component that can only ever lower
scores, in a tool whose failure mode is a seizure never shown.

The measured evidence for the design choices is in `docs/RESULTS.md` §3c.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.adaptive import (                                    # noqa: E402
    DEFAULT_FLOOR, DEFAULT_PERCENTILE, MIN_WINDOWS,
    adaptive_reference, adaptive_scale, is_noisy,
)


def quiet(n=100):
    """A recording the detector is confident about: almost all background."""
    return np.full(n, 0.01)


def noisy(n=100):
    """A pathological recording: most windows above the threshold."""
    return np.full(n, 0.95)


class NeverLoosens(unittest.TestCase):
    """The single most important property.

    If the reference were not floored, a quiet recording would be scaled *up*
    and would start proposing events the raw model never proposed — inventing
    false positives in the 54 % of recordings that currently produce none.
    """

    def test_a_quiet_recording_is_left_exactly_alone(self):
        p = quiet()
        out, ref = adaptive_scale(p)
        self.assertEqual(ref, DEFAULT_FLOOR)
        np.testing.assert_allclose(out, p)

    def test_scores_are_never_increased(self):
        rng = np.random.RandomState(0)
        for _ in range(50):
            p = rng.uniform(0, 1, 80)
            out, _ = adaptive_scale(p)
            self.assertTrue(np.all(out <= p + 1e-12),
                            'adaptive scaling raised a score')

    def test_a_noisy_recording_is_tightened(self):
        out, ref = adaptive_scale(noisy())
        self.assertGreater(ref, DEFAULT_FLOOR)
        self.assertTrue(np.all(out < noisy()))


class MonotoneWithinARecording(unittest.TestCase):
    """It must not reorder a recording's own windows.

    A per-recording rescale is a positive multiplier, so within-recording
    ranking — and therefore per-file AUC — is unchanged by construction. That
    is what makes it incapable of promoting background above a seizure that
    already outranked it.
    """

    def test_ordering_is_preserved(self):
        rng = np.random.RandomState(1)
        p = rng.uniform(0, 1, 200)
        out, _ = adaptive_scale(p)
        np.testing.assert_array_equal(np.argsort(p), np.argsort(out))

    def test_per_file_auc_is_unchanged(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.RandomState(2)
        p = rng.uniform(0, 1, 300)
        y = (rng.uniform(0, 1, 300) < 0.1).astype(int)
        out, _ = adaptive_scale(p)
        self.assertAlmostEqual(roc_auc_score(y, p), roc_auc_score(y, out),
                               places=12)


class TooShortToEstimate(unittest.TestCase):
    """A percentile of two values is not an estimate of a background level.

    Two recordings in this corpus are 19 and 23 seconds — about two windows
    each. Rescaling every score by the median of two numbers is arithmetic
    dressed up as inference.
    """

    def test_a_short_recording_is_a_no_op_even_if_it_looks_noisy(self):
        p = noisy(MIN_WINDOWS - 1)
        out, ref = adaptive_scale(p)
        self.assertEqual(ref, DEFAULT_FLOOR)
        np.testing.assert_allclose(out, p)

    def test_the_threshold_is_exactly_min_windows(self):
        self.assertTrue(is_noisy(noisy(MIN_WINDOWS)))
        self.assertFalse(is_noisy(noisy(MIN_WINDOWS - 1)))

    def test_an_empty_recording_does_not_raise(self):
        out, ref = adaptive_scale(np.array([]))
        self.assertEqual(ref, DEFAULT_FLOOR)
        self.assertEqual(out.size, 0)


class RefusedWindowsAreNotScores(unittest.TestCase):
    """A refused window carries 0.0 as a sentinel, not as a probability."""

    def test_refused_windows_are_excluded_from_the_reference(self):
        """Otherwise a noisy recording with many refusals looks quiet."""
        p = np.concatenate([noisy(60), np.zeros(60)])
        skip = np.concatenate([np.zeros(60, int), np.ones(60, int)])
        with_skip = adaptive_reference(p, skip)
        without = adaptive_reference(p, None)
        self.assertGreater(with_skip, without,
                           'refused windows dragged the reference down')

    def test_the_sentinel_stays_exactly_zero(self):
        p = np.concatenate([noisy(40), np.zeros(40)])
        skip = np.concatenate([np.zeros(40, int), np.ones(40, int)])
        out, _ = adaptive_scale(p, skip)
        self.assertTrue(np.all(out[skip != 0] == 0.0))


class PercentileChoice(unittest.TestCase):
    """Why the median and not the paper's 85th percentile.

    The reference must be computed over ALL windows, because at inference time
    nothing distinguishes background from seizure. At the 85th percentile a
    seizure-heavy recording inflates its own reference and suppresses the
    events it should surface — measured, the recordings tightened at pct 85
    hold 30 of 85 seizures, against 2 of 85 at the median.
    """

    def test_the_default_is_the_median(self):
        self.assertEqual(DEFAULT_PERCENTILE, 50.0)

    def test_a_seizure_minority_cannot_trigger_tightening_at_the_median(self):
        """20 % of windows ictal — far above the ~4 % seen in this corpus."""
        p = np.concatenate([np.full(80, 0.01), np.full(20, 0.99)])
        self.assertFalse(is_noisy(p, percentile=50.0))

    def test_the_same_recording_would_be_tightened_at_the_85th(self):
        p = np.concatenate([np.full(80, 0.01), np.full(20, 0.99)])
        self.assertTrue(is_noisy(p, percentile=85.0),
                        'this is the failure mode the median avoids')


if __name__ == '__main__':
    unittest.main()
