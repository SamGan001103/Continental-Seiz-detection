"""The pinned FastICA must reproduce scikit-learn 0.22.2 exactly.

This module exists to make the ICA stage independent of the installed
scikit-learn version, which is the single thing preventing this application
from running anywhere other than the machine it was developed on. Measured on
25 real windows:

    MNE      0.19.2 -> 0.23.4                   0.000000   (not the problem)
    sklearn  0.22.2 -> 0.24.2, unpinned         0.002149   (the problem)
    sklearn  0.22.2 -> 0.24.2, pinned           0.000000   (solved)

If any assertion here fails, the released weights are being fed a different
decomposition from the one they were fitted to, and every number in
`docs/RESULTS.md` is invalidated. That is why these are equality assertions
rather than tolerances.
"""
import os
import sys
import unittest
import warnings

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils import fastica_pinned                             # noqa: E402


def whitened(seed=0, n_features=19, n_samples=3000):
    """Data of the shape MNE hands to FastICA: already whitened, so
    ``whiten=False`` is the only path exercised."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n_features, n_samples))
    x -= x.mean(axis=1, keepdims=True)
    cov = np.cov(x)
    d, u = np.linalg.eigh(cov)
    w = (u * (1.0 / np.sqrt(np.maximum(d, 1e-12)))).dot(u.T)
    return w.dot(x).T          # (n_samples, n_features), as sklearn expects


class MatchesScikitLearn(unittest.TestCase):
    """Bit-identity against the scikit-learn actually installed here."""

    @classmethod
    def setUpClass(cls):
        import sklearn
        cls.sklearn_version = sklearn.__version__
        fastica_pinned.uninstall()          # make sure we compare against real
        from sklearn.decomposition import FastICA as RealFastICA
        cls.Real = RealFastICA

    def _both(self, X, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = self.Real(whiten=False, random_state=13, **kw).fit(X)
            b = fastica_pinned.FastICA(whiten=False, random_state=13,
                                       **kw).fit(X)
        return a.components_, b.components_

    def test_components_are_bit_identical(self):
        """Not 'close' — identical. A different unmixing is a different model."""
        if not self.sklearn_version.startswith('0.22'):
            self.skipTest('transcribed from 0.22.2; installed is {}'
                          .format(self.sklearn_version))
        a, b = self._both(whitened())
        np.testing.assert_array_equal(a, b)

    def test_identical_on_a_second_seed(self):
        if not self.sklearn_version.startswith('0.22'):
            self.skipTest('transcribed from 0.22.2')
        a, b = self._both(whitened(seed=7))
        np.testing.assert_array_equal(a, b)

    def test_identical_when_it_does_not_converge(self):
        """Non-convergence is baked into the operating point, so it must match.

        FastICA fails to converge on most windows of this corpus, and the
        returned unmixing is wherever the iteration stopped. Reproducing the
        converged case but not the stopped one would be worse than useless.
        """
        if not self.sklearn_version.startswith('0.22'):
            self.skipTest('transcribed from 0.22.2')
        a, b = self._both(whitened(seed=3), max_iter=2, tol=1e-12)
        np.testing.assert_array_equal(a, b)

    def test_iteration_count_matches(self):
        if not self.sklearn_version.startswith('0.22'):
            self.skipTest('transcribed from 0.22.2')
        X = whitened(seed=5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = self.Real(whiten=False, random_state=13).fit(X)
            b = fastica_pinned.FastICA(whiten=False, random_state=13).fit(X)
        self.assertEqual(a.n_iter_, b.n_iter_)


class Behaviour(unittest.TestCase):
    def tearDown(self):
        fastica_pinned.install()      # leave the process as production has it

    def test_whitening_is_refused_rather_than_silently_wrong(self):
        """Only the whiten=False path was transcribed; the other must raise."""
        with self.assertRaises(NotImplementedError):
            fastica_pinned.FastICA(whiten=True)

    def test_it_does_not_converge_quietly(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fastica_pinned.FastICA(whiten=False, random_state=13,
                                   max_iter=1).fit(whitened(seed=1))
        self.assertTrue(any('did not converge' in str(x.message) for x in w))

    def test_install_is_idempotent_and_reversible(self):
        import sklearn.decomposition as skd
        fastica_pinned.uninstall()
        original = skd.FastICA
        fastica_pinned.install()
        fastica_pinned.install()
        self.assertTrue(fastica_pinned.is_installed())
        fastica_pinned.uninstall()
        self.assertIs(skd.FastICA, original)


class ProductionUsesIt(unittest.TestCase):
    """Importing the preprocessing module must pin the decomposition.

    If this regresses, the application silently goes back to depending on
    whatever scikit-learn happens to be installed — which is exactly the
    condition that made it unportable.
    """

    def test_importing_preprocessing_installs_the_pin(self):
        fastica_pinned.uninstall()
        import importlib
        import utils.preprocessing
        importlib.reload(utils.preprocessing)
        self.assertTrue(fastica_pinned.is_installed())

    def test_mne_picks_up_the_patched_class(self):
        """MNE imports FastICA inside its fit method, so the patch must be on
        the module attribute, not on a name MNE bound at import time."""
        fastica_pinned.install()
        from sklearn.decomposition import FastICA as SeenByCaller
        self.assertIs(SeenByCaller, fastica_pinned.FastICA)


if __name__ == '__main__':
    unittest.main()
