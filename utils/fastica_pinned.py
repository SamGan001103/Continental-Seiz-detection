"""A version-pinned FastICA, so the ICA stage stops depending on scikit-learn.

Why this exists
---------------
`mne.preprocessing.ICA(method='fastica')` does not implement ICA. It whitens the
data, then calls `sklearn.decomposition.FastICA` and keeps `components_`. So the
numerics of this project's ICA stage are scikit-learn's, not MNE's.

Measured on 25 real windows (`experiments/diag_mne_version.py`):

    MNE      0.19.2 -> 0.23.4    max delta p(seizure)  0.000000   bit-identical
    sklearn  0.22.2 -> 0.24.2    max delta p(seizure)  0.002149   <-- the only
                                                                      sensitive
                                                                      component

That is the whole obstacle to running this application anywhere other than the
machine it was developed on. scikit-learn 0.22.2 has no macOS arm64 wheel, so
"pin scikit-learn" does not solve it; the decomposition has to stop depending on
the installed version at all.

This module is a faithful transcription of **scikit-learn 0.22.2's**
`_fastica.py`, restricted to the path MNE actually uses — `FastICA(whiten=False,
random_state=...)`, `.fit(X)`, read `.components_` — with sklearn's internal
helpers replaced by local equivalents so it depends on nothing but numpy and
scipy.

It is deliberately **not** an improvement. Every apparent oddity below is
reproduced on purpose:

* `_logcosh` mutates its input in place (`np.tanh(x, x)`) and computes `g_x`
  with a Python loop the original comments ask you not to vectorise.
* `_ica_par` returns whatever the fixed-point iteration reached when it hit
  `max_iter`, warning but not raising. On this corpus it frequently does not
  converge, and that non-convergence is baked into the trained operating point
  (`docs/ica_implementation_review.md` §3.5). "Fixing" it changes the model's
  behaviour: a capped `max_iter` once flipped a detection from p=0.902 to
  p=0.0014.
* The `w_init` draw is `random_state.normal(size=(n, n))` in exactly that
  order, because any other draw order gives a different — and equally valid, and
  equally wrong for these weights — decomposition.

Install with :func:`install`, which patches `sklearn.decomposition.FastICA`.
MNE imports that name *inside* its fit method, so patching the module attribute
is enough and no MNE source is touched.
"""
import warnings

import numpy as np
from scipy import linalg


class ConvergenceWarning(UserWarning):
    """Local stand-in for sklearn.exceptions.ConvergenceWarning."""


# ---------------------------------------------------------------------------
# contrast functions — verbatim from sklearn 0.22.2
# ---------------------------------------------------------------------------
def _logcosh(x, fun_args=None):
    alpha = fun_args.get('alpha', 1.0)

    x *= alpha
    gx = np.tanh(x, x)                 # apply the tanh inplace
    g_x = np.empty(x.shape[0])
    for i, gx_i in enumerate(gx):      # please don't vectorize.
        g_x[i] = (alpha * (1 - gx_i ** 2)).mean()
    return gx, g_x


def _exp(x, fun_args):
    exp = np.exp(-(x ** 2) / 2)
    gx = x * exp
    g_x = (1 - x ** 2) * exp
    return gx, g_x.mean(axis=-1)


def _cube(x, fun_args):
    return x ** 3, (3 * x ** 2).mean(axis=-1)


_CONTRAST = {'logcosh': _logcosh, 'exp': _exp, 'cube': _cube}


# ---------------------------------------------------------------------------
# the algorithm — verbatim from sklearn 0.22.2
# ---------------------------------------------------------------------------
def _sym_decorrelation(W):
    """W <- (W * W.T) ^ {-1/2} * W"""
    s, u = linalg.eigh(np.dot(W, W.T))
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)


def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA main loop."""
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in range(max_iter):
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_
                                - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        lim = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
        W = W1
        if lim < tol:
            break
    else:
        warnings.warn('FastICA did not converge. Consider increasing '
                      'tolerance or the maximum number of iterations.',
                      ConvergenceWarning)
    return W, ii + 1


def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """Deflationary FastICA. Present for completeness; MNE uses 'parallel'."""
    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    n_iter = []
    for j in range(n_components):
        w = w_init[j, :].copy()
        w /= np.sqrt((w ** 2).sum())
        for i in range(max_iter):
            gwtx, g_wtx = g(np.dot(w.T, X), fun_args)
            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w
            # Gram-Schmidt against the components already found
            w1 -= np.dot(np.dot(w1, W[:j].T), W[:j])
            w1 /= np.sqrt((w1 ** 2).sum())
            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break
        n_iter.append(i + 1)
        W[j, :] = w
    return W, max(n_iter)


def _check_random_state(seed):
    """Local stand-in for sklearn.utils.check_random_state."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{!r} cannot seed a numpy.random.RandomState'.format(seed))


def _check_array(X):
    """The subset of sklearn's check_array that this path exercises.

    ``FLOAT_DTYPES`` is ``(float64, float32, float16)`` and ``check_array``
    leaves an array alone if its dtype is already one of them, otherwise
    promotes to float64. ``copy`` is ``self.whiten``, i.e. False here.
    """
    X = np.asarray(X)
    if X.dtype not in (np.float64, np.float32, np.float16):
        X = X.astype(np.float64)
    if X.ndim != 2:
        raise ValueError('Expected 2D array, got {}D'.format(X.ndim))
    if X.shape[0] < 2:
        raise ValueError('Found array with {} sample(s); minimum 2.'
                         .format(X.shape[0]))
    return X


class FastICA(object):
    """scikit-learn 0.22.2's FastICA, restricted to the ``whiten=False`` path.

    Only the attributes MNE reads are populated: ``components_``, ``mixing_``,
    ``n_iter_``. Constructing with ``whiten=True`` raises rather than silently
    following an untranscribed path.
    """

    def __init__(self, n_components=None, algorithm='parallel', whiten=True,
                 fun='logcosh', fun_args=None, max_iter=200, tol=1e-4,
                 w_init=None, random_state=None):
        if whiten:
            raise NotImplementedError(
                'utils.fastica_pinned transcribes only the whiten=False path, '
                'which is the one MNE uses. Whitening is done by MNE before '
                'this is called.')
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state

    def fit(self, X, y=None):
        fun_args = {} if self.fun_args is None else self.fun_args
        random_state = _check_random_state(self.random_state)

        # sklearn transposes after validation, so from here X is
        # (n_features, n_samples) and the local names are the transposed ones.
        X = _check_array(X).T

        alpha = fun_args.get('alpha', 1.0)
        if not 1 <= alpha <= 2:
            raise ValueError('alpha must be in [1,2]')

        if self.fun in _CONTRAST:
            g = _CONTRAST[self.fun]
        elif callable(self.fun):
            def g(x, fun_args):
                return self.fun(x, **fun_args)
        else:
            raise ValueError('Unknown function {!r}'.format(self.fun))

        n_samples, n_features = X.shape

        n_components = self.n_components
        if n_components is not None:
            n_components = None
            warnings.warn('Ignoring n_components with whiten=False.')
        if n_components is None:
            n_components = min(n_samples, n_features)

        X1 = np.asarray(X, dtype=X.dtype)      # as_float_array(copy=False)

        w_init = self.w_init
        if w_init is None:
            w_init = np.asarray(
                random_state.normal(size=(n_components, n_components)),
                dtype=X1.dtype)
        else:
            w_init = np.asarray(w_init)
            if w_init.shape != (n_components, n_components):
                raise ValueError('w_init has invalid shape -- should be {}'
                                 .format((n_components, n_components)))

        kwargs = {'tol': self.tol, 'g': g, 'fun_args': fun_args,
                  'max_iter': self.max_iter, 'w_init': w_init}
        if self.algorithm == 'parallel':
            W, n_iter = _ica_par(X1, **kwargs)
        elif self.algorithm == 'deflation':
            W, n_iter = _ica_def(X1, **kwargs)
        else:
            raise ValueError('Invalid algorithm: must be either `parallel` or '
                             '`deflation`.')

        self.n_iter_ = n_iter
        self.components_ = W
        self.mixing_ = linalg.pinv(self.components_)
        self._unmixing = W
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return np.dot(self.components_, _check_array(X).T).T


_ORIGINAL = None


def install():
    """Patch ``sklearn.decomposition.FastICA`` to this implementation.

    MNE does ``from sklearn.decomposition import FastICA`` *inside* its fit
    method, so replacing the module attribute is sufficient and no MNE source
    is modified. Idempotent; returns the class that was replaced.
    """
    global _ORIGINAL
    import sklearn.decomposition as skd
    if _ORIGINAL is None:
        _ORIGINAL = skd.FastICA
    skd.FastICA = FastICA
    return _ORIGINAL


def uninstall():
    """Restore scikit-learn's own FastICA. For tests."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        import sklearn.decomposition as skd
        skd.FastICA = _ORIGINAL
        _ORIGINAL = None


def is_installed():
    import sklearn.decomposition as skd
    return skd.FastICA is FastICA
