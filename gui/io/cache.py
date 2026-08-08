"""Probability cache: a small .npz alongside each EDF."""
import os
import json
import hashlib
import numpy as np


def cache_path_for(edf_path):
    return os.path.splitext(edf_path)[0] + '.probs.npz'


def sha256_file(path, chunk_size=1 << 20):
    """Return the SHA-256 hex digest of a file's bytes, or None on error.

    Used as a content-integrity hash for provenance, unlike the size/mtime
    check used for cache invalidation."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(chunk_size), b''):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def save_probs(edf_path, window_starts, probs, meta=None, skip_code=None):
    """Write per-window probabilities to `<edf_basename>.probs.npz` next
    to the EDF file.

    window_starts : int array, seconds from file start
    probs         : float array, p(seizure) per window
    meta          : dict serialised to JSON in the 'meta' field
    skip_code     : optional per-window gui.io.infer.SKIP_* code
    """
    if meta is None:
        meta = {}
    meta = dict(meta)
    try:
        st = os.stat(edf_path)
        meta.setdefault('edf_size', int(st.st_size))
        meta.setdefault('edf_mtime', float(st.st_mtime))
        meta.setdefault('edf_basename', os.path.basename(edf_path))
    except OSError:
        pass
    save_probability_file(cache_path_for(edf_path), window_starts, probs, meta,
                          skip_code=skip_code)


def load_probs(edf_path):
    p = cache_path_for(edf_path)
    if not os.path.exists(p):
        return None
    loaded = load_probability_file(p)
    if loaded is None:
        return None
    starts = loaded['window_starts']
    probs = loaded['probs']
    meta = loaded['meta']
    try:
        st = os.stat(edf_path)
        if 'edf_size' not in meta or 'edf_mtime' not in meta:
            return None
        if int(meta['edf_size']) != int(st.st_size):
            return None
        cached_mtime = float(meta['edf_mtime'])
        if abs(cached_mtime - float(st.st_mtime)) > 1.0:
            return None
    except OSError:
        return None
    return loaded


CACHE_VERSION = 2   # v2 adds the skip_code array


def save_probability_file(path, window_starts, probs, meta=None,
                          skip_code=None):
    """Write a probability cache to an explicit path.

    This is used for non-EDF-sidecar outputs such as ZUNA probabilities under
    artifacts/. EDF sidecars should still go through save_probs() so source
    file size/mtime provenance is recorded.

    skip_code : optional per-window gui.io.infer.SKIP_* code. Recording it is
        what lets a consumer tell "the model scored this and said no" apart from
        "the pipeline never scored this". Omit it only for caches where that
        distinction genuinely does not exist.
    """
    if meta is None:
        meta = {}
    meta = dict(meta)
    arrays = {
        'window_starts': np.asarray(window_starts, dtype=np.int32),
        'probs': np.asarray(probs, dtype=np.float32),
    }
    if skip_code is not None:
        arrays['skip_code'] = np.asarray(skip_code, dtype=np.int8)
        meta.setdefault('cache_version', CACHE_VERSION)
        meta['has_skip_code'] = True
        meta['n_unscored_windows'] = int(np.count_nonzero(arrays['skip_code']))
    else:
        meta.setdefault('cache_version', 1)
        meta['has_skip_code'] = False
    meta['n_windows'] = int(arrays['probs'].size)
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_probability_file(path):
    """Load a probability cache from an explicit path.

    Returns a dict with ``window_starts``, ``probs``, ``meta``, ``skip_code``
    and ``skip_code_is_exact``.

    For a v2 cache ``skip_code`` is the array that was recorded at inference
    time. For a v1 cache it is *inferred* from ``probs == 0.0`` and
    ``skip_code_is_exact`` is False — a real softmax output is effectively never
    exactly 0.0, so the inference is sound, but callers that report on it should
    say which they had.
    """
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            starts = z['window_starts'].astype(np.int32, copy=False)
            probs = z['probs'].astype(np.float32, copy=False)
            meta = json.loads(str(z['meta']))
            raw_skip = (z['skip_code'].astype(np.int8, copy=False)
                        if 'skip_code' in z else None)
    except Exception:
        return None
    if starts.shape != probs.shape:
        return None
    if len(starts) and np.any(np.diff(starts) < 0):
        return None
    if len(probs) and not np.all(np.isfinite(probs)):
        return None

    exact = raw_skip is not None and raw_skip.shape == probs.shape
    if exact:
        skip_code = raw_skip
    else:
        # Legacy cache: reconstruct from the 0.0 sentinel that the older
        # inference path wrote for both interrupted data and ICA failure. The
        # reason is unrecoverable, so use the generic "interrupted" code.
        from gui.io.infer import SKIP_INTERRUPTED, SKIP_NONE
        skip_code = np.where(probs == 0.0, SKIP_INTERRUPTED,
                             SKIP_NONE).astype(np.int8)
    return {
        'window_starts': starts,
        'probs': probs,
        'meta': meta,
        'skip_code': skip_code,
        'skip_code_is_exact': bool(exact),
    }
