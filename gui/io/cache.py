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


def save_probs(edf_path, window_starts, probs, meta=None):
    """Write per-window probabilities to `<edf_basename>.probs.npz` next
    to the EDF file.

    window_starts : int array, seconds from file start
    probs         : float array, p(seizure) per window
    meta          : dict serialised to JSON in the 'meta' field
    """
    if meta is None:
        meta = {}
    meta = dict(meta)
    meta.setdefault('cache_version', 1)
    try:
        st = os.stat(edf_path)
        meta.setdefault('edf_size', int(st.st_size))
        meta.setdefault('edf_mtime', float(st.st_mtime))
        meta.setdefault('edf_basename', os.path.basename(edf_path))
    except OSError:
        pass
    save_probability_file(cache_path_for(edf_path), window_starts, probs, meta)


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


def save_probability_file(path, window_starts, probs, meta=None):
    """Write a probability cache to an explicit path.

    This is used for non-EDF-sidecar outputs such as ZUNA probabilities under
    artifacts/. EDF sidecars should still go through save_probs() so source
    file size/mtime provenance is recorded.
    """
    if meta is None:
        meta = {}
    meta = dict(meta)
    meta.setdefault('cache_version', 1)
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    np.savez_compressed(
        path,
        window_starts=np.asarray(window_starts, dtype=np.int32),
        probs=np.asarray(probs, dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )


def load_probability_file(path):
    """Load a probability cache from an explicit path."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            starts = z['window_starts'].astype(np.int32, copy=False)
            probs = z['probs'].astype(np.float32, copy=False)
            meta = json.loads(str(z['meta']))
    except Exception:
        return None
    if starts.shape != probs.shape:
        return None
    if len(starts) and np.any(np.diff(starts) < 0):
        return None
    if len(probs) and not np.all(np.isfinite(probs)):
        return None
    return {
        'window_starts': starts,
        'probs': probs,
        'meta': meta,
    }
