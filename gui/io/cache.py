"""Probability cache: a small .npz alongside each EDF.

Sidecar-next-to-the-EDF is the preferred location because it travels with the
recording and needs no bookkeeping. It is not always available: on a clinical
workstation the recordings usually sit on a read-only network share or a
PACS export directory, and a failed sidecar write would silently discard a
ten-minute inference run. When the sidecar cannot be written the cache falls
back to a per-user directory, and reads check both.
"""
import os
import json
import hashlib
import numpy as np

from gui.paths import writable_root


def cache_path_for(edf_path):
    return os.path.splitext(edf_path)[0] + '.probs.npz'


def fallback_cache_path_for(edf_path):
    """Per-user cache location, used when the EDF's directory is not writable.

    Keyed by a digest of the absolute path so two recordings with the same
    basename in different folders do not collide; the basename is kept as a
    prefix purely so the directory is legible to a human clearing it out.
    """
    p = os.path.abspath(edf_path)
    key = hashlib.sha256(p.encode('utf-8', 'replace')).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(p))[0]
    return os.path.join(writable_root(), 'cache',
                        '{}.{}.probs.npz'.format(stem, key))


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
    """Write per-window probabilities for an EDF, and return where they went.

    Prefers `<edf_basename>.probs.npz` beside the recording. Falls back to the
    per-user cache when that directory is not writable, rather than raising and
    discarding a multi-minute inference run.

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
    try:
        save_probability_file(cache_path_for(edf_path), window_starts, probs,
                              meta, skip_code=skip_code)
        return cache_path_for(edf_path)
    except (OSError, IOError):
        # Read-only recording directory. Do not lose the run.
        fb = fallback_cache_path_for(edf_path)
        save_probability_file(fb, window_starts, probs, meta,
                              skip_code=skip_code)
        return fb


def _fresh_for(edf_path, loaded):
    """True if a loaded cache still matches the EDF's size and mtime."""
    if loaded is None:
        return False
    meta = loaded['meta']
    try:
        st = os.stat(edf_path)
    except OSError:
        return False
    if 'edf_size' not in meta or 'edf_mtime' not in meta:
        return False
    if int(meta['edf_size']) != int(st.st_size):
        return False
    return abs(float(meta['edf_mtime']) - float(st.st_mtime)) <= 1.0


def load_probs(edf_path):
    for p in (cache_path_for(edf_path), fallback_cache_path_for(edf_path)):
        if not os.path.exists(p):
            continue
        loaded = load_probability_file(p)
        if _fresh_for(edf_path, loaded):
            return loaded
    return None


CACHE_VERSION = 2   # v2 adds the skip_code array


def environment_stamp():
    """What produced these probabilities, beyond the model weights.

    Caches written before this existed recorded the weights, the stride and the
    window length, but nothing about the *environment*. That turned out to
    matter: probabilities regenerated from unchanged EDFs with unchanged code
    differ from the stored caches by up to 0.107, while two fresh runs are
    bit-identical to each other. The pipeline is deterministic; the caches were
    simply produced under conditions nobody recorded, and with no stamp there is
    no way to tell which condition changed.

    Kept to cheap, always-available values so writing a cache never fails
    because provenance could not be collected.
    """
    stamp = {}
    try:
        import mne
        stamp['mne'] = str(mne.__version__)
    except Exception:
        pass
    try:
        import numpy
        stamp['numpy'] = str(numpy.__version__)
    except Exception:
        pass
    try:
        import scipy
        stamp['scipy'] = str(scipy.__version__)
    except Exception:
        pass
    # scikit-learn decides the ICA numerics and was the one library missing.
    # mne.preprocessing.ICA(method='fastica') whitens and then calls
    # sklearn.decomposition.FastICA, so the installed sklearn — not MNE — is
    # what the decomposition actually came from. Every cache written before
    # this omitted it, which is why a measured drift of up to 0.43 between two
    # stacks could not be attributed from the stamp alone.
    try:
        import sklearn
        stamp['sklearn'] = str(sklearn.__version__)
    except Exception:
        pass
    # ...and whether that version was allowed to matter. utils/fastica_pinned
    # replaces the implementation with a transcription of 0.22.2, so a cache
    # written with the pin installed is comparable to one written under a
    # different sklearn, and one written without it is not. Recording the
    # sklearn version without this flag would be actively misleading.
    #
    # This reads the state at write time. utils/preprocessing installs the pin
    # when it is imported, and any path that produced probabilities has
    # imported it — so a real cache always records True. Calling this function
    # in isolation reports False, which is accurate for that process and not a
    # statement about the cache.
    try:
        from utils.fastica_pinned import is_installed
        stamp['fastica_pinned'] = bool(is_installed())
    except Exception:
        pass
    try:
        import platform
        stamp['python'] = platform.python_version()
        stamp['host'] = platform.node()
        stamp['platform'] = platform.system()
        # arm64 vs x86-64: measured drift between two machines running the
        # *same* library versions is small but not zero, and cannot be seen
        # without this.
        stamp['machine'] = platform.machine()
    except Exception:
        pass
    # Never imported for the stamp's sake — by the time a cache is written the
    # inference has run, so this is either already loaded or genuinely absent.
    try:
        import sys
        tf = sys.modules.get('tensorflow')
        if tf is not None:
            stamp['tensorflow'] = str(tf.__version__)
            # oneDNN reorders floating-point operations on x86-64 and TF says
            # so at import. Off on arm64, on by default elsewhere.
            import os
            stamp['onednn'] = os.environ.get('TF_ENABLE_ONEDNN_OPTS', '1')
    except Exception:
        pass
    return stamp


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
    # Recorded under one key so a reader can tell at a glance whether two caches
    # came from the same environment, without comparing a dozen loose fields.
    meta.setdefault('env', environment_stamp())
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
