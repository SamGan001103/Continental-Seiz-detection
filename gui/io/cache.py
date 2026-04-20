"""Probability cache: a small .npz alongside each EDF."""
import os
import json
import numpy as np


def cache_path_for(edf_path):
    return os.path.splitext(edf_path)[0] + '.probs.npz'


def save_probs(edf_path, window_starts, probs, meta=None):
    """Write per-window probabilities to `<edf_basename>.probs.npz` next
    to the EDF file.

    window_starts : int array, seconds from file start
    probs         : float array, p(seizure) per window
    meta          : dict serialised to JSON in the 'meta' field
    """
    if meta is None:
        meta = {}
    np.savez_compressed(
        cache_path_for(edf_path),
        window_starts=np.asarray(window_starts, dtype=np.int32),
        probs=np.asarray(probs, dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )


def load_probs(edf_path):
    p = cache_path_for(edf_path)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {
            'window_starts': z['window_starts'],
            'probs': z['probs'],
            'meta': json.loads(str(z['meta'])),
        }
