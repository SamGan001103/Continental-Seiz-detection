"""The overview envelope, precomputed on the server instead of on the desk.

Why
---
The review application reads cached detector output and runs no model, but the
PICTURE was still being computed on the clinical workstation, and it is not
cheap. Measured on a 24-hour recording: the first zoom out past five minutes
blocks the interface for 48 seconds while the whole recording is filtered into
its min/max envelope, and every filter change throws that away and pays 39
seconds again.

That is the lag a reviewer actually meets, and it is the last thing in the
system that contradicts the two-part design -- the heavy work was supposed to
happen where the compute is.

The envelope is a good thing to ship. It is deterministic given the recording
and the display settings, it is bounded (ENVELOPE_MAX_COLS caps it at tens of
megabytes whatever the duration), and it is needed the moment a reviewer zooms
out, which they do immediately.

What is stored
--------------
    <stem>.envelope.npz    lo/hi arrays, the step they were built at, and the
                           display settings they are valid for

One envelope per display preset. A reviewer who moves to a combination the
bundle does not carry falls back to building it locally, exactly as before --
slower, but never wrong, and never refused.
"""
import os

import numpy as np

SUFFIX = '.envelope.npz'
MTIME_TOLERANCE_S = 2.0


def path_for(recording_path):
    return os.path.splitext(recording_path)[0] + SUFFIX


def settings_key(hp, lp, notch, montage, fs):
    """The display settings an envelope is valid for.

    Rounded, because these arrive from a combo box as floats and a 70.0 that
    reads back as 69.99999 must not silently invalidate a 38 MB artefact.
    """
    def num(x):
        return None if x is None else round(float(x), 4)
    return (num(hp), num(lp), num(notch), str(montage), int(round(float(fs))))


def _stamp(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return int(st.st_size), float(st.st_mtime)


def save(recording_path, key, step, lo, hi):
    """Write one envelope beside its recording."""
    out = path_for(recording_path)
    stamp = _stamp(recording_path)
    payload = {
        'lo': np.asarray(lo, dtype=np.float32),
        'hi': np.asarray(hi, dtype=np.float32),
        'step': np.array(int(step)),
        'hp': np.array(np.nan if key[0] is None else key[0]),
        'lp': np.array(np.nan if key[1] is None else key[1]),
        'notch': np.array(np.nan if key[2] is None else key[2]),
        'montage': np.array(str(key[3])),
        'fs': np.array(int(key[4])),
    }
    if stamp:
        payload['edf_size'] = np.array(stamp[0])
        payload['edf_mtime'] = np.array(stamp[1])
    tmp = out + '.part.npz'
    np.savez(tmp, **payload)
    if os.path.exists(out):
        os.remove(out)
    os.replace(tmp, out)
    return out


def load(recording_path, key):
    """The stored envelope for these display settings, or None.

    None covers every failure the same way -- absent, stale, damaged, or built
    for different settings -- because the caller's response to all of them is
    identical: build it locally. A cache that cannot be read is a slow open,
    never a wrong picture.
    """
    path = path_for(recording_path)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            stored = (
                None if np.isnan(z['hp']) else round(float(z['hp']), 4),
                None if np.isnan(z['lp']) else round(float(z['lp']), 4),
                None if np.isnan(z['notch']) else round(float(z['notch']), 4),
                str(z['montage']),
                int(z['fs']),
            )
            if stored != tuple(key):
                return None
            if 'edf_size' in z.files:
                now = _stamp(recording_path)
                if now is None or int(z['edf_size']) != now[0]:
                    return None
                if 'edf_mtime' in z.files and \
                        abs(float(z['edf_mtime']) - now[1]) > MTIME_TOLERANCE_S:
                    return None
            return int(z['step']), z['lo'], z['hi']
    except Exception:                                         # noqa: BLE001
        return None
