"""Design each FIR filter once, not 14,400 times.

The finding
-----------
Scoring a day-long recording filters every 12-second window five times: the
0.1 Hz high-pass before the ICA fit, then inside each of the two
`find_bads_eog` calls a 1-10 Hz band-pass of the 19 sources and another of the
proxy channel. Measured per call on a 3000-sample window:

    filter_data, 19 channels    15.3 ms   of which DESIGN   5.5 ms  (36 %)
    filter_data,  1 channel      9.0 ms   of which DESIGN   5.5 ms  (63 %)
    filter_data, 0.1 Hz         18.4 ms   of which DESIGN  10.6 ms  (57 %)

Designing the filter costs more than applying it whenever there are few
channels. And `mne.filter.filter_data` designs a fresh one on every call, even
though the design depends only on the sampling rate, the cut-offs and the
LENGTH of the data — never on its values. Every window in a recording has the
same sampling rate, the same cut-offs and the same length, so the same
coefficients are computed from scratch 14,400 times.

About 27 ms per window is spent re-deriving filters that are already known.

Why caching it cannot change the answer
---------------------------------------
`create_filter(data, sfreq, l_freq, h_freq, ...)` reads `data` only to learn
its length. Given identical arguments it returns identical coefficients — it is
a pure function of its parameters. Returning a remembered array instead of
recomputing an identical one cannot move a single bit, and the cache key
includes every argument that reaches the design, with the data reduced to its
length.

This follows the pattern `utils/fastica_pinned.py` already uses: replace a
library function deliberately, in one place, with an `install()` the caller
opts into, rather than scattering the change.
"""
import numpy as np

_cache = {}
_original = None
_stats = {'hits': 0, 'misses': 0}


def _key(data, args, kwargs):
    """Everything that reaches the design, with data reduced to its length."""
    if data is None:
        n = None
    else:
        try:
            n = int(np.asarray(data).shape[-1])
        except Exception:                                    # noqa: BLE001
            return None                     # not cacheable; fall through
    try:
        return (n, repr(args), repr(sorted(kwargs.items())))
    except Exception:                                        # noqa: BLE001
        return None


def install():
    """Memoise `mne.filter.create_filter`. Idempotent; returns True if it
    installed, False if it was already in place."""
    global _original
    import mne.filter as mf
    if _original is not None:
        return False
    _original = mf.create_filter

    def cached_create_filter(data=None, *args, **kwargs):
        key = _key(data, args, kwargs)
        if key is None:
            return _original(data, *args, **kwargs)
        hit = _cache.get(key)
        if hit is not None:
            _stats['hits'] += 1
            # Returned read-only so a caller cannot mutate the cached design
            # and quietly corrupt every later window.
            return hit
        _stats['misses'] += 1
        out = _original(data, *args, **kwargs)
        if isinstance(out, np.ndarray):
            out = out.copy()
            out.setflags(write=False)
            _cache[key] = out
        return out

    mf.create_filter = cached_create_filter
    return True


def uninstall():
    """Restore MNE's own function. Used by tests that compare the two."""
    global _original
    if _original is None:
        return False
    import mne.filter as mf
    mf.create_filter = _original
    _original = None
    _cache.clear()
    return True


def is_installed():
    return _original is not None


def stats():
    """(hits, misses, entries) — for reporting what the cache actually did."""
    return _stats['hits'], _stats['misses'], len(_cache)


def clear():
    _cache.clear()
    _stats['hits'] = 0
    _stats['misses'] = 0
