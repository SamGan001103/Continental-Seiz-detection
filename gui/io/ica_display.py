"""Show the reviewer what the detector saw: ICA artifact removal, on the DISPLAY.

Why this module exists
----------------------
The detector never scores the recording a clinician is looking at. It scores
`ica_arti_remove(window)` — a 19-component ICA decomposition with the top
EOG-correlated component for Fp1 and for Fp2 projected out (see
`utils/preprocessing.py`, and `docs/ica_implementation_review.md` for how that
matches the paper). Until now the reviewer had no way to see that signal, so
"the AI flagged this" could not be argued with: a detection driven by an eye
blink and a detection driven by rhythmic frontal activity look identical on a
raw trace.

This module re-runs the FROZEN function READ-ONLY, on a COPY, purely so the
cleaned trace can be drawn. It is the display path, and it obeys the display
path's one rule:

    DISPLAY  _raw_data -> [this module] -> apply_filters -> apply_montage -> view
    MODEL    _original_data -> ica_arti_remove -> _calc_stft -> ConvLSTM

Nothing here may be fed back to the model. The output is deliberately float32
(the caller's display precision, half the memory to cache) while the detector
consumes MNE's float64 — so an accidental reuse of a cached display array as
model input would not even reproduce the published scores. That is intentional.

Three things the reviewer needs to be told, which the frozen function does not
return and this module recovers:

1. **ICA failed.** `ica_arti_remove` returns None when `ica.fit` raises. The
   detector then skips the window entirely (`SKIP_ICA_FAILED` in
   `gui/io/infer.py`) — it produced no opinion at all.
2. **The raw pass-through.** When neither Fp1 nor Fp2 turns up an EOG
   component, `ica_arti_remove` returns *the untouched input* — not the
   0.1 Hz high-passed signal it just built, the original array. Roughly 10 % of
   windows take this branch (`tests/test_paper_conformance.py`
   `test_DEVIATION_no_eog_found_returns_the_raw_unfiltered_input`), so they
   reach the STFT preprocessed unlike their neighbours. A reviewer comparing
   two adjacent windows is entitled to know that one of them was never cleaned.
3. **Which components were removed.** `ica.exclude` is local to the frozen
   function. It is printed, so it is recovered by capturing stdout rather than
   by re-running a second ICA or by copying the branch logic here — a
   reimplementation would drift from the real thing exactly when it mattered.

Cost, measured on the development workstation over 12 consecutive windows of
synthetic 19-channel EEG, both supported stacks:

    per 12 s window   median 0.09-0.10 s, mean 0.13-0.15 s, worst 0.53-0.74 s
    30 s page, cold   0.33 s (Py3.6/MNE 0.19.2) - 0.60 s (Py3.11/MNE 1.12.1)
    30 s page, warm   < 0.005 s  (every repaint after the first)

So a page turn is one visible pause of about half a second and nothing after
that, which is interactive. Scrolling within a page is free; scrolling across a
block boundary costs one window. Cleaning a whole recording up front is not on
offer: at ~0.15 s per 12 s block that is ~0.75 s per minute of EEG, and a 24 h
ambulatory study would be ~18 min and 1.6 GB of cleaned signal.
"""
import ast
import contextlib
import io
import threading
import time
from collections import OrderedDict

import numpy as np

from gui.io.edf import CHANNELS_19, TARGET_FS
# SEGMENT_S is imported rather than restated so the display grid cannot drift
# away from the grid the windows were scored on; tests/test_paper_conformance.py
# pins it to the paper's 12 s.
from gui.io.infer import (SEGMENT_S, SKIP_ICA_FAILED, SKIP_INTERRUPTED,
                          SKIP_NONE, SKIP_SHORT)

# The stride between detector windows. `compute_probs_from_data(step_s=6)` is
# the default every cache in this repository was written with; a display block
# grid that did not land on multiples of this would show the reviewer a span
# that was never scored as a unit.
STEP_S = 6

# Windows kept in memory. 48 x 19 x 3000 float32 is ~11 MB, which is a couple of
# minutes of scrubbing at a 10 s page. The bound is the point: a 24 h ambulatory
# recording is 14 400 windows, and an unbounded dict of them is 3.3 GB.
CACHE_MAX_WINDOWS = 48


class IcaDisplayInfo(object):
    """What happened when the frozen ICA was re-run for one display window.

    This is the object the GUI shows the reviewer. Every field answers a
    question a clinician asked, or would be misled by not being able to ask:
    did the cleaning run, did it change anything, and which components went.
    """

    def __init__(self):
        self.fs = TARGET_FS
        self.start_s = None        # window start in the recording, seconds
        self.n_samples = 0
        self.seconds = 0.0         # wall time of the ICA run that produced it
        self.cached = False        # True when that time was not paid again

        self.failed = False        # ica_arti_remove returned None
        self.passthrough = False   # it returned its input, untouched
        self.interrupted = False   # detect_interupted_data rejected the window
        self.incomplete = False    # not a full SEGMENT_S window

        self.excluded_fp1 = None   # component index, or None if none was found
        self.excluded_fp2 = None
        self.excluded = []         # what ica.exclude held, in order
        self.indices_known = False  # False when stdout could not be parsed
        self.max_abs_change = 0.0  # µV; 0.0 whenever nothing was subtracted

    # ------------------------------------------------------------------
    def copy(self):
        """A detached copy.

        The cache holds one info per window and hands it to every repaint. The
        span assembler annotates what it gets back (a block can be cleaned and
        still have an uncovered tail), so callers must never receive the cached
        object itself — one annotated span would otherwise re-label that window
        for the rest of the session.
        """
        other = IcaDisplayInfo()
        other.__dict__.update(self.__dict__)
        other.excluded = list(self.excluded)
        return other

    @property
    def cleaned(self):
        """True only when the trace on screen actually differs from the raw."""
        return not (self.failed or self.passthrough or self.interrupted)

    @property
    def skip_code(self):
        """The `gui.io.infer.SKIP_*` code this window carries for the model.

        Shared vocabulary with the probability strip, so "the detector had no
        opinion here" is stated the same way in both places. Note that a
        pass-through window is SKIP_NONE: it *was* scored — just on an
        uncleaned signal.
        """
        if self.incomplete:
            return SKIP_SHORT
        if self.interrupted:
            return SKIP_INTERRUPTED
        if self.failed:
            return SKIP_ICA_FAILED
        return SKIP_NONE

    def label(self):
        """One line for the status bar, written for a clinician, not a dev."""
        if self.incomplete:
            return ('incomplete window — the detector never scored this span')
        if self.interrupted:
            return ('flat or interrupted signal — the detector rejected this '
                    'window before cleaning it')
        if self.failed:
            return ('ICA decomposition failed — the detector produced no score '
                    'for this window')
        if self.passthrough:
            return ('no EOG component found — the detector scored this window '
                    'on the uncleaned signal')
        if not self.indices_known:
            return 'ICA applied (removed components not recoverable)'
        parts = []
        if self.excluded_fp1 is not None:
            parts.append('Fp1 -> IC{}'.format(self.excluded_fp1))
        if self.excluded_fp2 is not None:
            parts.append('Fp2 -> IC{}'.format(self.excluded_fp2))
        return 'ICA removed {}'.format(', '.join(parts))

    def to_dict(self):
        """Plain types only, so this can go straight into a JSON session log."""
        return {
            'fs': int(self.fs),
            'start_s': self.start_s,
            'n_samples': int(self.n_samples),
            'seconds': float(self.seconds),
            'cached': bool(self.cached),
            'failed': bool(self.failed),
            'passthrough': bool(self.passthrough),
            'interrupted': bool(self.interrupted),
            'incomplete': bool(self.incomplete),
            'cleaned': bool(self.cleaned),
            'excluded_fp1': self.excluded_fp1,
            'excluded_fp2': self.excluded_fp2,
            'excluded': list(self.excluded),
            'indices_known': bool(self.indices_known),
            'max_abs_change': float(self.max_abs_change),
            'skip_code': int(self.skip_code),
        }

    def __repr__(self):
        return '<IcaDisplayInfo start={} {}>'.format(self.start_s, self.label())


# ---------------------------------------------------------------- stdout parse

# `ica_arti_remove` prints its EOG indices and its final exclude list. Capturing
# them is the only way to report what was removed without either (a) re-fitting
# a second ICA, which doubles a 0.2 s cost and can converge somewhere else, or
# (b) re-deriving the branch here, which would keep reporting the old rule after
# somebody changed the frozen file. The parse is deliberately strict and
# degrades to indices_known=False rather than guessing.
_EOG_PREFIX = 'eog_indices '
_EXCLUDE_PREFIX = 'ica.exclude '


def _parse_indices(text):
    """Return (fp1, fp2, exclude, known) from ica_arti_remove's stdout.

    MNE's own verbose output lands in the same buffer ("Creating RawArray...",
    "Zeroing out 1 ICA components"), so match on the exact prefixes and require
    the exact shape the frozen function emits: two `eog_indices` lines and one
    `ica.exclude` line. Anything else means another thread printed into our
    redirect and the answer cannot be trusted — say so instead of reporting a
    number the reviewer would have no way to check.
    """
    eog = []
    exclude = []
    n_exclude = 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(_EOG_PREFIX):
            eog.append(line[len(_EOG_PREFIX):].strip())
        elif line.startswith(_EXCLUDE_PREFIX):
            n_exclude += 1
            exclude = [line[len(_EXCLUDE_PREFIX):].strip()]
    if len(eog) != 2 or n_exclude != 1:
        return None, None, [], False
    parsed = []
    for raw in eog + exclude:
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None, None, [], False
        if not isinstance(value, list):
            return None, None, [], False
        try:
            parsed.append([int(v) for v in value])
        except (TypeError, ValueError):
            return None, None, [], False
    fp1 = parsed[0][0] if parsed[0] else None
    fp2 = parsed[1][0] if parsed[1] else None
    return fp1, fp2, parsed[2], True


# redirect_stdout swaps a process-global, so two display cleans running at once
# would shred each other's buffers. The lock also serialises the ICA itself,
# which costs nothing here: FastICA is already using every BLAS thread, and the
# cache means the second caller for a window usually never gets this far.
_STDOUT_LOCK = threading.Lock()


def _run_frozen_ica(segment, fs, channels):
    """Call the frozen ica_arti_remove, capturing what it prints.

    The redirect has a second, incidental benefit: in a PyInstaller --windowed
    build `sys.stdout` is not a real stream, and the authors' `print` calls are
    on the inference path too. For the duration of this call they have somewhere
    valid to go.
    """
    buf = io.StringIO()
    from utils.preprocessing import ica_arti_remove
    with _STDOUT_LOCK:
        with contextlib.redirect_stdout(buf):
            out = ica_arti_remove(segment, fs, channels)
    return out, buf.getvalue()


def _is_interrupted(segment, fs):
    """Did the detector reject this window before ICA ever ran?

    Mirrors `gui/io/infer.py` exactly, transpose included: the reviewer's
    question is "what did the detector do here", and for these windows the
    answer is "nothing at all", not "it cleaned the signal and said no".
    """
    from utils.preprocessing import detect_interupted_data
    buf = io.StringIO()
    with _STDOUT_LOCK:
        with contextlib.redirect_stdout(buf):
            return bool(detect_interupted_data(segment.transpose(), fs))


# ---------------------------------------------------------------------- cache

_CACHE = OrderedDict()          # (recording_id, start_key, fs) -> (array, info)
_CACHE_LOCK = threading.Lock()
_CACHE_LIMIT = CACHE_MAX_WINDOWS
_CACHE_HITS = 0
_CACHE_MISSES = 0


def recording_key(path):
    """A cache identity for a file that changes when the file does.

    Same size/mtime rule the probability sidecars use (`gui/io/cache.py`
    `_fresh_for`), so a recording that was re-exported under the same name does
    not serve a previous recording's cleaned traces.
    """
    import os
    p = os.path.abspath(path)
    try:
        st = os.stat(p)
        return '{}|{}|{}'.format(p, int(st.st_size), int(st.st_mtime))
    except OSError:
        # Better an un-shared cache than a wrong one.
        return '{}|nostat|{}'.format(p, time.time())


def set_cache_limit(n):
    """Change how many cleaned windows stay resident. Evicts immediately."""
    global _CACHE_LIMIT
    with _CACHE_LOCK:
        _CACHE_LIMIT = max(1, int(n))
        _evict_locked()


def cache_clear():
    """Drop every cleaned window. Call this when a new recording is opened."""
    global _CACHE_HITS, _CACHE_MISSES
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_HITS = 0
        _CACHE_MISSES = 0


def cache_stats():
    with _CACHE_LOCK:
        return {
            'size': len(_CACHE),
            'limit': _CACHE_LIMIT,
            'hits': _CACHE_HITS,
            'misses': _CACHE_MISSES,
        }


def _evict_locked():
    while len(_CACHE) > _CACHE_LIMIT:
        _CACHE.popitem(last=False)          # least recently used


def _cache_get(key):
    global _CACHE_HITS, _CACHE_MISSES
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            _CACHE_HITS += 1
            return _CACHE[key]
        _CACHE_MISSES += 1
    return None


def _cache_put(key, value):
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        _evict_locked()


# ------------------------------------------------------------------- cleaning

def _validate(data19, fs):
    data19 = np.asarray(data19)
    if data19.ndim != 2 or data19.shape[0] != len(CHANNELS_19):
        raise RuntimeError(
            'Display ICA requires shape (19, samples) in the canonical '
            'montage order, got {}'.format(data19.shape))
    fs = int(round(float(fs)))
    if fs <= 0:
        raise RuntimeError('Display ICA requires a positive sampling rate, '
                           'got {}'.format(fs))
    return data19, fs


def _clean_uncached(data19, fs, start_s, check_interrupted):
    """One ICA run. Returns (float32 array, IcaDisplayInfo).

    The caller's array is never touched: `np.array(..., copy=True)` up front,
    and the frozen function only ever sees that copy. This is the guarantee the
    whole module rests on — `_raw_data` and `_original_data` are the same object
    in `gui/app.py`, so a display routine that wrote through would silently
    change what the detector scores.
    """
    info = IcaDisplayInfo()
    info.fs = fs
    info.start_s = start_s
    info.n_samples = int(data19.shape[1])

    # A copy, and a copy of the SAME dtype the detector hands over. `infer.py`
    # slices float32 straight out of the loaded EDF, and `create_mne_raw` does
    # `data * 1e-6` before MNE widens it — so scaling in float32 and then
    # widening is not the same arithmetic as widening and then scaling. The
    # difference is ~1e-7 relative, which is nothing to look at and everything
    # to a decomposition: a different rounding can reorder the components, and
    # this module's whole claim is that the indices it reports are the ones the
    # detector actually removed.
    work = np.array(data19, copy=True)

    t0 = time.time()
    if check_interrupted and _is_interrupted(work, fs):
        # The detector never cleaned this window, so neither do we; showing a
        # cleaned trace here would claim the model saw something it did not.
        info.interrupted = True
        info.seconds = time.time() - t0
        return _freeze(work.astype(np.float32)), info

    out, printed = _run_frozen_ica(work, fs, list(CHANNELS_19))
    info.seconds = time.time() - t0

    if out is None:
        info.failed = True
        return _freeze(work.astype(np.float32)), info

    # THE pass-through test. Identity first, because the frozen function's last
    # statement is literally `return data` — the object we handed it comes
    # straight back. The value comparison is the belt to that braces: it holds
    # even if a future MNE makes the function return an equal-but-distinct
    # array, and it is what the requirement asks for ("compare output to input,
    # not reimplement the branch"). Reading `len(ica.exclude) == 0` out of the
    # printed line would be reimplementing it, so that line is only ever used to
    # cross-check, below.
    out = np.asarray(out)
    info.passthrough = bool(
        out is work or (out.shape == work.shape and np.array_equal(out, work)))

    fp1, fp2, excluded, known = _parse_indices(printed)
    info.excluded_fp1 = fp1
    info.excluded_fp2 = fp2
    info.excluded = excluded
    info.indices_known = known
    # If the printed exclude list and the array comparison disagree, something
    # printed into our buffer or the frozen file changed. Keep the array-based
    # verdict (it is the observable truth) and stop claiming to know the indices.
    if known and bool(excluded) == info.passthrough:
        info.indices_known = False
        info.excluded_fp1 = None
        info.excluded_fp2 = None
        info.excluded = []

    if not info.passthrough:
        info.max_abs_change = float(np.max(np.abs(out - work)))
    return _freeze(out.astype(np.float32)), info


def _freeze(arr):
    """Hand back a read-only array.

    Cached windows are shared by every repaint, and the pass-through branch
    returns the very array the reviewer's own copy came from. A caller that
    wrote into it would corrupt the cache for every later view of that window.
    The display chain downstream (`apply_filters`, `apply_montage`) allocates,
    so nothing legitimate is blocked by this.
    """
    # Only ever called on a freshly allocated array (the result of an astype),
    # because ascontiguousarray returns its argument unchanged when it already
    # matches — freezing a caller's array in place would be a surprising thing
    # for a read-only display helper to do.
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.setflags(write=False)
    return arr


def clean_for_display(data19, fs, recording_id=None, start_s=None,
                      check_interrupted=True, use_cache=True):
    """Re-run the detector's ICA on a copy, for display only.

    data19      : [19, N] in `gui.io.edf.CHANNELS_19` order. NEVER mutated.
    fs          : sampling rate; the detector only ever sees TARGET_FS.
    recording_id: cache identity — use `recording_key(edf_path)`. Without it
                  nothing is cached, because there is no safe key.
    start_s     : window start in the recording, for the cache key and for the
                  info object. Required for caching.
    check_interrupted : mirror the detector's pre-ICA rejection test.

    Returns (cleaned [19, N] float32 READ-ONLY, IcaDisplayInfo).

    A window that is not exactly SEGMENT_S long is still cleaned — the reviewer
    may be inspecting an arbitrary selection — but ICA on a different span is
    not the decomposition the model was given, so `info.incomplete` is set and
    the GUI should say so.
    """
    data19, fs = _validate(data19, fs)

    key = None
    if use_cache and recording_id is not None and start_s is not None:
        key = (str(recording_id), int(round(float(start_s) * fs)), int(fs))
        hit = _cache_get(key)
        if hit is not None:
            info = hit[1].copy()
            info.cached = True
            return hit[0], info

    cleaned, info = _clean_uncached(data19, fs, start_s, check_interrupted)
    info.incomplete = int(data19.shape[1]) != int(SEGMENT_S * fs)
    if key is not None:
        _cache_put(key, (cleaned, info))
        # Never hand out the stored object; see IcaDisplayInfo.copy().
        info = info.copy()
    return cleaned, info


# ----------------------------------------------------------- the visible span

def detector_window_starts(n_samples, fs, step_s=STEP_S):
    """The window starts the detector actually scored, in seconds.

    Copied in behaviour, not in code, from `compute_probs_from_data`: integer
    seconds, stride `step_s`, last start such that a full SEGMENT_S fits. Kept
    here so the display grid can be checked against the probability array the
    strip is drawing.
    """
    total_s = int(n_samples / float(fs))
    if total_s < SEGMENT_S:
        return []
    return list(range(0, total_s - SEGMENT_S + 1, int(step_s)))


def _aligned_block_starts(t0_s, t1_s, anchor_s):
    """SEGMENT_S-spaced block starts covering [t0, t1), anchored on the grid.

    Blocks tile without overlap, so every displayed sample comes from exactly
    one detector window and there is no blending to explain to a reviewer. The
    anchor is what lines the tiling up with a particular scored window: pass the
    selected event's window start and that window becomes one whole block, which
    is the case where "what you see is what was scored" has to be literally
    true. It is snapped to the detector's STEP_S grid first, because a block
    boundary at 3 s would correspond to no window that was ever scored.
    """
    anchor = int(round(float(anchor_s) / STEP_S)) * STEP_S
    k0 = int(np.floor((float(t0_s) - anchor) / float(SEGMENT_S)))
    starts = []
    s = anchor + k0 * SEGMENT_S
    while s < float(t1_s):
        starts.append(s)
        s += SEGMENT_S
    return starts


def clean_span_for_display(data19, fs, t0_s, t1_s, recording_id=None,
                           anchor_s=0, check_interrupted=True, use_cache=True):
    """Clean only what is on screen, in blocks matching the detector's grid.

    Cleaning a whole recording is not an option (see the module docstring: ~18
    minutes and 1.6 GB for a 24 h ambulatory study). A visible span is 1-3
    blocks, which is a third of a second, once.

    t0_s, t1_s : the visible range, seconds from file start.
    anchor_s   : a detector window start to align the block grid to (typically
                 the selected event's window). Snapped to the STEP_S grid.

    Returns (span [19, M] float32 WRITEABLE, span_start_s, blocks) where blocks
    is a list of (info, covered_from_s, covered_to_s). The array starts as a
    copy of the raw signal, so any part of the span the detector never scored —
    the tail past the last full window, or a lead-in before t=0 — is shown as
    the reviewer's own trace rather than silently invented. Those regions are
    reported by an `info` with `incomplete` set.
    """
    data19, fs = _validate(data19, fs)
    n = int(data19.shape[1])

    i0 = int(max(0, min(n, round(float(t0_s) * fs))))
    i1 = int(max(i0, min(n, round(float(t1_s) * fs))))
    span_start_s = i0 / float(fs)
    # A fresh, writeable copy: the cached per-window arrays are read-only and
    # shared, and the caller is about to filter and montage this one.
    out = np.array(data19[:, i0:i1], dtype=np.float32, copy=True)

    grid = detector_window_starts(n, fs)
    if not grid or i1 <= i0:
        info = IcaDisplayInfo()
        info.fs = fs
        info.start_s = span_start_s
        info.n_samples = i1 - i0
        info.incomplete = True
        return out, span_start_s, [(info, span_start_s, i1 / float(fs))]
    last_start = grid[-1]
    win = SEGMENT_S * fs

    blocks = []
    for s in _aligned_block_starts(span_start_s, i1 / float(fs), anchor_s):
        # The slice of the recording this block is responsible for.
        b_lo = max(float(s), span_start_s)
        b_hi = min(float(s) + SEGMENT_S, i1 / float(fs))
        if b_hi <= b_lo:
            continue
        # Which scored window can supply it. Blocks that fall off either end of
        # the grid borrow the nearest real window rather than inventing one, and
        # whatever it cannot reach stays raw.
        use = min(max(int(s), 0), int(last_start))
        lo = max(b_lo, float(use))
        hi = min(b_hi, float(use) + SEGMENT_S)
        if hi <= lo:
            info = IcaDisplayInfo()
            info.fs = fs
            info.start_s = b_lo
            info.n_samples = int(round((b_hi - b_lo) * fs))
            info.incomplete = True
            blocks.append((info, b_lo, b_hi))
            continue

        seg = data19[:, use * fs:use * fs + win]
        cleaned, info = clean_for_display(
            seg, fs, recording_id=recording_id, start_s=use,
            check_interrupted=check_interrupted, use_cache=use_cache)

        a = int(round((lo - use) * fs))
        b = int(round((hi - use) * fs))
        dst0 = int(round(lo * fs)) - i0
        # Two independent roundings of the same boundary can disagree by a
        # sample; clamp rather than let a one-sample race raise mid-repaint.
        count = max(0, min(b - a, out.shape[1] - dst0,
                           cleaned.shape[1] - a))
        if count:
            out[:, dst0:dst0 + count] = cleaned[:, a:a + count]
        if hi < b_hi:
            # Tail past the last scored window: left raw, and said so.
            info.incomplete = True
        blocks.append((info, lo, b_hi))

    return out, span_start_s, blocks


def summarise(blocks):
    """One status-bar line for a whole visible span.

    The pass-through count is the number that matters and the one nobody would
    otherwise see: it says how many of the windows on screen were scored without
    ever being cleaned.
    """
    infos = [b[0] if isinstance(b, tuple) else b for b in blocks]
    if not infos:
        return 'ICA: nothing to clean'
    n_pass = sum(1 for i in infos if i.passthrough)
    n_fail = sum(1 for i in infos if i.failed)
    n_int = sum(1 for i in infos if i.interrupted)
    n_ok = sum(1 for i in infos if i.cleaned)
    parts = ['{} cleaned'.format(n_ok)]
    if n_pass:
        parts.append('{} passed through UNCLEANED (no EOG component found)'
                     .format(n_pass))
    if n_fail:
        parts.append('{} ICA failure'.format(n_fail))
    if n_int:
        parts.append('{} rejected as interrupted'.format(n_int))
    return 'ICA over {} window(s): {}'.format(len(infos), '; '.join(parts))
