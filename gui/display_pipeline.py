"""Filter and montage the VISIBLE span of a recording, never the whole thing.

Why this exists
---------------
Until 2026-08-31 every filter or montage change ran `apply_filters` over the
entire recording, then `apply_montage` copied it, then the result was copied
again for the channel inspectors and once more on the way into SignalView.
`apply_filters` casts to float64 and runs up to three zero-phase passes, each
allocating a fresh full-length array. On a 58-minute TUH file that is a few
hundred megabytes and nobody notices. On a 24-hour ambulatory recording the raw
array alone is 1.6 GB, and the refresh path peaked somewhere above 10 GB. The
development machine has 16 GB; it froze hard enough to need a power cycle. A
ward PC has 8.

The fix is not a smaller copy. It is no copy: the display keeps ONE full-length
array — the raw signal the detector was fed — and everything the reviewer sees
is computed for the span on screen, at the moment it is on screen.

Two regimes, chosen by how much time the viewport covers:

* **Reviewing** (up to FULLRATE_MAX_SPAN_S of view): the raw samples for the
  span plus FILTER_PAD_S either side are filtered at full rate, the padding is
  trimmed, the montage is applied, and the result is decimated to pixel width.
  The padding absorbs the zero-phase filters' edge transients, so the interior
  is what whole-recording filtering would have produced — verified in
  tests/test_display_pipeline.py against `apply_filters` on the full array.

* **Overview** (wider than that): each pixel already covers many seconds of
  EEG, so the signal is decimated FIRST and the filters run on the decimated
  series, at its own (lower) rate, with any cutoff above the new Nyquist
  dropped. This is not the same numbers as full-rate filtering and does not
  claim to be — it is a navigation view, and what a reviewer needs from it is
  that slow drift does not swamp the trace and that the montage is right. Both
  hold. Morphology is judged at reviewing zooms, where the first regime is
  exact.

Cost is therefore bounded by the viewport, not the file: a 24-hour recording
and a 10-minute one redraw in the same time.

The detector's ICA view
-----------------------
`SplicedSource` stands in for the full-length copy `_display_source` used to
make. It presents the raw array with one span replaced by the cleaned block,
assembling only the samples actually requested. The raw array is never written
and never duplicated; tests/test_display_model_isolation.py holds that line.

Framework-free: numpy and gui.processing only, so it is testable without Qt
and — like gui.processing — must never be imported by the model path.
"""
import numpy as np

from gui.processing import (CH_NAMES_19, apply_filters, apply_montage,
                            decimate_for_display)

# Widest view that is filtered exactly at full rate. 300 s is ten clinical
# pages; beyond it a 1600-px display shows more than 0.2 s per pixel and the
# view is for navigation. Per redraw this bounds the full-rate work at
# (300 + 2*FILTER_PAD_S) s * 19 ch * 250 Hz of float64 — about 25 MB and a
# fraction of a second — regardless of recording length.
FULLRATE_MAX_SPAN_S = 300.0

# Edge margin discarded after filtering. The slowest display filter is the
# 0.3 Hz second-order high-pass, zero-phase (applied forward and back), whose
# transient is a few seconds; the notch and low-pass settle in well under a
# second. 10 s is comfortably past all of them, and the test measures it.
FILTER_PAD_S = 10.0

# How much signal is filtered and kept at once, so that panning is a slice
# rather than a re-filter. Several screens wide, floored so that a tight
# clinical page does not produce a block barely larger than itself. At 250 Hz
# a 120 s block is 19 ch x 30 000 samples x 4 bytes = 2.3 MB, which is nothing
# against the ~20 ms per redraw it saves while scrolling.
BLOCK_SPAN_FACTOR = 6
BLOCK_MIN_S = 120.0


class SplicedSource(object):
    """A full-length read-only view of `raw` with [i0, i0+len(block)) replaced.

    Nothing is copied at construction. Slicing assembles just the requested
    samples: the raw slice (a view where possible, copied only if the override
    overlaps it) with the overlapping part of `block` pasted in. Supports the
    two access patterns the display uses — `src[:, s0:s1]` and
    `src[row, s0:s1]` — with an optional step for the overview regime.
    """

    def __init__(self, raw, i0, block):
        raw = np.asarray(raw)
        block = np.asarray(block, dtype=raw.dtype)
        if block.ndim != 2 or block.shape[0] != raw.shape[0]:
            raise ValueError('block must be [channels, samples] matching raw')
        i0 = int(i0)
        if i0 < 0 or i0 + block.shape[1] > raw.shape[1]:
            raise ValueError('override span lies outside the recording')
        self._raw = raw
        self._i0 = i0
        self._block = block
        self.shape = raw.shape
        self.dtype = raw.dtype
        self.ndim = raw.ndim

    @property
    def override_span(self):
        return self._i0, self._i0 + self._block.shape[1]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        rows, cols = key
        if not isinstance(cols, slice):
            raise TypeError('SplicedSource supports slice access along time')
        start, stop, step = cols.indices(self.shape[1])
        step = max(1, step)
        out = np.array(self._raw[rows, start:stop:step])     # copy, small
        o0, o1 = self.override_span
        lo, hi = max(start, o0), min(stop, o1)
        if lo < hi:
            js = np.arange(lo, hi)
            sel = ((js - start) % step) == 0
            js = js[sel]
            if js.size:
                ks = (js - start) // step
                out[..., ks] = self._block[rows, js - o0]
        return out

    def __array__(self, dtype=None):
        # Full assembly. Legitimate for short recordings and tests; the display
        # never calls it, which is the point of the class.
        out = self[:, :]
        return out if dtype is None else out.astype(dtype)


class DisplayPipeline(object):
    """Renders filtered, montaged, decimated spans of one recording on demand.

    `source` is the raw [19, N] array or a SplicedSource over it. `montage` is
    a key of gui.processing.MONTAGES, or None to draw the rows as given (used
    when a caller hands SignalView an already-derived array, and by tests).
    """

    def __init__(self, source, fs, hp=None, lp=None, notch=None,
                 montage=None, row_labels=None):
        self.source = source
        self.fs = int(round(float(fs)))
        self.hp, self.lp, self.notch = hp, lp, notch
        self.montage = montage
        self.n_samples = int(source.shape[1])
        # (key, block_start, block_stop, rows) for the last full-rate block.
        self._block = None
        if montage is None:
            self.labels = list(row_labels if row_labels is not None
                               else CH_NAMES_19[:source.shape[0]])
        else:
            # Labels come from the montage table; a one-row probe costs nothing
            # and keeps the label logic in exactly one place.
            _, self.labels = apply_montage(
                np.zeros((source.shape[0], 1), dtype=np.float32), montage)
        self.n_rows = len(self.labels)

    # -- geometry --------------------------------------------------------
    @property
    def duration_s(self):
        return self.n_samples / float(self.fs)

    def _montage(self, data):
        if self.montage is None:
            return np.asarray(data, dtype=np.float32)
        out, _ = apply_montage(data, self.montage)
        return np.asarray(out, dtype=np.float32)

    # -- rendering -------------------------------------------------------
    def render(self, s0, s1, target_px):
        """Rows for samples [s0, s1), decimated to ~target_px points.

        Returns (rows_ds [K, M] float32, t_ds [M] float32 seconds, absolute).
        """
        s0 = max(0, int(s0))
        s1 = min(self.n_samples, int(s1))
        if s1 <= s0:
            k = self.n_rows
            return (np.zeros((k, 0), dtype=np.float32),
                    np.zeros(0, dtype=np.float32))
        span_s = (s1 - s0) / float(self.fs)
        if span_s <= FULLRATE_MAX_SPAN_S:
            return self._render_fullrate(s0, s1, target_px)
        return self._render_overview(s0, s1, target_px)

    def _render_fullrate(self, s0, s1, target_px):
        rows, base = self._fullrate_block(s0, s1)
        rows = rows[:, s0 - base:s1 - base]
        rows_ds, t_local = decimate_for_display(rows, self.fs,
                                                target_px=target_px)
        return rows_ds, (t_local + s0 / float(self.fs)).astype(np.float32)

    def _fullrate_block(self, s0, s1):
        """Filtered, montaged rows covering [s0, s1), from cache when possible.

        Scrolling re-asks for a span a little further along, and re-filtering
        from scratch each time made a scroll step cost ~20 ms. So a block
        several screens wide is filtered at once and kept; panning inside it is
        then only a slice.

        Correctness is unchanged because of WHERE the padding goes: the block
        is filtered with FILTER_PAD_S beyond each end and the padding
        discarded, so every sample inside the block was interior to the
        filtering. Any sub-range of it is therefore exactly what filtering that
        sub-range with its own padding would have produced — which is what
        tests/test_display_pipeline.py measures against whole-file filtering.

        The cache is dropped whenever the filters or montage change, since the
        key includes them.
        """
        key = (self.hp, self.lp, self.notch, self.montage, self.fs)
        cached = self._block
        if (cached is not None and cached[0] == key
                and cached[1] <= s0 and s1 <= cached[2]):
            return cached[3], cached[1]

        # Widen to several screens so a scroll crosses the boundary rarely.
        want = max(int(BLOCK_MIN_S * self.fs), (s1 - s0) * BLOCK_SPAN_FACTOR)
        mid = (s0 + s1) // 2
        b0 = max(0, mid - want // 2)
        b1 = min(self.n_samples, b0 + want)
        b0 = max(0, b1 - want)
        pad = int(FILTER_PAD_S * self.fs)
        p0, p1 = max(0, b0 - pad), min(self.n_samples, b1 + pad)
        raw = self.source[:, p0:p1]
        filt = apply_filters(raw, self.fs, hp=self.hp, lp=self.lp,
                             notch=self.notch)
        filt = filt[:, b0 - p0:b1 - p0]
        rows = self._montage(filt)
        self._block = (key, b0, b1, rows)
        return rows, b0

    def _render_overview(self, s0, s1, target_px):
        stride = max(1, (s1 - s0) // max(1, int(target_px)))
        raw_ds = self.source[:, s0:s1:stride]
        fs_ds = self.fs / float(stride)
        nyq = 0.5 * fs_ds
        # Cutoffs the decimated rate cannot represent are dropped rather than
        # passed to a filter design that would raise or alias into nonsense.
        lp = self.lp if (self.lp and self.lp < nyq) else None
        notch = self.notch if (self.notch and self.notch < nyq) else None
        hp = self.hp if (self.hp and self.hp < nyq) else None
        filt = apply_filters(raw_ds, fs_ds, hp=hp, lp=lp, notch=notch)
        rows = self._montage(filt)
        t = (s0 + np.arange(rows.shape[1]) * stride) / float(self.fs)
        return rows, t.astype(np.float32)

    def channel(self, row, s0, s1):
        """One montage row at full rate over [s0, s1), for the inspector.

        Only the raw channels that row depends on are filtered — one for a
        referential row, two for a bipolar pair, all nineteen for the common
        average — so an inspector window over ten minutes costs a few MB.
        Returns (values float32, t float32 absolute seconds).
        """
        s0 = max(0, int(s0))
        s1 = min(self.n_samples, int(s1))
        if s1 <= s0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        pad = int(FILTER_PAD_S * self.fs)
        p0, p1 = max(0, s0 - pad), min(self.n_samples, s1 + pad)
        if self.montage is None:
            raw = self.source[row, p0:p1][None, :]
            filt = apply_filters(raw, self.fs, self.hp, self.lp, self.notch)
            vals = filt[0, s0 - p0:s1 - p0]
        else:
            raw = self.source[:, p0:p1]
            filt = apply_filters(raw, self.fs, self.hp, self.lp, self.notch)
            rows, _ = apply_montage(filt, self.montage)
            vals = rows[row, s0 - p0:s1 - p0]
        t = (s0 + np.arange(vals.shape[0])) / float(self.fs)
        return (np.ascontiguousarray(vals, dtype=np.float32),
                t.astype(np.float32))
