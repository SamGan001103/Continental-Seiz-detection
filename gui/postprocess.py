"""Source-paper post-processing: per-second averaging, then discard/concatenate.

The detector emits one probability per overlapping window. The source method does
not threshold those windows directly; it first collapses them to a per-second
probability series and then applies two event-shaping rules. Reproducing the
published operating point requires all three steps, and until this module existed
the GUI and the evaluation scripts applied none of them — they thresholded raw
windows and merged whatever came out, which is why the pilot false-alarm rate was
an order of magnitude above the published figure.

The three stages, in the source method's own terms:

1. **Per-second averaging ("average method").** The predictor scores an
   M-second window but advances only 1 s, so every second is covered by several
   windows. The probability assigned to second *t* is the mean of every window
   overlapping *t*. Averaging is what suppresses isolated single-window spikes.

2. **Concatenate nearby events.** Two positive runs separated by less than
   ``max_gap_s`` are joined, so one seizure is not scored — and penalised — as
   several fragments.

3. **Discard short events.** A positive run shorter than ``min_duration_s`` is
   dropped. The published rationale is empirical: in the training corpus the
   shortest seizure lasts about 5 s and the shortest gap between two seizures is
   about 10 s, which is where the two defaults come from.

Ordering note: the rules are applied concatenate-then-discard. That order is
deliberate and it matters — a genuine seizure broken into three 3-second
fragments survives if it is joined first (9 s total), but is deleted entirely if
each fragment is length-filtered before joining. The upstream reference script
``post_process_code/discard.py`` interleaves the two tests in a single pass and
so implements a slightly stricter variant; ``strict_discard_first=True``
reproduces that behaviour for comparison.

numpy only, no Keras/Qt/MNE, so the same logic is testable headless, callable
from the evaluation scripts, and portable to the planned C/C++ core.
"""
import numpy as np

# Source-method defaults: shortest seizure ~5 s, shortest inter-seizure gap ~10 s.
MIN_EVENT_DURATION_S = 5.0
MAX_MERGE_GAP_S = 10.0


def per_second_probability(window_starts, probs, segment_s, duration_s=None,
                           skip_code=None):
    """Collapse overlapping window probabilities to one value per second.

    Each window starting at ``t`` covers ``[t, t + segment_s)``. A second is
    assigned the MEAN of every window covering it — the source method's
    "average method". Seconds covered by no window are 0.0.

    Returns ``(seconds, p_per_second)`` where ``seconds[i] == i`` (whole-file
    time base starting at 0, so callers can index by absolute recording time).

    skip_code : optional per-window gui.io.infer.SKIP_* code. Windows the
        pipeline declined to score are EXCLUDED from the mean rather than
        averaged in as 0.0. This matters and is one-directional: at 12 s/6 s
        every second is covered by two windows, so a genuine 0.9 sitting beside
        a refused window would otherwise average to 0.45 and fall below a 0.5
        threshold — suppressing a true detection purely because the neighbouring
        window hit an artifact. A second covered only by refused windows stays
        0.0 and is reported as uncovered.
    """
    starts = np.asarray(window_starts, dtype=np.float64).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if starts.size != probs.size:
        raise ValueError('window_starts and probs must be the same length')
    if skip_code is not None:
        skip_code = np.asarray(skip_code).ravel()
        if skip_code.size != starts.size:
            raise ValueError('skip_code must be the same length as probs')
    segment_s = float(segment_s)
    if segment_s <= 0:
        raise ValueError('segment_s must be positive')

    if duration_s is not None and duration_s > 0:
        n_sec = int(np.ceil(float(duration_s)))
    elif starts.size:
        n_sec = int(np.ceil(starts.max() + segment_s))
    else:
        n_sec = 0
    if n_sec <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    total = np.zeros(n_sec, dtype=np.float64)
    count = np.zeros(n_sec, dtype=np.float64)
    for i, (t, p) in enumerate(zip(starts, probs)):
        if skip_code is not None and skip_code[i]:
            continue          # no model score: contributes nothing, not 0.0
        lo = int(np.floor(t))
        hi = int(np.ceil(t + segment_s))
        if hi <= 0 or lo >= n_sec:
            continue
        lo = max(0, lo)
        hi = min(n_sec, hi)
        total[lo:hi] += p
        count[lo:hi] += 1.0

    covered = count > 0
    p_sec = np.zeros(n_sec, dtype=np.float64)
    p_sec[covered] = total[covered] / count[covered]
    return (np.arange(n_sec, dtype=np.int32),
            p_sec.astype(np.float32))


def _runs_above(p_sec, threshold):
    """Return [(start_s, stop_s)] for maximal runs with p >= threshold.

    Stop is exclusive, so a single positive second at index 4 yields (4, 5) —
    a 1-second event, not a zero-length one.
    """
    mask = np.asarray(p_sec, dtype=np.float64) >= float(threshold)
    if not mask.any():
        return []
    # Pad so transitions at the array edges are detected too.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(float(edges[i]), float(edges[i + 1]))
            for i in range(0, len(edges), 2)]


def concatenate_nearby(events, max_gap_s=MAX_MERGE_GAP_S):
    """Join events separated by a gap strictly smaller than ``max_gap_s``."""
    if not events:
        return []
    ordered = sorted((float(s), float(e)) for s, e in events)
    merged = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s - merged[-1][1] < float(max_gap_s):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def discard_short(events, min_duration_s=MIN_EVENT_DURATION_S):
    """Drop events shorter than ``min_duration_s``."""
    return [(s, e) for s, e in events
            if (e - s) >= float(min_duration_s)]


def shape_events(events, min_duration_s=MIN_EVENT_DURATION_S,
                 max_gap_s=MAX_MERGE_GAP_S, strict_discard_first=False):
    """Apply the source method's two event-shaping rules.

    strict_discard_first : reproduce ``post_process_code/discard.py``, which
        length-filters before joining and so deletes short fragments that the
        default order would have rescued by merging.
    """
    if strict_discard_first:
        return concatenate_nearby(
            discard_short(events, min_duration_s), max_gap_s)
    return discard_short(
        concatenate_nearby(events, max_gap_s), min_duration_s)


def decision_curve(window_starts, probs, segment_s, duration_s=None,
                   average=True, skip_code=None):
    """The series the detection threshold is actually applied to.

    Anything that DRAWS a threshold line must draw it over this, or the display
    contradicts the worklist. Measured before this existed: on 42 of 305 sample
    recordings the strip's per-second MAX crossed the line — up to 0.987 — while
    the events came from the per-second MEAN and the list was necessarily empty.

    Returns ``(seconds, values)``; with ``average=False`` the windows are
    expanded to per-second maxima, which is what raw-window thresholding
    effectively decides on.
    """
    if average:
        return per_second_probability(window_starts, probs, segment_s,
                                      duration_s=duration_s,
                                      skip_code=skip_code)
    starts = np.asarray(window_starts, dtype=np.float64).ravel()
    probs_a = np.asarray(probs, dtype=np.float64).ravel()
    if duration_s is not None and duration_s > 0:
        n_sec = int(np.ceil(float(duration_s)))
    elif starts.size:
        n_sec = int(np.ceil(starts.max() + float(segment_s)))
    else:
        n_sec = 0
    if n_sec <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    out = np.zeros(n_sec, dtype=np.float64)
    for i, (t, p) in enumerate(zip(starts, probs_a)):
        if skip_code is not None and np.asarray(skip_code).ravel()[i]:
            continue
        lo = max(0, int(np.floor(t)))
        hi = min(n_sec, int(np.ceil(t + float(segment_s))))
        if hi > lo:
            np.maximum(out[lo:hi], p, out=out[lo:hi])
    return np.arange(n_sec, dtype=np.int32), out.astype(np.float32)


def runs_discarded_by_shaping(window_starts, probs, threshold, segment_s,
                              duration_s=None,
                              min_duration_s=MIN_EVENT_DURATION_S,
                              max_gap_s=MAX_MERGE_GAP_S, average=True,
                              skip_code=None):
    """Spans that cross the threshold but are dropped by event shaping.

    Without these the reviewer sees the decision curve cross the line with
    nothing in the worklist and no explanation. They are a legitimate
    consequence of the source method's minimum-duration rule, not a display
    bug — but they have to be shown as *rejected*, not left invisible.

    Returns ``[(start_s, stop_s)]``.
    """
    _, p_sec = decision_curve(window_starts, probs, segment_s,
                              duration_s=duration_s, average=average,
                              skip_code=skip_code)
    raw = _runs_above(p_sec, threshold)
    kept = set(shape_events(raw, min_duration_s, max_gap_s))
    merged_kept = []
    for s, e in kept:
        merged_kept.append((float(s), float(e)))
    dropped = []
    for s, e in raw:
        inside = any(ks <= s and e <= ke for ks, ke in merged_kept)
        if not inside:
            dropped.append((float(s), float(e)))
    return dropped


def events_from_probs(window_starts, probs, threshold, segment_s,
                      duration_s=None, min_duration_s=MIN_EVENT_DURATION_S,
                      max_gap_s=MAX_MERGE_GAP_S, average=True,
                      strict_discard_first=False):
    """Full source-method decision stage: windows in, shaped events out.

    average : mean-collapse windows to per-second before thresholding (the
        source method). Set False to threshold raw windows — the behaviour the
        GUI had before this module, retained so the two can be compared.

    Set ``min_duration_s=0`` and ``max_gap_s=0`` to disable event shaping.

    Returns ``[(start_s, stop_s, peak_probability)]``.
    """
    segment_s = float(segment_s)
    if average:
        _, p_sec = per_second_probability(
            window_starts, probs, segment_s, duration_s=duration_s)
        raw = _runs_above(p_sec, threshold)
        score_at = lambda s, e: (                             # noqa: E731
            float(p_sec[int(s):int(e)].max()) if int(e) > int(s) else 0.0)
    else:
        starts = np.asarray(window_starts, dtype=np.float64).ravel()
        probs_a = np.asarray(probs, dtype=np.float64).ravel()
        hot = probs_a >= float(threshold)
        raw = [(float(t), float(t) + segment_s)
               for t, h in zip(starts, hot) if h]
        raw = concatenate_nearby(raw, max_gap_s=1e-9)  # join touching windows
        def score_at(s, e):                                    # noqa: E306
            sel = (starts + segment_s > s) & (starts < e) & hot
            return float(probs_a[sel].max()) if sel.any() else 0.0

    shaped = shape_events(raw, min_duration_s, max_gap_s,
                          strict_discard_first=strict_discard_first)
    if duration_s is not None and duration_s > 0:
        d = float(duration_s)
        shaped = [(max(0.0, s), min(d, e)) for s, e in shaped]
        shaped = [(s, e) for s, e in shaped if e > s]
    return [(s, e, score_at(s, e)) for s, e in shaped]
