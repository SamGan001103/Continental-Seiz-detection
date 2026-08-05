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


def per_second_probability(window_starts, probs, segment_s, duration_s=None):
    """Collapse overlapping window probabilities to one value per second.

    Each window starting at ``t`` covers ``[t, t + segment_s)``. A second is
    assigned the MEAN of every window covering it — the source method's
    "average method". Seconds covered by no window are 0.0.

    Returns ``(seconds, p_per_second)`` where ``seconds[i] == i`` (whole-file
    time base starting at 0, so callers can index by absolute recording time).

    Distinct from the ProbStrip display, which takes a per-second MAX. Max is
    the right choice for drawing attention on screen; mean is the right choice
    for scoring, because averaging is what removes isolated spikes.
    """
    starts = np.asarray(window_starts, dtype=np.float64).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if starts.size != probs.size:
        raise ValueError('window_starts and probs must be the same length')
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
    for t, p in zip(starts, probs):
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
