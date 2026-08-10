"""Per-recording adaptive score normalisation.

Motivation, measured on this corpus (`docs/RESULTS.md` §3b): the detector's
score distribution has a large per-recording offset. Across the 177 recordings
with at least 20 scored background windows, the 85th percentile of the
*background* score ranges from a median of 0.0032 to a maximum of 0.9885 — a
308x spread — and in 19 of them (11 %) the background 85th percentile already
exceeds the 0.5 detection threshold. The consequence is that **5 of 204
recordings produce 44 % of all false-positive windows** while 111 (54 %)
produce none at all.

A single global threshold cannot serve both populations. This module rescales
each recording against its own score distribution so that one threshold means
the same thing everywhere.

Relationship to the source paper
--------------------------------
Yang et al. apply a **PWI/PEI lens**: "the 85-percentile of PWI and PEI values
for each frequency band over the last two hours as adaptive thresholds",
applied as a confirmation stage to detections from the network.

This is **not a reimplementation of that lens** and must not be described as
one. PWI and PEI are periodicity statistics computed on the raw signal per
frequency band; this module works on the network's output probabilities. What
is carried over is the *principle* — an adaptive reference taken from the
recording's own recent distribution rather than a fixed constant — applied to
the quantity this project actually has evidence about.

Two deliberate departures, both because the deployment context differs:

**Whole recording, not a trailing two hours.** The paper's window is two hours
because its system streams. A review tool opens a recording that is already
complete, so the reference can use the whole file. That also removes the
precondition that blocked this work: no recording in this corpus reaches two
hours (longest 58.3 min), but every recording can supply its own distribution.

**Never loosens.** The reference is floored at the global threshold, so a quiet
recording is left exactly as it is and only a noisy one is tightened. Without
the floor, a recording whose scores are all near zero would be scaled *up* and
would start generating alarms that the raw model never proposed — inventing
false positives in the 54 % of recordings that currently produce none.
"""
import numpy as np

DEFAULT_PERCENTILE = 50.0     # see the note below on why NOT the paper's 85
DEFAULT_FLOOR = 0.5           # eval_config.THRESHOLD; see "never loosens"
MIN_WINDOWS = 20              # below this a percentile is not an estimate

# Why 50 and not the paper's 85. The reference has to be computed over ALL
# windows, because at inference time nothing distinguishes background from
# seizure. At the 85th percentile a seizure-heavy recording inflates its own
# reference and suppresses the very events it should surface: measured on this
# corpus, the recordings tightened at pct 85 contain 30 of the 85 seizures
# (35 %), and event sensitivity collapses from 49.4 % to 20 % by threshold 0.6.
# At the median, a recording must have more than half its windows above the
# threshold to be touched at all — which seizures cannot cause, since they
# occupy about 4 % of recorded time. The recordings tightened at pct 50 hold 2
# of 85 seizures (2 %).


def adaptive_reference(probs, skip_code=None, percentile=DEFAULT_PERCENTILE,
                       floor=DEFAULT_FLOOR, min_windows=MIN_WINDOWS):
    """This recording's own reference level.

    Computed over windows that actually carry a model score. Windows the
    pipeline refused are stored as 0.0 and would drag the percentile down,
    making a noisy recording look quiet and defeating the whole mechanism.

    Returns ``floor`` — making the rescaling a no-op — when the recording is
    quiet, and also when it is too short for a percentile to mean anything. A
    23-second recording yields two windows; a median of two values is not an
    estimate of a background level, and rescaling every score by it would be
    arithmetic dressed up as inference.
    """
    p = np.asarray(probs, dtype=float)
    if skip_code is not None:
        p = p[np.asarray(skip_code) == 0]
    if p.size < min_windows:
        return float(floor)
    return float(max(np.percentile(p, percentile), floor))


def adaptive_scale(probs, skip_code=None, percentile=DEFAULT_PERCENTILE,
                   floor=DEFAULT_FLOOR, min_windows=MIN_WINDOWS):
    """Rescale so this recording's reference level maps onto ``floor``.

    ``out = probs * (floor / reference)``.

    The transform is **strictly monotone within a recording**, which is the
    property that makes it safe: it cannot reorder a recording's own windows,
    so it cannot promote background above a seizure that already outranked it.
    Per-file AUC is therefore unchanged by construction. What changes is
    comparability *between* recordings — which is precisely what a single
    global threshold depends on.

    Returns ``(scaled, reference)``.
    """
    p = np.asarray(probs, dtype=float)
    ref = adaptive_reference(p, skip_code, percentile, floor, min_windows)
    scaled = p * (float(floor) / ref)
    if skip_code is not None:
        # A refused window carries 0.0 as a sentinel, not as a score. Rescaling
        # it is harmless arithmetically but the value must stay exactly 0.0 so
        # downstream skip detection on v1 caches keeps working.
        scaled = np.where(np.asarray(skip_code) == 0, scaled, 0.0)
    return scaled, ref


def is_noisy(probs, skip_code=None, percentile=DEFAULT_PERCENTILE,
             floor=DEFAULT_FLOOR, min_windows=MIN_WINDOWS):
    """True when the recording's own reference exceeds the global threshold.

    These are the recordings the mechanism exists for, and the ones worth
    flagging to a reviewer: a recording whose background routinely scores above
    the detection threshold will present a wall of candidates.
    """
    return adaptive_reference(
        probs, skip_code, percentile, floor, min_windows) > floor
