"""Interictal context for a seizure candidate.

Why this is the point of showing discharges at all
--------------------------------------------------
The clinician did not ask for a spike-annotation tool. They asked to see
interictal discharges ALONGSIDE the seizure detections, so the discharges
inform the judgement being made about each detection. Those are different
features, and the difference decides the whole design:

  * a detection in a patient whose left temporal region spikes all day is
    credible, and probably arises from that region;
  * a detection in a recording with no interictal activity anywhere in 24
    hours is far more likely to be artefact — chewing, electrode pop, movement;
  * a rise in spike rate in the minutes before a candidate supports it; a
    candidate that is merely a dense run of the same spikes the patient has all
    day is a different finding from a seizure.

So the useful number is never "how many spikes are there". It is "how does the
interictal activity AROUND THIS CANDIDATE compare with this patient's own
baseline" — the patient is their own control, which is what makes the
comparison meaningful when the detector's absolute threshold is unvalidated.

Pure arithmetic over marks, with no Qt and no model, so the decision rule can
be tested and argued about on its own.
"""

DEFAULT_HALO_S = 300.0      # five minutes either side of the candidate
# Below this many hours of recording outside the candidate's neighbourhood,
# a "baseline rate" is being extrapolated from too little and is not quoted.
MIN_BASELINE_H = 0.5
# A neighbourhood rate this many times the patient's baseline is worth the
# reviewer's attention. Deliberately not a threshold on anything automatic --
# nothing is accepted or rejected by it, it only decides whether a line of
# text is worth showing.
NOTABLE_RATIO = 2.0


def context_for_event(marks, start, stop, duration_s, halo_s=DEFAULT_HALO_S,
                      ictal=None):
    """How this candidate sits against the patient's own INTERICTAL baseline.

    marks      : every interictal mark on the recording (reviewer or detector)
    start, stop: the candidate's extent, in seconds
    duration_s : the whole recording's length, in seconds
    ictal      : other (start, stop) spans known to be seizures, excluded
                 along with this one

    Interictal means BETWEEN seizures, and that word is doing real work here.
    A spike detector slid across a seizure fires on the seizure — measured on
    pat_102_part02, 69% of a day's candidates fell in 4.6% of the recording,
    around and inside one event. Counting those as interictal activity would
    make the "independent evidence" for a candidate partly the candidate
    itself, which is circular and would read as corroboration. So every ictal
    span is removed from both the neighbourhood and the baseline, and the
    discharges inside this event are reported separately as `during` rather
    than folded into any rate.

    Returns counts and rates. `ratio` is None when there is not enough
    recording outside the neighbourhood to claim a baseline — quoting "12x
    baseline" off four minutes of EEG would be worse than saying nothing.
    """
    start, stop = float(start), float(stop)
    if stop < start:
        start, stop = stop, start
    halo_s = max(0.0, float(halo_s))
    duration_s = max(0.0, float(duration_s))

    lo, hi = start - halo_s, stop + halo_s
    times = []
    for m in marks or []:
        try:
            times.append(float(m['time']))
        except (KeyError, TypeError, ValueError):
            continue

    spans = _ictal_spans(ictal, start, stop)

    during = sum(1 for t in times if start <= t <= stop)
    total = len(times)
    # Interictal by construction: inside the halo, outside every seizure.
    near = sum(1 for t in times
               if lo <= t <= hi and not _in_any(t, spans))
    ictal_total = sum(1 for t in times if _in_any(t, spans))

    # The neighbourhood is clipped to the recording: a candidate two minutes
    # from the start has less than a full halo in front of it, and dividing by
    # the nominal width would halve its apparent rate. Seizure time is removed
    # from both spans for the same reason the marks are.
    near_span = max(0.0, min(hi, duration_s) - max(lo, 0.0))
    near_span = max(0.0, near_span - _overlap(spans, max(lo, 0.0),
                                              min(hi, duration_s)))
    away_span = max(0.0, duration_s - _overlap(spans, 0.0, duration_s))
    away_span = max(0.0, away_span - near_span)
    away = total - near - ictal_total

    near_rate = _per_hour(near, near_span)
    base_rate = _per_hour(away, away_span)
    ratio = None
    if (base_rate is not None and base_rate > 0.0
            and away_span / 3600.0 >= MIN_BASELINE_H and near_rate is not None):
        ratio = near_rate / base_rate

    return {
        'total': total,
        'during': during,
        'ictal': ictal_total,
        'near': near,
        'near_rate_per_h': near_rate,
        'baseline_rate_per_h': base_rate,
        'ratio': ratio,
        'halo_s': halo_s,
        'baseline_hours': away_span / 3600.0,
    }


def _ictal_spans(ictal, start, stop):
    """Every seizure span to exclude, this candidate's own included.

    Merged, because an overlapping pair would otherwise have its time
    subtracted twice and could drive a span negative.
    """
    spans = [(start, stop)]
    for pair in (ictal or []):
        try:
            a, b = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        spans.append((min(a, b), max(a, b)))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _in_any(t, spans):
    return any(a <= t <= b for a, b in spans)


def _overlap(spans, lo, hi):
    """Seconds of [lo, hi) covered by `spans`."""
    if hi <= lo:
        return 0.0
    return sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in spans)


def _per_hour(count, span_s):
    if span_s <= 0.0:
        return None
    return count / (span_s / 3600.0)


def describe(ctx):
    """One line a reviewer can read without doing arithmetic.

    Says what was counted and over what, never what to conclude. The detector
    is unvalidated on this data, so a sentence like "supports a seizure" would
    be asserting something nobody has earned.
    """
    if ctx['total'] == 0:
        return ('No interictal discharges anywhere in this recording — '
                'nothing to compare this candidate against.')
    bits = []
    if ctx['during']:
        # Said plainly as "inside", never added to a rate: a spike detector
        # fires on a seizure, so these are not evidence about the seizure.
        bits.append('{} discharge(s) inside the event (not counted as '
                    'interictal)'.format(ctx['during']))
    mins = int(round(ctx['halo_s'] / 60.0))
    if ctx['near_rate_per_h'] is not None:
        bits.append('{:.1f}/h within {} min'.format(
            ctx['near_rate_per_h'], mins))
    if ctx['baseline_rate_per_h'] is not None:
        bits.append('{:.1f}/h elsewhere'.format(ctx['baseline_rate_per_h']))
    line = '; '.join(bits) if bits else '{} total'.format(ctx['total'])
    if ctx['ratio'] is not None and ctx['ratio'] >= NOTABLE_RATIO:
        line += '  ({:.1f}x this patient’s baseline)'.format(ctx['ratio'])
    elif ctx['ratio'] is not None and ctx['ratio'] < 0.5:
        # Not `> 0` as well: a ratio of exactly zero means the neighbourhood is
        # SILENT while the patient spikes elsewhere, which is the strongest
        # version of this observation, and guarding against it excluded
        # precisely the case most worth showing.
        line += '  (quieter than this patient’s baseline)'
    return line


def density(marks, duration_s, bins, t0=0.0, t1=None):
    """Discharges per hour across the recording, for the overview strip.

    Individual marks are unreadable over 24 hours — one pixel is the better
    part of a minute — so what the strip can usefully carry is a rate curve.
    Returns a list of `bins` rates in discharges per hour.
    """
    bins = max(1, int(bins))
    t1 = float(duration_s if t1 is None else t1)
    t0 = float(t0)
    if t1 <= t0:
        return [0.0] * bins
    width = (t1 - t0) / bins
    counts = [0] * bins
    for m in marks or []:
        try:
            t = float(m['time'])
        except (KeyError, TypeError, ValueError):
            continue
        if t0 <= t < t1:
            counts[min(bins - 1, int((t - t0) / width))] += 1
        elif t == t1:
            counts[bins - 1] += 1
    per_h = 3600.0 / width
    return [c * per_h for c in counts]
