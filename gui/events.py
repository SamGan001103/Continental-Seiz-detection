"""Event-building helpers for the review GUI.

The GUI stores AI outputs as overlapping probability windows. These helpers
convert those windows into reviewer-facing event intervals and keep reviewed
state stable when the threshold changes.
"""

REVIEWED_STATUSES = ('accepted', 'edited', 'rejected')


def clamp_interval(start, stop, duration_s=None, min_len_s=0.1):
    """Return a sorted interval clamped to [0, duration_s]."""
    s = float(start)
    e = float(stop)
    if e < s:
        s, e = e, s
    if duration_s is not None and duration_s > 0:
        s = max(0.0, min(float(duration_s), s))
        e = max(0.0, min(float(duration_s), e))
    else:
        s = max(0.0, s)
        e = max(0.0, e)
    if e - s < min_len_s:
        if duration_s is not None and duration_s > 0:
            e = min(float(duration_s), s + min_len_s)
            if e - s < min_len_s:
                s = max(0.0, e - min_len_s)
        else:
            e = s + min_len_s
    return s, e


def _overlap(a_start, a_stop, b_start, b_stop):
    return max(0.0, min(a_stop, b_stop) - max(a_start, b_start))


def _match_score(new_ev, old_ev):
    ns = float(new_ev.get('source_start', new_ev['start']))
    ne = float(new_ev.get('source_stop', new_ev['stop']))
    os = float(old_ev.get('source_start', old_ev['start']))
    oe = float(old_ev.get('source_stop', old_ev['stop']))
    inter = _overlap(ns, ne, os, oe)
    if inter <= 0:
        return 0.0
    shorter = max(1e-6, min(ne - ns, oe - os))
    return inter / shorter


def build_proposed_events(window_starts, probs, threshold, segment_s,
                          duration_s=None):
    """Build proposed events from overlapping probability windows.

    Each positive probability covers [window_start, window_start + segment_s].
    Consecutive positive windows merge, and the stop time is the end of the
    last positive window rather than the next negative window start.
    """
    events = []
    in_event = False
    start = 0.0
    last_high_start = 0.0
    peak = 0.0
    segment_s = float(segment_s)

    for t, p in zip(window_starts, probs):
        t = float(t)
        p = float(p)
        if p >= threshold:
            if not in_event:
                in_event = True
                start = t
                peak = p
            else:
                peak = max(peak, p)
            last_high_start = t
        elif in_event:
            stop = last_high_start + segment_s
            start_c, stop_c = clamp_interval(start, stop, duration_s)
            events.append({
                'start': start_c,
                'stop': stop_c,
                'source_start': start,
                'source_stop': stop,
                'prob': peak,
                'status': 'proposed',
            })
            in_event = False

    if in_event:
        stop = last_high_start + segment_s
        start_c, stop_c = clamp_interval(start, stop, duration_s)
        events.append({
            'start': start_c,
            'stop': stop_c,
            'source_start': start,
            'source_stop': stop,
            'prob': peak,
            'status': 'proposed',
        })
    return events


def merge_review_state(proposals, previous_events, duration_s=None,
                       match_threshold=0.5):
    """Carry accepted/rejected/edited events through a threshold rebuild."""
    previous_events = list(previous_events or [])
    reviewed = [
        dict(ev) for ev in previous_events
        if ev.get('status') in REVIEWED_STATUSES
    ]
    used_prev = set()
    merged = []

    for prop in proposals:
        best_idx = None
        best_score = 0.0
        for idx, old in enumerate(reviewed):
            if idx in used_prev:
                continue
            score = _match_score(prop, old)
            if score > best_score:
                best_idx = idx
                best_score = score
        ev = dict(prop)
        if best_idx is not None and best_score >= match_threshold:
            old = reviewed[best_idx]
            used_prev.add(best_idx)
            ev['id'] = old.get('id')
            ev['status'] = old.get('status', 'proposed')
            ev['review_note'] = old.get('review_note')
            ev['prob'] = max(float(ev.get('prob', 0.0)),
                             float(old.get('prob', 0.0)))
            if ev['status'] in REVIEWED_STATUSES:
                ev['start'], ev['stop'] = clamp_interval(
                    old['start'], old['stop'], duration_s)
        merged.append(ev)

    for idx, old in enumerate(reviewed):
        if idx in used_prev:
            continue
        old['start'], old['stop'] = clamp_interval(
            old['start'], old['stop'], duration_s)
        merged.append(old)

    return assign_event_ids(merged)


def assign_event_ids(events):
    """Sort events by time and assign unique stable integer IDs."""
    out = [dict(ev) for ev in events]
    out.sort(key=lambda ev: (float(ev['start']), float(ev['stop'])))
    used = set()
    next_id = 1
    for ev in out:
        eid = ev.get('id')
        if not isinstance(eid, int) or eid <= 0 or eid in used:
            while next_id in used:
                next_id += 1
            eid = next_id
        ev['id'] = eid
        used.add(eid)
    return out


def rebuild_events(window_starts, probs, threshold, segment_s, duration_s=None,
                   previous_events=None, preserve_review=True):
    proposals = build_proposed_events(
        window_starts, probs, threshold, segment_s, duration_s=duration_s)
    if preserve_review:
        return merge_review_state(proposals, previous_events, duration_s)
    return assign_event_ids(proposals)
