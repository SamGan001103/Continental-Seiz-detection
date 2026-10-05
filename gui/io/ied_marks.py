"""Reviewer-marked interictal discharges, in a file that cannot be mistaken
for a reference annotation.

Why a separate format
---------------------
The reviewing neurologist asked for interictal discharges alongside seizure
detection, so that the spikes are CONTEXT for judging a seizure detection: a
candidate in a region that spikes all day reads very differently from one in a
patient with no interictal activity at all.

This file originally said the discharges could only ever be marked by hand,
because the seizure detector cannot produce them and this project trains
nothing. That was wrong, and it shaped the feature badly. An open-source
detector exists — SpikeNet1, weights published, no training required (see
`gui.io.ied_detect`) — so the marks in this file now come from two places:
automatically, which is the normal case and the one the clinician asked for,
and by hand, which is for confirming, rejecting, or adding the occasional one
the detector missed.

The obvious place to put them is the existing `.csv_bi`, and that is wrong
twice over. `.csv_bi` is **binary and term-based by construction**: two labels,
`seiz` and `bckg`, one label applying to all channels (NEDC AAREADME, TUSZ
v2.0.3/v2.0.6). Interictal marks are per-channel and are neither of those
labels. And there is no NEDC format that takes them either — the TUSZ
multi-class `.csv` label set is seizure TYPES only (fnsz, gnsz, spsz, cpsz,
absz, tnsz, cnsz, tcsz, atsz, mysz, seiz, bckg). The spike label `spsw` belongs
to TUEV, a different corpus with a different annotation format.

That last detail is the one that decided this design. **`spsw` is one character
away from `spsz`, "simple partial seizure"**, which is a real TUSZ label. Inside
a seizure-review tool, emitting `spsw` rows in a TUSZ-shaped file would produce
something valid in neither corpus and one typo away from asserting a seizure
type. The export's whole claim is that a reviewed `.csv_bi` is indistinguishable
from a TUSZ reference; bending it to carry spikes would destroy exactly the
property it exists to have.

So interictal marks go in their own sidecar, `<record>.ied.csv`, with a format
marker that is deliberately not a corpus name, and the same
not-for-clinical-use header every other export carries. Nothing downstream can
mistake it for ground truth, and `.csv_bi` is untouched.

If TUEV compatibility is ever wanted, that is a separate exporter written
against TUEV's actual format — not this one bent into a shape it does not fit.
"""
import os

from gui.io.csv_bi import NOT_FOR_CLINICAL_USE, TOOL_NAME

FORMAT_ID = 'continental_ied_v2'
# Deliberately NOT a NEDC/TUSZ label. `spsw` is TUEV's and sits one character
# from TUSZ's `spsz`; `ied` cannot be confused with either.
LABEL = 'ied'

# v2 added `source` and `score`, because the marks can now come from two very
# different places: a human who looked at the trace, and a detector that has
# not been validated on this data. A file that did not say which is which
# would let an unconfirmed machine guess be read later as a clinical finding —
# precisely the confusion this format exists to prevent. v1 files still read.
SOURCE_REVIEWER = 'reviewer'
COLUMNS_V2 = 'channel,time,label,source,score,note'
COLUMNS_V1 = 'channel,time,label,note'


def ied_path_for(path):
    """`<stem>.ied.csv` for a recording or an annotation file."""
    stem = path
    for suffix in ('.csv_bi', '.review.edf', '.csv', '.edf'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem + '.ied.csv'


def recording_stamp(source_path):
    """Size and mtime of the recording these marks belong to, or None.

    The probability cache already refuses to load against a recording whose
    size or mtime has changed, after copies through a USB drive silently
    invalidated them. This file needs the same guard and for a worse reason:
    a stale `.probs.npz` shows wrong probabilities, but stale marks assert a
    discharge AT A TIMESTAMP, and against the wrong recording that is a
    clinical claim about a moment that does not exist.
    """
    try:
        st = os.stat(source_path)
    except (OSError, TypeError):
        return None
    return {'size': int(st.st_size), 'mtime': float(st.st_mtime)}


def matches_recording(header, source_path, mtime_tolerance_s=2.0):
    """Whether marks carrying `header` belong to the recording at
    `source_path`. Unstamped files (v1, or written without a source) are
    accepted: they predate the guard and refusing them would silently discard
    a reviewer's real work."""
    if not header or 'edf_size' not in header:
        return True
    stamp = recording_stamp(source_path)
    if stamp is None:
        return False
    try:
        if int(header['edf_size']) != stamp['size']:
            return False
        if 'edf_mtime' not in header:
            return True
        return (abs(float(header['edf_mtime']) - stamp['mtime'])
                <= mtime_tolerance_s)
    except (TypeError, ValueError):
        return False


def write_ied_marks(path, bname, duration_s, marks, reviewer=None,
                    start_time=None, source_path=None):
    """Write reviewer-marked interictal discharges.

    marks : dicts with `time` (seconds from file start) and `channel` (the
        derivation label as displayed, e.g. 'Fp1-F7'), optionally `note`.

    A mark is a POINT, not an interval. A spike is a transient of tens of
    milliseconds and the reviewer clicks where they see it; inventing a
    duration would be inventing data.
    """
    marks = sorted(marks or [], key=lambda m: float(m['time']))
    with open(path, 'w') as f:
        f.write('# format = {}\n'.format(FORMAT_ID))
        f.write('# bname = {}\n'.format(bname))
        f.write('# duration = {:.2f} secs\n'.format(float(duration_s)))
        if start_time is not None:
            try:
                f.write('# start_time = {}\n'.format(start_time.isoformat()))
            except (AttributeError, ValueError):
                pass
        stamp = recording_stamp(source_path) if source_path else None
        if stamp:
            f.write('# edf_size = {}\n'.format(stamp['size']))
            f.write('# edf_mtime = {:.6f}\n'.format(stamp['mtime']))
        f.write('# tool = {}\n'.format(TOOL_NAME))
        f.write('# status = {}\n'.format(NOT_FOR_CLINICAL_USE))
        f.write('# note = interictal discharge marks. NOT a NEDC or TUSZ '
                'annotation file and NOT ground truth.\n')
        f.write('# note = the source column says where each mark came from: '
                '{} means a human marked it; anything else names an '
                'automatic detector whose output is UNVALIDATED on this data '
                'and is a candidate for review, not a finding.\n'
                .format(SOURCE_REVIEWER))
        if reviewer:
            f.write('# reviewer = {}\n'.format(reviewer))
        f.write('#\n')
        f.write(COLUMNS_V2 + '\n')
        for m in marks:
            score = m.get('score')
            f.write('{},{:.4f},{},{},{},{}\n'.format(
                str(m.get('channel') or '').replace(',', ' '),
                float(m['time']), LABEL,
                str(m.get('source') or SOURCE_REVIEWER).replace(',', ' '),
                '' if score is None else '{:.4f}'.format(float(score)),
                str(m.get('note') or '').replace(',', ' ')))
    return len(marks)


def read_ied_header(path):
    """The `# key = value` lines at the top of a marks file, as a dict."""
    header = {}
    if not os.path.exists(path):
        return header
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            body = line[1:].strip()
            if '=' not in body:
                continue
            key, _, value = body.partition('=')
            key = key.strip()
            # `note` appears twice and the rest are unique; keeping the first
            # of any repeat avoids a later line overwriting a stamp.
            header.setdefault(key, value.strip())
    return header


def read_ied_marks(path):
    """Read a file this module wrote, v1 or v2. Returns a list of dicts.

    The column header decides which layout to expect, because v1 put the note
    where v2 puts the source — reading a v1 file as v2 would quietly turn a
    reviewer's note into a detector name.
    """
    out = []
    if not os.path.exists(path):
        return out
    v2 = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('channel,'):
                v2 = line.startswith(COLUMNS_V2)
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                t = float(parts[1])
            except ValueError:
                continue
            mark = {'channel': parts[0] or None, 'time': t, 'label': parts[2]}
            if v2:
                mark['source'] = (parts[3] if len(parts) > 3 else
                                  SOURCE_REVIEWER) or SOURCE_REVIEWER
                try:
                    mark['score'] = (float(parts[4])
                                     if len(parts) > 4 and parts[4] else None)
                except ValueError:
                    mark['score'] = None
                mark['note'] = parts[5] if len(parts) > 5 else ''
            else:
                mark['source'] = SOURCE_REVIEWER
                mark['score'] = None
                mark['note'] = parts[3] if len(parts) > 3 else ''
            out.append(mark)
    return out


def summarise(marks):
    """Counts a reviewer would want: total, and per channel.

    Interictal burden is reported per recording and per derivation, because
    where the spikes are is most of the clinical information.
    """
    per_channel = {}
    per_source = {}
    for m in marks or []:
        ch = m.get('channel') or '(unknown)'
        per_channel[ch] = per_channel.get(ch, 0) + 1
        src = m.get('source') or SOURCE_REVIEWER
        per_source[src] = per_source.get(src, 0) + 1
    reviewer = per_source.get(SOURCE_REVIEWER, 0)
    return {'total': len(marks or []),
            'per_channel': dict(sorted(per_channel.items(),
                                       key=lambda kv: (-kv[1], kv[0]))),
            'per_source': dict(sorted(per_source.items(),
                                      key=lambda kv: (-kv[1], kv[0]))),
            # Kept apart so a status line can never add a human's confirmed
            # marks to a detector's unconfirmed guesses and report one number.
            'reviewer': reviewer,
            'automatic': len(marks or []) - reviewer}
