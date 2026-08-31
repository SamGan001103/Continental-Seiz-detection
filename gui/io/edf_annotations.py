"""Read the annotations the recording itself carries.

Until now the application read one kind of annotation: TUSZ `.csv_bi` reference
files, the seizure ground truth. Real clinical recordings carry a second kind
*inside* the EDF+ — what the technician typed while the study was running — and
we displayed none of it.

That is a clinical gap, not a cosmetic one. The Nihon Kohden recording Omid
circulated has 41 of them, and they are the highest-yield minutes in the file:

    1262 s  HVT START            hyperventilation begins
    1292 s  HVT 00:30            ... and its timings
    1770 s  PHOTO 9Hz            photic stimulation, by frequency
    1127 s  hoest                the technician noting a cough

Hyperventilation and photic stimulation are deliberate *seizure-activation
procedures*. A reviewer judging a run of rhythmic activity without knowing a
strobe was flashing at 9 Hz is missing the fact that decides the call — and it
is the most likely explanation for a false positive in exactly that stretch.
The notes are equally load-bearing in the other direction: "hoest" marks the
movement artefact a detector will happily score.

Three rules this module exists to keep:

**These are not ground truth.** They are what the technician observed, not an
adjudicated seizure label. They stay read-only, out of the event list, out of
the export, and out of any recovery metric. Drawing them in the reference
colour would quietly promote a note about a cough into a seizure annotation.

**They must never block a load.** A malformed or absent annotation channel
returns an empty list. A recording that opens today must still open after this
module exists; a file that has no annotations is the normal case, not an error.

**The text is the technician's, verbatim.** No translation, no normalisation —
the Dutch notes above stay Dutch. Cleaning up clinical free text is how meaning
gets lost, and a reviewer who cannot read a note is better served by seeing it
than by seeing our guess at it.
"""
import os


# EDF+ marks an instantaneous annotation with a duration of -1 (and some
# writers use 0). Either way there is nothing to span, so it draws as a line.
def _is_instant(duration):
    try:
        return float(duration) <= 0.0
    except (TypeError, ValueError):
        return True


def read_edf_annotations(path, limit=2000):
    """Annotations embedded in an EDF+ file.

    Returns a list of ``{'onset': float, 'duration': float, 'text': str,
    'instant': bool}``, sorted by onset. Never raises: an unreadable or
    annotation-free file gives ``[]``.

    ``limit`` caps how many are returned. Some vendors emit an annotation per
    impedance check, and a few thousand markers would bury the trace they are
    meant to inform; the caller is told when the cap bit so it can say so
    rather than silently showing part of the record.
    """
    if not path or not os.path.exists(path):
        return []
    try:
        import pyedflib
    except Exception:                                        # noqa: BLE001
        return []
    reader = None
    try:
        reader = pyedflib.EdfReader(os.path.abspath(path))
        onsets, durations, texts = reader.readAnnotations()
    except Exception:                                        # noqa: BLE001
        # A broken annotation channel must not stop a recording opening.
        return []
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:                                # noqa: BLE001
                pass

    out = []
    for onset, duration, text in zip(onsets, durations, texts):
        try:
            t = float(onset)
        except (TypeError, ValueError):
            continue
        label = ('' if text is None else str(text)).strip()
        if not label:
            # An empty annotation is a timekeeping record (EDF+ writes one per
            # data record); it marks nothing a reviewer can act on.
            continue
        try:
            dur = float(duration)
        except (TypeError, ValueError):
            dur = -1.0
        out.append({'onset': t, 'duration': dur, 'text': label,
                    'instant': _is_instant(dur)})
    out.sort(key=lambda a: a['onset'])
    return out[:int(limit)]


def annotation_summary(items, limit=2000):
    """A one-line description for the status bar, or '' if there are none."""
    if not items:
        return ''
    n = len(items)
    capped = ' (showing the first {})'.format(limit) if n >= limit else ''
    return '{} recording annotation{} from the EDF+{}'.format(
        n, '' if n == 1 else 's', capped)
