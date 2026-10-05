"""Write the reviewed events as an EDF+ annotation file, beside the recording.

Why
---
The reviewing neurologist asked what export format would be best. The survey
answer is blunt: **no clinical EEG system imports third-party seizure
annotations.** Natus, Nihon Kohden, Micromed, Persyst and Compumedics all
export their own annotations and import nobody else's. SCORE is a terminology
and a commercial reporting application, not an interchange file. The
standards-track answer, DICOM Supplement 217 neurophysiology waveforms, exists
on paper and is barely implemented.

What is left is the container the recording already arrived in. EDF+ carries
annotations in a reserved `EDF Annotations` signal, the specification is public
and free, and EDFbrowser, MNE-Python, MATLAB's Signal Processing Toolbox and
EDFlib all read it. So the reviewed events go out twice: as the `.csv_bi` the
evaluation needs, and as a small `.review.edf` a second tool can actually open.

The trap this module exists to avoid
------------------------------------
The obvious implementation is wrong in a way that passes every casual check.
`pyedflib` emits **one data record per annotation**, each one data-record long,
so a file carrying three events declares itself a three-second recording. It
reads back perfectly through `pyedflib.readAnnotations` and
`mne.read_annotations` — and then `mne.io.read_raw_edf`, the most common way
anyone will actually open it, **silently drops every event past the declared
duration**:

    RuntimeWarning: Omitted 2 annotation(s) that were outside data range.

On a day-long recording that is every event except the first. A warning, not an
error, and the file looks fine.

So the record count is padded with empty annotations until the file's declared
duration covers the recording. The EDF+ specification says an annotations-only
file should use a data-record duration of 0, which would make this unnecessary,
but `pyedflib` refuses that (`record_duration must be between 0.001 and 60`),
so 60-second records plus padding is the workable version. A 41,030-second
recording becomes 684 records and about 78 KB.

`tests/test_edf_review_annotations.py` asserts the `mne.io.read_raw_edf` path
specifically, because that is the one that fails quietly.
"""
import os

# pyedflib/EDFlib truncates annotation text at 40 characters, silently —
# reproduced in 0.1.22 and 0.1.42. The EDF+ spec itself sets no limit; this is
# our writer's budget, not the ecosystem's. Detail belongs in the provenance
# JSON, which has no such ceiling.
MAX_ANNOTATION_CHARS = 40
RECORD_SECONDS = 60
DISCLAIMER = 'NOT FOR CLINICAL USE - research'


def _label_for(ev):
    """A short, honest label. Kept under the 40-character ceiling."""
    status = (ev.get('status') or '').strip()
    if status == 'added':
        return 'seiz added by reviewer'
    prob = ev.get('prob')
    if prob is None:
        return 'seiz {}'.format(status or 'confirmed')[:MAX_ANNOTATION_CHARS]
    return 'seiz p={:.2f} {}'.format(
        float(prob), status or 'confirmed')[:MAX_ANNOTATION_CHARS]


def write_review_edf(path, events, duration_s, start_datetime=None):
    """Write `<path>`, an annotations-only EDF+ describing `events`.

    events        : dicts with `start`, `stop`, and optionally `status`/`prob`
    duration_s    : the SOURCE recording's duration. The file is padded to
                    cover it, which is what stops readers discarding events.
    start_datetime: the source recording's start, so the events sit at a real
                    wall-clock time rather than at an offset from nothing.

    Returns the number of annotations written, the disclaimer included.
    """
    import pyedflib

    n_records = max(1, int(-(-float(duration_s) // RECORD_SECONDS)))

    tmp = path + '.incoming'
    w = pyedflib.EdfWriter(tmp, 0, file_type=pyedflib.FILETYPE_EDFPLUS)
    written = 0
    try:
        # Belt and braces on the safety marker: the header field AND an
        # annotation. A reader that discards out-of-range annotations would
        # otherwise be able to lose the one statement that matters most.
        try:
            w.setEquipment('NOT FOR CLINICAL USE')
        except Exception:                                    # noqa: BLE001
            pass
        if start_datetime is not None:
            try:
                w.setStartdatetime(start_datetime)
            except Exception:                                # noqa: BLE001
                pass
        w.setDatarecordDuration(RECORD_SECONDS)

        w.writeAnnotation(0.0, -1, DISCLAIMER)
        written += 1
        for ev in events or []:
            onset = float(ev['start'])
            dur = max(0.0, float(ev['stop']) - onset)
            w.writeAnnotation(onset, dur, _label_for(ev))
            written += 1

        # Pad so the declared duration covers the recording. Without this the
        # file claims to be `written` records long and every later event is
        # dropped by mne.io.read_raw_edf with a warning and no error.
        while written < n_records:
            w.writeAnnotation(0.0, -1, '')
            written += 1
    finally:
        w.close()

    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)
    return written


def review_edf_path_for(csv_bi_path):
    """`<stem>.review.edf` beside the annotation file it accompanies."""
    stem = csv_bi_path
    for suffix in ('.csv_bi', '.csv'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem + '.review.edf'
