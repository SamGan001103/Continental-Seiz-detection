"""Read TUSZ `.csv_bi` reference files and write reviewer output in the
same format."""
import os

# Stamped into every exported annotation. An exported .csv_bi is
# indistinguishable in format from a TUSZ reference file, so without this a
# reviewed export could later be mistaken for clinical ground truth. read_csv_bi
# skips '#' lines, so these headers are format-safe in both directions.
TOOL_NAME = 'Continental Seizure Review (research prototype)'
NOT_FOR_CLINICAL_USE = (
    'research prototype, NOT for diagnostic or clinical use')


def read_csv_bi(path):
    """Return list of dicts: {start, stop, label, confidence}.
    Empty list if the file doesn't exist or has no events."""
    events = []
    if not path or not os.path.exists(path):
        return events
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('channel'):
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            events.append({
                'channel': parts[0],
                'start': float(parts[1]),
                'stop': float(parts[2]),
                'label': parts[3],
                'confidence': float(parts[4]),
            })
    return events


def write_csv_bi(path, bname, duration_s, events,
                 montage_file='$NEDC_NFC/lib/nedc_eas_default_montage.txt',
                 ai_source=None):
    """Write reviewer-confirmed events in TUSZ csv_bi format.

    events    : list of dicts with keys start, stop, label ('seiz'|'bckg'),
                and optional confidence (defaults to 1.0).
    ai_source : which probability source the reviewer worked from
                ('baseline' or 'zuna'). Recorded because a ZUNA-derived review
                was performed against a reconstructed signal, not original EEG.
    """
    with open(path, 'w') as f:
        f.write('# version = csv_v1.0.0\n')
        f.write('# bname = {}\n'.format(bname))
        f.write('# duration = {:.2f} secs\n'.format(duration_s))
        f.write('# montage_file = {}\n'.format(montage_file))
        f.write('# tool = {}\n'.format(TOOL_NAME))
        f.write('# status = {}\n'.format(NOT_FOR_CLINICAL_USE))
        if ai_source:
            f.write('# ai_source = {}{}\n'.format(
                ai_source,
                ' (RECONSTRUCTED signal, not original EEG)'
                if ai_source == 'zuna' else ''))
        f.write('#\n')
        f.write('channel,start_time,stop_time,label,confidence\n')
        for ev in events:
            conf = ev.get('confidence', 1.0)
            f.write('TERM,{:.4f},{:.4f},{},{:.4f}\n'.format(
                float(ev['start']), float(ev['stop']),
                ev.get('label', 'seiz'), float(conf)))
