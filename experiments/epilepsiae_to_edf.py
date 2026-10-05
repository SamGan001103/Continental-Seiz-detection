"""Turn EPILEPSIAE `.data` recordings into EDF the application can open.

Why this exists
---------------
The day-long patient data the lab supplied is EPILEPSIAE in its native format:
one-hour chunks of raw interleaved int16 (`.data`) described by a plain-text
sidecar (`.head`). The application reads EDF and nothing else, so none of it is
usable until it is converted.

Three things about the source make a naive converter produce a file that looks
correct and is quietly wrong.

**1. The hour blocks are not contiguous.** Measured on the three complete
patients, the recorder loses time between blocks: usually one second, sometimes
minutes, once 71.5 minutes. Concatenating pat_102's 173 blocks end to end drifts
by 5,465 s — over ninety minutes — by the end of the week, and every seizure
timestamp after the first gap lands somewhere it did not happen. Nothing errors.
So this tool places every block by its ABSOLUTE start time and fills the gaps,
rather than appending block after block.

**2. Blocks can overlap.** pat_102 has a pair whose timestamps overlap by 60 s.
The later block wins and the earlier one is truncated, so no sample is ever
written twice and the timeline stays monotonic.

**3. The zip is truncated.** The archive that arrived has no central directory,
so `zipfile` refuses it. It is also STORED — no compression — so every member's
bytes sit in the file verbatim behind a local header. This reads them at their
byte offsets and never needs the directory, which is also why nothing has to be
unpacked to disk first.

Gaps are written as zeros rather than skipped, because zeros keep wall-clock
alignment exact and the detector already handles them correctly:
`detect_interupted_data` rejects a window whose channels are near-DC, so a gap
is scored as "not assessed" and drawn as a grey band, which is the truth. A
`.gaps.json` sidecar records every one of them so nothing is hidden.

Seizure times come from `recording_info/Surface`, not from the signal archive,
and are written beside each EDF as a TUSZ-style `.csv_bi` — the format the
application already reads for reference annotations.

Layout of the raw data was determined empirically, not assumed: little-endian
int16, INTERLEAVED (sample 0 of every channel, then sample 1). Read
channel-major instead, the lag-1 autocorrelation of the result collapses from
0.89 to 0.10 — the difference between EEG and noise.

Usage
-----
    python experiments/epilepsiae_to_edf.py \
        --archive D:/D_Downloads/Epilepsiase.zip \
        --meta    D:/D_Downloads/Surface.zip \
        --out     D:/D_Downloads/epilepsiae_edf \
        --patients pat_102 pat_21902 pat_22602

Add `--dry-run` to print the plan — output files, durations, gaps and seizure
placement — without writing a byte. Do that first.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import re
import struct
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

XLNS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
ZIP_LOCAL_SIG = b'PK\x03\x04'
ZIP_DD_SIG = b'PK\x07\x08'
NAME_RE = re.compile(rb'^[\w./+\- ()\[\]]{1,200}$')
SCAN_CHUNK = 1 << 26

# Excel serial day zero. Excel's 1900 leap-year bug means serials >= 61 are
# offset by two days from a naive 1900-01-01 epoch; 1899-12-30 absorbs it.
EXCEL_EPOCH = _dt.datetime(1899, 12, 30)
DAY_S = 86400.0

DEFAULT_SPLIT_H = 24.0
# The 19 the detector needs, in the order utils/params_common_electrodes.txt
# lists them. Labels are written TUH-style and UPPERCASE because pyst matches a
# parameter name as a SUBSTRING of the EDF label, with whitespace stripped and
# no case folding: 'FP1' has to appear literally inside 'EEG FP1-REF'.
EEG_19 = ('FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ',
          'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2')


# ---------------------------------------------------------------------------
# reading the metadata workbook
# ---------------------------------------------------------------------------
def _xlsx_table(zf, member):
    """Rows of an xlsx worksheet as dicts. Avoids a pandas dependency in the
    environment that produces the thesis numbers."""
    xb = zipfile.ZipFile(io.BytesIO(zf.read(member)))
    shared = []
    if 'xl/sharedStrings.xml' in xb.namelist():
        root = ET.fromstring(xb.read('xl/sharedStrings.xml'))
        for si in root.findall(XLNS + 'si'):
            shared.append(''.join(t.text or '' for t in si.iter(XLNS + 't')))
    sheet = sorted(n for n in xb.namelist()
                   if n.startswith('xl/worksheets/sheet') and
                   n.endswith('.xml'))[0]
    rows = []
    for row in ET.fromstring(xb.read(sheet)).iter(XLNS + 'row'):
        cells = {}
        for c in row.findall(XLNS + 'c'):
            v = c.find(XLNS + 'v')
            t = c.get('t')
            if t == 's' and v is not None:
                val = shared[int(v.text)]
            elif t == 'inlineStr':
                isn = c.find(XLNS + 'is')
                val = ''.join(x.text or '' for x in isn.iter(XLNS + 't')) \
                    if isn is not None else ''
            else:
                val = v.text if v is not None else ''
            cells[re.match(r'([A-Z]+)', c.get('r', 'A1')).group(1)] = val
        if cells:
            rows.append(cells)
    if not rows:
        return []
    order = sorted(rows[0], key=lambda k: (len(k), k))
    hdr = [rows[0][k].strip() for k in order]
    return [{hdr[i]: r.get(order[i], '') for i in range(len(order))}
            for r in rows[1:]]


def read_metadata(meta_zip):
    """Return (blocks_by_patient, seizures_by_patient).

    blocks:   list of dicts with filename, t0 (datetime), n_samples, fs
    seizures: list of dicts with filename, onset (datetime), duration, label
    """
    zf = zipfile.ZipFile(meta_zip)
    names = zf.namelist()

    def pick(stem):
        for n in names:
            if n.endswith(stem):
                return n
        raise RuntimeError('{} not found in {}'.format(stem, meta_zip))

    blocks = _xlsx_table(zf, pick('epilepsia_recording_blocks.xlsx'))
    seiz = _xlsx_table(zf, pick('epilepsia_seizure_master.xlsx'))

    by_pat = defaultdict(list)
    idx2folder = {}
    for b in blocks:
        folder = (b.get('folder') or '').strip()
        if not folder:
            continue
        idx2folder[str(b.get('pat', '')).strip()] = folder
        try:
            t0 = EXCEL_EPOCH + _dt.timedelta(days=float(b['begin']))
            n = int(b['samples'])
            fs = int(round(n / 3600.0)) if n else 256
        except (KeyError, TypeError, ValueError):
            continue
        by_pat[folder].append({
            'filename': (b.get('filename') or '').strip(),
            't0': t0, 'n_samples': n, 'fs': fs,
            'n_channels': int(b.get('ch.') or 29),
            'factor': float(b.get('factor') or 0.165),
        })
    for v in by_pat.values():
        v.sort(key=lambda d: d['t0'])

    sz_by_pat = defaultdict(list)
    for s in seiz:
        folder = idx2folder.get(str(s.get('pat', '')).strip())
        if not folder:
            continue
        onset = _parse_stamp(s.get('onset'))
        if onset is None:
            continue
        try:
            dur = float(s.get('duration'))
        except (TypeError, ValueError):
            continue
        sz_by_pat[folder].append({
            'onset': onset, 'duration': dur,
            'label': (s.get('classif.') or '').strip() or 'seiz',
            'filename': (s.get('filename') or '').strip(),
        })
    for v in sz_by_pat.values():
        v.sort(key=lambda d: d['onset'])
    return dict(by_pat), dict(sz_by_pat)


def _parse_stamp(v):
    """The seizure sheet mixes two date styles and both appear in one column."""
    if v is None:
        return None
    s = str(v).strip().replace('\t', ' ')
    if not s:
        return None
    try:                       # an Excel serial that survived as a number
        return EXCEL_EPOCH + _dt.timedelta(days=float(s))
    except ValueError:
        pass
    for fmt in ('%d/%m/%y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
                '%d/%m/%y %H:%M', '%d/%m/%Y %H:%M'):
        try:
            return _dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# indexing the (possibly truncated) signal archive
# ---------------------------------------------------------------------------
def build_index(archive, cache_path=None, verbose=True):
    """Map `basename -> (payload_offset, payload_bytes)` by walking local
    headers. Works on a truncated zip, which `zipfile` cannot open at all."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return {k: tuple(v) for k, v in json.load(f).items()}

    size = os.path.getsize(archive)
    found = []
    with open(archive, 'rb') as f:
        off, tail = 0, b''
        while off < size:
            f.seek(off)
            buf = f.read(SCAN_CHUNK)
            if not buf:
                break
            data = tail + buf
            base = off - len(tail)
            i = 0
            while True:
                i = data.find(ZIP_LOCAL_SIG, i)
                if i < 0 or i + 30 > len(data):
                    break
                try:
                    v = struct.unpack('<IHHHHHIIIHH', data[i:i + 30])
                except struct.error:
                    break
                nlen, elen = v[9], v[10]
                if 0 < nlen <= 200 and i + 30 + nlen <= len(data):
                    raw = data[i + 30:i + 30 + nlen]
                    if NAME_RE.match(raw):
                        name = raw.decode('utf-8', 'replace')
                        found.append((base + i + 30 + nlen + elen, name,
                                      base + i))
                i += 4
            tail = data[-256:]
            off += len(buf)
            if verbose:
                sys.stderr.write('\r  indexing archive {:5.1f}%'.format(
                    100.0 * off / size))
    if verbose:
        sys.stderr.write('\n')

    # A member's payload ends where the next header begins, minus its data
    # descriptor. Only .data members matter and their true length is known from
    # the metadata, so record the span and let the reader bound it.
    found.sort()
    index = {}
    for k, (payload_off, name, hdr_off) in enumerate(found):
        end = found[k + 1][2] if k + 1 < len(found) else size
        index[os.path.basename(name)] = (payload_off, max(0, end - payload_off))
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(index, f)
    return index


def read_block(fh, offset, n_samples, n_channels, factor, span_bytes=None):
    """One hour of signal as float32 microvolts, shape (n_channels, n)."""
    want = n_samples * n_channels * 2
    if span_bytes is not None and span_bytes < want:
        want = (span_bytes // (n_channels * 2)) * n_channels * 2
    fh.seek(offset)
    raw = fh.read(want)
    usable = (len(raw) // (n_channels * 2)) * n_channels * 2
    a = np.frombuffer(raw[:usable], dtype='<i2')
    a = a.reshape(-1, n_channels).T                    # interleaved -> (ch, n)
    return a.astype(np.float32) * np.float32(factor)


# ---------------------------------------------------------------------------
# planning, kept pure so it can be tested without a 20 GB archive
# ---------------------------------------------------------------------------
def plan_timeline(blocks, present, split_hours=DEFAULT_SPLIT_H):
    """Lay blocks on an absolute timeline and cut it into output files.

    `present` is the set of block filenames that actually exist in the archive;
    absent ones become gaps rather than silently shortening the recording.

    Returns (parts, gaps). Each part is a dict with `index`, `t0`, `duration_s`
    and `pieces`, where a piece is (filename, offset_samples_into_part,
    skip_samples_into_block, n_samples). Overlaps are resolved by truncating
    the earlier block, so the timeline is strictly monotonic.
    """
    usable = [b for b in blocks if b['filename'] in present and b['n_samples']]
    if not usable:
        return [], []

    fs = usable[0]['fs']
    origin = usable[0]['t0']
    placed = []
    gaps = []
    cursor_end = None                # absolute end of the last placed sample
    for b in usable:
        start = (b['t0'] - origin).total_seconds()
        n = b['n_samples']
        skip = 0
        if cursor_end is not None:
            gap = start - cursor_end
            if gap < -1e-9:
                # Overlap: drop the front of this block so time never goes back.
                over = min(-gap, n / float(fs))
                skip = int(round(over * fs))
                start = cursor_end
                n -= skip
                gaps.append({'after': placed[-1][0], 'seconds': round(gap, 3),
                             'kind': 'overlap'})
            elif gap > 1e-9:
                gaps.append({'after': placed[-1][0], 'seconds': round(gap, 3),
                             'kind': 'gap'})
        if n <= 0:
            continue
        placed.append((b['filename'], start, skip, n, b))
        cursor_end = start + n / float(fs)

    total_s = cursor_end
    split_s = float(split_hours) * 3600.0 if split_hours else total_s
    n_parts = max(1, int(np.ceil(total_s / split_s - 1e-9)))

    parts = []
    for p in range(n_parts):
        p0 = p * split_s
        p1 = min(total_s, p0 + split_s)
        pieces = []
        for name, start, skip, n, b in placed:
            b0 = start
            b1 = start + n / float(fs)
            if b1 <= p0 or b0 >= p1:
                continue
            lo = max(b0, p0)
            hi = min(b1, p1)
            off_in_part = int(round((lo - p0) * fs))
            skip_in_block = skip + int(round((lo - b0) * fs))
            take = int(round((hi - lo) * fs))
            if take > 0:
                pieces.append((name, off_in_part, skip_in_block, take, b))
        parts.append({'index': p, 't0': origin + _dt.timedelta(seconds=p0),
                      'start_s': p0, 'duration_s': p1 - p0, 'fs': fs,
                      'pieces': pieces})
    return parts, gaps


def seizures_in_part(seizures, part):
    """Seizure intervals clipped to a part, in seconds from the part's start."""
    out = []
    t0 = part['t0']
    dur = part['duration_s']
    for s in seizures:
        rel = (s['onset'] - t0).total_seconds()
        stop = rel + float(s['duration'])
        if stop <= 0 or rel >= dur:
            continue
        out.append({'start': max(0.0, rel), 'stop': min(dur, stop),
                    'label': s['label'], 'clipped': rel < 0 or stop > dur})
    return out


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _signal_headers(elec_names, fs, factor, keep):
    import pyedflib                                          # noqa: F401
    # One identical digital/physical mapping for every channel, derived from the
    # source's own conversion factor. That makes the round trip exact and
    # sidesteps a flat channel (the photic marker is all zeros) producing an
    # invalid physical_min == physical_max.
    pmin, pmax = -32768 * factor, 32767 * factor
    heads = []
    for i in keep:
        name = elec_names[i]
        kind = 'EEG'
        if name.startswith('EOG'):
            kind = 'EOG'
        elif name == 'ECG':
            kind = 'ECG'
        elif name == 'EMG':
            kind = 'EMG'
        heads.append({
            'label': '{} {}-REF'.format(kind, name),
            'dimension': 'uV', 'sample_frequency': fs,
            'physical_min': pmin, 'physical_max': pmax,
            'digital_min': -32768, 'digital_max': 32767,
            'transducer': '', 'prefilter': '',
        })
    return heads


def write_part(archive_fh, index, part, out_path, elec_names, keep, factor,
               progress=None, edf_plus=False):
    """Stream one output EDF, a second at a time, zero-filling gaps.

    Written as plain EDF, not EDF+, and that choice is worth a minute of
    anyone's attention. EDF+ scatters its annotation records through the file,
    so a reader has to walk every data record at open time to collect them. A
    24-hour recording is 86,400 records, and pyedflib took **75 to 125 seconds
    just to open one** — repeatably, not a cold-cache effect. The same file as
    plain EDF opens instantly, because there is nothing to scan.

    Nothing is lost by it. Seizure times live in the `.csv_bi` beside the file,
    which is where the application looks for reference annotations anyway, and
    the start date and time that the display needs are in the plain EDF header
    exactly as they are in EDF+.
    """
    import pyedflib

    fs = part['fs']
    n_sec = int(round(part['duration_s']))
    n_ch = len(keep)
    heads = _signal_headers(elec_names, fs, factor, keep)

    tmp = out_path + '.incoming'
    w = pyedflib.EdfWriter(tmp, n_ch, file_type=(
        pyedflib.FILETYPE_EDFPLUS if edf_plus else pyedflib.FILETYPE_EDF))
    try:
        w.setSignalHeaders(heads)
        w.setStartdatetime(part['t0'])
        w.setDatarecordDuration(1)

        # Pieces sorted by where they land; one block is held at a time.
        pieces = sorted(part['pieces'], key=lambda p: p[1])
        cur_name, cur_data = None, None
        pi = 0
        for sec in range(n_sec):
            s0 = sec * fs
            rec = np.zeros((n_ch, fs), dtype=np.float64)
            while pi < len(pieces) and pieces[pi][1] + pieces[pi][3] <= s0:
                pi += 1
            for name, off, skip, take, b in pieces[pi:]:
                if off >= s0 + fs:
                    break
                lo = max(s0, off)
                hi = min(s0 + fs, off + take)
                if hi <= lo:
                    continue
                if name != cur_name:
                    o, span = index[name]
                    cur_data = read_block(archive_fh, o, b['n_samples'],
                                          b['n_channels'], factor, span)
                    cur_name = name
                src0 = skip + (lo - off)
                src1 = src0 + (hi - lo)
                if src1 > cur_data.shape[1]:
                    src1 = cur_data.shape[1]
                n = src1 - src0
                if n <= 0:
                    continue
                rec[:, lo - s0:lo - s0 + n] = cur_data[keep, src0:src1]
            w.blockWritePhysicalSamples(rec.ravel())
            if progress and sec % 3600 == 0:
                progress(sec, n_sec)
    finally:
        w.close()
    if os.path.exists(out_path):
        os.remove(out_path)
    os.rename(tmp, out_path)
    return n_sec


def write_csv_bi(path, bname, duration_s, events):
    """TUSZ-style reference annotation, the format the app already reads."""
    with open(path, 'w') as f:
        f.write('# version = csv_v1.0.0\n')
        f.write('# bname = {}\n'.format(bname))
        f.write('# duration = {:.2f} secs\n'.format(duration_s))
        f.write('# montage_file = (null)\n')
        f.write('# source = EPILEPSIAE recording_info/Surface\n')
        f.write('#\n')
        f.write('channel,start_time,stop_time,label,confidence\n')
        for e in events:
            f.write('TERM,{:.4f},{:.4f},seiz,1.0000\n'.format(
                e['start'], e['stop']))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--archive', required=True,
                    help='Epilepsiase.zip (may be truncated)')
    ap.add_argument('--meta', required=True,
                    help='Surface.zip from recording_info')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--patients', nargs='*', default=None,
                    help='e.g. pat_102 pat_21902; default every complete one')
    ap.add_argument('--split-hours', type=float, default=DEFAULT_SPLIT_H,
                    help='hours per output EDF (0 = one file per patient)')
    ap.add_argument('--channels', choices=('all', 'eeg19'), default='all')
    ap.add_argument('--edf-plus', action='store_true',
                    help='write EDF+ instead of plain EDF. Costs 75-125 s per '
                         'day-long file at OPEN time, every time, because a '
                         'reader must walk all 86,400 data records looking '
                         'for annotations. Only useful if something '
                         'downstream needs embedded annotations.')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan, write nothing')
    args = ap.parse_args(argv)

    blocks_by_pat, sz_by_pat = read_metadata(args.meta)
    cache = os.path.join(os.path.dirname(os.path.abspath(args.archive)),
                         '.epilepsiae_index.json')
    index = build_index(args.archive, cache_path=cache)
    present = set(index)

    wanted = args.patients or sorted(blocks_by_pat)
    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    total_files = 0
    for pat in wanted:
        blocks = blocks_by_pat.get(pat) or []
        have = [b for b in blocks if b['filename'] in present]
        if not have:
            print('  {}: no blocks present in the archive, skipping'.format(
                pat))
            continue
        parts, gaps = plan_timeline(blocks, present, args.split_hours)
        seiz = sz_by_pat.get(pat, [])
        elec = _elec_names(args.archive, index, have[0]['filename'])
        keep = ([elec.index(n) for n in EEG_19]
                if args.channels == 'eeg19' else list(range(len(elec))))

        print('\n  {}  {} blocks present of {}, {} parts, {} gaps'.format(
            pat, len(have), len(blocks), len(parts), len(gaps)))
        big = [g for g in gaps if abs(g['seconds']) > 5]
        if big:
            print('    gaps over 5 s: {} (largest {:.0f} s)'.format(
                len(big), max(abs(g['seconds']) for g in big)))

        fh = None if args.dry_run else open(args.archive, 'rb')
        try:
            for part in parts:
                stem = '{}_part{:02d}'.format(pat, part['index'] + 1)
                edf = os.path.join(args.out, stem + '.edf')
                evs = seizures_in_part(seiz, part)
                print('    {:<22} {:>6.2f} h  from {}  {} seizure(s)'.format(
                    stem + '.edf', part['duration_s'] / 3600.0,
                    part['t0'].strftime('%Y-%m-%d %H:%M'), len(evs)))
                if args.dry_run:
                    continue
                write_part(fh, index, part, edf, elec, keep,
                           have[0]['factor'],
                           progress=lambda s, n: sys.stderr.write(
                               '\r      {:.0f}%   '.format(100.0 * s / n)))
                sys.stderr.write('\r')
                write_csv_bi(os.path.splitext(edf)[0] + '.csv_bi', stem,
                             part['duration_s'], evs)
                with open(os.path.splitext(edf)[0] + '.gaps.json', 'w') as g:
                    json.dump({'patient': pat, 'part': part['index'] + 1,
                               'start': part['t0'].isoformat(),
                               'duration_s': part['duration_s'],
                               'gaps': gaps}, g, indent=2)
                total_files += 1
        finally:
            if fh:
                fh.close()

    print('\n  {} EDF file(s) written to {}'.format(
        total_files, os.path.abspath(args.out)) if not args.dry_run
        else '\n  dry run, nothing written')
    return 0


def _elec_names(archive, index, sample_block):
    """Electrode order from the matching .head, which travels with the data."""
    head_name = sample_block.replace('.data', '.head')
    if head_name not in index:
        raise RuntimeError('no .head for {} in the archive'.format(
            sample_block))
    off, span = index[head_name]
    with open(archive, 'rb') as f:
        f.seek(off)
        txt = f.read(min(span, 65536)).split(ZIP_DD_SIG)[0].decode(
            'utf-8', 'replace')
    m = re.search(r'elec_names=\[(.*?)\]', txt, re.S)
    if not m:
        raise RuntimeError('elec_names missing from {}'.format(head_name))
    return [x.strip() for x in m.group(1).split(',') if x.strip()]


if __name__ == '__main__':
    sys.exit(main())
