"""Read spans of a 19-channel recording from disk, instead of holding it in RAM.

`gui.io.edf.load_edf_19ch` reads a whole recording into memory: 1.6 GB of
float32 for 24 hours, and several gigabytes transiently while pyst holds every
file channel as float64. `gui/memory.py` refuses such a file rather than freeze
the machine — which is correct, and also means a ward PC could not review a
day of ambulatory EEG at all. That is the recording this project exists for.

`EdfSource` closes that gap for the DISPLAY. It presents the same interface the
display pipeline uses — `shape`, `dtype`, and slicing along time — while
reading only the samples asked for. A day-long file then costs a file handle
and whatever span is on screen, so opening one is bounded by the viewport
exactly as redrawing already is.

Channel selection is the risk, and it is handled by not reinventing it. The
detector's own loader picks its 19 channels through
`pyst.nedc_select_channels`, matching the labels in
`utils/params_common_electrodes.txt` under the parameter file's match mode.
This class calls that same code path on a ONE-SECOND read to learn which file
channel each of the 19 maps to, then reads those channels directly for every
later span. A display that showed different channels from the ones scored
would be the worst failure available to this application, so
`tests/test_edf_source.py` asserts the assembled array equals
`load_edf_19ch`'s output exactly, on real recordings.

Deliberately limited to files already at the model's 250 Hz. Resampling is a
whole-signal operation — an FFT resample of a span is not the corresponding
slice of the resampled whole — so a file at another rate cannot be served
span-by-span without changing the numbers, and `open_source` falls back to a
full load for those. TUH and the ambulatory recordings this is aimed at are
250 Hz.

Scoring is unaffected and still needs the full array: `compute_probs_from_data`
begins with `np.asarray(data, dtype=np.float32)`, which would materialise
everything. That is the right division — first-pass scoring of a day-long
recording is a batch job for `precompute_probs.py --workers N`, run before the
review, and the GUI then loads the cache in milliseconds.
"""
import os

import numpy as np

TARGET_FS = 250

# Stride past which seeking to each wanted sample beats reading through the
# gap. Break-even is where the skipped bytes take longer to read than a seek
# takes to perform — a few thousand samples on any disk this decade. Measured
# either side on a 24-hour recording: batching a whole-day overview read 1.6 GB
# and took 12 s; seeking reads 30 k samples and takes a fraction of a second.
SEEK_BEATS_READ_STRIDE = 1250


class EdfSource(object):
    """A read-only [19, N] view of an EDF on disk, read span by span.

    Supports `src[:, a:b]`, `src[i, a:b]` and a step, which is all the display
    pipeline uses. `np.asarray(src)` assembles the whole recording and is
    deliberately expensive: it is what this class exists to avoid.
    """

    # Above this stride a contiguous read is wasteful: the caller wants one
    # sample in every STRIDE, so reading the span whole means pulling gigabytes
    # off disk to discard almost all of it. A whole-day overview did exactly
    # that and took 35 s. Below it, one read and a stride is cheaper than
    # thousands of seeks.
    SPARSE_STRIDE = 64

    def __init__(self, path, channel_map, n_samples, fs=TARGET_FS,
                 reader=None):
        import pyedflib
        self._path = os.path.abspath(path)
        self._map = list(channel_map)          # our row i -> file channel index
        self.shape = (len(self._map), int(n_samples))
        self.dtype = np.dtype(np.float32)
        self.ndim = 2
        self.fs = int(fs)
        # The reader is handed in wherever one is already open: pyedflib's
        # constructor indexes the whole file, which is 28 s for a day-long
        # recording, and resolving the channel map needs a reader anyway.
        # Opening a second one doubled the cost of opening a file.
        self._reader = reader or pyedflib.EdfReader(self._path)

    # -- lifecycle -------------------------------------------------------
    def close(self):
        r = getattr(self, '_reader', None)
        if r is not None:
            try:
                r.close()
            except Exception:                                # noqa: BLE001
                pass
            self._reader = None

    def __del__(self):
        self.close()

    def __len__(self):
        return self.shape[0]

    # -- reading ---------------------------------------------------------
    def _rows_for(self, rows):
        if isinstance(rows, slice):
            return list(range(*rows.indices(self.shape[0]))), True
        if isinstance(rows, (int, np.integer)):
            n = int(rows)
            if n < 0:
                n += self.shape[0]
            return [n], False
        return [int(r) for r in rows], True

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        rows, cols = key
        if not isinstance(cols, slice):
            raise TypeError('EdfSource supports slice access along time')
        start, stop, step = cols.indices(self.shape[1])
        step = max(1, step)
        n = max(0, stop - start)
        idx, keep_2d = self._rows_for(rows)
        if n == 0 or not idx:
            out = np.zeros((len(idx), 0), dtype=np.float32)
            return out if keep_2d else out[0]
        if self._reader is None:
            raise RuntimeError('EdfSource is closed')
        if step >= self.SPARSE_STRIDE:
            out = self._read_sparse(idx, start, stop, step)
        else:
            # Read contiguously and stride afterwards: pyedflib has no strided
            # read, and at a small stride the contiguous block is what the file
            # layout wants anyway.
            out = np.empty((len(idx), n), dtype=np.float32)
            for k, r in enumerate(idx):
                out[k] = self._reader.readSignal(
                    self._map[r], start=start,
                    n=n).astype(np.float32, copy=False)
            if step > 1:
                out = out[:, ::step]
        return out if keep_2d else out[0]

    def _read_sparse(self, idx, start, stop, step):
        """Exactly `data[start:stop:step]`, without reading what falls between.

        Same values a contiguous read and stride would give — the positions are
        identical — but it touches n_out samples instead of (stop - start).
        Reads are batched into runs so a moderate stride still gets sequential
        I/O rather than one seek per point.
        """
        positions = np.arange(start, stop, step, dtype=np.int64)
        n_out = positions.size
        out = np.empty((len(idx), n_out), dtype=np.float32)
        if n_out == 0:
            return out
        # Cap the contiguous bytes pulled per read; a run covers as many
        # output points as fit, so a stride of 100 reads one block for ~80
        # points rather than 80 separate blocks.
        RUN_SAMPLES = 1 << 16
        per_run = max(1, int(RUN_SAMPLES // step))
        if step > SEEK_BEATS_READ_STRIDE:
            # Past this the gap between wanted samples costs more to read than
            # a seek costs to skip, and batching quietly reads the whole file:
            # at a whole-day zoom the stride is ~13,500 samples, so a "batch"
            # of four points pulled 54,000 samples to use four. That was 12 s
            # for one redraw.
            per_run = 1
        for k, r in enumerate(idx):
            ch = self._map[r]
            for b0 in range(0, n_out, per_run):
                b1 = min(n_out, b0 + per_run)
                p0 = int(positions[b0])
                p1 = int(positions[b1 - 1]) + 1
                block = self._reader.readSignal(ch, start=p0, n=p1 - p0)
                out[k, b0:b1] = block[positions[b0:b1] - p0]
        return out

    def __array__(self, dtype=None):
        out = self[:, :]
        return out if dtype is None else out.astype(dtype)


def resolve_channel_map(path, params_path_):
    """Which file channel each of the 19 model channels comes from.

    Uses the detector's own selection code (`pyst.nedc_get_pos` under the
    parameter file's match mode), so the mapping is by construction the same
    one `load_edf_19ch` would produce — including vendor label quirks like
    'EEG FP1-REF'.

    Returns (channel_map, fs, n_samples, open_reader). The reader is returned
    still open, because pyedflib's constructor indexes the entire file — 28 s
    for a day-long recording — and opening a second one to do the reading
    doubled the cost of opening a file. The caller owns it.
    """
    import pyedflib
    from utils import pyst

    params = pyst.nedc_load_parameters(params_path_)
    if params is None:
        raise RuntimeError('could not read the montage parameter file')
    chan_list = params.get(pyst.KEYWORD_CSEL).split(pyst.DELIM_COMMA)
    mmode = params[pyst.KEYWORD_MMODE]

    r = pyedflib.EdfReader(os.path.abspath(path))
    keep = False
    try:
        labels = r.getSignalLabels()
        rates = [float(r.getSampleFrequency(i))
                 for i in range(r.signals_in_file)]
        n_per_ch = [int(r.getNSamples()[i]) for i in range(r.signals_in_file)]
        mapping = []
        for lbl in chan_list:
            pos = pyst.nedc_get_pos(lbl, labels, mmode)
            if pos < 0:
                raise RuntimeError(
                    'Could not find the required 19 EEG channels in this EDF. '
                    'Expected TUH-style labels; missing: {}'.format(lbl))
            mapping.append(int(pos))
        fs_sel = {rates[p] for p in mapping}
        if len(fs_sel) != 1:
            raise RuntimeError(
                'the 19 EEG channels are not all at one sampling rate: '
                '{}'.format(sorted(fs_sel)))
        fs = fs_sel.pop()
        n = min(n_per_ch[p] for p in mapping)
        keep = True
    finally:
        if not keep:
            r.close()
    return mapping, fs, n, r


def open_source(path, params_path_):
    """A lazy EdfSource when the file allows it, else None.

    None means "load it the ordinary way": the caller keeps the existing
    behaviour rather than this module inventing a second numeric path.
    """
    try:
        mapping, fs, n, reader = resolve_channel_map(path, params_path_)
    except Exception:                                        # noqa: BLE001
        return None
    if int(round(fs)) != TARGET_FS:
        # Resampling cannot be done span-by-span without changing the numbers:
        # an FFT resample of a slice is not the corresponding slice of the
        # resampled whole. TUH recordings are 256 Hz and take this path, so
        # they load the ordinary way and the memory guard applies.
        try:
            reader.close()
        except Exception:                                    # noqa: BLE001
            pass
        return None
    try:
        return EdfSource(path, mapping, n, fs=TARGET_FS, reader=reader)
    except Exception:                                        # noqa: BLE001
        try:
            reader.close()
        except Exception:                                    # noqa: BLE001
            pass
        return None
