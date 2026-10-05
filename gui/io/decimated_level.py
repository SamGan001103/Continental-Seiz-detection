"""A coarse copy of the recording, held in RAM, for the zoomed-out regime.

The problem
-----------
Zooming out on a day-long recording was measured at 0.3-3.0 s per render, and
96-103% of that is one line: the strided read in
`DisplayPipeline._render_overview`. To draw 1600 points of a six-hour view the
reader walks the whole six hours off a 1.2 GB file, then throws away 99.97% of
what it read. A wheel gesture does that on every notch, synchronously, on the
GUI thread.

Nothing about that is fixed by a cache of render OUTPUT: the overview cache was
measured getting zero hits across a realistic zoom, because every notch asks
for a different (s0, s1, target_px) triple and a gesture never returns to a
bit-identical view. The fix has to make each render cheap, not rarer.

The shape
---------
One decimated level, built once: every Nth sample of every channel. At stride
64 a 24-hour 19-channel recording is 26 MB, and a wide render becomes a slice
of an in-memory array.

Stride 64 is chosen, not rounded to a power of two for neatness. The
zoomed-out regime begins at a ~180 s view, which needs 1600 points over 300 s,
or 5.3 Hz. Stride 64 of 256 Hz is 4 Hz, close enough that the narrowest wide
view loses little; anything coarser visibly under-samples it.

Every sample this serves is a REAL sample from the recording, at its true time.
The level never interpolates, and the effective stride is always rounded to a
multiple of its own, so the points drawn are a subset of the points the
full-rate path would have drawn. What changes is which subset, and only in the
regime where one pixel already spans many seconds.

Where it lives
--------------
On the SOURCE, not on DisplayPipeline. That object is rebuilt whenever a
filter, the montage or the sensitivity changes, which is constantly, and a
level rebuilt that often would cost more than it saves.

Why it is a file
----------------
Building it takes about 60 s on a 24-hour recording -- one pass over the whole
file -- which is far too long to make a reviewer wait on open. So it is written
beside the recording like `.probs.npz`, stamped with the recording's size and
mtime so a changed or replaced file cannot be served a stale level.
"""
import os

import numpy as np

DEFAULT_STRIDE = 64
SUFFIX = '.level{}.npz'
# Matches gui.io.cache: a copy through a filesystem that rounds timestamps
# should not invalidate an otherwise good level.
MTIME_TOLERANCE_S = 2.0


def path_for(recording_path, stride=DEFAULT_STRIDE):
    return os.path.splitext(recording_path)[0] + SUFFIX.format(int(stride))


class DecimatedLevel(object):
    """Every `stride`-th sample of every channel, in memory."""

    def __init__(self, data, stride, fs):
        self.data = np.asarray(data, dtype=np.float32)
        self.stride = int(stride)
        self.fs = float(fs)

    @property
    def nbytes(self):
        return int(self.data.nbytes)

    def covers(self, stride):
        """True when a request at `stride` can be served from this level.

        Below its own stride the level has fewer samples than the caller
        wants, and serving them would silently thin the trace.
        """
        return int(stride) >= self.stride

    def slice(self, s0, s1, stride):
        """Samples for [s0, s1) at approximately `stride`.

        Returns (rows, first_sample_index, effective_stride) in ORIGINAL
        sample coordinates, so the caller builds its time axis exactly as it
        would from a full-rate read. `effective_stride` is rounded up to a
        multiple of the level's own, which is what keeps every point a real
        sample at its real time rather than a nearby one.
        """
        # Rounded DOWN, never to nearest. Rounding to nearest sometimes lands
        # on a coarser stride than the caller asked for, and the trace is then
        # drawn from fewer points than the full-rate path would have used — at
        # a 600 s view, 75% of them. That is a visibly thinner trace, which is
        # exactly the kind of silent quality loss this whole module must not
        # introduce. Rounding down can only ever serve MORE points than were
        # asked for; the extra costs about a millisecond of paint against the
        # 1.7 s the level saves.
        step = max(1, int(stride) // self.stride)
        eff = step * self.stride
        # First level index at or after s0, so the window never starts before
        # the span the caller asked for.
        i0 = (int(s0) + self.stride - 1) // self.stride
        i1 = min(self.data.shape[1], (int(s1) + self.stride - 1) // self.stride)
        if i1 <= i0:
            return (np.zeros((self.data.shape[0], 0), dtype=np.float32),
                    int(s0), eff)
        rows = self.data[:, i0:i1:step]
        return rows, i0 * self.stride, eff

    # -- persistence ---------------------------------------------------
    def save(self, path, recording_path=None):
        meta = {'stride': self.stride, 'fs': self.fs}
        if recording_path:
            try:
                st = os.stat(recording_path)
                meta['edf_size'] = int(st.st_size)
                meta['edf_mtime'] = float(st.st_mtime)
            except OSError:
                pass
        tmp = path + '.part'
        # Written to a temporary name and moved into place, so a build
        # interrupted by a crash or a closed laptop cannot leave a truncated
        # level that later loads as if it were whole.
        np.savez(tmp, data=self.data, **{k: np.array(v)
                                         for k, v in meta.items()})
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp + '.npz' if os.path.exists(tmp + '.npz') else tmp, path)
        return path


def load(path, recording_path=None):
    """Read a level from disk, or None if absent, damaged or stale."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            data = z['data']
            stride = int(z['stride'])
            fs = float(z['fs'])
            size = int(z['edf_size']) if 'edf_size' in z.files else None
            mtime = float(z['edf_mtime']) if 'edf_mtime' in z.files else None
    except Exception:                                         # noqa: BLE001
        return None            # truncated or not a level; rebuild rather than
                               # fail the open
    if recording_path and size is not None:
        try:
            st = os.stat(recording_path)
        except OSError:
            return None
        if int(st.st_size) != size:
            return None
        if mtime is not None and abs(st.st_mtime - mtime) > MTIME_TOLERANCE_S:
            return None
    return DecimatedLevel(data, stride, fs)


def build(source, fs, stride=DEFAULT_STRIDE, chunk_s=3600.0, progress=None):
    """One chunked pass over `source`, producing a DecimatedLevel.

    Chunked so the transient footprint is an hour of signal rather than a day;
    the level itself is the only thing that stays.
    """
    stride = int(stride)
    n_ch, n = int(source.shape[0]), int(source.shape[1])
    out_len = (n + stride - 1) // stride
    level = np.empty((n_ch, out_len), dtype=np.float32)
    chunk = max(stride, int(chunk_s * fs))
    written = 0
    pos = 0
    while pos < n and written < out_len:
        end = min(n, pos + chunk)
        # Start this chunk on the global stride grid, so the samples land
        # exactly where one whole-file stride would have put them and a chunk
        # boundary cannot shift the phase.
        first = pos + (-pos % stride)
        if first < end:
            block = np.asarray(source[:, first:end:stride], dtype=np.float32)
            k = min(block.shape[1], out_len - written)
            if k > 0:
                level[:, written:written + k] = block[:, :k]
                written += k
        pos = end
        if progress is not None:
            progress(min(1.0, pos / float(n)))
    if written < out_len:
        level = level[:, :written]
    return DecimatedLevel(level, stride, fs)


def attach(source, recording_path, fs, stride=DEFAULT_STRIDE, build_if_missing=False,
           progress=None):
    """Give `source` a `.level`, from disk if one is there.

    Returns the level, or None. By default a missing level is NOT built here:
    the build is a minute of work and belongs in a precompute pass, not in the
    path that opens a recording for a waiting reviewer.
    """
    path = path_for(recording_path, stride)
    level = load(path, recording_path)
    if level is None and build_if_missing:
        level = build(source, fs, stride=stride, progress=progress)
        try:
            level.save(path, recording_path)
        except OSError:
            pass               # a read-only folder is not a reason to fail
    if level is not None:
        source.level = level
    return level
