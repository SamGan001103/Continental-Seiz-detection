"""Load a recording for SCORING without holding it three times over.

The problem
-----------
`load_edf_19ch` reads a whole recording into memory, and on the day-long
recordings this project is now aimed at that is expensive. Measured on a
22.3-hour, 29-channel, 256 Hz EPILEPSIAE file: **9.0 GB peak working set, 14 GB
of pagefile, 136 seconds** — before a single window is scored. The viewer never
pays it (it reads spans off disk through `gui/io/edf_source.EdfSource`), but
`precompute_probs.py` does, so a recording the application opens in 0.01 s can
be too heavy to score on the modest hardware this is meant to run on.

Three things cost that memory, and only one of them is the signal itself:

1. every channel is read, as float64, including the ten of 29 that are
   discarded a moment later,
2. the 256 Hz to 250 Hz resample is handed the whole 2D array, so the source
   and the resampled target are both alive at once, and
3. scipy's FFT resample allocates complex intermediates the size of the input.

So this reads one channel, resamples it, casts it down and stores it, then
drops it. Peak is the float32 result plus one channel's working set, instead of
every channel at float64 twice over.

Why this is allowed to exist
----------------------------
It is **bit-identical** to `load_edf_19ch`, not merely equivalent, and that was
established by measurement before anything was written:

* resampling row by row equals `resample(X, n, axis=1)` exactly — 0.0
  difference on real EEG at one minute, ten minutes and one hour;
* reading only the mapped channels equals reading all of them and selecting —
  exact equality on three real recordings, including a Nihon Kohden file whose
  vendor labels differ and a 58-minute TUH session;
* and on the day-long 256 Hz file, where the resample branch actually fires,
  the two loaders agree bit for bit.

That standard is not pedantry here. This project threw away a parallel scorer
that was 8.8x faster because pinning BLAS threads moved p(seizure) by 0.138, and
shipped a 3.3x one that is bitwise identical instead. `tests/test_edf_stream.py`
holds this to the same line.

The channel mapping is reused from `gui/io/edf_source.resolve_channel_map`, so
it is the same selection the lazy viewer path makes, by construction.
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.edf import TARGET_FS                              # noqa: E402
from gui.io.edf_source import resolve_channel_map             # noqa: E402
from gui.paths import params_path                             # noqa: E402


def load_edf_19ch_streaming(path, params_path_=None, out_path=None):
    """Return ``(signals[19, N], fs, duration_s)``, as `load_edf_19ch` does.

    Same values, same dtype, same row order — one channel at a time.

    out_path
        Write the result to a memory-mapped ``.npy`` at this path instead of
        holding it in RAM, and return the memmap. The array still behaves like
        an ndarray, so callers need no changes, but the operating system pages
        it in and out instead of the process carrying 1.5 GB of it.

        This is what makes a day-long recording scoreable on a small machine.
        Even after the per-channel loading above, the parent process peaked at
        3.1 GB — mostly the finished signal — and on a 16 GB box shared with a
        browser that is the difference between working and thrashing. With the
        signal mapped, the parent holds one channel and whatever pages the
        scoring loop is touching.

        The caller owns the file and should delete it when done.
    """
    mapping, fs, n_samples, reader = resolve_channel_map(
        path, params_path_ or params_path())
    fs = float(fs)
    if fs <= 0:
        try:
            reader.close()
        except Exception:                                     # noqa: BLE001
            pass
        raise RuntimeError('Invalid EDF sampling rate: {}'.format(fs))

    needs_resample = int(round(fs)) != TARGET_FS
    # The target length is computed exactly as load_edf_19ch computes it, from
    # the FULL width, so a per-channel loop cannot drift from the whole-array
    # call by a sample.
    n_out = (int(n_samples * TARGET_FS / fs) if needs_resample
             else int(n_samples))
    if out_path:
        from numpy.lib.format import open_memmap
        parent = os.path.dirname(os.path.abspath(out_path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        out = open_memmap(out_path, mode='w+', dtype=np.float32,
                          shape=(len(mapping), n_out))
    else:
        out = np.empty((len(mapping), n_out), dtype=np.float32)
    try:
        for row, ch in enumerate(mapping):
            x = reader.readSignal(ch)
            if needs_resample:
                from scipy.signal import resample
                x = resample(x, n_out)
            # Assigning into a float32 array casts elementwise, which is what
            # `.astype(np.float32)` on the whole array does.
            out[row] = x
            del x
    finally:
        try:
            reader.close()
        except Exception:                                     # noqa: BLE001
            pass

    fs_out = TARGET_FS if needs_resample else int(round(fs))
    return out, fs_out, out.shape[1] / float(fs_out)


def estimate_peak_bytes(n_channels_in_file, n_samples_source, fs_source):
    """Roughly what `load_edf_19ch_streaming` will peak at.

    Unlike `gui.memory.estimate_bytes`, which describes the eager loader, this
    one is dominated by the RESULT rather than by the intermediates: 19 rows of
    float32, plus one channel being worked on, plus scipy's complex
    intermediates for that one channel.
    """
    n_src = int(n_samples_source)
    n_out = int(n_src * TARGET_FS / float(fs_source))
    result = 19 * n_out * 4
    one_channel = n_src * 8 + n_out * 8
    fft_intermediates = 0
    if int(round(float(fs_source))) != TARGET_FS:
        fft_intermediates = 3 * max(n_src, n_out) * 8
    return result + one_channel + fft_intermediates
