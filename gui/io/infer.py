"""Shared inference helper used by the precompute CLI. Factored out from
run_inference.py so the GUI never has to import TF at runtime."""
import os
import sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.edf import load_edf_19ch, CHANNELS_19, TARGET_FS  # noqa: E402
from gui.paths import weights_path  # noqa: E402

SEGMENT_S = 12
WEIGHTS = weights_path()
_MODERN_TO_LEGACY = {
    'T7': 'T3',
    'T8': 'T4',
    'P7': 'T5',
    'P8': 'T6',
}


def _calc_stft(s_):
    import stft as stftpkg
    s = s_.transpose()
    d = stftpkg.spectrogram(s, framelength=250, centered=False)
    if d.ndim == 2:
        d = np.expand_dims(d, -1)
    d = np.transpose(d, (1, 2, 0))
    d = np.abs(d) + 1e-6
    d = d[:, :, 1:]
    d = np.log10(d)
    d[d <= 0] = 0
    return d.reshape(-1, d.shape[0], d.shape[1], d.shape[2])


def _build_model():
    # No os.chdir here. The old code changed the working directory into utils/
    # with the comment "keras import paths expect this cwd", but models/ sits at
    # the repository root and resolves through sys.path — the chdir was doing
    # nothing except mutating global state that a frozen build cannot satisfy.
    from models.deep_conv_lstm import ConvLstmNet
    m = ConvLstmNet(epochs=1).setup(
        (-1, 2 * SEGMENT_S - 1, len(CHANNELS_19), 125, 1))
    if not os.path.exists(WEIGHTS):
        raise RuntimeError(
            'Model weights not found at {}. The application folder is '
            'incomplete — re-copy it in full.'.format(WEIGHTS))
    m.model.load_weights(WEIGHTS)
    return m.model


def _normalise_channel_name(name):
    name = str(name).strip()
    if name.startswith('EEG '):
        name = name[4:]
    name = name.replace('-REF', '').replace('-LE', '')
    return _MODERN_TO_LEGACY.get(name, name)


def _coerce_channels(channels):
    return [_normalise_channel_name(ch) for ch in channels]


def load_signal_npz(npz_path):
    """Load a repo-format 19-channel signal NPZ.

    Expected arrays:
        data     : shape (19, samples)
        fs       : scalar sampling rate
        channels : optional channel labels
    """
    with np.load(npz_path, allow_pickle=False) as z:
        if 'data' not in z or 'fs' not in z:
            raise RuntimeError(
                'Signal NPZ must contain data and fs arrays: {}'.format(
                    npz_path))
        data = z['data'].astype(np.float32, copy=False)
        fs = int(round(float(np.asarray(z['fs']).reshape(-1)[0])))
        channels = None
        if 'channels' in z:
            channels = _coerce_channels(z['channels'])

    if data.ndim != 2:
        raise RuntimeError(
            'Signal NPZ data must be 2D, got shape {}'.format(data.shape))
    if channels is not None:
        if len(channels) != data.shape[0]:
            raise RuntimeError(
                'Signal NPZ has {} channel labels for {} data rows'.format(
                    len(channels), data.shape[0]))
        if channels != CHANNELS_19:
            if sorted(channels) != sorted(CHANNELS_19):
                raise RuntimeError(
                    'Signal NPZ channels do not match required 19-channel '
                    'montage: {}'.format(', '.join(channels)))
            order = [channels.index(ch) for ch in CHANNELS_19]
            data = data[order]
    elif data.shape[0] != len(CHANNELS_19):
        raise RuntimeError(
            'Signal NPZ without channel labels must have 19 rows, got {}'
            .format(data.shape[0]))
    return data.astype(np.float32, copy=False), fs, data.shape[1] / float(fs)


# Why a window carries no model score. Persisted alongside the probabilities so
# the GUI can distinguish "the model saw this and said no" from "the pipeline
# never scored this", which previously looked identical (both were 0.0).
SKIP_NONE = 0        # scored normally
SKIP_INTERRUPTED = 1  # detect_interupted_data rejected the raw segment
SKIP_ICA_FAILED = 2   # ica_arti_remove returned None
SKIP_SHORT = 3        # not a full window's worth of samples

SKIP_LABELS = {
    SKIP_NONE: 'scored',
    SKIP_INTERRUPTED: 'not assessed — interrupted or flat signal',
    SKIP_ICA_FAILED: 'not assessed — ICA decomposition failed',
    SKIP_SHORT: 'not assessed — incomplete window',
}


def compute_probs_from_data(data, fs, step_s=6, use_ica=True,
                            progress_cb=None, model=None):
    """Run inference on a 19-channel array.

    Returns ``(window_starts, probs, skip_code)``.

    ``skip_code`` is one of the SKIP_* constants per window. Windows the
    pipeline declined to score still carry ``probs == 0.0`` for backward
    compatibility, but the code array is what callers should test: a real
    softmax output is effectively never exactly 0.0, yet inferring the skip set
    from that coincidence is how a refused window came to be displayed as a
    confident negative.

    step_s       : stride in seconds between window starts
    use_ica      : apply ICA EOG removal per window (slow)
    progress_cb  : optional callable(i, n_total) for UI progress
    model        : optional already-loaded Keras model
    """
    from utils.preprocessing import detect_interupted_data, ica_arti_remove
    if model is None:
        model = _build_model()
    data = np.asarray(data, dtype=np.float32)
    fs = int(round(float(fs)))
    if data.ndim != 2 or data.shape[0] != len(CHANNELS_19):
        raise RuntimeError(
            'Model inference requires shape (19, samples), got {}'.format(
                data.shape))
    if fs != TARGET_FS:
        raise RuntimeError(
            'Model inference requires {} Hz data, got {} Hz'.format(
                TARGET_FS, fs))
    duration_s = data.shape[1] / float(fs)
    window_len = SEGMENT_S * fs
    total_s = int(duration_s)
    starts = list(range(0, total_s - SEGMENT_S + 1, step_s))
    probs = np.zeros(len(starts), dtype=np.float32)
    skip_code = np.zeros(len(starts), dtype=np.int8)
    for i, t in enumerate(starts):
        seg = data[:, t * fs:t * fs + window_len]
        if seg.shape[1] != window_len:
            # Trailing partial windows were never scored; mark the tail rather
            # than leaving it looking like a run of confident negatives.
            skip_code[i:] = SKIP_SHORT
            break
        if detect_interupted_data(seg.transpose(), fs):
            probs[i] = 0.0
            skip_code[i] = SKIP_INTERRUPTED
            if progress_cb is not None:
                progress_cb(i + 1, len(starts))
            continue
        if use_ica:
            proc = ica_arti_remove(seg, fs, CHANNELS_19)
            if proc is None:
                probs[i] = 0.0
                skip_code[i] = SKIP_ICA_FAILED
                if progress_cb is not None:
                    progress_cb(i + 1, len(starts))
                continue
        else:
            proc = seg
        x = _calc_stft(proc)
        x = np.expand_dims(x, -1)
        probs[i] = float(model.predict(x, verbose=0)[0, 1])
        if progress_cb is not None:
            progress_cb(i + 1, len(starts))
    return np.array(starts, dtype=np.int32), probs, skip_code


def compute_probs(edf_path, step_s=6, use_ica=True, progress_cb=None,
                  model=None, data=None, fs=None):
    """Run inference on a single EDF.

    Returns ``(window_starts, probs, skip_code)``.

    ``data``/``fs`` let a caller that has already read the EDF pass the array in
    rather than paying for a second full read of the same file.
    """
    if data is None or fs is None:
        data, fs, _duration_s = load_edf_19ch(edf_path)
    return compute_probs_from_data(
        data, fs, step_s=step_s, use_ica=use_ica,
        progress_cb=progress_cb, model=model)
