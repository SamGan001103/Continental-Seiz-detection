"""Shared inference helper used by the precompute CLI. Factored out from
run_inference.py so the GUI never has to import TF at runtime."""
import os
import sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.edf import load_edf_19ch, CHANNELS_19, TARGET_FS  # noqa: E402

SEGMENT_S = 12
WEIGHTS = os.path.join(REPO, 'convlstm_ICA_12_train.h5')


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
    os.chdir(os.path.join(REPO, 'utils'))  # keras import paths expect this cwd
    try:
        from models.deep_conv_lstm import ConvLstmNet
        m = ConvLstmNet(epochs=1).setup(
            (-1, 2 * SEGMENT_S - 1, len(CHANNELS_19), 125, 1))
        m.model.load_weights(WEIGHTS)
    finally:
        os.chdir(REPO)
    return m.model


def compute_probs(edf_path, step_s=6, use_ica=True, progress_cb=None):
    """Run inference on a single EDF. Return (window_starts, probs).

    step_s       : stride in seconds between window starts
    use_ica      : apply ICA EOG removal per window (slow)
    progress_cb  : optional callable(i, n_total) for UI progress
    """
    from utils.preprocessing import ica_arti_remove
    model = _build_model()
    data, fs, duration_s = load_edf_19ch(edf_path)
    window_len = SEGMENT_S * fs
    total_s = int(duration_s)
    starts = list(range(0, total_s - SEGMENT_S + 1, step_s))
    probs = np.zeros(len(starts), dtype=np.float32)
    for i, t in enumerate(starts):
        seg = data[:, t * fs:t * fs + window_len]
        if seg.shape[1] != window_len:
            break
        if use_ica:
            proc = ica_arti_remove(seg, fs, CHANNELS_19)
            if proc is None:
                proc = seg
        else:
            proc = seg
        x = _calc_stft(proc)
        x = np.expand_dims(x, -1)
        probs[i] = float(model.predict(x, verbose=0)[0, 1])
        if progress_cb is not None:
            progress_cb(i + 1, len(starts))
    return np.array(starts, dtype=np.int32), probs
