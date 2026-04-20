"""Signal processing used by the GUI — montages, filters, decimation.

Kept framework-free (numpy + scipy only) so the logic is testable and
reusable by the eventual C/C++ port."""
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

# Canonical 19-channel order used throughout the app (matches
# utils/params_common_electrodes.txt).
CH_NAMES_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
               'T3', 'C3', 'Cz', 'C4', 'T4',
               'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
CH_IDX = {n: i for i, n in enumerate(CH_NAMES_19)}


# ---------------------------------------------------------------------- montages

LONGITUDINAL_BIPOLAR = [
    ('Fp1-F7', 'Fp1', 'F7'), ('F7-T3', 'F7', 'T3'),
    ('T3-T5', 'T3', 'T5'),   ('T5-O1', 'T5', 'O1'),
    ('Fp2-F8', 'Fp2', 'F8'), ('F8-T4', 'F8', 'T4'),
    ('T4-T6', 'T4', 'T6'),   ('T6-O2', 'T6', 'O2'),
    ('Fp1-F3', 'Fp1', 'F3'), ('F3-C3', 'F3', 'C3'),
    ('C3-P3', 'C3', 'P3'),   ('P3-O1', 'P3', 'O1'),
    ('Fp2-F4', 'Fp2', 'F4'), ('F4-C4', 'F4', 'C4'),
    ('C4-P4', 'C4', 'P4'),   ('P4-O2', 'P4', 'O2'),
    ('Fz-Cz', 'Fz', 'Cz'),   ('Cz-Pz', 'Cz', 'Pz'),
]

TRANSVERSE_BIPOLAR = [
    ('F7-Fp1', 'F7', 'Fp1'), ('Fp1-Fp2', 'Fp1', 'Fp2'), ('Fp2-F8', 'Fp2', 'F8'),
    ('F7-F3', 'F7', 'F3'),   ('F3-Fz', 'F3', 'Fz'),
    ('Fz-F4', 'Fz', 'F4'),   ('F4-F8', 'F4', 'F8'),
    ('T3-C3', 'T3', 'C3'),   ('C3-Cz', 'C3', 'Cz'),
    ('Cz-C4', 'Cz', 'C4'),   ('C4-T4', 'C4', 'T4'),
    ('T5-P3', 'T5', 'P3'),   ('P3-Pz', 'P3', 'Pz'),
    ('Pz-P4', 'Pz', 'P4'),   ('P4-T6', 'P4', 'T6'),
    ('T5-O1', 'T5', 'O1'),   ('O1-O2', 'O1', 'O2'), ('O2-T6', 'O2', 'T6'),
]

MONTAGES = {
    'Common Average': None,         # sentinel
    'Longitudinal Bipolar': LONGITUDINAL_BIPOLAR,
    'Transverse Bipolar': TRANSVERSE_BIPOLAR,
    'Referential (raw)': 'raw',     # sentinel
}


def apply_montage(data19, montage_name):
    """data19: [19, N] in canonical order. Return (display_data [K, N], labels)."""
    if montage_name == 'Referential (raw)':
        return data19.copy(), list(CH_NAMES_19)
    if montage_name == 'Common Average':
        avg = data19.mean(axis=0, keepdims=True)
        return (data19 - avg), list(CH_NAMES_19)
    spec = MONTAGES[montage_name]
    rows, labels = [], []
    for label, a, b in spec:
        rows.append(data19[CH_IDX[a]] - data19[CH_IDX[b]])
        labels.append(label)
    return np.stack(rows, axis=0), labels


# ---------------------------------------------------------------------- filters


def _butter_sos(cutoff, fs, btype, order=4):
    nyq = 0.5 * fs
    if isinstance(cutoff, (tuple, list)):
        wn = [c / nyq for c in cutoff]
    else:
        wn = cutoff / nyq
    return butter(order, wn, btype=btype, output='sos')


def _notch_sos(f0, fs, q=30.0):
    b, a = iirnotch(f0 / (0.5 * fs), q)
    return tf2sos(b, a)


def apply_filters(data, fs, hp=None, lp=None, notch=None):
    """Return filtered copy. hp/lp in Hz or None; notch in Hz or None.

    Uses zero-phase sosfiltfilt so no phase distortion. Safe to call with
    all parameters = None (returns a copy).
    """
    out = np.ascontiguousarray(data, dtype=np.float64)
    if hp is not None and hp > 0:
        sos = _butter_sos(hp, fs, 'highpass', order=2)
        out = sosfiltfilt(sos, out, axis=1)
    if lp is not None and lp > 0 and lp < fs / 2:
        sos = _butter_sos(lp, fs, 'lowpass', order=4)
        out = sosfiltfilt(sos, out, axis=1)
    if notch is not None and notch > 0:
        sos = _notch_sos(notch, fs)
        out = sosfiltfilt(sos, out, axis=1)
    return out.astype(np.float32)


# ---------------------------------------------------------------------- decimation


def decimate_for_display(data, fs, target_px=8000):
    """Uniform-stride decimation to ~target_px points along time axis.

    Adaptive re-decimation in SignalView keeps the point count roughly
    proportional to viewport pixel width at every zoom level. Stride
    decimation is used instead of min/max envelope because an envelope
    produces solid vertical bars when the signal has sustained peaks at
    or above the clip level (looks like black boxes).

    Returns (data_ds [K, M], time_ds [M])."""
    n = data.shape[1]
    stride = max(1, n // target_px)
    t_ds = np.arange(0, n, stride, dtype=np.float32) / fs
    data_ds = data[:, ::stride]
    return data_ds, t_ds
