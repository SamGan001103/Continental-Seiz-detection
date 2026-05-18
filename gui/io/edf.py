"""EDF loading for the GUI. Reuses the repo's existing read_edf_elec + the
19-channel montage declared in utils/params_common_electrodes.txt."""
import os
import sys
import numpy as np
from scipy.signal import resample

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.pyst import read_edf_elec  # noqa: E402

CHANNELS_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
               'T3', 'C3', 'Cz', 'C4', 'T4',
               'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
TARGET_FS = 250


def load_edf_19ch(path):
    """Return (signals[19, N_samples], fs=TARGET_FS, duration_s).

    Resamples to 250 Hz if the source sampling rate is higher; otherwise
    passes through. Uses the repo's partial-match montage logic so
    'EEGFP1-REF' → Fp1 regardless of vendor naming."""
    path = os.path.abspath(path)
    cwd = os.getcwd()
    try:
        os.chdir(os.path.join(REPO, 'utils'))  # params lookup is cwd-relative
        try:
            fs, data = read_edf_elec(path)
        except SystemExit as ex:
            raise RuntimeError(
                'Could not find the required 19 EEG channels in this EDF. '
                'Expected TUH-style labels for: {}'.format(
                    ', '.join(CHANNELS_19))) from ex
    finally:
        os.chdir(cwd)
    fs = float(fs)
    if fs <= 0:
        raise RuntimeError('Invalid EDF sampling rate: {}'.format(fs))
    if fs != TARGET_FS:
        data = resample(data, int(data.shape[1] * TARGET_FS / fs), axis=1)
        fs = TARGET_FS
    fs = int(round(fs))          # pyedflib returns float; callers slice with it
    duration_s = data.shape[1] / fs
    return data.astype(np.float32), fs, duration_s
