"""EDF loading for the GUI. Reuses the repo's existing read_edf_elec + the
19-channel montage declared in utils/params_common_electrodes.txt."""
import os
import sys
import numpy as np
from scipy.signal import resample

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.paths import params_path  # noqa: E402
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
    # The montage file is passed as an absolute path. This used to os.chdir into
    # utils/ because pyst resolved a bare filename against the cwd — which
    # mutated global process state on every load and has no equivalent in a
    # frozen build, where utils/ is a directory inside the bundle.
    try:
        fs, data = read_edf_elec(path, parameters=params_path())
    except SystemExit as ex:
        raise RuntimeError(
            'Could not find the required 19 EEG channels in this EDF. '
            'Expected TUH-style labels for: {}'.format(
                ', '.join(CHANNELS_19))) from ex
    fs = float(fs)
    if fs <= 0:
        raise RuntimeError('Invalid EDF sampling rate: {}'.format(fs))
    if fs != TARGET_FS:
        data = resample(data, int(data.shape[1] * TARGET_FS / fs), axis=1)
        fs = TARGET_FS
    fs = int(round(fs))          # pyedflib returns float; callers slice with it
    duration_s = data.shape[1] / fs
    return data.astype(np.float32), fs, duration_s


def read_start_datetime(path):
    """When the recording began, from the EDF header, or None.

    Every EDF carries a start date and time and this application was reading
    neither, so the x axis could only ever count seconds. On a day-long
    recording that is the difference between "48,213 seconds" and "03:22" —
    and clock time is what every nursing note, video review and clinician's
    recollection is written in, so it is the only thing the trace can be
    cross-referenced against.

    Returns None rather than raising: a missing or malformed header date is a
    reason to fall back to elapsed time, never a reason not to open a
    recording. Some vendors write a placeholder date (1985-01-01 is the EDF
    spec's own "unknown"), which is treated as absent.
    """
    import datetime
    try:
        import pyedflib
        r = pyedflib.EdfReader(os.path.abspath(path))
    except Exception:                                        # noqa: BLE001
        return None
    try:
        dt = r.getStartdatetime()
    except Exception:                                        # noqa: BLE001
        return None
    finally:
        try:
            r.close()
        except Exception:                                    # noqa: BLE001
            pass
    if not isinstance(dt, datetime.datetime):
        return None
    # The EDF spec uses 1985-01-01 as "date unknown", and a year of 1900 or
    # earlier is a writer that left the field empty. Either way the clock time
    # may still be real, but the date is not, so do not present it as one.
    if dt.year <= 1900:
        return None
    return dt
