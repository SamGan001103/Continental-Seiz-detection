"""Decide, BEFORE loading, whether a recording will fit in this machine's RAM.

A ward PC has 8 GB and no one at the keyboard who can diagnose a swap storm.
On 2026-08-31 the development machine (16 GB) was frozen to a power cycle by
opening a 24-hour recording; a freeze is the worst possible failure for a tool
that is meant to sit on many hospital PCs, because it takes down everything
else the clinician had open and leaves no message. A refusal with numbers is
strictly better, and it costs one header read.

What the estimate covers
------------------------
`gui.io.edf.load_edf_19ch` reads every channel in the file through pyst (which
holds them as float64), selects the 19 the detector needs, resamples with an
FFT when the file is not at 250 Hz (complex intermediates on top), and finally
casts to float32. So the transient is several times the array that survives:

    resident   19 x N250 x 4 bytes                  (kept for the session)
    transient  C  x Nsrc x 8 bytes                  (pyst, all file channels)
             + 19 x max(Nsrc, N250) x 24 bytes      (FFT resample, if any)

where N250 is the sample count at 250 Hz and C the channel count in the file.
Deliberately pessimistic; the point is to refuse the files that would freeze
the machine, not to admit every file that might just squeak through.
"""
import ctypes
import os
import sys

TARGET_FS = 250
BYTES_F32, BYTES_F64, BYTES_FFT = 4, 8, 24
# Headroom beyond the recording itself, as a floor plus a proportion.
#
# A large FIXED margin was wrong twice over. It DOUBLE-COUNTED: the check runs
# inside the already-running application, so Qt's 0.13 GB and TensorFlow's
# 0.38 GB are allocated and therefore already excluded from `available_bytes`;
# reserving them again asks for memory we are holding. And it did not scale —
# at 1.5 GB fixed, a two-minute recording needing 30 MB was refused on a busy
# machine, which on a ward PC is a tool that refuses ordinary work all day.
#
# What the headroom must actually cover is the snapshot going stale between
# the check and the load, and leaving the OS enough not to page. Both scale
# with the size of the read, so the margin does too, with a floor for the
# small files where a fixed slab would dominate.
#
# The estimate it sits on is already deliberately pessimistic (every file
# channel as float64 AND resample intermediates, simultaneously), so this is
# slack on top of a worst case, not on top of a guess.
MIN_HEADROOM_BYTES = 1 << 28           # 256 MB floor
HEADROOM_FRACTION = 0.25               # ...or a quarter of the read, whichever is larger


def _headroom(load_bytes):
    return max(MIN_HEADROOM_BYTES, int(load_bytes * HEADROOM_FRACTION))


def available_bytes():
    """Physical memory currently available, or None if it cannot be read."""
    if sys.platform.startswith('win'):
        class _MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullAvailPhys)
        except Exception:                                   # noqa: BLE001
            return None
        return None
    try:
        pages = os.sysconf('SC_AVPHYS_PAGES')
        return int(pages * os.sysconf('SC_PAGE_SIZE'))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:                                       # noqa: BLE001
        return None


def estimate_bytes(n_channels_in_file, n_samples_source, fs_source):
    """(resident, transient) byte estimate for loading one recording."""
    n_src = int(n_samples_source)
    n_250 = int(n_src * TARGET_FS / float(fs_source))
    resident = 19 * n_250 * BYTES_F32
    transient = int(n_channels_in_file) * n_src * BYTES_F64
    if int(round(float(fs_source))) != TARGET_FS:
        transient += 19 * max(n_src, n_250) * BYTES_FFT
    return resident, transient


def edf_open_estimate(path, available=None):
    """Header-only check. Returns a dict; `fits` is the decision.

    `available` is injectable for tests; None reads the machine.
    """
    import pyedflib
    r = pyedflib.EdfReader(path)
    try:
        n_ch = int(r.signals_in_file)
        fs = max(float(r.getSampleFrequency(i)) for i in range(n_ch))
        n_src = int(round(r.file_duration * fs))
        duration_s = float(r.file_duration)
    finally:
        r.close()
    resident, transient = estimate_bytes(n_ch, n_src, fs)
    needed = resident + transient + _headroom(resident + transient)
    avail = available_bytes() if available is None else int(available)
    fits = True if avail is None else needed <= avail
    return {
        'path': path, 'duration_s': duration_s, 'channels': n_ch,
        'fs': fs, 'resident_bytes': resident, 'transient_bytes': transient,
        'needed_bytes': needed, 'available_bytes': avail, 'fits': fits,
    }


def describe_refusal(est):
    """The dialog text for a recording that can be neither held nor streamed.

    This is now a much rarer message than when it was written, and it says
    something different. A large recording is no longer refused for being
    large — it is opened span-by-span off disk. This text appears only when
    that also failed, which in practice means the file's 19 EEG channels
    cannot be resolved or they are not all at one sampling rate.

    So the size figures are the reason it cannot simply be loaded, and the
    channel problem is the reason it cannot be streamed either. Saying only
    the first would send a reviewer off to close programs and buy RAM for a
    file that would still not open on an empty machine.
    """
    gb = lambda b: '{:.1f} GB'.format(b / 1e9)     # noqa: E731
    hours = est['duration_s'] / 3600.0
    text = (
        'This recording is {:.1f} hours long ({} channels at {:.0f} Hz), and '
        'it could not be opened either way.\n\n'
        'It is too large to load into memory: that needs about {} — {} to '
        'hold the signal and {} more while it is read — and this computer has '
        '{} available. Loading it anyway would push the machine into swap and '
        'can freeze it.\n\n'
        'Reading it directly from disk instead, which is how large recordings '
        'are normally opened here, also failed. That usually means the 19 EEG '
        'channels this detector needs could not be found in the file, or they '
        'are not all recorded at the same sampling rate.\n\n'
        'What to check first: that this is a 19-channel EEG with standard '
        '10-20 labels. Failing that, more free memory would let it load the '
        'ordinary way.'
    ).format(hours, est['channels'], est['fs'], gb(est['needed_bytes']),
             gb(est['resident_bytes']), gb(est['transient_bytes']),
             'unknown' if est['available_bytes'] is None
             else gb(est['available_bytes']))
    return text
