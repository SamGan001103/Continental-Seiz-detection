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
# Headroom reserved beyond the recording itself, from measurement rather than
# taste: Qt plus the main window is 0.13 GB and TensorFlow plus the loaded
# detector another 0.38 GB, so the application occupies about 0.5 GB before a
# single sample is read. The rest is slack — `available_bytes` is a snapshot,
# and the moment a machine starts paging a clinician is looking at a frozen
# screen. At 1.0 GB the estimate let a 24-hour recording through on a machine
# with 6 GB free, which is the case that caused the freeze this module exists
# to prevent; that is the wrong side to be wrong on.
SAFETY_MARGIN_BYTES = 3 << 29          # 1.5 GB


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
    needed = resident + transient + SAFETY_MARGIN_BYTES
    avail = available_bytes() if available is None else int(available)
    fits = True if avail is None else needed <= avail
    return {
        'path': path, 'duration_s': duration_s, 'channels': n_ch,
        'fs': fs, 'resident_bytes': resident, 'transient_bytes': transient,
        'needed_bytes': needed, 'available_bytes': avail, 'fits': fits,
    }


def describe_refusal(est):
    """The dialog text for a recording that will not fit. Numbers, and the
    way out — never just 'too big'."""
    gb = lambda b: '{:.1f} GB'.format(b / 1e9)     # noqa: E731
    hours = est['duration_s'] / 3600.0
    text = (
        'This recording is {:.1f} hours long ({} channels at {:.0f} Hz).\n\n'
        'Opening it needs about {} of memory — {} to hold the signal and '
        '{} more while it is read — and this computer currently has {} '
        'available.\n\n'
        'Loading it anyway would push the machine into swap and can freeze '
        'it, so the recording was not opened.\n\n'
        'What works: close other programs and try again, use a computer with '
        'more memory, or split the recording into shorter files.'
    ).format(hours, est['channels'], est['fs'], gb(est['needed_bytes']),
             gb(est['resident_bytes']), gb(est['transient_bytes']),
             'unknown' if est['available_bytes'] is None
             else gb(est['available_bytes']))
    return text
