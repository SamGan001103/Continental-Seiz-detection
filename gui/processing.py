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


# ---------------------------------------------------------------------- spectrum
#
# Quantitative EEG for the review panel: how much power the visible slice holds
# in each clinical band, and — the part that matters — which of those numbers the
# display filters have already decided. With this app's defaults (HP 1.0, LP 70,
# notch 60) an unannotated band table reports the filter, not the brain.
#
# These imports sit here instead of in the module header so the filter/montage
# block above stays byte-identical; tests/test_display_model_isolation.py pins
# that code, and other work in flight assumes it does not move.
from collections import OrderedDict, namedtuple      # noqa: E402

from scipy.signal import welch                       # noqa: E402
try:                       # scipy >= 1.15 renamed sosfreqz to freqz_sos
    from scipy.signal import freqz_sos as _sos_freq_response
except ImportError:        # the 3.6 stack ships scipy 1.2, which has neither
    from scipy.signal import sosfreqz as _sos_freq_response


# Standard clinical bands, in the order a reviewer reads them. Half-open in
# spirit ([lo, hi)) but integrated closed, so contiguous bands share an edge
# node and the five powers sum to the total exactly.
BANDS = OrderedDict([
    ('delta', (0.5, 4.0)),
    ('theta', (4.0, 8.0)),
    ('alpha', (8.0, 13.0)),
    ('beta', (13.0, 30.0)),
    ('gamma', (30.0, 80.0)),
])

# (compromised, retained, note) — `retained` is the fraction of a flat spectrum
# the display filter chain leaves in that band, so the GUI can show a scale
# factor rather than a vague warning.
BandCaveat = namedtuple('BandCaveat', 'compromised retained note')

# One row of the band table, ready to render.
BandRow = namedtuple('BandRow', 'name lo hi power ratio compromised note')

# Below this fraction of surviving power a band is called compromised: the
# reviewer must not read it as physiology. 0.9 is deliberately strict — a 10%
# bite out of a band is already larger than the differences people interpret.
COMPROMISED_BELOW = 0.9


# ------------------------------------------------------------- power spectrum


def _pow2_at_most(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _default_nperseg(n, fs):
    """Welch segment length giving ~0.5 Hz bins on a 12 s slice.

    Resolution is fs/nperseg. This asked for 2*fs samples and then rounded DOWN
    to a power of two, which at fs=250 means 500 -> 256, i.e. 0.98 Hz bins - not
    the ~0.5 Hz the docstring promised. The cost fell entirely on delta: with
    0.98 Hz bins the whole 0.5-0.98 Hz sub-band is interpolated between the DC
    bin, which detrend='constant' forces to zero, and the first real bin.
    Measured on unfiltered pure tones of known power, the estimate returned 31%
    of the truth at 0.6 Hz and 55% at 0.8 Hz, while 10 Hz came back at 99.8%.
    Delta is the band clinicians read for slowing, and it was the one the
    estimator quietly under-reported.

    Rounding UP to the next power of two fixes it: 512 at fs=250 gives 0.49 Hz
    bins, which is what was always claimed. A 12 s slice is 3000 samples, so
    that is still ~11 half-overlapping averaging segments - enough for a stable
    estimate. Clamped to the slice length so a short selection degrades to
    coarser bins instead of raising.

    Re-measured after the change, same tones:

        0.6 Hz -> 113%      0.8 Hz -> 113%
        1.0 Hz ->  99%     10.0 Hz -> 100%

    Sub-1 Hz now reads about 13% HIGH rather than 69% low. That residual is
    spectral leakage - a 0.6 Hz tone does not fit in a 2 s window, so a Hann
    window spreads it across neighbouring bins - and it is a property of the
    estimate, not a defect to chase. It errs toward reporting more delta than
    is there, which is the safer direction for a band read as slowing, but it
    means sub-1 Hz band power should be read as an indication rather than a
    measurement.
    """
    n = int(n)
    if n < 8:                      # too short for a windowed estimate at all
        return n
    want = _pow2_at_least(max(2, int(round(2.0 * float(fs)))))
    return max(8, min(want, _pow2_at_most(n)))


def _pow2_at_least(v):
    p = 1
    while p < int(v):
        p *= 2
    return p


def psd(x, fs, nperseg=None):
    """Welch power spectral density of a display slice, in uV^2/Hz.

    x : [N] or [K, N] in uV (display-path units). Returns (freqs [F],
    power [F] or [K, F]).

    Window, overlap, detrend and scaling are all passed explicitly rather than
    left to scipy's defaults: the two supported stacks are eight years of scipy
    apart (1.2 on Python 3.6, 1.17 on 3.11) and these numbers go in a thesis, so
    both must produce the same figure. Per-segment mean removal
    (detrend='constant') is what keeps the amplifier's DC offset — every EEG has
    one — out of the delta band, where it would otherwise dominate everything.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if n < 4:
        raise ValueError(
            'need at least 4 samples to estimate a spectrum, got {}'.format(n))
    if nperseg is None:
        nperseg = _default_nperseg(n, fs)
    nperseg = int(max(2, min(int(nperseg), n)))
    freqs, power = welch(x, fs=float(fs), window='hann', nperseg=nperseg,
                         noverlap=nperseg // 2, detrend='constant',
                         scaling='density', axis=-1)
    return freqs, power


# ------------------------------------------------------------- band integrals


def _interp_at(f, p, x):
    """Linearly interpolate p (spectrum along the last axis) at frequency x."""
    j = int(np.searchsorted(f, x))
    if j <= 0:
        return np.asarray(p[..., 0])
    if j >= f.size:
        return np.asarray(p[..., -1])
    f0, f1 = float(f[j - 1]), float(f[j])
    if f1 == f0:
        return np.asarray(p[..., j])
    w = (float(x) - f0) / (f1 - f0)
    return np.asarray(p[..., j - 1] * (1.0 - w) + p[..., j] * w)


def _trapz_last_axis(values, nodes):
    """Trapezoid rule along the last axis over an explicit, uneven node grid."""
    return np.sum(0.5 * (values[..., 1:] + values[..., :-1]) * np.diff(nodes),
                  axis=-1)


def _band_integral(freqs, power, lo, hi):
    """Integrate the PSD across [lo, hi], interpolating the two edge values.

    Summing the bins that happen to fall inside a band mis-counts up to a whole
    bin at each edge, and makes the answer depend on where the FFT grid lands
    relative to 13 Hz — so the same recording read at two segment lengths would
    give two different beta powers. Interpolating the endpoints also makes
    contiguous bands additive: the shared edge becomes a node in both integrals,
    so the five band powers sum to the 0.5-80 Hz total exactly.
    """
    f = np.asarray(freqs, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    lo = max(float(lo), float(f[0]))
    hi = min(float(hi), float(f[-1]))     # nothing above Nyquist was recorded
    if hi <= lo:
        return np.zeros(p.shape[:-1]) if p.ndim > 1 else np.float64(0.0)
    inside = (f > lo) & (f < hi)
    nodes = np.concatenate(([lo], f[inside], [hi]))
    vals = np.concatenate((_interp_at(f, p, lo)[..., None],
                           p[..., inside],
                           _interp_at(f, p, hi)[..., None]), axis=-1)
    return _trapz_last_axis(vals, nodes)


def _scalarise(v):
    return float(v) if np.ndim(v) == 0 else v


def band_powers(freqs, power):
    """Absolute power per clinical band, in uV^2 (PSD integrated, not summed).

    Returns an OrderedDict in BANDS order: band -> uV^2 (a float for a single
    channel, a [K] array if `power` came from a [K, N] slice).
    """
    out = OrderedDict()
    for name, (lo, hi) in BANDS.items():
        out[name] = _scalarise(_band_integral(freqs, power, lo, hi))
    return out


def _total_from_powers(powers):
    vals = list(powers.values())
    total = vals[0]
    for p in vals[1:]:
        total = total + p
    return _scalarise(total)


def _ratios_from_powers(powers):
    """Split out so band_table integrates the spectrum once, not three times."""
    total = _total_from_powers(powers)
    out = OrderedDict()
    for name, p in powers.items():
        if np.ndim(total) == 0:
            out[name] = (p / total) if total > 0 else 0.0
        else:
            out[name] = np.where(total > 0, p / np.where(total > 0, total, 1.0),
                                 0.0)
    return out


def total_band_power(freqs, power):
    """Total power across the analysed range (0.5-80 Hz), in uV^2.

    Deliberately the sum of the five bands rather than one integral over
    0.5-80 Hz: the two differ in the last digits (band edges are extra trapezoid
    nodes) and a table whose percentages do not add to 100 invites the reviewer
    to go looking for the missing power.
    """
    return _total_from_powers(band_powers(freqs, power))


def band_ratios(freqs, power):
    """Relative band power — each band as a fraction of the 0.5-80 Hz total.

    Returns an OrderedDict in BANDS order; the values sum to 1. A dead or
    disconnected electrode has no power at all, and 0/0 would put 'nan%' in
    every row of the table, so a flat slice reports zeros instead.
    """
    return _ratios_from_powers(band_powers(freqs, power))


# --------------------------------------------------- what the filters removed


def _display_chain_sos(fs, hp, lp, notch):
    """The sections apply_filters would build, under its exact conditions.

    Mirroring the *conditions* matters as much as the coefficients: apply_filters
    silently drops a low-pass at or above Nyquist and ignores non-positive
    cutoffs, and a caveat warning about a filter that never ran would be its own
    kind of misinformation. Only the fundamental notch is applied there, so
    nothing here may claim the harmonics are notched either.
    """
    out = []
    if hp is not None and hp > 0:
        out.append(_butter_sos(hp, fs, 'highpass', order=2))
    if lp is not None and lp > 0 and lp < fs / 2:
        out.append(_butter_sos(lp, fs, 'lowpass', order=4))
    if notch is not None and notch > 0:
        try:
            out.append(_notch_sos(notch, fs))
        except ValueError:
            # A notch at or above Nyquist cannot be built. apply_filters would
            # raise on the same input; the band table must not be the thing that
            # takes the window down, so report the chain without it.
            pass
    return out


def _power_gain(sos_list, freqs_hz, fs):
    """|H(f)|**4 for a chain of sos sections at the given frequencies.

    Fourth power, not squared: sosfiltfilt runs each filter forwards and back,
    so the amplitude spectrum is multiplied by |H|**2 and the power spectrum —
    which is what a band table integrates — by |H|**4. Reporting |H|**2 would
    understate the bite the 70 Hz low-pass takes out of gamma by half.
    """
    w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) / float(fs)
    gain = np.ones(w.shape, dtype=np.float64)
    for sos in sos_list:
        # worN in rad/sample rather than the fs= keyword, which scipy 1.2 (the
        # Python 3.6 stack) does not have.
        _, h = _sos_freq_response(sos, worN=w)
        gain = gain * (np.abs(h) ** 4)
    return gain


def _band_grid(lo, hi, notch):
    """Frequencies to evaluate a filter response on across one band.

    Uniform sampling alone misses the notch: at Q=30 its -3 dB width is only
    f0/30 (2 Hz at 60 Hz), so a 0.4 Hz grid straddles the null and the estimate
    of what it removes depends on luck. Extra nodes clustered on f0 pin it.
    """
    grid = np.linspace(lo, hi, 257)
    if notch is not None and lo < notch < hi:
        offs = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75,
                         1.0, 1.5, 2.0, 3.0])
        extra = np.concatenate((notch - offs, notch + offs))
        grid = np.unique(np.concatenate((grid, extra[(extra > lo) &
                                                     (extra < hi)])))
    return grid


# Severity wording keyed on how much of the band survives that one filter.
_SEVERITY = ((0.02, 'removed by'),
             (0.5, 'mostly removed by'),
             (0.9, 'attenuated by'),
             (0.98, 'slightly attenuated by'))


def _severity_clause(retained, hz, kind):
    for limit, verb in _SEVERITY:
        if retained < limit:
            return '{} the {:g} Hz {} filter'.format(verb, hz, kind)
    return None


def _join_clauses(clauses):
    if len(clauses) == 1:
        return clauses[0]
    return '{} and {}'.format(', '.join(clauses[:-1]), clauses[-1])


_CAVEAT_CACHE = {}


def filter_caveats(fs, hp=None, lp=None, notch=None):
    """Which clinical bands the DISPLAY filters have already decided.

    Returns an OrderedDict band -> BandCaveat(compromised, retained, note).

    Without this the band table is misinformation. At the app's defaults the
    1 Hz high-pass has taken a quarter of the delta band's power — nearly all of
    it below 1 Hz — before anything was integrated, and the 70 Hz low-pass plus
    the 60 Hz notch leave only about 63% of gamma. Neither loss is visible in the
    numbers themselves, so delta looks small and gamma looks quiet for reasons
    that have nothing to do with the patient.

    `retained` is the fraction of a FLAT spectrum that survives the chain across
    that band — what the filters would leave of WHITE NOISE — computed from the
    frequency response of the very same sos sections apply_filters builds. So it
    is a measured scale factor rather than a hand-set warning level.

    It is NOT the fraction of true power a real recording retains, and the
    difference is large at the edges of the range. EEG is roughly 1/f, so its
    power piles up exactly where a high-pass cuts. Measured against pink noise
    at the app defaults (HP 1.0, LP 70, notch 60):

        band    this figure says    1/f actually retains
        delta          0.749                0.345
        gamma          0.632                0.797

    Delta is out by a factor of 2.2, in the reassuring direction. Treat this as
    "the filters attenuate this band" and not as a number to quote or correct
    by. An earlier version of this docstring claimed a band reading 63% "really
    does read about 63% of the true power"; that is false on any spectrum that
    is not flat, which is all of them.

    This describes the filters only. It cannot know about a bad electrode, and
    `retained` is not a correction factor to divide by: the filters removed that
    power, they did not scale it.
    """
    key = (float(fs), hp, lp, notch)
    hit = _CAVEAT_CACHE.get(key)
    if hit is not None:
        return hit

    nyq = float(fs) / 2.0
    chain = _display_chain_sos(fs, hp, lp, notch)
    # Each filter's own contribution, so the sentence can name the culprit.
    single = [('hp', hp, _display_chain_sos(fs, hp, None, None), 'high-pass'),
              ('lp', lp, _display_chain_sos(fs, None, lp, None), 'low-pass'),
              ('notch', notch, _display_chain_sos(fs, None, None, notch), None)]

    out = OrderedDict()
    for name, (lo, hi) in BANDS.items():
        width = hi - lo
        top = min(hi, nyq)
        if top <= lo:
            # The recording never contained this band; nothing was filtered out
            # because nothing was ever there to filter.
            out[name] = BandCaveat(
                True, 0.0,
                'not recorded: the whole band is above the {:g} Hz Nyquist '
                'limit of this {:g} Hz recording'.format(nyq, fs))
            continue

        grid = _band_grid(lo, top, notch)
        # _trapz_last_axis rather than np.trapz: numpy 2 renamed that to
        # np.trapezoid and the module has to import on numpy 1.16 as well.
        retained = float(
            _trapz_last_axis(_power_gain(chain, grid, fs), grid) / width)

        clauses = []
        notched_inside = False
        for kind, hz, sos_list, label in single:
            if not sos_list:
                continue
            # Divided by the RECORDED width (lo..top), not the full band width.
            # Dividing by `width` charged Nyquist truncation to every filter in
            # the chain, so at fs=100 a 1 Hz high-pass was reported as having
            # "mostly removed" gamma - a band it does not touch, and of which
            # 100% of the recorded part survives. `retained` above stays on the
            # full width, because there the truncation genuinely is lost power.
            own_width = max(1e-12, float(top) - float(lo))
            own = float(
                _trapz_last_axis(_power_gain(sos_list, grid, fs), grid)
                / own_width)
            if kind == 'notch':
                if lo < hz < top:
                    notched_inside = True
                    clauses.append('notched at {:g} Hz'.format(hz))
                continue
            clause = _severity_clause(own, hz, label)
            if clause is not None:
                clauses.append(clause)
        if top < hi:
            clauses.append(
                'measured only to the {:g} Hz Nyquist limit of this {:g} Hz '
                'recording'.format(nyq, fs))

        # A notch inside the band always counts, however little of the band's
        # total power it takes: it is a hole at one frequency, not a gentle
        # slope, and whatever sat at 60 Hz — mains or brain — is gone from a
        # number the reviewer would otherwise compare across recordings.
        compromised = retained < COMPROMISED_BELOW or notched_inside
        if clauses:
            note = '{} ({:.0f}% of the band survives)'.format(
                _join_clauses(clauses), 100.0 * retained)
        else:
            note = ''
        out[name] = BandCaveat(compromised, retained, note)

    if len(_CAVEAT_CACHE) > 64:       # toolbar settings, not a hot key space
        _CAVEAT_CACHE.clear()
    _CAVEAT_CACHE[key] = out
    return out


def band_table(x, fs, hp=None, lp=None, notch=None, nperseg=None):
    """One call for the review panel: band powers plus their filter caveats.

    `x` is the slice the reviewer is looking at — post apply_filters, post
    montage — and hp/lp/notch are the settings that produced it. Passing the
    displayed signal is the point: the table must describe what is on screen,
    and the caveats then say which rows are the filter talking.

    Returns a list of BandRow(name, lo, hi, power uV^2, ratio, compromised,
    note) in BANDS order. For a [K, N] input, power and ratio are [K] arrays.
    """
    freqs, power = psd(x, fs, nperseg=nperseg)
    powers = band_powers(freqs, power)
    ratios = _ratios_from_powers(powers)
    caveats = filter_caveats(fs, hp=hp, lp=lp, notch=notch)
    rows = []
    for name, (lo, hi) in BANDS.items():
        cav = caveats[name]
        rows.append(BandRow(name, lo, hi, powers[name], ratios[name],
                            cav.compromised, cav.note))
    return rows
