"""Automatic interictal-discharge candidates, from SpikeNet1.

Why this exists
---------------
The reviewing clinician asked to see interictal epileptiform discharges beside
the seizure detections, and recalled that open-source detectors for them exist.
They do, and one is directly usable: **SpikeNet1** (Jing J, Sun H, ... Westover
MB, *Development of Expert-Level Automated Detection of Epileptiform Discharges
During Electroencephalogram Interpretation*, JAMA Neurol 2020;77(1):103-108,
PMID 31633740). Unusually for this field its trained weights are published in
the repository itself rather than behind a credentialed portal, so it can be
run without a data-use agreement.

What it does and does not give you
----------------------------------
SpikeNet1 emits ONE number per one-second window for the whole montage. It does
not say which electrode the discharge is at, and this module does not pretend
otherwise: `channel` on a detected mark is always None. A crude
maximum-amplitude hint is available via `channel_hint`, but it comes from the
signal, not from the model, and it is kept out of the mark so that nothing
downstream can mistake a heuristic for a detection.

The scores are NOT calibrated probabilities on a continuous recording. The
published figures are for classifying candidate events that a different stage
had already proposed; sliding the model across unselected EEG is a different
task with a different operating point. A threshold therefore has to be chosen
against labelled data, and until that is done anything this produces is
**unvalidated and must be presented to a reviewer as a candidate, never as a
finding**. Measured on `pat_102_part02`, background scores sit at a median of
0.02 and threshold 0.8 yields roughly two candidates an hour.

Relation to the seizure detector
--------------------------------
Completely separate, deliberately. Different model, different window length
(1 s against 12 s), different sampling rate (128 Hz against the detector's),
and no ICA — which is why it costs about 0.2 min per hour of EEG against the
seizure path's ~54 min per 24 h. Nothing here touches `gui.io.infer`, the
seizure weights, or any display module.

Licensing
---------
The upstream repository contradicts itself: LICENSE.txt says CC BY-NC 4.0,
README.md says CC BY-ND 4.0. Non-commercial academic use with citation
satisfies either reading. The weights are therefore NOT vendored into this
repository; `experiments/fetch_spikenet.py` downloads them to
`models/spikenet1/` on request, and `available()` reports whether that has
happened. Clarify with the authors before any distribution beyond research.
"""
import os
import warnings

import numpy as np

from gui.paths import resource

# SpikeNet's own channel order. Not alphabetical, not the detector's order --
# the weights were trained against exactly this sequence, so it is a contract.
MONOPOLAR = ('FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
             'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2')

# The 18 bipolar derivations appended to the 19 common-average ones to make the
# 37 input channels the model expects.
BIPOLAR = ('FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6',
           'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
           'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ')

FS = 128                    # the rate the weights were trained at
WINDOW = 128                # one second
DEFAULT_STEP = 16           # eight windows a second
DEFAULT_THRESHOLD = 0.8
DEFAULT_NOTCH_HZ = 50.0     # mains; 60.0 in North America
REFRACTORY_S = 0.5          # below this, two hits are one discharge

MODEL_DIR = resource('models', 'spikenet1')
STRUCTURE = 'spikenet1.o_structure.txt'
WEIGHTS = 'spikenet1.o_weights.h5'

CITATION = ('Jing J, Sun H, Kim JA, et al. Development of Expert-Level '
            'Automated Detection of Epileptiform Discharges During '
            'Electroencephalogram Interpretation. JAMA Neurol. '
            '2020;77(1):103-108. PMID 31633740.')
SOURCE = 'spikenet1'


def model_paths(model_dir=None):
    d = model_dir or MODEL_DIR
    return os.path.join(d, STRUCTURE), os.path.join(d, WEIGHTS)


def available(model_dir=None):
    """True when the weights have been fetched and can be loaded."""
    return all(os.path.exists(p) for p in model_paths(model_dir))


def missing_reason(model_dir=None):
    """A sentence to show a user when the detector cannot run, or None."""
    if available(model_dir):
        return None
    d = model_dir or MODEL_DIR
    return ('SpikeNet weights not found in {}. Run '
            '`python experiments/fetch_spikenet.py` to download them '
            '(1.3 MB, no account needed).'.format(d))


def load_model(model_dir=None):
    """Load SpikeNet1.

    Imported lazily: the review interface must open on a machine with no
    TensorFlow at all, and must not pay the import cost for a feature nobody
    has switched on.

    Keras 3 CANNOT load this model -- the saved config is keras_version 2.2.2
    and Keras 3 raises `TypeError: Could not locate class 'Model'`. The process
    must therefore have TF_USE_LEGACY_KERAS=1 set BEFORE tensorflow is
    imported. Setting it here would be too late if something else already
    imported TF, so this checks and says so plainly instead of failing deep
    inside the deserialiser.
    """
    structure, weights = model_paths(model_dir)
    if not os.path.exists(weights):
        raise RuntimeError(missing_reason(model_dir))
    if os.environ.get('TF_USE_LEGACY_KERAS') != '1':
        import sys
        if 'tensorflow' in sys.modules:
            raise RuntimeError(
                'TensorFlow was imported without TF_USE_LEGACY_KERAS=1, and '
                'Keras 3 cannot load SpikeNet1 (its config is Keras 2.2.2). '
                'Set that variable before importing tensorflow.')
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
    from tensorflow.keras.models import model_from_json
    with open(structure) as fh:
        model = model_from_json(fh.read())
    model.load_weights(weights)
    return model


def _electrode_key(name):
    """Reduce a channel label to a bare electrode name.

    The recordings reach us with TUH-style labels ('EEG FP1-REF'), EPILEPSIAE
    ones, and plain ones, so matching has to be on the electrode alone.
    """
    key = str(name).upper().strip()
    for junk in ('EEG ', 'EEG-', '-REF', '-LE', '-AVG', 'REF'):
        key = key.replace(junk, '')
    return key.strip().strip('-').strip()


def channel_index(names):
    """Row indices of SpikeNet's 19 electrodes within `names`.

    Raises if any is absent, because silently substituting a channel would
    change what the model sees without changing what it reports.
    """
    lookup = {}
    for i, n in enumerate(names):
        lookup.setdefault(_electrode_key(n), i)
    missing = [e for e in MONOPOLAR if e not in lookup]
    if missing:
        raise ValueError(
            'SpikeNet needs all 19 electrodes; missing: {}'
            .format(', '.join(missing)))
    return [lookup[e] for e in MONOPOLAR]


def preprocess(data_uv, fs, notch_hz=DEFAULT_NOTCH_HZ):
    """19 channels of microvolts at `fs` -> 37 channels at 128 Hz.

    `data_uv` must already be in SpikeNet's channel order and in MICROVOLTS.
    The model's output is amplitude-sensitive -- the same recording scaled by
    ten moves the median score by an order of magnitude -- so a caller handing
    this volts from MNE without the 1e6 will get plausible nonsense rather than
    an error.
    """
    from mne.filter import filter_data, notch_filter
    import scipy.signal as ss

    x = np.asarray(data_uv, dtype='float64')
    if x.shape[0] != len(MONOPOLAR):
        raise ValueError('expected {} channels, got {}'
                         .format(len(MONOPOLAR), x.shape[0]))
    if int(round(fs)) != FS:
        # resample_poly, not resample. scipy's `resample` is FFT-based and
        # therefore GLOBAL: every output sample depends on the whole input, so
        # resampling a chunk gives different numbers from resampling the same
        # span inside a longer segment, and a chunked run disagrees with a
        # single-pass run on borderline windows. resample_poly is a local FIR
        # decimation, so chunks agree with the whole once the filter edges are
        # discarded -- which is what makes detect_in_chunks trustworthy.
        # 256 -> 128 is exactly 1:2, the common case here.
        from math import gcd
        src = int(round(float(fs)))
        g = gcd(FS, src) or 1
        x = ss.resample_poly(x, FS // g, src // g, axis=1)
    if x.shape[1] < min_segment_s() * FS:
        # MNE emits a RuntimeWarning here that is easy to miss in a log. Say it
        # in our own terms, because the consequence is specific: the high-pass
        # distorts both ends and a discharge near them can be smeared away or
        # invented. `detect_in_chunks` exists to avoid this.
        warnings.warn(
            'segment is {:.1f} s; the 0.5 Hz high-pass needs at least {:.1f} s '
            'to settle, so the ends are distorted and marks near them are '
            'unreliable. Use detect_in_chunks() for long recordings.'
            .format(x.shape[1] / float(FS), min_segment_s()),
            RuntimeWarning, stacklevel=2)
    x = filter_data(x, FS, 0.5, None, method='fir', verbose=False)
    if notch_hz:
        x = notch_filter(x, FS, float(notch_hz), method='fir', verbose=False)
    pairs = np.array([[MONOPOLAR.index(p.split('-')[0]),
                       MONOPOLAR.index(p.split('-')[1])] for p in BIPOLAR])
    average = x - x.mean(axis=0)
    bipolar = x[pairs[:, 0]] - x[pairs[:, 1]]
    return np.concatenate([average, bipolar], axis=0).astype('float32')


def score_windows(model, eeg37, step=DEFAULT_STEP, batch_size=256,
                  grid_offset=0):
    """Slide the model across a preprocessed segment.

    Returns (scores, offsets_in_seconds). Both are empty when the segment is
    shorter than one window.

    `grid_offset` shifts the first window so that windows land on a grid fixed
    to the START OF THE RECORDING rather than to this segment. Without it each
    chunk starts its own grid, the windows land at a different phase relative
    to a discharge, and a chunked run reports marks a few tens of milliseconds
    off from a single-pass run — with borderline ones flipping either way.
    """
    n = eeg37.shape[1]
    step = int(step)
    first = int(grid_offset) % step
    if n < WINDOW + first:
        return np.zeros(0, 'float32'), np.zeros(0, 'float64')
    starts = np.arange(first, n - WINDOW + 1, step)
    batch = np.stack([eeg37[:, s:s + WINDOW] for s in starts])
    batch = batch.transpose(0, 2, 1)[:, :, None, :]
    scores = model.predict(batch, batch_size=batch_size, verbose=0).ravel()
    return scores, starts / float(FS)


def is_flat(eeg37, floor_uv=1.0):
    """Gaps in a converted recording are zero-filled, and a detector run over
    one would report whatever it makes of silence. Treated as not assessed,
    the same way the seizure path treats a near-DC window."""
    return float(np.abs(eeg37).max()) < floor_uv


def merge(scores, offsets, threshold=DEFAULT_THRESHOLD,
          refractory_s=REFRACTORY_S, t0=0.0):
    """Supra-threshold windows -> discrete candidate marks.

    With a 16-sample step a single discharge trips eight consecutive windows.
    Reporting eight marks for one spike would make the count meaningless and
    the display unreadable, so runs closer together than the refractory period
    collapse to their highest-scoring window.
    """
    hits = np.flatnonzero(np.asarray(scores) > float(threshold))
    if not len(hits):
        return []
    marks, run = [], [hits[0]]
    for i in hits[1:]:
        if offsets[i] - offsets[run[-1]] <= refractory_s:
            run.append(i)
        else:
            marks.append(_peak(scores, offsets, run, t0))
            run = [i]
    marks.append(_peak(scores, offsets, run, t0))
    return marks


def _peak(scores, offsets, run, t0):
    best = max(run, key=lambda i: scores[i])
    return {
        'time': float(t0 + offsets[best] + WINDOW / (2.0 * FS)),
        'score': float(scores[best]),
        'channel': None,          # SpikeNet localises nothing; see module docs
        'source': SOURCE,
        'label': 'ied',
        'note': 'auto, unconfirmed',
    }


def channel_hint(eeg37, t_rel):
    """Which bipolar derivation is largest at `t_rel`, as a REVIEW AID ONLY.

    This is maximum absolute amplitude in the window, not a model output, and
    it is deliberately not written onto the mark. A noisy electrode wins this
    comparison as easily as a real discharge does.
    """
    centre = int(round(t_rel * FS))
    a = max(0, centre - WINDOW // 2)
    b = min(eeg37.shape[1], a + WINDOW)
    if b <= a:
        return None
    seg = np.abs(eeg37[len(MONOPOLAR):, a:b])
    return BIPOLAR[int(np.argmax(seg.max(axis=1)))]


def filter_edge_s():
    """Seconds at each end that the 0.5 Hz high-pass distorts.

    MNE builds an 845-tap FIR at 128 Hz for a 0.5 Hz transition, which is 6.6
    seconds of kernel. Filtering a segment shorter than that does not raise —
    it returns a distorted answer and a RuntimeWarning nobody reads. Worse,
    when a long recording is cut into chunks, every chunk boundary gets the
    same distortion, and a discharge sitting near one is either smeared away
    or invented. `detect_in_chunks` overlaps by this margin and discards the
    marks inside it for exactly that reason.

    Derived from the filter MNE actually builds rather than hard-coded, so it
    stays correct if the filter design changes under us.
    """
    from mne.filter import create_filter
    kernel = create_filter(None, FS, 0.5, None, method='fir', verbose=False)
    return (len(kernel) // 2) / float(FS)


def min_segment_s():
    """Shortest segment that can be filtered without edge distortion."""
    return 2.0 * filter_edge_s()


def detect_in_chunks(read, n_samples, fs, names=None, chunk_s=300.0,
                     threshold=DEFAULT_THRESHOLD, step=DEFAULT_STEP,
                     notch_hz=DEFAULT_NOTCH_HZ, model=None, model_dir=None,
                     progress=None):
    """Run the detector over a long recording without loading it.

    `read(a, b)` returns samples [a, b) as channels x samples in MICROVOLTS;
    `n_samples` is the length of the recording in samples at `fs`. Memory stays
    bounded by `chunk_s` regardless of how long the recording is, which is the
    point — these are 24-hour files.

    Each chunk is read with `filter_edge_s()` of context on both sides and the
    marks falling in that context are dropped, so a discharge is detected
    exactly once and never from distorted signal.
    """
    if model is None:
        model = load_model(model_dir)
    fs = float(fs)
    edge = filter_edge_s()
    chunk_s = max(float(chunk_s), min_segment_s() * 2.0)
    edge_n = int(round(edge * fs))
    body_n = int(round(chunk_s * fs))
    out = []
    pos = 0
    while pos < n_samples:
        a = max(0, pos - edge_n)
        b = min(n_samples, pos + body_n + edge_n)
        if b - a < int(round(min_segment_s() * fs)):
            break
        data = read(a, b)
        if names is not None:
            data = data[channel_index(names), :]
        eeg = preprocess(data, fs, notch_hz=notch_hz)
        if not is_flat(eeg):
            # Where this chunk begins on the recording-wide 128 Hz grid, so
            # every chunk scores the same windows the whole file would.
            a128 = int(round(a * FS / fs))
            scores, offsets = score_windows(model, eeg, step=step,
                                            grid_offset=-a128)
            found = merge(scores, offsets, threshold=threshold, t0=a / fs)
            # Keep only what belongs to this chunk's body. The leading context
            # was already covered by the previous chunk; the trailing context
            # will be covered by the next.
            lo = pos / fs
            hi = min(n_samples, pos + body_n) / fs
            out.extend(m for m in found if lo <= m['time'] < hi)
        if progress is not None:
            progress(min(1.0, (pos + body_n) / float(n_samples)))
        pos += body_n
    return out


def detect(data_uv, fs, names=None, threshold=DEFAULT_THRESHOLD,
           step=DEFAULT_STEP, notch_hz=DEFAULT_NOTCH_HZ, t0=0.0, model=None,
           model_dir=None):
    """Candidate interictal discharges in one segment of EEG.

    `data_uv` is channels x samples in MICROVOLTS. If `names` is given the
    rows are reordered into SpikeNet's order first; otherwise they are assumed
    to be in it already.

    Returns a list of marks in the shape `gui.io.ied_marks` writes, each
    carrying its score and `source='spikenet1'` so a reviewer's own marks stay
    distinguishable from the machine's proposals.
    """
    x = np.asarray(data_uv)
    if names is not None:
        x = x[channel_index(names), :]
    eeg = preprocess(x, fs, notch_hz=notch_hz)
    if is_flat(eeg):
        return []
    if model is None:
        model = load_model(model_dir)
    scores, offsets = score_windows(model, eeg, step=step)
    return merge(scores, offsets, threshold=threshold, t0=t0)
