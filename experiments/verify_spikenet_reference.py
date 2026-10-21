"""Check this repository's SpikeNet integration against the authors' own output.

    python experiments/verify_spikenet_reference.py

The seizure detector could be checked against a published AUC. The interictal
detector could not: there is no annotated interictal ground truth in any
dataset available here, so until now the only evidence was that the input
contract was implemented as specified -- channel order, the 37-channel
montage expansion, 128 Hz, the 0.5 Hz high-pass and the notch. That verifies
the model is being fed what it expects. It says nothing about whether the
same numbers come out.

The upstream repository settles it. It ships both an example recording and
the scores its own script produced from that recording:

    Data/raw/sample_eeg.mat        19 channels, 23040 samples, 128 Hz
    Data/ssd/SSD_sample_eeg.mat    `yp`, one score per sample

Running this repository's preprocessing and scoring on their input and
comparing against their output is an end-to-end replication check, and it is
exact rather than statistical: the weights are fixed, the model is
deterministic, so any difference is a difference in my code.

Result when written: **max absolute difference 0.000000 over 22,913 windows**,
Pearson r = 1.000000, and identical decisions at every threshold.

Note on the comparison. Their script steps the window by ONE sample and pads
the score sequence at both ends to the sample count, so `yp` is per-sample.
This repository steps by 16 by default -- 16x fewer windows for a day-long
recording -- so the check runs at step 1 to line the two up. The step is a
cost choice, not a difference in the model.
"""
import os
import sys

# Before anything imports TensorFlow: SpikeNet's config is Keras 2.2.2 and
# Keras 3 cannot load it.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BASE = 'https://raw.githubusercontent.com/bdsp-core/SpikeNet1/main/Data/'
FILES = (('raw', 'sample_eeg.mat'), ('ssd', 'SSD_sample_eeg.mat'))
CACHE = os.path.join(REPO, 'artifacts', 'spikenet_reference')


def fetch():
    """The authors' example recording and the scores they published for it."""
    import urllib.request
    os.makedirs(CACHE, exist_ok=True)
    for sub, name in FILES:
        out = os.path.join(CACHE, name)
        if os.path.exists(out):
            continue
        print('  fetching {} ...'.format(name))
        with urllib.request.urlopen(BASE + sub + '/' + name, timeout=180) as r:
            body = r.read()
        with open(out, 'wb') as fh:
            fh.write(body)
    return [os.path.join(CACHE, n) for _, n in FILES]


def main():
    import h5py
    import numpy as np
    from scipy.io import loadmat
    from gui.io import ied_detect as DET

    src, ref = fetch()
    # v7.3 .mat is HDF5, and stores arrays transposed.
    with h5py.File(src, 'r') as f:
        data = np.array(f['data']).T
        fs = float(np.array(f['Fs']).ravel()[0])
        names = [''.join(chr(c) for c in np.array(f[r]).ravel())
                 for r in np.array(f['channels']).ravel()]
    yp = np.asarray(loadmat(ref)['yp']).ravel()

    print('  reference input : {} ch x {} samples @ {:g} Hz'
          .format(data.shape[0], data.shape[1], fs))
    same_order = ([n.upper() for n in names]
                  == [n.upper() for n in DET.MONOPOLAR])
    print('  channel order matches this repository: {}'.format(same_order))
    if not same_order:
        raise SystemExit('channel order differs; the comparison would be '
                         'meaningless:\n  theirs {}\n  mine   {}'
                         .format(names, list(DET.MONOPOLAR)))

    if not DET.available():
        raise SystemExit('SpikeNet weights not present. Run '
                         'experiments/fetch_spikenet.py first.')

    eeg37 = DET.preprocess(data, fs, notch_hz=60.0)   # their script's notch
    model = DET.load_model()
    mine, _ = DET.score_windows(model, eeg37, step=1, batch_size=1000)
    mine = np.asarray(mine).ravel()

    n_win = data.shape[1] - DET.WINDOW + 1
    pad = (yp.size - n_win) // 2
    theirs = yp[pad:pad + n_win]
    k = min(theirs.size, mine.size)
    a, b = theirs[:k], mine[:k]
    worst = float(np.abs(a - b).max())

    print()
    print('  windows compared        : {}'.format(k))
    print('  max absolute difference : {:.3e}'.format(worst))
    print('  mean absolute difference: {:.3e}'.format(
        float(np.abs(a - b).mean())))
    print('  Pearson correlation     : {:.6f}'.format(
        float(np.corrcoef(a, b)[0, 1])))
    for thr in (0.3, 0.5, 0.8):
        print('  identical decisions at {:.1f}: {:.4f}%'.format(
            thr, 100.0 * float(((a >= thr) == (b >= thr)).mean())))

    ok = worst < 1e-5
    print()
    print('  VERDICT: {}'.format(
        'reproduces the authors\' published output' if ok
        else 'DOES NOT reproduce - investigate before relying on any mark'))
    return 0 if ok else 1


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    sys.exit(main())
