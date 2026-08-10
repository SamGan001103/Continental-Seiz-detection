"""Does the paper's Keras 2.0 / TF 1.4.0 give different numbers from ours?

The source paper states "our model is implemented in Python 3.6 with the use of
Keras 2.0 and Tensorflow 1.4.0" (§2.4.1). This project runs **Keras 2.2.5 /
TF 1.15**. That is a deviation from the published pipeline, and — like the MNE
one — it had never been measured.

The MNE deviation turned out to be numerically irrelevant (0.19.2 and 0.20.0
are bit-identical). This checks whether the same holds for the framework that
actually evaluates the network.

Method
------
The forward pass is isolated from everything else. Stage 1 computes the STFT
input tensors from real recordings **once**, using the current environment, and
saves them. Stage 2 loads those identical tensors, builds the model, loads the
released weights and predicts — and is run once per framework version. Nothing
but Keras/TF differs between the two runs.

    # 1. tensors, computed once
    python experiments/diag_tf_version.py --make-inputs --out inputs.npz

    # 2. once per version; the second prepends an isolated install
    python experiments/diag_tf_version.py --inputs inputs.npz --out a.npz
    python experiments/diag_tf_version.py --inputs inputs.npz --out b.npz \
        --lib-path /tmp/tf140

    # 3. compare
    python experiments/diag_tf_version.py --compare a.npz b.npz

Two framework versions cannot coexist in one process, hence the split.
"""
from __future__ import print_function

import argparse
import os
import sys


def _prepend(path):
    """Must run before any keras/tensorflow import, including transitive."""
    if path:
        sys.path.insert(0, os.path.abspath(path))


def make_inputs(out, n_windows, edf=None):
    """Compute STFT tensors from real recordings, once, in this environment."""
    import csv
    import numpy as np
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import mne
    mne.set_log_level('ERROR')
    from gui.io.edf import load_edf_19ch, CHANNELS_19
    from gui.io.infer import _calc_stft, SEGMENT_S
    from utils.preprocessing import detect_interupted_data, ica_arti_remove

    if edf is None:
        m = os.path.join(repo, 'artifacts', 'zuna_thesis', 'manifest.csv')
        with open(m) as f:
            rows = [r for r in csv.DictReader(f)
                    if (r.get('cohort') or '') == 'seizure']
        rows.sort(key=lambda r: float(r['duration_s']), reverse=True)
        edf = os.path.normpath(os.path.join(repo, rows[0]['edf']))

    data, fs, _ = load_edf_19ch(edf)
    fs = int(fs)
    wl = SEGMENT_S * fs
    tensors, t = [], 0
    while len(tensors) < n_windows and t * fs + wl <= data.shape[1]:
        seg = data[:, t * fs:t * fs + wl]
        t += 6
        if detect_interupted_data(seg.transpose(), fs):
            continue
        proc = ica_arti_remove(seg, fs, CHANNELS_19)
        if proc is None:
            continue
        tensors.append(np.expand_dims(_calc_stft(proc), -1)[0])
    x = np.asarray(tensors, dtype=np.float32)
    np.savez(out, x=x, edf=os.path.basename(edf))
    print('wrote {} tensors of shape {} from {}'.format(
        x.shape[0], x.shape[1:], os.path.basename(edf)))
    return 0


def predict(inputs, out):
    import numpy as np
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo not in sys.path:
        sys.path.append(repo)
    import keras
    import tensorflow as tf
    from gui.io.infer import _build_model

    print('Keras {} / TF {}'.format(keras.__version__, tf.__version__))
    print('  keras from: {}'.format(os.path.dirname(keras.__file__)[:70]))

    x = np.load(inputs)['x']
    model = _build_model()
    probs = np.asarray(
        [float(model.predict(x[i:i + 1], verbose=0)[0, 1])
         for i in range(x.shape[0])], dtype=np.float64)
    np.savez(out, probs=probs, keras=str(keras.__version__),
             tf=str(tf.__version__))
    print('wrote {} predictions'.format(probs.size))
    return 0


def compare(a_path, b_path):
    import numpy as np
    a, b = np.load(a_path), np.load(b_path)
    print('A: Keras {} / TF {}'.format(str(a['keras']), str(a['tf'])))
    print('B: Keras {} / TF {}'.format(str(b['keras']), str(b['tf'])))
    pa, pb = a['probs'], b['probs']
    n = min(pa.size, pb.size)
    d = np.abs(pa[:n] - pb[:n])
    print('windows compared : {}'.format(n))
    print('|difference| in p(seizure)')
    print('  max     : {:.3e}'.format(d.max()))
    print('  mean    : {:.3e}'.format(d.mean()))
    print('  exact   : {}/{}'.format(int((d == 0).sum()), n))
    print('  > 1e-6  : {}'.format(int((d > 1e-6).sum())))
    print('  > 0.01  : {}'.format(int((d > 0.01).sum())))
    flip = int(((pa[:n] >= 0.5) != (pb[:n] >= 0.5)).sum())
    print('  crossing the 0.5 threshold : {}'.format(flip))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib-path', default=None,
                    help='directory holding an alternative keras/tensorflow')
    ap.add_argument('--make-inputs', action='store_true')
    ap.add_argument('--inputs', default=None)
    ap.add_argument('--edf', default=None)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', default=None)
    ap.add_argument('--compare', nargs=2, default=None)
    args = ap.parse_args(argv)

    if args.compare:
        return compare(*args.compare)
    if args.make_inputs:
        return make_inputs(args.out, args.n, args.edf)
    _prepend(args.lib_path)
    return predict(args.inputs, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
