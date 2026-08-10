"""Does the paper's MNE v0.20 give different numbers from the pinned 0.19.2?

The source paper states "our artifact removal is implemented in Python 3.6 with
the use of library MNE v0.20". This project pins **0.19.2**, which is a
deviation from the published pipeline and — more importantly — possibly a
deviation from whatever generated the released weights' training features.

That deviation had never been measured. A separate, much larger figure exists
in `docs/deployment_roadmap.md` — probabilities moving by up to 0.90 — but that
was measured against **MNE 1.12**, twelve minor versions away, and does not
license any claim about 0.20.

    # 1. install 0.20 somewhere isolated, without touching seiz36
    python -m pip install --no-deps --target /tmp/mne020 mne==0.20

    # 2. run once per version; the second prepends the isolated copy
    python experiments/diag_mne_version.py --out a.npz
    python experiments/diag_mne_version.py --out b.npz --mne-path /tmp/mne020

    # 3. compare
    python experiments/diag_mne_version.py --compare a.npz b.npz

Two MNE versions cannot coexist in one process, hence the two-pass design.
"""
from __future__ import print_function

import argparse
import os
import sys


def _prepend(path):
    """Must happen before any mne import, including transitive ones."""
    if path:
        sys.path.insert(0, os.path.abspath(path))


def run(edf, n_windows, out, seed_note):
    import numpy as np
    import mne
    mne.set_log_level('ERROR')

    from gui.io.edf import load_edf_19ch, CHANNELS_19
    from gui.io.infer import _calc_stft, _build_model, SEGMENT_S
    from utils.preprocessing import detect_interupted_data, ica_arti_remove

    print('MNE version in use : {}'.format(mne.__version__))
    print('MNE loaded from    : {}'.format(os.path.dirname(mne.__file__)))

    data, fs, _dur = load_edf_19ch(edf)
    fs = int(fs)
    wl = SEGMENT_S * fs
    model = _build_model()

    starts, probs, states = [], [], []
    t = 0
    while len(starts) < n_windows and (t + SEGMENT_S) * fs <= data.shape[1]:
        seg = data[:, t * fs:t * fs + wl]
        if seg.shape[1] != wl:
            break
        if detect_interupted_data(seg.transpose(), fs):
            starts.append(t)
            probs.append(0.0)
            states.append(1)
            t += 6
            continue
        proc = ica_arti_remove(seg, fs, CHANNELS_19)
        if proc is None:
            starts.append(t)
            probs.append(0.0)
            states.append(2)
            t += 6
            continue
        x = np.expand_dims(_calc_stft(proc), -1)
        starts.append(t)
        probs.append(float(model.predict(x, verbose=0)[0, 1]))
        states.append(0)
        t += 6

    np.savez(out, starts=np.array(starts), probs=np.array(probs),
             states=np.array(states), mne_version=str(mne.__version__),
             note=str(seed_note))
    print('wrote {} ({} windows, {} scored)'.format(
        out, len(starts), int((np.array(states) == 0).sum())))
    return 0


def compare(a_path, b_path):
    import numpy as np
    a, b = np.load(a_path), np.load(b_path)
    va = str(a['mne_version'])
    vb = str(b['mne_version'])
    print('A: MNE {}   B: MNE {}'.format(va, vb))

    sa, sb = a['starts'], b['starts']
    n = min(len(sa), len(sb))
    if not np.array_equal(sa[:n], sb[:n]):
        print('window grids differ — not comparable')
        return 1
    pa, pb = a['probs'][:n], b['probs'][:n]
    ka = (a['states'][:n] == 0) & (b['states'][:n] == 0)

    print('windows compared      : {} ({} scored in both)'.format(n,
                                                                  int(ka.sum())))
    if not ka.any():
        print('nothing scored in both')
        return 1
    d = np.abs(pa[ka] - pb[ka])
    print('|difference| in p(seizure)')
    print('  max    : {:.6f}'.format(d.max()))
    print('  mean   : {:.6f}'.format(d.mean()))
    print('  median : {:.6f}'.format(np.median(d)))
    print('  exact matches : {}/{}'.format(int((d == 0).sum()), d.size))
    print('  moved > 0.01  : {}'.format(int((d > 0.01).sum())))
    print('  moved > 0.10  : {}'.format(int((d > 0.10).sum())))

    flip = int(((pa[ka] >= 0.5) != (pb[ka] >= 0.5)).sum())
    print('  windows crossing the 0.5 threshold : {}'.format(flip))
    print()
    # The pipeline is not bit-reproducible even at a FIXED version
    # (docs/RESULTS.md §9 measured 0.107 between two runs of the same code), so
    # a version difference only means something if it exceeds that floor.
    print('Compare against the same-version re-run floor of 0.107 recorded in')
    print('RESULTS.md §9. A max difference at or below that is indistinguishable')
    print('from ICA\'s own non-determinism and licenses no claim either way.')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mne-path', default=None,
                    help='directory holding an alternative mne package')
    ap.add_argument('--edf', default=None)
    ap.add_argument('--n', type=int, default=25)
    ap.add_argument('--out', default=None)
    ap.add_argument('--compare', nargs=2, default=None)
    args = ap.parse_args(argv)

    if args.compare:
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        return compare(*args.compare)

    _prepend(args.mne_path)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo not in sys.path:
        sys.path.append(repo)

    edf = args.edf
    if edf is None:
        import csv
        m = os.path.join(repo, 'artifacts', 'zuna_thesis', 'manifest.csv')
        with open(m) as f:
            rows = [r for r in csv.DictReader(f)
                    if (r.get('cohort') or '') == 'seizure']
        rows.sort(key=lambda r: float(r['duration_s']), reverse=True)
        edf = os.path.normpath(os.path.join(repo, rows[0]['edf']))
    print('EDF: {}'.format(os.path.basename(edf)))
    return run(edf, args.n, args.out, args.mne_path or 'default')


if __name__ == '__main__':
    raise SystemExit(main())
