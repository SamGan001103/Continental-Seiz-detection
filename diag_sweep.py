"""Sweep probabilities across all seizure windows in N files.

For each known-seizure reference interval in each file, we compute the
model's max probability over the 12-s windows that overlap the interval.
This tells us the model's best chance on each true seizure event, without
the hit/miss threshold getting in the way.
"""
import os
import sys
import glob
import time
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from utils.pyst import read_edf_elec
from utils.preprocessing import ica_arti_remove
from models.deep_conv_lstm import ConvLstmNet
from scipy.signal import resample
import stft as stftpkg


CHS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
       'T3', 'C3', 'Cz', 'C4', 'T4',
       'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
FS = 250
WIN_S = 12
WIN_N = FS * WIN_S


def calc_stft(s_):
    s = s_.transpose()
    d = stftpkg.spectrogram(s, framelength=250, centered=False)
    if d.ndim == 2:
        d = np.expand_dims(d, -1)
    d = np.transpose(d, (1, 2, 0))
    d = np.abs(d) + 1e-6
    d = d[:, :, 1:]
    d = np.log10(d)
    d[d <= 0] = 0
    return d.reshape(-1, d.shape[0], d.shape[1], d.shape[2])


def parse_ref(p):
    events = []
    if not os.path.exists(p):
        return events
    for line in open(p):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('channel'):
            continue
        parts = line.split(',')
        if len(parts) >= 5 and parts[3] == 'seiz':
            events.append((float(parts[1]), float(parts[2])))
    return events


def main(n_files=8, mode='ica'):
    os.chdir(os.path.join(REPO, 'utils'))
    model = ConvLstmNet(epochs=1).setup((-1, 23, 19, 125, 1))
    model.model.load_weights(os.path.join(REPO, 'convlstm_ICA_12_train.h5'))
    os.chdir(REPO)

    edfs = sorted(glob.glob(os.path.join(REPO, 'sample_data', '**', '*.edf'),
                            recursive=True))
    seiz_edfs = [e for e in edfs if parse_ref(e[:-4] + '.csv_bi')]
    seiz_edfs = seiz_edfs[:n_files]

    print('mode =', mode, '|', n_files, 'files')
    all_ref_maxp, all_bg_maxp = [], []
    per_file_rows = []

    for edf in seiz_edfs:
        fname = os.path.basename(edf)
        refs = parse_ref(edf[:-4] + '.csv_bi')
        os.chdir(os.path.join(REPO, 'utils'))
        fs, data = read_edf_elec(edf)
        os.chdir(REPO)
        if fs > FS:
            data = resample(data, int(data.shape[1] * FS / fs), axis=1)
        total_s = data.shape[1] // FS

        # For each 12-s window starting every 6s (50% overlap, fast enough).
        starts = list(range(0, total_s - WIN_S + 1, 6))
        probs = []
        for t in starts:
            seg = data[:, t * FS:t * FS + WIN_N]
            if mode == 'ica':
                proc = ica_arti_remove(seg, FS, CHS)
                if proc is None:
                    proc = seg
            else:
                proc = seg
            x = calc_stft(proc)
            x = np.expand_dims(x, -1)
            p = float(model.model.predict(x, verbose=0)[0, 1])
            probs.append((t, p))
        probs = np.array(probs, dtype=[('t', int), ('p', float)])

        ref_max = []
        for g0, g1 in refs:
            mask = (probs['t'] + WIN_S >= g0) & (probs['t'] <= g1)
            if mask.any():
                ref_max.append(float(probs['p'][mask].max()))
        all_ref_maxp.extend(ref_max)

        bg_mask = np.ones(len(probs), dtype=bool)
        for g0, g1 in refs:
            bg_mask &= ~((probs['t'] + WIN_S >= g0) & (probs['t'] <= g1))
        bg_probs = probs['p'][bg_mask]
        all_bg_maxp.append(float(bg_probs.max()) if len(bg_probs) else 0.0)

        per_file_rows.append((fname, len(refs), ref_max,
                              float(bg_probs.max()) if len(bg_probs) else 0.0,
                              float(bg_probs.mean()) if len(bg_probs) else 0.0))
        print('  {}  refs={}  max-prob-per-seiz={}  bg-max={:.2f}  bg-mean={:.3f}'
              .format(fname, len(refs),
                      ['%.2f' % v for v in ref_max],
                      float(bg_probs.max()) if len(bg_probs) else 0.0,
                      float(bg_probs.mean()) if len(bg_probs) else 0.0))

    print('\n== aggregate ==')
    arr = np.array(all_ref_maxp)
    print('seizures scored:', len(arr))
    if len(arr):
        for thr in [0.3, 0.5, 0.7, 0.9, 0.95]:
            print('  recall @ thr {:.2f}: {}/{} = {:.1%}'.format(
                thr, int((arr >= thr).sum()), len(arr),
                (arr >= thr).mean()))
        print('  per-seizure max-prob: median={:.3f} mean={:.3f} max={:.3f}'
              .format(float(np.median(arr)), float(arr.mean()), float(arr.max())))
    bg = np.array(all_bg_maxp)
    print('  bg-max mean across files: {:.3f}'.format(float(bg.mean())))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--mode', choices=['ica', 'raw'], default='ica')
    args = ap.parse_args()
    main(n_files=args.n, mode=args.mode)
