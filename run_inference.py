"""Inference-only runner for Continental-Seiz-detection 19-channel ConvLSTM.

Reads EDFs from sample_data/, applies the repo's preprocessing pipeline
(19-ch montage, resample to 250 Hz, 12s windows, bad-data skip, ICA
artifact removal, STFT), runs the pretrained model, thresholds, and
emits per-file event predictions compared against TUSZ .csv_bi refs.

Usage (from repo root):
    python run_inference.py --n 3         # first 3 files containing seizures
    python run_inference.py --file <edf>  # a single EDF path
"""
import argparse
import glob
import os
import sys
import time
import numpy as np

# pyst.read_edf_elec looks up params_common_electrodes.txt relative to cwd,
# so we cd into utils/ before any EDF reads.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import eval_config as cfg  # canonical operating point (threshold/step)  # noqa: E402

from gui.postprocess import events_from_probs  # noqa: E402
from utils.pyst import read_edf_elec
from utils.preprocessing import detect_interupted_data, ica_arti_remove
from models.deep_conv_lstm import ConvLstmNet
from scipy.signal import resample
import stft


def calc_stft(s_):
    """Replicates utils/ICA_load_data_elec.py calc_stft (kept standalone to
    avoid that module's `from pyst import ...` cwd-dependent import)."""
    s = s_.transpose()
    stft_data = stft.spectrogram(s, framelength=250, centered=False)
    if stft_data.ndim == 2:
        stft_data = np.expand_dims(stft_data, -1)
    stft_data = np.transpose(stft_data, (1, 2, 0))
    stft_data = np.abs(stft_data) + 1e-6
    stft_data = stft_data[:, :, 1:]
    stft_data = np.log10(stft_data)
    stft_data[stft_data <= 0] = 0
    stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                  stft_data.shape[1],
                                  stft_data.shape[2])
    return stft_data


CHS_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
          'T3', 'C3', 'Cz', 'C4', 'T4',
          'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
SEGMENT_S = cfg.SEGMENT_S
FS = cfg.FS
WINDOW = FS * SEGMENT_S  # 3000 samples
WEIGHTS = os.path.join(REPO_ROOT, 'convlstm_ICA_12_train.h5')


def find_sample_edfs():
    edfs = sorted(glob.glob(os.path.join(REPO_ROOT, 'sample_data', '**', '*.edf'),
                            recursive=True))
    return edfs


def ref_path(edf):
    return edf[:-4] + '.csv_bi'


def parse_ref(csv_bi_path):
    events = []
    if not os.path.exists(csv_bi_path):
        return events
    with open(csv_bi_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('channel'):
                continue
            parts = line.split(',')
            if len(parts) >= 5 and parts[3] == 'seiz':
                events.append((float(parts[1]), float(parts[2])))
    return events


def merge_events(per_sec, min_gap=1, min_len=1):
    """Convert per-second binary predictions to (start, stop) event list."""
    events = []
    in_ev = False
    start = 0
    for i, v in enumerate(per_sec):
        if v and not in_ev:
            in_ev = True
            start = i
        elif not v and in_ev:
            in_ev = False
            if i - start >= min_len:
                events.append((start, i))
    if in_ev and len(per_sec) - start >= min_len:
        events.append((start, len(per_sec)))
    # merge close events
    merged = []
    for s, e in events:
        if merged and s - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def any_overlap(pred, gts, tol=5):
    """Does predicted window (p0,p1) overlap any gt (g0,g1) within tolerance?"""
    p0, p1 = pred
    for g0, g1 in gts:
        if p1 + tol >= g0 and p0 - tol <= g1:
            return (g0, g1)
    return None


def run_file(model, edf_path, threshold=0.95, step_s=1, verbose=True):
    """Run inference on a single EDF. Returns dict with probs, events, refs."""
    t_start = time.time()
    fsamp, data = read_edf_elec(edf_path)
    if fsamp > FS:
        data = resample(data, int(data.shape[1] * FS / fsamp), axis=1)
    total_samples = data.shape[1]
    total_s = total_samples // FS
    n_windows = (total_s - SEGMENT_S) // step_s + 1
    if verbose:
        print('  duration: {}s, windows: {}'.format(total_s, n_windows))
    probs = np.zeros(n_windows, dtype=np.float32)
    skipped = 0
    for i in range(n_windows):
        s0 = i * step_s * FS
        seg = data[:, s0:s0 + WINDOW]
        if seg.shape[1] != WINDOW:
            break
        if detect_interupted_data(seg.transpose(), FS):
            probs[i] = 0
            skipped += 1
            continue
        ica_seg = ica_arti_remove(seg, FS, CHS_19)
        if ica_seg is None:
            probs[i] = 0
            skipped += 1
            continue
        stft_in = calc_stft(ica_seg)  # (1, 23, 19, 125)
        stft_in = np.expand_dims(stft_in, -1)  # (1, 23, 19, 125, 1)
        p = model.predict(stft_in, verbose=0)[0, 1]
        probs[i] = p
        if verbose and p >= threshold:
            print('    t={}s  p={:.3f}'.format(i * step_s, p))
    # Window index i starts at second i*step_s and covers SEGMENT_S seconds.
    # Building events straight off the index array (as this script used to do)
    # reports every time in window-index units, so with the canonical stride of
    # 6 s all printed event times came out 6x compressed and were then compared
    # against reference times in seconds. Share the one event builder instead,
    # so this CLI, the GUI and evaluate_baseline.py cannot drift apart again.
    starts = np.arange(n_windows, dtype=np.int32) * int(step_s)
    events = [(s, e) for s, e, _p in events_from_probs(
        starts, probs, threshold, SEGMENT_S, duration_s=float(total_s))]
    refs = parse_ref(ref_path(edf_path))
    dt = time.time() - t_start
    return {'edf': edf_path, 'probs': probs, 'events': events,
            'refs': refs, 'skipped': skipped, 'elapsed_s': dt,
            'n_windows': n_windows, 'step_s': int(step_s),
            'window_starts': starts}


def summarize(result):
    edf = os.path.basename(result['edf'])
    events, refs, probs = result['events'], result['refs'], result['probs']
    print('\n== {} =='.format(edf))
    print('  refs ({} seizures):'.format(len(refs)))
    for g0, g1 in refs:
        print('    {:>7.1f} - {:>7.1f}s  ({:.0f}s)'.format(g0, g1, g1 - g0))
    if len(probs):
        print('  prob stats: max={:.3f} p90={:.3f} mean={:.3f}'.format(
            float(np.max(probs)), float(np.percentile(probs, 90)), float(np.mean(probs))))
        step_s = result.get('step_s', 1)
        top = np.argsort(-probs)[:8]
        top_str = ', '.join('t{}s={:.2f}'.format(int(i) * step_s,
                                                 float(probs[i]))
                            for i in top)
        print('  top probs: {}'.format(top_str))
    print('  preds ({} events):'.format(len(events)))
    hits = 0
    for p0, p1 in events:
        match = any_overlap((p0, p1), refs)
        tag = '[HIT {:.1f}-{:.1f}]'.format(*match) if match else '[FP]'
        if match:
            hits += 1
        print('    {:>7.1f} - {:>7.1f}s  {}'.format(p0, p1, tag))
    if refs:
        covered = sum(1 for g in refs if any_overlap(g, events))
        print('  detected {}/{} reference seizures; {} false-positive events'
              .format(covered, len(refs), len(events) - hits))
    print('  skipped windows: {}/{};  elapsed: {:.1f}s'
          .format(result['skipped'], result['n_windows'], result['elapsed_s']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default=None)
    ap.add_argument('--n', type=int, default=3,
                    help='number of seizure-containing files to run')
    ap.add_argument('--threshold', type=float, default=cfg.THRESHOLD,
                    help='detection threshold (default: canonical eval_config)')
    ap.add_argument('--step', type=int, default=cfg.STEP_S,
                    help='stride between windows in seconds (default: canonical)')
    args = ap.parse_args()

    # Resolve --file against the invocation directory BEFORE the chdir below,
    # otherwise a relative path is silently reinterpreted under utils/ and the
    # run fails with a confusing "failed to open" on a path the user never gave.
    explicit = os.path.abspath(args.file) if args.file else None

    os.chdir(os.path.join(REPO_ROOT, 'utils'))  # params file resolution

    print('Building model and loading weights...')
    model = ConvLstmNet(epochs=1).setup((-1, 2 * SEGMENT_S - 1, 19, 125, 1))
    model.model.load_weights(WEIGHTS)
    print('OK\n')

    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit('No such EDF: {}'.format(explicit))
        targets = [explicit]
    else:
        all_edfs = find_sample_edfs()
        targets = []
        for e in all_edfs:
            refs = parse_ref(ref_path(e))
            if refs:
                targets.append(e)
            if len(targets) >= args.n:
                break

    results = []
    for edf in targets:
        print('>> {}'.format(edf))
        try:
            r = run_file(model.model, edf, threshold=args.threshold,
                         step_s=args.step, verbose=False)
            summarize(r)
            results.append(r)
        except Exception as ex:
            print('  ERROR:', ex)

    print('\n==== overall ====')
    tot_ref, tot_hit, tot_fp = 0, 0, 0
    for r in results:
        tot_ref += len(r['refs'])
        tot_hit += sum(1 for g in r['refs'] if any_overlap(g, r['events']))
        tot_fp += sum(1 for e in r['events'] if not any_overlap(e, r['refs']))
    print('ran {} files; detected {}/{} reference seizures; {} false-positive events'
          .format(len(results), tot_hit, tot_ref, tot_fp))


if __name__ == '__main__':
    main()
