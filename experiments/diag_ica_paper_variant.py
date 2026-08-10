"""What do the deviations from the paper's ICA description actually cost?

`docs/ica_implementation_review.md` §2b lists four places where
`utils/preprocessing.ica_arti_remove` departs from the published method. Three
are testable by re-running inference with the deviation removed:

  A. **top-1 vs all** — the paper says "we remove **those** independent
     sources" (every source correlated with Fp1/Fp2). The code appends only the
     single highest-scoring component per channel, discarding roughly half of
     what it flagged.
  B. **the 0.1 Hz high-pass** — the paper mentions no filter, and MNE has to
     design an 8251-sample filter for a 3000-sample window to honour it.
  C. **the raw pass-through** — when no EOG component is found the code returns
     the *unfiltered* input, so those windows are preprocessed differently from
     their neighbours.

This script does **not** modify the production function. `ica_arti_remove` is
what generated the training features, so inference must stay bit-faithful to
it; the variant below is a parameterised copy used only for measurement. The
output is a thesis result — "deviation X is worth Y AUC" — not a patch.

    python experiments/diag_ica_paper_variant.py --variant paper
    python experiments/diag_ica_paper_variant.py --variant all_components

Needs TensorFlow and MNE (seiz36).
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mne                                                    # noqa: E402
from mne.preprocessing import ICA                             # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402

from utils.preprocessing import create_mne_raw, detect_interupted_data  # noqa: E402
from gui.io.cache import cache_path_for, load_probability_file  # noqa: E402
from gui.io.edf import load_edf_19ch, CHANNELS_19, TARGET_FS  # noqa: E402
from gui.io.infer import _calc_stft, _build_model, SEGMENT_S  # noqa: E402
from experiments.compare_zuna import read_reference_events    # noqa: E402
from experiments.evaluate_baseline import window_labels       # noqa: E402

MANIFEST = os.path.join(REPO, 'artifacts', 'zuna_thesis', 'manifest.csv')

VARIANTS = {
    # as trained and as deployed — the control, must match the cache
    'as_trained':     dict(remove_all=False, highpass=0.1,  passthrough_raw=True),
    # A only: remove every flagged component, as the paper's wording says
    'all_components': dict(remove_all=True,  highpass=0.1,  passthrough_raw=True),
    # B only: drop the unrealisable filter the paper never mentions
    'no_highpass':    dict(remove_all=False, highpass=None, passthrough_raw=True),
    # C only: keep preprocessing consistent when no EOG component is found
    'no_passthrough': dict(remove_all=False, highpass=0.1,  passthrough_raw=False),
    # everything the paper actually describes, and nothing it does not
    'paper':          dict(remove_all=True,  highpass=None, passthrough_raw=False),
}


def ica_variant(data, sfreq, chs, remove_all, highpass, passthrough_raw):
    """A parameterised copy of ica_arti_remove. Never used for deployment.

    Kept deliberately close to the original so the diff is the variable under
    test and nothing else.
    """
    raw = create_mne_raw(data, sfreq, chs)
    work = raw.copy()
    work.load_data()
    if highpass is not None:
        work.filter(l_freq=highpass, h_freq=None, verbose=False)

    ica = ICA(n_components=19, random_state=13)
    try:
        ica.fit(work, verbose=False)
    except Exception:
        return None

    ica.exclude = []
    e1, _ = ica.find_bads_eog(work, threshold=2.0, ch_name='Fp1', verbose=False)
    e2, _ = ica.find_bads_eog(work, threshold=2.0, ch_name='Fp2', verbose=False)
    if remove_all:
        ica.exclude = sorted(set(list(e1) + list(e2)))
    else:
        if len(e1) > 0:
            ica.exclude.append(e1[0])
        if len(e2) > 0:
            ica.exclude.append(e2[0])

    if len(ica.exclude) > 0:
        rec = work.copy()
        rec.load_data()
        # No verbose= here: MNE 0.19.2's ICA.apply does not take it, and the
        # production function does not pass it either. Keeping the call
        # identical is the point of this copy.
        ica.apply(rec)
        return rec.get_data() * 1e6, len(set(list(e1) + list(e2))), len(ica.exclude)
    if passthrough_raw:
        # the original behaviour: the ORIGINAL, unfiltered input
        return data, len(set(list(e1) + list(e2))), 0
    # consistent behaviour: same preprocessing as every other window
    return work.get_data() * 1e6, len(set(list(e1) + list(e2))), 0


def probs_for(edf, model, cfgv, step_s=6):
    data, fs, _dur = load_edf_19ch(edf)
    fs = int(fs)
    wl = SEGMENT_S * fs
    total = int(data.shape[1] / fs)
    starts = list(range(0, total - SEGMENT_S + 1, step_s))
    out, keep = [], []
    flagged = removed = 0
    for t in starts:
        seg = data[:, t * fs:t * fs + wl]
        if seg.shape[1] != wl:
            break
        if detect_interupted_data(seg.transpose(), fs):
            out.append(0.0)
            keep.append(False)
            continue
        got = ica_variant(seg, fs, CHANNELS_19, **cfgv)
        if got is None:
            out.append(0.0)
            keep.append(False)
            continue
        proc, nf, nr = got
        flagged += nf
        removed += nr
        x = np.expand_dims(_calc_stft(proc), -1)
        out.append(float(model.predict(x, verbose=0)[0, 1]))
        keep.append(True)
    n = len(out)
    return (np.array(starts[:n], dtype=float), np.array(out, dtype=np.float32),
            np.array(keep, dtype=bool), flagged, removed)


def pooled(labels, scores):
    y = np.concatenate(labels)
    s = np.concatenate(scores)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else float('nan')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='paper', choices=sorted(VARIANTS))
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args(argv)
    cfgv = VARIANTS[args.variant]

    mne.set_log_level('ERROR')
    rows = [r for r in csv.DictReader(open(MANIFEST))
            if (r.get('cohort') or '') == 'seizure']
    print('variant: {}  {}'.format(args.variant, cfgv))
    print('Building model...')
    model = _build_model()

    labs, base_s, var_s, stems = [], [], [], []
    tot_flagged = tot_removed = 0
    print('\n{:<22} {:>5} {:>10} {:>10}'.format(
        'stem', 'npos', 'AUC_trained', 'AUC_var'))
    print('-' * 52)
    for r in rows:
        edf = os.path.normpath(os.path.join(REPO, r['edf']))
        stem = os.path.splitext(os.path.basename(edf))[0]
        cache = load_probability_file(cache_path_for(edf))
        if cache is None:
            continue
        refs = read_reference_events(os.path.splitext(edf)[0] + '.csv_bi')

        s_v, p_v, k_v, nf, nr = probs_for(edf, model, cfgv)
        tot_flagged += nf
        tot_removed += nr

        s_b = np.asarray(cache['window_starts'])
        p_b = np.asarray(cache['probs'])
        k_b = np.asarray(cache['skip_code']) == 0

        b = {int(s): i for i, s in enumerate(s_b) if k_b[i]}
        v = {int(s): i for i, s in enumerate(s_v) if k_v[i]}
        shared = sorted(set(b) & set(v))
        if not shared:
            continue
        lab = window_labels(np.array(shared, dtype=float), refs)
        labs.append(lab)
        base_s.append(p_b[[b[s] for s in shared]])
        var_s.append(p_v[[v[s] for s in shared]])
        stems.append(stem)

        def f(y, s):
            return ('n/a' if not (0 < y.sum() < y.size)
                    else '{:.3f}'.format(roc_auc_score(y, s)))
        print('{:<22} {:>5} {:>10} {:>10}'.format(
            stem, int(lab.sum()), f(lab, base_s[-1]), f(lab, var_s[-1])))

    if not labs:
        print('nothing comparable')
        return 1

    a_b = pooled(labs, base_s)
    a_v = pooled(labs, var_s)
    delta = a_v - a_b
    print('-' * 52)
    print('components flagged {}, removed {}'.format(tot_flagged, tot_removed))
    print('pooled AUC  as-trained: {:.4f}   {}: {:.4f}'.format(
        a_b, args.variant, a_v))
    print('delta ({} - as_trained): {:+.4f}'.format(args.variant, delta))

    n = len(labs)
    rng = np.random.RandomState(args.seed)
    d = []
    for _ in range(args.boot):
        idx = rng.randint(0, n, n)
        L = [labs[i] for i in idx]
        if sum(int(l.sum()) for l in L) == 0:
            continue
        x = pooled(L, [base_s[i] for i in idx])
        y = pooled(L, [var_s[i] for i in idx])
        if not (np.isnan(x) or np.isnan(y)):
            d.append(y - x)
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print('\ncluster bootstrap by recording over {} files'.format(n))
    print('  DELTA 95 % CI : [{:+.4f}, {:+.4f}]'.format(lo, hi))
    print('  interval {} zero'.format('CROSSES' if lo < 0 < hi else 'excludes'))

    dest = os.path.join(REPO, 'artifacts', 'zuna_thesis',
                        'ica_variant_{}.json'.format(args.variant))
    with open(dest, 'w') as f:
        json.dump({'variant': args.variant, 'config': cfgv, 'files': stems,
                   'auc_as_trained': a_b, 'auc_variant': a_v, 'delta': delta,
                   'delta_ci95': [float(lo), float(hi)],
                   'components_flagged': tot_flagged,
                   'components_removed': tot_removed}, f, indent=2,
                  sort_keys=True)
        f.write('\n')
    print('wrote {}'.format(os.path.relpath(dest, REPO)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
