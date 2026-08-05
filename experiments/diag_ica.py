"""Diagnostic: does ICA help or hurt, per file, at the window level?

For each seizure file in the manifest, compute window-level ROC-AUC with the
existing ICA-on cache vs a freshly computed ICA-off run, to test whether the
low-AUC outliers are caused by ICA removing ictal activity (over-aggressive
EOG rejection) rather than a genuine model failure.

Run in seiz36 (needs TF for the ICA-off pass).
"""
from __future__ import print_function
import csv
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sklearn.metrics import roc_auc_score
from gui.io.cache import cache_path_for, load_probability_file
from gui.io.infer import compute_probs, _build_model
from experiments.compare_zuna import read_reference_events
from experiments.evaluate_baseline import window_labels

MANIFEST = os.path.join(REPO, 'artifacts', 'zuna_thesis', 'manifest.csv')


def auc_for(starts, probs, refs):
    labels = window_labels(starts, refs)
    keep = probs != 0.0
    lab, sc = labels[keep], probs[keep]
    n_pos = int(lab.sum())
    if n_pos == 0 or n_pos == lab.size:
        return None, n_pos, int((~keep).sum()), lab, sc
    return float(roc_auc_score(lab, sc)), n_pos, int((~keep).sum()), lab, sc


def main():
    rows = [r for r in csv.DictReader(open(MANIFEST))
            if (r.get('cohort') or '') == 'seizure']
    print('Building model...')
    model = _build_model()

    pool = {'on': ([], []), 'off': ([], [])}
    print('\n{:<22} {:>5} {:>8} {:>8} {:>10} {:>10}'.format(
        'stem', 'npos', 'AUC_on', 'AUC_off', 'zero_on', 'zero_off'))
    print('-' * 70)
    for r in rows:
        edf = os.path.normpath(os.path.join(REPO, r['edf']))
        stem = os.path.splitext(os.path.basename(edf))[0]
        refs = read_reference_events(os.path.splitext(edf)[0] + '.csv_bi')

        cache = load_probability_file(cache_path_for(edf))
        if cache is None:
            print('{:<22} no ICA-on cache, skipping'.format(stem))
            continue
        s_on = np.asarray(cache['window_starts'])
        p_on = np.asarray(cache['probs'])

        s_off, p_off = compute_probs(edf, step_s=6, use_ica=False, model=model)

        auc_on, npos, z_on, lab_on, sc_on = auc_for(s_on, p_on, refs)
        auc_off, _, z_off, lab_off, sc_off = auc_for(s_off, p_off, refs)

        pool['on'][0].append(lab_on); pool['on'][1].append(sc_on)
        pool['off'][0].append(lab_off); pool['off'][1].append(sc_off)

        def f(x):
            return 'n/a' if x is None else '{:.3f}'.format(x)
        print('{:<22} {:>5} {:>8} {:>8} {:>10} {:>10}'.format(
            stem, npos, f(auc_on), f(auc_off), z_on, z_off))

    def pooled(key):
        L = np.concatenate(pool[key][0]); S = np.concatenate(pool[key][1])
        return roc_auc_score(L, S) if 0 < L.sum() < L.size else None
    print('-' * 70)
    print('POOLED window AUC   ICA-on: {:.4f}   ICA-off: {:.4f}'.format(
        pooled('on'), pooled('off')))


if __name__ == '__main__':
    main()
