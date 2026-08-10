"""Generate the thesis figures from committed artefacts.

One command, so every figure regenerates from the same numbers as docs/RESULTS.md
and none is hand-edited. Writes PNG (300 dpi) and PDF (vector, for LaTeX).

    python experiments/thesis_figures.py --out docs/figures

Figures:
  4.2  ROC curve, both protocols, with the published 0.84 marked
  4.3  Threshold sweep: sensitivity and false alarms per 24 h
  4.4  Peak model score per reference seizure — the separation failure
  4.5  Decision-stage ablation

Needs matplotlib in seiz36:  python -m pip install matplotlib==3.0.3
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import eval_config as cfg  # noqa: E402
from gui.io.cache import cache_path_for, load_probability_file  # noqa: E402
from experiments.evaluate_baseline import iter_manifest_rows, load_file  # noqa: E402

J = os.path.join(REPO, 'artifacts', 'zuna_thesis', 'baseline_eval')
PUBLISHED_AUC = 0.84

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'figure.dpi': 110,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def _save(fig, out_dir, name):
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, '{}.{}'.format(name, ext))
        fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  wrote {}.png / .pdf'.format(name))


# --------------------------------------------------------------------------
def _pooled_scores(manifest):
    """(labels, scores) per protocol, recomputed from the caches.

    The curves are derived here rather than read from the JSON because
    replicate_paper_auc.py stores summary statistics, not the curve arrays.
    Deriving them keeps the figure and the reported AUC from ever diverging.
    """
    from experiments.replicate_paper_auc import paper_protocol_labels
    out = {'paper': ([], []), 'project': ([], [])}
    for edf, coh in iter_manifest_rows(os.path.abspath(manifest)):
        r = load_file(edf, coh)
        if r is None or not r['has_reference']:
            continue
        lab, keep = paper_protocol_labels(r['starts'], r['refs'])
        m = keep & r['keep']
        if m.any():
            out['paper'][0].append(lab[m])
            out['paper'][1].append(r['probs'][m])
        m2 = r['keep']
        if m2.any():
            out['project'][0].append(r['labels'][m2])
            out['project'][1].append(r['probs'][m2])
    return {k: (np.concatenate(a), np.concatenate(b))
            for k, (a, b) in out.items() if a}


def fig_roc(out_dir, manifest):
    """Fig 4.2 — ROC under both protocols, published value marked."""
    from sklearn.metrics import roc_curve, roc_auc_score
    with open(os.path.join(J, 'paper_protocol_auc_full.json')) as f:
        d = json.load(f)
    pooled = _pooled_scores(manifest)

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for key, jkey, label, colour in (
            ('paper', 'paper_pure_windows',
             'Paper protocol (windows fully inside)', '#1f6fb4'),
            ('project', 'project_any_overlap',
             'Project protocol (any overlap)', '#b02020')):
        y, s = pooled[key]
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        b = ((d['pooled'].get(jkey) or {}).get('bootstrap_by_patient')
             or (d['pooled'].get(jkey) or {}).get('bootstrap') or {})
        ax.plot(fpr, tpr, color=colour, lw=1.7,
                label='{}\nAUC {:.2f}  95% CI [{:.2f}, {:.2f}]'.format(
                    label, auc, b.get('ci_lo', float('nan')),
                    b.get('ci_hi', float('nan'))))
    ax.plot([0, 1], [0, 1], color='#999999', ls='--', lw=0.9)
    # The published 0.84 is an AREA, not a point on a curve, so it must not be
    # drawn as a line on these axes — that would imply it is a true-positive
    # rate. It belongs in the annotation.
    ax.text(0.42, 0.30,
            'Source paper reports\nAUC {:.2f} (TUH v1.5.1 dev).\n'
            'It lies inside both intervals.'.format(PUBLISHED_AUC),
            fontsize=8, color='#2c8a3d', ha='left', va='top')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Window-level discrimination, 206 recordings / 28 patients')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', frameon=False)
    _save(fig, out_dir, 'fig4_2_roc')


def fig_threshold_sweep(out_dir):
    """Fig 4.3 — the sensitivity / false-alarm trade-off."""
    with open(os.path.join(J, 'full_scorable.json')) as f:
        d = json.load(f)
    sweep = sorted(d['event_sweep'], key=lambda s: s['threshold'])
    thr = [s['threshold'] for s in sweep]
    sens = [100 * s['pooled']['sensitivity'] for s in sweep]
    fa = [s['pooled']['false_alarms_per_24h'] for s in sweep]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(fa, sens, 'o-', color='#1f6fb4', lw=1.6, ms=5)
    for x, y, t in zip(fa, sens, thr):
        ax.annotate('{:g}'.format(t), (x, y), textcoords='offset points',
                    xytext=(6, -9), fontsize=8, color='#444444')
    ax.set_xlabel('False alarms per 24 h')
    ax.set_ylabel('Event sensitivity (%)')
    ax.set_title('Operating points (labels are detection thresholds)\n'
                 '206 recordings, 27.8 h, 85 seizures')
    ax.grid(alpha=0.25)
    _save(fig, out_dir, 'fig4_3_threshold_sweep')


def fig_separation(out_dir, manifest):
    """Fig 4.4 — peak model score inside each reference seizure.

    The most important new figure: it shows that ~22 % of seizures produce no
    model response at all, so their misses are a SEPARATION failure that no
    threshold can recover — not a threshold-tuning problem.
    """
    peaks = []
    for edf, coh in iter_manifest_rows(os.path.abspath(manifest)):
        r = load_file(edf, coh)
        if r is None or not r['has_reference'] or not r['refs']:
            continue
        c = load_probability_file(cache_path_for(edf))
        if c is None:
            continue
        s = np.asarray(c['window_starts'])
        p = np.asarray(c['probs'])
        for ref in r['refs']:
            m = (s + cfg.SEGMENT_S > ref['start']) & (s < ref['stop'])
            peaks.append(float(p[m].max()) if m.any() else 0.0)
    peaks = np.array(peaks)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bins = np.logspace(-5, 0, 26)
    ax.hist(np.clip(peaks, 1e-5, 1.0), bins=bins, color='#1f6fb4',
            edgecolor='white', lw=0.5)
    ax.set_xscale('log')
    ax.axvline(0.5, color='#b02020', ls='--', lw=1.3)
    ax.axvline(0.01, color='#e08a00', ls=':', lw=1.3)
    n_lost = int((peaks < 0.01).sum())
    ax.annotate('threshold 0.5', (0.5, ax.get_ylim()[1] * 0.92),
                color='#b02020', fontsize=8, ha='right', rotation=90,
                va='top')
    ax.annotate('0.01 — below this no usable\nthreshold recovers the seizure',
                (0.01, ax.get_ylim()[1] * 0.55), color='#a06000', fontsize=8,
                ha='left')
    ax.set_xlabel('Peak model score reached inside the seizure (log scale)')
    ax.set_ylabel('Reference seizures')
    ax.set_title('Why misses are a separation problem, not a threshold problem\n'
                 '{} of {} seizures ({:.0f} %) peak below 0.01'.format(
                     n_lost, peaks.size, 100.0 * n_lost / peaks.size))
    _save(fig, out_dir, 'fig4_4_separation')
    return peaks


def fig_ablation(out_dir):
    """Fig 4.5 — what the source method's decision stage is worth."""
    with open(os.path.join(J, 'postproc_ablation_full.json')) as f:
        d = json.load(f)
    order = [('raw', 'raw windows'), ('shape', '+ event shaping'),
             ('avg', '+ per-second averaging'), ('avg+shape', 'source method')]
    sens, fa, labels = [], [], []
    for key, lbl in order:
        sw = d['configurations'].get(key)
        if not sw:
            continue
        row = [s['pooled'] for s in sw if abs(s['threshold'] - 0.5) < 1e-9][0]
        sens.append(100 * row['sensitivity'])
        fa.append(row['false_alarms_per_24h'])
        labels.append(lbl)

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(5.6, 3.4))
    ax1.bar(x - 0.19, fa, 0.38, color='#b02020', label='FA / 24 h')
    ax1.set_ylabel('False alarms per 24 h', color='#b02020')
    ax1.tick_params(axis='y', colors='#b02020')
    ax2 = ax1.twinx()
    ax2.bar(x + 0.19, sens, 0.38, color='#1f6fb4', label='sensitivity')
    ax2.set_ylabel('Event sensitivity (%)', color='#1f6fb4')
    ax2.tick_params(axis='y', colors='#1f6fb4')
    ax2.spines['right'].set_visible(True)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha='right')
    ax1.set_title('The decision stage, not the model, cuts false alarms 2.1x\n'
                  '(threshold 0.5)')
    _save(fig, out_dir, 'fig4_5_ablation')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest',
                    default='artifacts/zuna_thesis/manifest_full.csv')
    ap.add_argument('--out', default='docs/figures')
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('writing figures to {}'.format(out_dir))

    fig_roc(out_dir, args.manifest)
    fig_threshold_sweep(out_dir)
    peaks = fig_separation(out_dir, args.manifest)
    fig_ablation(out_dir)

    print('\nseparation summary for the caption:')
    for lo, hi, lab in [(0, 0.001, 'no response at all'),
                        (0.001, 0.01, 'unreachable by any usable threshold'),
                        (0.01, 0.5, 'reachable by lowering the threshold'),
                        (0.5, 1.01, 'detected at the default threshold')]:
        n = int(((peaks >= lo) & (peaks < hi)).sum())
        print('  {:<42} {:2d}  ({:4.1f} %)'.format(
            lab, n, 100.0 * n / peaks.size))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
