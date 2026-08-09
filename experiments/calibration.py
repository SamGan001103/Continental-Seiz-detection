"""Is the detector's score a calibrated probability? (No.) Can it be made one? (Mostly.)

The literature review treats calibration as a *trust requirement*: modern networks are
systematically over-confident (Guo et al., ICML 2017), and an uncalibrated strip presented as a
probability misleads exactly the reviewer it is meant to help. The supervisor flagged the same
thing. This script measures it and fits the standard post-hoc corrections.

Post-hoc calibration does not touch the weights, so it is compatible with the inference-only
policy. It reads cached probabilities: no TensorFlow, no re-inference.

    python experiments/calibration.py --manifest artifacts/zuna_thesis/manifest_full.csv \
        --out artifacts/zuna_thesis/baseline_eval/calibration.json \
        --svg artifacts/zuna_thesis/baseline_eval/reliability.svg

------------------------------------------------------------------------------------
GROUPING — the thing that is easy to get wrong, and that inverted this analysis once
------------------------------------------------------------------------------------
The 203 scorable recordings come from only **28 patients**, and only 14 of those contribute a
single positive window. Splitting cross-validation folds by *recording* therefore puts other
recordings of the same patient in the training fold for ~95 % of test windows, and a free
monotone map (isotonic) exploits that. Measured, same data, only the group key changed:

                    by recording   by patient   leave-one-patient-out
      temperature       0.0694       0.0694            0.0708
      platt             0.0106       0.0109            0.0107
      isotonic          0.0085       0.0149            0.0168     <- flatters itself

`--group-by patient` is the default for that reason. `recording` is retained only so the
difference can be reproduced.

------------------------------------------------------------------------------------
UNITS — three different questions, three different numbers
------------------------------------------------------------------------------------
Calibration is fitted per 12-second **window**, because that is what the network emits. But the
GUI thresholds the per-**second** mean (`gui/postprocess.per_second_probability`) and presents
**events**. mean(f(p)) != f(mean(p)), so these are genuinely different objects:

    window   raw ECE 0.072   P(ictal | score >= 0.5) = 0.29
    second   raw ECE 0.059   P(ictal | score >= 0.5) = 0.35   <- averaging is itself a
    event                    42 of 289 proposals are real (0.15)   partial calibrator

Always say which object a number describes. `--granularity` selects the fitted unit.

------------------------------------------------------------------------------------
WHAT IS MEASURED
------------------------------------------------------------------------------------
  reliability  observed ictal frequency vs predicted score per bin. Equal-mass bins are the
               default because ~75 % of windows score below 0.01 and equal-width puts three
               quarters of the data in one bar.
  ECE          sample-weighted mean |mean_predicted - observed_frequency| over bins — the
               positive-class reliability form (Naeini et al. 2015; Brocker 2009), NOT Guo's
               confidence-vs-accuracy form, which is near-meaningless at a 5 % base rate where
               predicting "background" everywhere scores 95 % accuracy. Reported with a
               bootstrap CI resampling the grouping unit.
  MCE          worst-bin gap. **Partition-dependent** (0.27 at 5 bins to 0.69 at 200) — a
               diagnostic, never a headline, and never quote a "reduction" in it.
  Brier        with Murphy's REL - RES + UNC decomposition. The identity omits the within-bin
               term, so it misses Brier slightly; the residual is reported explicitly.
  log loss     strictly proper, and what temperature scaling minimises.
  ROC-AUC      to show calibration buys no discrimination. Every map is monotone *within a
               fold*, and per-fold AUC is recorded so that claim is regenerable rather than
               asserted.

METHODS: temperature (p' = sigmoid(logit(p)/T)), Platt (sigmoid(a*logit(p)+b)), isotonic.
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

import eval_config as cfg  # noqa: E402
from gui.io.cache import cache_path_for, load_probability_file  # noqa: E402
from gui.postprocess import per_second_probability  # noqa: E402
from experiments.evaluate_baseline import iter_manifest_rows, load_file  # noqa: E402
from experiments.replicate_paper_auc import paper_protocol_labels  # noqa: E402

EPS = 1e-6          # logit clipping; also the floor used for log loss


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def patient_of(stem):
    """TUSZ patient id is the stem prefix: aaaaaarq_s016_t007 -> aaaaaarq."""
    return str(stem).split('_')[0]


def collect(manifest, protocol='project', granularity='window'):
    """Return (scores, labels, groups_by_recording, stats).

    granularity : 'window' scores the network's own 12 s output;
                  'second' collapses to the per-second mean the GUI thresholds.
    Windows the pipeline never scored are dropped — they carry no model output,
    so they say nothing about the model's calibration. Their count is reported,
    because the GUI currently paints them as p = 0.0.
    """
    P, Y, G = [], [], []
    stats = {'n_manifest_rows': 0, 'skipped_no_cache': 0,
             'skipped_no_reference': 0, 'skipped_no_scored_windows': 0,
             'n_unscored_windows': 0, 'n_unscored_positive': 0,
             'duplicate_stems': []}
    seen = set()
    for edf, cohort in iter_manifest_rows(os.path.abspath(manifest)):
        stats['n_manifest_rows'] += 1
        rec = load_file(edf, cohort)
        if rec is None:
            stats['skipped_no_cache'] += 1
            continue
        if not rec['has_reference']:
            stats['skipped_no_reference'] += 1
            continue
        cache = load_probability_file(cache_path_for(edf))
        if cache is None:
            stats['skipped_no_cache'] += 1
            continue
        if rec['stem'] in seen:
            # Same stem under two montage directories. Benign here (same key ->
            # same fold) but the whole validity argument rests on this key, so
            # make a collision loud rather than lucky.
            stats['duplicate_stems'].append(rec['stem'])
        seen.add(rec['stem'])

        skip = np.asarray(cache['skip_code'])
        scored = skip == 0
        stats['n_unscored_windows'] += int((~scored).sum())
        stats['n_unscored_positive'] += int(rec['labels'][~scored].sum())

        if granularity == 'second':
            _, p_sec = per_second_probability(
                rec['starts'], rec['probs'], cfg.SEGMENT_S,
                duration_s=rec['duration_s'], skip_code=skip)
            if not p_sec.size:
                stats['skipped_no_scored_windows'] += 1
                continue
            y_sec = np.zeros(p_sec.size, dtype=np.int32)
            for ref in rec['refs']:
                a = max(0, int(np.floor(float(ref['start']))))
                b = min(p_sec.size, int(np.ceil(float(ref['stop']))))
                if b > a:
                    y_sec[a:b] = 1
            # Seconds covered only by refused windows read 0.0 and are not
            # model output; drop them for the same reason as unscored windows.
            covered = p_sec > 0
            if not covered.any():
                stats['skipped_no_scored_windows'] += 1
                continue
            P.append(p_sec[covered].astype(np.float64))
            Y.append(y_sec[covered])
            G.append(np.full(int(covered.sum()), rec['stem']))
            continue

        if protocol == 'paper':
            labels, keep = paper_protocol_labels(rec['starts'], rec['refs'])
            mask = scored & keep
        else:
            labels, mask = rec['labels'], scored
        if not mask.any():
            stats['skipped_no_scored_windows'] += 1
            continue
        P.append(rec['probs'][mask].astype(np.float64))
        Y.append(labels[mask])
        G.append(np.full(int(mask.sum()), rec['stem']))

    if not P:
        raise SystemExit('No scorable units found.')
    return np.concatenate(P), np.concatenate(Y), np.concatenate(G), stats


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def reliability(p, y, n_bins=15, strategy='quantile'):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    if strategy == 'quantile':
        edges = np.unique(np.percentile(p, np.linspace(0, 100, n_bins + 1)))
        edges = np.concatenate(([0.0], edges[1:-1], [1.0 + 1e-9]))
    else:
        edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n == 0:
            continue
        bins.append({'lo': float(lo), 'hi': float(min(hi, 1.0)), 'n': n,
                     'mean_predicted': float(p[m].mean()),
                     'observed_frequency': float(y[m].mean()),
                     'n_positive': int(y[m].sum())})
    return bins


def ece_mce(bins, n_total):
    if not bins or not n_total:
        return None, None
    gaps = [(b['n'] / float(n_total),
             abs(b['mean_predicted'] - b['observed_frequency'])) for b in bins]
    return float(sum(w * g for w, g in gaps)), float(max(g for _, g in gaps))


def ece_of(p, y, n_bins=15, strategy='quantile'):
    return ece_mce(reliability(p, y, n_bins, strategy), np.asarray(y).size)[0]


def brier_decomposition(p, y, n_bins=15):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    rel = res = 0.0
    for b in reliability(p, y, n_bins, 'quantile'):
        w = b['n'] / float(y.size)
        rel += w * (b['mean_predicted'] - b['observed_frequency']) ** 2
        res += w * (b['observed_frequency'] - base) ** 2
    unc = base * (1.0 - base)
    return {'brier': brier, 'reliability': float(rel), 'resolution': float(res),
            'uncertainty': float(unc), 'base_rate': base,
            # REL - RES + UNC omits the within-bin variance term; report the
            # residual rather than pretending the identity is exact.
            'decomposition_residual': float(brier - (rel - res + unc))}


def log_loss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(p, y):
    y = np.asarray(y)
    if y.sum() in (0, y.size):
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def summarise(p, y, n_bins=15):
    n = int(np.asarray(y).size)
    uni, qua = reliability(p, y, n_bins, 'uniform'), reliability(p, y, n_bins, 'quantile')
    ece_u, mce_u = ece_mce(uni, n)
    ece_q, mce_q = ece_mce(qua, n)
    out = {'n': n, 'n_positive': int(np.asarray(y).sum()),
           'ece_quantile': ece_q, 'mce_quantile': mce_q,
           'ece_uniform': ece_u, 'mce_uniform': mce_u,
           'n_bins_used_quantile': len(qua), 'n_bins_used_uniform': len(uni),
           'log_loss': log_loss(p, y), 'log_loss_eps': EPS,
           'roc_auc': roc_auc(p, y), 'mean_predicted': float(np.mean(p)),
           'n_exactly_0_or_1': int(np.sum((p <= 0) | (p >= 1))),
           'bins_quantile': qua, 'bins_uniform': uni}
    out.update(brier_decomposition(p, y, n_bins))
    return out


# --------------------------------------------------------------------------
# calibrators
# --------------------------------------------------------------------------
def fit_temperature(p, y):
    from scipy.optimize import minimize_scalar
    z, yy = _logit(p), np.asarray(y, dtype=np.float64)

    def nll(log_t):
        q = np.clip(_sigmoid(z / np.exp(log_t)), EPS, 1 - EPS)
        return -np.mean(yy * np.log(q) + (1 - yy) * np.log(1 - q))
    r = minimize_scalar(nll, bounds=(-4.0, 4.0), method='bounded')
    return {'kind': 'temperature', 'T': float(np.exp(r.x))}


def fit_platt(p, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(solver='lbfgs', C=1e6)
    lr.fit(_logit(p).reshape(-1, 1), np.asarray(y))
    return {'kind': 'platt', 'a': float(lr.coef_[0][0]),
            'b': float(lr.intercept_[0])}


def fit_isotonic(p, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64))
    # Isotonic emits hard 0 and 1, so its log loss would otherwise depend on
    # EPS. Clip to a data-driven floor instead of an arbitrary constant.
    return {'kind': 'isotonic', 'model': iso,
            'clip': 1.0 / (2.0 * max(len(p), 1))}


def apply_calibrator(cal, p):
    p = np.asarray(p, dtype=np.float64)
    if cal['kind'] == 'temperature':
        return _sigmoid(_logit(p) / cal['T'])
    if cal['kind'] == 'platt':
        return _sigmoid(cal['a'] * _logit(p) + cal['b'])
    if cal['kind'] == 'isotonic':
        c = cal.get('clip', 0.0)
        return np.clip(cal['model'].predict(p), c, 1.0 - c)
    raise ValueError(cal['kind'])


FITTERS = {'temperature': fit_temperature, 'platt': fit_platt,
           'isotonic': fit_isotonic}


# --------------------------------------------------------------------------
# grouped cross-validation
# --------------------------------------------------------------------------
def make_groups(rec_groups, group_by):
    if group_by == 'patient':
        return np.array([patient_of(s) for s in rec_groups])
    return np.asarray(rec_groups)


def folds_for(groups, k, seed=13, loo=False):
    uniq = np.array(sorted(set(groups)))
    if loo:
        return [{u} for u in uniq]
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    return [set(uniq[i::k]) for i in range(k)]


def cross_validated(p, y, groups, method, k=5, seed=13, loo=False):
    """Out-of-fold calibrated scores plus per-fold diagnostics."""
    out = np.full(p.shape, np.nan, dtype=np.float64)
    params, fold_auc, n_passthrough = [], [], 0
    for held in folds_for(groups, k, seed, loo):
        te = np.array([g in held for g in groups])
        tr = ~te
        if not te.any() or not tr.any():
            continue
        if np.asarray(y)[tr].sum() == 0:
            out[te] = p[te]
            n_passthrough += int(te.sum())
            continue
        cal = FITTERS[method](p[tr], y[tr])
        out[te] = apply_calibrator(cal, p[te])
        params.append({kk: vv for kk, vv in cal.items() if kk != 'model'})
        a, b = roc_auc(p[te], y[te]), roc_auc(out[te], y[te])
        if a is not None and b is not None:
            fold_auc.append({'raw': a, 'calibrated': b, 'delta': b - a,
                             'n': int(te.sum())})
    miss = np.isnan(out)
    if miss.any():
        out[miss] = p[miss]
        n_passthrough += int(miss.sum())
    return out, {'fold_parameters': params, 'fold_auc': fold_auc,
                 'n_passthrough': n_passthrough}


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------
def bootstrap_ece(scores_by_method, y, groups, n_boot=2000, seed=13, n_bins=15):
    """Cluster bootstrap over the grouping unit, for ECE and paired differences.

    Bins are recomputed inside each draw — otherwise the interval understates
    the binning contribution to the estimate's variability.
    """
    uniq = np.array(sorted(set(groups)))
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.RandomState(seed)
    names = list(scores_by_method)
    draws = {m: [] for m in names}
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_g[uniq[i]] for i in pick])
        yy = y[sel]
        if yy.sum() == 0 or yy.sum() == yy.size:
            continue
        for m in names:
            draws[m].append(ece_of(scores_by_method[m][sel], yy, n_bins))
    out = {}
    for m in names:
        d = np.array([v for v in draws[m] if v is not None])
        out[m] = ({'ci_lo': float(np.percentile(d, 2.5)),
                   'ci_hi': float(np.percentile(d, 97.5)),
                   'n_boot': int(d.size)} if d.size else None)
    # paired differences between every method pair
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = np.array(draws[a]), np.array(draws[b])
            n = min(da.size, db.size)
            if n < 10:
                continue
            diff = da[:n] - db[:n]
            lo, hi = np.percentile(diff, 2.5), np.percentile(diff, 97.5)
            pairs['{} - {}'.format(a, b)] = {
                'ci_lo': float(lo), 'ci_hi': float(hi),
                'separable': bool(lo > 0 or hi < 0)}
    return out, pairs


def threshold_table(p, y, groups, thresholds=(0.5, 0.8, 0.9), n_boot=2000,
                    seed=13):
    """P(ictal | score >= t) with a cluster-bootstrap CI over recordings."""
    uniq = np.array(sorted(set(groups)))
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.RandomState(seed)
    rows = []
    for t in thresholds:
        m = p >= t
        if not m.any():
            continue
        draws = []
        for _ in range(n_boot):
            pick = rng.randint(0, len(uniq), len(uniq))
            sel = np.concatenate([idx_by_g[uniq[i]] for i in pick])
            mm = p[sel] >= t
            if mm.any():
                draws.append(y[sel][mm].mean())
        draws = np.array(draws)
        rows.append({'threshold': float(t), 'n': int(m.sum()),
                     'observed_ictal': float(y[m].mean()),
                     'ci_lo': float(np.percentile(draws, 2.5)) if draws.size else None,
                     'ci_hi': float(np.percentile(draws, 97.5)) if draws.size else None})
    return rows


# --------------------------------------------------------------------------
# reliability diagram — log x, because 13 of 15 bins sit below 0.08
# --------------------------------------------------------------------------
def write_svg(path, curves, title, subtitle=''):
    W, H, ML, MR, MT, MB = 660, 520, 78, 24, 58, 74
    pw, ph = W - ML - MR, H - MT - MB
    lo = 1e-4

    def X(v):
        v = max(float(v), lo)
        return ML + pw * (np.log10(v) - np.log10(lo)) / (0 - np.log10(lo))

    def Y(v):
        return H - MB - float(v) * ph

    col = {'raw': '#b02020', 'temperature': '#1f6fb4',
           'platt': '#2c8a3d', 'isotonic': '#7b4fa8'}
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'viewBox="0 0 %d %d" font-family="system-ui,sans-serif">' % (W, H, W, H),
         '<rect width="%d" height="%d" fill="white"/>' % (W, H)]
    for e in (-4, -3, -2, -1, 0):
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ececec"/>'
                 % (X(10.0 ** e), Y(0), X(10.0 ** e), Y(1)))
        s.append('<text x="%.1f" y="%.1f" font-size="11" fill="#555" '
                 'text-anchor="middle">%s</text>'
                 % (X(10.0 ** e), Y(0) + 18,
                    '1' if e == 0 else '10<tspan font-size="8" dy="-4">%d</tspan>' % e))
    for t in np.linspace(0, 1, 6):
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ececec"/>'
                 % (ML, Y(t), ML + pw, Y(t)))
        s.append('<text x="%.1f" y="%.1f" font-size="11" fill="#555" '
                 'text-anchor="end">%.1f</text>' % (ML - 8, Y(t) + 4, t))
    # identity, drawn only where data exist
    pts = [(10 ** e) for e in np.linspace(np.log10(lo), 0, 60)]
    s.append('<polyline fill="none" stroke="#999" stroke-dasharray="5,4" '
             'points="%s"/>' % ' '.join('%.1f,%.1f' % (X(v), Y(v)) for v in pts))
    s.append('<text x="%.1f" y="%.1f" font-size="10" fill="#888">perfect '
             'calibration</text>' % (X(0.28), Y(0.34)))
    for i, (name, bins) in enumerate(curves):
        c = col.get(name, '#444')
        pp = [(b['mean_predicted'], b['observed_frequency']) for b in bins]
        if pp:
            s.append('<polyline fill="none" stroke="%s" stroke-width="2" '
                     'points="%s"/>' % (c, ' '.join('%.1f,%.1f' % (X(a), Y(b))
                                                    for a, b in pp)))
            for a, b in pp:
                s.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>'
                         % (X(a), Y(b), c))
        s.append('<rect x="%.1f" y="%.1f" width="11" height="11" fill="%s"/>'
                 % (ML + 12, MT + 6 + i * 18, c))
        s.append('<text x="%.1f" y="%.1f" font-size="12" fill="#333">%s</text>'
                 % (ML + 29, MT + 16 + i * 18, name))
    s.append('<text x="%.1f" y="%.1f" font-size="13" fill="#222" '
             'text-anchor="middle">predicted score (log scale)</text>'
             % (ML + pw / 2.0, H - 22))
    s.append('<text x="16" y="%.1f" font-size="13" fill="#222" '
             'text-anchor="middle" transform="rotate(-90 16 %.1f)">observed '
             'ictal frequency</text>' % (Y(0.5), Y(0.5)))
    s.append('<text x="%.1f" y="26" font-size="14" fill="#111" '
             'text-anchor="middle">%s</text>' % (W / 2.0, title))
    if subtitle:
        s.append('<text x="%.1f" y="44" font-size="11" fill="#666" '
                 'text-anchor="middle">%s</text>' % (W / 2.0, subtitle))
    s.append('</svg>')
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    # Explicit UTF-8 with a declaration: Python 3.6's open() uses the locale
    # encoding, which on Windows is cp1252 and silently produces bytes no XML
    # parser will accept for any non-ASCII character in the subtitle.
    import io as _io
    with _io.open(path, 'w', encoding='utf-8') as f:
        f.write(u'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(u'\n'.join(s) + u'\n')


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--protocol', default='project', choices=['project', 'paper'])
    ap.add_argument('--granularity', default='window', choices=['window', 'second'])
    ap.add_argument('--group-by', default='patient', choices=['patient', 'recording'],
                    help='CV grouping unit. patient is correct here: 203 '
                         'recordings come from 28 patients.')
    ap.add_argument('--loo', action='store_true',
                    help='Leave-one-group-out instead of k folds.')
    ap.add_argument('--bins', type=int, default=15)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--out', default=None)
    ap.add_argument('--svg', default=None)
    args = ap.parse_args(argv)

    p, y, rec_groups, stats = collect(args.manifest, args.protocol,
                                      args.granularity)
    groups = make_groups(rec_groups, args.group_by)
    n_rec, n_pat = len(set(rec_groups)), len(set(make_groups(rec_groups, 'patient')))

    print('\n{} {}s from {} recordings / {} patients, {} positive ({:.2f}%)'
          .format(p.size, args.granularity, n_rec, n_pat, int(y.sum()),
                  100.0 * y.mean()))
    print('grouping: {} ({} groups, {})'.format(
        args.group_by, len(set(groups)),
        'leave-one-out' if args.loo else '{}-fold'.format(args.folds)))
    print('excluded: {} no cache, {} no .csv_bi, {} no scored unit; '
          '{} unscored windows carrying {} positives'.format(
              stats['skipped_no_cache'], stats['skipped_no_reference'],
              stats['skipped_no_scored_windows'],
              stats['n_unscored_windows'], stats['n_unscored_positive']))
    if stats['duplicate_stems']:
        print('WARNING duplicate stems (same key -> same fold): {}'
              .format(', '.join(sorted(set(stats['duplicate_stems'])))))

    scores = {'raw': p}
    diag = {}
    for m in ('temperature', 'platt', 'isotonic'):
        q, d = cross_validated(p, y, groups, m, args.folds, loo=args.loo)
        scores[m], diag[m] = q, d

    results = {k: summarise(v, y, args.bins) for k, v in scores.items()}
    ece_ci, pairs = bootstrap_ece(scores, y, rec_groups, args.n_boot,
                                  n_bins=args.bins)

    print('\n{:<13} {:>7} {:>18} {:>7} {:>9} {:>9} {:>8}'.format(
        '', 'ECE', '95% CI', 'MCE', 'Brier', 'log loss', 'ROC-AUC'))
    print('-' * 78)
    for name in ('raw', 'temperature', 'platt', 'isotonic'):
        r, ci = results[name], ece_ci.get(name)
        print('{:<13} {:>7.4f} {:>18} {:>7.2f} {:>9.5f} {:>9.4f} {:>8.4f}'.format(
            name, r['ece_quantile'],
            '[{:.4f}, {:.4f}]'.format(ci['ci_lo'], ci['ci_hi']) if ci else 'n/a',
            r['mce_quantile'], r['brier'], r['log_loss'],
            r['roc_auc'] or float('nan')))
    print('MCE is partition-dependent ({} equal-mass bins) — diagnostic only.'
          .format(results['raw']['n_bins_used_quantile']))

    # Only name a winner when the paired interval excludes zero.
    ranked = sorted(('temperature', 'platt', 'isotonic'),
                    key=lambda m: results[m]['ece_quantile'])
    top, second = ranked[0], ranked[1]
    # pairs are keyed in the order the methods were enumerated, so try both.
    pair = (pairs.get('{} - {}'.format(top, second)) or
            pairs.get('{} - {}'.format(second, top)))
    separable = bool(pair and pair['separable'])
    print('\nBest point estimate: {} (ECE {:.4f}).'.format(
        top, results[top]['ece_quantile']))
    if pair:
        print('Paired vs {}: 95% CI [{:+.4f}, {:+.4f}] — {}.'.format(
            second, pair['ci_lo'], pair['ci_hi'],
            'separable' if separable else 'NOT separable at this sample size'))
    if not separable:
        print('Report as a tie. Prefer platt: fewer parameters, seed-stable, '
              'and its intercept is\nprevalence-shiftable, which isotonic has '
              'no parameter to do.')

    fp = diag['temperature']['fold_parameters']
    if fp:
        print('Temperatures across folds: {}'.format(
            ', '.join('{:.2f}'.format(f['T']) for f in fp)))
    for m in ('temperature', 'platt', 'isotonic'):
        d = [f['delta'] for f in diag[m]['fold_auc']]
        if d:
            print('{:<12} within-fold AUC delta: min {:+.6f} max {:+.6f} '
                  '{}'.format(m, min(d), max(d),
                              '(identical — monotone)' if max(abs(x) for x in d) < 1e-9
                              else '(ties from quantisation)'))

    thr = threshold_table(p, y, rec_groups, n_boot=args.n_boot)
    print('\nWhat a {} score threshold actually means (CI resamples recordings):'
          .format(args.granularity))
    for r in thr:
        print('  >= {:.1f}: {:6d} {}s, {:.1%} ictal  95% CI [{:.1%}, {:.1%}]'
              .format(r['threshold'], r['n'], args.granularity,
                      r['observed_ictal'], r['ci_lo'], r['ci_hi']))
    print('All figures are conditional on this corpus base rate of {:.4f}.'
          .format(results['raw']['base_rate']))

    if args.svg:
        write_svg(args.svg,
                  [(n, results[n]['bins_quantile'])
                   for n in ('raw', 'platt', 'isotonic')],
                  'Reliability of the per-{} seizure score'.format(args.granularity),
                  '{} equal-mass bins · out-of-fold, grouped by {} · n={} '
                  '({:.2f}% ictal)'.format(results['raw']['n_bins_used_quantile'],
                                           args.group_by, p.size,
                                           100.0 * y.mean()))
        print('\nwrote {}'.format(args.svg))

    if args.out:
        payload = {
            'protocol': args.protocol, 'granularity': args.granularity,
            'group_by': args.group_by, 'loo': bool(args.loo),
            'n_folds': args.folds, 'n_bins_requested': args.bins,
            'n_recordings': n_rec, 'n_patients': n_pat,
            'collection_stats': stats, 'config': cfg.as_dict(),
            'results': results, 'ece_bootstrap_ci': ece_ci,
            'ece_paired_differences': pairs,
            'threshold_table': thr,
            'diagnostics': {m: {'fold_parameters': d['fold_parameters'],
                                'fold_auc': d['fold_auc'],
                                'n_passthrough': d['n_passthrough']}
                            for m, d in diag.items()},
            'best_point_estimate': top,
            'best_is_separable': separable,
            'recommended': top if separable else 'platt',
            'note': (
                'Out-of-fold under cross-validation grouped by {}. Grouping by '
                'RECORDING leaks: 203 recordings come from 28 patients, so ~95% '
                'of test windows would have the same patient in training, which '
                'flatters isotonic specifically. Each calibrator is monotone '
                'within its fold, so calibration corrects the MEANING of the '
                'score, not the ranking; per-fold AUC deltas are recorded above. '
                'Every calibrated probability is conditional on a base rate of '
                '{:.4f} on a seizure-enriched corpus and is not transportable to '
                'ambulatory prevalence without re-shifting Platt\'s intercept '
                'under a label-shift assumption; isotonic has no such parameter. '
                'MCE is partition-dependent and is a diagnostic only.'
            ).format(args.group_by, results['raw']['base_rate']),
        }
        out = os.path.abspath(args.out)
        parent = os.path.dirname(out)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        with open(out, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')
        print('wrote {}'.format(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
