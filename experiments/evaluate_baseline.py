"""Reproducible baseline evaluation for the seizure-detection pipeline.

This is the committed generator for the numbers WS1 promises. It reads the
per-window probability caches that already exist next to each EDF
(`<edf>.probs.npz`) plus the TUSZ `.csv_bi` references, and produces, from one
command, all three evaluation views:

  1. Window-level (paper-style)  : ROC-AUC and PR-AUC over per-window labels,
                                   pooled across files and per file, with the
                                   pooled ROC and PR curve arrays for plotting.
  2. Event-level (GUI view)      : a threshold sweep giving event sensitivity,
                                   false alarms per 24 h, and mean onset/offset
                                   error at each threshold, pooled and split by
                                   cohort (seizure vs background files).
  3. Clinical-review simulation  : per-file "sent for review" decision (>=1
                                   above-threshold candidate) and the reviewer
                                   alarm burden, so the human-in-the-loop triage
                                   load can be reported honestly.

It needs NO TensorFlow and re-runs no inference: it only scores cached probs, so
it is fast and fully reproducible. Run it in the seiz36 env (numpy + sklearn).

Window labelling: a window starting at t is positive if [t, t + SEGMENT_S]
overlaps any `seiz` reference interval.

Skipped windows: the inference path writes a probability of exactly 0.0 for
windows it skips (interrupted data, or ICA failure). A real softmax output is
effectively never exactly 0.0, so by default these are EXCLUDED from the
window-level AUC (which is meant to measure model quality, not preprocessing
drop-outs) and the count is reported. They are KEPT for event-level scoring,
because a skipped window is a genuine non-detection in the deployed pipeline.

Example:
    python experiments/evaluate_baseline.py \
        --manifest artifacts/zuna_thesis/manifest.csv \
        --name baseline26 \
        --out artifacts/zuna_thesis/baseline_eval/baseline26.json
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

import eval_config as cfg  # noqa: E402
from gui.io.cache import cache_path_for, load_probability_file  # noqa: E402
from experiments.compare_zuna import (  # noqa: E402
    read_reference_events, read_reference_duration, summarize_run,
)

SEGMENT_S = cfg.SEGMENT_S


# --------------------------------------------------------------------------
# file discovery
# --------------------------------------------------------------------------
def iter_manifest_rows(manifest_path):
    """Yield (edf_abspath, cohort) from a manifest CSV with an 'edf' column."""
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            edf = row.get('edf')
            if not edf:
                continue
            edf_abs = edf if os.path.isabs(edf) else os.path.join(REPO, edf)
            cohort = row.get('cohort') or None
            yield os.path.normpath(edf_abs), cohort


def iter_edf_args(paths):
    """Yield (edf_abspath, None) from explicit EDF paths/globs."""
    import glob
    for p in paths:
        expanded = glob.glob(p, recursive=True) or [p]
        for e in expanded:
            if e.lower().endswith('.edf'):
                yield os.path.abspath(e), None


# --------------------------------------------------------------------------
# window-level labelling
# --------------------------------------------------------------------------
def window_labels(window_starts, ref_events, segment_s=SEGMENT_S):
    """1 if [t, t+segment_s] overlaps any seiz reference, else 0."""
    labels = np.zeros(len(window_starts), dtype=np.int32)
    for i, t in enumerate(window_starts):
        w0 = float(t)
        w1 = w0 + float(segment_s)
        for ref in ref_events:
            if min(w1, float(ref['stop'])) - max(w0, float(ref['start'])) > 0.0:
                labels[i] = 1
                break
    return labels


def _safe_auc(labels, scores, kind):
    """ROC-AUC or PR-AUC, or None when undefined (one class / empty)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0:
        return None
    pos = int(labels.sum())
    if kind == 'roc':
        if pos == 0 or pos == labels.size:   # ROC-AUC needs both classes
            return None
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    if pos == 0:                             # PR-AUC needs >=1 positive
        return None
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, scores))


# --------------------------------------------------------------------------
# per-file load + score
# --------------------------------------------------------------------------
def load_file(edf_abs, cohort, exclude_zero=True):
    """Load one file's cached probs + refs; return a record or None if no cache."""
    cache = cache_path_for(edf_abs)
    loaded = load_probability_file(cache)
    if loaded is None:
        return None
    starts = np.asarray(loaded['window_starts'], dtype=np.int32)
    probs = np.asarray(loaded['probs'], dtype=np.float32)
    ref_path = os.path.splitext(edf_abs)[0] + '.csv_bi'
    refs = read_reference_events(ref_path)
    duration_s = read_reference_duration(ref_path)
    if duration_s is None and starts.size:
        duration_s = float(starts.max()) + float(SEGMENT_S)

    labels = window_labels(starts, refs)
    zero_mask = (probs == 0.0)
    keep = ~zero_mask if exclude_zero else np.ones(len(probs), dtype=bool)

    inferred_cohort = cohort or ('seizure' if refs else 'nonseizure')
    return {
        'edf': edf_abs,
        'stem': os.path.splitext(os.path.basename(edf_abs))[0],
        'cohort': inferred_cohort,
        'starts': starts,
        'probs': probs,
        'labels': labels,
        'keep': keep,
        'n_excluded_zero': int(zero_mask.sum()),
        'refs': refs,
        'duration_s': float(duration_s or 0.0),
    }


# --------------------------------------------------------------------------
# mode 1: window-level AUC
# --------------------------------------------------------------------------
def window_level(records):
    all_labels, all_scores = [], []
    per_file = []
    for r in records:
        lab = r['labels'][r['keep']]
        sc = r['probs'][r['keep']]
        all_labels.append(lab)
        all_scores.append(sc)
        per_file.append({
            'stem': r['stem'],
            'cohort': r['cohort'],
            'n_windows': int(r['keep'].sum()),
            'n_pos_windows': int(lab.sum()),
            'n_excluded_zero': r['n_excluded_zero'],
            'window_auc': _safe_auc(lab, sc, 'roc'),
            'window_pr_auc': _safe_auc(lab, sc, 'pr'),
        })
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    scores = np.concatenate(all_scores) if all_scores else np.array([])

    out = {
        'pooled_window_auc': _safe_auc(labels, scores, 'roc'),
        'pooled_window_pr_auc': _safe_auc(labels, scores, 'pr'),
        'n_windows': int(labels.size),
        'n_pos_windows': int(labels.sum()) if labels.size else 0,
        'window_auc_by_file': {p['stem']: p['window_auc'] for p in per_file},
        'per_file': per_file,
    }
    # pooled curves for plotting (only if both classes present)
    if labels.size and 0 < int(labels.sum()) < labels.size:
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(labels, scores)
        prec, rec, _ = precision_recall_curve(labels, scores)
        out['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        out['pr_curve'] = {'precision': prec.tolist(), 'recall': rec.tolist()}
    return out


# --------------------------------------------------------------------------
# mode 2: event-level threshold sweep (pooled + per cohort)
# --------------------------------------------------------------------------
def _empty_acc():
    return {'hits': 0, 'refs': 0, 'fp': 0, 'dup': 0, 'duration_s': 0.0,
            'onset': [], 'offset': [], 'files': 0}


def _finalise_acc(acc):
    days = acc['duration_s'] / 86400.0
    return {
        'files': acc['files'],
        'n_refs': acc['refs'],
        'hits': acc['hits'],
        'misses': acc['refs'] - acc['hits'],
        'false_positives': acc['fp'],
        # Extra predictions landing inside an already-detected seizure. Alarm
        # burden for the reviewer, but not false detections.
        'duplicate_detections': acc['dup'],
        'sensitivity': (acc['hits'] / acc['refs']) if acc['refs'] else None,
        'false_alarms_per_24h': (acc['fp'] / days) if days > 0 else None,
        'mean_onset_abs_error_s': (
            float(np.mean(acc['onset'])) if acc['onset'] else None),
        'mean_offset_abs_error_s': (
            float(np.mean(acc['offset'])) if acc['offset'] else None),
    }


def event_sweep(records, thresholds, tolerance_s, postprocess=None):
    sweep = []
    for thr in thresholds:
        pooled = _empty_acc()
        by_cohort = {}
        for r in records:
            row = summarize_run(
                r['stem'], r['starts'], r['probs'], r['refs'],
                threshold=thr, duration_s=r['duration_s'],
                tolerance_s=tolerance_s, postprocess=postprocess)
            cohort = r['cohort']
            acc_targets = [pooled, by_cohort.setdefault(cohort, _empty_acc())]
            for acc in acc_targets:
                acc['files'] += 1
                acc['hits'] += row['hits']
                acc['refs'] += row['n_refs']
                acc['fp'] += row['false_positives']
                acc['dup'] += row.get('duplicate_detections', 0)
                acc['duration_s'] += row['duration_s']
                acc['onset'].extend(
                    m['onset_abs_error_s'] for m in row['matches'])
                acc['offset'].extend(
                    m['offset_abs_error_s'] for m in row['matches'])
        sweep.append({
            'threshold': float(thr),
            'pooled': _finalise_acc(pooled),
            'by_cohort': {c: _finalise_acc(a) for c, a in by_cohort.items()},
        })
    return sweep


# --------------------------------------------------------------------------
# mode 3: clinical-review simulation
# --------------------------------------------------------------------------
def review_simulation(records, threshold):
    """Reviewer-triage view at one operating threshold.

    A file is 'sent for review' if it has >=1 window at/above threshold. We
    report how many seizure files would reach a reviewer (file-level recall),
    how many background files raise a (false) review flag, and the candidate
    alarm burden a reviewer would actually step through.
    """
    seiz_files = [r for r in records if r['cohort'] == 'seizure']
    bg_files = [r for r in records if r['cohort'] != 'seizure']

    def flagged(r):
        return bool(np.any(r['probs'] >= threshold))

    def n_candidate_windows(r):
        return int(np.count_nonzero(r['probs'] >= threshold))

    seiz_flagged = sum(1 for r in seiz_files if flagged(r))
    bg_flagged = sum(1 for r in bg_files if flagged(r))
    total_candidates = sum(n_candidate_windows(r) for r in records)
    total_hours = sum(r['duration_s'] for r in records) / 3600.0
    return {
        'threshold': float(threshold),
        'seizure_files': len(seiz_files),
        'seizure_files_flagged': seiz_flagged,
        'seizure_file_recall': (
            seiz_flagged / len(seiz_files)) if seiz_files else None,
        'background_files': len(bg_files),
        'background_files_false_flagged': bg_flagged,
        'background_file_false_flag_rate': (
            bg_flagged / len(bg_files)) if bg_files else None,
        'total_candidate_windows': total_candidates,
        'total_recording_hours': round(total_hours, 3),
        'candidate_windows_per_hour': (
            total_candidates / total_hours) if total_hours > 0 else None,
    }


# --------------------------------------------------------------------------
def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--manifest', help='CSV with an "edf" column (and optional "cohort").')
    src.add_argument('--edfs', nargs='+', help='Explicit EDF paths or globs.')
    ap.add_argument('--name', default='baseline', help='Label for this run.')
    ap.add_argument('--out', default=None, help='Output JSON path.')
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=cfg.THRESHOLD_GRID,
                    help='Threshold grid for the event sweep.')
    ap.add_argument('--review-threshold', type=float, default=cfg.THRESHOLD,
                    help='Operating threshold for the review simulation.')
    ap.add_argument('--tolerance', type=float, default=cfg.EVENT_TOLERANCE_S)
    ap.add_argument('--keep-zero', action='store_true',
                    help='Keep prob==0 (skipped) windows in the AUC (default: exclude).')
    pp = ap.add_mutually_exclusive_group()
    pp.add_argument('--postprocess', dest='postprocess', action='store_true',
                    default=None,
                    help='Apply the source method decision stage (per-second '
                         'averaging, concatenate <%gs, discard <%gs). Default '
                         'comes from eval_config.USE_SOURCE_POSTPROCESSING.'
                         % (cfg.MAX_MERGE_GAP_S, cfg.MIN_EVENT_DURATION_S))
    pp.add_argument('--no-postprocess', dest='postprocess',
                    action='store_false',
                    help='Threshold raw windows instead (pre-fix behaviour).')
    return ap


def _fmt(v, p=3):
    return 'n/a' if v is None else ('{:.' + str(p) + 'f}').format(v)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.manifest:
        sources = list(iter_manifest_rows(os.path.abspath(args.manifest)))
    else:
        sources = list(iter_edf_args(args.edfs))

    records, missing = [], []
    for edf_abs, cohort in sources:
        rec = load_file(edf_abs, cohort, exclude_zero=not args.keep_zero)
        (records if rec is not None else missing).append(
            rec if rec is not None else edf_abs)

    if not records:
        raise SystemExit(
            'No probability caches found. Run precompute_probs.py first.\n'
            'Looked for <edf>.probs.npz next to {} EDF(s).'.format(len(sources)))

    postprocess = (cfg.USE_SOURCE_POSTPROCESSING if args.postprocess is None
                   else args.postprocess)
    win = window_level(records)
    sweep = event_sweep(records, args.thresholds, args.tolerance,
                        postprocess=postprocess)
    review = review_simulation(records, args.review_threshold)

    payload = {
        'name': args.name,
        'config': dict(cfg.as_dict(), use_source_postprocessing=postprocess),
        'exclude_zero_prob_windows': not args.keep_zero,
        'n_files_scored': len(records),
        'n_files_missing_cache': len(missing),
        'missing_caches': [os.path.basename(m) for m in missing],
        'window_level': win,
        'event_sweep': sweep,
        'review_simulation': review,
    }

    # --- console summary ---
    print('\n=== {} : {} files scored ({} missing cache) ==='.format(
        args.name, len(records), len(missing)))
    print('Window-level: pooled ROC-AUC = {}   pooled PR-AUC = {}   '
          '({} windows, {} positive)'.format(
              _fmt(win['pooled_window_auc']), _fmt(win['pooled_window_pr_auc']),
              win['n_windows'], win['n_pos_windows']))
    print('\nEvent sweep (pooled) — source post-processing {}:'.format(
        'ON (avg + concat <{:g}s + discard <{:g}s)'.format(
            cfg.MAX_MERGE_GAP_S, cfg.MIN_EVENT_DURATION_S)
        if postprocess else 'OFF (raw window threshold)'))
    print('  {:>6}  {:>5}  {:>7}  {:>5}  {:>9}  {:>5}'.format(
        'thr', 'sens', 'hits', 'fp', 'fp/24h', 'dup'))
    for s in sweep:
        p = s['pooled']
        print('  {:>6}  {:>5}  {:>7}  {:>5}  {:>9}  {:>5}'.format(
            _fmt(s['threshold'], 2), _fmt(p['sensitivity'], 3),
            '{}/{}'.format(p['hits'], p['n_refs']),
            p['false_positives'], _fmt(p['false_alarms_per_24h'], 1),
            p['duplicate_detections']))
    rv = review
    print('\nReview simulation @ thr {}: seizure-file recall {} ({}/{}), '
          'background false-flag {} ({}/{}), {} candidate windows over {} h'.format(
              _fmt(rv['threshold'], 2), _fmt(rv['seizure_file_recall'], 3),
              rv['seizure_files_flagged'], rv['seizure_files'],
              _fmt(rv['background_file_false_flag_rate'], 3),
              rv['background_files_false_flagged'], rv['background_files'],
              rv['total_candidate_windows'], _fmt(rv['total_recording_hours'], 1)))
    if missing:
        print('\nNOTE: {} file(s) had no cache and were skipped: {}'.format(
            len(missing), ', '.join(os.path.basename(m) for m in missing[:8])
            + (' ...' if len(missing) > 8 else '')))

    if args.out:
        out = os.path.abspath(args.out)
        parent = os.path.dirname(out)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        with open(out, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')
        print('\nwrote {}'.format(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
