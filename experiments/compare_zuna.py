"""Compare detector results on original EDF data and ZUNA-reconstructed data.

This script keeps the detector fixed. The only intended difference between
rows is the signal source:
    baseline : original EDF -> existing ConvLSTM path
    zuna     : ZUNA 19-channel NPZ -> existing ConvLSTM path
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.events import rebuild_events  # noqa: E402
from gui.io.cache import load_probs  # noqa: E402
from gui.io.csv_bi import read_csv_bi  # noqa: E402

SEGMENT_S = 12


def ref_path_for_edf(edf_path):
    return os.path.splitext(edf_path)[0] + '.csv_bi'


def read_reference_events(csv_bi_path):
    refs = []
    for ev in read_csv_bi(csv_bi_path):
        if ev.get('label') != 'seiz':
            continue
        start = float(ev['start'])
        stop = float(ev['stop'])
        if stop > start:
            refs.append({
                'start': start,
                'stop': stop,
                'label': 'seiz',
            })
    refs.sort(key=lambda ev: (ev['start'], ev['stop']))
    return refs


def read_reference_duration(csv_bi_path):
    if not csv_bi_path or not os.path.exists(csv_bi_path):
        return None
    with open(csv_bi_path) as f:
        for line in f:
            line = line.strip()
            if not line.lower().startswith('# duration'):
                continue
            try:
                return float(line.split('=', 1)[1].strip().split()[0])
            except (IndexError, ValueError):
                return None
    return None


def load_probability_file(path):
    if not os.path.exists(path):
        raise SystemExit('Probability NPZ not found: {}'.format(path))
    with np.load(path, allow_pickle=False) as z:
        starts = z['window_starts'].astype(np.int32, copy=False)
        probs = z['probs'].astype(np.float32, copy=False)
        meta = {}
        if 'meta' in z:
            try:
                meta = json.loads(str(z['meta']))
            except ValueError:
                meta = {}
    if starts.shape != probs.shape:
        raise RuntimeError(
            'Probability file has mismatched starts/probs shapes: {}'.format(
                path))
    return starts, probs, meta


def save_probability_file(path, window_starts, probs, meta):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    np.savez_compressed(
        path,
        window_starts=np.asarray(window_starts, dtype=np.int32),
        probs=np.asarray(probs, dtype=np.float32),
        meta=np.array(json.dumps(dict(meta or {}))),
    )


def infer_step_s(window_starts):
    starts = np.asarray(window_starts)
    if starts.size < 2:
        return None
    diffs = np.diff(starts)
    if diffs.size == 0:
        return None
    return int(round(float(np.median(diffs))))


def validate_step(window_starts, requested_step_s, label):
    actual = infer_step_s(window_starts)
    if actual is not None and int(actual) != int(requested_step_s):
        raise RuntimeError(
            '{} probabilities use step {}s, but --step is {}s'.format(
                label, actual, requested_step_s))


def _overlap_seconds(a, b):
    return max(0.0, min(float(a['stop']), float(b['stop'])) -
               max(float(a['start']), float(b['start'])))


def _within_tolerance(pred, ref, tolerance_s):
    return (float(pred['stop']) + tolerance_s >= float(ref['start']) and
            float(pred['start']) - tolerance_s <= float(ref['stop']))


def match_events(pred_events, ref_events, tolerance_s=5.0):
    """Greedily match predictions to references, one prediction per ref."""
    unmatched_refs = set(range(len(ref_events)))
    matches = []
    false_positive_indexes = []

    for pred_idx, pred in enumerate(pred_events):
        best_ref_idx = None
        best_key = None
        for ref_idx in unmatched_refs:
            ref = ref_events[ref_idx]
            if not _within_tolerance(pred, ref, tolerance_s):
                continue
            overlap = _overlap_seconds(pred, ref)
            onset_err = abs(float(pred['start']) - float(ref['start']))
            offset_err = abs(float(pred['stop']) - float(ref['stop']))
            key = (overlap, -onset_err, -offset_err)
            if best_key is None or key > best_key:
                best_key = key
                best_ref_idx = ref_idx
        if best_ref_idx is None:
            false_positive_indexes.append(pred_idx)
            continue
        unmatched_refs.remove(best_ref_idx)
        matches.append({
            'pred_index': pred_idx,
            'ref_index': best_ref_idx,
            'overlap_s': _overlap_seconds(pred, ref_events[best_ref_idx]),
            'onset_abs_error_s': abs(
                float(pred['start']) -
                float(ref_events[best_ref_idx]['start'])),
            'offset_abs_error_s': abs(
                float(pred['stop']) -
                float(ref_events[best_ref_idx]['stop'])),
        })

    return matches, false_positive_indexes, sorted(unmatched_refs)


def _mean_or_none(values):
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def probability_stats(probs):
    probs = np.asarray(probs, dtype=np.float32)
    if probs.size == 0:
        return {'max_prob': None, 'p90_prob': None, 'mean_prob': None}
    return {
        'max_prob': float(np.max(probs)),
        'p90_prob': float(np.percentile(probs, 90)),
        'mean_prob': float(np.mean(probs)),
    }


def summarize_run(name, window_starts, probs, ref_events, threshold,
                  duration_s=None, tolerance_s=5.0, segment_s=SEGMENT_S,
                  source=None):
    pred_events = rebuild_events(
        window_starts, probs, threshold, segment_s,
        duration_s=duration_s, preserve_review=False)
    if duration_s is None:
        if len(window_starts):
            duration_s = float(np.max(window_starts)) + float(segment_s)
        elif ref_events:
            duration_s = max(float(ev['stop']) for ev in ref_events)
        else:
            duration_s = 0.0

    matches, fp_indexes, miss_indexes = match_events(
        pred_events, ref_events, tolerance_s=tolerance_s)
    duration_days = float(duration_s) / 86400.0 if duration_s else 0.0
    false_alarm_rate = (
        float(len(fp_indexes)) / duration_days if duration_days > 0 else None)

    out = {
        'name': name,
        'source': source,
        'threshold': float(threshold),
        'duration_s': float(duration_s),
        'n_refs': len(ref_events),
        'n_pred': len(pred_events),
        'hits': len(matches),
        'misses': len(miss_indexes),
        'false_positives': len(fp_indexes),
        'sensitivity': (
            float(len(matches)) / float(len(ref_events))
            if ref_events else None),
        'false_alarms_per_24h': false_alarm_rate,
        'mean_onset_abs_error_s': _mean_or_none(
            [m['onset_abs_error_s'] for m in matches]),
        'mean_offset_abs_error_s': _mean_or_none(
            [m['offset_abs_error_s'] for m in matches]),
        'pred_events': pred_events,
        'reference_events': list(ref_events),
        'matches': matches,
        'false_positive_indexes': fp_indexes,
        'missed_reference_indexes': miss_indexes,
    }
    out.update(probability_stats(probs))
    return out


def load_baseline_probs(args):
    if args.baseline_probs:
        starts, probs, meta = load_probability_file(args.baseline_probs)
        validate_step(starts, args.step, 'baseline')
        return starts, probs, 'probability file', args.baseline_probs, meta

    if not args.recompute_baseline:
        cached = load_probs(args.edf)
        if cached is not None:
            starts = cached['window_starts']
            validate_step(starts, args.step, 'baseline cache')
            return (
                starts,
                cached['probs'],
                'EDF sidecar cache',
                os.path.splitext(args.edf)[0] + '.probs.npz',
                cached.get('meta', {}),
            )

    t0 = time.time()
    from gui.io.infer import compute_probs  # noqa: E402
    starts, probs = compute_probs(
        args.edf, step_s=args.step, use_ica=not args.baseline_no_ica)
    return starts, probs, 'computed in {:.1f}s'.format(time.time() - t0), (
        args.edf), {
            'step_s': args.step,
            'segment_s': SEGMENT_S,
            'use_ica': not args.baseline_no_ica,
        }


def load_zuna_probs(args):
    if args.zuna_probs:
        starts, probs, meta = load_probability_file(args.zuna_probs)
        validate_step(starts, args.step, 'ZUNA')
        return starts, probs, 'probability file', args.zuna_probs, meta
    if not args.zuna_npz:
        return None
    if not os.path.exists(args.zuna_npz):
        raise SystemExit(
            'ZUNA 19-channel NPZ not found: {}\n\n'
            'Generate it first with the bridge commands in docs/zuna_bridge.md:\n'
            '  1. conda create -n zuna-eeg python=3.10 -y\n'
            '  2. conda run -n zuna-eeg pip install zuna mne\n'
            '  3. conda run -n zuna-eeg python utils\\zuna_bridge.py prepare ...\n'
            '  4. conda run -n zuna-eeg python utils\\zuna_bridge.py run ...\n'
            '  5. conda run -n zuna-eeg python utils\\zuna_bridge.py export-npz ...\n\n'
            'Until that file exists, run baseline-only by omitting --zuna-npz.'
            .format(args.zuna_npz))

    from gui.io.infer import compute_probs_from_data, load_signal_npz  # noqa: E402
    data, fs, duration_s = load_signal_npz(args.zuna_npz)
    t0 = time.time()
    starts, probs = compute_probs_from_data(
        data, fs, step_s=args.step, use_ica=not args.zuna_no_ica)
    meta = {
        'step_s': args.step,
        'segment_s': SEGMENT_S,
        'use_ica': not args.zuna_no_ica,
        'fs': fs,
        'duration_s': duration_s,
        'source_npz': args.zuna_npz,
    }
    if args.zuna_probs_out:
        save_probability_file(args.zuna_probs_out, starts, probs, meta)
    return starts, probs, 'computed in {:.1f}s'.format(time.time() - t0), (
        args.zuna_npz), meta


def _format_number(value, width, precision=3):
    if value is None:
        return '{:>{}}'.format('n/a', width)
    return ('{:>' + str(width) + '.' + str(precision) + 'f}').format(
        float(value))


def print_table(rows):
    print('')
    print('{:<12} {:>4} {:>4} {:>4} {:>4} {:>6} {:>9} {:>9} {:>7}'.format(
        'branch', 'refs', 'pred', 'hit', 'miss', 'fp', 'fp/24h',
        'onset', 'max_p'))
    for row in rows:
        print('{:<12} {:>4} {:>4} {:>4} {:>4} {:>6} {} {} {}'.format(
            row['name'][:12],
            row['n_refs'],
            row['n_pred'],
            row['hits'],
            row['misses'],
            row['false_positives'],
            _format_number(row['false_alarms_per_24h'], 9, 2),
            _format_number(row['mean_onset_abs_error_s'], 9, 2),
            _format_number(row['max_prob'], 7, 3),
        ))


def write_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description='Compare baseline detector output against ZUNA output.')
    ap.add_argument('--edf', required=True,
                    help='Original EDF path used for baseline and labels.')
    ap.add_argument('--zuna-npz', default=None,
                    help='Repo-format 19-channel NPZ exported by zuna_bridge.')
    ap.add_argument('--zuna-probs', default=None,
                    help='Precomputed probability NPZ for the ZUNA branch.')
    ap.add_argument('--zuna-probs-out', default=None,
                    help='Optional path to save computed ZUNA probabilities.')
    ap.add_argument('--baseline-probs', default=None,
                    help='Optional baseline probability NPZ override.')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--step', type=int, default=6)
    ap.add_argument('--tolerance', type=float, default=5.0,
                    help='Seconds of event matching tolerance.')
    ap.add_argument('--out', default=None,
                    help='Optional JSON result path.')
    ap.add_argument('--recompute-baseline', action='store_true',
                    help='Ignore the EDF sidecar .probs.npz cache.')
    ap.add_argument('--baseline-no-ica', action='store_true',
                    help='Skip ICA for a recomputed baseline branch.')
    ap.add_argument('--zuna-no-ica', action='store_true',
                    help='Skip ICA when scoring ZUNA NPZ data.')
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    args.edf = os.path.abspath(args.edf)

    ref_path = ref_path_for_edf(args.edf)
    ref_events = read_reference_events(ref_path)
    duration_s = read_reference_duration(ref_path)

    rows = []
    starts, probs, mode, source, meta = load_baseline_probs(args)
    rows.append(summarize_run(
        'baseline', starts, probs, ref_events,
        threshold=args.threshold,
        duration_s=duration_s,
        tolerance_s=args.tolerance,
        source={'mode': mode, 'path': source, 'meta': meta}))

    zuna_loaded = load_zuna_probs(args)
    if zuna_loaded is not None:
        starts, probs, mode, source, meta = zuna_loaded
        rows.append(summarize_run(
            'zuna', starts, probs, ref_events,
            threshold=args.threshold,
            duration_s=duration_s,
            tolerance_s=args.tolerance,
            source={'mode': mode, 'path': source, 'meta': meta}))

    payload = {
        'edf': args.edf,
        'reference_csv_bi': ref_path,
        'threshold': float(args.threshold),
        'step_s': int(args.step),
        'segment_s': SEGMENT_S,
        'tolerance_s': float(args.tolerance),
        'rows': rows,
    }

    print_table(rows)
    if zuna_loaded is None:
        print('\nZUNA row not run: pass --zuna-npz or --zuna-probs.')
    if args.out:
        write_json(args.out, payload)
        print('\nwrote {}'.format(args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
