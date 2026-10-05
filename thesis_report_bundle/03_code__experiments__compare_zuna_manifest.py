"""Run and aggregate original-vs-ZUNA comparisons for a manifest CSV."""
from __future__ import print_function

import argparse
import csv
import json
import os
import subprocess
import sys


def build_arg_parser():
    ap = argparse.ArgumentParser(description='Compare ZUNA outputs by manifest.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--zuna-npz-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--step', type=int, default=6)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--allow-missing', action='store_true')
    return ap


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')


def aggregate(results):
    out = {}
    for item in results:
        for row in item['comparison']['rows']:
            branch = row['name']
            agg = out.setdefault(branch, {
                'n_files': 0,
                'duration_s': 0.0,
                'n_refs': 0,
                'n_pred': 0,
                'hits': 0,
                'misses': 0,
                'false_positives': 0,
            })
            agg['n_files'] += 1
            agg['duration_s'] += float(row['duration_s'])
            for key in ('n_refs', 'n_pred', 'hits', 'misses',
                        'false_positives'):
                agg[key] += int(row[key])
    for agg in out.values():
        agg['sensitivity'] = (
            float(agg['hits']) / float(agg['n_refs'])
            if agg['n_refs'] else None)
        agg['false_alarms_per_24h'] = (
            float(agg['false_positives']) / (agg['duration_s'] / 86400.0)
            if agg['duration_s'] else None)
    return out


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    compare_script = os.path.join(repo, 'experiments', 'compare_zuna.py')

    if not os.path.exists(args.manifest):
        raise SystemExit('manifest not found: {}'.format(args.manifest))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(args.manifest, newline='') as f:
        rows = list(csv.DictReader(f))

    results = []
    missing = []
    for row in rows:
        stem = row['stem']
        edf = os.path.abspath(row['edf'])
        zuna_npz = os.path.join(args.zuna_npz_dir, stem + '.zuna_19ch.npz')
        if not os.path.exists(zuna_npz):
            missing.append({'stem': stem, 'path': zuna_npz})
            print('MISSING {}'.format(stem))
            if args.allow_missing:
                continue
            raise SystemExit('ZUNA NPZ missing: {}'.format(zuna_npz))

        result_path = os.path.join(args.out_dir, stem + '.zuna_compare.json')
        probs_path = os.path.join(args.out_dir, stem + '.zuna.probs.npz')
        log_path = os.path.join(args.out_dir, stem + '.compare.log')
        cmd = [
            args.python,
            compare_script,
            '--edf', edf,
            '--zuna-npz', zuna_npz,
            '--zuna-probs-out', probs_path,
            '--threshold', str(args.threshold),
            '--step', str(args.step),
            '--out', result_path,
        ]
        print('COMPARE {}'.format(stem))
        with open(log_path, 'w') as log:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True)
            for line in proc.stdout:
                log.write(line)
            ret = proc.wait()
        if ret:
            raise SystemExit(
                'comparison failed for {} with exit code {}'.format(
                    stem, ret))

        results.append({
            'manifest_row': row,
            'comparison_path': result_path,
            'probability_path': probs_path,
            'log_path': log_path,
            'comparison': load_json(result_path),
        })

    summary = {
        'manifest': os.path.abspath(args.manifest),
        'threshold': float(args.threshold),
        'step_s': int(args.step),
        'n_manifest_rows': len(rows),
        'n_completed': len(results),
        'missing': missing,
        'aggregate': aggregate(results),
        'results': results,
    }
    summary_path = os.path.join(args.out_dir, 'zuna_manifest_summary.json')
    write_json(summary_path, summary)
    print('wrote {}'.format(summary_path))
    for name, agg in sorted(summary['aggregate'].items()):
        print('{} files={} refs={} pred={} hits={} misses={} fp={} sens={} fp24h={}'.format(
            name,
            agg['n_files'],
            agg['n_refs'],
            agg['n_pred'],
            agg['hits'],
            agg['misses'],
            agg['false_positives'],
            agg['sensitivity'],
            agg['false_alarms_per_24h'],
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
