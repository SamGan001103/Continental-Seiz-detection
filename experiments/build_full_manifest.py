"""Build an evaluation manifest over every TUSZ EDF available locally.

The committed manifest (`artifacts/zuna_thesis/manifest.csv`) is 26
seizure-enriched files, chosen for the ZUNA study. That is too small and too
skewed to estimate the detector's window-level AUC with any precision — a
file-level bootstrap over it puts the 95 % interval at roughly [0.68, 0.92].

This script enumerates all EDFs under `sample_data/`, reads each one's `.csv_bi`
for its seizure count and duration, and writes a manifest with a `cohort` column
(`seizure` / `nonseizure`) so `evaluate_baseline.py` can report the two groups
separately. Including the background-only files matters: they are the only
specificity evidence available, and the seizure-enriched subset has none.

    python experiments/build_full_manifest.py --out artifacts/zuna_thesis/manifest_full.csv
"""
from __future__ import print_function

import argparse
import csv
import glob
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import eval_config as cfg  # noqa: E402
from gui.io.cache import cache_path_for, load_probability_file  # noqa: E402
from gui.io.csv_bi import read_csv_bi  # noqa: E402
from experiments.compare_zuna import read_reference_duration  # noqa: E402


def duration_for(edf, ref):
    """Recording duration in seconds, matching evaluate_baseline.load_file.

    Prefer the `.csv_bi` duration field; fall back to the probability cache's
    last window start plus one segment. Without the fallback the totals here
    undercount badly — a `.csv_bi` missing that field silently contributed
    zero, which made the manifest report 28.2 h for a corpus the evaluation
    scored as 41.9 h.
    """
    d = read_reference_duration(ref)
    if d:
        return float(d)
    cache = load_probability_file(cache_path_for(edf))
    if cache is not None and len(cache['window_starts']):
        return float(max(cache['window_starts'])) + float(cfg.SEGMENT_S)
    return 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='sample_data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cached-only', action='store_true',
                    help='Only include files that already have a probs cache.')
    args = ap.parse_args(argv)

    root = os.path.join(REPO, args.root)
    edfs = sorted(glob.glob(os.path.join(root, '**', '*.edf'), recursive=True))

    rows = []
    n_seiz_files = n_bg_files = 0
    total_dur = seiz_dur = 0.0
    n_seizures = 0
    for edf in edfs:
        cached = os.path.exists(cache_path_for(edf))
        if args.cached_only and not cached:
            continue
        ref = os.path.splitext(edf)[0] + '.csv_bi'
        events = [e for e in read_csv_bi(ref) if e.get('label') == 'seiz']
        duration = duration_for(edf, ref)
        s_dur = sum(float(e['stop']) - float(e['start']) for e in events)

        cohort = 'seizure' if events else 'nonseizure'
        if events:
            n_seiz_files += 1
            n_seizures += len(events)
            seiz_dur += s_dur
        else:
            n_bg_files += 1
        total_dur += duration

        rows.append({
            'stem': os.path.splitext(os.path.basename(edf))[0],
            'cohort': cohort,
            'duration_s': round(duration, 1),
            'seiz_count': len(events),
            'seiz_total_s': round(s_dur, 2),
            'cached': int(cached),
            'edf': os.path.relpath(edf, REPO).replace('\\', '/'),
        })

    out = os.path.abspath(args.out)
    parent = os.path.dirname(out)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(out, 'w') as f:
        w = csv.DictWriter(f, fieldnames=[
            'stem', 'cohort', 'duration_s', 'seiz_count', 'seiz_total_s',
            'cached', 'edf'], lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    n_cached = sum(r['cached'] for r in rows)
    print('wrote {}  ({} files)'.format(out, len(rows)))
    print('  seizure-bearing : {:>4}  ({} seizures, {:.2f} h ictal)'.format(
        n_seiz_files, n_seizures, seiz_dur / 3600.0))
    print('  background-only : {:>4}'.format(n_bg_files))
    print('  total duration  : {:.2f} h'.format(total_dur / 3600.0))
    print('  with probs cache: {:>4} / {}'.format(n_cached, len(rows)))
    if total_dur > 0 and seiz_dur > 0:
        print('  background:seizure duration ratio = {:.1f}:1'.format(
            (total_dur - seiz_dur) / seiz_dur))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
