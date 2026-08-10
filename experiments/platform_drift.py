"""Measure how much p(seizure) moves when the same recording is scored on a
different machine.

    on Windows:   python experiments/platform_drift.py --export artifacts/drift_ref_windows.npz
    on the Mac:   python experiments/platform_drift.py --compare artifacts/drift_ref_windows.npz

Why this exists
---------------
`docs/portability.md` reports a median drift of 0.0001 and a maximum of 0.136,
and `docs/known_issues.md` §1 leans on those numbers to argue the two stacks are
interchangeable. They were measured over **74 windows from a single machine**,
comparing Python 3.6 against Python 3.11 on that same Windows box. That is a
*stack* comparison, not a *platform* comparison: same CPU, same BLAS, same
memory layout.

The first Windows-vs-macOS measurement moved a peak by 0.0735 — inside the
published envelope, but 700× the published median, and drawn from one
recording. Two different pairs are being described by one number. This script
replaces the assumption with a distribution.

What it does NOT establish
--------------------------
Nothing here says which machine is right. The ICA does not converge
(`known_issues.md` §4), so neither decomposition is the true one and the drift
is not an error to be fixed — it is the width of an operating point. The only
question worth asking is whether it is wide enough to change a decision a
reviewer would see.

Alignment
---------
Two probability arrays are only comparable window-for-window. An earlier
version of the ZUNA comparison silently compared arms computed over different
window sets and produced a confident wrong answer, so this refuses to compare
anything it cannot align exactly, and reports what it dropped.
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import platform
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _manifest_rows():
    p = os.path.join(REPO, 'artifacts', 'zuna_thesis', 'manifest_full.csv')
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _local_edf(stem):
    """Absolute path to a recording, or None. Tries the manifest, then a walk.

    The manifest stores repository-relative POSIX paths. On a machine where the
    corpus lives elsewhere (the Mac keeps it under ~/Downloads), --corpus
    supplies the root to search instead.
    """
    for row in _manifest_rows():
        if row['stem'] == stem:
            p = os.path.join(REPO, row['edf'])
            if os.path.exists(p):
                return os.path.abspath(p)
    return None


def _walk_index(root):
    """stem -> sorted list of every EDF with that stem under a directory.

    A list, not a single path, because the stem is not unique. TUH records some
    sessions under two montages, so
    `.../s010_2015_08_27/01_tcp_ar/aaaaaqvx_s010_t004.edf` and
    `.../s010_2015_08_27/03_tcp_ar_a/aaaaaqvx_s010_t004.edf` are different
    recordings with the same stem — five such collisions are in
    manifest_full.csv, and two of them differ in duration (600 s against 601 s).
    Keeping the first match silently compared one montage's cached
    probabilities against the other montage's signal and reported the result as
    platform drift.
    """
    index = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith('.edf'):
                index.setdefault(os.path.splitext(fn)[0], []).append(
                    os.path.join(dirpath, fn))
    return {k: sorted(v) for k, v in index.items()}


# ---------------------------------------------------------------- export


def export(out_path):
    """Bundle every local GUI cache into one file for another machine."""
    from gui.io.cache import load_probs

    rows = _manifest_rows()
    stems, starts, probs, skips, offsets, envs = [], [], [], [], [0], []
    edfs = []
    inexact = []
    for row in rows:
        stem = row['stem']
        # This row's own path, not a lookup by stem. _local_edf returns the
        # FIRST manifest row carrying the stem, and five stems appear twice —
        # the same session under the 01_tcp_ar and 03_tcp_ar_a montages. Going
        # through it exported one montage's cache twice under one stem and
        # never exported the other recording at all, which is why the two
        # entries for aaaaaqvx_s010_t004 were numerically identical.
        edf = os.path.join(REPO, row['edf'])
        if not os.path.exists(edf):
            continue
        edf = os.path.abspath(edf)
        try:
            rec = load_probs(edf)
        except Exception:
            rec = None
        if not rec:
            continue
        p = np.asarray(rec['probs'], dtype=np.float32)
        s = np.asarray(rec['window_starts'], dtype=np.int64)
        sk = rec.get('skip_code')
        sk = (np.zeros(len(p), dtype=np.int8) if sk is None
              else np.asarray(sk, dtype=np.int8))
        if not rec.get('skip_code_is_exact', False):
            inexact.append(stem)
        stems.append(stem)
        edfs.append(row['edf'])
        starts.append(s)
        probs.append(p)
        skips.append(sk)
        offsets.append(offsets[-1] + len(p))
        meta = rec.get('meta') or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        envs.append(meta.get('env', {}))

    if not stems:
        print('no local caches found — nothing to export')
        return 1

    # One env per cache, but they should all agree; record the set so a
    # comparison against a mixed-provenance export is visibly suspect.
    uniq = {json.dumps(e, sort_keys=True) for e in envs}
    np.savez_compressed(
        out_path,
        stems=np.array(stems),
        edfs=np.array(edfs),
        offsets=np.array(offsets, dtype=np.int64),
        window_starts=np.concatenate(starts),
        probs=np.concatenate(probs),
        skip_code=np.concatenate(skips),
        provenance=json.dumps({
            'exported_on': platform.node(),
            'platform': platform.system(),
            'python': sys.version.split()[0],
            'cache_envs': [json.loads(u) for u in sorted(uniq)],
            'n_recordings': len(stems),
            'n_windows': int(offsets[-1]),
            'skip_code_inexact_for': inexact,
        }),
    )
    size = os.path.getsize(out_path)
    print('exported {} recordings, {} windows -> {} ({:.0f} KB)'.format(
        len(stems), offsets[-1], out_path, size / 1024.0))
    if len(uniq) > 1:
        print('WARNING: caches came from {} different environments; this '
              'export mixes provenance'.format(len(uniq)))
    for u in sorted(uniq):
        print('   env: {}'.format(u))
    if inexact:
        print('note: {} caches predate the skip_code array; their skipped '
              'windows are inferred'.format(len(inexact)))
    return 0


# --------------------------------------------------------------- compare


def compare(ref_path, corpus=None, limit=None, threshold=0.5, out_json=None,
            shortest_first=False):
    """Re-score locally and report the drift distribution against the export."""
    from gui.io.edf import load_edf_19ch
    from gui.io.infer import compute_probs_from_data

    z = np.load(ref_path, allow_pickle=False)
    stems = [str(s) for s in z['stems']]
    offsets = z['offsets']
    ref_starts, ref_probs, ref_skip = (z['window_starts'], z['probs'],
                                       z['skip_code'])
    prov = json.loads(str(z['provenance']))
    print('reference: {} recordings, {} windows, exported on {} ({})'.format(
        prov['n_recordings'], prov['n_windows'], prov['exported_on'],
        prov['platform']))
    for e in prov.get('cache_envs', []):
        print('   env: {}'.format(json.dumps(e, sort_keys=True)))
    print('this machine: {} {} python {}'.format(
        platform.system(), platform.machine(), sys.version.split()[0]))
    print()

    index = _walk_index(corpus) if corpus else {}

    deltas = []
    per_rec = []
    missing, misaligned, ambiguous = [], [], []

    # Iterate over positions in `stems`, never over a reordered copy of it.
    # `offsets` is indexed by a recording's position in the export, so pairing
    # it with a position in some other sequence hands each recording another
    # recording's reference window — silently, whenever the two happen to have
    # the same window count. That is the ZUNA mistake this file exists to avoid,
    # and slicing a prefix only avoids it by accident.
    order = list(range(len(stems)))
    if shortest_first:
        # Coverage of the drift *distribution* comes from many recordings, not
        # from many windows in a few. Manifest order opens with a 3337 s
        # recording (555 windows), so it buys the fewest recordings per hour of
        # any possible ordering.
        dur = {r['stem']: float(r['duration_s'] or 0) for r in _manifest_rows()}
        order.sort(key=lambda j: dur.get(stems[j], float('inf')))
    if limit is not None:
        order = order[:limit]

    # A stem that is not unique cannot be resolved to one recording, and the
    # export records stems only. Refusing is the sole correct option: guessing
    # compares one montage's cached probabilities against the other montage's
    # signal, and the grid check waves that through whenever the two happen to
    # have the same window count — which is usual, since the montages cover the
    # same minutes. Before this, the five colliding stems supplied 22 of 72
    # apparent decision changes.
    stem_counts = {}
    for s in stems:
        stem_counts[s] = stem_counts.get(s, 0) + 1

    for i in order:
        stem = stems[i]
        found = index.get(stem)
        if found is None:
            one = _local_edf(stem)
            found = [one] if one else []
        if stem_counts[stem] > 1 or len(found) > 1:
            ambiguous.append(stem)
            continue
        edf = found[0] if found else None
        if edf is None:
            missing.append(stem)
            continue
        lo, hi = offsets[i], offsets[i + 1]
        rs, rp, rk = ref_starts[lo:hi], ref_probs[lo:hi], ref_skip[lo:hi]
        try:
            data, fs, _dur = load_edf_19ch(edf)
            starts, probs, skip = compute_probs_from_data(data, fs)
        except Exception as ex:                        # noqa: BLE001
            missing.append('{} (unreadable: {})'.format(stem, ex))
            continue
        starts = np.asarray(starts, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float32)
        skip = np.asarray(skip, dtype=np.int8)

        # Refuse to compare misaligned grids. Intersecting would quietly change
        # what the number means, which is precisely the ZUNA mistake.
        if starts.shape != rs.shape or not np.array_equal(starts, rs):
            misaligned.append(stem)
            continue

        both = (skip == 0) & (rk == 0)
        if not both.any():
            continue
        d = np.abs(probs[both] - rp[both])
        flipped = int(((probs[both] >= threshold) !=
                       (rp[both] >= threshold)).sum())
        deltas.append(d)
        per_rec.append({
            'stem': stem, 'n': int(both.sum()),
            'median': float(np.median(d)), 'max': float(d.max()),
            'decisions_changed': flipped,
            'ref_max': float(rp[both].max()),
            'local_max': float(probs[both].max()),
        })
        print('  {:<22} n={:<4} median {:.6f}  max {:.4f}  flips {}'.format(
            stem, int(both.sum()), float(np.median(d)), float(d.max()),
            flipped))

    if not deltas:
        print('\nnothing could be compared.')
        return 1

    all_d = np.concatenate(deltas)
    flips = sum(r['decisions_changed'] for r in per_rec)
    n_win = int(all_d.size)
    summary = {
        'n_recordings': len(per_rec),
        'n_windows': n_win,
        'median': float(np.median(all_d)),
        'p95': float(np.percentile(all_d, 95)),
        'p99': float(np.percentile(all_d, 99)),
        'max': float(all_d.max()),
        'decisions_changed': flips,
        'threshold': threshold,
        'reference_platform': prov['platform'],
        'local_platform': '{} {}'.format(platform.system(),
                                         platform.machine()),
        'not_found': len(missing),
        'misaligned': misaligned,
        'ambiguous_stems': sorted(set(ambiguous)),
    }
    print()
    print('=' * 62)
    print('recordings compared : {}'.format(summary['n_recordings']))
    print('windows compared    : {}'.format(n_win))
    print('median |delta|      : {:.6f}'.format(summary['median']))
    print('p95    |delta|      : {:.6f}'.format(summary['p95']))
    print('p99    |delta|      : {:.6f}'.format(summary['p99']))
    print('max    |delta|      : {:.6f}'.format(summary['max']))
    print('decisions changed   : {} of {} windows at threshold {}'.format(
        flips, n_win, threshold))
    if missing:
        print('not found locally   : {}'.format(len(missing)))
    if misaligned:
        print('MISALIGNED (dropped): {} -> {}'.format(
            len(misaligned), ', '.join(misaligned[:5])))
    if ambiguous:
        print('AMBIGUOUS  (dropped): {} -> {}'.format(
            len(set(ambiguous)), ', '.join(sorted(set(ambiguous))[:5])))
        print('                      stem maps to more than one recording; '
              'the export records stems only, so it cannot be resolved')
    print('=' * 62)

    if out_json:
        with open(out_json, 'w') as f:
            json.dump({'summary': summary, 'per_recording': per_rec,
                       'reference_provenance': prov}, f, indent=2)
        print('wrote {}'.format(out_json))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', metavar='OUT.npz')
    ap.add_argument('--compare', metavar='REF.npz')
    ap.add_argument('--corpus', help='root to search for EDFs, if the corpus '
                                     'is not under the repository')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--shortest-first', action='store_true',
                    dest='shortest_first',
                    help='compare the shortest recordings first, to cover more '
                         'recordings per hour of compute')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--json', dest='out_json')
    a = ap.parse_args(argv)
    if a.export:
        return export(a.export)
    if a.compare:
        return compare(a.compare, corpus=a.corpus, limit=a.limit,
                       threshold=a.threshold, out_json=a.out_json,
                       shortest_first=a.shortest_first)
    ap.error('give --export or --compare')


if __name__ == '__main__':
    raise SystemExit(main())
