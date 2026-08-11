"""Do two machines running the SAME modern stack agree, window and event?

`platform_drift.py` compares a machine against the Windows legacy caches, which
answers "has the stack changed the answer". This answers a different question:
with the library versions held identical and only the CPU differing, do two
builds of the shipped application propose the same events?

That is the question behind shipping Windows and macOS as one product, and it
cannot be answered from two separate runs against the legacy reference — those
tell you how each differs from a third thing, not from each other.

Usage, on each machine:

    python experiments/modern_stack_agreement.py --score windows --limit 40
    python experiments/modern_stack_agreement.py --score macos   --limit 40

then, once both files exist:

    python experiments/modern_stack_agreement.py --compare windows macos
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

OUTDIR = os.path.join(REPO, 'artifacts', 'zuna_thesis')


def _rows():
    p = os.path.join(REPO, 'artifacts', 'zuna_thesis', 'manifest_full.csv')
    with open(p) as f:
        return list(csv.DictReader(f))


def score(tag, limit):
    """Score the shortest `limit` recordings and save every window."""
    from gui.io.edf import load_edf_19ch
    from gui.io.infer import compute_probs_from_data
    import tensorflow as tf

    rows = [r for r in _rows() if os.path.exists(os.path.join(REPO, r['edf']))]
    # Shortest first: coverage of the distribution comes from many recordings.
    # Ambiguous stems are dropped, as in platform_drift - five TUH sessions
    # appear twice under two montages and cannot be resolved by stem.
    seen = {}
    for r in rows:
        seen[r['stem']] = seen.get(r['stem'], 0) + 1
    rows = [r for r in rows if seen[r['stem']] == 1]
    rows.sort(key=lambda r: float(r['duration_s'] or 0))
    rows = rows[:limit]

    stems, starts, probs, skips, offs = [], [], [], [], [0]
    for i, r in enumerate(rows):
        edf = os.path.join(REPO, r['edf'])
        try:
            data, fs, _dur = load_edf_19ch(edf)
            st, pr, sk = compute_probs_from_data(data, fs)
        except Exception as ex:                          # noqa: BLE001
            print('  skip %s (%s)' % (r['stem'], ex))
            continue
        stems.append(r['stem'])
        starts.append(np.asarray(st, dtype=np.int64))
        probs.append(np.asarray(pr, dtype=np.float64))   # float64: the point is
        skips.append(np.asarray(sk, dtype=np.int8))      # exact comparison
        offs.append(offs[-1] + len(pr))
        print('  %3d/%d  %-22s %d windows' % (i + 1, len(rows), r['stem'],
                                              len(pr)))

    out = os.path.join(OUTDIR, 'agree_%s.npz' % tag)
    np.savez_compressed(
        out,
        stems=np.array(stems),
        offsets=np.array(offs, dtype=np.int64),
        window_starts=np.concatenate(starts),
        probs=np.concatenate(probs),
        skip_code=np.concatenate(skips),
        provenance=json.dumps({
            'tag': tag, 'host': platform.node(),
            'platform': platform.system(), 'machine': platform.machine(),
            'python': sys.version.split()[0],
            'tensorflow': tf.__version__,
            'tf_keras': tf.keras.__name__,
            'legacy_keras': os.environ.get('TF_USE_LEGACY_KERAS'),
            'numpy': np.__version__,
        }))
    print('wrote %s  (%d recordings, %d windows)' % (out, len(stems), offs[-1]))
    return 0


def compare(tag_a, tag_b, threshold=0.5):
    from experiments.platform_drift import _events, _compare_events

    za = np.load(os.path.join(OUTDIR, 'agree_%s.npz' % tag_a), allow_pickle=False)
    zb = np.load(os.path.join(OUTDIR, 'agree_%s.npz' % tag_b), allow_pickle=False)
    pa, pb = json.loads(str(za['provenance'])), json.loads(str(zb['provenance']))
    for p in (pa, pb):
        print('%-8s %-8s %-8s python %-8s tf %-8s keras=%s numpy %s' % (
            p['tag'], p['platform'], p['machine'], p['python'],
            p['tensorflow'], p.get('legacy_keras'), p['numpy']))
    print()

    sa = {s: i for i, s in enumerate([str(x) for x in za['stems']])}
    sb = {s: i for i, s in enumerate([str(x) for x in zb['stems']])}
    shared = [s for s in sa if s in sb]

    deltas, ev = [], {'a': 0, 'b': 0, 'matched': 0, 'lost': 0, 'gained': 0,
                      'recordings_changed': 0}
    flips = 0
    worst = []
    for stem in shared:
        ia, ib = sa[stem], sb[stem]
        A = za['probs'][za['offsets'][ia]:za['offsets'][ia + 1]]
        B = zb['probs'][zb['offsets'][ib]:zb['offsets'][ib + 1]]
        ska = za['skip_code'][za['offsets'][ia]:za['offsets'][ia + 1]]
        skb = zb['skip_code'][zb['offsets'][ib]:zb['offsets'][ib + 1]]
        st = za['window_starts'][za['offsets'][ia]:za['offsets'][ia + 1]]
        if A.shape != B.shape:
            continue
        both = (ska == 0) & (skb == 0)
        if not both.any():
            continue
        d = np.abs(A[both] - B[both])
        deltas.append(d)
        flips += int(((A[both] >= threshold) != (B[both] >= threshold)).sum())
        dur = float(st[-1] + 12)
        ea = _events(st, np.where(ska == 0, A, 0.0), threshold, dur)
        eb = _events(st, np.where(skb == 0, B, 0.0), threshold, dur)
        m, lost, gained, _sh = _compare_events(ea, eb)
        ev['a'] += len(ea); ev['b'] += len(eb); ev['matched'] += m
        ev['lost'] += lost; ev['gained'] += gained
        if lost or gained:
            ev['recordings_changed'] += 1
        if d.size:
            worst.append((float(d.max()), stem))

    x = np.concatenate(deltas)
    print('recordings compared : %d' % len(shared))
    print('windows compared    : %d' % x.size)
    print('bitwise identical   : %d of %d  (%.1f %%)' % (
        int((x == 0).sum()), x.size, 100.0 * (x == 0).sum() / x.size))
    print('median |delta|      : %.3e' % float(np.median(x)))
    print('p99    |delta|      : %.3e' % float(np.percentile(x, 99)))
    print('max    |delta|      : %.3e' % float(x.max()))
    print('windows changing a decision : %d of %d' % (flips, x.size))
    print('-' * 58)
    print('events %-8s : %d' % (tag_a, ev['a']))
    print('events %-8s : %d' % (tag_b, ev['b']))
    print('  matched           : %d' % ev['matched'])
    print('  present in %s only : %d' % (tag_a, ev['lost']))
    print('  present in %s only : %d' % (tag_b, ev['gained']))
    print('  recordings whose event list differs : %d of %d' % (
        ev['recordings_changed'], len(shared)))
    worst.sort(reverse=True)
    print()
    print('largest per-recording differences:')
    for v, s in worst[:5]:
        print('   %-22s %.4f' % (s, v))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--score', metavar='TAG')
    ap.add_argument('--compare', nargs=2, metavar=('TAG_A', 'TAG_B'))
    ap.add_argument('--limit', type=int, default=40)
    ap.add_argument('--threshold', type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.score:
        return score(a.score, a.limit)
    if a.compare:
        return compare(a.compare[0], a.compare[1], a.threshold)
    ap.error('give --score TAG or --compare A B')


if __name__ == '__main__':
    raise SystemExit(main())
