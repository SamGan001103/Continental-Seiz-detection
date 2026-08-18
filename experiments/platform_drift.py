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


def _sha256(path, chunk=1 << 20):
    """Fingerprint of a recording, so two machines can prove they scored the same one."""
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


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


def _events(window_starts, probs, threshold, duration_s):
    """The events a reviewer would actually be shown, for one probability array.

    A window is an internal unit. What a reviewer steps through is a *shaped
    event*: per-second averaging, threshold, merge runs closer than
    MAX_MERGE_GAP_S, discard runs shorter than MIN_EVENT_DURATION_S. A window
    that flips at the threshold may vanish inside an existing event, extend
    one, create one, or be discarded for being too short - so the window count
    and the event count answer different questions, and only the second is a
    claim about what somebody sees.

    Reads eval_config for the shaping parameters, exactly as gui/events.py
    does, so this measures the shipped decision stage and not a variant of it.
    """
    from gui.postprocess import events_from_probs
    from gui.io.infer import SEGMENT_S
    import eval_config as _cfg
    return events_from_probs(
        window_starts, probs, threshold, SEGMENT_S,
        duration_s=duration_s,
        average=_cfg.USE_PER_SECOND_AVERAGING,
        min_duration_s=(_cfg.MIN_EVENT_DURATION_S
                        if _cfg.USE_SOURCE_POSTPROCESSING else 0.0),
        max_gap_s=(_cfg.MAX_MERGE_GAP_S
                   if _cfg.USE_SOURCE_POSTPROCESSING else 0.0))


def _compare_events(ref_ev, loc_ev):
    """Match two event lists by overlap in time.

    Overlap of any length counts as the same event. A stricter criterion (say
    50 % reciprocal overlap) would report boundary movement as a lost event
    plus a gained one, which overstates what changed: a reviewer looking at a
    seizure whose marked end moved by two seconds sees one event, moved.

    Matching is greedy over reference events in time order. Events are few per
    recording and non-overlapping by construction, so greedy is exact here.

    Returns (matched, lost, gained, max_boundary_shift_s).
    """
    used = set()
    matched = 0
    shift = 0.0
    for rs, re_, _rp in ref_ev:
        for j, (ls, le, _lp) in enumerate(loc_ev):
            if j in used:
                continue
            if ls < re_ and rs < le:            # any temporal overlap
                used.add(j)
                matched += 1
                shift = max(shift, abs(ls - rs), abs(le - re_))
                break
    return matched, len(ref_ev) - matched, len(loc_ev) - len(used), shift


# ---------------------------------------------------------------- export


def export(out_path):
    """Bundle every local GUI cache into one file for another machine."""
    from gui.io.cache import load_probs

    rows = _manifest_rows()
    stems, starts, probs, skips, offsets, envs = [], [], [], [], [0], []
    edfs = []
    edf_sha = []
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
        # The recording's own fingerprint, not just its name.
        #
        # This comparison assumes both machines score the SAME signal, and
        # nothing was checking it. The two corpora in use are separate
        # downloads of TUSZ: on this machine the five stems that appear twice
        # are byte-identical copies, and on the other they are not - one of
        # them produced 22 window flips, which identical files cannot do. So
        # the corpora demonstrably disagree somewhere, and every disagreement
        # is indistinguishable from platform drift once it reaches the
        # probabilities. Window-grid alignment cannot catch it: two different
        # recordings of the same duration have the same window count.
        #
        # Hashing 300-odd EDFs costs a few minutes once. Reporting a data
        # difference as a portability finding would cost considerably more.
        edf_sha.append(_sha256(edf))
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
        edf_sha256=np.array(edf_sha),
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
    missing, misaligned, ambiguous, different_data = [], [], [], []
    ref_sha = ([str(x) for x in z['edf_sha256']]
               if 'edf_sha256' in z.files else None)
    if ref_sha is None:
        print('NOTE: this reference predates recording fingerprints, so it '
              'cannot prove both machines scored the same signal.')
        print()
    ev_totals = {'ref': 0, 'local': 0, 'matched': 0, 'lost': 0, 'gained': 0,
                 'recordings_changed': 0, 'max_shift_s': 0.0}

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
        # Prove it is the same recording before attributing anything to the
        # platform. Two separate downloads of TUSZ are in use and they are known
        # to disagree on at least one file; a differing signal produces a
        # differing probability, and nothing downstream can tell that apart from
        # drift. Same duration means same window count, so the grid check is
        # blind to it.
        if ref_sha is not None:
            try:
                if _sha256(edf) != ref_sha[i]:
                    different_data.append(stem)
                    continue
            except Exception:
                pass

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

        # Events, on the full arrays rather than the compared subset: shaping
        # is sensitive to gaps, so dropping windows out of the middle would
        # split events that the GUI would keep whole. Windows either side
        # marked unscored are zeroed on both arms, because an unscored window
        # cannot contribute a detection to a reviewer either way.
        rp_full = np.where(rk == 0, ref_probs[lo:hi], 0.0)
        lp_full = np.where(skip == 0, probs, 0.0)
        ref_ev = _events(rs, rp_full, threshold, _dur)
        loc_ev = _events(starts, lp_full, threshold, _dur)
        ev_match, ev_lost, ev_gained, ev_shift = _compare_events(ref_ev,
                                                                 loc_ev)
        ev_totals['ref'] += len(ref_ev)
        ev_totals['local'] += len(loc_ev)
        ev_totals['matched'] += ev_match
        ev_totals['lost'] += ev_lost
        ev_totals['gained'] += ev_gained
        if ev_lost or ev_gained:
            ev_totals['recordings_changed'] += 1
        ev_totals['max_shift_s'] = max(ev_totals['max_shift_s'], ev_shift)

        per_rec.append({
            'stem': stem, 'n': int(both.sum()),
            'median': float(np.median(d)), 'max': float(d.max()),
            'decisions_changed': flipped,
            'ref_max': float(rp[both].max()),
            'local_max': float(probs[both].max()),
            'events_ref': len(ref_ev), 'events_local': len(loc_ev),
            'events_matched': ev_match, 'events_lost': ev_lost,
            'events_gained': ev_gained,
            'event_max_shift_s': round(ev_shift, 3),
        })
        # Checkpoint after every recording. This run takes over an hour and a
        # segfault in a native library - which happened at recording 88 of 120 -
        # otherwise discards every window-level percentile computed so far. The
        # per-recording lines survive in stdout, but the raw deltas the
        # percentiles need do not. Writing a few thousand floats each time costs
        # nothing against the cost of re-scoring the corpus.
        if out_json:
            try:
                with open(out_json + '.partial', 'w') as _f:
                    json.dump({'complete': False,
                               'per_recording': per_rec,
                               'events_so_far': dict(ev_totals),
                               'deltas': [float(x) for a in deltas for x in a]},
                              _f)
            except Exception:                          # noqa: BLE001
                pass

        print('  {:<22} n={:<4} median {:.6f}  max {:.4f}  flips {:<3} '
              'events {}->{}{}'.format(
                  stem, int(both.sum()), float(np.median(d)), float(d.max()),
                  flipped, len(ref_ev), len(loc_ev),
                  '  (-{} +{})'.format(ev_lost, ev_gained)
                  if (ev_lost or ev_gained) else ''))

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
        # Not decoration. Without these a comparison cannot be interpreted
        # later, and two studies were written up wrong for exactly that reason.
        'local_numerics': _local_numerics_stack(),
        'not_found': len(missing),
        'misaligned': misaligned,
        'ambiguous_stems': sorted(set(ambiguous)),
        'different_data_stems': sorted(set(different_data)),
        'events': dict(ev_totals),
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
    # The window figure is internal. These are the events a reviewer is shown.
    print('-' * 62)
    print('events (reference)  : {}'.format(ev_totals['ref']))
    print('events (this machine): {}'.format(ev_totals['local']))
    print('  matched           : {}'.format(ev_totals['matched']))
    print('  lost              : {}   (in the reference, absent here)'.format(
        ev_totals['lost']))
    print('  gained            : {}   (here, absent in the reference)'.format(
        ev_totals['gained']))
    print('  recordings with any event change: {} of {}'.format(
        ev_totals['recordings_changed'], summary['n_recordings']))
    print('  largest boundary shift on a matched event: {:.1f} s'.format(
        ev_totals['max_shift_s']))
    if missing:
        print('not found locally   : {}'.format(len(missing)))
    if misaligned:
        print('MISALIGNED (dropped): {} -> {}'.format(
            len(misaligned), ', '.join(misaligned[:5])))
    if different_data:
        print('DIFFERENT RECORDING (dropped): {} -> {}'.format(
            len(different_data), ', '.join(different_data[:5])))
        print('   the local file is not the one the reference was built from;')
        print('   comparing them would report a data difference as drift.')
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


def _local_numerics_stack():
    """What actually decides the numbers, recorded so a run can be interpreted.

    This function exists because its absence cost two studies. The summary block
    below stamped host, platform, python, numpy, scipy and MNE -- everything
    except the two libraries that do the arithmetic. Both arms of the drift study
    behind docs/portability.md therefore ran Keras 3 against a Keras 2 reference
    and stored nothing that said so, and the difference was written up as
    platform drift. It is a bigger effect than the platform: on one recording the
    Keras version moves p(seizure) from 0.5990 to 0.4602, across a decision
    threshold, while two machines on the same Keras agree to ~1e-9.

    Keras is asked for by the name `tf.keras` resolves to, not by
    `keras.__version__`. The standalone Keras 3 package reports 3.x even when
    tf.keras is correctly bound to Keras 2, so the version string answers a
    different question than the one being asked.
    """
    info = {}
    try:
        import tensorflow as tf
        info['tensorflow'] = str(tf.__version__)
        try:
            name = tf.keras.__name__
            info['tf_keras_module'] = name
            info['keras_major'] = 2 if name.startswith('tf_keras') else 3
        except Exception as ex:                                 # noqa: BLE001
            info['tf_keras_module'] = 'unavailable: {}'.format(ex)
    except Exception as ex:                                     # noqa: BLE001
        info['tensorflow'] = 'unavailable: {}'.format(ex)
    info['TF_USE_LEGACY_KERAS'] = os.environ.get('TF_USE_LEGACY_KERAS')
    info['python'] = platform.python_version()
    return info


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
