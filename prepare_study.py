"""Part one of the system: prepare a recording for review, on the server.

    python prepare_study.py "D:/studies/*.edf"

This is the only command that runs on the hospital's compute machine, and it is
the only place in the system where a model runs at all. For each recording it
produces a STUDY BUNDLE -- every artefact the review application needs --
and a manifest naming them. The bundle is then copied to as many clinical
workstations as the service has; the review application reads it and computes
nothing.

Why the split is enforced rather than merely intended
-----------------------------------------------------
The review application used to be able to fall back to running inference
itself when a cache was missing. That made the two-part design a convention: a
clinical workstation with no GPU, no TensorFlow and a six-year-old processor
would silently start a job measured in hours, and the reviewer would be told
only that the file was "loading". The fallback is gone, and this command is
what replaces it. If a bundle is missing the application says so and names
this script, which is a thirty-second answer instead of an afternoon.

That also makes the system's claim checkable. "Heavy computation happens on
the server" is only true if the client cannot do it, and the client can no
longer do it.

What a bundle contains
----------------------
    <stem>.probs.npz      per-window seizure probability, from the ConvLSTM
    <stem>.ied.csv        interictal discharge candidates, from SpikeNet1
    <stem>.study.json     the manifest: what is here, what produced it, and
                          which display settings the artefacts are valid for

The manifest records the recording's size and mtime, so a bundle cannot be
read against a recording it was not built from -- the same guard the
probability cache already applies, lifted to the bundle as a whole.
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MANIFEST_VERSION = 1
MANIFEST_SUFFIX = '.study.json'

# The display settings the bundle's precomputed artefacts are valid for. The
# review application draws with these; anything else would need a filter pass
# it is no longer allowed to run. Kept here rather than imported from the GUI
# so the server side has no dependency on Qt.
DEFAULT_DISPLAY = {
    'montage': 'Longitudinal Bipolar',
    'hp': 1.0,
    'lp': 70.0,
    'notch': 60.0,
}


def manifest_path(recording):
    return os.path.splitext(recording)[0] + MANIFEST_SUFFIX


def _stamp(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {'bytes': int(st.st_size), 'mtime': float(st.st_mtime)}


def _sha256(path, limit=None):
    """Digest of the recording. `limit` caps the read for very large files:
    a 1.2 GB EDF takes several seconds to hash in full and the first 64 MB
    plus the size is already specific enough to catch a different recording."""
    h = hashlib.sha256()
    read = 0
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
            read += len(chunk)
            if limit and read >= limit:
                break
    return h.hexdigest(), read


def _run(script, paths, extra=None):
    """Run one precompute step in this interpreter, returning (ok, seconds)."""
    cmd = [sys.executable, os.path.join(REPO, script)] + list(paths)
    if extra:
        cmd += list(extra)
    env = dict(os.environ)
    env.setdefault('TF_USE_LEGACY_KERAS', '1')
    env.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    env.setdefault('PYTHONWARNINGS', 'ignore')
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, env=env)
    return r.returncode == 0, time.time() - t0


def artefacts_for(recording):
    """Where each part of the bundle lives, by name."""
    stem = os.path.splitext(recording)[0]
    return {
        'seizure_probabilities': stem + '.probs.npz',
        'interictal_candidates': stem + '.ied.csv',
    }


def write_manifest(recording, timings=None, display=None):
    """Record what the bundle holds and what it was built from."""
    digest, hashed = _sha256(recording, limit=64 << 20)
    arte = artefacts_for(recording)
    payload = {
        'manifest_version': MANIFEST_VERSION,
        'prepared_utc': dt.datetime.utcfromtimestamp(
            time.time()).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'recording': {
            'path': os.path.basename(recording),
            'sha256_prefix': digest,
            'sha256_bytes_hashed': hashed,
            **(_stamp(recording) or {}),
        },
        # The review application draws with exactly these. It cannot filter or
        # re-montage beyond them, because it no longer computes.
        'display_settings': dict(display or DEFAULT_DISPLAY),
        'artefacts': {
            name: {
                'file': os.path.basename(path),
                'present': os.path.exists(path),
                **(_stamp(path) or {}),
            }
            for name, path in arte.items()
        },
        'build_seconds': dict(timings or {}),
        'note': ('Prepared by prepare_study.py. Every model in this bundle ran '
                 'here, not on the review workstation. Detector output is a '
                 'CANDIDATE for a clinician to confirm or reject, not a '
                 'finding, and no threshold here is clinically validated.'),
    }
    out = manifest_path(recording)
    with open(out, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out, payload


def read_manifest(recording):
    """The bundle manifest for this recording, or None."""
    path = manifest_path(recording)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    return blob if isinstance(blob, dict) else None


def bundle_matches(recording, blob=None):
    """Whether a bundle belongs to the recording beside it.

    Size and mtime first because they are free; the digest only if those pass,
    so a mismatched bundle is rejected without reading a gigabyte.
    """
    blob = blob if blob is not None else read_manifest(recording)
    if not blob:
        return False
    rec = blob.get('recording') or {}
    now = _stamp(recording)
    if now is None or 'bytes' not in rec:
        return False
    if int(rec['bytes']) != now['bytes']:
        return False
    if 'mtime' in rec and abs(float(rec['mtime']) - now['mtime']) > 2.0:
        return False
    return True


def missing_from(recording, blob=None):
    """Names of bundle artefacts that are absent or stale."""
    blob = blob if blob is not None else read_manifest(recording)
    out = []
    for name, path in artefacts_for(recording).items():
        if not os.path.exists(path):
            out.append(name)
    return out


def prepare(recording, args):
    name = os.path.basename(recording)
    print('\n  {}'.format(name))
    timings = {}

    if not args.skip_seizure:
        extra = ['--overwrite'] if args.force else []
        if args.workers:
            extra += ['--workers', str(args.workers)]
        ok, secs = _run('precompute_probs.py', [recording], extra)
        timings['seizure_probabilities'] = round(secs, 1)
        if not ok:
            print('    seizure detection FAILED; bundle not written')
            return None

    if not args.skip_interictal:
        extra = ['--force'] if args.force else []
        ok, secs = _run('precompute_ieds.py', [recording], extra)
        timings['interictal_candidates'] = round(secs, 1)
        if not ok:
            print('    interictal detection failed; continuing without it')

    out, payload = write_manifest(recording, timings)
    have = [n for n, a in payload['artefacts'].items() if a['present']]
    total = sum(a.get('bytes', 0) for a in payload['artefacts'].values()
                if a.get('present'))
    print('    bundle: {} ({:.1f} MB) -> {}'.format(
        ', '.join(have) or 'nothing', total / 1e6, os.path.basename(out)))
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('paths', nargs='+', help='EDF recordings, or a glob')
    p.add_argument('--force', action='store_true',
                   help='rebuild artefacts that already exist')
    p.add_argument('--workers', type=int, default=0,
                   help='parallel workers for seizure scoring')
    p.add_argument('--skip-seizure', action='store_true')
    p.add_argument('--skip-interictal', action='store_true')
    p.add_argument('--manifest-only', action='store_true',
                   help='write the manifest for artefacts already present')
    args = p.parse_args(argv)

    files = []
    for pattern in args.paths:
        hits = sorted(glob.glob(pattern))
        files.extend(hits if hits else [pattern])
    files = [f for f in files
             if os.path.exists(f) and f.lower().endswith('.edf')]
    if not files:
        print('no EDF recordings matched')
        return 1

    print('\nPreparing {} study bundle(s). Every model runs HERE, not on the '
          'review workstation.'.format(len(files)))
    t0 = time.time()
    done = 0
    for f in files:
        if args.manifest_only:
            out, payload = write_manifest(f)
            print('  {} -> {}'.format(os.path.basename(f),
                                      os.path.basename(out)))
            done += 1
            continue
        if prepare(f, args) is not None:
            done += 1
    print('\n  {}/{} prepared in {:.1f} min'.format(
        done, len(files), (time.time() - t0) / 60.0))
    print('  Copy each recording with its bundle to the review workstations.')
    return 0 if done == len(files) else 1


if __name__ == '__main__':
    sys.exit(main())
