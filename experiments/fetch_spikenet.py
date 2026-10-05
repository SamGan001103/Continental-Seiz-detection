"""Fetch the SpikeNet1 weights into models/spikenet1/.

The weights are not vendored into this repository on purpose. The upstream
project states two different licences in two different files -- LICENSE.txt
says CC BY-NC 4.0, README.md says CC BY-ND 4.0 -- and until that is clarified
with the authors the safe course is to download them on the user's own
machine rather than redistribute them in ours.

    python experiments/fetch_spikenet.py

Downloads 1.3 MB from github.com/bdsp-core/SpikeNet1, checks the weights
against a known digest, and writes a NOTICE file recording the citation and
the licence ambiguity next to them.
"""
import hashlib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BASE = 'https://raw.githubusercontent.com/bdsp-core/SpikeNet1/main/model/'
FILES = {
    'spikenet1.o_structure.txt': None,
    # Digest of the file as published; verified on 2026-10-01.
    'spikenet1.o_weights.h5': '22bee5b468a105d50b09eceb1ac1a7e1',
}

NOTICE = """SpikeNet1 -- interictal epileptiform discharge detector
=======================================================

Downloaded by experiments/fetch_spikenet.py from
https://github.com/bdsp-core/SpikeNet1

NOT part of this project. Redistributed nowhere; fetched locally on request.

Cite:
  Jing J, Sun H, Kim JA, Herlopian A, Karakis I, Ng M, Halford JJ, Maus D,
  Chan F, Dolatshahi M, Muniz C, Chu C, Sacca V, Pathmanathan J, Ge W,
  Dauwels J, Lam A, Cole AJ, Cash SS, Westover MB. Development of
  Expert-Level Automated Detection of Epileptiform Discharges During
  Electroencephalogram Interpretation. JAMA Neurol. 2020;77(1):103-108.
  PMID 31633740; PMCID PMC6806668.

LICENCE -- UNRESOLVED. The upstream repository contradicts itself:
  LICENSE.txt : CC BY-NC 4.0  (Attribution-NonCommercial, no commercial use)
  README.md   : CC BY-ND 4.0  (Attribution-NoDerivatives, no modified copies)
Non-commercial academic use with citation satisfies either reading, which is
what this project does. Clarify with the authors before any distribution or
commercial use.

VALIDATION -- NONE HERE. The published performance is for classifying candidate
events already proposed by another stage, not for sliding the model across
continuous EEG. No threshold has been validated on this project's data. Every
mark it produces is a candidate for a human to confirm or reject.
"""


def fetch(dest=None):
    from gui.paths import resource
    dest = dest or resource('models', 'spikenet1')
    if not os.path.isdir(dest):
        os.makedirs(dest)
    try:
        from urllib.request import urlopen
    except ImportError:                                   # pragma: no cover
        from urllib2 import urlopen                        # noqa: F401

    for name, digest in FILES.items():
        out = os.path.join(dest, name)
        if os.path.exists(out) and _ok(out, digest):
            print('  have    {}'.format(name))
            continue
        print('  fetch   {} ...'.format(name), end=' ')
        sys.stdout.flush()
        body = urlopen(BASE + name, timeout=60).read()
        with open(out, 'wb') as fh:
            fh.write(body)
        if not _ok(out, digest):
            os.remove(out)
            raise RuntimeError(
                '{} did not match its expected digest and was discarded. The '
                'upstream file may have changed; check the repository before '
                'trusting it.'.format(name))
        print('{:,} bytes'.format(len(body)))

    notice = os.path.join(dest, 'NOTICE.txt')
    with open(notice, 'w') as fh:
        fh.write(NOTICE)
    print('\n  into    {}'.format(dest))
    print('  licence ambiguity and citation recorded in NOTICE.txt')
    return dest


def _ok(path, digest):
    if digest is None:
        return os.path.getsize(path) > 0
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest() == digest


if __name__ == '__main__':
    fetch(sys.argv[1] if len(sys.argv) > 1 else None)
