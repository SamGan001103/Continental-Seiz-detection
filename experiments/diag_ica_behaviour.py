"""Audit what ica_arti_remove actually does, window by window.

Measures the things that decide whether the ICA stage is implemented correctly:
  - how often FastICA fails to converge
  - how many components find_bads_eog flags, vs how many the code removes
  - how often NO eog component is found  -> which return path fires
  - whether the two return paths give differently-preprocessed data
  - the sample-count vs component-count ratio for a stable decomposition
"""
from __future__ import print_function
import os
import sys
import warnings
import numpy as np

REPO = r'c:\Users\User\Continental-Seiz-detection'
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import mne  # noqa
from mne.preprocessing import ICA  # noqa
from gui.io.edf import load_edf_19ch, CHANNELS_19  # noqa
import utils.preprocessing as pp  # noqa

mne.set_log_level('ERROR')

SEG, STEP, FS = 12, 6, 250
FILES = [
    'sample_data/v2.0.0/edf/eval/aaaaaarq/s017_2014_06_29/01_tcp_ar/aaaaaarq_s017_t000.edf',
    'sample_data/v2.0.0/edf/eval/aaaaajru/s031_2014_03_15/01_tcp_ar/aaaaajru_s031_t002.edf',
]

tot = dict(n=0, nonconv=0, no_eog=0, one_eog=0, two_eog=0,
           flagged=0, removed=0, fit_fail=0)
identical_to_input = 0
maxdiff_when_no_eog = []

for path in FILES:
    if not os.path.exists(path):
        print('missing', path)
        continue
    data, fs, dur = load_edf_19ch(path)
    starts = range(0, int(dur) - SEG + 1, STEP)
    print('\n=== %s  (%.0f s, %d windows) ===' % (
        os.path.basename(path), dur, len(list(starts))))

    for t in range(0, int(dur) - SEG + 1, STEP):
        seg = data[:, t * fs:t * fs + SEG * fs]
        if seg.shape[1] != SEG * fs:
            break
        if pp.detect_interupted_data(seg.transpose(), fs):
            continue
        tot['n'] += 1

        raw = pp.create_mne_raw(seg, fs, CHANNELS_19)
        filt = raw.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            filt.load_data().filter(l_freq=0.1, h_freq=None, verbose=False)

        ica = ICA(n_components=19, random_state=13)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                ica.fit(filt, verbose=False)
            except Exception:
                tot['fit_fail'] += 1
                continue
            if any('did not converge' in str(x.message).lower() or
                   'convergence' in str(x.message).lower() for x in w):
                tot['nonconv'] += 1

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            e1, _ = ica.find_bads_eog(filt, threshold=2.0, ch_name='Fp1',
                                      verbose=False)
            e2, _ = ica.find_bads_eog(filt, threshold=2.0, ch_name='Fp2',
                                      verbose=False)
        flagged = set(e1) | set(e2)
        removed = set()
        if e1:
            removed.add(e1[0])
        if e2:
            removed.add(e2[0])
        tot['flagged'] += len(flagged)
        tot['removed'] += len(removed)

        if not removed:
            tot['no_eog'] += 1
            # This is the branch that returns the ORIGINAL, UNFILTERED data.
            out = seg
            identical_to_input += 1
            # what the filtered path would have returned instead
            maxdiff_when_no_eog.append(
                float(np.max(np.abs(filt.get_data() * 1e6 - seg))))
        elif len(removed) == 1:
            tot['one_eog'] += 1
        else:
            tot['two_eog'] += 1

print('\n' + '=' * 66)
n = max(tot['n'], 1)
print('windows analysed                : %d' % tot['n'])
print('FastICA fit raised              : %d' % tot['fit_fail'])
print('FastICA did NOT converge        : %d  (%.0f%%)' % (
    tot['nonconv'], 100.0 * tot['nonconv'] / n))
print('no EOG component found          : %d  (%.0f%%)   <- returns RAW data' % (
    tot['no_eog'], 100.0 * tot['no_eog'] / n))
print('  1 component removed           : %d' % tot['one_eog'])
print('  2 components removed          : %d' % tot['two_eog'])
print('components FLAGGED by find_bads : %d' % tot['flagged'])
print('components actually REMOVED     : %d   (discarded %d by taking [0])' % (
    tot['removed'], tot['flagged'] - tot['removed']))
if maxdiff_when_no_eog:
    print('\nWhen no EOG is found the function returns the unfiltered input,')
    print('so those windows skip the 0.1 Hz high-pass the other windows get.')
    print('  max |filtered - raw| on those windows: %.2f uV (median %.2f)' % (
        max(maxdiff_when_no_eog), float(np.median(maxdiff_when_no_eog))))

print('\n--- decomposition stability ---')
print('samples per window   : %d' % (SEG * FS))
print('components requested : 19  (full rank, no PCA reduction)')
print('samples per component: %.0f' % (SEG * FS / 19.0))
print('Common rule of thumb : k * n^2 samples, k>=20, n=19 -> >= %d samples'
      % (20 * 19 * 19))
print('                       we have %d, i.e. %.1fx SHORT' % (
    SEG * FS, (20 * 19 * 19) / float(SEG * FS)))
