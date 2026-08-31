"""A cache must survive being copied to the stick it ships on.

Probability caches sit beside their recording and are matched on the EDF's size
and modification time. That is the right check — it catches a recording that
was regenerated or edited — but it was tuned on NTFS and the application is
delivered on a FAT32 USB stick.

FAT32 stores mtime at two-second resolution and rounds UP. Measured copying the
demo set to the stick: 2 of 9 recordings drifted 1.5 s and 1.7 s, past the
1.0 s tolerance, and arrived stale. Nothing warns about that. The app simply
rescores the recording when it is opened — which for the 58-minute file is
minutes of frozen window, and would have happened for the first time in front
of a clinician.

So the tolerance is the filesystem's granularity plus a rounding allowance. The
size check is what still catches an edited recording; a genuine re-export
differs by far more than two seconds anyway.
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.cache import (MTIME_TOLERANCE_S, _fresh_for,        # noqa: E402
                          load_probs, save_probs)

FS = 250
CH = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ',
      'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']


def _write_edf(path, seconds=30):
    import pyedflib
    labels = ['EEG {}-REF'.format(c) for c in CH]
    sigs = np.zeros((len(labels), seconds * FS))

    def _attempt(rate_key):
        w = pyedflib.EdfWriter(path, len(labels),
                               file_type=pyedflib.FILETYPE_EDFPLUS)
        try:
            w.setSignalHeaders([{
                'label': lab, 'dimension': 'uV', rate_key: FS,
                'physical_min': -3276.8, 'physical_max': 3276.7,
                'digital_min': -32768, 'digital_max': 32767,
                'prefilter': '', 'transducer': ''} for lab in labels])
            w.writeSamples([np.ascontiguousarray(s) for s in sigs])
        finally:
            try:
                w.close()
            except Exception:                                # noqa: BLE001
                pass

    try:
        _attempt('sample_frequency')
    except (FutureWarning, KeyError, TypeError, ValueError):
        _attempt('sample_rate')
    return path


class ToleratesFat32Timestamps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='freshness')
        self.edf = _write_edf(os.path.join(self.tmp, 'r.edf'))
        starts = np.arange(0, 19, 6, dtype=np.int32)
        save_probs(self.edf, starts,
                   np.zeros(starts.size, dtype=np.float32),
                   skip_code=np.zeros(starts.size, dtype=np.int8),
                   meta={'step_s': 6, 'segment_s': 12, 'use_ica': True,
                         'fs': FS, 'weights': 'convlstm_ICA_12_train.h5'})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _shift_mtime(self, seconds):
        st = os.stat(self.edf)
        os.utime(self.edf, (st.st_atime, st.st_mtime + seconds))

    def test_the_tolerance_covers_fat32_granularity(self):
        self.assertGreaterEqual(
            MTIME_TOLERANCE_S, 2.0,
            'FAT32 stores mtime at two-second resolution and rounds up, so a '
            'tolerance under 2 s invalidates caches merely by copying them to '
            'the stick the application ships on')

    def test_a_cache_survives_the_drift_a_stick_actually_causes(self):
        """1.7 s was measured on the real demo stick; 1.5 s on another file."""
        for drift in (1.5, 1.7, 2.0):
            self._shift_mtime(drift)
            self.assertIsNotNone(
                load_probs(self.edf),
                'a {} s timestamp shift invalidated the cache — that is a '
                'copy to a USB stick, not an edited recording'.format(drift))
            self._shift_mtime(-drift)

    def test_a_genuinely_different_recording_is_still_rejected(self):
        """The tolerance must not become a hole."""
        self._shift_mtime(600.0)
        self.assertIsNone(load_probs(self.edf),
                          'a ten-minute timestamp shift was accepted; the '
                          'staleness check no longer means anything')

    def test_a_changed_size_is_rejected_whatever_the_timestamp(self):
        """Size is what catches an edit that preserved the mtime."""
        loaded = load_probs(self.edf)
        self.assertIsNotNone(loaded)
        meta = dict(loaded['meta'])
        meta['edf_size'] = int(meta['edf_size']) + 1
        self.assertFalse(
            _fresh_for(self.edf, {'meta': meta}),
            'a recording of a different size was accepted as unchanged')

    def test_a_cache_with_no_stamp_is_not_trusted(self):
        self.assertFalse(_fresh_for(self.edf, {'meta': {}}))


if __name__ == '__main__':
    unittest.main()
