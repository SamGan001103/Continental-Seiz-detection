"""The overview envelope, prepared on the server rather than on the desk.

Measured on a 24-hour recording before this existed: the first zoom out past
five minutes blocked the interface for 48 seconds while the whole recording
was filtered into its min/max envelope, and every filter change paid it again.
That was the lag a reviewer actually met, and it was the last thing in the
system still doing heavy work on the clinical workstation.

The envelope is cacheable because it is deterministic given the recording and
the display settings. These tests pin the part that makes caching safe rather
than merely fast: it must be impossible to draw a picture of one recording, or
one filter setting, while claiming another.

Every failure resolves to None and a local rebuild. A cache that cannot be read
is a slow open; it must never be a wrong trace.
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

from gui.io import envelope_cache as EC                 # noqa: E402


class TheKeyIsTheDisplaySettings(unittest.TestCase):
    def test_the_same_settings_make_the_same_key(self):
        a = EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar', 256)
        b = EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar', 256)
        self.assertEqual(a, b)

    def test_float_noise_from_a_combo_box_does_not_invalidate_it(self):
        """These arrive as floats from the UI. A 70.0 that reads back as
        69.999999 must not throw away a 36 MB artefact."""
        a = EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar', 256)
        b = EC.settings_key(1.0, 69.999999, 60.0, 'Longitudinal Bipolar', 256)
        self.assertEqual(a, b)

    def test_a_different_filter_is_a_different_key(self):
        a = EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar', 256)
        for other in ((0.5, 70.0, 60.0), (1.0, 35.0, 60.0),
                      (1.0, 70.0, 50.0)):
            self.assertNotEqual(
                a, EC.settings_key(*(other + ('Longitudinal Bipolar', 256))))

    def test_a_different_montage_is_a_different_key(self):
        self.assertNotEqual(
            EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar', 256),
            EC.settings_key(1.0, 70.0, 60.0, 'Referential', 256))

    def test_none_cutoffs_survive_the_round_trip(self):
        k = EC.settings_key(None, None, None, 'Referential', 250)
        self.assertEqual(k[:3], (None, None, None))


class ItNeverServesTheWrongPicture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rec = os.path.join(self.tmp, 'rec.edf')
        with open(self.rec, 'wb') as f:
            f.write(b'x' * 8192)
        self.key = EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar',
                                   256)
        self.lo = np.full((18, 120), -3.5, dtype=np.float32)
        self.hi = np.full((18, 120), 4.25, dtype=np.float32)
        EC.save(self.rec, self.key, 64, self.lo, self.hi)

    def test_what_was_stored_comes_back(self):
        got = EC.load(self.rec, self.key)
        self.assertIsNotNone(got)
        step, lo, hi = got
        self.assertEqual(step, 64)
        np.testing.assert_array_equal(lo, self.lo)
        np.testing.assert_array_equal(hi, self.hi)

    def test_different_settings_are_refused(self):
        other = EC.settings_key(0.5, 70.0, 60.0, 'Longitudinal Bipolar', 256)
        self.assertIsNone(EC.load(self.rec, other))

    def test_a_changed_recording_is_refused(self):
        """A bundle matched to the wrong EEG would draw one patient's trace
        under another's name."""
        with open(self.rec, 'wb') as f:
            f.write(b'x' * 16384)
        self.assertIsNone(EC.load(self.rec, self.key))

    def test_a_missing_recording_is_refused(self):
        os.remove(self.rec)
        self.assertIsNone(EC.load(self.rec, self.key))

    def test_a_damaged_file_is_refused_rather_than_raising(self):
        with open(EC.path_for(self.rec), 'wb') as f:
            f.write(b'\x00\xff not an npz')
        self.assertIsNone(EC.load(self.rec, self.key))

    def test_a_missing_cache_is_simply_absent(self):
        os.remove(EC.path_for(self.rec))
        self.assertIsNone(EC.load(self.rec, self.key))

    def test_the_file_sits_beside_the_recording(self):
        self.assertTrue(EC.path_for('/x/y.edf').endswith('y.envelope.npz'))

    def test_writing_twice_replaces_rather_than_accumulates(self):
        EC.save(self.rec, self.key, 64, self.lo * 2, self.hi * 2)
        step, lo, hi = EC.load(self.rec, self.key)
        np.testing.assert_array_equal(lo, self.lo * 2)
        self.assertFalse(os.path.exists(EC.path_for(self.rec) + '.part.npz'))


if __name__ == '__main__':
    unittest.main()
