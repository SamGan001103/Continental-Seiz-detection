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


class EachPresetIsItsOwnFile(unittest.TestCase):
    """The application offers 432 combinations and a bundle carries about six,
    so the preset has to be in the name. It is also the only reason a second
    preset does not overwrite the first."""

    def test_the_name_says_what_the_settings_were(self):
        tag = EC.preset_tag({'hp': 1.0, 'lp': 70.0, 'notch': 60.0,
                             'montage': 'Longitudinal Bipolar'})
        self.assertEqual(tag, 'hp1-lp70-n60-longitudinal-bipolar')

    def test_a_decimal_point_does_not_become_an_extension(self):
        tag = EC.preset_tag({'hp': 0.5, 'lp': 70.0, 'notch': 60.0,
                             'montage': 'Common Average'})
        self.assertEqual(tag, 'hp0p5-lp70-n60-common-average')
        self.assertTrue(EC.path_for('/x/y.edf', {
            'hp': 0.5, 'lp': 70.0, 'notch': 60.0,
            'montage': 'Common Average'}).endswith('.npz'))

    def test_a_filter_that_is_off_is_named_not_omitted(self):
        tag = EC.preset_tag({'hp': None, 'lp': None, 'notch': None,
                             'montage': 'Referential (raw)'})
        self.assertEqual(tag, 'hpoff-lpoff-noff-referential-raw')

    def test_every_shipped_preset_gets_a_distinct_name(self):
        presets = [{'hp': hp, 'lp': lp, 'notch': n, 'montage': m}
                   for hp in (None, 0.3, 0.5, 1.0, 3.0, 5.0)
                   for lp in (None, 15.0, 30.0, 35.0, 50.0, 70.0)
                   for n in (None, 50.0, 60.0)
                   for m in ('Common Average', 'Longitudinal Bipolar',
                             'Transverse Bipolar', 'Referential (raw)')]
        tags = set(EC.preset_tag(p) for p in presets)
        self.assertEqual(len(tags), len(presets))

    def test_the_sample_rate_is_not_in_the_name(self):
        """A recording has one rate, so it is not a choice to name -- but it
        is still stored, and a mismatch is still refused."""
        a = EC.path_for('/x/y.edf', EC.preset_of(
            EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar', 256)))
        b = EC.path_for('/x/y.edf', EC.preset_of(
            EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar', 500)))
        self.assertEqual(a, b)


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
        self.path = EC.save(self.rec, self.key, 64, self.lo, self.hi)

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

    def test_a_different_sample_rate_is_refused_though_the_name_matches(self):
        """The rate is not in the file name, so this is the stored key doing
        the work rather than the path."""
        other = EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar', 500)
        self.assertEqual(EC.path_for(self.rec, EC.preset_of(other)),
                         self.path)
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
        with open(self.path, 'wb') as f:
            f.write(b'\x00\xff not an npz')
        self.assertIsNone(EC.load(self.rec, self.key))

    def test_a_missing_cache_is_simply_absent(self):
        os.remove(self.path)
        self.assertIsNone(EC.load(self.rec, self.key))

    def test_the_file_sits_beside_the_recording(self):
        self.assertTrue(self.path.startswith(os.path.join(self.tmp, 'rec.')))
        self.assertTrue(EC.path_for('/x/y.edf').endswith('y.envelope.npz'))

    def test_writing_twice_replaces_rather_than_accumulates(self):
        EC.save(self.rec, self.key, 64, self.lo * 2, self.hi * 2)
        step, lo, hi = EC.load(self.rec, self.key)
        np.testing.assert_array_equal(lo, self.lo * 2)
        self.assertFalse(os.path.exists(self.path + '.part.npz'))

    def test_a_second_preset_does_not_overwrite_the_first(self):
        """The bundle ships several presets. When they shared one file name,
        precompute wrote six and kept the last, and five of the six filter
        settings a reviewer could pick silently rebuilt on the workstation."""
        other = EC.settings_key(0.5, 70.0, 60.0, 'Longitudinal Bipolar', 256)
        lo2 = np.full((18, 120), -9.0, dtype=np.float32)
        p2 = EC.save(self.rec, other, 64, lo2, self.hi)
        self.assertNotEqual(p2, self.path)
        np.testing.assert_array_equal(EC.load(self.rec, self.key)[1], self.lo)
        np.testing.assert_array_equal(EC.load(self.rec, other)[1], lo2)

    def test_a_study_prepared_before_presets_had_names_still_reads(self):
        """The unnamed file earlier versions wrote is still accepted, on the
        strength of the settings inside it rather than its name."""
        legacy = EC.path_for(self.rec)
        os.replace(self.path, legacy)
        got = EC.load(self.rec, self.key)
        self.assertIsNotNone(got)
        np.testing.assert_array_equal(got[1], self.lo)


class AskingIsCheaperThanLoading(unittest.TestCase):
    """The redraw path has to know whether a render is about to take forty
    seconds before it calls it, and must not read 36 MB to find out."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rec = os.path.join(self.tmp, 'rec.edf')
        with open(self.rec, 'wb') as f:
            f.write(b'x' * 8192)
        self.key = EC.settings_key(1.0, 70.0, 60.0, 'Longitudinal Bipolar',
                                   256)
        self.path = EC.save(self.rec, self.key, 64,
                            np.zeros((18, 120), dtype=np.float32),
                            np.ones((18, 120), dtype=np.float32))

    def test_it_agrees_with_load_when_the_envelope_is_there(self):
        self.assertTrue(EC.has(self.rec, self.key))
        self.assertIsNotNone(EC.load(self.rec, self.key))

    def test_it_agrees_with_load_when_it_is_not(self):
        other = EC.settings_key(3.0, 15.0, 60.0, 'Transverse Bipolar', 256)
        self.assertFalse(EC.has(self.rec, other))
        self.assertIsNone(EC.load(self.rec, other))

    def test_a_stale_envelope_is_not_counted_as_present(self):
        with open(self.rec, 'wb') as f:
            f.write(b'x' * 16384)
        self.assertFalse(EC.has(self.rec, self.key))

    def test_a_damaged_envelope_is_not_counted_as_present(self):
        with open(self.path, 'wb') as f:
            f.write(b'rubbish')
        self.assertFalse(EC.has(self.rec, self.key))


if __name__ == '__main__':
    unittest.main()
