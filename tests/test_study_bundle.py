"""The study bundle: the interface between the system's two halves.

The system is in two parts. One prepares a recording on the hospital's compute
machine, where the patient records and the computing power are; the other is a
review application on a clinical workstation, which reads what that produced
and runs no model at all. The bundle is the whole of the contract between
them, so these pin what it has to guarantee:

  * it names every artefact it carries, so a half-built bundle is visible
    rather than silently short;
  * it cannot be read against a recording it was not built from, because a
    bundle matched to the wrong EEG would put seizure proposals at timestamps
    that belong to someone else;
  * it records the display settings its artefacts are valid for, since the
    review client can no longer recompute them;
  * it says in the file itself that detector output is a candidate and not a
    finding.

No Qt, no model: the manifest is JSON and the rules about it are arithmetic.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from server import prepare_study as PS   # noqa: E402


class TheManifestDescribesTheBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rec = os.path.join(self.tmp, 'study.edf')
        with open(self.rec, 'wb') as f:
            f.write(b'0' * 8192)

    def _artefacts(self, which=('seizure_probabilities',
                                'interictal_candidates')):
        paths = PS.artefacts_for(self.rec)
        for name in which:
            with open(paths[name], 'wb') as f:
                f.write(b'x' * 128)

    def test_it_names_where_each_artefact_lives(self):
        a = PS.artefacts_for(self.rec)
        self.assertTrue(a['seizure_probabilities'].endswith('.probs.npz'))
        self.assertTrue(a['interictal_candidates'].endswith('.ied.csv'))

    def test_writing_it_records_what_is_present(self):
        self._artefacts()
        out, blob = PS.write_manifest(self.rec)
        self.assertTrue(os.path.exists(out))
        for name in ('seizure_probabilities', 'interictal_candidates'):
            self.assertTrue(blob['artefacts'][name]['present'])
            self.assertGreater(blob['artefacts'][name]['bytes'], 0)

    def test_a_half_built_bundle_is_visible_not_silent(self):
        """Preparing can fail partway. A manifest that claimed everything was
        there would send a reviewer looking for proposals that do not exist."""
        self._artefacts(('seizure_probabilities',))
        _, blob = PS.write_manifest(self.rec)
        self.assertTrue(blob['artefacts']['seizure_probabilities']['present'])
        self.assertFalse(blob['artefacts']['interictal_candidates']['present'])
        self.assertEqual(PS.missing_from(self.rec),
                         ['interictal_candidates'])

    def test_it_records_the_display_settings_the_artefacts_assume(self):
        """The client cannot recompute, so the bundle has to say what it was
        built for rather than leaving the reviewer to guess."""
        self._artefacts()
        _, blob = PS.write_manifest(self.rec)
        d = blob['display_settings']
        self.assertEqual(d['montage'], 'Longitudinal Bipolar')
        for key in ('hp', 'lp', 'notch'):
            self.assertIn(key, d)

    def test_it_says_detections_are_candidates_not_findings(self):
        self._artefacts()
        _, blob = PS.write_manifest(self.rec)
        note = blob['note'].lower()
        self.assertIn('candidate', note)
        self.assertIn('not a finding', note)
        self.assertIn('clinically validated', note)
        self.assertIn('no threshold', note)

    def test_it_records_that_the_models_ran_on_the_server(self):
        self._artefacts()
        _, blob = PS.write_manifest(self.rec)
        self.assertIn('not on the review workstation', blob['note'])


class ABundleNeverOutlivesItsRecording(unittest.TestCase):
    """A bundle read against the wrong EEG would place seizure proposals at
    timestamps belonging to a different patient. Same guard the probability
    cache already applies, lifted to the bundle as a whole."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rec = os.path.join(self.tmp, 'study.edf')
        with open(self.rec, 'wb') as f:
            f.write(b'0' * 8192)
        PS.write_manifest(self.rec)

    def test_it_matches_the_recording_it_was_built_from(self):
        self.assertTrue(PS.bundle_matches(self.rec))

    def test_a_recording_of_a_different_size_is_refused(self):
        with open(self.rec, 'wb') as f:
            f.write(b'0' * 16384)
        self.assertFalse(PS.bundle_matches(self.rec))

    def test_a_missing_recording_is_refused(self):
        os.remove(self.rec)
        self.assertFalse(PS.bundle_matches(self.rec))

    def test_a_missing_manifest_is_not_a_match(self):
        os.remove(PS.manifest_path(self.rec))
        self.assertIsNone(PS.read_manifest(self.rec))
        self.assertFalse(PS.bundle_matches(self.rec))

    def test_a_damaged_manifest_does_not_raise(self):
        with open(PS.manifest_path(self.rec), 'w') as f:
            f.write('{ this is not json')
        self.assertIsNone(PS.read_manifest(self.rec))
        self.assertFalse(PS.bundle_matches(self.rec))

    def test_the_manifest_is_readable_json_with_a_version(self):
        with open(PS.manifest_path(self.rec)) as f:
            blob = json.load(f)
        self.assertEqual(blob['manifest_version'], PS.MANIFEST_VERSION)
        self.assertIn('prepared_utc', blob)


class TheRecordingIsIdentifiedByContent(unittest.TestCase):
    def test_two_different_recordings_get_different_digests(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        digests = []
        for i, body in enumerate((b'a' * 4096, b'b' * 4096)):
            rec = os.path.join(tmp, 'r{}.edf'.format(i))
            with open(rec, 'wb') as f:
                f.write(body)
            _, blob = PS.write_manifest(rec)
            digests.append(blob['recording']['sha256_prefix'])
        self.assertNotEqual(digests[0], digests[1])

    def test_hashing_is_capped_so_a_gigabyte_is_not_read_twice(self):
        """A 1.2 GB EDF takes seconds to hash in full; the cap keeps preparing
        a study from paying that twice for no extra certainty."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        rec = os.path.join(tmp, 'big.edf')
        with open(rec, 'wb') as f:
            f.write(b'z' * (3 << 20))
        _, hashed = PS._sha256(rec, limit=1 << 20)
        self.assertLessEqual(hashed, (1 << 20) + (1 << 20))


if __name__ == '__main__':
    unittest.main()
