"""Interictal marks have to survive, and have to stay attributable.

Three things were wrong before the automatic detector arrived, and all three
only bite after a reviewer has done real work:

  * marks were never autosaved, so a crash lost every one of them while the
    seizure decisions beside them were recovered;
  * `.ied.csv` was written on export and never read back, so marks were
    invisible the moment the recording was closed, and anything a detector
    found ahead of time could not reach the screen at all;
  * the tally added a reviewer's marks to a detector's proposals, which would
    report an unconfirmed machine guess as something a clinician asserted.

These exercise MainWindow's logic with no display, the way the other
app-level tests do.
"""
import io
import os
import shutil
import tempfile
import unittest

from gui.io.ied_marks import read_ied_marks, write_ied_marks

try:
    from PyQt5 import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.app import MainWindow
    HAVE_QT = True
except Exception:                                    # pragma: no cover
    HAVE_QT = False


REVIEWER = {'time': 12.0, 'channel': 'Fp1-F7', 'note': 'sharp'}
AUTO = {'time': 40.0, 'channel': None, 'source': 'spikenet1',
        'score': 0.93, 'note': 'auto, unconfirmed'}


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class MarksAreAutosaved(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.w = MainWindow()
        self.w._reviewer_id = 'test'
        self.w._edf_path = os.path.join(self.tmp, 'rec.edf')
        self.w._duration_s = 300.0
        self.w._events = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _autosaved(self):
        import json
        with open(self.w._autosave_path()) as f:
            return json.load(f)

    def test_a_mark_reaches_the_autosave_file(self):
        self.w._ied_marks = [dict(REVIEWER)]
        self.w._autosave_review()
        blob = self._autosaved()
        self.assertEqual(len(blob['interictal_marks']), 1)
        self.assertEqual(blob['interictal_marks'][0]['time'], 12.0)

    def test_marks_alone_are_worth_recovering(self):
        """A session can be entirely spike marking, with no seizure candidate
        touched. Requiring `events` threw that whole session away."""
        self.w._ied_marks = [dict(REVIEWER)]
        self.w._autosave_review()
        self.assertIsNotNone(self.w._read_autosave())

    def test_an_empty_session_still_writes_nothing_to_recover(self):
        self.w._ied_marks = []
        self.w._autosave_review()
        self.assertIsNone(self.w._read_autosave())


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheSidecarIsReadBack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.w = MainWindow()
        self.edf = os.path.join(self.tmp, 'rec.edf')
        self.side = os.path.join(self.tmp, 'rec.ied.csv')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_marks_written_on_export_come_back_on_open(self):
        write_ied_marks(self.side, 'rec', 300.0, [dict(REVIEWER)])
        self.assertEqual(self.w._load_ied_sidecar(self.edf), 1)
        self.assertEqual(self.w._ied_marks[0]['channel'], 'Fp1-F7')
        self.assertEqual(self.w._ied_marks[0]['note'], 'sharp')

    def test_detector_candidates_come_back_with_their_source(self):
        write_ied_marks(self.side, 'rec', 300.0, [dict(AUTO)])
        self.w._load_ied_sidecar(self.edf)
        self.assertEqual(self.w._ied_marks[0]['source'], 'spikenet1')
        self.assertAlmostEqual(self.w._ied_marks[0]['score'], 0.93, places=4)

    def test_they_arrive_in_time_order(self):
        write_ied_marks(self.side, 'rec', 300.0,
                        [dict(AUTO), dict(REVIEWER)])
        self.w._load_ied_sidecar(self.edf)
        times = [m['time'] for m in self.w._ied_marks]
        self.assertEqual(times, sorted(times))

    def test_no_sidecar_is_not_an_error(self):
        self.assertEqual(self.w._load_ied_sidecar(self.edf), 0)
        self.assertEqual(self.w._ied_marks, [])

    def test_a_damaged_sidecar_does_not_block_opening_the_recording(self):
        """Opening a recording must not fail because a sidecar is corrupt —
        the signal is the thing the reviewer came for."""
        with open(self.side, 'wb') as f:
            f.write(b'\x00\xff not a csv at all\x00')
        self.assertEqual(self.w._load_ied_sidecar(self.edf), 0)


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class ConfirmedAndProposedAreCountedApart(unittest.TestCase):
    def setUp(self):
        self.w = MainWindow()

    def test_the_label_never_shows_one_combined_total(self):
        self.w._ied_marks = [dict(REVIEWER), dict(AUTO), dict(AUTO)]
        self.w._update_ied_summary()
        text = self.w._ied_label.text()
        self.assertIn('IED 1', text)
        self.assertIn('2 proposed', text)
        self.assertNotIn('IED 3', text)

    def test_the_tooltip_says_proposals_are_unvalidated(self):
        self.w._ied_marks = [dict(AUTO)]
        self.w._update_ied_summary()
        tip = self.w._ied_label.toolTip()
        self.assertIn('UNVALIDATED', tip)
        self.assertIn('spikenet1', tip)
        self.assertIn('not findings', tip.replace('are not \n', 'are not '))

    def test_it_no_longer_claims_the_detector_cannot_find_these(self):
        self.w._ied_marks = [dict(REVIEWER)]
        self.w._update_ied_summary()
        self.assertNotIn('The detector does not find these',
                         self.w._ied_label.toolTip())

    def test_proposals_do_not_invent_a_derivation(self):
        """SpikeNet scores the whole montage. Grouping its marks under a
        channel named '' would have put them all on one fictional lane."""
        self.w._ied_marks = [dict(AUTO)]
        self.w._update_ied_summary()
        self.assertNotIn('(unknown)', self.w._ied_label.text())

    def test_nothing_marked_hides_the_label(self):
        self.w._ied_marks = []
        self.w._update_ied_summary()
        self.assertFalse(self.w._ied_label.isVisible())


class RoundTripThroughTheFileKeepsTheDistinction(unittest.TestCase):
    """No Qt needed: the file format is the contract both halves rely on."""

    def test_a_reviewer_mark_and_a_proposal_survive_together(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        p = os.path.join(tmp, 'r.ied.csv')
        write_ied_marks(p, 'r', 300.0, [dict(REVIEWER), dict(AUTO)])
        back = read_ied_marks(p)
        by_source = {m['source']: m for m in back}
        self.assertEqual(set(by_source), {'reviewer', 'spikenet1'})
        self.assertIsNone(by_source['reviewer']['score'])
        self.assertAlmostEqual(by_source['spikenet1']['score'], 0.93, places=4)


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class StaleMarksAreRefused(unittest.TestCase):
    """The probability cache already refuses to load against a recording whose
    size or mtime changed — that was the USB-copy bug. Marks need the same
    guard for a worse reason: a stale probability shows a wrong number, but a
    stale mark asserts a DISCHARGE AT A TIMESTAMP, and against the wrong
    recording that is a claim about a moment which does not exist."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.w = MainWindow()
        self.edf = os.path.join(self.tmp, 'rec.edf')
        self.side = os.path.join(self.tmp, 'rec.ied.csv')
        with io.open(self.edf, 'wb') as f:
            f.write(b'x' * 4096)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self):
        write_ied_marks(self.side, 'rec', 300.0, [dict(REVIEWER)],
                        source_path=self.edf)

    def test_marks_load_against_the_recording_they_were_made_on(self):
        self._write()
        self.assertEqual(self.w._load_ied_sidecar(self.edf), 1)

    def test_a_recording_that_changed_size_refuses_its_marks(self):
        self._write()
        with io.open(self.edf, 'wb') as f:
            f.write(b'x' * 8192)          # a different recording, same name
        self.assertEqual(self.w._load_ied_sidecar(self.edf), 0)
        self.assertEqual(self.w._ied_marks, [])

    def test_a_missing_recording_refuses_stamped_marks(self):
        self._write()
        os.remove(self.edf)
        self.assertEqual(self.w._load_ied_sidecar(self.edf), 0)

    def test_unstamped_marks_are_still_accepted(self):
        """Files written before the guard existed, and v1 files. Refusing
        them would silently discard a reviewer's real work."""
        write_ied_marks(self.side, 'rec', 300.0, [dict(REVIEWER)])
        self.assertEqual(self.w._load_ied_sidecar(self.edf), 1)

    def test_the_reviewer_is_told_rather_than_left_wondering(self):
        self._write()
        with io.open(self.edf, 'wb') as f:
            f.write(b'x' * 8192)
        self.w._load_ied_sidecar(self.edf)
        self.assertIn('different version',
                      self.w.statusBar().currentMessage())


if __name__ == '__main__':
    unittest.main()
