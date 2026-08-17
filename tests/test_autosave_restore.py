"""The autosave must be readable, not just writable.

The application has always written a crash-recovery record after every decision,
and `_confirm_discard_review` tells the reviewer where it is when they abandon a
session. Nothing ever read it back, so the promise could not be kept: a reviewer
whose machine died forty candidates into a recording lost all of it, while a file
containing their work sat on disk beside the recording.

These tests pin the read path. They drive the restore logic directly rather than
through the dialog, because a modal cannot be clicked headlessly — the dialog is
stubbed to answer Yes.
"""
import json
import os
import shutil
import tempfile
import unittest

try:
    from PyQt5 import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.app import MainWindow
    HAVE_QT = True
except Exception:                                    # pragma: no cover
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class AutosaveRestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.w = MainWindow()
        self.w._edf_path = os.path.join(self.tmp, 'rec.edf')
        # Stub the modal, and remember the real one. Replacing exec_ on the
        # CLASS and not restoring it would leak into every test module that
        # runs afterwards — including test_review_guards, whose whole subject
        # is discard dialogs, where an always-Yes stub would mask a real
        # failure rather than cause a visible one.
        self._real_exec = QtWidgets.QMessageBox.exec_
        self._answer(QtWidgets.QMessageBox.Yes)

    def tearDown(self):
        QtWidgets.QMessageBox.exec_ = self._real_exec
        self.w.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _answer(self, button):
        QtWidgets.QMessageBox.exec_ = lambda self_: button

    def _write_autosave(self, events):
        path = self.w._autosave_path()
        with open(path, 'w') as f:
            json.dump({'edf': self.w._edf_path, 'ai_source': 'baseline',
                       'threshold': 0.5, 'events': events}, f)
        return path

    # -- reading -------------------------------------------------------
    def test_reads_back_what_it_wrote(self):
        self.w._events = [{'start': 10.0, 'stop': 22.0, 'status': 'accepted',
                           'prob': 0.9, 'review_note': None}]
        self.w._autosave_review()
        blob = self.w._read_autosave()
        self.assertIsNotNone(blob, 'the file it just wrote could not be read')
        self.assertEqual(len(blob['events']), 1)

    def test_missing_file_is_not_an_error(self):
        self.assertIsNone(self.w._read_autosave())

    def test_a_truncated_file_is_ignored(self):
        """A crash mid-write is exactly when this file gets truncated."""
        with open(self.w._autosave_path(), 'w') as f:
            f.write('{"events": [{"start": 1.0, "sto')
        self.assertIsNone(self.w._read_autosave())

    # -- restoring -----------------------------------------------------
    def test_decisions_are_restored_onto_rebuilt_proposals(self):
        """The proposals are rebuilt from probabilities, so matching is by span."""
        self._write_autosave([
            {'start': 10.0, 'stop': 22.0, 'status': 'accepted',
             'review_note': None, 'model_score_uncalibrated': 0.9},
            {'start': 40.0, 'stop': 52.0, 'status': 'rejected',
             'review_note': 'chewing', 'model_score_uncalibrated': 0.6}])
        # Fresh objects, as _rebuild_events_from_probs would produce.
        self.w._events = [
            {'start': 10.0, 'stop': 22.0, 'status': 'proposed', 'prob': 0.9},
            {'start': 40.0, 'stop': 52.0, 'status': 'proposed', 'prob': 0.6},
            {'start': 70.0, 'stop': 82.0, 'status': 'proposed', 'prob': 0.7}]
        self.w._offer_autosave_restore()

        self.assertEqual(self.w._events[0]['status'], 'accepted')
        self.assertEqual(self.w._events[1]['status'], 'rejected')
        self.assertEqual(self.w._events[1]['review_note'], 'chewing')
        self.assertEqual(self.w._events[2]['status'], 'proposed',
                         'an untouched candidate must stay untouched')

    def test_added_events_come_back(self):
        """The case the tool exists to demonstrate.

        A seizure the detector never proposed has no candidate to match against,
        so it must be re-inserted rather than looked up.
        """
        self._write_autosave([
            {'start': 100.0, 'stop': 118.0, 'status': 'added',
             'review_note': 'missed by detector',
             'model_score_uncalibrated': None}])
        self.w._events = [
            {'start': 10.0, 'stop': 22.0, 'status': 'proposed', 'prob': 0.9}]
        self.w._offer_autosave_restore()

        added = [e for e in self.w._events if e['status'] == 'added']
        self.assertEqual(len(added), 1, 'the added seizure was not restored')
        self.assertEqual(added[0]['start'], 100.0)
        self.assertEqual(added[0]['review_note'], 'missed by detector')

    def test_restored_events_stay_in_time_order(self):
        self._write_autosave([
            {'start': 5.0, 'stop': 17.0, 'status': 'added',
             'review_note': None, 'model_score_uncalibrated': None}])
        self.w._events = [
            {'start': 60.0, 'stop': 72.0, 'status': 'proposed', 'prob': 0.8}]
        self.w._offer_autosave_restore()
        starts = [e['start'] for e in self.w._events]
        self.assertEqual(starts, sorted(starts))

    def test_restoring_marks_the_session_unsaved(self):
        """Restored work is unexported work; the discard guard must see it."""
        self._write_autosave([
            {'start': 10.0, 'stop': 22.0, 'status': 'accepted',
             'review_note': None, 'model_score_uncalibrated': 0.9}])
        self.w._events = [
            {'start': 10.0, 'stop': 22.0, 'status': 'proposed', 'prob': 0.9}]
        self.w._dirty = False
        self.w._offer_autosave_restore()
        self.assertTrue(self.w._dirty)
        self.assertEqual(len(self.w._reviewed_events()), 1)

    def test_declining_changes_nothing_and_keeps_the_file(self):
        self._answer(QtWidgets.QMessageBox.No)
        path = self._write_autosave([
            {'start': 10.0, 'stop': 22.0, 'status': 'accepted',
             'review_note': None, 'model_score_uncalibrated': 0.9}])
        self.w._events = [
            {'start': 10.0, 'stop': 22.0, 'status': 'proposed', 'prob': 0.9}]
        self.w._offer_autosave_restore()
        self.assertEqual(self.w._events[0]['status'], 'proposed')
        self.assertTrue(os.path.exists(path),
                        'declining must not delete the reviewer\'s only copy')

    def test_an_autosave_with_no_events_prompts_nothing(self):
        self._write_autosave([])
        self.w._events = [
            {'start': 10.0, 'stop': 22.0, 'status': 'proposed', 'prob': 0.9}]
        self.w._offer_autosave_restore()
        self.assertEqual(self.w._events[0]['status'], 'proposed')


if __name__ == '__main__':
    unittest.main()
