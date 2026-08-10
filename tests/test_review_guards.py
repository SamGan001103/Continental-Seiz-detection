"""Guards on the two destructive paths (B3, B5).

Nothing a reviewer does reaches disk until they choose Export, so opening
another recording, closing the window, or a careless Save dialog could all
destroy a session. These tests pin the guards without needing a display: they
exercise MainWindow's logic with the dialogs stubbed out.
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
class ReviewGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.w = MainWindow()
        # Export prompts once for a reviewer id; offscreen that modal would
        # block forever. Pre-set it rather than stubbing Qt globally.
        self.w._reviewer_id = 'test'
        self.w._edf_path = os.path.join(self.tmp, 'rec.edf')
        self.w._duration_s = 300.0
        self.w._events = [
            {'id': 1, 'start': 10.0, 'stop': 20.0, 'prob': 0.9,
             'status': 'proposed'},
            {'id': 2, 'start': 50.0, 'stop': 60.0, 'prob': 0.8,
             'status': 'proposed'},
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------------------------------------------------------------- B3
    def test_a_clean_session_never_prompts(self):
        self.assertFalse(self.w._dirty)
        self.assertTrue(self.w._confirm_discard_review('open another'))

    def test_accepting_an_event_marks_the_session_dirty(self):
        self.w._on_accept(1)
        self.assertTrue(self.w._dirty)

    def test_rejecting_an_event_marks_the_session_dirty(self):
        self.w._on_reject(1)
        self.assertTrue(self.w._dirty)

    def test_editing_an_extent_marks_the_session_dirty(self):
        self.w._on_region_edited(1, 12.0, 25.0)
        self.assertTrue(self.w._dirty)

    def test_dirty_title_carries_a_marker(self):
        self.w._on_accept(1)
        self.assertTrue(self.w.windowTitle().startswith('*'))

    def test_autosave_records_decisions_beside_the_edf(self):
        self.w._on_accept(1)
        self.w._on_reject(2)
        path = self.w._autosave_path()
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            payload = json.load(f)
        statuses = sorted(e['status'] for e in payload['events'])
        self.assertEqual(statuses, ['accepted', 'rejected'])
        # It must not be mistakable for an annotation file.
        self.assertFalse(path.endswith('.csv_bi'))

    def test_autosave_is_cleared_once_the_review_is_exported(self):
        self.w._on_accept(1)
        path = self.w._autosave_path()
        self.assertTrue(os.path.exists(path))
        self.w._dirty = False
        self.w._clear_autosave()
        self.assertFalse(os.path.exists(path))

    def test_autosave_reports_where_it_actually_wrote(self):
        self.w._on_accept(1)
        self.assertEqual(os.path.abspath(self.w._autosave_written_to),
                         os.path.abspath(self.w._autosave_path()))

    def test_autosave_falls_back_when_the_recording_folder_is_read_only(self):
        """Clinical recordings normally live on a read-only share.

        Losing crash recovery there — silently — would leave the reviewer
        believing their decisions were protected when they were not.
        """
        blocker = os.path.join(self.tmp, 'blocker')
        with open(blocker, 'wb') as f:
            f.write(b'not a directory')
        self.w._autosave_path = lambda: os.path.join(
            blocker, 'sub', 'x.review.autosave.json')

        self.w._on_accept(1)
        where = self.w._autosave_written_to
        self.assertIsNotNone(where, 'autosave silently wrote nowhere')
        self.assertEqual(os.path.abspath(where),
                         os.path.abspath(self.w._autosave_fallback_path()))
        self.assertTrue(os.path.exists(where))
        with open(where) as f:
            self.assertEqual(
                [e['status'] for e in json.load(f)['events']], ['accepted'])
        self.w._clear_autosave()
        self.assertFalse(os.path.exists(where))

    def test_the_discard_prompt_warns_when_nothing_could_be_autosaved(self):
        self.w._autosave_path = lambda: None
        self.w._autosave_fallback_path = lambda: None
        self.w._on_accept(1)
        self.assertIsNone(self.w._autosave_written_to)

    def test_only_reviewed_events_count_as_unsaved_work(self):
        self.assertEqual(self.w._reviewed_events(), [])
        self.w._on_accept(1)
        self.assertEqual(len(self.w._reviewed_events()), 1)

    # ---------------------------------------------------------------- B5
    def test_export_refuses_to_overwrite_the_reference_file(self):
        ref = self.w._reference_path()
        self.assertTrue(ref.endswith('.csv_bi'))
        self.assertEqual(os.path.basename(ref), 'rec.csv_bi')

        with open(ref, 'w') as f:
            f.write('# reference\n')
        with open(ref) as f:
            before = f.read()

        critical = []
        QtWidgets.QMessageBox.critical = staticmethod(
            lambda *a, **k: critical.append(a[1]))
        self.w._export_preflight = lambda: True
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(
            lambda *a, **k: (ref, ''))

        self.w._on_accept(1)
        self.w._export_reviewed()

        with open(ref) as f:
            after = f.read()
        self.assertEqual(after, before, 'reference was overwritten')
        self.assertTrue(critical, 'no refusal was shown')

    def test_export_writes_when_the_target_is_not_the_reference(self):
        out = os.path.join(self.tmp, 'rec.reviewed.csv_bi')
        self.w._export_preflight = lambda: True
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(
            lambda *a, **k: (out, ''))

        self.w._on_accept(1)
        self.w._export_reviewed()

        self.assertTrue(os.path.exists(out))
        with open(out) as f:
            text = f.read()
        self.assertIn('# status = ', text)          # disclaimer survives
        self.assertIn('TERM,', text)                 # the event was written
        self.assertFalse(self.w._dirty)              # session is clean again


if __name__ == '__main__':
    unittest.main()
