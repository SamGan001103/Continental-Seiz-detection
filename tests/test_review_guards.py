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


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class ShortcutsAreUnambiguous(unittest.TestCase):
    """Two registrations of one key sequence disable it, silently.

    Qt answers a duplicate with "QAction::event: Ambiguous shortcut overload"
    on stderr and then fires neither handler. Nothing raises, nothing is
    logged to the application's own log, and the menu item still shows the
    shortcut next to it — so the only way to notice is to press the key and
    watch nothing happen.

    Four keys were dead this way: N, Ctrl+Z, Ctrl+R and Ctrl+O, each declared
    once as a menu QAction and once again as a QShortcut. N is the only way to
    record a seizure the detector never proposed, which is the capability the
    human-in-the-loop argument rests on.
    """

    def test_no_key_sequence_is_registered_twice(self):
        from collections import Counter
        w = MainWindow()
        try:
            seqs = Counter()
            for a in w.findChildren(QtWidgets.QAction):
                for s in a.shortcuts():
                    if s.toString():
                        seqs[s.toString()] += 1
            for s in w.findChildren(QtWidgets.QShortcut):
                if s.key().toString():
                    seqs[s.key().toString()] += 1
            dupes = {k: v for k, v in seqs.items() if v > 1}
            self.assertEqual(dupes, {},
                             'these key sequences are registered more than '
                             'once, so Qt will fire none of them: {}'
                             .format(sorted(dupes)))
        finally:
            w.close()

    def test_the_add_missed_event_key_is_bound(self):
        """N specifically, because it is the one the thesis figure needs."""
        w = MainWindow()
        try:
            bound = [a for a in w.findChildren(QtWidgets.QAction)
                     if any(s.toString() == 'N' for s in a.shortcuts())]
            self.assertEqual(len(bound), 1,
                             'N should be bound exactly once; found %d'
                             % len(bound))
        finally:
            w.close()


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class AddedEventsCountAsReviewWork(unittest.TestCase):
    """A session of only added events is still a session.

    _reviewed_events() feeds the unsaved-work prompt, the autosave payload and
    the session summary. It listed accepted/rejected/edited and omitted
    'added', so a reviewer who only marked seizures the detector missed had an
    empty list: closing the window discarded the work without asking, and no
    autosave existed to recover it.
    """

    def test_added_events_are_review_work(self):
        w = MainWindow()
        try:
            w._events = [{'start': 10.0, 'stop': 22.0, 'status': 'added',
                          'prob': None}]
            self.assertEqual(len(w._reviewed_events()), 1,
                             'an added event is unexported review work and '
                             'must trigger the discard guard')
        finally:
            w.close()

    def test_every_exported_status_counts_as_review_work(self):
        from gui.events import EXPORTED_STATUSES
        w = MainWindow()
        try:
            for status in EXPORTED_STATUSES:
                w._events = [{'start': 0.0, 'stop': 12.0, 'status': status,
                              'prob': None}]
                self.assertEqual(
                    len(w._reviewed_events()), 1,
                    '%r is written on export but does not count as unsaved '
                    'work, so it can be discarded without a prompt' % status)
        finally:
            w.close()


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class ClosedPanelsCanBeReopened(unittest.TestCase):
    """A dock with no menu entry is gone for good, as far as a user is concerned.

    Qt puts every dock in a right-click menu on the toolbar, but nobody finds
    it — the electrode map was added without a View entry and the first person
    to close it had no way back. A panel that cannot be restored is worse than
    one that was never added: the reviewer now knows the feature exists and
    cannot reach it.
    """

    def test_every_dock_has_a_view_menu_entry(self):
        w = MainWindow()
        w.show()
        try:
            view = [m for m in w.menuBar().actions()
                    if 'View' in m.text()][0].menu()
            texts = ' '.join(a.text() for a in view.actions())
            for dock in w.findChildren(QtWidgets.QDockWidget):
                title = dock.windowTitle()
                key = title.split()[0].strip('&')
                self.assertIn(
                    key.lower(), texts.lower(),
                    'the {!r} dock has no View menu entry, so closing it is '
                    'irreversible for a user'.format(title))
        finally:
            w.close()

    def test_closing_and_reopening_the_electrode_map_works(self):
        w = MainWindow()
        w.show()
        try:
            view = [m for m in w.menuBar().actions()
                    if 'View' in m.text()][0].menu()
            act = [a for a in view.actions() if 'Electrode' in a.text()][0]
            self.assertTrue(w._head_dock.isVisible())
            w._head_dock.close()
            self.assertFalse(w._head_dock.isVisible())
            self.assertFalse(act.isChecked(),
                             'the menu entry must untick when the dock closes, '
                             'or it lies about the panel state')
            act.trigger()
            self.assertTrue(w._head_dock.isVisible(),
                            'the View menu entry did not bring the panel back')
        finally:
            w.close()
