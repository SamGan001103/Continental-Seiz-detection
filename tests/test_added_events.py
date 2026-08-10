"""Human-originated events and undo.

These exist because the thesis's primary figure — reviewer-in-the-loop event
recovery — is the count of seizures the reviewer found that the detector did
not. Before this, no such event could be created: the worklist derived entirely
from suprathreshold runs and export filtered to accepted/edited, so a reviewer
could only subtract. The figure was unmeasurable and the tool inherited the
model's 49 % sensitivity ceiling.
"""
import os
import shutil
import tempfile
import unittest

from gui.events import (
    EXPORTED_STATUSES, REVIEWED_STATUSES, merge_review_state,
)

try:
    from PyQt5 import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.app import MainWindow
    HAVE_QT = True
except Exception:                                    # pragma: no cover
    HAVE_QT = False


class EventModelTests(unittest.TestCase):
    def test_added_is_reviewed_so_it_survives_a_threshold_rebuild(self):
        self.assertIn('added', REVIEWED_STATUSES)

    def test_added_is_exported(self):
        self.assertIn('added', EXPORTED_STATUSES)

    def test_rejected_is_not_exported(self):
        self.assertNotIn('rejected', EXPORTED_STATUSES)

    def test_a_manual_event_survives_a_rebuild_that_proposes_nothing(self):
        """The case that matters: the detector found nothing at all."""
        added = {'id': 1, 'start': 90.0, 'stop': 110.0, 'prob': None,
                 'status': 'added', 'source_start': 90.0, 'source_stop': 110.0}
        merged = merge_review_state([], [added], duration_s=200.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['status'], 'added')

    def test_a_manual_event_keeps_prob_none_when_absorbed_by_a_proposal(self):
        """Lowering the threshold may make the AI propose the same region.

        The human keeps ownership, and the score must not become 0.0 — that
        would read as a confident negative in the list and the ledger.
        """
        added = {'id': 1, 'start': 90.0, 'stop': 110.0, 'prob': None,
                 'status': 'added', 'source_start': 90.0, 'source_stop': 110.0}
        prop = {'start': 92.0, 'stop': 108.0, 'prob': 0.6, 'status': 'proposed',
                'source_start': 92.0, 'source_stop': 108.0}
        merged = merge_review_state([prop], [added], duration_s=200.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['status'], 'added')
        self.assertEqual(merged[0]['prob'], 0.6)     # real score now known

    def test_prob_stays_none_when_nothing_scored_it(self):
        added = {'id': 1, 'start': 90.0, 'stop': 110.0, 'prob': None,
                 'status': 'added', 'source_start': 90.0, 'source_stop': 110.0}
        merged = merge_review_state([], [added], duration_s=200.0)
        self.assertIsNone(merged[0]['prob'])


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class AddAndUndoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.w = MainWindow()
        self.w._reviewer_id = 'test'
        self.w._edf_path = os.path.join(self.tmp, 'rec.edf')
        self.w._duration_s = 300.0
        self.w._events = []
        self.w._undo_stack = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add(self):
        self.w.signal_view.center_on(100.0, half_span=15.0)
        self.w._add_event_at_view()

    def test_adding_creates_an_event_with_no_model_score(self):
        self._add()
        self.assertEqual(len(self.w._events), 1)
        ev = self.w._events[0]
        self.assertEqual(ev['status'], 'added')
        self.assertIsNone(ev['prob'])

    def test_adding_marks_the_session_dirty(self):
        self._add()
        self.assertTrue(self.w._dirty)

    def test_undo_removes_an_added_event(self):
        self._add()
        self.w._undo()
        self.assertEqual(len(self.w._events), 0)

    def test_undo_restores_a_rejected_event(self):
        self.w._events = [{'id': 1, 'start': 10.0, 'stop': 20.0, 'prob': 0.9,
                           'status': 'proposed'}]
        self.w._on_reject(1)
        self.assertEqual(self.w._events[0]['status'], 'rejected')
        self.w._undo()
        self.assertEqual(self.w._events[0]['status'], 'proposed')

    def test_undo_restores_an_accepted_event(self):
        self.w._events = [{'id': 1, 'start': 10.0, 'stop': 20.0, 'prob': 0.9,
                           'status': 'proposed'}]
        self.w._on_accept(1)
        self.w._undo()
        self.assertEqual(self.w._events[0]['status'], 'proposed')

    def test_undo_on_an_empty_stack_is_harmless(self):
        self.w._undo()          # must not raise
        self.assertEqual(len(self.w._events), 0)

    def test_undo_stack_is_bounded(self):
        self.w._events = [{'id': 1, 'start': 10.0, 'stop': 20.0, 'prob': 0.9,
                           'status': 'proposed'}]
        for _ in range(self.w.UNDO_DEPTH + 20):
            self.w._push_undo('x')
        self.assertLessEqual(len(self.w._undo_stack), self.w.UNDO_DEPTH)

    def test_revert_restores_the_detector_extent(self):
        self.w._events = [{'id': 1, 'start': 12.0, 'stop': 25.0, 'prob': 0.9,
                           'status': 'edited',
                           'source_start': 10.0, 'source_stop': 20.0}]
        self.w.event_list.set_events(self.w._events)
        self.w._revert_extent(1)
        ev = self.w._events[0]
        self.assertAlmostEqual(ev['start'], 10.0)
        self.assertAlmostEqual(ev['stop'], 20.0)
        self.assertEqual(ev['status'], 'proposed')

    def test_revert_declines_on_a_human_added_event(self):
        """There is no detector extent to go back to."""
        self._add()
        eid = self.w._events[0]['id']
        before = (self.w._events[0]['start'], self.w._events[0]['stop'])
        self.w._revert_extent(eid)
        after = (self.w._events[0]['start'], self.w._events[0]['stop'])
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
