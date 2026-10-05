import unittest

from gui.events import clamp_interval, rebuild_events


class GuiEventTests(unittest.TestCase):
    def test_single_positive_window_uses_full_segment_span(self):
        events = rebuild_events(
            [0, 6], [0.9, 0.1],
            threshold=0.5, segment_s=12,
            preserve_review=False)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['start'], 0.0)
        self.assertEqual(events[0]['stop'], 12.0)

    def test_threshold_rebuild_preserves_reviewed_event(self):
        previous = [{
            'id': 7, 'start': 10.0, 'stop': 22.0,
            'source_start': 12.0, 'source_stop': 24.0,
            'prob': 0.95, 'status': 'accepted',
        }]

        events = rebuild_events(
            [12, 18, 24], [0.9, 0.8, 0.1],
            threshold=0.7, segment_s=12,
            duration_s=100,
            previous_events=previous,
            preserve_review=True)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['id'], 7)
        self.assertEqual(events[0]['status'], 'accepted')
        self.assertEqual(events[0]['start'], 10.0)
        self.assertEqual(events[0]['stop'], 22.0)

    def test_reviewed_event_survives_when_threshold_excludes_it(self):
        previous = [{
            'id': 3, 'start': 4.0, 'stop': 16.0,
            'prob': 0.6, 'status': 'edited',
        }]

        events = rebuild_events(
            [0, 6], [0.4, 0.3],
            threshold=0.8, segment_s=12,
            duration_s=30,
            previous_events=previous,
            preserve_review=True)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['id'], 3)
        self.assertEqual(events[0]['status'], 'edited')

    def test_clamp_interval_sorts_and_bounds_edits(self):
        self.assertEqual(clamp_interval(50, -5, duration_s=40), (0.0, 40.0))


if __name__ == '__main__':
    unittest.main()
