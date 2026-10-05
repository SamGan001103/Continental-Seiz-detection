"""The EPILEPSIAE converter must not quietly move the signal in time.

The source recordings are one-hour blocks that are NOT contiguous. Measured on
the three complete patients supplied by the lab:

    pat_102     173 blocks, 5,465 s of gap, one pair overlapping by 60 s
    pat_21902    97 blocks, 1,353 s of gap, two gaps of ~10 minutes
    pat_22602   118 blocks,   788 s of gap

A converter that appends block after block produces an EDF of exactly the right
shape, full of real signal, in which every seizure after the first gap is at the
wrong time — by up to ninety minutes for pat_102. Nothing raises, nothing looks
wrong, and the evaluation silently measures the wrong seconds.

So these tests are about placement, not about EDF. They run on synthetic block
lists, need neither the 20 GB archive nor pyedflib, and each one pins a
property that a naive implementation gets wrong.
"""
import datetime as dt
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from experiments.epilepsiae_to_edf import (           # noqa: E402
    plan_timeline, seizures_in_part)

FS = 256
HOUR = 3600
T0 = dt.datetime(2002, 9, 30, 10, 58, 46)


def block(name, t0, seconds=HOUR):
    return {'filename': name, 't0': t0, 'n_samples': seconds * FS, 'fs': FS,
            'n_channels': 29, 'factor': 0.165}


def chain(n, gap_s=0.0, start=T0):
    """n one-hour blocks with a constant gap between them."""
    out, t = [], start
    for i in range(n):
        out.append(block('b{:04d}.data'.format(i), t))
        t = t + dt.timedelta(seconds=HOUR + gap_s)
    return out


def all_names(blocks):
    return {b['filename'] for b in blocks}


def part_end_s(part):
    return part['start_s'] + part['duration_s']


class PlacementFollowsTheClock(unittest.TestCase):
    def test_contiguous_blocks_produce_no_gaps(self):
        blocks = chain(5)
        parts, gaps = plan_timeline(blocks, all_names(blocks), split_hours=0)
        self.assertEqual(gaps, [])
        self.assertAlmostEqual(parts[0]['duration_s'], 5 * HOUR, places=6)

    def test_a_one_second_gap_is_recorded_and_not_absorbed(self):
        blocks = chain(3, gap_s=1.0)
        parts, gaps = plan_timeline(blocks, all_names(blocks), split_hours=0)
        self.assertEqual(len(gaps), 2)
        self.assertTrue(all(abs(g['seconds'] - 1.0) < 1e-6 for g in gaps))
        # Wall-clock span, not signal length: 3 hours plus the two lost seconds.
        self.assertAlmostEqual(parts[0]['duration_s'], 3 * HOUR + 2, places=6)

    def test_the_drift_that_motivates_this_module(self):
        """173 blocks at one second apiece is pat_102's real shape."""
        blocks = chain(173, gap_s=1.0)
        parts, _ = plan_timeline(blocks, all_names(blocks), split_hours=0)
        span = parts[0]['duration_s']
        naive = 173 * HOUR              # what appending would produce
        self.assertAlmostEqual(span - naive, 172.0, places=6,
                               msg='the timeline must carry the lost time, '
                                   'otherwise the last block starts 172 s '
                                   'earlier than it really did')
        last = max(parts[0]['pieces'], key=lambda p: p[1])
        self.assertAlmostEqual(last[1] / float(FS), 172 * (HOUR + 1),
                               places=3)

    def test_a_large_gap_keeps_later_blocks_in_the_right_place(self):
        blocks = chain(2) + [block('late.data',
                                   T0 + dt.timedelta(seconds=2 * HOUR + 4290))]
        parts, gaps = plan_timeline(blocks, all_names(blocks), split_hours=0)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0]['seconds'], 4290.0, places=3)
        late = [p for p in parts[0]['pieces'] if p[0] == 'late.data'][0]
        self.assertAlmostEqual(late[1] / float(FS), 2 * HOUR + 4290, places=3)

    def test_a_missing_block_becomes_a_gap_not_a_shortening(self):
        """An hour absent from the truncated archive must still cost an hour
        of timeline, or everything after it slides an hour early."""
        blocks = chain(4)
        present = all_names(blocks) - {'b0001.data'}
        parts, gaps = plan_timeline(blocks, present, split_hours=0)
        self.assertAlmostEqual(parts[0]['duration_s'], 4 * HOUR, places=6)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0]['seconds'], HOUR, places=6)
        third = [p for p in parts[0]['pieces'] if p[0] == 'b0002.data'][0]
        self.assertAlmostEqual(third[1] / float(FS), 2 * HOUR, places=3)


class OverlapsNeverRewindTheClock(unittest.TestCase):
    def test_an_overlapping_block_is_truncated_at_the_front(self):
        """pat_102 really contains a pair overlapping by 60 s."""
        blocks = [block('a.data', T0),
                  block('b.data', T0 + dt.timedelta(seconds=HOUR - 60))]
        parts, gaps = plan_timeline(blocks, all_names(blocks), split_hours=0)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]['kind'], 'overlap')
        pieces = {p[0]: p for p in parts[0]['pieces']}
        self.assertEqual(pieces['b.data'][2], 60 * FS,
                         'the overlapping head of the later block must be '
                         'skipped, not written twice')
        self.assertAlmostEqual(pieces['b.data'][1] / float(FS), HOUR,
                               places=3)
        self.assertAlmostEqual(parts[0]['duration_s'], 2 * HOUR - 60,
                               places=3)

    def test_the_timeline_is_strictly_monotonic(self):
        blocks = [block('a.data', T0),
                  block('b.data', T0 + dt.timedelta(seconds=HOUR - 60)),
                  block('c.data', T0 + dt.timedelta(seconds=2 * HOUR))]
        parts, _ = plan_timeline(blocks, all_names(blocks), split_hours=0)
        ends = []
        for _, off, _, take, _ in sorted(parts[0]['pieces'],
                                         key=lambda p: p[1]):
            if ends:
                self.assertGreaterEqual(off, ends[-1],
                                        'pieces must not overlap in the '
                                        'output timeline')
            ends.append(off + take)


class SplittingIntoDayFiles(unittest.TestCase):
    def test_parts_tile_the_recording_without_overlap_or_hole(self):
        blocks = chain(50)
        parts, _ = plan_timeline(blocks, all_names(blocks), split_hours=24)
        self.assertEqual(len(parts), 3)
        self.assertAlmostEqual(parts[0]['start_s'], 0.0)
        for a, b in zip(parts, parts[1:]):
            self.assertAlmostEqual(part_end_s(a), b['start_s'], places=6)
        self.assertAlmostEqual(part_end_s(parts[-1]), 50 * HOUR, places=6)

    def test_a_block_spanning_a_boundary_appears_in_both_parts(self):
        blocks = chain(25)
        parts, _ = plan_timeline(blocks, all_names(blocks), split_hours=24)
        names0 = {p[0] for p in parts[0]['pieces']}
        names1 = {p[0] for p in parts[1]['pieces']}
        self.assertEqual(len(parts), 2)
        self.assertIn('b0024.data', names1)
        self.assertNotIn('b0024.data', names0)

    def test_split_hours_zero_means_one_file(self):
        blocks = chain(30)
        parts, _ = plan_timeline(blocks, all_names(blocks), split_hours=0)
        self.assertEqual(len(parts), 1)


class SeizuresLandWhereTheyHappened(unittest.TestCase):
    def _parts(self, n=50, gap=1.0, split=24):
        blocks = chain(n, gap_s=gap)
        return plan_timeline(blocks, all_names(blocks), split_hours=split)[0]

    def test_a_seizure_is_placed_by_absolute_time(self):
        parts = self._parts()
        onset = T0 + dt.timedelta(seconds=30 * HOUR)
        ev = seizures_in_part([{'onset': onset, 'duration': 55.1,
                                'label': 'CP'}], parts[1])
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev[0]['start'], 30 * HOUR - 24 * HOUR, places=3)
        self.assertAlmostEqual(ev[0]['stop'] - ev[0]['start'], 55.1, places=3)

    def test_a_seizure_outside_a_part_is_not_reported_in_it(self):
        parts = self._parts()
        onset = T0 + dt.timedelta(seconds=30 * HOUR)
        self.assertEqual(
            seizures_in_part([{'onset': onset, 'duration': 55.1,
                               'label': 'CP'}], parts[0]), [])

    def test_a_seizure_straddling_a_boundary_is_clipped_into_both(self):
        parts = self._parts()
        onset = T0 + dt.timedelta(seconds=24 * HOUR - 20)
        sz = [{'onset': onset, 'duration': 60.0, 'label': 'SG'}]
        first = seizures_in_part(sz, parts[0])
        second = seizures_in_part(sz, parts[1])
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertTrue(first[0]['clipped'] and second[0]['clipped'])
        self.assertAlmostEqual(first[0]['stop'], parts[0]['duration_s'],
                               places=3)
        self.assertAlmostEqual(second[0]['start'], 0.0, places=3)
        total = ((first[0]['stop'] - first[0]['start']) +
                 (second[0]['stop'] - second[0]['start']))
        self.assertAlmostEqual(total, 60.0, places=3)

    def test_gaps_shift_a_later_seizure_and_the_plan_follows(self):
        """With one second lost per hour, a seizure 30 hours in by the CLOCK is
        not 30 hours in by SAMPLE COUNT. The absolute placement is the one that
        matches the annotation."""
        parts = self._parts(gap=1.0)
        onset = T0 + dt.timedelta(seconds=30 * (HOUR + 1))
        ev = seizures_in_part([{'onset': onset, 'duration': 10.0,
                                'label': 'UC'}], parts[1])
        self.assertEqual(len(ev), 1)
        expected = 30 * (HOUR + 1) - parts[1]['start_s']
        self.assertAlmostEqual(ev[0]['start'], expected, places=3)


if __name__ == '__main__':
    unittest.main()
