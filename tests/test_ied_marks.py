"""Interictal marks must never be mistakable for a reference annotation.

The reviewing neurologist asked for interictal discharges alongside seizures.
The detector cannot supply them, so the reviewer marks them — and the question
that decides the design is where those marks are written.

Not in the `.csv_bi`. That format is binary and term-based by construction:
two labels, `seiz` and `bckg`, one label across all channels. Interictal marks
are per-channel and are neither label. And no NEDC format takes them: the TUSZ
multi-class `.csv` label set is seizure TYPES only, and the spike label `spsw`
belongs to TUEV, a different corpus.

`spsw` is also one character from `spsz` — "simple partial seizure", a real
TUSZ label. In a seizure-review tool that is a genuinely dangerous near-miss,
and it is why these marks get their own file with a format marker that is
deliberately not a corpus name.

The export's claim is that a reviewed `.csv_bi` is indistinguishable from a
TUSZ reference. These tests exist to keep that true.
"""
import datetime as dt
import io
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.ied_marks import (                       # noqa: E402
    FORMAT_ID, LABEL, SOURCE_REVIEWER, ied_path_for, read_ied_marks, summarise,
    write_ied_marks)

START = dt.datetime(2002, 10, 1, 10, 58, 46)


def _marks():
    return [
        {'time': 120.5, 'channel': 'Fp1-F7', 'note': 'sharp'},
        {'time': 12.25, 'channel': 'T3-T5'},
        {'time': 300.0, 'channel': 'Fp1-F7', 'note': ''},
    ]


class TheFileCannotBeMistakenForAReference(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='ied')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, 'r.ied.csv')
        write_ied_marks(self.path, 'r', 600.0, _marks(), reviewer='SG',
                        start_time=START)
        with open(self.path) as f:
            self.text = f.read()

    def test_it_declares_a_format_that_is_not_a_corpus_name(self):
        self.assertIn('# format = {}'.format(FORMAT_ID), self.text)
        # The FORMAT MARKER is what a reader keys off; the prose below it is
        # allowed to name TUSZ precisely because it is saying this is not one.
        for corpus in ('tusz', 'nedc', 'tuev', 'csv_v1.0.0'):
            self.assertNotIn(corpus, FORMAT_ID.lower())
        self.assertNotIn('# version = csv', self.text.lower(),
                         'that header belongs to csv_bi and would make a '
                         'reader parse this as a TUSZ reference')

    def test_it_never_uses_a_tusz_or_tuev_label(self):
        """`spsw` is TUEV's and is one character from TUSZ's `spsz`. Neither
        may appear in the DATA — the tool's own name legitimately contains
        'Seizure', so this checks the rows, which is what a parser reads."""
        rows = [ln for ln in self.text.splitlines()
                if ln and not ln.startswith('#')
                and not ln.startswith('channel,')]
        self.assertTrue(rows)
        for row in rows:
            label = row.split(',')[2]
            self.assertEqual(label, LABEL)
            for bad in ('spsw', 'spsz', 'seiz', 'bckg', 'fnsz', 'gnsz',
                        'cpsz'):
                self.assertNotIn(bad, row.lower(),
                                 '{!r} must not appear in an interictal '
                                 'row'.format(bad))
        self.assertEqual(LABEL, 'ied')

    def test_it_carries_the_not_for_clinical_use_header(self):
        self.assertIn('NOT for diagnostic or clinical use', self.text)

    def test_it_says_it_is_not_ground_truth(self):
        """v1 could claim the file was "not a detector output", because
        nothing could produce these but a human. v2 carries SpikeNet
        candidates too, so that claim would now be false. The protection it
        provided has to be replaced by a stronger one: every row says where it
        came from, and the header says an automatic source is unvalidated."""
        self.assertIn('NOT ground truth', self.text)
        self.assertIn('UNVALIDATED', self.text)
        self.assertIn('source column', self.text)
        self.assertNotIn('not a detector output', self.text,
                         'the file can now carry detector candidates, so '
                         'this claim would be a lie')

    def test_it_sits_beside_the_recording_under_its_own_extension(self):
        self.assertEqual(ied_path_for('/x/y.edf'), '/x/y.ied.csv')
        self.assertEqual(ied_path_for('/x/y_reviewed.csv_bi'),
                         '/x/y_reviewed.ied.csv')
        self.assertEqual(ied_path_for('/x/y.review.edf'), '/x/y.ied.csv')


class MarksRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='ied')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, 'r.ied.csv')

    def test_what_goes_in_comes_back(self):
        write_ied_marks(self.path, 'r', 600.0, _marks())
        got = read_ied_marks(self.path)
        self.assertEqual(len(got), 3)
        self.assertEqual([round(m['time'], 2) for m in got],
                         [12.25, 120.5, 300.0], 'marks must be time-ordered')
        self.assertEqual(got[0]['channel'], 'T3-T5')
        self.assertTrue(all(m['label'] == LABEL for m in got))

    def test_a_mark_is_a_point_not_an_interval(self):
        """A spike is tens of milliseconds and the reviewer clicks where they
        see it. Inventing a duration would be inventing data."""
        write_ied_marks(self.path, 'r', 600.0, _marks())
        with open(self.path) as f:
            header = [ln for ln in f if ln.startswith('channel,')][0]
        self.assertNotIn('duration', header)
        self.assertNotIn('stop', header)

    def test_a_note_with_a_comma_does_not_break_the_row(self):
        write_ied_marks(self.path, 'r', 600.0,
                        [{'time': 1.0, 'channel': 'Fp1-F7',
                          'note': 'sharp, then slow'}])
        got = read_ied_marks(self.path)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]['channel'], 'Fp1-F7')

    def test_no_marks_still_writes_a_readable_file(self):
        n = write_ied_marks(self.path, 'r', 600.0, [])
        self.assertEqual(n, 0)
        self.assertEqual(read_ied_marks(self.path), [])
        with open(self.path) as f:
            self.assertIn('# format', f.read())

    def test_reading_a_file_that_is_not_there_is_not_an_error(self):
        self.assertEqual(read_ied_marks(os.path.join(self.tmp, 'no.csv')), [])


class WhatAReviewerWantsCounted(unittest.TestCase):
    def test_total_and_per_channel(self):
        s = summarise(_marks())
        self.assertEqual(s['total'], 3)
        self.assertEqual(s['per_channel']['Fp1-F7'], 2)
        self.assertEqual(s['per_channel']['T3-T5'], 1)

    def test_channels_are_ordered_by_burden(self):
        """Where the spikes are is most of the clinical information, so the
        busiest derivation comes first."""
        s = summarise(_marks())
        self.assertEqual(list(s['per_channel'])[0], 'Fp1-F7')

    def test_nothing_marked_is_reported_as_nothing(self):
        s = summarise([])
        self.assertEqual(s['total'], 0)
        self.assertEqual(s['per_channel'], {})
        # v2 added the source split; an empty recording has none of either.
        self.assertEqual(s['reviewer'], 0)
        self.assertEqual(s['automatic'], 0)


class ReviewerMarksAndMachineCandidatesStayApart(unittest.TestCase):
    """The whole point of v2. A detector candidate that has not been confirmed
    must never be countable, or readable, as something a clinician asserted."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='ied2')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, 'r.ied.csv')
        self.marks = [
            {'time': 10.0, 'channel': 'Fp1-F7', 'note': 'sharp'},
            {'time': 20.0, 'channel': None, 'source': 'spikenet1',
             'score': 0.97, 'note': 'auto, unconfirmed'},
            {'time': 30.0, 'channel': None, 'source': 'spikenet1',
             'score': 0.88, 'note': 'auto, unconfirmed'},
        ]
        write_ied_marks(self.path, 'r', 600.0, self.marks)
        self.back = read_ied_marks(self.path)

    def test_a_mark_without_a_source_is_the_reviewer(self):
        self.assertEqual(self.back[0]['source'], SOURCE_REVIEWER)

    def test_the_detector_is_named_on_its_own_rows(self):
        self.assertEqual(self.back[1]['source'], 'spikenet1')
        self.assertEqual(self.back[2]['source'], 'spikenet1')

    def test_the_score_round_trips(self):
        self.assertAlmostEqual(self.back[1]['score'], 0.97, places=4)

    def test_a_reviewer_mark_has_no_score(self):
        self.assertIsNone(self.back[0]['score'])

    def test_the_two_are_counted_separately(self):
        s = summarise(self.back)
        self.assertEqual(s['total'], 3)
        self.assertEqual(s['reviewer'], 1)
        self.assertEqual(s['automatic'], 2)
        self.assertEqual(s['per_source']['spikenet1'], 2)

    def test_the_label_column_is_still_where_a_parser_expects_it(self):
        rows = [ln for ln in io.open(self.path).read().splitlines()
                if ln and not ln.startswith('#')
                and not ln.startswith('channel,')]
        for row in rows:
            self.assertEqual(row.split(',')[2], LABEL)


V1_FILE = """# format = continental_ied_v1
#
channel,time,label,note
Fp1-F7,12.5000,ied,sharp
"""


class V1FilesStillRead(unittest.TestCase):
    """Marks exported before the detector existed must not be silently
    reinterpreted — v1 put the NOTE where v2 puts the SOURCE, so reading one
    as the other would turn 'sharp' into a detector name."""

    def test_a_v1_file_reads_as_reviewer_marks_with_its_notes_intact(self):
        tmp = tempfile.mkdtemp(prefix='ied1')
        self.addCleanup(shutil.rmtree, tmp, True)
        p = os.path.join(tmp, 'old.ied.csv')
        with io.open(p, 'w') as f:
            f.write(V1_FILE)
        back = read_ied_marks(p)
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]['note'], 'sharp')
        self.assertEqual(back[0]['source'], SOURCE_REVIEWER)
        self.assertIsNone(back[0]['score'])


if __name__ == '__main__':
    unittest.main()


class MarksBelongToOneRecordingAndOneMontage(unittest.TestCase):
    """Two bugs this pins, both found by audit after the feature was written.

    A mark names the derivation it was seen on, and that name only means
    something inside one montage — 'Fp1-F7' has no counterpart in a referential
    view. So the montage travels with the mark; without it an exported mark
    cannot be interpreted later, and a reviewer who changed montage would watch
    their marks silently disappear from the trace.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='ied')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, 'r.ied.csv')

    def test_the_writer_accepts_and_ignores_extra_keys(self):
        """The app attaches `montage`; the file format does not carry it yet,
        and the writer must not choke on a key it does not write."""
        marks = [{'time': 5.0, 'channel': 'Fp1-F7',
                  'montage': 'Longitudinal Bipolar', 'note': ''}]
        write_ied_marks(self.path, 'r', 60.0, marks)
        self.assertEqual(len(read_ied_marks(self.path)), 1)

    def test_marks_without_a_channel_are_still_written(self):
        """Degraded rather than dropped: losing a reviewer's mark because a
        field is missing is worse than writing it with a blank channel.

        Read back as None rather than '' since v2, because SpikeNet localises
        nothing and every one of its marks arrives without a channel. None
        says "not known"; '' would be a derivation whose name is empty, and
        summarise would have grouped them all under it."""
        write_ied_marks(self.path, 'r', 60.0, [{'time': 5.0}])
        got = read_ied_marks(self.path)
        self.assertEqual(len(got), 1)
        self.assertIsNone(got[0]['channel'])

    def test_summarise_copes_with_a_missing_channel(self):
        s = summarise([{'time': 1.0}, {'time': 2.0, 'channel': 'T3-T5'}])
        self.assertEqual(s['total'], 2)
        self.assertIn('(unknown)', s['per_channel'])
