"""The EDF+ companion must still contain the events when someone opens it.

A reviewing neurologist asked what export format would be best. The honest
survey answer is that no clinical EEG system imports third-party seizure
annotations, so the useful target is the container the recording already came
in: EDF+, whose annotation signal EDFbrowser, MNE, MATLAB and EDFlib all read.

The reason this file has tests at all is that the obvious implementation is
wrong in a way nothing casual catches. `pyedflib` writes one data record per
annotation, so a file with three events declares itself three seconds long. It
reads back perfectly through `pyedflib.readAnnotations` AND through
`mne.read_annotations`. Only `mne.io.read_raw_edf` — the most likely way anyone
opens it — notices, and it discards every out-of-range event with a
`RuntimeWarning` rather than an error. On a day-long recording that is every
event but the first.

So the writer pads the record count, and the test below asserts the
`read_raw_edf` path specifically. An assertion on `readAnnotations` alone would
pass against the broken version.
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.edf_review_annotations import (          # noqa: E402
    DISCLAIMER, MAX_ANNOTATION_CHARS, review_edf_path_for, write_review_edf)

DAY_S = 41030.0            # a real EPILEPSIAE part length
START = dt.datetime(2002, 10, 1, 10, 58, 46)


def _events():
    return [
        {'start': 12.5, 'stop': 70.6, 'status': 'accepted', 'prob': 0.91},
        {'start': 6235.0, 'stop': 6290.1, 'status': 'edited', 'prob': 0.42},
        {'start': 41000.0, 'stop': 41055.0, 'status': 'added', 'prob': None},
    ]


class EventsSurviveTheReaderPeopleActuallyUse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='reviewedf')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, 'r.review.edf')

    def test_mne_read_raw_edf_keeps_every_event(self):
        """THE test. Without padding this returns one annotation of four and
        warns; with padding it returns all of them."""
        mne = _mne()
        write_review_edf(self.path, _events(), DAY_S, START)
        raw = mne.io.read_raw_edf(self.path, preload=False, verbose='ERROR')
        onsets = sorted(round(float(o), 1) for o in raw.annotations.onset)
        for expected in (12.5, 6235.0, 41000.0):
            self.assertIn(expected, onsets,
                          'event at {} s was dropped by read_raw_edf — the '
                          'record count is not covering the recording'.format(
                              expected))

    def test_the_declared_duration_covers_the_recording(self):
        write_review_edf(self.path, _events(), DAY_S, START)
        import pyedflib
        r = pyedflib.EdfReader(self.path)
        try:
            self.assertGreaterEqual(
                r.file_duration, DAY_S,
                'the file declares itself shorter than the recording it '
                'describes, which is what makes readers discard events')
        finally:
            r.close()

    def test_pyedflib_reads_them_back_too(self):
        write_review_edf(self.path, _events(), DAY_S, START)
        import pyedflib
        r = pyedflib.EdfReader(self.path)
        try:
            onsets, _durs, texts = r.readAnnotations()
        finally:
            r.close()
        self.assertIn(DISCLAIMER, [str(t) for t in texts])
        got = sorted(round(float(o), 1) for o in onsets)
        for expected in (12.5, 6235.0, 41000.0):
            self.assertIn(expected, got)

    def test_a_short_recording_does_not_get_padded_pointlessly(self):
        write_review_edf(self.path, _events()[:1], 30.0, START)
        self.assertLess(os.path.getsize(self.path), 4096)


class TheFileSaysWhatItIs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='reviewedf')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, 'r.review.edf')

    def test_the_disclaimer_is_the_first_annotation(self):
        write_review_edf(self.path, _events(), DAY_S, START)
        import pyedflib
        r = pyedflib.EdfReader(self.path)
        try:
            onsets, _d, texts = r.readAnnotations()
        finally:
            r.close()
        first = [str(t) for o, t in sorted(zip(onsets, texts)) if str(t)][0]
        self.assertEqual(first, DISCLAIMER)

    def test_the_disclaimer_is_also_in_the_header(self):
        """An annotation can be discarded by a strict reader; the header field
        cannot. The safety marker goes in both."""
        write_review_edf(self.path, _events(), DAY_S, START)
        import pyedflib
        r = pyedflib.EdfReader(self.path)
        try:
            self.assertIn('NOT FOR CLINICAL USE', r.getEquipment())
        finally:
            r.close()

    def test_the_start_time_is_the_recordings_own(self):
        write_review_edf(self.path, _events(), DAY_S, START)
        import pyedflib
        r = pyedflib.EdfReader(self.path)
        try:
            self.assertEqual(r.getStartdatetime(), START)
        finally:
            r.close()

    def test_labels_stay_inside_the_forty_character_ceiling(self):
        """pyedflib truncates silently at 40. A label that reaches the file
        cut in half would be worse than a short one written on purpose."""
        write_review_edf(self.path, _events(), DAY_S, START)
        import pyedflib
        r = pyedflib.EdfReader(self.path)
        try:
            _o, _d, texts = r.readAnnotations()
        finally:
            r.close()
        for t in texts:
            self.assertLessEqual(len(str(t)), MAX_ANNOTATION_CHARS)

    def test_a_reviewer_added_event_says_so(self):
        write_review_edf(self.path, _events(), DAY_S, START)
        import pyedflib
        r = pyedflib.EdfReader(self.path)
        try:
            _o, _d, texts = r.readAnnotations()
        finally:
            r.close()
        joined = ' | '.join(str(t) for t in texts)
        self.assertIn('added by reviewer', joined,
                      'an event the detector never proposed must be '
                      'distinguishable from one it did')


class Naming(unittest.TestCase):
    def test_it_sits_beside_the_annotation_file(self):
        self.assertEqual(review_edf_path_for('/x/y_reviewed.csv_bi'),
                         '/x/y_reviewed.review.edf')

    def test_a_plain_csv_works_too(self):
        self.assertEqual(review_edf_path_for('/x/y.csv'), '/x/y.review.edf')


def _mne():
    import mne
    return mne


if __name__ == '__main__':
    unittest.main()


class OnlyConfirmedEventsReachTheCompanion(unittest.TestCase):
    """A rejected event must never appear in the EDF+ file.

    The csv_bi carries only EXPORTED_STATUSES — accepted, edited, added. The
    companion has to carry exactly the same set. The natural mistake when
    wiring it up is to hand it the whole event list, which is the one the
    reviewer's REJECTIONS are also in, and the file would then assert that the
    reviewer confirmed seizures they explicitly threw out.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='reviewedf')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, 'r.review.edf')

    def _texts(self):
        import pyedflib
        r = pyedflib.EdfReader(self.path)
        try:
            onsets, _d, texts = r.readAnnotations()
        finally:
            r.close()
        return [(round(float(o), 1), str(t)) for o, t in zip(onsets, texts)]

    def test_the_writer_only_writes_what_it_is_given(self):
        confirmed = [{'start': 10.0, 'stop': 40.0, 'status': 'accepted',
                      'prob': 0.8}]
        write_review_edf(self.path, confirmed, 600.0, START)
        onsets = [o for o, t in self._texts() if t]
        self.assertIn(10.0, onsets)
        self.assertNotIn(999.0, onsets)

    def test_the_status_of_each_event_is_carried_through(self):
        evs = [{'start': 10.0, 'stop': 40.0, 'status': 'accepted',
                'prob': 0.8},
               {'start': 100.0, 'stop': 130.0, 'status': 'added',
                'prob': None}]
        write_review_edf(self.path, evs, 600.0, START)
        got = dict(self._texts())
        self.assertIn('accepted', got.get(10.0, ''))
        self.assertIn('added by reviewer', got.get(100.0, ''))

    def test_an_export_with_nothing_confirmed_is_still_a_valid_file(self):
        write_review_edf(self.path, [], 600.0, START)
        texts = [t for _o, t in self._texts() if t]
        self.assertEqual(texts, [DISCLAIMER],
                         'an empty review must still say what the file is')
