"""Recording annotations are context, and must never become ground truth.

Real clinical recordings carry annotations inside the EDF+ — what the
technician typed while the study ran. We displayed none of them. The Nihon
Kohden recording Omid circulated has 41, and they are the highest-yield minutes
in the file: `HVT START` and `PHOTO 9Hz` mark deliberate seizure-activation
procedures, and `hoest` (cough) marks movement artefact a detector will happily
score. A reviewer judging a run of rhythmic activity without knowing a strobe
was flashing is missing the fact that decides the call.

Three properties are tested here because getting any of them wrong is worse
than not having the feature:

* **They are not ground truth.** They stay out of the event list and out of the
  export. A note about a cough promoted into a seizure annotation would corrupt
  the one file this application produces for other clinicians to read.

* **They never block a load.** A malformed or missing annotation channel gives
  an empty list, because a recording that opened yesterday must open today.

* **They are drawn unmistakably.** Green belongs to the reference seizures and
  the status palette to the detector's proposals; a third kind of mark sharing
  either colour is a mark that will be misread.
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.edf_annotations import (annotation_summary,        # noqa: E402
                                    read_edf_annotations)

try:
    import PyQt5  # noqa: F401
    HAVE_QT = True
except Exception:                                              # noqa: BLE001
    HAVE_QT = False

if HAVE_QT:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtWidgets

FS = 250
CH = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ',
      'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']


def _write_edf_with_annotations(path, annots, seconds=60):
    import pyedflib
    labels = ['EEG {}-REF'.format(c) for c in CH]
    sigs = np.zeros((len(labels), seconds * FS))

    def _attempt(rate_key):
        w = pyedflib.EdfWriter(path, len(labels),
                               file_type=pyedflib.FILETYPE_EDFPLUS)
        try:
            w.setSignalHeaders([{
                'label': lab, 'dimension': 'uV', rate_key: FS,
                'physical_min': -3276.8, 'physical_max': 3276.7,
                'digital_min': -32768, 'digital_max': 32767,
                'prefilter': '', 'transducer': ''} for lab in labels])
            w.writeSamples([np.ascontiguousarray(s) for s in sigs])
            for onset, dur, text in annots:
                w.writeAnnotation(onset, dur, text)
        finally:
            try:
                w.close()
            except Exception:                                  # noqa: BLE001
                pass

    try:
        _attempt('sample_frequency')
    except (FutureWarning, KeyError, TypeError, ValueError):
        _attempt('sample_rate')
    return path


def _write_cache_for(path, seconds=60):
    """A probability cache beside the fixture, so opening it needs no model.

    Without one, load_edf runs inference — which imports TensorFlow into a
    process that has already loaded Qt, and on Windows that combination fails
    to initialise (docs/known_issues.md 1, the frozen app's own conflict
    reappearing in the test runner). The crash is silent: the process dies
    mid-test with no failure reported. These tests are about drawing
    annotations, so the cache keeps the model out of them entirely.
    """
    import numpy as np
    from gui.io.cache import save_probs
    starts = np.arange(0, max(1, seconds - 12 + 1), 6, dtype=np.int32)
    probs = np.zeros(starts.size, dtype=np.float32)
    save_probs(path, starts, probs,
               skip_code=np.zeros(starts.size, dtype=np.int8),
               meta={'step_s': 6, 'segment_s': 12, 'use_ica': True, 'fs': FS,
                     'weights': 'convlstm_ICA_12_train.h5'})


class ReadingThem(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='edfann')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_clinical_markers_come_back_with_their_text(self):
        p = _write_edf_with_annotations(
            os.path.join(self.tmp, 'a.edf'),
            [(10.0, -1.0, 'HVT START'), (20.0, -1.0, 'PHOTO 9Hz'),
             (30.0, -1.0, 'hoest')])
        got = read_edf_annotations(p)
        texts = [a['text'] for a in got]
        for want in ('HVT START', 'PHOTO 9Hz', 'hoest'):
            self.assertIn(want, texts,
                          '{!r} was lost; these are the markers that explain '
                          'why the detector fired'.format(want))

    def test_the_technicians_text_is_not_normalised(self):
        """Dutch notes stay Dutch — cleaning clinical free text loses meaning."""
        p = _write_edf_with_annotations(
            os.path.join(self.tmp, 'b.edf'),
            [(5.0, -1.0, 'ademh bevestigen')])
        got = read_edf_annotations(p)
        self.assertEqual([a['text'] for a in got], ['ademh bevestigen'])

    def test_they_are_sorted_by_time(self):
        p = _write_edf_with_annotations(
            os.path.join(self.tmp, 'c.edf'),
            [(30.0, -1.0, 'third'), (5.0, -1.0, 'first'),
             (12.0, -1.0, 'second')])
        got = read_edf_annotations(p)
        self.assertEqual([a['text'] for a in got], ['first', 'second', 'third'])
        self.assertEqual([a['onset'] for a in got], sorted(a['onset']
                                                           for a in got))

    def test_a_negative_duration_marks_an_instant(self):
        p = _write_edf_with_annotations(
            os.path.join(self.tmp, 'd.edf'), [(10.0, -1.0, 'mark')])
        got = read_edf_annotations(p)
        self.assertTrue(got[0]['instant'])

    def test_a_real_duration_is_a_span(self):
        p = _write_edf_with_annotations(
            os.path.join(self.tmp, 'e.edf'), [(10.0, 25.0, 'HV period')])
        got = [a for a in read_edf_annotations(p) if a['text'] == 'HV period']
        self.assertTrue(got, 'the spanning annotation was dropped')
        self.assertFalse(got[0]['instant'])
        self.assertAlmostEqual(got[0]['duration'], 25.0, places=3)

    def test_a_file_with_none_gives_an_empty_list(self):
        p = _write_edf_with_annotations(os.path.join(self.tmp, 'f.edf'), [])
        self.assertEqual(read_edf_annotations(p), [])

    def test_a_missing_file_does_not_raise(self):
        self.assertEqual(
            read_edf_annotations(os.path.join(self.tmp, 'nope.edf')), [])

    def test_a_corrupt_file_does_not_raise(self):
        """A recording that opened yesterday must open today."""
        p = os.path.join(self.tmp, 'junk.edf')
        with open(p, 'wb') as f:
            f.write(b'this is not an EDF file at all')
        self.assertEqual(read_edf_annotations(p), [])

    def test_the_count_is_capped_and_the_cap_is_disclosed(self):
        p = _write_edf_with_annotations(
            os.path.join(self.tmp, 'g.edf'),
            [(float(i) * 0.01 + 1.0, -1.0, 'imp {}'.format(i))
             for i in range(60)], seconds=60)
        got = read_edf_annotations(p, limit=10)
        self.assertLessEqual(len(got), 10)
        summary = annotation_summary(got, limit=10)
        self.assertIn('showing the first 10', summary,
                      'a truncated annotation list must say so, or the '
                      'reviewer believes they have seen the whole record')

    def test_the_summary_is_empty_when_there_are_none(self):
        self.assertEqual(annotation_summary([]), '')


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class DrawingThem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(['-platform', 'offscreen']))

    def _view(self):
        from gui.widgets.signal_view import SignalView
        v = SignalView()
        data = np.zeros((19, FS * 120), dtype=np.float32)
        v.set_signal(data, FS, CH)
        return v

    def test_markers_are_added_and_replaced_not_accumulated(self):
        v = self._view()
        try:
            v.set_source_annotations([{'onset': 10.0, 'duration': -1.0,
                                       'text': 'HVT START', 'instant': True}])
            self.assertEqual(len(v._annot_items), 1)
            v.set_source_annotations([
                {'onset': 20.0, 'duration': -1.0, 'text': 'a', 'instant': True},
                {'onset': 30.0, 'duration': -1.0, 'text': 'b', 'instant': True}])
            self.assertEqual(len(v._annot_items), 2,
                             'markers accumulated across calls, so a second '
                             'recording would show the first one\'s notes')
        finally:
            v.close()

    def test_loading_a_new_signal_clears_the_previous_markers(self):
        v = self._view()
        try:
            v.set_source_annotations([{'onset': 10.0, 'duration': -1.0,
                                       'text': 'old', 'instant': True}])
            data = np.zeros((19, FS * 60), dtype=np.float32)
            v.set_signal(data, FS, CH)
            self.assertEqual(
                v._annot_items, [],
                'the previous recording\'s technician notes survived a new '
                'file being opened')
        finally:
            v.close()

    def test_they_are_not_drawn_in_the_reference_colour(self):
        """Green is ground truth. A cough must not look like a seizure."""
        v = self._view()
        try:
            v.set_references([(10.0, 20.0)])
            v.set_source_annotations([{'onset': 40.0, 'duration': -1.0,
                                       'text': 'hoest', 'instant': True}])
            ref_pen = v._ref_items[0].lines[0].pen.color().getRgb()[:3]
            ann_pen = v._annot_items[0].pen.color().getRgb()[:3]
            self.assertNotEqual(
                ref_pen, ann_pen,
                'recording annotations share the reference colour, so '
                'technician context reads as adjudicated ground truth')
        finally:
            v.close()

    def _crowded(self, v, n=20, step=3.0):
        """Markers every `step` seconds — the PHOTO-stimulation pattern."""
        items = [{'onset': 5.0 + i * step, 'duration': -1.0,
                  'text': 'PHOTO {}Hz'.format(i), 'instant': True}
                 for i in range(n)]
        v.set_source_annotations(items)
        return items

    def _visible_label_positions(self, v, x0, x1):
        W = float(v._pw.viewport().width() or 1500)
        span = max(1e-9, x1 - x0)
        out = []
        for it, a in zip(v._annot_items, v._source_annotations):
            lab = getattr(it, 'label', None)
            # Shown-ness is opacity, not visibility: pyqtgraph 0.11.1's
            # InfLineLabel.setVisible(True) raises (see _set_label_shown).
            if lab is not None and lab.opacity() > 0.5:
                out.append((a['onset'] - x0) / span * W)
        return sorted(out)

    def test_labels_thin_out_instead_of_overlapping(self):
        """Crowded text over the EEG is worse than no text.

        Measured on the real clinical recording before this existed: at a
        five-minute view, 16 of 18 label pairs overlapped, and across the whole
        recording 35 of 40. The result was an unreadable smear laid over the
        trace the reviewer is trying to read.
        """
        from gui.widgets.signal_view import MIN_LABEL_GAP_PX
        v = self._view()
        try:
            self._crowded(v, n=20, step=3.0)
            pi = v._pw.getPlotItem()
            pi.setXRange(0.0, 70.0, padding=0)     # all 20 on screen at once
            v._redecimate_visible()
            pos = self._visible_label_positions(v, 0.0, 70.0)
            self.assertTrue(pos, 'every label was hidden; at least the first '
                                 'of a crowded group must still be readable')
            gaps = [pos[i + 1] - pos[i] for i in range(len(pos) - 1)]
            self.assertTrue(
                all(g >= MIN_LABEL_GAP_PX - 1e-6 for g in gaps),
                'visible labels are still closer than {} px: {}'.format(
                    MIN_LABEL_GAP_PX, [round(g, 1) for g in gaps]))
        finally:
            v.close()

    def test_the_lines_are_all_still_drawn_when_labels_are_thinned(self):
        """Hiding text must not hide the fact that a marker is there."""
        v = self._view()
        try:
            items = self._crowded(v, n=20, step=3.0)
            pi = v._pw.getPlotItem()
            pi.setXRange(0.0, 70.0, padding=0)
            v._redecimate_visible()
            self.assertEqual(len(v._annot_items), len(items))
            self.assertTrue(all(it.isVisible() for it in v._annot_items),
                            'a marker line was hidden along with its label')
        finally:
            v.close()

    def test_zooming_in_brings_the_labels_back(self):
        v = self._view()
        try:
            self._crowded(v, n=20, step=3.0)
            pi = v._pw.getPlotItem()
            pi.setXRange(0.0, 70.0, padding=0)
            v._redecimate_visible()
            wide = len(self._visible_label_positions(v, 0.0, 70.0))
            pi.setXRange(5.0, 15.0, padding=0)     # a couple of markers only
            v._redecimate_visible()
            near = len(self._visible_label_positions(v, 5.0, 15.0))
            self.assertGreater(
                near, 0, 'zoomed in to two markers and neither was labelled')
            self.assertLess(wide, 20,
                            'nothing was thinned at the wide zoom, so this '
                            'test is not exercising the thinning')
        finally:
            v.close()

    def test_labels_off_screen_are_not_drawn(self):
        v = self._view()
        try:
            self._crowded(v, n=20, step=3.0)
            pi = v._pw.getPlotItem()
            pi.setXRange(0.0, 8.0, padding=0)
            v._redecimate_visible()
            for it, a in zip(v._annot_items, v._source_annotations):
                lab = getattr(it, 'label', None)
                if lab is not None and a['onset'] > 8.0:
                    self.assertLessEqual(
                        lab.opacity(), 0.5,
                        'a label for a marker at {} s is drawn while the view '
                        'ends at 8 s'.format(a['onset']))
        finally:
            v.close()

    def test_an_empty_list_is_safe(self):
        v = self._view()
        try:
            v.set_source_annotations([])
            self.assertEqual(v._annot_items, [])
        finally:
            v.close()


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class TheyStayOutOfTheReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(['-platform', 'offscreen']))

    def setUp(self):
        # Two things would otherwise make these tests depend on the machine
        # rather than the code. The memory guard reads real free memory, so a
        # busy build box refuses a 60-second fixture; and its refusal is a
        # MODAL dialog, which in an offscreen test process blocks with no
        # output at all. Both are correct in the application and wrong in a
        # test, so free memory is pinned generous and the dialogs are stubbed.
        import gui.memory as gm
        self._saved_avail = gm.available_bytes
        gm.available_bytes = lambda: 64 << 30
        self._saved_warn = QtWidgets.QMessageBox.warning
        self._saved_crit = QtWidgets.QMessageBox.critical
        QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: None)
        QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: None)

    def tearDown(self):
        import gui.memory as gm
        gm.available_bytes = self._saved_avail
        QtWidgets.QMessageBox.warning = self._saved_warn
        QtWidgets.QMessageBox.critical = self._saved_crit

    def test_the_export_writes_no_annotation_from_the_recording(self):
        """The one file this app produces for other clinicians to read."""
        import io
        with io.open(os.path.join(REPO, 'gui', 'app.py'),
                     encoding='utf-8') as f:
            src = f.read()
        export_start = src.index('def _do_export')  \
            if 'def _do_export' in src else src.index('write_csv_bi(path')
        export_src = src[export_start:export_start + 4000]
        self.assertNotIn(
            '_source_annots', export_src,
            'the export path references the recording\'s own annotations; '
            'technician notes must never be written as seizure labels')

    def test_they_are_actually_drawn_after_a_real_load(self):
        """The bug this catches: loaded, counted, and invisible.

        set_source_annotations was called during load_edf, and
        _refresh_signal_view ran afterwards and cleared the plot. The status
        bar said "41 recording annotations" while the trace showed none — a
        screen stating something false, which is the failure mode this
        application treats as most serious.
        """
        import numpy as np
        from gui.app import MainWindow
        tmp = tempfile.mkdtemp(prefix='annload')
        try:
            p = _write_edf_with_annotations(
                os.path.join(tmp, 'load.edf'),
                [(10.0, -1.0, 'HVT START'), (20.0, -1.0, 'PHOTO 9Hz')],
                seconds=60)
            _write_cache_for(p, 60)
            w = MainWindow()
            try:
                w.load_edf(p)
                self.assertEqual(len(w._source_annots), 2)
                self.assertEqual(
                    len(w.signal_view._annot_items), 2,
                    'annotations were loaded and counted but never drawn')
            finally:
                w.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_they_survive_a_filter_and_montage_change(self):
        """Every rebuild must restore them, or they vanish on first use."""
        from gui.app import MainWindow
        tmp = tempfile.mkdtemp(prefix='annrefresh')
        try:
            p = _write_edf_with_annotations(
                os.path.join(tmp, 'r.edf'),
                [(10.0, -1.0, 'HVT START')], seconds=60)
            _write_cache_for(p, 60)
            w = MainWindow()
            try:
                w.load_edf(p)
                w.cb_lp.setCurrentText('30')
                self.assertEqual(len(w.signal_view._annot_items), 1,
                                 'a filter change erased the markers')
                w.cb_montage.setCurrentIndex(1)
                self.assertEqual(len(w.signal_view._annot_items), 1,
                                 'a montage change erased the markers')
            finally:
                w.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_main_window_keeps_them_separate_from_events(self):
        from gui.app import MainWindow
        w = MainWindow()
        try:
            self.assertEqual(w._source_annots, [],
                             'annotation state is not initialised, so a '
                             'status message before any file opens can reach '
                             'an undefined attribute')
            self.assertTrue(hasattr(w, '_events'))
            self.assertIsNot(w._source_annots, w._events)
        finally:
            w.close()


if __name__ == '__main__':
    unittest.main()
