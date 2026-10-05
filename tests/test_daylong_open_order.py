"""A day-long recording must show signal, not 19 flat lines.

The bug this pins was invisible and total. `open_source` keeps a
`pyedflib.EdfReader` open on the recording, deliberately, because constructing
one indexes the whole file. Anything that opens a SECOND reader on the same
path afterwards hits edflib's refusal to open a file twice — and tearing down
that refused open frees the handle the first reader is holding.

The first reader then survives. Every read through it returns the zero-filled
buffer it allocated, so the reviewer is shown a 24-hour recording as 19
perfectly flat traces at every zoom, montage and filter setting, with no
dialog, no status message and nothing in the log. A reviewer either concludes
the recording is corrupt or, worse, reads it as genuinely isoelectric.

Measured on `pat_102_part02.edf`: 34.98 uV of signal becomes 0.00, purely from
the ORDER of three calls in `load_edf`. The only symptom is
`read -1, less than N requested!!!` on stderr from the C layer, which is
invisible in a frozen build — and which was dismissed for a whole session as a
harmless over-read past the end of the file.

So the rule is: every header read that needs its own reader happens BEFORE the
lazy source opens one. These tests hold that, without needing a 1.2 GB file.
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

FS = 256
CH = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ',
      'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']


def _write_edf(path, seconds=120):
    import pyedflib
    rng = np.random.default_rng(7)
    n = seconds * FS
    sigs = np.cumsum(rng.standard_normal((len(CH), n)), axis=1)
    sigs *= 40.0 / np.maximum(sigs.std(axis=1, keepdims=True), 1e-9)

    def _attempt(rate_key):
        w = pyedflib.EdfWriter(path, len(CH), file_type=pyedflib.FILETYPE_EDF)
        try:
            w.setSignalHeaders([{
                'label': 'EEG {}-REF'.format(c), 'dimension': 'uV',
                rate_key: FS,
                'physical_min': -5000.0, 'physical_max': 5000.0,
                'digital_min': -32768, 'digital_max': 32767,
                'prefilter': '', 'transducer': ''} for c in CH])
            w.writeSamples([np.ascontiguousarray(s) for s in sigs])
        finally:
            try:
                w.close()
            except Exception:                                # noqa: BLE001
                pass

    try:
        _attempt('sample_frequency')
    except (FutureWarning, KeyError, TypeError, ValueError):
        _attempt('sample_rate')
    return path


class ASecondReaderKillsTheFirst(unittest.TestCase):
    """The mechanism, pinned directly. If this ever stops being true the
    ordering rule below can be relaxed — but until then it is load-bearing."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='openorder')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.edf = _write_edf(os.path.join(self.tmp, 'r.edf'))

    def _sample(self, src):
        a = np.asarray(src[:, 10 * FS:20 * FS], dtype=float)
        return float(a.std())

    def test_header_reads_after_the_lazy_open_destroy_the_signal(self):
        from gui.io.edf import read_start_datetime
        from gui.io.edf_annotations import read_edf_annotations
        from gui.io.edf_source import open_source
        from gui.paths import params_path

        src = open_source(self.edf, params_path())
        self.assertIsNotNone(src)
        try:
            read_edf_annotations(self.edf)
            read_start_datetime(self.edf)
            after = self._sample(src)
        finally:
            src.close()
        # Documents the hazard. If this ASSERTION starts failing, edflib has
        # changed and the comment in load_edf should be revisited.
        self.assertEqual(after, 0.0,
                         'the hazard this ordering rule exists for appears to '
                         'be gone; re-check gui/app.py load_edf before '
                         'relying on that')

    def test_header_reads_before_the_lazy_open_are_fine(self):
        from gui.io.edf import read_start_datetime
        from gui.io.edf_annotations import read_edf_annotations
        from gui.io.edf_source import open_source
        from gui.paths import params_path

        read_edf_annotations(self.edf)
        read_start_datetime(self.edf)
        src = open_source(self.edf, params_path())
        self.assertIsNotNone(src)
        try:
            self.assertGreater(self._sample(src), 1.0,
                               'reading the headers first must leave the '
                               'lazy source working')
        finally:
            src.close()


class LoadEdfReadsHeadersFirst(unittest.TestCase):
    """The rule, asserted against the source so it cannot regress by someone
    moving a line."""

    def test_the_header_reads_come_before_open_source(self):
        src = open(os.path.join(REPO, 'gui', 'app.py'),
                   encoding='utf-8').read()
        i_ann = src.index('self._source_annots = read_edf_annotations(path)')
        i_start = src.index('self._start_time = read_start_datetime(path)')
        i_open = src.index('lazy_source = open_source(path, params_path())')
        self.assertLess(i_ann, i_open,
                        'read_edf_annotations must run BEFORE open_source or '
                        'every day-long recording displays flat traces')
        self.assertLess(i_start, i_open,
                        'read_start_datetime must run BEFORE open_source for '
                        'the same reason')

    def test_neither_header_read_happens_twice(self):
        """They were moved, not copied. A leftover second call would put the
        bug straight back."""
        src = open(os.path.join(REPO, 'gui', 'app.py'),
                   encoding='utf-8').read()
        self.assertEqual(
            src.count('read_edf_annotations(path)'), 1)
        self.assertEqual(
            src.count('read_start_datetime(path)'), 1)


if __name__ == '__main__':
    unittest.main()
