"""The prototype disclaimer must reach every surface that leaves the app.

An exported .csv_bi is byte-format-identical to a TUSZ reference file, so
without these headers a reviewed export can later be mistaken for clinical
ground truth. These tests exist so that can never regress silently.
"""
import os
import shutil
import tempfile
import unittest

from gui.io.csv_bi import (
    NOT_FOR_CLINICAL_USE, TOOL_NAME, read_csv_bi, write_csv_bi,
)


class ExportDisclaimerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, 'out.csv_bi')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, **kw):
        write_csv_bi(self.path, bname='rec', duration_s=100.0,
                     events=[{'start': 10.0, 'stop': 20.0, 'label': 'seiz'}],
                     **kw)
        with open(self.path) as f:
            return f.read()

    def test_export_states_the_tool_and_that_it_is_not_clinical(self):
        text = self._write()
        self.assertIn('# tool = ' + TOOL_NAME, text)
        self.assertIn('# status = ' + NOT_FOR_CLINICAL_USE, text)

    def test_zuna_export_is_marked_as_a_reconstructed_signal(self):
        text = self._write(ai_source='zuna')
        self.assertIn('# ai_source = zuna', text)
        self.assertIn('RECONSTRUCTED', text)

    def test_baseline_export_records_its_source_without_the_warning(self):
        text = self._write(ai_source='baseline')
        self.assertIn('# ai_source = baseline', text)
        self.assertNotIn('RECONSTRUCTED', text)

    def test_headers_do_not_break_the_reader_round_trip(self):
        """The disclaimer must not corrupt the format it is embedded in."""
        self._write(ai_source='zuna')
        events = read_csv_bi(self.path)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]['start'], 10.0)
        self.assertAlmostEqual(events[0]['stop'], 20.0)
        self.assertEqual(events[0]['label'], 'seiz')

    def test_omitting_ai_source_still_carries_the_disclaimer(self):
        text = self._write()
        self.assertIn('# status = ', text)
        self.assertNotIn('# ai_source', text)


class BannerConstantTests(unittest.TestCase):
    def test_gui_constants_say_not_for_diagnostic_use(self):
        # Imported lazily: PyQt5 may be unavailable in a headless checkout.
        try:
            from gui.app import PROTOTYPE_BANNER, TITLE_SUFFIX
        except ImportError:
            self.skipTest('PyQt5 not available')
        for text in (PROTOTYPE_BANNER, TITLE_SUFFIX):
            self.assertIn('NOT FOR DIAGNOSTIC USE', text.upper())


if __name__ == '__main__':
    unittest.main()
