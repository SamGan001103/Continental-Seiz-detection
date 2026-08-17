r"""The 19-channel montage must parse identically from a CRLF or an LF checkout.

`utils/params_common_electrodes.txt` names the channels the detector is fed, and
git hands it to Windows with CRLF and to macOS with LF. The two frozen builds on
the demo stick differ by exactly those 24 bytes -- 608 against 584, byte-identical
after normalisation -- and that is the entire difference between the bundled data
on the two platforms once the model weights are hashed (both 90c046ee80b50499).

Harmless as it stands: `pyst.nedc_load_parameters` opens the file in text mode,
so Python translates the line endings before the parser sees them.

Pinned anyway, because the risk is not this parser but the next reader of the
file, and the failure would be silent. Splitting on "\n" instead leaves "\r"
welded to the last field on every line; that channel name stops matching the EDF
header, the channel is dropped from the array the model scores, and it happens on
one platform only. Nothing raises. The scores just quietly disagree across
machines -- and having spent this project's worth of effort establishing that
cross-machine agreement comes from the Keras version, the next hunt would start
at the model and not at a text file's line endings.

Written as its own module rather than appended to test_montage_tables.py so the
escape sequences above live in a file authored directly, not through a shell
heredoc: three separate bugs in this repository have come from "\n" arriving as a
real newline inside a string literal.
"""
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui import paths
from utils import pyst

CR = b'\x0d'
LF = b'\x0a'
CRLF = CR + LF


class ParameterFileIsNewlineAgnostic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='eol')
        with open(paths.params_path(), 'rb') as f:
            self.lf_bytes = f.read().replace(CRLF, LF)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _parse(self, name, data):
        path = os.path.join(self.tmp, name + '.txt')
        with open(path, 'wb') as f:
            f.write(data)
        params = pyst.nedc_load_parameters(path)
        self.assertIsNotNone(params, '%s variant failed to parse at all' % name)
        return params

    def test_the_fixture_actually_differs(self):
        """Guard the test itself: two identical inputs would prove nothing."""
        crlf_bytes = self.lf_bytes.replace(LF, CRLF)
        self.assertNotEqual(self.lf_bytes, crlf_bytes)
        self.assertEqual(len(crlf_bytes) - len(self.lf_bytes),
                         self.lf_bytes.count(LF),
                         'the CRLF fixture is not one extra byte per line')

    def test_crlf_and_lf_parse_to_the_same_parameters(self):
        lf = self._parse('lf', self.lf_bytes)
        crlf = self._parse('crlf', self.lf_bytes.replace(LF, CRLF))
        self.assertEqual(lf, crlf,
                         'the montage parses differently from a CRLF checkout')

    def test_no_parsed_value_carries_a_carriage_return(self):
        """The specific corruption, on the file as it stands on this machine."""
        params = pyst.nedc_load_parameters(paths.params_path())
        self.assertIsNotNone(params)
        for key, value in params.items():
            self.assertNotIn('\r', str(key))
            self.assertNotIn('\r', str(value),
                             'parsed value for %r carries a carriage return, so '
                             'the last field on its line is corrupt' % (key,))

    def test_the_channel_list_survives_both_encodings(self):
        """The list that becomes the model's input, specifically.

        The equality above would also pass if both variants parsed to nothing
        useful, so pull out the channel selection and check it is the real
        19-channel montage in both cases.
        """
        selections = []
        for name, data in (('lf', self.lf_bytes),
                           ('crlf', self.lf_bytes.replace(LF, CRLF))):
            params = self._parse(name, data)
            channels = None
            for value in params.values():
                if isinstance(value, dict):
                    for inner in value.values():
                        if isinstance(inner, str) and ',' in inner:
                            parts = [p.strip() for p in inner.split(',')]
                            if len(parts) >= 19:
                                channels = parts
                elif isinstance(value, str) and ',' in value:
                    parts = [p.strip() for p in value.split(',')]
                    if len(parts) >= 19:
                        channels = parts
            self.assertIsNotNone(
                channels, '%s: no channel list found in the parsed parameters'
                % name)
            self.assertTrue(all(c and '\r' not in c for c in channels),
                            '%s: channel names are corrupt: %r' % (name, channels))
            selections.append(channels)

        self.assertEqual(selections[0], selections[1],
                         'the channel list the model is fed depends on the '
                         'checkout line endings')


if __name__ == '__main__':
    unittest.main()
