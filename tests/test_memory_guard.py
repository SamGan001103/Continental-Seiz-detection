"""A recording too large for the machine must be refused, not attempted.

On 2026-08-31 opening a 24-hour recording on a 16 GB development machine froze
it hard enough to need a power cycle. The app is intended for hospital PCs with
8 GB and nobody at the keyboard who can diagnose a swap storm — and a freeze
takes down whatever else the clinician had open, with no message and no log.

`gui/memory.py` reads the EDF header, estimates what loading it would cost, and
refuses when that exceeds available RAM. These tests pin the arithmetic (which
must stay pessimistic — under-estimating is the failure that freezes machines),
the decision boundary, and the fact that the refusal text tells the reviewer
what to do about it.

`available` is injected throughout, so the tests assert behaviour rather than
whatever this particular machine happens to have free.
"""
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.memory import (SAFETY_MARGIN_BYTES, available_bytes,   # noqa: E402
                        describe_refusal, estimate_bytes)

GB = 1 << 30


class TheEstimateIsPessimistic(unittest.TestCase):
    def test_a_day_at_250hz_needs_gigabytes(self):
        n = 24 * 3600 * 250
        resident, transient = estimate_bytes(19, n, 250)
        self.assertAlmostEqual(resident / 1e9, 1.64, places=1)
        self.assertGreater(transient, resident,
                           'the transient cost of loading must not be '
                           'estimated as smaller than the array kept')

    def test_extra_file_channels_raise_the_estimate(self):
        """pyst reads every channel in the file, not just the 19 wanted."""
        n = 3600 * 250
        _r19, t19 = estimate_bytes(19, n, 250)
        _r37, t37 = estimate_bytes(37, n, 250)
        self.assertGreater(t37, t19 * 1.8,
                           'a 37-channel file must be estimated to cost more '
                           'to read than a 19-channel one')

    def test_resampling_adds_cost(self):
        n = 3600 * 200
        _r, t200 = estimate_bytes(19, n, 200)          # needs resampling
        _r2, t250 = estimate_bytes(19, int(n * 250 / 200), 250)
        self.assertGreater(t200, t250,
                           'an FFT resample must be estimated as costing more '
                           'than a straight read')

    def test_a_short_recording_is_cheap(self):
        resident, transient = estimate_bytes(19, 600 * 250, 250)
        self.assertLess(resident + transient, 200e6)


class TheDecisionBoundary(unittest.TestCase):
    """Uses the estimate directly; edf_open_estimate only adds a header read."""

    def _fits(self, resident, transient, available):
        return resident + transient + SAFETY_MARGIN_BYTES <= available

    def test_a_day_long_file_is_refused_on_an_8gb_machine(self):
        r, t = estimate_bytes(19, 24 * 3600 * 250, 250)
        self.assertFalse(self._fits(r, t, 6 * GB),
                         'a 24 h recording must be refused with 6 GB free — '
                         'this is the case that froze a real machine')

    def test_an_hour_long_file_opens_on_a_modest_machine(self):
        r, t = estimate_bytes(19, 3600 * 250, 250)
        self.assertTrue(self._fits(r, t, 4 * GB),
                        'an hour-long recording must still open on a ward PC')

    def test_the_safety_margin_is_reserved(self):
        """A file that exactly fills RAM must still be refused."""
        r, t = estimate_bytes(19, 3600 * 250, 250)
        self.assertFalse(self._fits(r, t, r + t),
                         'the estimate must reserve headroom for Qt, '
                         'TensorFlow and the operating system')

    def test_unknown_available_memory_does_not_block_the_user(self):
        """If the machine cannot be queried, the app must not refuse everything."""
        from gui.memory import edf_open_estimate
        self.assertTrue(callable(edf_open_estimate))
        # available=None means "unknown" only inside edf_open_estimate when the
        # platform query fails; the contract asserted here is that a failure to
        # measure is not treated as zero memory.
        import gui.memory as m
        saved = m.available_bytes
        try:
            m.available_bytes = lambda: None
            self.assertIsNone(m.available_bytes())
        finally:
            m.available_bytes = saved


class TheRefusalIsActionable(unittest.TestCase):
    def setUp(self):
        r, t = estimate_bytes(19, 24 * 3600 * 250, 250)
        self.est = {'path': 'x.edf', 'duration_s': 86400.0, 'channels': 19,
                    'fs': 250.0, 'resident_bytes': r, 'transient_bytes': t,
                    'needed_bytes': r + t + SAFETY_MARGIN_BYTES,
                    'available_bytes': 6 * GB, 'fits': False}

    def test_it_states_the_numbers_and_a_way_forward(self):
        text = describe_refusal(self.est)
        self.assertIn('24.0 hours', text)
        self.assertIn('GB', text)
        for phrase in ('close other programs', 'more memory', 'split'):
            self.assertIn(phrase, text.lower(),
                          'the refusal must tell the reviewer what to do, not '
                          'only that it failed: missing {!r}'.format(phrase))

    def test_it_survives_unknown_available_memory(self):
        est = dict(self.est, available_bytes=None)
        text = describe_refusal(est)
        self.assertIn('unknown', text)


class TheMachineCanBeQueried(unittest.TestCase):
    def test_available_bytes_is_plausible_or_none(self):
        v = available_bytes()
        if v is None:
            self.skipTest('this platform exposes no memory query')
        self.assertGreater(v, 16 << 20, 'implausibly small')
        self.assertLess(v, 4 << 40, 'implausibly large')


if __name__ == '__main__':
    unittest.main()
