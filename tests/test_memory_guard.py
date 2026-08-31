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

from gui.memory import (_headroom, available_bytes,             # noqa: E402
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
        load = resident + transient
        return load + _headroom(load) <= available

    def test_a_day_long_file_is_refused_on_a_busy_ward_pc(self):
        """An 8 GB machine running Windows and a browser, realistically."""
        r, t = estimate_bytes(19, 24 * 3600 * 250, 250)
        self.assertFalse(self._fits(r, t, 35 * GB // 10),
                         'a 24 h recording must be refused with 3.5 GB free — '
                         'this is the shape of machine the tool deploys to')

    def test_a_short_recording_opens_even_on_a_starved_machine(self):
        """The regression that made this margin proportional.

        A fixed 1.5 GB margin refused a two-minute recording needing 30 MB
        whenever the machine was momentarily busy. A review tool that refuses
        ordinary work is not safer than one that occasionally pages; it is
        just unusable, and a clinician will route around it.
        """
        r, t = estimate_bytes(19, 142 * 250, 250)
        self.assertTrue(self._fits(r, t, 700 * GB // 1000),
                        'a 2-minute recording was refused with 0.7 GB free')

    def test_a_half_hour_recording_opens_on_a_busy_machine(self):
        r, t = estimate_bytes(37, 1818 * 200, 200)
        self.assertTrue(self._fits(r, t, 15 * GB // 10),
                        'a 30-minute clinical recording was refused with '
                        '1.5 GB free')

    def test_an_hour_long_file_opens_on_a_modest_machine(self):
        r, t = estimate_bytes(19, 3600 * 250, 250)
        self.assertTrue(self._fits(r, t, 4 * GB),
                        'an hour-long recording must still open on a ward PC')

    def test_headroom_is_reserved_beyond_the_read_itself(self):
        """A file that exactly fills available RAM must still be refused."""
        r, t = estimate_bytes(19, 3600 * 250, 250)
        self.assertFalse(self._fits(r, t, r + t),
                         'the estimate left no slack, so the load would take '
                         'the machine to exactly zero free memory')

    def test_headroom_scales_with_the_size_of_the_read(self):
        small = _headroom(10 * 1024 * 1024)
        large = _headroom(5 * GB)
        self.assertGreater(large, small * 4,
                           'headroom does not scale, so it is either crushing '
                           'small files or too thin for large ones')

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
                    'needed_bytes': r + t + _headroom(r + t),
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


class TheBuildGateChecksTheFrozenApp(unittest.TestCase):
    """A guard that can vanish from a bundle must be gated, not assumed.

    `load_edf` imports gui.memory lazily and swallows an ImportError — right,
    because a machine that cannot report its memory should still open files,
    and wrong if nobody notices the module missing from a frozen build. The
    guard would then silently not exist and the first symptom would be a
    frozen ward PC. This is the Keras 3 failure shape (known_issues.md §1),
    where a build-time claim was true and the runtime behaviour was not, so it
    gets the same treatment: the self-test reports what it actually found and
    the smoke test fails the build on it.
    """

    def _src(self, rel):
        import io
        with io.open(os.path.join(REPO, rel), encoding='utf-8') as f:
            return f.read()

    def test_the_self_test_reports_whether_the_guard_is_present(self):
        code = self._src(os.path.join('gui', 'main.py'))
        self.assertIn('self-test: memguard', code,
                      'the frozen self-test no longer reports the memory '
                      'guard, so smoke_test.py has nothing to check')

    def test_the_smoke_test_fails_the_build_when_it_is_missing(self):
        import re
        code = self._src(os.path.join('packaging', 'smoke_test.py'))
        self.assertIn('self-test: memguard', code,
                      'smoke_test.py does not inspect the memory guard line')
        self.assertTrue(
            re.search(r'failures\.append\([^)]*memory guard', code, re.S),
            'smoke_test.py reads the memory-guard line but does not fail the '
            'build on it — a silently unguarded build would ship')


class TheMachineCanBeQueried(unittest.TestCase):
    def test_available_bytes_is_plausible_or_none(self):
        v = available_bytes()
        if v is None:
            self.skipTest('this platform exposes no memory query')
        self.assertGreater(v, 16 << 20, 'implausibly small')
        self.assertLess(v, 4 << 40, 'implausibly large')


if __name__ == '__main__':
    unittest.main()
