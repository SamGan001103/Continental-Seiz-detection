"""Band power is only meaningful if it survives the display filter chain.

Quantitative EEG numbers look authoritative on screen, which is exactly the
danger: at this app's defaults (HP 1.0, LP 70, notch 60) a delta figure is
mostly a statement about the high-pass and a gamma figure is a statement about
the low-pass and the mains notch. These tests pin two things:

  * the arithmetic — a 10 Hz sinusoid lands in alpha at the right absolute
    power, bands integrate rather than sum, ratios add to 1;
  * the honesty of the caveats — every band the helper calls compromised is
    measurably compromised, and the fraction it claims survives is the fraction
    that actually survives when the same signal is put through apply_filters.

The second is the one worth having. A warning nobody has checked against the
filter it warns about is decoration.
"""
import os
import sys
import time
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.processing import (BANDS, COMPROMISED_BELOW, apply_filters,
                            band_powers, band_ratios, band_table,
                            filter_caveats, psd, total_band_power)

FS = 250.0


def _sine(freq_hz, amp_uv=20.0, seconds=12.0, fs=FS, phase=0.4):
    t = np.arange(int(seconds * fs)) / fs
    return (amp_uv * np.sin(2 * np.pi * freq_hz * t + phase)).astype(np.float32)


def _noise(seconds=60.0, fs=FS, amp_uv=30.0, seed=3):
    rng = np.random.RandomState(seed)
    return (amp_uv * rng.randn(int(seconds * fs))).astype(np.float32)


class TheBandsThemselves(unittest.TestCase):
    """The table's rows are a clinical convention, not a free parameter."""

    def test_bands_are_the_standard_clinical_five_in_order(self):
        self.assertEqual(list(BANDS), ['delta', 'theta', 'alpha', 'beta',
                                       'gamma'])
        self.assertEqual(list(BANDS.values()),
                         [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0),
                          (30.0, 80.0)])

    def test_bands_are_contiguous(self):
        """A gap would silently vanish from every percentage in the table."""
        edges = list(BANDS.values())
        for (_, hi), (lo, _) in zip(edges[:-1], edges[1:]):
            self.assertEqual(hi, lo)


class ASinusoidLandsInOneBand(unittest.TestCase):
    """The arithmetic, checked against a signal whose spectrum we know."""

    def setUp(self):
        self.freqs, self.power = psd(_sine(10.0, amp_uv=20.0), FS)

    def test_a_10_hz_sinusoid_is_alpha_and_nothing_else(self):
        ratios = band_ratios(self.freqs, self.power)
        self.assertGreater(ratios['alpha'], 0.98)
        for name in ('delta', 'theta', 'beta', 'gamma'):
            self.assertLess(ratios[name], 0.01,
                            '{} picked up {:.3f} of a pure 10 Hz tone'
                            .format(name, ratios[name]))

    def test_absolute_power_is_in_uv_squared(self):
        """A 20 uV sinusoid carries A**2/2 = 200 uV**2. Units are the whole
        point of a table a clinician reads."""
        alpha = band_powers(self.freqs, self.power)['alpha']
        self.assertAlmostEqual(alpha, 200.0, delta=20.0)

    def test_the_psd_resolves_the_bands_it_reports(self):
        """~0.5-1 Hz bins, or alpha and theta are the same measurement."""
        df = float(self.freqs[1] - self.freqs[0])
        self.assertLessEqual(df, 1.0)
        self.assertGreaterEqual(df, 0.2)

    def test_psd_does_not_mutate_its_input(self):
        x = _sine(10.0)
        before = x.copy()
        psd(x, FS)
        np.testing.assert_array_equal(x, before)

    def test_a_multichannel_slice_gives_one_row_per_channel(self):
        data = np.stack([_sine(f) for f in (2.0, 6.0, 10.0, 20.0, 40.0)])
        freqs, power = psd(data, FS)
        self.assertEqual(power.shape[0], 5)
        ratios = band_ratios(freqs, power)
        loudest = [max(BANDS, key=lambda b: ratios[b][k]) for k in range(5)]
        self.assertEqual(loudest, ['delta', 'theta', 'alpha', 'beta', 'gamma'])


class BandsAddUp(unittest.TestCase):
    def setUp(self):
        self.freqs, self.power = psd(_noise(), FS)

    def test_band_powers_sum_to_the_total(self):
        powers = band_powers(self.freqs, self.power)
        self.assertAlmostEqual(sum(powers.values()),
                               total_band_power(self.freqs, self.power),
                               places=9)

    def test_ratios_sum_to_one(self):
        self.assertAlmostEqual(sum(band_ratios(self.freqs, self.power).values()),
                               1.0, places=9)

    def test_ratios_sum_to_one_for_a_pure_tone_too(self):
        freqs, power = psd(_sine(10.0), FS)
        self.assertAlmostEqual(sum(band_ratios(freqs, power).values()),
                               1.0, places=9)

    def test_a_flat_channel_reports_zeros_not_nan(self):
        """A disconnected electrode is a real recording condition; 0/0 would
        put 'nan%' in every row."""
        freqs, power = psd(np.zeros(int(12 * FS), dtype=np.float32), FS)
        ratios = band_ratios(freqs, power)
        for name, r in ratios.items():
            self.assertEqual(r, 0.0, '{} was {}'.format(name, r))

    def test_integration_is_not_a_bin_sum(self):
        """Summing bins scales with segment length; integrating does not.

        The same signal read at two Welch resolutions must give the same beta
        power, otherwise the number depends on a setting the reviewer cannot
        see. A bare bin sum would differ by the ratio of the bin widths (2x).
        """
        x = _noise()
        f1, p1 = psd(x, FS, nperseg=256)
        f2, p2 = psd(x, FS, nperseg=1024)
        b1 = band_powers(f1, p1)['beta']
        b2 = band_powers(f2, p2)['beta']
        self.assertAlmostEqual(b1 / b2, 1.0, delta=0.10)


class CaveatsNameTheFilterThatDidIt(unittest.TestCase):
    def test_gamma_under_a_30_hz_lowpass_is_compromised(self):
        cav = filter_caveats(FS, hp=1.0, lp=30.0, notch=60.0)['gamma']
        self.assertTrue(cav.compromised)
        self.assertIn('30 Hz low-pass', cav.note)
        self.assertLess(cav.retained, 0.05)

    def test_a_40_hz_signal_really_is_gone_under_that_lowpass(self):
        """The caveat is checked against the filter, not merely asserted.

        A 40 Hz tone is 'gamma' by the table's definition; under a 30 Hz
        low-pass almost none of it reaches the screen, and the table would
        otherwise report the residue as physiology.
        """
        x = _sine(40.0, amp_uv=20.0)[None, :]
        y = apply_filters(x, FS, hp=1.0, lp=30.0, notch=60.0)[0]
        before = band_powers(*psd(x[0], FS))['gamma']
        after = band_powers(*psd(y, FS))['gamma']
        self.assertLess(after / before, 0.02,
                        'the 30 Hz low-pass left {:.1%} of a 40 Hz tone'
                        .format(after / before))

    def test_the_app_defaults_compromise_delta_and_gamma_only(self):
        """The specific claim the review panel has to make on first open."""
        cav = filter_caveats(FS, hp=1.0, lp=70.0, notch=60.0)
        flagged = [n for n, c in cav.items() if c.compromised]
        self.assertEqual(flagged, ['delta', 'gamma'])
        self.assertIn('high-pass', cav['delta'].note)
        self.assertIn('low-pass', cav['gamma'].note)

    def test_a_notch_inside_a_band_is_called_out_by_frequency(self):
        cav = filter_caveats(FS, hp=None, lp=None, notch=20.0)['beta']
        self.assertTrue(cav.compromised)
        self.assertIn('notched at 20 Hz', cav.note)

    def test_no_filters_means_no_caveats(self):
        for name, c in filter_caveats(FS).items():
            self.assertFalse(c.compromised, name)
            self.assertEqual(c.note, '')
            self.assertAlmostEqual(c.retained, 1.0, places=6)

    def test_a_filter_that_apply_filters_skips_is_not_reported(self):
        """apply_filters ignores a low-pass at or above Nyquist. Warning about
        a filter that never ran is its own kind of misinformation."""
        cav = filter_caveats(FS, hp=None, lp=200.0, notch=None)
        for name, c in cav.items():
            self.assertEqual(c.note, '', name)

    def test_a_band_above_nyquist_is_declared_unrecorded(self):
        cav = filter_caveats(100.0, hp=1.0, lp=None, notch=None)['gamma']
        self.assertTrue(cav.compromised)
        self.assertIn('Nyquist', cav.note)


class TheRetainedFractionIsMeasuredNotGuessed(unittest.TestCase):
    """The number in the sentence must match what the filters actually do.

    White noise is flat by construction, so the fraction of band power left
    after apply_filters is exactly what `retained` predicts. If these drift
    apart, the panel is quoting a percentage it made up.
    """

    def _measured(self, hp, lp, notch, seed=11):
        x = _noise(seconds=120.0, seed=seed)[None, :]
        y = apply_filters(x, FS, hp=hp, lp=lp, notch=notch)[0]
        before = band_powers(*psd(x[0], FS))
        after = band_powers(*psd(y, FS))
        return dict((n, after[n] / before[n]) for n in BANDS)

    def test_prediction_matches_measurement_at_the_app_defaults(self):
        predicted = filter_caveats(FS, hp=1.0, lp=70.0, notch=60.0)
        measured = self._measured(1.0, 70.0, 60.0)
        for name in BANDS:
            self.assertAlmostEqual(
                predicted[name].retained, measured[name], delta=0.06,
                msg='{}: caveat claims {:.2f} survives, apply_filters left '
                    '{:.2f}'.format(name, predicted[name].retained,
                                    measured[name]))

    def test_prediction_matches_measurement_for_a_review_preset(self):
        predicted = filter_caveats(FS, hp=0.5, lp=15.0, notch=50.0)
        measured = self._measured(0.5, 15.0, 50.0, seed=5)
        for name in ('delta', 'theta', 'alpha', 'beta'):
            self.assertAlmostEqual(
                predicted[name].retained, measured[name], delta=0.06,
                msg='{}: caveat claims {:.2f}, measured {:.2f}'.format(
                    name, predicted[name].retained, measured[name]))

    def test_every_uncompromised_band_really_is_intact(self):
        """The direction that matters clinically: a silent row must be safe."""
        for hp, lp, notch in ((1.0, 70.0, 60.0), (0.5, 35.0, 50.0),
                              (None, None, None), (3.0, 30.0, None)):
            predicted = filter_caveats(FS, hp=hp, lp=lp, notch=notch)
            measured = self._measured(hp, lp, notch, seed=17)
            for name in BANDS:
                if predicted[name].compromised:
                    continue
                self.assertGreater(
                    measured[name], COMPROMISED_BELOW - 0.06,
                    '{} is unflagged under HP={} LP={} notch={} but only '
                    '{:.2f} of it survived'.format(name, hp, lp, notch,
                                                   measured[name]))


class TheTableTheGuiRenders(unittest.TestCase):
    def test_band_table_returns_one_annotated_row_per_band(self):
        rows = band_table(_sine(10.0), FS, hp=1.0, lp=70.0, notch=60.0)
        self.assertEqual([r.name for r in rows], list(BANDS))
        self.assertAlmostEqual(sum(r.ratio for r in rows), 1.0, places=9)
        by_name = dict((r.name, r) for r in rows)
        self.assertTrue(by_name['gamma'].compromised)
        self.assertFalse(by_name['alpha'].compromised)
        self.assertEqual(by_name['alpha'].note, '')
        self.assertGreater(by_name['alpha'].ratio, 0.9)

    def test_rows_carry_their_band_edges_for_the_header(self):
        for row in band_table(_noise(seconds=12.0), FS):
            self.assertEqual((row.lo, row.hi), BANDS[row.name])


class ItIsFastEnoughToScrubWith(unittest.TestCase):
    """This recomputes on every cursor move; a slow version is a broken one.

    The budgets are ~50x the measured cost on the slower of the two supported
    stacks, so this fails on an algorithmic regression rather than on a busy
    laptop.
    """

    def _time(self, x, n=5):
        band_table(x, FS, hp=1.0, lp=70.0, notch=60.0)      # warm the cache
        t0 = time.time()
        for _ in range(n):
            band_table(x, FS, hp=1.0, lp=70.0, notch=60.0)
        return (time.time() - t0) / n

    def test_a_12_second_single_channel_slice(self):
        dt = self._time(_noise(seconds=12.0))
        self.assertLess(dt, 0.10, 'took {:.1f} ms'.format(dt * 1e3))

    def test_a_300_second_single_channel_slice(self):
        dt = self._time(_noise(seconds=300.0))
        self.assertLess(dt, 0.50, 'took {:.1f} ms'.format(dt * 1e3))


if __name__ == '__main__':
    unittest.main()
