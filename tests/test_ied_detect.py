"""The automatic interictal-discharge detector.

The clinician asked for interictal discharges beside the seizure detections and
recalled that open-source detectors exist. SpikeNet1 is one, and its weights
are published rather than gated. These tests pin the parts that are easy to get
silently wrong: the channel contract, the window arithmetic, and the rule that
one discharge produces one mark.

Most run without the weights present, so they work on a machine that has never
fetched them. The handful that need a loaded model skip instead of failing.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io import ied_detect as D                 # noqa: E402

HAVE_WEIGHTS = D.available()


class TheChannelContractIsExact(unittest.TestCase):
    """The weights were trained against one specific electrode order. Getting
    it wrong does not raise — it just feeds the model a scrambled head."""

    def test_nineteen_electrodes_in_spikenets_own_order(self):
        self.assertEqual(len(D.MONOPOLAR), 19)
        self.assertEqual(D.MONOPOLAR[0], 'FP1')
        self.assertEqual(D.MONOPOLAR[-1], 'O2')
        # Not alphabetical, and not the seizure detector's order either.
        self.assertNotEqual(list(D.MONOPOLAR), sorted(D.MONOPOLAR))

    def test_eighteen_bipolar_pairs_make_thirty_seven_inputs(self):
        self.assertEqual(len(D.BIPOLAR), 18)
        self.assertEqual(len(D.MONOPOLAR) + len(D.BIPOLAR), 37)

    def test_every_bipolar_pair_refers_to_electrodes_we_have(self):
        for pair in D.BIPOLAR:
            a, b = pair.split('-')
            self.assertIn(a, D.MONOPOLAR)
            self.assertIn(b, D.MONOPOLAR)

    def test_tuh_style_labels_are_matched(self):
        names = ['EEG {}-REF'.format(e) for e in D.MONOPOLAR]
        self.assertEqual(D.channel_index(names), list(range(19)))

    def test_order_is_corrected_not_assumed(self):
        names = ['EEG {}-REF'.format(e) for e in reversed(D.MONOPOLAR)]
        idx = D.channel_index(names)
        self.assertEqual(idx, list(range(18, -1, -1)))

    def test_extra_channels_are_ignored_not_fed_to_the_model(self):
        names = (['ECG EKG-REF', 'EEG SP1-REF']
                 + ['EEG {}-LE'.format(e) for e in D.MONOPOLAR])
        self.assertEqual(D.channel_index(names), list(range(2, 21)))

    def test_a_missing_electrode_is_an_error_not_a_substitution(self):
        names = ['EEG {}-REF'.format(e) for e in D.MONOPOLAR[:-1]]
        with self.assertRaises(ValueError) as cm:
            D.channel_index(names)
        self.assertIn('O2', str(cm.exception))


class PreprocessingMatchesWhatTheWeightsExpect(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)

    def test_it_produces_thirty_seven_channels(self):
        x = self.rng.randn(19, 128 * 4) * 30
        self.assertEqual(D.preprocess(x, 128).shape[0], 37)

    def test_it_resamples_to_128_hz(self):
        x = self.rng.randn(19, 256 * 4) * 30
        self.assertEqual(D.preprocess(x, 256).shape[1], 128 * 4)

    def test_already_at_128_it_keeps_its_length(self):
        x = self.rng.randn(19, 128 * 4) * 30
        self.assertEqual(D.preprocess(x, 128).shape[1], 128 * 4)

    def test_the_first_nineteen_rows_are_common_average(self):
        x = self.rng.randn(19, 128 * 4) * 30
        out = D.preprocess(x, 128, notch_hz=None)
        # A common-average montage sums to zero across its channels.
        self.assertLess(float(np.abs(out[:19].sum(axis=0)).max()), 1e-2)

    def test_the_wrong_channel_count_is_refused(self):
        with self.assertRaises(ValueError):
            D.preprocess(self.rng.randn(18, 512), 128)

    def test_a_recording_gap_is_detected_as_flat(self):
        self.assertTrue(D.is_flat(np.zeros((37, 512), 'float32')))

    def test_real_signal_is_not_flat(self):
        self.assertFalse(D.is_flat(self.rng.randn(37, 512).astype('float32')
                                   * 30))


class OneDischargeMakesOneMark(unittest.TestCase):
    """At a 16-sample step a single spike trips eight consecutive windows.
    Reporting eight marks would make the count meaningless."""

    def setUp(self):
        self.offsets = np.arange(0, 10, 0.125)          # 8 per second

    def test_a_run_of_hits_collapses_to_one(self):
        scores = np.zeros_like(self.offsets)
        scores[8:16] = 0.95
        self.assertEqual(len(D.merge(scores, self.offsets)), 1)

    def test_the_mark_lands_on_the_highest_scoring_window(self):
        scores = np.zeros_like(self.offsets)
        scores[8:16] = 0.95
        scores[12] = 0.99
        mark = D.merge(scores, self.offsets)[0]
        self.assertAlmostEqual(mark['score'], 0.99, places=6)
        # Reported at the centre of its window, not its leading edge.
        self.assertAlmostEqual(
            mark['time'], self.offsets[12] + D.WINDOW / (2.0 * D.FS), places=6)

    def test_two_discharges_a_second_apart_stay_two(self):
        scores = np.zeros_like(self.offsets)
        scores[8] = 0.95
        scores[24] = 0.95                                # 2 s later
        self.assertEqual(len(D.merge(scores, self.offsets)), 2)

    def test_nothing_above_threshold_is_no_marks(self):
        self.assertEqual(D.merge(np.full_like(self.offsets, 0.1),
                                 self.offsets), [])

    def test_the_threshold_is_respected(self):
        scores = np.zeros_like(self.offsets)
        scores[8] = 0.85
        self.assertEqual(len(D.merge(scores, self.offsets, threshold=0.9)), 0)
        self.assertEqual(len(D.merge(scores, self.offsets, threshold=0.8)), 1)

    def test_t0_offsets_the_mark_into_the_recording(self):
        scores = np.zeros_like(self.offsets)
        scores[8] = 0.95
        a = D.merge(scores, self.offsets)[0]['time']
        b = D.merge(scores, self.offsets, t0=1000.0)[0]['time']
        self.assertAlmostEqual(b - a, 1000.0, places=6)


class AMarkSaysWhereItCameFrom(unittest.TestCase):
    """An unconfirmed machine guess must never be readable as a clinical
    finding, so every mark names its source and carries no channel."""

    def setUp(self):
        offsets = np.arange(0, 4, 0.125)
        scores = np.zeros_like(offsets)
        scores[8] = 0.97
        self.mark = D.merge(scores, offsets)[0]

    def test_it_names_the_detector(self):
        self.assertEqual(self.mark['source'], 'spikenet1')
        self.assertEqual(D.SOURCE, 'spikenet1')

    def test_it_claims_no_channel_because_the_model_localises_nothing(self):
        self.assertIsNone(self.mark['channel'])

    def test_it_is_flagged_unconfirmed(self):
        self.assertIn('unconfirmed', self.mark['note'])

    def test_it_uses_the_same_label_as_a_reviewer_mark(self):
        from gui.io.ied_marks import LABEL
        self.assertEqual(self.mark['label'], LABEL)

    def test_it_carries_its_score(self):
        self.assertAlmostEqual(self.mark['score'], 0.97, places=6)


class TheChannelHintIsNotADetection(unittest.TestCase):
    def test_it_finds_the_loudest_derivation(self):
        eeg = np.zeros((37, 512), 'float32')
        eeg[19 + 3] = 100.0                      # the fourth bipolar pair
        self.assertEqual(D.channel_hint(eeg, 2.0), D.BIPOLAR[3])

    def test_it_is_kept_off_the_mark_itself(self):
        offsets = np.arange(0, 4, 0.125)
        scores = np.zeros_like(offsets)
        scores[8] = 0.97
        self.assertIsNone(D.merge(scores, offsets)[0]['channel'])


class WithoutTheWeightsItSaysSoUsefully(unittest.TestCase):
    def test_a_missing_model_explains_how_to_get_it(self):
        reason = D.missing_reason('/nonexistent/spikenet')
        self.assertIsNotNone(reason)
        self.assertIn('fetch_spikenet', reason)

    def test_available_is_false_when_it_is_not_there(self):
        self.assertFalse(D.available('/nonexistent/spikenet'))

    def test_loading_a_missing_model_raises_the_same_explanation(self):
        with self.assertRaises(RuntimeError) as cm:
            D.load_model('/nonexistent/spikenet')
        self.assertIn('fetch_spikenet', str(cm.exception))


# --------------------------------------------------------------------------
# The model-backed checks run in a SUBPROCESS, and that is not fastidiousness.
# TensorFlow's DLL refuses to initialise in a process where Qt has already been
# loaded -- "DllMain returned false", reproduced directly -- and the whole test
# suite is full of Qt. In-process these tests either skipped spuriously or
# aborted the entire run with STATUS_STACK_BUFFER_OVERRUN, taking 600 unrelated
# tests with them. A clean interpreter per probe costs a few seconds and makes
# the coverage real.
# --------------------------------------------------------------------------

PROBE = r"""
import json, os, sys
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, r'{repo}')
import numpy as np
from gui.io import ied_detect as D

out = {{}}
model = D.load_model()
out['params'] = int(model.count_params())
out['input_shape'] = list(model.input_shape)
out['output_shape'] = list(model.output_shape)

s, o = D.score_windows(model, np.zeros((37, 64), 'float32'))
out['short_segment_scores'] = len(s)

rng = np.random.RandomState(1)
eeg = rng.randn(37, 128 * 4).astype('float32')
s, _ = D.score_windows(model, eeg, step=16)
out['window_count'] = len(s)

eeg = rng.randn(37, 128 * 3).astype('float32') * 30
s, _ = D.score_windows(model, eeg)
out['in_unit_range'] = bool(((s >= 0.0) & (s <= 1.0)).all())
b, _ = D.score_windows(model, (eeg * 10).astype('float32'))
out['amplitude_sensitive'] = float(np.abs(s - b).max())

names = ['EEG {{}}-REF'.format(e) for e in D.MONOPOLAR]
out['gap_gives_nothing'] = (
    D.detect(np.zeros((19, 256 * 10)), 256, names=names, model=model) == [])

# Chunked must equal single-pass. This is the property that makes
# detect_in_chunks usable on a 24-hour file, and it did NOT hold when written.
fs, dur = 256, 200
data = rng.randn(19, fs * dur) * 20.0
# FOCAL, not on every channel. Both of SpikeNet's montages are differential
# -- common-average subtracts the mean, bipolar subtracts pairs -- so a
# transient added identically to all 19 channels cancels to nothing and the
# model sees flat noise. An earlier version of this test did exactly that and
# silently compared two EMPTY lists, proving nothing at all.
focus = [D.MONOPOLAR.index(e) for e in ('F7', 'T3', 'T5')]
for t in range(5, dur - 5, 7):
    c = t * fs
    data[np.ix_(focus, range(c, c + 12))] += np.hanning(12) * 400.0
single = D.detect(data, fs, names=names, threshold=0.5, model=model)
out['single_count'] = len(single)
agree = {{}}
for chunk_s in (60.0, 37.0, 25.0):
    ch = D.detect_in_chunks(lambda a, b: data[:, a:b], data.shape[1], fs,
                            names=names, chunk_s=chunk_s, threshold=0.5,
                            model=model)
    agree[str(chunk_s)] = ([round(m['time'], 6) for m in ch]
                           == [round(m['time'], 6) for m in single])
out['chunking_agrees'] = agree

seam = rng.randn(19, fs * 120) * 20.0
seam[np.ix_(focus, range(60 * fs, 60 * fs + 12))] += np.hanning(12) * 400.0
marks = D.detect_in_chunks(lambda a, b: seam[:, a:b], seam.shape[1], fs,
                           names=names, chunk_s=60.0, threshold=0.5,
                           model=model)
out['on_seam'] = sum(1 for m in marks if abs(m['time'] - 60.0) < 1.0)

print('@@' + json.dumps(out))
"""


def _probe():
    """Run the model probe once; cached for the class."""
    import subprocess
    env = dict(os.environ)
    env['TF_USE_LEGACY_KERAS'] = '1'
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['PYTHONWARNINGS'] = 'ignore'
    env.pop('QT_QPA_PLATFORM', None)
    r = subprocess.run([sys.executable, '-c', PROBE.format(repo=REPO)],
                       cwd=REPO, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    text = r.stdout.decode('utf-8', 'replace')
    for line in text.splitlines():
        if line.startswith('@@'):
            import json
            return json.loads(line[2:])
    raise unittest.SkipTest(
        'model probe produced no result (TensorFlow unavailable?): '
        + r.stderr.decode('utf-8', 'replace')[-800:])


@unittest.skipUnless(HAVE_WEIGHTS,
                     'SpikeNet weights not fetched; run '
                     'experiments/fetch_spikenet.py')
class TheRealModel(unittest.TestCase):
    _cache = None

    @classmethod
    def setUpClass(cls):
        if TheRealModel._cache is None:
            TheRealModel._cache = _probe()
        cls.out = TheRealModel._cache

    def test_the_published_architecture(self):
        self.assertEqual(self.out['params'], 318849)
        self.assertEqual(self.out['input_shape'], [None, 128, 1, 37])
        self.assertEqual(self.out['output_shape'], [None, 1])

    def test_a_segment_shorter_than_one_window_scores_nothing(self):
        self.assertEqual(self.out['short_segment_scores'], 0)

    def test_the_window_count_follows_the_step(self):
        self.assertEqual(self.out['window_count'], (128 * 4 - 128) // 16 + 1)

    def test_every_score_is_a_probability(self):
        self.assertTrue(self.out['in_unit_range'])

    def test_the_output_is_amplitude_sensitive(self):
        """Documented because it is a trap: the same EEG in volts rather than
        microvolts scores completely differently, and nothing raises."""
        self.assertGreater(self.out['amplitude_sensitive'], 0.01)

    def test_a_recording_gap_yields_no_candidates(self):
        self.assertTrue(self.out['gap_gives_nothing'])

    def test_chunking_a_recording_changes_nothing(self):
        """The property that makes detect_in_chunks usable on a 24-hour file.

        It did NOT hold when this was written. scipy's FFT `resample` is
        global, so a chunk resampled differently from the same span inside a
        longer segment; and each chunk started its own window grid, so windows
        landed at a different phase against a discharge. Both were invisible --
        the marks merely moved and borderline ones flipped.
        """
        self.assertTrue(self.out['single_count'] > 0)
        for chunk_s, agrees in sorted(self.out['chunking_agrees'].items()):
            self.assertTrue(agrees,
                            'chunk_s={} disagreed with a single pass'
                            .format(chunk_s))

    def test_a_chunk_boundary_does_not_duplicate_a_mark(self):
        self.assertLessEqual(self.out['on_seam'], 1,
                             'a discharge on the seam was reported twice')


if __name__ == '__main__':
    unittest.main()
