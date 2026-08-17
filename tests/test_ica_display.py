"""The display-side ICA must be able to lie about nothing.

`gui/io/ica_display.py` re-runs the frozen `ica_arti_remove` so a reviewer can
see the signal the detector was actually given. Three ways that feature could
quietly become a clinical hazard, and one way it could become useless:

* **It writes through.** `_raw_data` and `_original_data` are the same numpy
  object in `gui/app.py`. A display routine that mutated its input would change
  what the detector scores, and every existing test would still pass.
* **It hides the pass-through.** `ica_arti_remove` returns its input untouched
  when neither Fp1 nor Fp2 yields an EOG component (~10 % of windows; see
  `test_paper_conformance.test_DEVIATION_no_eog_found_returns_the_raw_...`).
  Drawing that window as "ICA cleaned" tells the reviewer the detector saw
  something it did not.
* **It reports the wrong components.** The excluded indices are recovered from
  what the frozen function prints. If that recovery silently stops working, the
  GUI must say "not recoverable", never a stale or invented index.
* **It re-runs ICA on every repaint.** 0.2-0.3 s per window; without a bounded
  cache, scrubbing is unusable and a long recording exhausts memory.

The pass-through fixture is a *search*, not a fixed seed. Which segments take
that branch is a property of the installed MNE and FastICA, and it differs
between the two supported stacks — a seed hard-coded from one of them would
have made this test a statement about the developer's laptop.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    from gui.io import ica_display
    from gui.io.edf import CHANNELS_19
    from gui.io.infer import SEGMENT_S, SKIP_ICA_FAILED, SKIP_INTERRUPTED
    HAVE_STACK = True
except Exception:                                    # pragma: no cover
    HAVE_STACK = False

FS = 250
N_WIN = 12 * FS


def eeg_like(seed):
    """A 12 s, 19-channel segment with EEG-ish structure.

    Band-limited and sinusoid-dominated on purpose: FastICA converges on this
    in ~0.2 s, where it hits its iteration limit on white noise and takes ten
    times as long. A test suite that took a minute per assertion would not be
    run.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(N_WIN) / float(FS)
    rows = []
    for i in range(19):
        f = 6.0 + 0.7 * ((i + seed) % 11)
        rows.append(30.0 * np.sin(2 * np.pi * f * t + 0.37 * i + 0.11 * seed)
                    + 10.0 * rng.randn(N_WIN))
    return np.stack(rows).astype(np.float32)


def with_blink(seed):
    """The same, plus a slow high-amplitude frontal deflection.

    This is the case the ICA exists for: something big and eye-like on Fp1/Fp2
    that `find_bads_eog` will pick out as a component.
    """
    data = eeg_like(seed)
    t = np.arange(N_WIN) / float(FS)
    blink = 150.0 * np.sin(2 * np.pi * 0.4 * t)
    data[CHANNELS_19.index('Fp1')] += blink
    data[CHANNELS_19.index('Fp2')] += 0.9 * blink
    return data


# Seeds observed to take the raw pass-through branch on both supported stacks
# (Python 3.6 / MNE 0.19.2 and Python 3.11 / MNE 1.12.1). Searched in order so
# a stack where the first few no longer qualify still finds one.
PASSTHROUGH_CANDIDATES = (1, 2, 3, 4, 12, 13, 14, 15, 16, 23)


@unittest.skipUnless(HAVE_STACK, 'MNE/pyedflib inference stack not available')
class TheCallersArrayIsNeverTouched(unittest.TestCase):
    """The one rule: the display path must not reach the detector's input."""

    def setUp(self):
        ica_display.cache_clear()
        self.data = with_blink(0)

    def test_clean_for_display_does_not_mutate_its_input(self):
        before = self.data.copy()
        ica_display.clean_for_display(self.data, FS)
        np.testing.assert_array_equal(
            self.data, before,
            'clean_for_display modified its input. _raw_data and '
            '_original_data are the same object on load, so this silently '
            'changes what the detector scores.')

    def test_the_cleaned_array_shares_no_memory_with_the_input(self):
        out, _info = ica_display.clean_for_display(self.data, FS)
        self.assertFalse(np.shares_memory(out, self.data))

    def test_the_pass_through_case_also_shares_no_memory(self):
        """The branch that hands back its own argument is the dangerous one."""
        data, info = _first_passthrough(self)
        out, _ = ica_display.clean_for_display(data, FS)
        self.assertTrue(info.passthrough)
        self.assertFalse(
            np.shares_memory(out, data),
            'the pass-through branch returned a view onto the caller array; '
            'a later display edit would reach the detector')

    def test_a_read_only_input_is_accepted(self):
        """Cached display arrays are read-only, and get passed around."""
        ro = self.data.copy()
        ro.setflags(write=False)
        out, _info = ica_display.clean_for_display(ro, FS)
        self.assertEqual(out.shape, ro.shape)

    def test_the_span_variant_does_not_mutate_its_input(self):
        data = np.concatenate([with_blink(0), eeg_like(1)], axis=1)
        before = data.copy()
        ica_display.clean_span_for_display(data, FS, 0.0, 12.0)
        np.testing.assert_array_equal(data, before)

    def test_output_is_read_only_so_the_cache_cannot_be_corrupted(self):
        out, _info = ica_display.clean_for_display(
            self.data, FS, recording_id='rec', start_s=0)
        with self.assertRaises(ValueError):
            out[0, 0] = 1.0

    def test_the_assembled_span_is_writeable(self):
        """The caller filters and montages it; a frozen span would break that."""
        data = np.concatenate([with_blink(0), eeg_like(1)], axis=1)
        out, _t0, _blocks = ica_display.clean_span_for_display(
            data, FS, 0.0, 12.0)
        out[0, 0] = 1.0          # must not raise


@unittest.skipUnless(HAVE_STACK, 'MNE/pyedflib inference stack not available')
class TheRawPassThroughIsReported(unittest.TestCase):
    """~10 % of windows reach the model uncleaned. The reviewer must be told."""

    def setUp(self):
        ica_display.cache_clear()

    def test_a_segment_with_no_eog_component_is_flagged(self):
        data, info = _first_passthrough(self)
        self.assertTrue(info.passthrough)
        self.assertFalse(info.cleaned)
        self.assertFalse(info.failed)
        self.assertIsNone(info.excluded_fp1)
        self.assertIsNone(info.excluded_fp2)
        self.assertEqual(info.excluded, [])
        self.assertEqual(info.max_abs_change, 0.0)
        self.assertIn('no EOG component', info.label())

    def test_the_pass_through_output_equals_the_input(self):
        """Not "close to" — the frozen function returns the array itself."""
        data, info = _first_passthrough(self)
        out, _ = ica_display.clean_for_display(data, FS)
        np.testing.assert_array_equal(np.asarray(out), data)

    def test_the_displayed_trace_is_the_detectors_own_ica(self):
        """The point of the whole feature, stated as an equality.

        `gui/io/infer.py` calls `ica_arti_remove(seg, fs, CHANNELS_19)` on a
        float32 slice of the loaded EDF. This module must produce that array and
        not merely something like it — which is why the working copy keeps the
        caller's dtype instead of being widened to float64 for convenience.
        `create_mne_raw` scales by 1e-6 *before* MNE widens, so widening first
        is different arithmetic, and a decomposition is entitled to notice.
        """
        from utils.preprocessing import ica_arti_remove
        seg = with_blink(0)
        direct = ica_arti_remove(np.array(seg, copy=True), FS,
                                 list(CHANNELS_19))
        self.assertIsNotNone(direct, 'fixture no longer exercises the ICA')
        shown, info = ica_display.clean_for_display(
            seg, FS, check_interrupted=False)
        self.assertTrue(info.cleaned)
        np.testing.assert_array_equal(
            np.asarray(shown), np.asarray(direct).astype(np.float32),
            'the trace shown to the reviewer is not the array the detector '
            'was given, so "this is what the AI saw" would be false')

    def test_a_cleaned_window_is_not_flagged_as_pass_through(self):
        _out, info = ica_display.clean_for_display(with_blink(0), FS)
        self.assertFalse(info.passthrough)
        self.assertTrue(info.cleaned)
        self.assertGreater(info.max_abs_change, 0.0)

    def test_the_flag_comes_from_comparing_arrays_not_from_the_printed_list(self):
        """A stub that returns its input must still be seen as a pass-through.

        The requirement is that the branch is detected by comparison, so it
        keeps working if the frozen function stops printing, or prints
        something new. Patching the module's own import site is the only way to
        assert that without editing utils/preprocessing.py.
        """
        import utils.preprocessing as pp
        real = pp.ica_arti_remove

        def echo(data, sfreq, chs=None):
            print('eog_indices [4]')      # deliberately inconsistent
            print('eog_indices [5]')
            print('ica.exclude [4, 5]')
            return data

        pp.ica_arti_remove = echo
        try:
            _out, info = ica_display.clean_for_display(
                eeg_like(0), FS, check_interrupted=False)
        finally:
            pp.ica_arti_remove = real
        self.assertTrue(info.passthrough,
                        'pass-through was inferred from the printed exclude '
                        'list rather than from the returned array')
        self.assertFalse(info.indices_known,
                         'the printed list contradicted the array and was '
                         'reported anyway')
        self.assertIsNone(info.excluded_fp1)


@unittest.skipUnless(HAVE_STACK, 'MNE/pyedflib inference stack not available')
class ExcludedComponentsAreReported(unittest.TestCase):
    """"Which components did it remove" is the reviewer's follow-up question."""

    def setUp(self):
        ica_display.cache_clear()

    def test_indices_are_recovered_for_a_cleaned_window(self):
        _out, info = ica_display.clean_for_display(with_blink(0), FS)
        self.assertTrue(info.indices_known,
                        'the excluded indices could not be recovered from the '
                        'frozen function; the GUI has nothing to show')
        self.assertTrue(info.excluded,
                        'a cleaned window reported an empty exclude list')
        for idx in info.excluded:
            self.assertIsInstance(idx, int)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(CHANNELS_19))

    def test_fp1_and_fp2_are_reported_separately(self):
        """The frozen function takes the top component from each, separately."""
        _out, info = ica_display.clean_for_display(with_blink(0), FS)
        found = [i for i in (info.excluded_fp1, info.excluded_fp2)
                 if i is not None]
        self.assertTrue(found, 'neither Fp1 nor Fp2 reported a component, yet '
                                'the window was cleaned')
        for idx in found:
            self.assertIn(idx, info.excluded)
        self.assertIn('IC{}'.format(found[0]), info.label())

    def test_unparsable_output_reports_nothing_rather_than_guessing(self):
        self.assertEqual(ica_display._parse_indices(''),
                         (None, None, [], False))
        self.assertEqual(
            ica_display._parse_indices('eog_indices [1]\nica.exclude [1]\n'),
            (None, None, [], False))      # only one eog line: shape is wrong
        self.assertEqual(
            ica_display._parse_indices(
                'eog_indices [1]\neog_indices [2]\n'
                'ica.exclude [1, 2]\nica.exclude [3]\n'),
            (None, None, [], False))      # something else printed into us
        self.assertEqual(
            ica_display._parse_indices(
                'eog_indices [nope]\neog_indices []\nica.exclude []\n'),
            (None, None, [], False))

    def test_a_well_formed_capture_parses(self):
        self.assertEqual(
            ica_display._parse_indices(
                'Creating RawArray with float64 data\n'
                'eog_indices []\n'
                'eog_indices [7, 11]\n'
                'ica.exclude [7]\n'
                'Zeroing out 1 ICA components\n'),
            (None, 7, [7], True))


@unittest.skipUnless(HAVE_STACK, 'MNE/pyedflib inference stack not available')
class TheWindowsTheDetectorRefusedAreDistinguished(unittest.TestCase):
    """"No score" and "a score of no" must not look the same here either."""

    def setUp(self):
        ica_display.cache_clear()

    def test_an_interrupted_window_is_not_shown_as_cleaned(self):
        flat = np.zeros((19, N_WIN), dtype=np.float32)
        _out, info = ica_display.clean_for_display(flat, FS)
        self.assertTrue(info.interrupted)
        self.assertFalse(info.cleaned)
        self.assertEqual(info.skip_code, SKIP_INTERRUPTED)

    def test_an_ica_failure_is_reported_as_a_failure(self):
        flat = np.zeros((19, N_WIN), dtype=np.float32)
        _out, info = ica_display.clean_for_display(
            flat, FS, check_interrupted=False)
        self.assertTrue(info.failed)
        self.assertEqual(info.skip_code, SKIP_ICA_FAILED)
        self.assertIn('failed', info.label())

    def test_a_short_window_is_flagged_as_incomplete(self):
        short = with_blink(0)[:, :6 * FS]
        _out, info = ica_display.clean_for_display(short, FS)
        self.assertTrue(info.incomplete,
                        'a 6 s span is not the decomposition the model was '
                        'given, and must not be presented as one')

    def test_a_wrong_shaped_array_is_refused(self):
        with self.assertRaises(RuntimeError):
            ica_display.clean_for_display(np.zeros((18, N_WIN)), FS)
        with self.assertRaises(RuntimeError):
            ica_display.clean_for_display(np.zeros(N_WIN), FS)


@unittest.skipUnless(HAVE_STACK, 'MNE/pyedflib inference stack not available')
class TheCacheIsCorrectAndBounded(unittest.TestCase):
    """Scrubbing repaints constantly; a 24 h study is 14 400 windows."""

    def setUp(self):
        ica_display.cache_clear()
        self.limit = ica_display.cache_stats()['limit']
        self.data = with_blink(0)

    def tearDown(self):
        ica_display.set_cache_limit(self.limit)
        ica_display.cache_clear()

    def test_a_second_call_returns_the_identical_array(self):
        a, info_a = ica_display.clean_for_display(
            self.data, FS, recording_id='rec', start_s=0)
        b, info_b = ica_display.clean_for_display(
            self.data, FS, recording_id='rec', start_s=0)
        self.assertIs(a, b, 'the cache re-ran ICA instead of returning the '
                            'array it already had')
        np.testing.assert_array_equal(a, b)
        self.assertFalse(info_a.cached)
        self.assertTrue(info_b.cached)
        self.assertEqual(ica_display.cache_stats()['hits'], 1)

    def test_the_cached_info_is_a_copy_not_the_stored_object(self):
        """A span annotates the info it gets back; the cache must not learn it."""
        _a, info_a = ica_display.clean_for_display(
            self.data, FS, recording_id='rec', start_s=0)
        info_a.incomplete = True
        _b, info_b = ica_display.clean_for_display(
            self.data, FS, recording_id='rec', start_s=0)
        self.assertFalse(info_b.incomplete)

    def test_different_windows_do_not_collide(self):
        a, _ = ica_display.clean_for_display(
            self.data, FS, recording_id='rec', start_s=0)
        b, _ = ica_display.clean_for_display(
            eeg_like(0), FS, recording_id='rec', start_s=6)
        self.assertIsNot(a, b)
        self.assertEqual(ica_display.cache_stats()['size'], 2)

    def test_a_different_recording_does_not_collide(self):
        ica_display.clean_for_display(
            self.data, FS, recording_id='rec-a', start_s=0)
        ica_display.clean_for_display(
            self.data, FS, recording_id='rec-b', start_s=0)
        self.assertEqual(ica_display.cache_stats()['size'], 2)
        self.assertEqual(ica_display.cache_stats()['hits'], 0)

    def test_the_cache_is_bounded_and_evicts_least_recently_used(self):
        ica_display.set_cache_limit(3)
        for start in (0, 6, 12, 18, 24):
            ica_display.clean_for_display(
                self.data, FS, recording_id='rec', start_s=start)
            self.assertLessEqual(ica_display.cache_stats()['size'], 3,
                                 'the cache grew past its limit; a long '
                                 'recording would exhaust memory')
        self.assertEqual(ica_display.cache_stats()['size'], 3)
        # 0 and 6 were evicted; 24 is still resident.
        ica_display.clean_for_display(
            self.data, FS, recording_id='rec', start_s=24)
        self.assertEqual(ica_display.cache_stats()['hits'], 1)

    def test_lowering_the_limit_evicts_immediately(self):
        for start in (0, 6, 12):
            ica_display.clean_for_display(
                self.data, FS, recording_id='rec', start_s=start)
        ica_display.set_cache_limit(1)
        self.assertEqual(ica_display.cache_stats()['size'], 1)

    def test_nothing_is_cached_without_a_recording_id(self):
        """No safe key means no cache, rather than a key that could be wrong."""
        ica_display.clean_for_display(self.data, FS)
        self.assertEqual(ica_display.cache_stats()['size'], 0)

    def test_the_recording_key_changes_when_the_file_does(self):
        import tempfile
        fd, path = tempfile.mkstemp(suffix='.edf')
        try:
            os.write(fd, b'x' * 16)
            os.close(fd)
            first = ica_display.recording_key(path)
            with open(path, 'ab') as f:
                f.write(b'y' * 32)
            self.assertNotEqual(first, ica_display.recording_key(path))
        finally:
            if os.path.exists(path):
                os.remove(path)


@unittest.skipUnless(HAVE_STACK, 'MNE/pyedflib inference stack not available')
class TheVisibleSpanLinesUpWithWhatWasScored(unittest.TestCase):
    """Blocks must be detector windows, or the display is a different signal."""

    def setUp(self):
        ica_display.cache_clear()

    def test_the_window_grid_matches_the_inference_grid(self):
        """Same rule as compute_probs_from_data: int seconds, stride 6."""
        for n_seconds in (11, 12, 13, 30, 31, 305):
            n = n_seconds * FS
            expected = list(range(0, int(n / float(FS)) - SEGMENT_S + 1,
                                  ica_display.STEP_S))
            self.assertEqual(
                ica_display.detector_window_starts(n, FS), expected,
                'display blocks would not line up with the scored windows '
                'for a {} s recording'.format(n_seconds))

    def test_a_short_recording_has_no_scored_windows(self):
        self.assertEqual(ica_display.detector_window_starts(5 * FS, FS), [])

    def test_blocks_are_segment_spaced_and_anchor_snaps_to_the_grid(self):
        starts = ica_display._aligned_block_starts(0.0, 30.0, 0)
        self.assertEqual(starts, [0, 12, 24])
        # An anchor of 18 s is on the detector's 6 s grid: keep it.
        self.assertEqual(
            ica_display._aligned_block_starts(18.0, 30.0, 18), [18])
        # 20 s is not; snap to 18 so the block is a window that was scored.
        self.assertEqual(
            ica_display._aligned_block_starts(18.0, 30.0, 20), [18])
        for starts in (ica_display._aligned_block_starts(5.0, 40.0, 0),
                       ica_display._aligned_block_starts(5.0, 40.0, 18)):
            self.assertTrue(all(s % ica_display.STEP_S == 0 for s in starts))
            self.assertTrue(all(b - a == SEGMENT_S
                                for a, b in zip(starts, starts[1:])))

    def test_a_thirty_second_span_is_cleaned_block_by_block(self):
        data = np.concatenate(
            [with_blink(0), eeg_like(0), with_blink(1)], axis=1)   # 36 s
        out, t0, blocks = ica_display.clean_span_for_display(
            data, FS, 0.0, 30.0, recording_id='rec')
        self.assertEqual(t0, 0.0)
        self.assertEqual(out.shape, (19, 30 * FS))
        self.assertEqual(len(blocks), 3)
        for info, lo, hi in blocks:
            self.assertEqual(info.fs, FS)
            self.assertGreater(hi, lo)
        self.assertIn('ICA over 3 window(s)', ica_display.summarise(blocks))

    def test_the_span_actually_differs_from_the_raw_signal(self):
        data = np.concatenate([with_blink(0), with_blink(1)], axis=1)
        out, _t0, blocks = ica_display.clean_span_for_display(
            data, FS, 0.0, 24.0, recording_id='rec')
        cleaned_any = any(info.cleaned for info, _lo, _hi in blocks)
        self.assertTrue(cleaned_any)
        self.assertFalse(np.allclose(out, data[:, :24 * FS]),
                         'the span came back identical to the raw signal, so '
                         'the toggle would show the reviewer nothing')

    def test_the_unscored_tail_is_shown_raw_and_flagged(self):
        """20 s recording: the detector scored windows at 0 s and 6 s only.

        The last one ends at 18 s, so the block the display grid puts at 12 s
        runs two seconds past anything that was ever scored. Those two seconds
        must come back raw and flagged, not as a decomposition the detector
        never made.
        """
        data = np.concatenate([with_blink(0), eeg_like(0)], axis=1)[:, :20 * FS]
        out, _t0, blocks = ica_display.clean_span_for_display(
            data, FS, 0.0, 20.0, recording_id='rec')
        self.assertEqual(out.shape, (19, 20 * FS))
        self.assertTrue(any(info.incomplete for info, _lo, _hi in blocks),
                        'no block reported that part of the span was never '
                        'scored')

    def test_a_recording_shorter_than_one_window_is_all_raw(self):
        data = with_blink(0)[:, :5 * FS]
        out, t0, blocks = ica_display.clean_span_for_display(data, FS, 0.0, 5.0)
        np.testing.assert_array_equal(out, data)
        self.assertEqual(t0, 0.0)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(blocks[0][0].incomplete)

    def test_the_span_reuses_the_cache_across_repaints(self):
        data = np.concatenate([with_blink(0), eeg_like(0)], axis=1)
        ica_display.clean_span_for_display(
            data, FS, 0.0, 24.0, recording_id='rec')
        misses = ica_display.cache_stats()['misses']
        ica_display.clean_span_for_display(
            data, FS, 0.0, 24.0, recording_id='rec')
        self.assertEqual(ica_display.cache_stats()['misses'], misses,
                         'a repaint of the same span re-ran ICA')

    def test_summarise_names_the_pass_through_windows(self):
        info = ica_display.IcaDisplayInfo()
        info.passthrough = True
        text = ica_display.summarise([(info, 0.0, 12.0)])
        self.assertIn('UNCLEANED', text)
        self.assertIn('1 passed through', text)

    def test_info_serialises_to_plain_types(self):
        _out, info = ica_display.clean_for_display(with_blink(0), FS)
        payload = info.to_dict()
        import json
        json.loads(json.dumps(payload))
        self.assertIn('passthrough', payload)
        self.assertIn('excluded', payload)


# ---------------------------------------------------------------- fixtures

_PASSTHROUGH = {}


def _first_passthrough(case):
    """The first candidate segment this stack refuses to find an EOG in.

    Cached across tests because each miss costs a real ICA run. Fails loudly
    rather than skipping: if no candidate takes the branch, the branch has
    stopped being reachable on this stack and the ~10 % figure the GUI reports
    to the reviewer is no longer true of it.
    """
    if 'hit' in _PASSTHROUGH:
        seed = _PASSTHROUGH['hit']
        data = eeg_like(seed)
        _out, info = ica_display.clean_for_display(data, FS)
        return data, info
    tried = []
    for seed in PASSTHROUGH_CANDIDATES:
        data = eeg_like(seed)
        _out, info = ica_display.clean_for_display(data, FS)
        tried.append(seed)
        if info.passthrough:
            _PASSTHROUGH['hit'] = seed
            return data, info
    case.fail(
        'no candidate segment took ica_arti_remove\'s raw pass-through branch '
        'on this stack (tried seeds {}). Either the frozen function changed, '
        'or this MNE/FastICA finds an EOG component in everything - in which '
        'case the ~10 % pass-through rate the GUI reports is wrong for this '
        'build.'.format(tried))


if __name__ == '__main__':
    unittest.main()
