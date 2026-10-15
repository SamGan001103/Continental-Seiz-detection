"""A recording too large to hold in memory must still open.

The first version of the memory guard refused them. That was the wrong answer,
and it took being told so to see it: day-long ambulatory recordings are what
this tool exists for, and one that declines them is not safer than one that
freezes — it is useless, which is a different way of not working.

So the guard now chooses a strategy instead of a verdict:

* **Short** → the ordinary loader, unchanged. Resampled to 250 Hz, the same
  bytes as before this existed. Small files must not change behaviour at all,
  and the first test here holds that.

* **Longer than EAGER_MAX_DURATION_S, or too large to hold** → `EdfSource`,
  read span-by-span off disk at the file's own rate. A 24-hour recording costs
  ~0.15 GB and 1 ms to open, against ~2 minutes and a 5 GB peak held.

* **Cannot be held AND cannot be read lazily** → refused, because now there
  genuinely is no way to show it without taking the machine down. A recording
  that merely prefers streaming and cannot get it is still loaded whole.

Length rather than free memory, because the original test -- does it FIT --
got the answer wrong in the direction of the better machine. The same day-long
recording streamed instantly on 16 GB, where 21 GB of estimate did not fit,
and took two minutes once 64 GB made it fit. Holding it buys only the 250 Hz
resample the ICA view needs, which that view already declines without.

The interesting property is the second one. The detector needs 250 Hz; the
DISPLAY does not, and nothing on that side assumes it — every widget takes fs
as a parameter and every time is in seconds. So a trace drawn at a TUH file's
native 256 Hz lines up with probabilities computed at 250 Hz, which the
alignment test below checks against real ground truth rather than by assertion.

Resampling a span was considered and rejected on measurement: an FFT resample
of a slice differs from the corresponding slice of the resampled whole by up to
1200 uV at the edges on a real TUH recording. That is why the detector-input
(ICA) view declines at native rate rather than approximating — it promises to
show what the model was fed, and a nearly-right version of that is worse than
none, because the reviewer would be judging the detector on a trace it never
saw.
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

try:
    import PyQt5  # noqa: F401
    HAVE_QT = True
except Exception:                                            # noqa: BLE001
    HAVE_QT = False

if HAVE_QT:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtWidgets

CH = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ',
      'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']


def _write_edf(path, seconds=60, fs=250, seed=3):
    import pyedflib
    rng = np.random.RandomState(seed)
    labels = ['EEG {}-REF'.format(c) for c in CH]
    sigs = rng.randn(len(labels), int(seconds * fs)) * 40.0

    def _attempt(rate_key):
        w = pyedflib.EdfWriter(path, len(labels),
                               file_type=pyedflib.FILETYPE_EDFPLUS)
        try:
            w.setSignalHeaders([{
                'label': lab, 'dimension': 'uV', rate_key: fs,
                'physical_min': -3276.8, 'physical_max': 3276.7,
                'digital_min': -32768, 'digital_max': 32767,
                'prefilter': '', 'transducer': ''} for lab in labels])
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


def _write_cache(path, seconds, event=None):
    """A 250 Hz probability cache, as precompute_probs would leave."""
    from gui.io.cache import save_probs
    starts = np.arange(0, max(1, int(seconds) - 12 + 1), 6, dtype=np.int32)
    probs = np.zeros(starts.size, dtype=np.float32)
    if event is not None:
        a, b = event
        probs[(starts + 12 > a) & (starts < b)] = 0.95
    save_probs(path, starts, probs,
               skip_code=np.zeros(starts.size, dtype=np.int8),
               meta={'step_s': 6, 'segment_s': 12, 'use_ica': True,
                     'fs': 250, 'weights': 'convlstm_ICA_12_train.h5'})


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class StrategyByFileSize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(['-platform', 'offscreen']))

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='bigopen')
        import gui.memory as gm
        self._saved_avail = gm.available_bytes
        self._saved_warn = QtWidgets.QMessageBox.warning
        self.warnings = []
        QtWidgets.QMessageBox.warning = staticmethod(
            lambda *a, **k: self.warnings.append(a[1] if len(a) > 1 else 'warn'))
        from gui.app import MainWindow
        self.w = MainWindow()

    def tearDown(self):
        import gui.memory as gm
        gm.available_bytes = self._saved_avail
        if getattr(self, '_saved_estimate', None) is not None:
            gm.edf_open_estimate = self._saved_estimate
        QtWidgets.QMessageBox.warning = self._saved_warn
        self.w.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _plenty(self):
        import gui.memory as gm
        gm.available_bytes = lambda: 64 << 30

    def _none(self):
        import gui.memory as gm
        gm.available_bytes = lambda: 1

    def _claim_duration(self, hours, fits=True):
        """Report a short real file as a long one.

        The policy reads the header estimate, so this is the one honest way to
        test a day-long recording without writing a day of samples: the file
        on disk stays small and readable, and only the estimate the decision
        is made on is substituted.
        """
        import gui.memory as gm
        self._saved_estimate = gm.edf_open_estimate
        real = gm.edf_open_estimate

        def claimed(path, available=None):
            est = dict(real(path, available=available))
            est['duration_s'] = hours * 3600.0
            est['fits'] = fits
            return est
        gm.edf_open_estimate = claimed

    def test_a_file_that_fits_takes_the_ordinary_path_unchanged(self):
        """Small recordings must behave exactly as before any of this existed."""
        self._plenty()
        p = _write_edf(os.path.join(self.tmp, 'small.edf'), seconds=60, fs=256)
        _write_cache(p, 60)
        self.w.load_edf(p)
        self.assertIsNotNone(self.w._raw_data)
        self.assertEqual(int(round(self.w._fs)), 250,
                         'a file that fits must still be resampled to the '
                         "detector's rate by the ordinary loader")
        self.assertIsInstance(self.w._raw_data, np.ndarray,
                              'a file that fits must be loaded, not streamed')
        self.assertEqual(self.warnings, [])

    def test_a_long_recording_is_streamed_even_with_memory_to_spare(self):
        """The regression: a 64 GB workstation was punished for its memory.

        The same 24-hour file opened in 1 ms on a 16 GB machine, where the
        estimate did not fit, and took two minutes once 64 GB made it fit --
        reading the whole recording to produce something only the ICA view
        wants. Length decides this now, not the machine.
        """
        self._plenty()
        self._claim_duration(24, fits=True)
        p = _write_edf(os.path.join(self.tmp, 'daylong.edf'), seconds=60,
                       fs=256)
        _write_cache(p, 60)
        self.w.load_edf(p)
        self.assertIsNotNone(self.w._raw_data)
        self.assertNotIsInstance(
            self.w._raw_data, np.ndarray,
            'a day-long recording was loaded whole because it happened to '
            'fit, which is the two-minute open this policy exists to avoid')
        self.assertEqual(int(round(self.w._fs)), 256,
                         'the streamed path must serve the native rate')
        self.assertEqual(self.warnings, [])

    def test_an_hour_long_recording_is_still_loaded_whole(self):
        """The threshold is a threshold, not a ban on holding anything."""
        self._plenty()
        self._claim_duration(1, fits=True)
        p = _write_edf(os.path.join(self.tmp, 'hour.edf'), seconds=60, fs=256)
        _write_cache(p, 60)
        self.w.load_edf(p)
        self.assertIsInstance(
            self.w._raw_data, np.ndarray,
            'a recording at the threshold was streamed, so short files lose '
            'the detector-input view for no gain')
        self.assertEqual(int(round(self.w._fs)), 250)

    def test_a_long_recording_that_will_not_stream_is_loaded_not_refused(self):
        """Preferring the stream must not become a new way to refuse a file.

        It fits, so there is a working answer; slow and correct beats a
        dialog.
        """
        self._plenty()
        self._claim_duration(24, fits=True)
        p = _write_edf(os.path.join(self.tmp, 'nostream.edf'), seconds=60,
                       fs=256)
        _write_cache(p, 60)
        import gui.io.edf_source as es
        saved = es.open_source
        es.open_source = lambda *a, **k: None
        try:
            self.w.load_edf(p)
        finally:
            es.open_source = saved
        self.assertIsInstance(self.w._raw_data, np.ndarray)
        self.assertEqual(self.warnings, [],
                         'a recording the machine can hold was refused')

    def test_a_file_too_large_opens_lazily_at_its_own_rate(self):
        self._none()
        p = _write_edf(os.path.join(self.tmp, 'big.edf'), seconds=60, fs=256)
        _write_cache(p, 60)
        self.w.load_edf(p)
        self.assertIsNotNone(self.w._raw_data,
                             'a recording too large to hold was refused '
                             'instead of being read from disk')
        self.assertEqual(int(round(self.w._fs)), 256,
                         'the lazy path must serve the native rate')
        self.assertEqual(self.warnings, [],
                         'the reviewer was shown a refusal for a file that '
                         'did in fact open')

    def test_events_scored_at_250hz_land_correctly_on_a_256hz_trace(self):
        """The property the whole native-rate decision rests on."""
        self._none()
        p = _write_edf(os.path.join(self.tmp, 'align.edf'), seconds=120, fs=256)
        _write_cache(p, 120, event=(48.0, 72.0))
        self.w.load_edf(p)
        self.assertTrue(self.w._events, 'no event was proposed from the cache')
        ev = self.w._events[0]
        self.assertGreaterEqual(ev['start'], 40.0)
        self.assertLessEqual(ev['stop'], 80.0)
        self.assertAlmostEqual(self.w._duration_s, 120.0, delta=1.0,
                               msg='duration is wrong at native rate, so every '
                                   'time on screen is wrong')

    def test_the_detector_input_view_declines_at_native_rate(self):
        self._none()
        p = _write_edf(os.path.join(self.tmp, 'ica.edf'), seconds=60, fs=256)
        _write_cache(p, 60)
        self.w.load_edf(p)
        self.w.chk_ica.setChecked(True)
        self.w._display_source()
        self.assertFalse(
            self.w.chk_ica.isChecked(),
            'the detector-input view stayed on at 256 Hz — it would be showing '
            'a trace the model never saw')
        self.assertIn('250 Hz', self.w.statusBar().currentMessage(),
                      'the reviewer was not told why the view turned itself off')

    def test_a_250hz_file_still_gets_the_detector_input_view(self):
        """The decline must be conditional, not a blanket disable."""
        self._none()
        p = _write_edf(os.path.join(self.tmp, 'native250.edf'), seconds=60,
                       fs=250)
        _write_cache(p, 60)
        self.w.load_edf(p)
        self.assertEqual(int(round(self.w._fs)), 250)
        self.w.chk_ica.setChecked(True)
        self.w._display_source()
        self.assertTrue(self.w.chk_ica.isChecked(),
                        'the ICA view was disabled on a 250 Hz recording, '
                        'where it is perfectly valid')

    def test_an_unreadable_oversized_file_is_still_refused(self):
        """No lazy path available means refusing is the honest answer."""
        self._none()
        import pyedflib
        p = os.path.join(self.tmp, 'nochannels.edf')
        # Same rate-key feature detection as everywhere else: the legacy 3.6
        # stack's pyedflib wants sample_rate, the modern one sample_frequency.
        def _attempt(rate_key):
            w = pyedflib.EdfWriter(p, 2, file_type=pyedflib.FILETYPE_EDFPLUS)
            try:
                w.setSignalHeaders([{
                    'label': lab, 'dimension': 'uV', rate_key: 250,
                    'physical_min': -3276.8, 'physical_max': 3276.7,
                    'digital_min': -32768, 'digital_max': 32767,
                    'prefilter': '', 'transducer': ''}
                    for lab in ('sine', 'ramp')])
                w.writeSamples([np.zeros(250 * 10), np.zeros(250 * 10)])
            finally:
                try:
                    w.close()
                except Exception:                            # noqa: BLE001
                    pass

        try:
            _attempt('sample_frequency')
        except (FutureWarning, KeyError, TypeError, ValueError):
            _attempt('sample_rate')
        self.w.load_edf(p)
        self.assertTrue(
            self.warnings,
            'a recording that can be neither held nor streamed must be '
            'refused with an explanation, not attempted')

    def test_an_unprepared_recording_opens_and_says_what_to_run(self):
        """No study bundle — open anyway, and name the server command.

        The application no longer scores anything itself, at any size: every
        model runs where the study was prepared. So the one thing it owes a
        reviewer who opens an unprepared recording is the command that
        prepares it. Saying only "no proposals" would leave them stuck.
        """
        self._none()
        p = _write_edf(os.path.join(self.tmp, 'unscored.edf'), seconds=60,
                       fs=256)
        self.w.load_edf(p)
        self.assertIsNotNone(
            self.w._raw_data,
            'an unprepared recording must still open for review; the '
            'reviewer may be looking for what the detector missed')
        msg = self.w.statusBar().currentMessage()
        self.assertIn('prepare_study.py', msg,
                      'the reviewer was not told how to get proposals')

    def test_the_review_client_never_runs_the_model_itself(self):
        """The two-part split has to be enforced, not merely intended.

        While the application could fall back to scoring, "heavy computation
        happens on the server" was a convention a missing cache could break —
        on a clinical workstation with no GPU and often no TensorFlow.
        """
        import gui.app as app_module
        self.assertFalse(
            hasattr(app_module.MainWindow, '_ensure_probs_with_progress'),
            'the in-app inference fallback is back; the review client must '
            'not be able to run the detector')


if __name__ == '__main__':
    unittest.main()
