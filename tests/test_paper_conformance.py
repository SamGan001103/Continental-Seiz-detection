"""Does the pipeline still match the published method?

Yang et al., *Continental generalization of an AI system for clinical seizure
recognition*, Expert Syst. Appl. 207:118083 (2022); preprint arXiv:2103.10900.
§2.3 is the entire preprocessing specification, quoted in
`docs/ica_implementation_review.md` §1 and re-verified against the arXiv source
on 2026-08-10.

Why this file exists
--------------------
The conformance argument was prose in a document. Prose does not fail when
somebody changes `framelength` or drops a channel — and the repository contains
no copy of the paper, so a reader cannot even check the prose against the
source. These tests make the paper's numeric claims executable.

They deliberately assert **the current behaviour, deviations included**. Four
places depart from the paper (`ica_implementation_review.md` §2b rows 5-8) and
they must NOT be "fixed": `utils/ICA_load_data_elec.py` imports the same
`ica_arti_remove`, so the released weights were fitted to the code as written,
not to the paper as worded. A test that failed on those deviations would be
inviting somebody to break the model in the name of fidelity. Instead each
deviation is pinned *as a deviation*, with the paper's wording in the assertion
message, so that changing it is a deliberate act rather than an accident.
"""
import inspect
import os
import re
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class SegmentationMatchesPaper(unittest.TestCase):
    """"We split EEG signals into 12-second segments"."""

    def test_segment_length_is_12_seconds(self):
        from gui.io.infer import SEGMENT_S
        self.assertEqual(SEGMENT_S, 12)

    def test_sampling_rate_is_250_hz(self):
        from gui.io.edf import TARGET_FS
        self.assertEqual(TARGET_FS, 250)

    def test_nineteen_electrodes(self):
        """"decompose the signal into 19 independent components"."""
        from gui.io.edf import CHANNELS_19
        self.assertEqual(len(CHANNELS_19), 19)

    def test_the_two_eog_reference_channels_are_present(self):
        """"detected from two EEG channels, namely 'FP1' and 'FP2'"."""
        from gui.io.edf import CHANNELS_19
        self.assertIn('Fp1', CHANNELS_19)
        self.assertIn('Fp2', CHANNELS_19)


class IcaMatchesPaper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils import preprocessing
        cls.src = inspect.getsource(preprocessing.ica_arti_remove)

    def test_nineteen_independent_components(self):
        self.assertIn('n_components=19', self.src)

    def test_eog_identification_uses_fp1_and_fp2(self):
        self.assertIn("ch_name='Fp1'", self.src)
        self.assertIn("ch_name='Fp2'", self.src)

    # -- the four deviations, pinned AS deviations ----------------------
    def test_DEVIATION_only_the_top_component_per_channel_is_removed(self):
        """Paper: "We remove **those** independent sources" — plural, all of them.

        The code appends only e1[0] and e2[0]. Measured over 98 windows: 212
        components flagged, 106 removed. Pinned because the training features
        were generated this way; see docs/ica_implementation_review.md 2b row 5.
        """
        self.assertIn('eog_indices1[0]', self.src)
        self.assertIn('eog_indices2[0]', self.src)

    def test_DEVIATION_a_high_pass_filter_the_paper_never_mentions(self):
        """The paper specifies no filter of any kind before ICA.

        MNE designs an 8251-sample (33 s) filter for a 3000-sample (12 s)
        window and warns that distortion is likely. Pinned, not fixed.
        """
        self.assertIn('l_freq=0.1', self.src)

    def test_DEVIATION_eog_threshold_is_unspecified_and_non_default(self):
        """The paper gives no threshold. MNE's default is 3.0; this uses 2.0."""
        self.assertIn('threshold=2.0', self.src)

    def test_DEVIATION_no_eog_found_returns_the_raw_unfiltered_input(self):
        """~10 % of windows reach the STFT preprocessed unlike their neighbours.

        The paper is silent on this case. The final `return data` hands back the
        untouched input rather than the filtered signal.
        """
        self.assertTrue(self.src.rstrip().endswith('return data'),
                        'the raw pass-through branch changed shape; it is '
                        'load-bearing for the trained operating point')


class StftMatchesPaper(unittest.TestCase):
    """"STFT ... window length of 250 (or 1 second) and 50% overlapping, then
    remove the DC component ... shape will become (n x 23 x 125)"."""

    @classmethod
    def setUpClass(cls):
        from gui.io.infer import _calc_stft, SEGMENT_S
        from gui.io.edf import CHANNELS_19, TARGET_FS
        sig = np.random.RandomState(0).randn(
            len(CHANNELS_19), SEGMENT_S * TARGET_FS).astype(np.float32)
        cls.out = _calc_stft(sig)

    def test_window_length_250(self):
        from utils import preprocessing  # noqa: F401  (import side-effect free)
        from gui.io import infer
        self.assertIn('framelength=250', inspect.getsource(infer._calc_stft))

    def test_twenty_three_time_frames_implies_fifty_percent_overlap(self):
        """23 frames from 3000 samples at length 250 is only possible at hop 125."""
        self.assertEqual(self.out.shape[1], 23)
        n, fl = 3000, 250
        self.assertEqual((n - fl) // 125 + 1, 23)
        self.assertNotEqual((n - fl) // 250 + 1, 23)   # 0 % overlap would give 12

    def test_dc_component_removed_leaving_125_bins(self):
        self.assertEqual(self.out.shape[3], 125)
        self.assertIn('[:, :, 1:]', inspect.getsource(
            __import__('gui.io.infer', fromlist=['_calc_stft'])._calc_stft))

    def test_electrode_axis_is_nineteen(self):
        from gui.io.edf import CHANNELS_19
        self.assertEqual(self.out.shape[2], len(CHANNELS_19))

    def test_model_input_shape_matches_the_tensor(self):
        """(-1, 2*12-1, 19, 125, 1) == (-1, 23, 19, 125, 1)."""
        from gui.io import infer
        src = inspect.getsource(infer._build_model)
        self.assertIn('2 * SEGMENT_S - 1', src)
        self.assertIn('125', src)


class DocumentedEnvironmentDeviation(unittest.TestCase):
    """The paper names MNE v0.20; this project pins 0.19.2.

    Not something to silently "fix" — changing MNE changes ICA numerics, and
    the weights were fitted to whatever the authors ran. The requirement is
    that the mismatch stays *disclosed*, so it cannot quietly stop being
    mentioned.
    """

    def test_python_version_matches_the_paper(self):
        self.assertTrue(sys.version.startswith('3.6'))

    def test_mne_version_deviation_is_documented(self):
        import mne
        review = os.path.join(REPO, 'docs', 'ica_implementation_review.md')
        with open(review, encoding='utf-8') as f:
            text = f.read()
        if mne.__version__.startswith('0.20'):
            return          # no deviation to document
        self.assertIn('0.19.2', text,
                      'MNE differs from the paper v0.20 but the review no '
                      'longer records which version is installed')
        self.assertIn('v0.20', text,
                      'the review no longer states what the paper specifies')

    def test_the_paper_specification_is_quoted_in_the_repo(self):
        """The repo has no copy of the paper; the quote is the only record."""
        review = os.path.join(REPO, 'docs', 'ica_implementation_review.md')
        with open(review, encoding='utf-8') as f:
            text = f.read()
        for phrase in ('12-second segments',
                       '19 independent components',
                       'We remove those independent sources',
                       'MNE v0.20'):
            self.assertIn(phrase, text,
                          'lost the paper wording for: {}'.format(phrase))


class TrainingUsedTheSameFunction(unittest.TestCase):
    """The whole "do not fix ICA" argument depends on this being true."""

    def test_the_training_loader_imports_the_deployed_ica(self):
        p = os.path.join(REPO, 'utils', 'ICA_load_data_elec.py')
        with open(p, encoding='utf-8', errors='replace') as f:
            text = f.read()
        self.assertTrue(
            re.search(r'from\s+preprocessing\s+import[^\n]*ica_arti_remove',
                      text),
            'the training loader no longer imports the deployed '
            'ica_arti_remove — the "baked into the weights" argument in '
            'RESULTS.md 4 would no longer hold')

    def test_training_and_inference_both_skip_on_ica_failure(self):
        p = os.path.join(REPO, 'utils', 'ICA_load_data_elec.py')
        with open(p, encoding='utf-8', errors='replace') as f:
            text = f.read()
        self.assertIn('ica_filt_s is None', text)


if __name__ == '__main__':
    unittest.main()
