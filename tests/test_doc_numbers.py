"""Every number in the shipped documents must agree with RESULTS.md.

`docs/RESULTS.md` is this project's single source of truth. Two documents are
bundled into the application and read by people who will never open the
repository — `INTENDED_USE.md` by the clinician and any ethics reviewer,
`DEPLOYMENT.md` by whoever installs it. A figure that drifts out of step there
is worse than one that drifts in a working note, because the reader has no way
to notice.

This suite exists because two real errors got as far as a committed document,
and both had the same cause: **stating a proportion without naming its
denominator**, in a project where "event level" and "window level" are
routinely different objects.

  - "22 % of *missed* seizures produce no model response" — it is 22 % of *all*
    85 reference seizures.
  - the replacement, "over half of everything the detector misses" — that used
    the 36 seizures whose peak *window* score is below 0.5, while the
    surrounding text had just established an *event*-level denominator of 43.

A grep-based check cannot prove a document is correct. What it can do is fail
the moment a headline figure is changed in one place and not the other, and
force the derived percentages to stay arithmetically consistent with the counts
they are derived from.
"""
import io
import os
import re
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCS = os.path.join(REPO, 'docs')


def read(*parts):
    with io.open(os.path.join(*parts), encoding='utf-8') as f:
        return f.read()


class NumbersAgreeWithResults(unittest.TestCase):
    """Headline figures must appear in RESULTS.md as well as in the doc."""

    @classmethod
    def setUpClass(cls):
        cls.results = read(DOCS, 'RESULTS.md')
        cls.intended = read(DOCS, 'INTENDED_USE.md')
        cls.deployment = read(DOCS, 'DEPLOYMENT.md')

    def _both(self, label, doc, needle, results_needle=None):
        results_needle = results_needle or needle
        self.assertIn(needle, doc, '{}: missing from the shipped document'
                      .format(label))
        self.assertIn(results_needle, self.results,
                      '{}: not backed by RESULTS.md'.format(label))

    # -- discrimination -------------------------------------------------
    def test_window_auc(self):
        self._both('window AUC', self.intended, '0.89')

    def test_patient_level_interval_is_the_one_quoted(self):
        """The by-patient interval is wider; quoting the by-file one flatters."""
        self._both('patient CI', self.intended, '[0.73, 0.95]')

    def test_published_auc(self):
        self._both('published AUC', self.intended, '0.84')

    # -- event level ----------------------------------------------------
    def test_event_sensitivity(self):
        self._both('event sensitivity', self.intended, '48.2')

    def test_false_alarm_rate(self):
        self._both('false alarms', self.intended, '204.4')

    def test_detected_count(self):
        self.assertIn('41 of 85', self.intended)
        self.assertIn('41/85', self.results)

    # -- corpus ---------------------------------------------------------
    def test_corpus_description(self):
        for needle in ('206', '27.8', '28 patients'):
            self._both('corpus ' + needle, self.intended, needle)

    # -- calibration ----------------------------------------------------
    def test_calibration_error(self):
        self._both('raw ECE', self.intended, '0.072')
        self._both('Platt ECE', self.intended, '0.011')

    def test_score_of_half_is_not_half_a_probability(self):
        """The single most misreadable number in the interface."""
        self.assertIn('29 %', self.intended)
        self.assertIn('0.287', self.results)

    # -- runtime --------------------------------------------------------
    def test_runtime_factor(self):
        self._both('runtime', self.deployment, '0.043')


class DerivedProportionsAreConsistent(unittest.TestCase):
    """Percentages must follow from the counts stated alongside them.

    Both documented errors were of this kind: the count was right and the
    percentage derived from it was not.
    """

    TOTAL_SEIZURES = 85
    DETECTED_AT_DEFAULT = 41
    NO_MODEL_RESPONSE = 16        # peak window score < 0.01

    @classmethod
    def setUpClass(cls):
        cls.intended = read(DOCS, 'INTENDED_USE.md')

    def test_share_of_all_seizures(self):
        pct = 100.0 * self.NO_MODEL_RESPONSE / self.TOTAL_SEIZURES
        self.assertAlmostEqual(pct, 18.8, places=1)
        self.assertIn('19 %', self.intended)

    def test_share_of_missed_seizures_uses_the_event_level_denominator(self):
        missed = self.TOTAL_SEIZURES - self.DETECTED_AT_DEFAULT
        self.assertEqual(missed, 44)
        pct = 100.0 * self.NO_MODEL_RESPONSE / missed
        self.assertAlmostEqual(pct, 36.4, places=1)
        self.assertIn('36 %', self.intended)

    def test_the_superseded_claim_is_gone(self):
        """'over half' was computed against a window-level denominator."""
        self.assertNotIn('over half', self.intended)

    def test_the_document_names_the_denominator(self):
        """The fix is not the number, it is saying what it is a share of."""
        self.assertIn('44 seizures', self.intended)


class ShippedDocumentsExist(unittest.TestCase):
    """The spec bundles these; a rename would silently ship nothing."""

    def test_documents_the_spec_bundles_are_present(self):
        spec = read(REPO, 'packaging', 'SeizureReview.spec')
        named = re.findall(r"'(\w+\.md)'", spec)
        self.assertTrue(named, 'spec no longer names any documents')
        for name in set(named):
            found = (os.path.exists(os.path.join(DOCS, name))
                     or os.path.exists(os.path.join(REPO, name)))
            self.assertTrue(found, '{} is bundled but does not exist'
                            .format(name))

    def test_intended_use_states_it_is_not_a_medical_device(self):
        text = read(DOCS, 'INTENDED_USE.md').lower()
        self.assertIn('not a medical device', text)
        self.assertIn('must not be used unsupervised', text)


if __name__ == '__main__':
    unittest.main()
