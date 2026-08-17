"""The bipolar montage tables must actually do what their labels say.

`LONGITUDINAL_BIPOLAR` and `TRANSVERSE_BIPOLAR` in gui/processing.py are
hand-typed 18-row tables of (label, channel_a, channel_b), and the label is the
only thing the clinician ever sees. A transposed pair - ('Fp1-F7', 'F7', 'Fp1')
- or a row that names T4 where it means T3 puts a left-temporal discharge on a
trace captioned right-temporal, and nothing downstream would notice: the trace
still looks like EEG, the detector never sees the montage at all (it scores the
referential array on the frozen model path), and the lateralisation call is made
from the caption. In a tool built so a clinician can localise a finding, that is
the worst failure available, and until now the tables had no coverage.

So these tests take the tables' word for nothing. Labels are re-parsed and
checked against the arithmetic they claim; anatomy is checked against a 10-20
grid hardcoded below, deliberately re-typed here rather than imported, so that a
single typo cannot satisfy both the table and its test.

Read-only by construction - gui/processing.py is not this file's to change. If
an assertion here fires, the table is wrong, not the test.
"""
import os
import re
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.processing import (CH_IDX, CH_NAMES_19, LONGITUDINAL_BIPOLAR,
                            MONTAGES, TRANSVERSE_BIPOLAR, apply_montage)

# The standard 10-20 layout as a coarse anatomical grid. Written out here
# independently of gui/processing.py on purpose: this is the test's own source
# of truth, and if it agreed with the table by construction it would prove
# nothing.
#   ap  : anterior -> posterior level; 0 = Fp row, 1 = F, 2 = C, 3 = P, 4 = O.
#         T3/T4 sit at the central level and T5/T6 at the parietal level, which
#         is what makes the temporal chain a chain.
#   lat : -2 far left, -1 left, 0 midline, +1 right, +2 far right.
TEN_TWENTY = {
    'Fp1': (0, -1), 'Fp2': (0, 1),
    'F7': (1, -2), 'F3': (1, -1), 'Fz': (1, 0), 'F4': (1, 1), 'F8': (1, 2),
    'T3': (2, -2), 'C3': (2, -1), 'Cz': (2, 0), 'C4': (2, 1), 'T4': (2, 2),
    'T5': (3, -2), 'P3': (3, -1), 'Pz': (3, 0), 'P4': (3, 1), 'T6': (3, 2),
    'O1': (4, -1), 'O2': (4, 1),
}

# Left/right counterparts, derived from the grid rather than typed a second
# time: the mirror of an electrode is the one at the same anterior-posterior
# level with the opposite lateral offset. Midline electrodes mirror onto
# themselves.
MIRROR = {}
for _name, _pos in TEN_TWENTY.items():
    for _other, _other_pos in TEN_TWENTY.items():
        if _other_pos == (_pos[0], -_pos[1]):
            MIRROR[_name] = _other

# The double banana and the standard transverse rows, as a clinician would
# recite them. Held separately from the tables so that "the chains link end to
# end" is checked against the montage a reader expects, not against whatever
# the table happens to contain.
EXPECTED_LONGITUDINAL_CHAINS = [
    ['Fp1', 'F7', 'T3', 'T5', 'O1'],        # left temporal
    ['Fp2', 'F8', 'T4', 'T6', 'O2'],        # right temporal
    ['Fp1', 'F3', 'C3', 'P3', 'O1'],        # left parasagittal
    ['Fp2', 'F4', 'C4', 'P4', 'O2'],        # right parasagittal
    ['Fz', 'Cz', 'Pz'],                     # midline
]

EXPECTED_TRANSVERSE_CHAINS = [
    ['F7', 'Fp1', 'Fp2', 'F8'],             # frontopolar, anchored on F7/F8
    ['F7', 'F3', 'Fz', 'F4', 'F8'],         # frontal
    ['T3', 'C3', 'Cz', 'C4', 'T4'],         # central
    ['T5', 'P3', 'Pz', 'P4', 'T6'],         # parietal
    ['T5', 'O1', 'O2', 'T6'],               # occipital, anchored on T5/T6
]

TABLES = [('LONGITUDINAL_BIPOLAR', LONGITUDINAL_BIPOLAR),
          ('TRANSVERSE_BIPOLAR', TRANSVERSE_BIPOLAR)]


def parse_label(label):
    """Split 'Fp1-F7' into ('Fp1', 'F7').

    The label is what the clinician reads, so the tests derive the expected
    arithmetic from it and never from the tuple that is supposed to implement
    it. Raises on anything that is not exactly two names.
    """
    parts = label.split('-')
    if len(parts) != 2:
        raise ValueError('label {!r} is not a two-electrode derivation'
                         .format(label))
    return parts[0], parts[1]


def chains_of(table):
    """Group consecutive rows into chains, linking row b -> next row a.

    A bipolar chain is exactly this: the second electrode of one derivation is
    the first electrode of the next. Rebuilding the chains from the table is
    how a broken link (F7-T3 followed by T5-O1, with T3-T5 dropped or mistyped)
    becomes visible.
    """
    chains = []
    for _label, a, b in table:
        if chains and chains[-1][-1] == a:
            chains[-1].append(b)
        else:
            chains.append([a, b])
    return chains


class TableShapeTests(unittest.TestCase):
    """The cheap structural facts, checked first so failures elsewhere are
    about anatomy rather than about a malformed row."""

    def test_each_table_has_eighteen_rows(self):
        # 19 referential electrodes yield 18 bipolar derivations in both the
        # standard longitudinal and the standard transverse montage. A 17- or
        # 19-row table means a row was lost or duplicated in hand-typing.
        for name, table in TABLES:
            self.assertEqual(18, len(table),
                             '{} has {} rows, expected 18'
                             .format(name, len(table)))

    def test_every_row_is_a_label_and_two_channels(self):
        for name, table in TABLES:
            for row in table:
                self.assertEqual(
                    3, len(row),
                    '{}: row {!r} is not (label, channel_a, channel_b)'
                    .format(name, row))

    def test_labels_are_unique(self):
        # A duplicated label puts two different derivations under one caption;
        # the clinician has no way to tell which trace is which.
        for name, table in TABLES:
            labels = [label for label, _a, _b in table]
            dupes = sorted(set(l for l in labels if labels.count(l) > 1))
            self.assertEqual([], dupes,
                             '{} has duplicate labels: {}'.format(name, dupes))

    def test_derivations_are_unique(self):
        for name, table in TABLES:
            pairs = [(a, b) for _label, a, b in table]
            dupes = sorted(set(p for p in pairs if pairs.count(p) > 1))
            self.assertEqual([], dupes,
                             '{} derives the same pair twice: {}'
                             .format(name, dupes))

    def test_row_count_matches_label_count(self):
        for name, table in TABLES:
            self.assertEqual(len(table), len(set(l for l, _a, _b in table)),
                             '{}: row count and distinct label count differ'
                             .format(name))

    def test_no_row_subtracts_a_channel_from_itself(self):
        # Would render a permanently flat trace, which reads as a dead
        # electrode rather than as a table bug.
        for name, table in TABLES:
            for label, a, b in table:
                self.assertNotEqual(a, b,
                                    '{}: {} subtracts {} from itself'
                                    .format(name, label, a))

    def test_electrode_names_never_contain_the_separator(self):
        # parse_label splits on '-'; if an electrode name ever contained one
        # the parse would silently mis-split and every label test below would
        # be checking the wrong thing.
        for name in CH_NAMES_19:
            self.assertNotIn('-', name)

    def test_the_test_grid_covers_exactly_the_canonical_nineteen(self):
        # Guards the tests themselves: if CH_NAMES_19 ever changes, the
        # anatomy assertions must be revisited rather than quietly skipped.
        self.assertEqual(sorted(CH_NAMES_19), sorted(TEN_TWENTY.keys()))
        self.assertEqual(19, len(CH_NAMES_19))
        self.assertEqual(19, len(set(CH_NAMES_19)))


class ChannelsExistTests(unittest.TestCase):
    def test_every_named_channel_is_in_the_nineteen_channel_montage(self):
        # A name that is not in CH_IDX raises KeyError inside apply_montage at
        # the moment the clinician switches montage - i.e. mid-review, with a
        # recording open.
        for name, table in TABLES:
            for label, a, b in table:
                for ch in (a, b):
                    self.assertIn(ch, CH_IDX,
                                  '{}: row {} names {!r}, which is not one of '
                                  'the 19 channels'.format(name, label, ch))

    def test_every_electrode_named_in_a_label_exists_too(self):
        for name, table in TABLES:
            for label, _a, _b in table:
                for ch in parse_label(label):
                    self.assertIn(ch, CH_IDX,
                                  '{}: label {!r} names {!r}, which is not one '
                                  'of the 19 channels'.format(name, label, ch))

    def test_every_electrode_is_displayed_by_every_montage(self):
        # An electrode present in the recording but absent from the montage is
        # invisible to the reviewer, and invisibly so. Both standard montages
        # use all 19.
        for name, table in TABLES:
            used = set()
            for _label, a, b in table:
                used.add(a)
                used.add(b)
            missing = sorted(set(CH_NAMES_19) - used)
            self.assertEqual([], missing,
                             '{} never displays {}'.format(name, missing))


class LabelMatchesDerivationTests(unittest.TestCase):
    """The transposition check. The label is parsed, not trusted."""

    def test_label_names_the_channels_in_the_order_they_are_subtracted(self):
        for name, table in TABLES:
            for label, a, b in table:
                want_a, want_b = parse_label(label)
                self.assertEqual(
                    (want_a, want_b), (a, b),
                    '{}: row labelled {!r} actually computes {} - {}. A '
                    'transposed pair inverts the trace and, where the two '
                    'electrodes are on opposite sides, mislateralises the '
                    'finding.'.format(name, label, a, b))

    def test_labels_are_written_in_the_canonical_capitalisation(self):
        # 'FP1-F7' or 'fp1-f7' would still parse but would not match the
        # channel names, so the arithmetic tests below could not check it.
        for name, table in TABLES:
            for label, _a, _b in table:
                self.assertTrue(
                    re.match(r'^[A-Za-z]+z?\d*-[A-Za-z]+z?\d*$', label),
                    '{}: label {!r} is not of the form Name-Name'
                    .format(name, label))
                for ch in parse_label(label):
                    self.assertIn(ch, CH_NAMES_19)


class LongitudinalAnatomyTests(unittest.TestCase):
    """The double banana, checked link by link."""

    def test_chains_link_end_to_end(self):
        self.assertEqual(EXPECTED_LONGITUDINAL_CHAINS,
                         chains_of(LONGITUDINAL_BIPOLAR),
                         'the longitudinal chains are not the standard double '
                         'banana; a link is broken, reordered or mistyped')

    def test_every_row_runs_anterior_to_posterior(self):
        # Longitudinal bipolar convention: the anterior electrode is the one
        # subtracted from. Reverse a row and its deflections invert, which a
        # reader interprets as a phase reversal - the exact cue used to
        # localise a focus.
        for label, a, b in LONGITUDINAL_BIPOLAR:
            ap_a = TEN_TWENTY[a][0]
            ap_b = TEN_TWENTY[b][0]
            self.assertEqual(
                1, ap_b - ap_a,
                '{}: {} (level {}) to {} (level {}) is not a single step '
                'front to back'.format(label, a, ap_a, b, ap_b))

    def test_every_row_stays_on_one_side_of_the_head(self):
        # No longitudinal derivation may straddle the midline: subtracting a
        # right electrode from a left one produces a trace that belongs to
        # neither hemisphere.
        for label, a, b in LONGITUDINAL_BIPOLAR:
            side_a = np.sign(TEN_TWENTY[a][1])
            side_b = np.sign(TEN_TWENTY[b][1])
            self.assertEqual(side_a, side_b,
                             '{} crosses the midline: {} and {} are on '
                             'opposite sides'.format(label, a, b))

    def test_left_and_right_chains_mirror_each_other(self):
        rows = set((a, b) for _label, a, b in LONGITUDINAL_BIPOLAR)
        for label, a, b in LONGITUDINAL_BIPOLAR:
            mirrored = (MIRROR[a], MIRROR[b])
            self.assertIn(
                mirrored, rows,
                '{} has no counterpart {}-{}. An asymmetric montage means the '
                'two hemispheres are not comparable trace for trace, which is '
                'how lateralisation is read.'
                .format(label, mirrored[0], mirrored[1]))

    def test_the_mirror_of_every_chain_is_also_a_chain(self):
        chains = chains_of(LONGITUDINAL_BIPOLAR)
        as_tuples = set(tuple(c) for c in chains)
        for chain in chains:
            mirrored = tuple(MIRROR[ch] for ch in chain)
            self.assertIn(mirrored, as_tuples,
                          'chain {} has no mirror image in the montage'
                          .format(chain))

    def test_the_midline_chain_is_present_and_is_the_only_midline_one(self):
        midline = [c for c in chains_of(LONGITUDINAL_BIPOLAR)
                   if all(TEN_TWENTY[ch][1] == 0 for ch in c)]
        self.assertEqual([['Fz', 'Cz', 'Pz']], midline)


class TransverseAnatomyTests(unittest.TestCase):
    """Transverse rows sweep across the head at one level, left to right."""

    def test_chains_link_end_to_end(self):
        self.assertEqual(EXPECTED_TRANSVERSE_CHAINS,
                         chains_of(TRANSVERSE_BIPOLAR),
                         'the transverse chains are not the standard rows; a '
                         'link is broken, reordered or mistyped')

    def test_every_row_runs_left_to_right(self):
        # Transverse convention: the more leftward electrode is subtracted
        # from. A row typed backwards inverts that trace relative to its
        # neighbours in the same sweep, which fakes a phase reversal at the
        # midline.
        for label, a, b in TRANSVERSE_BIPOLAR:
            lat_a = TEN_TWENTY[a][1]
            lat_b = TEN_TWENTY[b][1]
            self.assertLess(lat_a, lat_b,
                            '{}: {} (lateral {}) is not left of {} (lateral '
                            '{})'.format(label, a, lat_a, b, lat_b))

    def test_each_sweep_holds_one_anterior_posterior_level(self):
        """Interior electrodes of a sweep share one level.

        The two outermost electrodes are allowed to sit one level away,
        because the standard frontopolar and occipital sweeps have no
        electrode of their own at the lateral extreme and anchor on F7/F8 and
        T5/T6 respectively. Anything else - a P-level electrode spliced into
        the central sweep - is a typo.
        """
        for chain in chains_of(TRANSVERSE_BIPOLAR):
            interior = chain[1:-1]
            levels = set(TEN_TWENTY[ch][0] for ch in interior)
            self.assertEqual(1, len(levels),
                             'sweep {} spans several levels internally: {}'
                             .format(chain, sorted(levels)))
            level = levels.pop()
            for ch in (chain[0], chain[-1]):
                ap, lat = TEN_TWENTY[ch]
                if ap != level:
                    self.assertEqual(
                        1, abs(ap - level),
                        'sweep {} anchors on {}, {} levels away'
                        .format(chain, ch, abs(ap - level)))
                    self.assertEqual(
                        2, abs(lat),
                        'sweep {} anchors on {}, which is not at the lateral '
                        'extreme'.format(chain, ch))

    def test_each_sweep_starts_on_the_left_and_ends_on_the_right(self):
        for chain in chains_of(TRANSVERSE_BIPOLAR):
            self.assertLess(TEN_TWENTY[chain[0]][1], 0,
                            'sweep {} does not start on the left'
                            .format(chain))
            self.assertGreater(TEN_TWENTY[chain[-1]][1], 0,
                               'sweep {} does not end on the right'
                               .format(chain))

    def test_each_sweep_crosses_the_midline_exactly_once(self):
        """A sweep goes left side -> right side once and does not come back.

        Midline electrodes are dropped before counting, because a sweep that
        passes through Fz sits at lateral 0 for one step; that is a stop on
        the midline, not a second crossing. What is being ruled out is a sweep
        that wanders back to the side it came from, which is what a
        transposed or misplaced row looks like.
        """
        for chain in chains_of(TRANSVERSE_BIPOLAR):
            sides = [TEN_TWENTY[ch][1] for ch in chain
                     if TEN_TWENTY[ch][1] != 0]
            crossings = sum(1 for i in range(len(sides) - 1)
                            if np.sign(sides[i]) != np.sign(sides[i + 1]))
            self.assertEqual(1, crossings,
                             'sweep {} changes side {} times'
                             .format(chain, crossings))

    def test_no_sweep_visits_the_midline_more_than_once(self):
        # Two midline electrodes in one sweep would mean two different levels
        # were spliced together - Fz and Cz in the same row, say.
        for chain in chains_of(TRANSVERSE_BIPOLAR):
            midline = [ch for ch in chain if TEN_TWENTY[ch][1] == 0]
            self.assertGreaterEqual(1, len(midline),
                                    'sweep {} touches the midline at {}'
                                    .format(chain, midline))

    def test_sweeps_are_left_right_symmetric(self):
        # Mirroring a transverse row also reverses it: the mirror of A-B is
        # mirror(B)-mirror(A), because the sweep runs the other way.
        rows = set((a, b) for _label, a, b in TRANSVERSE_BIPOLAR)
        for label, a, b in TRANSVERSE_BIPOLAR:
            mirrored = (MIRROR[b], MIRROR[a])
            self.assertIn(mirrored, rows,
                          '{} has no counterpart {}-{}'
                          .format(label, mirrored[0], mirrored[1]))


class ApplyMontageArithmeticTests(unittest.TestCase):
    """The tables are only half the story - apply_montage has to honour them.

    These run the real function on real numbers and recompute every row from
    the label, so a correct table wired to the wrong index would still fail.
    """

    def setUp(self):
        rng = np.random.RandomState(11)
        fs = 250
        t = np.arange(10 * fs) / float(fs)
        # Distinct amplitude, frequency and offset per channel so that no two
        # rows can coincide by accident and a swapped index shows up.
        self.data = np.stack([
            ((20.0 + 3.0 * i) * np.sin(2 * np.pi * (4.0 + 0.7 * i) * t)
             + 5.0 * rng.randn(t.size) + float(i))
            for i in range(19)]).astype(np.float32)

    def _check(self, montage_name, table):
        out, labels = apply_montage(self.data, montage_name)
        self.assertEqual([l for l, _a, _b in table], labels,
                         '{}: apply_montage returned different labels from '
                         'the table'.format(montage_name))
        self.assertEqual(len(labels), out.shape[0],
                         '{}: {} rows returned for {} labels'
                         .format(montage_name, out.shape[0], len(labels)))
        self.assertEqual(self.data.shape[1], out.shape[1])
        for i, label in enumerate(labels):
            a, b = parse_label(label)
            expected = self.data[CH_IDX[a]] - self.data[CH_IDX[b]]
            np.testing.assert_array_equal(
                out[i], expected,
                'row {} labelled {!r} is not {} minus {}'
                .format(i, label, a, b))

    def test_longitudinal_rows_are_the_arithmetic_the_labels_claim(self):
        self._check('Longitudinal Bipolar', LONGITUDINAL_BIPOLAR)

    def test_transverse_rows_are_the_arithmetic_the_labels_claim(self):
        self._check('Transverse Bipolar', TRANSVERSE_BIPOLAR)

    def test_polarity_is_not_merely_magnitude(self):
        """assert_array_equal above would still pass on |a - b|; this will not.

        Each channel is held at a constant equal to its index, so every row
        must equal index(a) - index(b) exactly, sign included. A transposed
        pair flips the sign of a nonzero constant.
        """
        flat = np.tile(np.arange(19, dtype=np.float32).reshape(19, 1), (1, 64))
        for montage_name, table in [('Longitudinal Bipolar',
                                     LONGITUDINAL_BIPOLAR),
                                    ('Transverse Bipolar',
                                     TRANSVERSE_BIPOLAR)]:
            out, labels = apply_montage(flat, montage_name)
            for i, label in enumerate(labels):
                a, b = parse_label(label)
                want = float(CH_IDX[a] - CH_IDX[b])
                self.assertNotEqual(0.0, want,
                                    '{}: {} cannot test polarity'
                                    .format(montage_name, label))
                np.testing.assert_allclose(
                    out[i], want, rtol=0, atol=1e-5,
                    err_msg='{}: row {!r} came out as {} but {} - {} is {}'
                            .format(montage_name, label, out[i][0], a, b,
                                    want))

    def test_a_left_sided_discharge_only_lights_left_sided_traces(self):
        """The failure this whole file exists to prevent.

        Put activity on T3 (left mid-temporal) and nothing anywhere else. Only
        the traces whose labels name T3 may move, and no label that moves may
        name a right-hemisphere electrode.
        """
        fs = 250
        t = np.arange(8 * fs) / float(fs)
        burst = (120.0 * np.sin(2 * np.pi * 6.0 * t)).astype(np.float32)
        data = np.zeros((19, t.size), dtype=np.float32)
        data[CH_IDX['T3']] = burst

        for montage_name in ('Longitudinal Bipolar', 'Transverse Bipolar'):
            out, labels = apply_montage(data, montage_name)
            active = [labels[i] for i in range(len(labels))
                      if np.any(out[i] != 0.0)]
            expected = [l for l in labels if 'T3' in parse_label(l)]
            self.assertEqual(
                expected, active,
                '{}: a discharge on T3 appears on {} but should appear on {}'
                .format(montage_name, active, expected))
            for label in active:
                for ch in parse_label(label):
                    self.assertLessEqual(
                        TEN_TWENTY[ch][1], 0,
                        '{}: left-sided activity is displayed on {}, which '
                        'names the right-hemisphere electrode {}'
                        .format(montage_name, label, ch))

    def test_the_referential_and_average_montages_keep_all_nineteen(self):
        # Not table-driven, but the same contract: the labels the widget shows
        # must match the rows it is handed.
        for montage_name in ('Referential (raw)', 'Common Average'):
            out, labels = apply_montage(self.data, montage_name)
            self.assertEqual(list(CH_NAMES_19), labels)
            self.assertEqual(len(labels), out.shape[0])

    def test_every_registered_montage_returns_matching_rows_and_labels(self):
        # MONTAGES is what populates the combo box, so anything in it must be
        # dispatchable. A key added without a branch in apply_montage would
        # raise on selection.
        for montage_name in MONTAGES:
            out, labels = apply_montage(self.data, montage_name)
            self.assertEqual(len(labels), out.shape[0],
                             '{}: {} rows for {} labels'
                             .format(montage_name, out.shape[0], len(labels)))
            self.assertEqual(len(labels), len(set(labels)),
                             '{}: duplicate labels returned'
                             .format(montage_name))
            self.assertEqual(self.data.shape[1], out.shape[1])


if __name__ == '__main__':
    unittest.main()
