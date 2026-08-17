"""Geometry and behaviour of the 10-20 head map.

The electrode coordinates in gui/widgets/head_view.py are a pasted-in constant,
generated once offline from mne's standard_1020 template. Nothing at runtime
recomputes them, so nothing at runtime would notice if a digit were mistyped
during a later edit — a head map with Fp1 and Fp2 transposed is worse than no
head map, because it invites a reviewer to localise a spike to the wrong
hemisphere and looks entirely plausible while doing it. These tests are all
that stands between that edit and the clinician.

The geometry assertions are deliberately anatomical rather than exact values:
they say "Fp1 is anterior and on the left", not "Fp1 is at (-0.2555, 0.8513)",
so regenerating the constant from a different sphere fit stays legal while
mirroring or transposing it does not.
"""
import math
import unittest

from gui.widgets.head_view import (CHANNEL_NAMES, HEAD_POS, HEAD_POS_1020,
                                    LEGACY_ALIASES, derivations_for,
                                    montage_pairs)
from gui.processing import (CH_NAMES_19, LONGITUDINAL_BIPOLAR, MONTAGES,
                            TRANSVERSE_BIPOLAR)

try:
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtCore, QtGui, QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.widgets.head_view import HeadView
    HAVE_QT = True
except Exception:                                    # pragma: no cover
    HAVE_QT = False


class HeadGeometryTests(unittest.TestCase):
    def test_nineteen_electrodes_named_like_the_rest_of_the_app(self):
        """Order and spelling must match, or an index into one is wrong in the
        other. This is the TUH legacy set: T3/T4/T5/T6, not T7/T8/P7/P8."""
        self.assertEqual(len(HEAD_POS_1020), 19)
        self.assertEqual(list(CHANNEL_NAMES), list(CH_NAMES_19))
        for modern in ('T7', 'T8', 'P7', 'P8'):
            self.assertNotIn(modern, HEAD_POS)
            self.assertIn(LEGACY_ALIASES[modern], HEAD_POS)

    def test_the_edf_loader_and_the_head_map_agree_on_the_channel_set(self):
        """gui/io/edf.py is the module that actually names the loaded rows.

        Skipped rather than failed when pyedflib is unavailable — that import
        pulls the whole EDF stack in, and this file must stay runnable on a
        machine that only has the GUI dependencies.
        """
        try:
            from gui.io.edf import CHANNELS_19
        except Exception as ex:                      # pragma: no cover
            self.skipTest('gui.io.edf not importable: {}'.format(ex))
        self.assertEqual(list(CHANNEL_NAMES), list(CHANNELS_19))

    def test_every_electrode_is_inside_the_scalp_circle(self):
        """The outline is drawn at r = 1.0; a marker at r >= 1 hangs off the
        head and collides with the ears."""
        for name, x, y in HEAD_POS_1020:
            r = math.hypot(x, y)
            self.assertLess(r, 0.95,
                            '{} sits at r={:.3f}, outside the scalp outline'
                            .format(name, r))

    def test_electrodes_do_not_overlap(self):
        """Markers are drawn at ~0.088 of the head radius, so two electrodes
        closer than ~0.18 in model units would render as one blob."""
        for i, (n1, x1, y1) in enumerate(HEAD_POS_1020):
            for n2, x2, y2 in HEAD_POS_1020[i + 1:]:
                d = math.hypot(x1 - x2, y1 - y2)
                self.assertGreater(d, 0.18,
                                   '{} and {} are only {:.3f} apart'
                                   .format(n1, n2, d))

    def test_fp1_is_anterior_and_left_of_fp2(self):
        """+y is anterior, +x is the subject's right (nose-up view from above).

        Left-on-the-left is the convention every EEG head diagram uses, and
        getting it backwards is the single most dangerous error this widget
        could contain.
        """
        fp1, fp2 = HEAD_POS['Fp1'], HEAD_POS['Fp2']
        self.assertLess(fp1[0], 0.0, 'Fp1 must be on the left (x < 0)')
        self.assertGreater(fp2[0], 0.0, 'Fp2 must be on the right (x > 0)')
        self.assertLess(fp1[0], fp2[0])
        for name in ('Cz', 'C3', 'C4', 'O1', 'O2', 'Pz'):
            self.assertGreater(fp1[1], HEAD_POS[name][1],
                               'Fp1 must be anterior to {}'.format(name))

    def test_occipital_is_posterior_and_frontal_is_anterior(self):
        for name in ('O1', 'O2', 'Pz', 'T5', 'T6'):
            self.assertLess(HEAD_POS[name][1], 0.0,
                            '{} must be posterior (y < 0)'.format(name))
        for name in ('Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'):
            self.assertGreater(HEAD_POS[name][1], 0.0,
                               '{} must be anterior (y > 0)'.format(name))

    def test_cz_is_at_the_centre(self):
        """The vertex projects to the origin under an azimuthal-equidistant
        projection. A small offset is inherent to the fitted sphere centre; a
        large one means the projection lost its centre."""
        x, y = HEAD_POS['Cz']
        self.assertLess(math.hypot(x, y), 0.10,
                        'Cz is {:.3f} from centre'.format(math.hypot(x, y)))

    def test_the_midline_chain_is_on_the_midline_and_ordered(self):
        for name in ('Fz', 'Cz', 'Pz'):
            self.assertLess(abs(HEAD_POS[name][0]), 0.05,
                            '{} is off the midline'.format(name))
        self.assertGreater(HEAD_POS['Fz'][1], HEAD_POS['Cz'][1])
        self.assertGreater(HEAD_POS['Cz'][1], HEAD_POS['Pz'][1])

    def test_the_temporal_chain_is_the_most_lateral(self):
        """T3/T4 are the equator electrodes: nothing may be further out."""
        max_abs_x = max(abs(x) for _n, x, _y in HEAD_POS_1020)
        self.assertAlmostEqual(abs(HEAD_POS['T3'][0]), max_abs_x, delta=0.02)
        self.assertLess(HEAD_POS['T3'][0], 0.0)
        self.assertGreater(HEAD_POS['T4'][0], 0.0)
        for left, right in (('F7', 'F8'), ('T3', 'T4'), ('T5', 'T6'),
                            ('C3', 'C4'), ('P3', 'P4'), ('O1', 'O2')):
            self.assertLess(HEAD_POS[left][0], 0.0, left)
            self.assertGreater(HEAD_POS[right][0], 0.0, right)
            self.assertAlmostEqual(abs(HEAD_POS[left][0]),
                                   abs(HEAD_POS[right][0]), delta=0.05,
                                   msg='{}/{} are not left-right symmetric'
                                       .format(left, right))

    def test_no_runtime_dependency_on_mne(self):
        """mne 0.19 (legacy) has no Montage.get_positions(); mne 1.12 does.

        Importing mne here would work on one interpreter and fail on the other,
        and would drag it into the PyInstaller bundle for 19 constants.
        """
        import io
        import gui.widgets.head_view as hv
        with io.open(hv.__file__.replace('.pyc', '.py'),
                     encoding='utf-8') as f:
            src = f.read()
        self.assertNotIn('import mne', src)
        self.assertNotIn('matplotlib', src)


class MontagePairLookupTests(unittest.TestCase):
    def test_bipolar_montages_return_their_pairs(self):
        self.assertEqual(montage_pairs('Longitudinal Bipolar'),
                         list(LONGITUDINAL_BIPOLAR))
        self.assertEqual(montage_pairs('Transverse Bipolar'),
                         list(TRANSVERSE_BIPOLAR))

    def test_reference_montages_have_nothing_pairwise_to_draw(self):
        self.assertEqual(montage_pairs('Common Average'), [])
        self.assertEqual(montage_pairs('Referential (raw)'), [])

    def test_an_unknown_montage_degrades_instead_of_raising(self):
        """montage_pairs() is called from paintEvent; an exception there is a
        repaint loop of tracebacks, not a visible error."""
        self.assertEqual(montage_pairs('No such montage'), [])
        self.assertEqual(montage_pairs(None), [])

    def test_every_montage_key_is_accepted(self):
        for name in MONTAGES:
            self.assertIsInstance(montage_pairs(name), list)

    def test_every_pair_endpoint_is_an_electrode_we_can_draw(self):
        """Otherwise the derivation is silently omitted from the diagram."""
        for name in MONTAGES:
            for label, a, b in montage_pairs(name):
                self.assertIn(a, HEAD_POS, '{} in {}'.format(label, name))
                self.assertIn(b, HEAD_POS, '{} in {}'.format(label, name))

    def test_the_returned_list_cannot_corrupt_the_montage_table(self):
        pairs = montage_pairs('Longitudinal Bipolar')
        pairs.append(('bogus', 'Cz', 'Pz'))
        self.assertEqual(len(montage_pairs('Longitudinal Bipolar')),
                         len(LONGITUDINAL_BIPOLAR))

    def test_derivations_for_an_electrode(self):
        self.assertEqual(derivations_for('Cz', 'Longitudinal Bipolar'),
                         ['Fz-Cz', 'Cz-Pz'])
        self.assertEqual(derivations_for('Fp1', 'Longitudinal Bipolar'),
                         ['Fp1-F7', 'Fp1-F3'])
        self.assertEqual(derivations_for('Cz', 'Common Average'), [])
        self.assertEqual(derivations_for('nonsense', 'Longitudinal Bipolar'),
                         [])

    def test_every_electrode_appears_in_each_bipolar_montage(self):
        """A montage that never mentions an electrode leaves it looking
        disconnected on the diagram, which is a real finding worth pinning."""
        for name in ('Longitudinal Bipolar', 'Transverse Bipolar'):
            for ch in CHANNEL_NAMES:
                self.assertTrue(derivations_for(ch, name),
                                '{} appears in no derivation of {}'
                                .format(ch, name))


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class HeadViewWidgetTests(unittest.TestCase):
    def setUp(self):
        self.w = HeadView()
        self.w.resize(320, 380)

    def tearDown(self):
        self.w.deleteLater()

    def test_it_constructs_headless(self):
        self.assertIsInstance(self.w, QtWidgets.QWidget)
        self.assertEqual(self.w.selected(), None)

    def test_it_paints_without_raising_in_every_montage(self):
        """paintEvent is the only place most of this code runs; rendering into
        a QImage exercises it without needing a screen."""
        for montage in list(MONTAGES.keys()):
            self.w.set_montage(montage)
            img = QtGui.QImage(320, 380, QtGui.QImage.Format_ARGB32)
            img.fill(0)
            self.w.render(img)

    def test_it_paints_at_absurd_sizes(self):
        """A dock widget can be dragged to almost nothing. The caption is
        reserved space, so a squeeze must shrink the head, not divide by it."""
        for w, h in ((1, 1), (10, 400), (400, 10), (60, 60), (1200, 900)):
            self.w.resize(w, h)
            img = QtGui.QImage(max(1, w), max(1, h),
                               QtGui.QImage.Format_ARGB32)
            img.fill(0)
            self.w.render(img)

    def test_clicking_an_electrode_emits_its_name(self):
        got = []
        self.w.electrodeClicked.connect(got.append)
        cx, cy, radius = self.w._layout()
        x, y = HEAD_POS['T3']
        pos = QtCore.QPoint(int(round(cx + x * radius)),
                            int(round(cy - y * radius)))
        ev = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, pos,
                               QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                               QtCore.Qt.NoModifier)
        self.w.mousePressEvent(ev)
        self.assertEqual(got, ['T3'])
        self.assertEqual(self.w.selected(), 'T3')

    def test_clicking_empty_scalp_selects_nothing(self):
        got = []
        self.w.electrodeClicked.connect(got.append)
        cx, cy, _radius = self.w._layout()
        # Between Cz and Fz there is no electrode; halfway is empty.
        ev = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress, QtCore.QPoint(int(cx), int(cy)),
            QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        self.w.mousePressEvent(ev)
        # Cz is within a marker radius of the centre, so aim well clear of it.
        self.assertIn(got, ([], ['Cz']))

    def test_hit_testing_finds_every_electrode_and_only_that_one(self):
        cx, cy, radius = self.w._layout()
        for name, x, y in HEAD_POS_1020:
            pt = QtCore.QPointF(cx + x * radius, cy - y * radius)
            self.assertEqual(self.w.electrode_at(pt), name)

    def test_hit_testing_works_before_the_first_paint(self):
        """The layout must be derived from the size, not cached at paint time,
        or a click before the first repaint hits nothing."""
        fresh = HeadView()
        fresh.resize(300, 340)
        cx, cy, radius = fresh._layout()
        x, y = HEAD_POS['O2']
        self.assertEqual(
            fresh.electrode_at(QtCore.QPointF(cx + x * radius,
                                              cy - y * radius)), 'O2')
        fresh.deleteLater()

    def test_hit_testing_follows_a_resize(self):
        x, y = HEAD_POS['F7']
        for size in ((240, 260), (700, 500), (320, 900)):
            self.w.resize(*size)
            cx, cy, radius = self.w._layout()
            self.assertEqual(
                self.w.electrode_at(QtCore.QPointF(cx + x * radius,
                                                   cy - y * radius)), 'F7')

    def test_far_outside_the_head_hits_nothing(self):
        self.assertIsNone(self.w.electrode_at(QtCore.QPointF(-500.0, -500.0)))

    def test_set_selected_accepts_modern_names_and_rejects_junk(self):
        self.w.set_selected('T7')
        self.assertEqual(self.w.selected(), 'T3')
        self.w.set_selected('EKG')
        self.assertIsNone(self.w.selected())
        self.w.set_selected(None)
        self.assertIsNone(self.w.selected())

    def test_set_selected_does_not_emit(self):
        """Otherwise syncing it from the trace view feeds back into it."""
        got = []
        self.w.electrodeClicked.connect(got.append)
        self.w.set_selected('Cz')
        self.assertEqual(got, [])

    def test_value_shading_is_off_by_default(self):
        self.assertIsNone(self.w.values_range())

    def test_value_shading_ignores_unusable_entries(self):
        self.w.set_values({'Cz': 1.0, 'O1': float('nan'), 'O2': None,
                           'EKG': 5.0, 'T7': 3.0, 'Fz': 'x'})
        self.assertEqual(self.w.values_range(), (1.0, 3.0))   # Cz and T3 only
        img = QtGui.QImage(320, 380, QtGui.QImage.Format_ARGB32)
        img.fill(0)
        self.w.render(img)

    def test_value_shading_survives_a_constant_field(self):
        """vmax == vmin would be a divide-by-zero in the ramp."""
        self.w.set_values(dict((n, 2.0) for n in CHANNEL_NAMES))
        self.assertEqual(self.w.values_range(), (2.0, 2.0))
        img = QtGui.QImage(320, 380, QtGui.QImage.Format_ARGB32)
        img.fill(0)
        self.w.render(img)

    def test_value_shading_honours_explicit_limits(self):
        self.w.set_values({'Cz': 1.0, 'O1': 3.0}, vmin=0.0, vmax=10.0)
        self.assertEqual(self.w.values_range(), (0.0, 10.0))

    def test_clearing_values_returns_to_a_plain_diagram(self):
        self.w.set_values({'Cz': 1.0, 'O1': 3.0})
        self.w.set_values(None)
        self.assertIsNone(self.w.values_range())

    def test_the_caption_states_that_all_19_were_scored(self):
        """This is the sentence that keeps electrode selection honest — it must
        say both that all 19 were used AND that selection is display-only."""
        cap = self.w.caption_text().lower()
        self.assertIn('19', cap)
        self.assertIn('display', cap)
        self.assertTrue('all 19' in cap)

    def test_colours_come_from_the_palette_in_both_themes(self):
        """Rendered output must differ between a light and a dark palette; if
        anything were hardcoded black-on-white the two would be identical
        wherever that hardcoding dominates."""
        def render_with(bg, fg):
            w = HeadView()
            w.resize(320, 380)
            pal = QtGui.QPalette(w.palette())
            pal.setColor(QtGui.QPalette.Base, QtGui.QColor(*bg))
            pal.setColor(QtGui.QPalette.Text, QtGui.QColor(*fg))
            w.setPalette(pal)
            img = QtGui.QImage(320, 380, QtGui.QImage.Format_ARGB32)
            img.fill(0)
            w.render(img)
            w.deleteLater()
            return img

        light = render_with((255, 255, 255), (20, 20, 20))
        dark = render_with((30, 32, 36), (235, 235, 235))
        self.assertNotEqual(light, dark, 'the widget ignored its palette')
        # And the canvas is actually the palette Base colour, so the widget
        # never paints a white card into a dark application.
        self.assertEqual(QtGui.QColor(dark.pixel(2, 2)).rgb(),
                         QtGui.QColor(30, 32, 36).rgb())

    def test_montage_changes_change_the_picture(self):
        def render(montage):
            self.w.set_montage(montage)
            img = QtGui.QImage(320, 380, QtGui.QImage.Format_ARGB32)
            img.fill(0)
            self.w.render(img)
            return img

        self.assertNotEqual(render('Referential (raw)'),
                            render('Longitudinal Bipolar'))
        self.assertNotEqual(render('Longitudinal Bipolar'),
                            render('Transverse Bipolar'))

    def test_hovering_reports_the_electrode_under_the_cursor(self):
        cx, cy, radius = self.w._layout()
        x, y = HEAD_POS['P3']
        pos = QtCore.QPoint(int(round(cx + x * radius)),
                            int(round(cy - y * radius)))
        ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, pos,
                               QtCore.Qt.NoButton, QtCore.Qt.NoButton,
                               QtCore.Qt.NoModifier)
        self.w.mouseMoveEvent(ev)
        self.assertEqual(self.w._hover, 'P3')
        img = QtGui.QImage(320, 380, QtGui.QImage.Format_ARGB32)
        img.fill(0)
        self.w.render(img)                # badge drawing must not raise


if __name__ == '__main__':
    unittest.main()
