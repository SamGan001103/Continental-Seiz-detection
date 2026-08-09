"""Regression tests for the P1 findings of the first cognitive walkthrough.

Each test names the finding it pins. See
docs/usability/cognitive_walkthrough_results.md for the issue list and severity
scoring. The point of these is that a re-run of the walkthrough should show the
Q2/Q3 failures collapse — if one of these breaks, that result silently reverts.
"""
import unittest

try:
    from PyQt5 import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ['-platform', 'offscreen'])
    from gui.app import MainWindow
    from gui.widgets.event_list import STATUS_STYLE, EventList
    from gui.widgets.signal_view import SignalView
    HAVE_QT = True
except Exception:                                    # pragma: no cover
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, 'PyQt5 not available')
class WalkthroughP1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = MainWindow()

    # ---------------------------------------------------------- U-02
    def test_menu_bar_exists_with_help(self):
        titles = [a.text() for a in self.w.menuBar().actions()]
        self.assertTrue(any('File' in t for t in titles), titles)
        self.assertTrue(any('View' in t for t in titles), titles)
        self.assertTrue(any('Help' in t for t in titles), titles)

    def test_keyboard_shortcuts_are_discoverable_on_f1(self):
        help_menu = [a for a in self.w.menuBar().actions()
                     if 'Help' in a.text()][0].menu()
        keys = [a for a in help_menu.actions() if 'Keyboard' in a.text()]
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0].shortcut().toString(), 'F1')

    def test_shortcut_dialog_text_comes_from_the_live_docstring(self):
        """So the dialog cannot drift from the shortcuts actually wired."""
        import gui.app as A
        self.assertIn('Keyboard shortcuts:', A.__doc__)
        for key in ('Space', 'J / K', 'Enter'):
            self.assertIn(key, A.__doc__)

    # ---------------------------------------------------------- U-04
    def test_channel_inspector_reachable_without_double_click(self):
        self.assertTrue(hasattr(self.w, 'a_inspector'))
        self.assertEqual(self.w.a_inspector.shortcut().toString(), 'Ctrl+I')
        self.assertTrue(self.w.a_inspector.toolTip())

    # ---------------------------------------------------------- U-07
    def test_every_event_status_has_a_distinct_colour(self):
        colours = [v[0] for v in STATUS_STYLE.values()]
        self.assertEqual(len(colours), len(set(colours)),
                         'two statuses share a colour')

    def test_status_is_never_colour_alone(self):
        """Colour-blind reviewers must still be able to read the status."""
        for status, (_rgb, glyph, tip) in STATUS_STYLE.items():
            self.assertTrue(glyph, status)
            self.assertTrue(tip, status)

    def test_status_cell_carries_colour_glyph_and_tooltip(self):
        lst = EventList()
        for status in ('proposed', 'accepted', 'rejected', 'edited'):
            item = lst._status_cell(status)
            self.assertIn(status, item.text())
            self.assertTrue(item.toolTip())
            rgb = item.background().color().getRgb()[:3]
            self.assertEqual(rgb, STATUS_STYLE[status][0])

    def test_event_list_and_signal_view_agree_on_status_colours(self):
        for status, (rgb, _g, _t) in STATUS_STYLE.items():
            self.assertEqual(SignalView.STATUS_COLOURS[status][:3], rgb,
                             'list and trace disagree on ' + status)

    # ---------------------------------------------------------- U-08
    def test_accepted_does_not_share_the_reference_bands_green(self):
        """Reference bands are (50,160,70). A reviewer's own confirmation must
        not look like the answer key."""
        r, g, b = SignalView.STATUS_COLOURS['accepted'][:3]
        self.assertFalse(g > r and g > b,
                         'accepted is still green-dominant like the reference')

    # ---------------------------------------------------------- U-09
    def test_blind_mode_exists_and_is_checkable(self):
        self.assertTrue(self.w.a_blind.isCheckable())
        self.assertEqual(self.w.a_blind.shortcut().toString(), 'Ctrl+B')

    def test_blind_mode_hides_references_and_stays_in_sync(self):
        self.w.a_blind.setChecked(True)
        self.assertFalse(self.w.chk_refs.isChecked())
        self.w.a_blind.setChecked(False)
        self.assertTrue(self.w.chk_refs.isChecked())
        # the toolbar route must drive the menu item too
        self.w.chk_refs.setChecked(False)
        self.assertTrue(self.w.a_blind.isChecked())
        self.w.chk_refs.setChecked(True)


if __name__ == '__main__':
    unittest.main()
