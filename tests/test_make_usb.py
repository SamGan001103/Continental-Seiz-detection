"""The USB assembler must not destroy a good build to write a bad one.

`make_usb.py` refills a slot on a memory stick. It used to copy straight over
the destination, which `copy_tree` removes first (make_usb.py:258) -- so an
interrupted copy left the slot EMPTY: the previous, working build deleted and the
new one incomplete. That is not a hypothetical. It happened once, mid-refill, and
the stick sat there with 16 KB where 1.4 GB of verified application had been.

Two things covered here, found together and easy to confuse. The 16 KB was a
stale `NOT_BUILT.txt` that had been sitting beside the build all along, not
something the failure wrote; the failure only removed the build that was hiding
it. `MakeUsbPlaceholders` covers that half.

The moment it happens is the worst one available. Refilling a stick is something
you do shortly before showing the application to somebody, and a USB write is
precisely the operation that gets interrupted -- a pulled cable, a full volume,
an impatient eject. Losing the old build buys nothing in exchange.

So the copy goes to a staging directory and is swapped in only once it has
finished. These tests hold that line by breaking the copy on purpose.
"""
import os
import shutil
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import tempfile

# Loaded by path, not as `packaging.make_usb`. That import resolves to this
# repository on the 3.6 stack and to the PyPI `packaging` distribution on the
# modern one, so it is forbidden -- see test_packaging_name.py, which caught this
# very file doing it. The first version of these tests passed on 3.6 for exactly
# the reason the guard exists.
_MAKE_USB = os.path.join(REPO, 'packaging', 'make_usb.py')
if sys.version_info >= (3, 5):
    import importlib.util
    _spec = importlib.util.spec_from_file_location('_make_usb_under_test',
                                                   _MAKE_USB)
    make_usb = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(make_usb)
else:                                                       # pragma: no cover
    import imp
    make_usb = imp.load_source('_make_usb_under_test', _MAKE_USB)


def _write(path, text):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    f = open(path, 'w')
    try:
        f.write(text)
    finally:
        f.close()


def _read(path):
    f = open(path)
    try:
        return f.read()
    finally:
        f.close()


class MakeUsbCopyIsAtomic(unittest.TestCase):
    """A failed copy leaves the slot as it was, not empty."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='usbtest')
        self.dest = os.path.join(self.tmp, 'stick')
        self.dist = os.path.join(self.tmp, 'dist', 'SeizureReview')
        # a plausible new build
        _write(os.path.join(self.dist, 'SeizureReview.exe'), 'new binary')
        _write(os.path.join(self.dist, '_internal', 'weights.h5'), 'new weights')
        self._real_copy_tree = make_usb.copy_tree

    def tearDown(self):
        make_usb.copy_tree = self._real_copy_tree
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _slot(self):
        return os.path.join(self.dest, make_usb.current_platform(),
                            'SeizureReview')

    def _plant_previous_build(self):
        """Put a working build in the slot, as a real stick would have."""
        _write(os.path.join(self._slot(), 'SeizureReview.exe'), 'OLD binary')
        _write(os.path.join(self._slot(), '_internal', 'weights.h5'),
               'OLD weights')

    def _run(self):
        return make_usb.main(['--dest', self.dest, '--dist', self.dist,
                              '--skip-source'])

    @staticmethod
    def _failing_copy_tree(wrote_partial):
        """A stand-in for copy_tree that fails the way the real one does.

        Faithfulness matters here, and a first version of these tests got it
        wrong: a mock that merely raises does not reproduce the fault, because
        the destruction is done by copy_tree itself -- it removes the
        destination before writing a byte (make_usb.py:258). Stubbing that out
        along with the failure makes the test pass against the buggy code, which
        is the worst outcome a regression test can have. So the stub clears the
        destination first, exactly as the real function does, and only then dies.
        """
        def stub(src, dst):
            if os.path.exists(dst):
                shutil.rmtree(dst)                  # what copy_tree does first
            if wrote_partial:
                _write(os.path.join(dst, 'SeizureReview.exe'), 'half a binary')
            raise IOError('cable pulled')
        return stub

    # -- the regression itself --------------------------------------------

    def test_interrupted_copy_leaves_the_previous_build_intact(self):
        self._plant_previous_build()

        # Get far enough to have written something, then die -- the shape of a
        # real interruption, not a clean refusal before any I/O.
        make_usb.copy_tree = self._failing_copy_tree(wrote_partial=True)
        self.assertRaises(Exception, self._run)

        binary = os.path.join(self._slot(), 'SeizureReview.exe')
        self.assertTrue(os.path.isfile(binary),
                        'the previous build was destroyed by a failed copy')
        self.assertEqual(_read(binary), 'OLD binary',
                         'the slot holds a partial copy, not the old build')
        self.assertEqual(
            _read(os.path.join(self._slot(), '_internal', 'weights.h5')),
            'OLD weights')

    def test_a_clean_failure_before_any_write_also_preserves_the_build(self):
        """The other shape of interruption: a full volume, refused up front."""
        self._plant_previous_build()

        make_usb.copy_tree = self._failing_copy_tree(wrote_partial=False)
        self.assertRaises(Exception, self._run)

        self.assertEqual(
            _read(os.path.join(self._slot(), 'SeizureReview.exe')),
            'OLD binary', 'the previous build did not survive')

    # -- and the swap still delivers the new build ------------------------

    def test_successful_copy_replaces_the_previous_build(self):
        self._plant_previous_build()
        self._run()

        binary = os.path.join(self._slot(), 'SeizureReview.exe')
        self.assertEqual(_read(binary), 'new binary',
                         'the new build did not replace the old one')
        self.assertEqual(
            _read(os.path.join(self._slot(), '_internal', 'weights.h5')),
            'new weights')

    def test_no_staging_directory_is_left_behind(self):
        """A stray `.incoming` would double the space used on the stick.

        On a 30 GB stick carrying two 1.4 GB builds that is survivable; the
        reason it matters is that a leftover staging directory is also a
        directory named `SeizureReview.incoming` sitting next to the real one,
        which is exactly the sort of thing somebody double-clicks.
        """
        self._plant_previous_build()
        self._run()

        slot_parent = os.path.dirname(self._slot())
        leftovers = [n for n in os.listdir(slot_parent)
                     if n.endswith('.incoming') or n.endswith('.previous')]
        self.assertEqual(leftovers, [],
                         'staging directories left on the stick: %r' % leftovers)

    def test_first_ever_copy_works_with_no_previous_build(self):
        """The swap must not assume there is something to swap out."""
        self._run()
        self.assertEqual(
            _read(os.path.join(self._slot(), 'SeizureReview.exe')),
            'new binary')

    def test_a_stale_staging_directory_does_not_poison_the_next_run(self):
        """Recovery after the failure above: the retry must still work.

        The rerun that recovered the real stick had to cope with whatever the
        crashed run left behind. If a stale `.incoming` made the retry fail,
        the fix would have converted a recoverable loss into a stuck one.
        """
        self._plant_previous_build()
        stale = self._slot() + '.incoming'
        _write(os.path.join(stale, 'SeizureReview.exe'), 'junk from last time')
        _write(os.path.join(stale, 'stray.txt'), 'junk')

        self._run()

        self.assertEqual(
            _read(os.path.join(self._slot(), 'SeizureReview.exe')),
            'new binary')
        self.assertFalse(os.path.exists(os.path.join(self._slot(), 'stray.txt')),
                         'stale staging content survived into the new build')


class MakeUsbPlaceholders(unittest.TestCase):
    """A filled slot must stop claiming to be empty.

    The placeholder is written into `<plat>/NOT_BUILT.txt` and the build lands
    in `<plat>/SeizureReview/`, so nothing ever removed the note. A real stick
    was found carrying a verified 1.4 GB Windows build and, next to it, a file
    instructing the reader to go and build Windows.

    Worth a test rather than a shrug, because the audience for that file is
    somebody who has been handed the stick and does not know which half to
    believe -- and on the run where the build was briefly missing, the stale
    placeholder was the only thing left in the directory and led me to the wrong
    explanation for the failure.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='usbph')
        self.dest = os.path.join(self.tmp, 'stick')
        self.dist = os.path.join(self.tmp, 'dist', 'SeizureReview')
        _write(os.path.join(self.dist, 'SeizureReview.exe'), 'new binary')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _plat_dir(self):
        return os.path.join(self.dest, make_usb.current_platform())

    def _run(self):
        return make_usb.main(['--dest', self.dest, '--dist', self.dist,
                              '--skip-source'])

    def test_filling_a_slot_removes_its_stale_placeholder(self):
        _write(os.path.join(self._plat_dir(), 'NOT_BUILT.txt'),
               'run packaging/build_app.bat')
        self._run()

        self.assertTrue(os.path.isdir(os.path.join(self._plat_dir(),
                                                   'SeizureReview')))
        self.assertFalse(
            os.path.isfile(os.path.join(self._plat_dir(), 'NOT_BUILT.txt')),
            'the stick still tells the reader this platform is not built')

    def test_unbuilt_platforms_keep_their_placeholder(self):
        """The removal must be conditional, or the stick loses its instructions."""
        self._run()

        others = [p for p in make_usb.PLATFORM_DIRS
                  if p != make_usb.current_platform()]
        for p in others:
            self.assertTrue(
                os.path.isfile(os.path.join(self.dest, p, 'NOT_BUILT.txt')),
                'no build and no placeholder for %s -- an empty directory with '
                'no explanation' % p)

if __name__ == '__main__':
    unittest.main()
