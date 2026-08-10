"""Path resolution that has to survive leaving the developer's machine.

Every failure guarded here is one that appears only on the target machine — a
hospital workstation with no repository checkout, no writable recording folder,
and no working directory anyone controls. They cannot be caught by running the
GUI from the repo root, which is why they are pinned here.
"""
import os
import shutil
import stat
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui import paths                                       # noqa: E402
from gui.io import cache                                    # noqa: E402


class ForeignWorkingDirectory(unittest.TestCase):
    """The app must not depend on the process working directory.

    The loader used to os.chdir into utils/ so that a bare montage filename
    resolved. That is invisible from the repo root and fatal in a frozen build,
    where there is no utils/ to chdir into.
    """

    def setUp(self):
        self._cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_montage_parameters_load_from_any_directory(self):
        from utils.pyst import nedc_load_parameters
        self.assertIsNotNone(
            nedc_load_parameters('params_common_electrodes.txt'),
            'montage parameters must resolve without a chdir into utils/')

    def test_montage_parameters_load_by_absolute_path(self):
        from utils.pyst import nedc_load_parameters
        self.assertIsNotNone(nedc_load_parameters(paths.params_path()))

    def test_no_module_calls_chdir_at_import_or_load(self):
        """A chdir anywhere in the load path is a latent packaging failure."""
        import gui.io.edf
        import gui.io.infer
        for mod in (gui.io.edf, gui.io.infer):
            with open(mod.__file__.replace('.pyc', '.py'),
                      encoding='utf-8') as f:
                src = f.read()
            code = '\n'.join(l for l in src.splitlines()
                             if not l.lstrip().startswith('#'))
            self.assertNotIn('os.chdir(', code,
                             '{} changes the working directory'.format(
                                 mod.__name__))


class BundledResources(unittest.TestCase):
    def test_weights_resolve_to_a_real_file(self):
        self.assertTrue(os.path.exists(paths.weights_path()),
                        paths.weights_path())

    def test_params_resolve_to_a_real_file(self):
        self.assertTrue(os.path.exists(paths.params_path()),
                        paths.params_path())

    def test_writable_root_stays_namespaced_when_localappdata_is_unusable(self):
        """It must never resolve to a bare home or temp directory.

        Doing so would scatter generic logs/, cache/ and autosave/ folders into
        the user's home, with names too generic for anyone to attribute.
        """
        saved_frozen = getattr(sys, 'frozen', None)
        saved_local = os.environ.get('LOCALAPPDATA')
        saved_app = os.environ.get('APPDATA')
        try:
            sys.frozen = True
            os.environ['LOCALAPPDATA'] = os.path.join(
                tempfile.mkdtemp(), 'nope', 'still-nope')
            os.environ.pop('APPDATA', None)
            root = paths.writable_root()
            self.assertEqual(os.path.basename(root), 'SeizureReview')
            self.assertTrue(os.path.isdir(root))
        finally:
            if saved_frozen is None:
                del sys.frozen
            else:
                sys.frozen = saved_frozen
            if saved_local is None:
                os.environ.pop('LOCALAPPDATA', None)
            else:
                os.environ['LOCALAPPDATA'] = saved_local
            if saved_app is not None:
                os.environ['APPDATA'] = saved_app

    def test_writable_root_is_never_the_bundle_when_frozen(self):
        """sys._MEIPASS is deleted on exit; writing a review there loses it."""
        saved = getattr(sys, 'frozen', None)
        saved_meipass = getattr(sys, '_MEIPASS', None)
        try:
            sys.frozen = True
            sys._MEIPASS = tempfile.mkdtemp()
            self.assertNotEqual(os.path.abspath(paths.writable_root()),
                                os.path.abspath(sys._MEIPASS))
        finally:
            if saved is None:
                del sys.frozen
            else:
                sys.frozen = saved
            if saved_meipass is None:
                if hasattr(sys, '_MEIPASS'):
                    del sys._MEIPASS
            else:
                sys._MEIPASS = saved_meipass


class Provenance(unittest.TestCase):
    """An export must identify the model and build that produced it.

    On a frozen build there is no repository and the target machine has no git,
    so the naive implementations of both of these return None — silently
    stripping provenance from exactly the annotations that will be relied on.
    """

    def test_weights_hash_matches_the_reviewed_model(self):
        import eval_config as cfg
        from gui.app import _weights_sha256
        self.assertEqual(_weights_sha256(), cfg.WEIGHTS_SHA256)

    def test_frozen_build_reads_its_commit_from_the_bundled_stamp(self):
        import json
        from gui.app import _git_commit
        bundle = tempfile.mkdtemp()
        with open(os.path.join(bundle, 'build_info.json'), 'w') as f:
            json.dump({'commit': 'abc1234'}, f)
        saved = getattr(sys, 'frozen', None)
        saved_meipass = getattr(sys, '_MEIPASS', None)
        try:
            sys.frozen = True
            sys._MEIPASS = bundle
            self.assertEqual(_git_commit(), 'abc1234')
        finally:
            if saved is None:
                del sys.frozen
            else:
                sys.frozen = saved
            if saved_meipass is None:
                if hasattr(sys, '_MEIPASS'):
                    del sys._MEIPASS
            else:
                sys._MEIPASS = saved_meipass
            shutil.rmtree(bundle, ignore_errors=True)

    def test_a_frozen_build_with_no_stamp_reports_none_rather_than_raising(self):
        from gui.app import _git_commit
        bundle = tempfile.mkdtemp()
        saved = getattr(sys, 'frozen', None)
        saved_meipass = getattr(sys, '_MEIPASS', None)
        try:
            sys.frozen = True
            sys._MEIPASS = bundle
            self.assertIsNone(_git_commit())
        finally:
            if saved is None:
                del sys.frozen
            else:
                sys.frozen = saved
            if saved_meipass is None:
                if hasattr(sys, '_MEIPASS'):
                    del sys._MEIPASS
            else:
                sys._MEIPASS = saved_meipass
            shutil.rmtree(bundle, ignore_errors=True)


class ReadOnlyRecordingDirectory(unittest.TestCase):
    """Recordings normally live on a read-only share on a clinical PC.

    Inference takes minutes. Losing it because the sidecar could not be written
    next to the EDF would be the single most infuriating failure in the field.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.edf = os.path.join(self.tmp, 'rec.edf')
        with open(self.edf, 'wb') as f:
            f.write(b'0' * 64)
        self._home = paths.writable_root
        self.store = tempfile.mkdtemp()
        paths.writable_root = lambda: self.store
        cache.writable_root = lambda: self.store

    def tearDown(self):
        paths.writable_root = self._home
        cache.writable_root = self._home
        os.chmod(self.tmp, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.store, ignore_errors=True)

    def _save(self):
        return cache.save_probs(
            self.edf, np.arange(4, dtype=np.int32),
            np.array([0.1, 0.2, 0.9, 0.3], dtype=np.float32),
            meta={'source': 'test'},
            skip_code=np.zeros(4, dtype=np.int8))

    def test_sidecar_is_preferred_when_the_directory_is_writable(self):
        where = self._save()
        self.assertEqual(os.path.abspath(where),
                         os.path.abspath(cache.cache_path_for(self.edf)))

    def test_a_sidecar_cache_is_read_back(self):
        self._save()
        got = cache.load_probs(self.edf)
        self.assertIsNotNone(got)
        self.assertEqual(len(got['probs']), 4)

    def test_falls_back_when_the_sidecar_cannot_be_written(self):
        """Simulated with a genuine OS-level write failure.

        The sidecar is directed under a path component that exists as a regular
        file, so the write raises NotADirectoryError — the same OSError family a
        read-only share raises, without needing to manipulate Windows ACLs.
        """
        blocker = os.path.join(self.tmp, 'blocker')
        with open(blocker, 'wb') as f:
            f.write(b'not a directory')
        real = cache.cache_path_for
        cache.cache_path_for = lambda p: os.path.join(
            blocker, 'sub', 'x.probs.npz')
        try:
            where = self._save()
        finally:
            cache.cache_path_for = real
        self.assertEqual(os.path.abspath(where),
                         os.path.abspath(cache.fallback_cache_path_for(self.edf)))
        self.assertTrue(os.path.exists(where))

    def test_the_fallback_cache_is_found_on_read(self):
        cache.save_probability_file(
            cache.fallback_cache_path_for(self.edf),
            np.arange(4, dtype=np.int32),
            np.array([0.1, 0.2, 0.9, 0.3], dtype=np.float32),
            meta={'edf_size': os.stat(self.edf).st_size,
                  'edf_mtime': os.stat(self.edf).st_mtime},
            skip_code=np.zeros(4, dtype=np.int8))
        self.assertIsNotNone(cache.load_probs(self.edf))

    def test_fallback_keys_do_not_collide_across_directories(self):
        other = os.path.join(tempfile.mkdtemp(), 'rec.edf')
        self.assertNotEqual(cache.fallback_cache_path_for(self.edf),
                            cache.fallback_cache_path_for(other))

    def test_a_stale_cache_is_rejected_in_both_locations(self):
        self._save()
        with open(self.edf, 'ab') as f:      # EDF changed after caching
            f.write(b'more')
        self.assertIsNone(cache.load_probs(self.edf))


if __name__ == '__main__':
    unittest.main()
