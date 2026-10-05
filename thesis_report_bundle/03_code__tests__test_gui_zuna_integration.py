import os
import tempfile
import unittest

import numpy as np

from gui.io.cache import load_probability_file, save_probability_file
from gui.io.zuna import zuna_session_paths


class GuiZunaIntegrationTests(unittest.TestCase):
    def test_explicit_probability_cache_round_trip(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, 'zuna.probs.npz')

        save_probability_file(
            path,
            window_starts=[0, 6, 12],
            probs=[0.1, 0.9, 0.2],
            meta={'source': 'full_zuna'})
        loaded = load_probability_file(path)

        self.assertEqual(loaded['meta']['source'], 'full_zuna')
        self.assertEqual(loaded['window_starts'].tolist(), [0, 6, 12])
        np.testing.assert_allclose(loaded['probs'], [0.1, 0.9, 0.2])

    def test_malformed_probability_cache_loads_as_missing(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, 'bad.probs.npz')
        np.savez_compressed(path, probs=np.asarray([0.1], dtype=np.float32))

        self.assertIsNone(load_probability_file(path))

    def test_zuna_session_paths_are_under_gui_artifacts(self):
        paths = zuna_session_paths(
            os.path.join('sample_data', 'patient', 'recording.edf'))

        self.assertIn(os.path.join('artifacts', 'gui_zuna'), paths['root'])
        self.assertTrue(paths['fif'].endswith(
            os.path.join('fif', 'recording.fif')))
        self.assertTrue(paths['npz'].endswith('recording.zuna_19ch.npz'))
        self.assertTrue(paths['probs'].endswith('recording.zuna.probs.npz'))

    def test_zuna_session_path_changes_when_edf_changes(self):
        tmp = tempfile.mkdtemp()
        edf = os.path.join(tmp, 'recording.edf')
        with open(edf, 'wb') as f:
            f.write(b'first')
        first = zuna_session_paths(edf)['root']

        with open(edf, 'wb') as f:
            f.write(b'second payload')
        second = zuna_session_paths(edf)['root']

        self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
