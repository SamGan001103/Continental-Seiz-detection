"""The two detectors must not be able to affect each other.

There are now two models in this application, and they are unrelated:

    SEIZURE   _original_data -> ICA -> log-STFT -> ConvLSTM (convlstm_ICA_12_train.h5)
    INTERICTAL raw 19ch -> 128 Hz, 0.5 Hz HP, notch -> 37ch -> CNN (spikenet1.o_weights.h5)

Different weights, different sampling rate, different montage, different window
length, and no ICA on the interictal side at all. Nothing about one should be
reachable from the other, because the seizure path's reproduction argument
rests on it producing exactly the published model's scores: anything that
perturbs its input, its filters or its FastICA patch invalidates that
retroactively, with nothing on screen to show it happened.

Until now that was true only by how the code happened to be written. These
tests make it a property of the repository. They are cheap -- mostly import
bookkeeping -- so they can run on every change.
"""
import os
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Anything belonging to the seizure path. If importing or running the
# interictal detector pulls in one of these, the two are no longer separable.
SEIZURE_MODULES = (
    'gui.io.infer',
    'gui.processing',
    'models.deep_conv_lstm',
    'utils.preprocessing',
    'utils.ICA_load_data_elec',
)

PROBE = r'''
import json, sys
before = set(sys.modules)
from gui.io import ied_detect
pulled = sorted(m for m in set(sys.modules) - before)
print(json.dumps({"pulled": pulled}))
'''


def _run(code):
    env = dict(os.environ)
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['PYTHONWARNINGS'] = 'ignore'
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in reversed(out.stdout.decode('utf-8', 'replace').splitlines()):
        line = line.strip()
        if line.startswith('{'):
            import json
            return json.loads(line)
    raise AssertionError('probe produced no result:\n{}'
                         .format(out.stderr.decode('utf-8', 'replace')[-2000:]))


class TheInterictalDetectorDoesNotReachTheSeizurePath(unittest.TestCase):
    def test_importing_it_pulls_in_no_seizure_code(self):
        pulled = _run(PROBE)['pulled']
        leaked = [m for m in pulled
                  if any(m == s or m.startswith(s + '.')
                         for s in SEIZURE_MODULES)]
        self.assertEqual(leaked, [], 'the interictal detector imported '
                                     'seizure-path modules: {}'.format(leaked))

    def test_it_does_not_import_tensorflow_just_by_being_imported(self):
        """The review window must open on a machine with no TensorFlow, and
        must not pay a multi-second import for a feature nobody switched on.
        The model load is deliberately lazy."""
        pulled = _run(PROBE)['pulled']
        self.assertNotIn('tensorflow', pulled)

    def test_it_never_touches_the_seizure_weights(self):
        from gui.io import ied_detect
        _, weights = ied_detect.model_paths()
        self.assertIn('spikenet', os.path.basename(weights).lower())
        self.assertNotIn('convlstm', os.path.basename(weights).lower())

    def test_its_source_names_no_seizure_module_outside_prose(self):
        """Grep alone cannot tell an import from a docstring saying 'this does
        not import X', so this checks the parsed import statements."""
        import ast
        path = os.path.join(REPO, 'gui', 'io', 'ied_detect.py')
        with open(path) as fh:
            tree = ast.parse(fh.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        for mod in SEIZURE_MODULES:
            self.assertNotIn(mod, names)


class TheTwoModelsAgreeOnNothingByAccident(unittest.TestCase):
    """If these ever coincide, a later change could feed one model's prepared
    input to the other and the shapes would not complain."""

    def test_the_sampling_rates_differ(self):
        from gui.io import ied_detect
        self.assertEqual(ied_detect.FS, 128)

    def test_the_channel_count_is_the_expanded_montage_not_nineteen(self):
        from gui.io import ied_detect
        self.assertEqual(len(ied_detect.MONOPOLAR) + len(ied_detect.BIPOLAR),
                         37)

    def test_the_window_is_one_second_not_twelve(self):
        from gui.io import ied_detect
        self.assertEqual(ied_detect.WINDOW / float(ied_detect.FS), 1.0)

    def test_the_outputs_go_to_different_files(self):
        from gui.io.cache import cache_path_for
        from gui.io.ied_marks import ied_path_for
        edf = '/x/rec.edf'
        self.assertNotEqual(cache_path_for(edf), ied_path_for(edf))
        self.assertTrue(cache_path_for(edf).endswith('.probs.npz'))
        self.assertTrue(ied_path_for(edf).endswith('.ied.csv'))


class NoSharedMutableState(unittest.TestCase):
    def test_the_interictal_path_does_not_install_a_global_patch(self):
        """`fastica_pinned` patches FastICA process-wide for the seizure path.
        The interictal path uses no ICA, so it must neither install nor
        uninstall anything -- an uninstall would silently change the seizure
        detector's output for the rest of the session."""
        import ast
        path = os.path.join(REPO, 'gui', 'io', 'ied_detect.py')
        with open(path) as fh:
            src = fh.read()
        tree = ast.parse(src)
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertNotIn('install', called)
        self.assertNotIn('uninstall', called)
        self.assertNotIn('fastica_pinned', src)


if __name__ == '__main__':
    unittest.main()
