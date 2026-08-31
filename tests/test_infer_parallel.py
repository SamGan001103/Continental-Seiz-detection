"""The parallel scorer must produce the serial scorer's exact bytes.

`gui/io/infer_parallel.py` exists to make day-long recordings tractable — the
per-window ICA is 90 % of scoring time, and fanning it across cores divides
wall-clock by the worker count. Its entire licence to exist is that it changes
the schedule and nothing else: same windows, same seeded ICA per window, same
predict() call pattern, same floats.

"Same" here means `np.array_equal`, not allclose. This project has already
shipped one build whose numbers were silently different while every check
passed (the Keras 3 selection, docs/known_issues.md §1); a parallel path that
was merely "close" would be the same class of defect with a better excuse. If
this test fails, the parallel module is wrong — fix it or delete it, but do
not loosen the comparison.

Runs in a SUBPROCESS, and that is load-bearing twice over:

* The suite's GUI tests import PyQt5 into the runner, and on Windows a
  process that has loaded Qt can no longer initialise TensorFlow's native
  library (known_issues.md §1 — the same conflict the frozen app had). Run
  in-process, this test skips with a DLL error whenever it executes after
  any GUI test, which in discovery order is always. A first version did
  exactly that: green standalone, silently skipped in the full suite.

* Windows multiprocessing spawn re-imports the parent's __main__, so the
  scorer requires an importable main module — the subprocess gives it this
  file; the unittest runner's __main__ would also have worked, but the Qt
  problem forces the subprocess anyway.

The same comparison was also run on a real recording — ma0844az, the
30-minute Nihon Kohden export, 302 windows — and matched bitwise at 3.3× the
serial speed on 20 workers. That is recorded here rather than re-run; half an
hour of EEG does not belong in a unit test.
"""
import os
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _compare_main():
    """Runs in a clean interpreter: score both ways, compare bitwise."""
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    from gui.io.infer import compute_probs_from_data, _build_model
    from gui.io.infer_parallel import compute_probs_parallel

    fs = 250
    rng = np.random.RandomState(7)
    data = rng.randn(19, 40 * fs).astype(np.float32)
    data = uniform_filter1d(data, size=9, axis=1) * 40.0
    data[:, 12 * fs:24 * fs] = 0.0        # exercise the interrupted path

    model = _build_model()
    s_starts, s_probs, s_skip = compute_probs_from_data(data, fs, model=model)
    p_starts, p_probs, p_skip = compute_probs_parallel(
        data, fs, workers=2, model=model)

    assert np.array_equal(s_starts, p_starts), 'starts diverge'
    assert np.array_equal(s_skip, p_skip), \
        'skip codes diverge: serial %r parallel %r' % (
            s_skip.tolist(), p_skip.tolist())
    assert np.array_equal(s_probs, p_probs), \
        'probabilities diverge, max |delta| = %r' % (
            float(np.abs(s_probs - p_probs).max()))
    assert int((s_skip == 0).sum()) > 0, 'nothing was scored'
    assert int((s_skip != 0).sum()) > 0, \
        'the flatlined window was not rejected; the skip path went untested'
    assert p_starts.dtype == np.int32 and p_probs.dtype == np.float32 \
        and p_skip.dtype == np.int8, 'dtype contract broken'
    print('PARALLEL-BITWISE-OK windows=%d scored=%d' % (
        len(s_starts), int((s_skip == 0).sum())))


class ParallelEqualsSerial(unittest.TestCase):
    def test_bitwise_identical_in_a_clean_interpreter(self):
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), '--compare'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=REPO, timeout=600)
        out = proc.stdout.decode('utf-8', 'replace')
        if proc.returncode != 0 and (
                'DLL load failed' in out
                or 'No module named' in out
                or 'weights' in out.lower() and 'not found' in out.lower()):
            self.skipTest('inference stack unavailable in a clean '
                          'interpreter:\n' + out[-1500:])
        self.assertEqual(
            proc.returncode, 0,
            'parallel-vs-serial comparison failed:\n' + out[-3000:])
        self.assertIn(
            'PARALLEL-BITWISE-OK', out,
            'comparison exited 0 without reporting success:\n' + out[-1500:])


if __name__ == '__main__':
    if '--compare' in sys.argv:
        _compare_main()
    else:
        unittest.main()
