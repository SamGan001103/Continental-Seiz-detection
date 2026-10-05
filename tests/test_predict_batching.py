"""Batched prediction is opt-in, and the default must stay bit-identical.

Why the option exists: spreading the ICA across workers does not speed up
`model.predict`, so once the workers are busy the serial predict is roughly
three quarters of what is left. Measured on a 24-hour recording, 12 workers
bought 1.7x rather than 12x. Batching addresses exactly that — 42.3 ms per
window at a batch of one, 7.1 ms at 32, 5.0 ms at 48 — which is the difference
between thirty minutes per recording and twelve on the hardware this is meant
to run on.

Why it is off by default: a batched matmul reduces in a different order, so the
probabilities move. This project already threw away an 8.8x parallel scorer
over a 0.138 shift in p(seizure) and shipped a 3.3x bit-identical one instead.
So the option is allowed to exist, but it has to announce itself: the default
is unchanged, and any cache built with it records that it was.

These tests need no TensorFlow — they pin the plumbing and the promises. The
numerical bound is a measurement, reported in the docstrings, not something a
unit test can assert without a model.
"""
import argparse
import inspect
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class TheDefaultIsTheBitIdenticalPath(unittest.TestCase):
    def test_the_parallel_scorer_defaults_to_one_window_per_call(self):
        from gui.io import infer_parallel
        sig = inspect.signature(infer_parallel.compute_probs_parallel)
        self.assertIn('batch_size', sig.parameters)
        self.assertEqual(
            sig.parameters['batch_size'].default, 1,
            'batching must be opt-in: it is not bit-identical to the serial '
            'loop, and every cache in the project was built without it')

    def test_the_command_line_defaults_to_one(self):
        import precompute_probs
        src = inspect.getsource(precompute_probs.main)
        self.assertIn("'--batch'", src)
        self.assertIn('default=1', src)

    def test_run_defaults_to_one(self):
        import precompute_probs
        sig = inspect.signature(precompute_probs.run)
        self.assertEqual(sig.parameters['batch_size'].default, 1)


class ACacheSaysHowItWasScored(unittest.TestCase):
    """A batched cache and an unbatched one are not bit-comparable, and
    nothing else in the file records which it is."""

    def test_the_batch_size_is_written_into_the_cache_metadata(self):
        import precompute_probs
        src = inspect.getsource(precompute_probs.run)
        self.assertIn('predict_batch_size', src)

    def test_it_sits_alongside_the_other_settings_that_change_the_answer(self):
        import precompute_probs
        src = inspect.getsource(precompute_probs.run)
        for key in ('step_s', 'use_ica', 'weights', 'predict_batch_size'):
            self.assertIn(key, src)


class TheFlagExplainsWhatItCosts(unittest.TestCase):
    """An option that silently changes numbers is the failure mode this
    project keeps running into. The help text has to say so."""

    def test_the_help_text_names_the_measured_deviation(self):
        import precompute_probs
        parser = None
        src = inspect.getsource(precompute_probs.main)
        self.assertIn('1.3e-09', src,
                      'the help must quote the measured bound, not just say '
                      '"slightly different"')
        self.assertIn('bit-identical', src)
        del parser

    def test_the_docstring_warns_the_caches_are_not_comparable(self):
        from gui.io import infer_parallel
        doc = infer_parallel.compute_probs_parallel.__doc__ or ''
        self.assertIn('batch_size', doc)
        self.assertIn('bit-identical', doc)
        self.assertTrue(
            'not bit-comparable' in doc or 'not bit-comparable' in
            (infer_parallel.__doc__ or ''),
            'the docstring must say that a batched cache cannot be compared '
            'bit for bit with an unbatched one')


class BatchingPreservesOrderAndSkips(unittest.TestCase):
    """The accumulate-and-flush loop must not reorder results or score a
    window the pipeline refused. Exercised without TensorFlow by replaying the
    same logic over a fake model."""

    def _run(self, codes, batch_size):
        import numpy as np
        probs = np.zeros(len(codes), dtype=np.float32)
        pending_idx, pending_x = [], []

        class FakeModel(object):
            def predict(self, x, verbose=0, batch_size=None):
                n = x.shape[0]
                # second column = the value encoded in the input
                return np.stack([np.zeros(n), x[:, 0, 0, 0, 0]], axis=1)

        model = FakeModel()

        def flush():
            if not pending_idx:
                return
            if len(pending_idx) == 1:
                probs[pending_idx[0]] = float(
                    model.predict(pending_x[0], verbose=0)[0, 1])
            else:
                out = model.predict(np.concatenate(pending_x, axis=0),
                                    batch_size=len(pending_x), verbose=0)
                for k, idx in enumerate(pending_idx):
                    probs[idx] = float(out[k, 1])
            del pending_idx[:]
            del pending_x[:]

        for i, code in enumerate(codes):
            if code == 0:
                x = np.full((1, 1, 1, 1, 1), float(i + 1))
                pending_idx.append(i)
                pending_x.append(x)
                if len(pending_idx) >= batch_size:
                    flush()
        flush()
        return probs

    def test_every_scored_window_gets_its_own_value(self):
        codes = [0] * 10
        for bs in (1, 3, 4, 10, 32):
            got = self._run(codes, bs)
            self.assertEqual(list(got), [float(i + 1) for i in range(10)],
                             'batch_size={} reordered results'.format(bs))

    def test_refused_windows_stay_zero_and_do_not_shift_the_rest(self):
        codes = [0, 1, 0, 2, 0, 0, 3, 0]
        expected = [1.0, 0.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0]
        for bs in (1, 2, 5, 64):
            self.assertEqual(list(self._run(codes, bs)), expected,
                             'batch_size={} mishandled skipped windows'.format(
                                 bs))

    def test_a_trailing_partial_batch_is_still_scored(self):
        """Seven windows at a batch of four leaves three that must not be
        silently dropped — they would read as confident negatives."""
        got = self._run([0] * 7, 4)
        self.assertEqual(list(got), [float(i + 1) for i in range(7)])

    def test_no_scored_windows_at_all_is_survivable(self):
        self.assertEqual(list(self._run([1, 2, 3], 8)), [0.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
