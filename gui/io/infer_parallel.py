"""Score a recording with the per-window preprocessing fanned across cores.

Why this exists
---------------
A day of ambulatory EEG is ~14,400 twelve-second windows, and the pipeline
spends 90 % of its time inside `ica_arti_remove` — measured at 440 ms per
window against 46 ms for the model and 3 ms for the STFT. Scored serially that
is just under two hours per day-long recording, which turns "open the file"
into "come back after lunch".

The artefact removal cannot be made faster by changing it: it is the authors'
code, the released weights were fitted to signals cleaned by exactly this
routine, and this project has already measured what a silently substituted
numeric stack does to the probabilities (docs/known_issues.md §1). What CAN
change is the schedule. Each window's ICA is a pure function of that window's
samples — `utils/preprocessing.ica_arti_remove` builds a fresh
`ICA(n_components=19, random_state=13)` per call, so there is no cross-window
state and no unseeded randomness. Running the windows on 20 cores instead of
one changes which core computes each number and nothing else.

What this module promises
-------------------------
Bit-identical output to `compute_probs_from_data`. Not "close", identical:
`tests/test_infer_parallel.py` asserts `np.array_equal` on probs and skip
codes against the serial path, and the first real recording scored both ways
(ma0844az, 302 windows, a Nihon Kohden export) matched bitwise.

Three decisions keep that promise cheap to trust:

* The serial loop in `gui/io/infer.py` is untouched. This module reuses its
  per-window pieces (`detect_interupted_data`, `ica_arti_remove`,
  `_calc_stft`) rather than re-implementing anything.

* Workers do preprocessing only. `model.predict` stays in the parent, one
  window at a time, in ascending index order — exactly the serial call
  pattern. Batching windows through predict() would be ~10× less TensorFlow
  overhead and is deliberately NOT done: a batched matmul may reduce in a
  different order, and this project does not ship "should be identical".

* Workers never import TensorFlow. `gui.io.infer` defers that import into
  `_build_model`, so a worker importing this module pays for numpy, MNE and
  sklearn — a few seconds once per worker, amortised over thousands of
  windows.

* Workers keep the parent's BLAS threading. Pinning them to one thread each
  is 8.8× instead of 3.3× and was rejected because it CHANGES THE OUTPUT:
  measured max |Δp| of 0.138 against the serial reference, thread count
  alone, one machine. The full measurement is at the pool-creation site
  below. 3.3× is what bit-identical costs on a 24-core machine, and that is
  the price this project pays.

The trailing-short-window rule is reproduced from the serial loop: the first
window with fewer than SEGMENT_S*fs samples marks itself and every later
window SKIP_SHORT, and none of them are dispatched.

Frozen builds
-------------
Not wired into the GUI yet. multiprocessing under PyInstaller on Windows
means freeze_support() and spawn re-exec of the bundled executable, which is
its own project; until that is done and gated, this is a CLI capability
(`precompute_probs.py --workers N`) for preparing caches ahead of a review
session. The GUI already loads those caches instantly.
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.io.infer import (_calc_stft, CHANNELS_19, SEGMENT_S,  # noqa: E402
                          SKIP_ICA_FAILED, SKIP_INTERRUPTED, SKIP_NONE,
                          SKIP_SHORT, TARGET_FS)


def _preprocess_one(args):
    """detect -> ICA -> STFT for one window. Runs in a worker process.

    Returns (index, skip_code, stft_or_none). Top-level function because
    Windows multiprocessing uses spawn, which pickles by qualified name.
    """
    i, seg, fs = args
    from utils.preprocessing import detect_interupted_data, ica_arti_remove
    if detect_interupted_data(seg.transpose(), fs):
        return (i, SKIP_INTERRUPTED, None)
    proc = ica_arti_remove(seg, fs, CHANNELS_19)
    if proc is None:
        return (i, SKIP_ICA_FAILED, None)
    x = _calc_stft(proc)
    return (i, SKIP_NONE, np.expand_dims(x, -1))


def _silence_worker():
    """The per-window pipeline prints MNE's running commentary; one worker
    doing that is noisy, twenty interleaving are unreadable."""
    sys.stdout = open(os.devnull, 'w')


def compute_probs_parallel(data, fs, step_s=6, workers=None,
                           progress_cb=None, model=None):
    """`compute_probs_from_data`, same numbers, wall-clock divided by ~workers.

    Returns ``(window_starts, probs, skip_code)`` with identical dtypes and
    identical values to the serial function. `workers=None` uses cpu_count-2,
    capped at 16 — the parent needs a core for predict(), and past the memory
    bandwidth the extra workers only queue.
    """
    import multiprocessing

    # Windows multiprocessing is spawn-only: every worker re-imports the
    # parent's __main__ module. A parent whose __main__ is not an importable
    # file — piped stdin, a REPL, `python -c` — cannot be re-imported, and
    # what that looks like in practice is not an error but a HANG: each
    # worker dies at startup with `OSError: Invalid argument: '<stdin>'`,
    # imap waits forever for results that will never come, and the caller
    # stares at a silent ten-minute timeout. Measured, not imagined — the
    # first benchmark of this module did exactly that. Refuse up front with
    # an explanation instead.
    main_mod = sys.modules.get('__main__')
    main_file = getattr(main_mod, '__file__', None)
    if sys.platform.startswith('win') and (
            main_file is None or not os.path.exists(main_file)):
        raise RuntimeError(
            'compute_probs_parallel needs an importable __main__ module: '
            'Windows spawn re-imports the calling script in every worker, '
            'and this process was started from {} which cannot be '
            're-imported (workers would crash and the pool would hang). '
            'Run from a .py file, or use the serial '
            'compute_probs_from_data.'.format(
                '<stdin>/REPL' if main_file is None else repr(main_file)))

    if model is None:
        from gui.io.infer import _build_model
        model = _build_model()
    data = np.asarray(data, dtype=np.float32)
    fs = int(round(float(fs)))
    if data.ndim != 2 or data.shape[0] != len(CHANNELS_19):
        raise RuntimeError(
            'Model inference requires shape (19, samples), got {}'.format(
                data.shape))
    if fs != TARGET_FS:
        raise RuntimeError(
            'Model inference requires {} Hz data, got {} Hz'.format(
                TARGET_FS, fs))

    duration_s = data.shape[1] / float(fs)
    window_len = SEGMENT_S * fs
    total_s = int(duration_s)
    starts = list(range(0, total_s - SEGMENT_S + 1, step_s))
    probs = np.zeros(len(starts), dtype=np.float32)
    skip_code = np.zeros(len(starts), dtype=np.int8)

    # The serial loop marks the first short window AND everything after it
    # SKIP_SHORT and stops. Shape checks are free, so resolve that boundary
    # up front and dispatch only full windows.
    n_full = len(starts)
    for i, t in enumerate(starts):
        if data[:, t * fs:t * fs + window_len].shape[1] != window_len:
            n_full = i
            skip_code[i:] = SKIP_SHORT
            break

    if workers is None:
        workers = max(1, min(16, (os.cpu_count() or 4) - 2))

    # A generator, consumed lazily: Pool.imap pulls tasks as workers free up,
    # so at no point is more than ~workers*chunksize windows' worth of segment
    # data pickled and in flight. concurrent.futures' Executor.map was
    # rejected here because it drains the whole iterable up front — for a
    # day-long recording that is ~3 GB of queued segments, in the exact
    # scenario this module exists to make tractable.
    tasks = ((i, data[:, starts[i] * fs:starts[i] * fs + window_len], fs)
             for i in range(n_full))

    done = 0
    # Workers inherit the parent's BLAS threading untouched. DO NOT pin them
    # to one thread, tempting as it is: it was tried, it is 8.8x instead of
    # 3.3x, and it CHANGES THE NUMBERS. Measured on ma0844az (302 windows),
    # single-threaded workers against the multi-threaded serial reference:
    #
    #     probs bitwise equal : False
    #     max |delta|         : 0.138            (at window 58)
    #
    # The BLAS thread count changes reduction order somewhere inside
    # FastICA's linear algebra, the never-converging ICA amplifies the
    # low-order-bit difference chaotically, and 0.138 in p(seizure) crosses
    # decision thresholds. This is the same amplification mechanism
    # documented across machines in docs/known_issues.md 4, reproduced on one
    # machine by a thread count alone — worth knowing in its own right.
    #
    # With inherited threading the same run IS bitwise identical, at 3.3x for
    # 20 workers on 24 cores (the workers contend for cores, which is where
    # the other 6x goes). This project ships identical, not fast-but-close.
    pool = multiprocessing.Pool(processes=workers,
                                initializer=_silence_worker)
    try:
        # imap (not imap_unordered): results arrive in index order, so
        # predict() runs in ascending order — the serial loop's exact call
        # pattern. Workers stay ahead of the parent by a bounded prefetch.
        for i, code, x in pool.imap(_preprocess_one, tasks, chunksize=4):
            assert i == done, 'imap broke ordering: got %d at %d' % (i, done)
            skip_code[i] = code
            if code == SKIP_NONE:
                probs[i] = float(model.predict(x, verbose=0)[0, 1])
            done += 1
            if progress_cb is not None:
                progress_cb(done, n_full)
    finally:
        pool.terminate()
        pool.join()

    return np.array(starts, dtype=np.int32), probs, skip_code
