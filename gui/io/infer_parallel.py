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
  overhead and is deliberately NOT the default: a batched matmul may reduce in
  a different order, and this project does not ship "should be identical".

  It is now available as an explicit opt-in (`batch_size`), because once the
  ICA is spread across workers the serial predict() becomes roughly three
  quarters of what is left, and that matters for the hardware this is meant to
  run on. Measured per window: 42.3 ms at a batch of one, 7.1 ms at 32, 5.0 ms
  at 48. The price is a real but tiny divergence — see `batch_size` below for
  the measured bound. Off by default, so no cache changes unless someone asks
  for it and records that they did.

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


_WORKER_MODEL = [None]


def _score_one(args):
    """The WHOLE window pipeline in a worker: detect, ICA, STFT, predict.

    The alternative — workers preprocess, parent predicts — puts the two
    stages in series. The parent spends 43 ms predicting each window, during
    which it cannot drain the queue and the workers stall, which is why twelve
    of them bought 1.7x rather than 12x. Threading the parent's predict does
    not help either: measured, `model.predict` holds the interpreter lock and
    only 22 % of it overlaps with other Python work.

    So the prediction moves here. Each worker keeps one model, built once and
    reused for every window it is given, and every call is still a batch of
    ONE on the same code path — which is what keeps this bit-identical rather
    than merely close. The cost is a copy of TensorFlow per worker.

    **Measured, and it is SLOWER. Do not turn this on.** On a 91-window
    recording, against a 314 ms/window serial reference:

        4 workers, predict in parent     118.8 ms/window    2.64x
        4 workers, predict in workers    160.9 ms/window    1.95x
        6 workers, predict in workers    151.0 ms/window    2.08x

    All three are bit-identical, so the option is numerically sound and simply
    does not pay. The reason is thread contention: TensorFlow spawns its own
    intra-op pool in every process, so four workers each running a model
    oversubscribe the machine far worse than one parent doing it, and the ICA
    in those same workers is competing for the same threads. Removing the
    serialisation cost less than the contention it created.

    Kept, documented and off, so the next person to have this idea can read
    the number instead of spending an evening on it.
    """
    i, seg, fs = args
    from utils.preprocessing import detect_interupted_data, ica_arti_remove
    if detect_interupted_data(seg.transpose(), fs):
        return (i, SKIP_INTERRUPTED, 0.0)
    proc = ica_arti_remove(seg, fs, CHANNELS_19)
    if proc is None:
        return (i, SKIP_ICA_FAILED, 0.0)
    if _WORKER_MODEL[0] is None:
        from gui.io.infer import _build_model
        _WORKER_MODEL[0] = _build_model()
    x = np.expand_dims(_calc_stft(proc), -1)
    return (i, SKIP_NONE,
            float(_WORKER_MODEL[0].predict(x, verbose=0)[0, 1]))


def compute_probs_parallel(data, fs, step_s=6, workers=None,
                           progress_cb=None, model=None, batch_size=1,
                           predict_in_workers=False):
    """`compute_probs_from_data`, same numbers, wall-clock divided by ~workers.

    Returns ``(window_starts, probs, skip_code)`` with identical dtypes and
    identical values to the serial function. `workers=None` uses cpu_count-2,
    capped at 16 — the parent needs a core for predict(), and past the memory
    bandwidth the extra workers only queue.

    batch_size
        How many windows to hand `model.predict` at once. **1 is the default
        and the only value that is bit-identical to the serial loop.**

        Anything larger trades that for speed, and the trade is worth
        understanding before taking it. Spreading the ICA across workers does
        not speed up predict(), so once the workers are busy the serial
        predict becomes roughly three quarters of the remaining cost: at 12
        workers this module reached only 1.7x, not 12x. Batching addresses
        exactly that bottleneck — 42.3 ms per window at 1, 7.1 ms at 32,
        5.0 ms at 48.

        What it costs: a batched matmul reduces in a different order, so the
        probabilities move. Measured on real windows, batch 32 against batch 1:
        maximum difference 1.3e-09, and no decision changed at the 0.5
        threshold. For scale, this project accepts 6.8e-09 between TensorFlow
        1.15 and 2.21, and measured 0.138 from BLAS thread count alone. So the
        deviation is small — but it is a deviation, it is opt-in, and a cache
        built with it is not bit-comparable to one built without it.
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
    # np.array on the slice, not the slice itself: `data` may be a memmap (the
    # parent maps a day-long recording rather than holding 1.5 GB of it), and
    # a memmap slice pickled to a worker would carry the mapping with it. One
    # window is 228 KB either way; this just guarantees it is 228 KB of plain
    # bytes. Same values, same dtype.
    tasks = ((i, np.array(data[:, starts[i] * fs:starts[i] * fs + window_len]),
              fs) for i in range(n_full))

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
        if predict_in_workers:
            # Whole pipeline per worker: nothing is serialised in the parent,
            # so the scaling is no longer capped by predict's 43 ms.
            for i, code, p in pool.imap(_score_one, tasks, chunksize=4):
                assert i == done, 'imap broke ordering: got %d at %d' % (
                    i, done)
                skip_code[i] = code
                probs[i] = p
                done += 1
                if progress_cb is not None:
                    progress_cb(done, n_full)
            return np.array(starts, dtype=np.int32), probs, skip_code

        batch_size = max(1, int(batch_size))
        pending_idx = []
        pending_x = []

        def _flush():
            """Score whatever has accumulated. At batch_size 1 this is the
            original call, unchanged, so the default path is untouched."""
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

        for i, code, x in pool.imap(_preprocess_one, tasks, chunksize=4):
            assert i == done, 'imap broke ordering: got %d at %d' % (i, done)
            skip_code[i] = code
            if code == SKIP_NONE:
                pending_idx.append(i)
                pending_x.append(x)
                if len(pending_idx) >= batch_size:
                    _flush()
            done += 1
            if progress_cb is not None:
                progress_cb(done, n_full)
        _flush()
    finally:
        pool.terminate()
        pool.join()

    return np.array(starts, dtype=np.int32), probs, skip_code
