"""Canonical inference and evaluation configuration.

Single source of truth for the operating point used across the whole project.
Before this module existed, run_inference.py (threshold 0.95 / step 1), the GUI
compare path (threshold 0.5 / step 6) and diag_sweep.py disagreed, so "the
baseline number" depended on which script was run. Every script that produces a
reportable number should import these constants instead of hard-coding its own.

Nothing here imports TensorFlow, so it is safe to import anywhere (GUI, CLI,
eval) without pulling in the heavy detector stack.
"""

# --- detector / preprocessing ---
SEGMENT_S = 12          # window length fed to the ConvLSTM (seconds)
STEP_S = 6              # stride between window starts (seconds)
FS = 250               # model sampling rate (Hz)
USE_ICA = True         # per-window ICA EOG removal before STFT
WEIGHTS = 'convlstm_ICA_12_train.h5'   # 19-channel single-scale ConvLSTM weights

# --- event scoring ---
THRESHOLD = 0.5         # canonical operating threshold for event-level reporting
EVENT_TOLERANCE_S = 5.0  # matching tolerance between predicted and reference events

# --- source-method decision stage (see gui/postprocess.py) ---
# The source method does not threshold raw windows. Its decision stage has two
# independent parts, and omitting them was the single largest reason this
# project's event-level false-alarm rate sat far above the published figure.
# Both are kept separately switchable because they behave differently here, and
# experiments/ablate_postprocessing.py measures all four combinations.
#
# Measured on the 26-file manifest at threshold 0.5 (sensitivity, FP/24 h):
#     raw         0.292   171.7      (neither rule — the original behaviour)
#     shape       0.292   143.1      (event shaping only)
#     avg         0.250    57.2      (per-second averaging only)
#     avg+shape   0.250    57.2      (the source method)
#
# EVENT SHAPING concatenates positive runs closer than MAX_MERGE_GAP_S, then
# discards runs shorter than MIN_EVENT_DURATION_S. Both durations are the source
# method's own, derived from its training corpus: shortest seizure ~5 s,
# shortest gap between two seizures ~10 s. It costs no sensitivity and mainly
# removes fragmentation.
#
# PER-SECOND AVERAGING collapses the overlapping windows to one mean value per
# second before thresholding. It does the bulk of the false-alarm suppression
# (171.7 -> 57.2) at a cost of one of 24 reference seizures at this threshold.
#
# IMPORTANT — averaging is stride-dependent, and the effect is strong. At the
# canonical 6 s stride each second is covered by only two 12 s windows, so the
# mean is a mild 2-point smoother and is beneficial. At a 1 s stride each second
# is covered by twelve windows and the mean is dominated by windows that merely
# clip the edge of a seizure: on aaaaatao_s003_t000 the model peaks at p=0.98
# inside the reference seizure, yet stride-1 averaging pulls every second below
# 0.5 and the event vanishes entirely. The source method averaged over 3, 5 and
# 7 second windows, where this dilution is far weaker. So averaging must not be
# enabled at fine strides with these single-scale 12 s weights without re-running
# the ablation. Anything that changes STEP_S should re-check this.
USE_SOURCE_POSTPROCESSING = True   # event shaping (concatenate + discard)
USE_PER_SECOND_AVERAGING = True    # valid at STEP_S = 6; see stride note above
MIN_EVENT_DURATION_S = 5.0   # discard predicted events shorter than this
MAX_MERGE_GAP_S = 10.0       # concatenate predicted events closer than this

# --- threshold sweep grid (high -> low) ---
THRESHOLD_GRID = [0.5, 0.3, 0.1, 0.05, 0.01]


def as_dict():
    """Return the canonical config as a plain dict for provenance stamping."""
    return {
        'segment_s': SEGMENT_S,
        'step_s': STEP_S,
        'fs': FS,
        'use_ica': USE_ICA,
        'weights': WEIGHTS,
        'threshold': THRESHOLD,
        'event_tolerance_s': EVENT_TOLERANCE_S,
        'use_source_postprocessing': USE_SOURCE_POSTPROCESSING,
        'use_per_second_averaging': USE_PER_SECOND_AVERAGING,
        'min_event_duration_s': MIN_EVENT_DURATION_S,
        'max_merge_gap_s': MAX_MERGE_GAP_S,
    }
