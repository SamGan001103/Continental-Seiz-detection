# Reproduction status: what matches the source paper, what does not, and why

*BMET4111 Thesis — Sam Gan, University of Sydney. Supervisor: Prof. Omid Kavehei.*
*All numbers below were measured on public TUH/TUSZ v2.0.0 `eval` data. Inference only; no
weights were retrained or modified.*

This document answers one question directly: **does the code in this repository reproduce the
source paper's result?** It supersedes the framing in `methodology_statements.md` §1, which
compared the implementation against the wrong paper.

---

## 1. Which paper is the target

The repository descends from two NeuroSyd publications, and it matters which one the pretrained
weights belong to.

| | Yang et al., SPMB 2020 (two-channel) | Yang et al., *Expert Syst. Appl.* 2022 (continental generalization) |
|---|---|---|
| Channels | 2 (F3–F7, P3–O1) | 19 |
| Window | blended 3 s / 5 s / 7 s | 12 s |
| Artifact removal | not ICA-based | per-window ICA |
| Headline | 2.04 % sens @ 0.17 FA/24 h (TAES, Neureka challenge) | 0.84 AUROC on TUH; 76.68 % @ ~56 FA/24 h on RPAH; 92.19 % with a human arbiter |

The weights this project runs are **`convlstm_ICA_12_train.h5`** — 19 channels, 12-second
windows, ICA-denoised, input tensor `(23, 19, 125)`. Every one of those properties matches the
**2022 continental-generalization paper** and none matches the two-channel SPMB paper. The
2-channel weights the SPMB method requires are not in the repository, and its 3/5/7-second
loaders (`utils/load_data_elec_{3,5,7}s.py`) are unused dead code.

**Conclusion: the correct reproduction target is the 2022 continental-generalization paper.**
`methodology_statements.md` §1 currently measures the implementation against the two-channel
paper and concludes "not a reproduction". That conclusion is an artefact of comparing against
the wrong method; against the right one the project is a genuine partial reproduction with a
quantifiable gap.

> The 0.84 AUROC figure is taken from secondary summaries of the paper, not from the PDF, which
> is paywalled on ScienceDirect and 403s on bioRxiv. **Verify it against the published PDF
> before quoting it in the thesis**, and note which split it refers to (train/dev/eval).

---

## 2. What is faithful

Checked line by line against the training code that produced the weights:

- **STFT is byte-identical.** `gui/io/infer.py:_calc_stft` and the training-time
  `utils/ICA_load_data_elec.py:calc_stft` perform the same operations in the same order:
  `stft.spectrogram(framelength=250, centered=False)` → transpose `(1,2,0)` → `abs + 1e-6` →
  drop the DC bin → `log10` → clamp negatives to 0. No divergence.
- **Channel montage and order match.** Both resolve the same 19 channels in the same order via
  `params_common_electrodes.txt`.
- **Window geometry matches.** 12 s at 250 Hz = 3000 samples → 23 STFT frames × 19 channels ×
  125 bins, which is exactly the saved model's input tensor. The model loads with no shape
  mismatch, no padding workaround, and no silent reshaping.
- **ICA is applied at inference exactly as at training** — a fresh 19-component FastICA fitted
  per 12-second window, with Fp1/Fp2 EOG-correlation component rejection.

So the detector is not misconfigured. Divergence in the numbers is not caused by a broken
inference path.

---

## 3. What was wrong, and what fixing it changed

Three defects were found and fixed. All three inflated the reported false-alarm rate; none
required touching the model.

### 3.1 The source method's decision stage was never implemented

The source method does not threshold raw windows. It (i) collapses the overlapping window
probabilities to a per-second mean, (ii) concatenates positive runs less than 10 s apart, and
(iii) discards runs shorter than 5 s. The 5 s and 10 s constants are the paper's own, derived
from its training corpus.

None of this ran. `post_process_code/{overlap,discard,clean}.py` implement parts of it but are
standalone scripts operating on hardcoded `/Users/yikai/...` text files, and nothing in the GUI
or the evaluation path imports them. The decision stage now lives in **`gui/postprocess.py`**,
is configured from `eval_config.py`, and is shared by the GUI, `run_inference.py` and
`experiments/evaluate_baseline.py` so they cannot drift apart again.

### 3.2 Fragments of a detected seizure were counted as false alarms

`experiments/compare_zuna.py:match_events` matched one prediction per reference, then charged
**every** remaining prediction as a false positive — including predictions landing inside a
seizure that had already been matched. A single sub-threshold window in the middle of a seizure
splits it in two, and the second half was booked as a false alarm against a seizure the detector
had in fact found.

`match_events` now returns a fourth class, `duplicate_indexes`: predictions that overlap an
already-claimed reference. They are reported (a reviewer really does have to dismiss them) but
excluded from FP/24 h. At threshold 0.5, 6 of 18 "false positives" across the manifest were
fragments of correctly detected seizures.

### 3.3 `run_inference.py` reported event times in window-index units

`merge_events` built events from the window-index array, so with the canonical 6-second stride
every printed event time was 6× compressed and was then compared against reference times in
seconds — making the script's own hit/false-positive tally meaningless. It now shares
`gui/postprocess.events_from_probs`. A relative `--file` path was also being re-resolved under
`utils/` after the `chdir`, producing a confusing "failed to open" on a path the user never
typed; it is now resolved before the `chdir`.

### Measured effect

26 files, same cached probabilities, threshold 0.5
(`python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest.csv`):

| configuration | sensitivity | FP/24 h | duplicates |
|---|---|---|---|
| as originally reported (fragments counted as FPs) | 0.292 | **257.6** | — |
| raw windows, FP-counting fixed | 0.292 | 171.7 | 6 |
| \+ event shaping (concatenate <10 s, discard <5 s) | 0.292 | 143.1 | 2 |
| \+ per-second averaging | 0.250 | 57.2 | 0 |
| **the source method (averaging + shaping)** | **0.250** | **57.2** | **0** |

**The reported false-alarm rate falls 4.5× (257.6 → 57.2 per 24 h) with no change to the model.**
Event shaping is free — it costs no sensitivity. Per-second averaging does the bulk of the
suppression but costs one of 24 reference seizures at this threshold.

For a triage tool a missed seizure is worse than an extra candidate the reviewer dismisses, so
`USE_PER_SECOND_AVERAGING = False` (shaping only: 0.292 / 143.1) is a defensible operating point
for the thesis even though it is less faithful to the paper. Both are one-line switches in
`eval_config.py` and the ablation script regenerates the table.

**Averaging is stride-dependent and this is a real constraint.** At the canonical 6 s stride each
second is covered by only two 12 s windows, so the mean is a mild smoother. At a 1 s stride each
second is covered by twelve windows and the mean is dominated by windows that merely clip the
edge of a seizure: on `aaaaatao_s003_t000` the model peaks at p = 0.98 *inside* the reference
seizure, yet stride-1 averaging pulls every second below 0.5 and the event vanishes entirely.
The source method averaged over 3/5/7-second windows where this dilution is far weaker. Any
change to `STEP_S` must re-run the ablation.

---

## 4. What still does not match, and why

After the fixes, the honest position is a **partial reproduction with a quantified gap**.

**Window level.** Pooled ROC-AUC over the 26-file manifest is **0.723** (857 windows, 134
positive; PR-AUC 0.430) against the paper's reported **0.84** on TUH. Gap ≈ 0.12.

Four candidate explanations, none yet isolated:

1. **Corpus version.** The weights were trained on TUSZ v1.5.1; this evaluation uses v2.0.0
   `eval`. v2.0.0 re-annotated and re-partitioned the corpus, so the label conventions and the
   patient split both differ. This is the most likely single contributor and is not fixable
   without v1.5.1.
2. **Subset size and selection.** 26 seizure-enriched files is a small, non-random slice of a
   ~1000-file eval split. The confidence interval on 0.723 is wide.
3. **Which split the 0.84 refers to.** If it is a dev/validation figure rather than held-out
   eval, the comparison is not like-for-like. Resolve this from the PDF.
4. **ICA.** See below.

**ICA is currently costing accuracy.** Comparing the cached ICA-on probabilities against a fresh
ICA-off pass over the 13 seizure files (`python experiments/diag_ica.py`):

| | pooled window ROC-AUC |
|---|---|
| ICA on (as trained, as deployed) | 0.7157 |
| ICA off | **0.7417** |

Turning ICA **off** improves discrimination slightly while running ~30× faster. This is
consistent with the fragility the progress report flagged: MNE emits
`filter_length (8251) is longer than the signal (3000), distortion is likely` on every window,
because a 0.1 Hz high-pass on a 12-second segment is not physically meaningful, and a
19-component FastICA fitted to 3000 samples is under-determined.

**Do not "fix" this by removing or restructuring ICA.** The features the model was trained on
were produced by this same non-converged per-window ICA (`utils/ICA_load_data_elec.py:131`), so
the non-convergence is baked into the operating point. Independent testing showed that capping
`max_iter` at 50 flipped 2 of 25 windows across the 0.5 threshold, one confident detection going
p = 0.902 → 0.0014. Fitting ICA once per recording instead of per window is 9.6× faster but
correlates only r = 0.66 with the deployed configuration. Any ICA change must ship with an
AUC/sensitivity delta from `evaluate_baseline.py`. The ICA-on/ICA-off comparison is a legitimate
thesis result; silently switching the deployed configuration is not.

**A methodological caveat that bounds everything above.** One file in the manifest,
`aaaaaqtw_s002_t012`, has all 49 of its windows rejected by `detect_interupted_data` and scores
0.0 throughout, despite containing a 27-second reference seizure. Those windows are excluded
from the AUC (correctly — they measure preprocessing drop-out, not model quality) but they are
guaranteed misses at the event level. The skip rate should be reported alongside sensitivity.

---

## 5. Claims the thesis can now make

- The inference pipeline is a **faithful reconstruction** of the 2022 continental-generalization
  detector: identical STFT, montage, window geometry, and ICA procedure, with the pretrained
  weights loading and running as trained.
- Reproducing the paper's **decision stage** — not the model — accounts for a **4.5× reduction**
  in the reported false-alarm rate (257.6 → 57.2 FP/24 h at threshold 0.5). Two of the three
  contributing defects were scoring bugs, not modelling choices.
- The remaining window-level gap is **0.723 vs 0.84 AUROC**, with corpus version (v2.0.0 vs
  v1.5.1) the leading candidate explanation and the subset size a close second.
- **ICA as deployed costs both accuracy and 30× runtime** (0.7157 vs 0.7417 pooled AUC), but is
  retained because the model was trained on its output. This is reported as a finding, not
  acted on.

## 6. Claims the thesis still must not make

- That any number here reproduces the **two-channel** SPMB result. That method was never run and
  its weights are absent.
- That 0.84 is a verified comparison point until it is read out of the published PDF, with its
  split identified.
- That the displayed probabilities are calibrated. They remain raw softmax.
- That 57.2 FP/24 h is a clinical false-alarm rate. It comes from 26 short, seizure-enriched
  TUSZ clips totalling ~1.7 h and is not a deployment figure.

---

## Reproducing these numbers

```bash
# window-level AUC + event sweep + review simulation, all three views
python experiments/evaluate_baseline.py --manifest artifacts/zuna_thesis/manifest.csv \
    --name baseline26 --out artifacts/zuna_thesis/baseline_eval/baseline26.json

# the four-way decision-stage ablation in §3
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest.csv \
    --out artifacts/zuna_thesis/baseline_eval/postproc_ablation.json

# ICA on vs off (needs TensorFlow; recomputes the ICA-off pass)
python experiments/diag_ica.py
```

The first two read only the cached `<edf>.probs.npz` files, need no TensorFlow, and re-run no
inference. All three run in the `seiz36` environment.
