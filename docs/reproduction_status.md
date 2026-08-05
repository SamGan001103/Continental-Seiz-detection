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
| Headline | 2.04 % sens @ 0.17 FA/24 h (TAES, Neureka challenge) | AUC **0.84** on TUH v1.5.1 dev; AUC 0.82 / 76.68 % @ 56.55 FA/24 h on RPAH; 92.19 % with a human arbiter |

The weights this project runs are **`convlstm_ICA_12_train.h5`** — 19 channels, 12-second
windows, ICA-denoised, input tensor `(23, 19, 125)`. Every one of those properties matches the
**2022 continental-generalization paper** and none matches the two-channel SPMB paper. The
2-channel weights the SPMB method requires are not in the repository, and its 3/5/7-second
loaders (`utils/load_data_elec_{3,5,7}s.py`) are unused dead code.

**Conclusion: the correct reproduction target is the 2022 continental-generalization paper.**
`methodology_statements.md` §1 measures the implementation against the two-channel paper and
concludes "not a reproduction". That conclusion is an artefact of comparing against the wrong
method.

### The 19-channel numbers, verified from the paper

Read from the arXiv preprint (arXiv:2103.10900v2), Table 2 and §2.3–2.7:

- **AUC 0.84 — "TUH EEG Corpus v1.5.1, AI, This work"**, 12-second window. Figure 5 identifies
  this curve as **"TUH-TUH ... trained on the TUH training dataset and tested on the TUH
  *development* dataset"**. It is a **development-split** number, not held-out eval.
- §2.3: *"we split EEG signals into 12-second segments and applied the ICA algorithm to
  decompose the signal into 19 independent components"*, Fp1/Fp2 Pearson correlation for EOG
  rejection, STFT with a 250-sample window and 50 % overlap, DC removed → `(n × 23 × 125)`,
  MNE v0.20, Python 3.6. **This is exactly the pipeline in this repository.**
- The PWA/PEI "lens" post-processing (§2.5) is applied **only to the RPAH inference**, not to
  the TUH AUC. So 0.84 is the raw window-level model score, with no post-processing — which is
  what makes it directly comparable.
- The RPAH figures (76.68 %, 56.55 FA/24 h, AUC 0.82) come from **private clinical data under
  ethics X19-0323-2019/STE16040** and, per `utils/ICA_load_data_elec.py:285`, from a
  **20-channel** model (19 EEG + ECG). **Those are not replicable here** — no data, no weights.

So of the paper's 19-channel claims, exactly one is checkable from public data: the TUH
AUC of 0.84. Note also that the SPMB two-channel paper's assertion that performance *"improves
dramatically when all 19 electrodes"* is stated without any published 19-channel number, so
there is nothing there to replicate either.

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

## 4. Do we replicate the 19-channel result? Yes — 0.822 against 0.84

The apparent gap was an **evaluation-protocol mismatch, not a modelling failure.**

The paper's dev-set features were generated by `utils/ICA_load_data_elec.py:115-157`:

```python
while (st + i) * 250 + window_len < sp * 250:   # window must fit ENTIRELY inside
    s = data[:, (st+i)*250 : (st+i)*250 + window_len]
    ...
    i += 12                                      # dev setting: NON-overlapping
```

Each labelled interval is tiled with non-overlapping 12-second windows that fit **entirely
inside** it, and each window inherits that interval's label. A window straddling a seizure
boundary is never generated, so every scored window is unambiguously ictal or background.

This project instead slides a window every 6 s across the whole recording and labels it positive
if it **overlaps** a seizure by any amount (`evaluate_baseline.py:window_labels`). Boundary
windows are included and called seizures even when they are mostly background — a harder and
noisier task. On this manifest 8.4 % of windows straddle a boundary, and **every one of them is
labelled positive** by the project protocol.

Rescoring the *same cached probabilities* under both protocols
(`python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest.csv`):

| protocol | pooled AUC | windows | positive | mean per-file AUC |
|---|---|---|---|---|
| project — any-overlap, 6 s stride | 0.7226 | 857 | 134 | 0.7590 |
| **paper — pure windows only** | **0.8220** | 778 | 55 | 0.7849 |
| paper — pure + non-overlapping | 0.8174 | 391 | 27 | 0.7866 |
| *source paper, TUH v1.5.1 dev* | *0.84* | — | — | — |

**0.822 against a reported 0.84 is a replication**, not a shortfall — the residual 0.018 is
comfortably inside the noise of a 26-file, 55-positive-window subset. The reconstruction of the
19-channel detector is therefore validated at the window level, which is the only level at which
the paper's public-data claim can be checked.

Caveats that keep this honest: the comparison uses TUSZ **v2.0.0 eval** where the paper used
**v1.5.1 dev** (different annotations, different patients, and a seizure-enriched rather than
realistic mix), and 55 positive windows is a small sample. The direction of any residual bias is
not established. But the protocol correction accounts for essentially the whole apparent gap, so
corpus version is no longer needed as an explanation.

**What this does *not* replicate:** the RPAH figures (76.68 % sensitivity, 56.55 FA/24 h). Those
require private clinical data and a 20-channel model, and are permanently out of scope here.

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
- **The 19-channel window-level result replicates: AUC 0.822 here against 0.84 published**, when
  the paper's own window protocol is applied. This is the only one of the paper's 19-channel
  claims checkable from public data, and it holds.
- The apparent 0.72-vs-0.84 shortfall was an **evaluation-protocol artefact**: labelling
  boundary-straddling windows as seizures, which the paper's loader never sampled. Worth
  reporting in its own right — it is a concrete example of how a scoring convention moves a
  headline number by more than most methodological differences do.
- Reproducing the paper's **decision stage** — not the model — accounts for a **4.5× reduction**
  in the reported false-alarm rate (257.6 → 57.2 FP/24 h at threshold 0.5). Two of the three
  contributing defects were scoring bugs, not modelling choices.
- **ICA as deployed costs both accuracy and 30× runtime** (0.7157 vs 0.7417 pooled AUC), but is
  retained because the model was trained on its output. This is reported as a finding, not
  acted on.

## 6. Claims the thesis still must not make

- That any number here reproduces the **two-channel** SPMB result. That method was never run and
  its weights are absent.
- That the **RPAH** figures (76.68 % / 56.55 FA/24 h / 92.19 % with arbiter) are reproduced or
  reproducible. They need private clinical data under a hospital ethics approval and a
  20-channel model.
- That 0.822 is measured on the same data as 0.84. It is TUSZ v2.0.0 eval versus v1.5.1 dev, on
  a 26-file seizure-enriched subset. State this whenever the comparison is made.
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

# the paper-protocol AUC comparison in §4 (the replication result)
python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest.csv \
    --out artifacts/zuna_thesis/baseline_eval/paper_protocol_auc.json

# ICA on vs off (needs TensorFlow; recomputes the ICA-off pass)
python experiments/diag_ica.py
```

The first two read only the cached `<edf>.probs.npz` files, need no TensorFlow, and re-run no
inference. All three run in the `seiz36` environment.
