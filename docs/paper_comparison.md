# Replication vs. the source paper — side-by-side

*BMET4111 Thesis — Sam Gan, University of Sydney. Supervisor: Prof. Omid Kavehei.*

**Source paper:** Y. Yang, N. D. Truong, C. Maher, A. Nikpour, O. Kavehei,
"Continental generalization of a human-in-the-loop AI system for clinical seizure recognition",
*Expert Systems with Applications* **207**:118083, 2022 (preprint arXiv:2103.10900v2).
Numbers below are from its **Table 2**, **Table 1**, **Fig. 5** and **§2.3–2.7**.

**This work:** inference-only re-run of the same pretrained weights
(`convlstm_ICA_12_train.h5`) on public TUSZ v2.0.0. No retraining, no fine-tuning.

---

## 1. Is it the same detector? — configuration match

| Component | Source paper (§2.3, §2.4) | This work | Match |
|---|---|---|---|
| Channels | 19 | 19 | ✅ |
| Window length | 12 s | 12 s | ✅ |
| Resample rate | 250 Hz | 250 Hz | ✅ |
| Artifact removal | ICA → 19 independent components (BSS) | identical | ✅ |
| EOG rejection | Pearson correlation vs Fp1 and Fp2 | identical | ✅ |
| STFT | 250-sample window, 50 % overlap, DC bin removed | identical | ✅ |
| Input tensor | `(n × 23 × 125)` | `(23, 19, 125)` | ✅ |
| Model | 3 × ConvLSTM (16/32/64) + FC 256 + FC 2 | identical, 384,846 params | ✅ |
| Weights | trained on TUH train split | **the same file, not retrained** | ✅ |
| Python | 3.6 | 3.6.15 | ✅ |
| MNE | v0.20 | **v0.19.2** | ⚠ minor |
| Keras / TF | 2.0 / 1.4.0 | **2.2.5 / 1.15.0** | ⚠ minor |

The pipeline is a faithful reconstruction. The two ⚠ rows are library-version differences that
could not be avoided — the repository's `seiz36` environment is what the weights load under.
They are worth disclosing because MNE's ICA implementation is the numerically sensitive stage
(a 0.19 → 1.12 jump changes probabilities by up to 0.90; a 0.19 → 0.20 jump is far smaller but
was not separately quantified).

---

## 2. Evaluation data

| | Source paper (TUH dev) | This work | Note |
|---|---|---|---|
| Corpus | TUSZ **v1.5.1**, `dev` split | TUSZ **v2.0.0**, `eval` split | different version *and* split |
| Files | 1,013 | 305 | ~30 % as many |
| Files with seizures | 280 | 44 | |
| Seizures | 673 | 85 | |
| Background duration | 154.1 h | 41.1 h | |
| Seizure duration | 16.2 h | 1.16 h | |
| **Total duration** | **170.3 h** | **42.24 h** | ~25 % as much |
| **Background : seizure** | **9.5 : 1** | **35.4 : 1** | **ours is harder** |

The set used here is smaller but proportionally **much more background-heavy**, so it is a
sterner specificity test than the paper's own development split, not a softer one.

---

## 3. The comparable result — window-level AUC on TUH

This is **the only claim in the paper that can be checked against public data.**

| Method | Dataset | Protocol | AUC | 95 % CI |
|---|---|---|---|---|
| **Source paper** | TUH v1.5.1 dev | — (its own feature loader) | **0.84** | not reported |
| **This work** | TUSZ v2.0.0 eval | **paper protocol** (pure windows) | **0.8811** | **[0.820, 0.932]** |
| This work | TUSZ v2.0.0 eval | paper protocol, non-overlapping | 0.8776 | [0.814, 0.930] |
| This work | TUSZ v2.0.0 eval | project protocol (any-overlap) | 0.7941 | [0.738, 0.843] |

**Verdict: replicated.** 0.881 vs 0.84, with the published value inside the interval and
P(AUC ≥ 0.84) = 0.92 under a file-level bootstrap. Measured over 22,803 windows (503 positive)
from 303 files.

Two things had to be right to see this:

1. **Window protocol.** The paper's feature loader
   (`utils/ICA_load_data_elec.py:115-157`) tiles each annotated interval with non-overlapping
   12-second windows that fit **entirely inside** it, so every scored window is unambiguously
   ictal or background. Labelling boundary-straddling windows as seizures instead — as a naive
   sliding-window evaluation does — costs **~0.09 AUC**.
2. **Sample size.** On the original 26-file seizure-enriched subset the same measurement gave
   0.822 with CI [0.673, 0.925] — too wide to distinguish from 0.84 (P = 0.39).

Confidence intervals resample whole **files**, not windows: windows within one recording share
patient, montage and artifact regime, so a window-level interval ([0.759, 0.880] on the subset)
treats correlated windows as independent evidence and is too narrow to quote.

---

## 4. What the paper reports that **cannot** be compared

| Result | Dataset | Value | Why not comparable |
|---|---|---|---|
| AUC | RPAH, 1,006 sessions | 0.82 | private clinical data (ethics X19-0323-2019/STE16040) |
| Sensitivity | RPAH, 1,006 sessions | 76.68 % | private data; **20-channel** model (19 EEG + ECG); PWA/PEI lens; SDR metric |
| FA / 24 h | RPAH, 1,006 sessions | 56.55 | as above; SDR merges alarms within 30 s into one |
| Sensitivity + arbiter | RPAH, 66-session pilot | 92.19 % | private data; requires a human expert arbiter |
| Review time | RPAH, 66-session pilot | 90 → 7.62 min | private data; clinical workflow study |
| **Sensitivity** | **TUH v1.5.1** | **not published** | Table 2 leaves this column blank for TUH |
| **FA / 24 h** | **TUH v1.5.1** | **not published** | Table 2 leaves this column blank for TUH |

> ### ⚠ The 56.55 trap
> Our event-level false-alarm figures land in the same numeric neighbourhood as the paper's
> **56.55 FA/24 h**. **This is coincidence.** Theirs is 14,590 h of private RPAH data through a
> 20-channel model with the PWA/PEI lens, scored by SDR (which merges alarms within 30 s). Ours
> is 42 h of public TUSZ through a 19-channel model with concatenate/discard shaping, scored by
> per-event matching at 5 s tolerance. Different data, model, post-processing, metric, and two
> orders of magnitude of recording time. Never present them as a comparison.

The 2020 SPMB two-channel paper asserts performance "improves dramatically when all 19
electrodes" but publishes **no 19-channel number**, so there is nothing to replicate there either.

---

## 5. Our event-level results — characterisation, not replication

**No published TUH counterpart exists** (§4), so these describe this system only.
305 files, 85 reference seizures, 42.24 h, threshold 0.5.

### Decision-stage ablation

| Configuration | Sensitivity | Hits | FP/24 h | Duplicates |
|---|---|---|---|---|
| raw windows (no decision stage) | 0.553 | 47/85 | 485.4 | 33 |
| \+ event shaping (concat <10 s, discard <5 s) | 0.541 | 46/85 | 319.8 | 11 |
| per-second averaging only | 0.482 | 41/85 | 237.8 | 12 |
| **full source method (averaging + shaping)** | **0.482** | **41/85** | **223.5** | **9** |

Reproducing the paper's *decision stage* — not its model — more than halves the false-alarm
rate, for 7 percentage points of sensitivity.

### Threshold sweep (source method)

| Threshold | Sensitivity | Hits | FP | FP/24 h |
|---|---|---|---|---|
| 0.50 | 0.482 | 41/85 | 390 | 223.5 |
| 0.30 | 0.565 | 48/85 | 570 | 326.7 |
| 0.10 | 0.588 | 50/85 | 695 | 398.3 |
| 0.05 | 0.635 | 54/85 | 781 | 447.6 |
| 0.01 | 0.741 | 63/85 | 967 | 554.2 |

### Reviewer-triage view (threshold 0.5)

| Metric | Value |
|---|---|
| Seizure-file recall | **0.750** (33/44) |
| Background false-flag rate | 0.487 (127/261) |
| Candidate windows | 1,985 over 41.9 h |
| Window ROC-AUC / PR-AUC | 0.794 / 0.201 |

---

## 6. Summary

| Question | Answer |
|---|---|
| Is the pipeline the same detector? | Yes — configuration matches on every axis except two library minor versions. |
| Does the TUH window AUC replicate? | **Yes. 0.881 [0.820, 0.932] vs 0.84 published.** |
| Do the false positives replicate? | **No published TUH false-positive figure exists to replicate.** |
| Does the TUH sensitivity replicate? | **No published TUH sensitivity figure exists to replicate.** |
| Do the RPAH headline figures replicate? | **No, and they never can** — private data, 20-channel model. |
| Does the two-channel SPMB result replicate? | No — that method was never run; its weights are absent. |

**One-line claim the thesis is entitled to make:** *the 19-channel continental-generalization
detector was reconstructed from its published description and reproduces its only publicly
checkable result, window-level AUC 0.881 (95 % CI [0.820, 0.932]) against 0.84 reported, on
42 hours of public TUSZ.*

---

## Regenerating every number above

```bash
python experiments/build_full_manifest.py  --out artifacts/zuna_thesis/manifest_full.csv
python experiments/replicate_paper_auc.py  --manifest artifacts/zuna_thesis/manifest_full.csv
python experiments/evaluate_baseline.py    --manifest artifacts/zuna_thesis/manifest_full.csv --name full303
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest_full.csv
```
