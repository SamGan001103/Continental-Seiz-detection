# Reproduction vs. the source paper — side-by-side

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
| Files scored | 1,013 | 206 | 99 further cached files have no annotation and are excluded |
| Files with seizures | 280 | 44 | |
| Seizures | 673 | 85 | |
| Total duration | 170.3 h | 27.8 h | ~16 % as much |

The paper-side column is from its Table 1 (train/dev summary); cite that table directly if the
figures are reproduced in the thesis.

**On excluded files.** 99 of the 305 locally cached recordings have no `.csv_bi`. `read_csv_bi`
returns `[]` for a missing file and for a seizure-free one alike, so scoring them as background
would assert ground truth nobody has — and at least 6 of them contain seizures according to
their per-channel `.csv`. They are excluded from every number in this document.

---

## 3. The comparable result — window-level AUC on TUH

This is **the only claim in the paper that can be checked against public data.**

| Method | Dataset | Protocol | AUC | 95 % CI |
|---|---|---|---|---|
| **Source paper** | TUH v1.5.1 dev | its own feature loader | **0.84** | not reported |
| **This work** | TUSZ v2.0.0 eval | **paper protocol** (pure windows) | **0.89** | **[0.83, 0.94]** |
| This work | TUSZ v2.0.0 eval | paper protocol, non-overlapping | 0.89 | [0.82, 0.94] |
| This work | TUSZ v2.0.0 eval | project protocol (any-overlap) | 0.80 | [0.74, 0.85] |

**Verdict: reproduced.** 0.84 lies inside the interval, and a one-sided file-level bootstrap
gives p ≈ 0.06 — which does not reject the published value. The honest claim is **statistically
indistinguishable from the published result**, not better than it. Measured over 15,211 windows
(503 positive) from 206 recordings.

Two things had to be right to see this:

1. **Window protocol.** The paper's loader (`utils/ICA_load_data_elec.py:115`) tiles each
   annotated interval with windows that fit **entirely inside** it. Labelling boundary-straddling
   windows as seizures instead — as a naive sliding-window evaluation does — costs **~0.09 AUC**.
   That exclusion drops 1.8 % of all windows but **36.2 % of the positive class** (285 of 788);
   state that figure, because quoting only the 1.8 % understates it badly. It is nonetheless
   faithful rather than selective: `txt_file/ref_dev.txt` tiles all 1,013 dev recordings with
   contiguous intervals and no gaps, so the source loop could not emit a boundary window.
2. **Sample size.** On the original 26-file seizure-enriched subset the same measurement gave
   0.82 with CI [0.67, 0.93] — too wide to distinguish anything.

Confidence intervals resample whole **files**, not windows: windows within one recording share
patient, montage and artifact regime.

---

## 4. What the paper reports that **cannot** be compared

| Result | Dataset | Value | Why not comparable |
|---|---|---|---|
| AUC | RPAH, 1,006 sessions | 0.82 | private clinical data (ethics X19-0323-2019/STE16040) |
| Sensitivity | RPAH, 1,006 sessions | 76.68 % | private data; **20-channel** model (19 EEG + ECG); PWI/PEI lens; SDR metric |
| FA / 24 h | RPAH, 1,006 sessions | 56.55 | as above; SDR merges alarms within 30 s into one |
| Sensitivity + arbiter | RPAH, 66-session pilot | 92.19 % | private data; requires a human expert arbiter |
| Review time | RPAH, 66-session pilot | 90 → 7.62 min | private data; clinical workflow study |
| **Sensitivity** | **TUH v1.5.1** | **not published** | Table 2 leaves this column blank for TUH |
| **FA / 24 h** | **TUH v1.5.1** | **not published** | Table 2 leaves this column blank for TUH |

> ### ⚠ The 56.55 trap
> Our event-level false-alarm figures land in the same numeric neighbourhood as the paper's
> **56.55 FA/24 h**. **This is coincidence.** Theirs is 14,590 h of private RPAH data through a
> 20-channel model with the PWI/PEI lens, scored by SDR (which merges alarms within 30 s). Ours
> is 27.8 h of public TUSZ through a 19-channel model with concatenate/discard shaping, scored
> by per-event matching at 5 s tolerance. Different data, model, post-processing, metric, and
> about two and a half orders of magnitude of recording time. Never present them as a comparison.

The 2020 SPMB two-channel paper asserts performance "improves dramatically when all 19
electrodes" but publishes **no 19-channel number**, so there is nothing to replicate there either.

---

## 5. Our event-level results — characterisation, not replication

**No published TUH counterpart exists** (§4), so these describe this system only.
206 annotated files, 85 reference seizures, 27.8 h, threshold 0.5.

### Decision-stage ablation

| Configuration | Sensitivity | Hits | FP/24 h | Duplicates |
|---|---|---|---|---|
| raw windows (no decision stage) | 0.565 | 48/85 | 478.0 | 32 |
| \+ event shaping (concat <10 s, discard <5 s) | 0.553 | 47/85 | 314.6 | 11 |
| per-second averaging only | 0.494 | 42/85 | 237.8 | 13 |
| **full source method (averaging + shaping)** | **0.494** | **42/85** | **222.9** | **10** |

Reproducing the paper's *decision stage* — not its model — cuts the false-alarm rate **2.1×**
for 7 percentage points of sensitivity.

### Threshold sweep (source method)

| Threshold | Sensitivity | Hits | FP | FP/24 h |
|---|---|---|---|---|
| 0.50 | 0.494 | 42/85 | 237 | 204.4 |
| 0.30 | 0.565 | 48/85 | 352 | 303.6 |
| 0.10 | 0.588 | 50/85 | 457 | 394.2 |
| 0.05 | 0.647 | 55/85 | 522 | 450.2 |
| 0.01 | 0.718 | 61/85 | 650 | 560.6 |

### Reviewer-triage view (threshold 0.5)

| Metric | Value |
|---|---|
| Seizure-file recall | **0.750** (33/44) |
| Background false-flag rate | 0.451 (73/162) |
| Candidate windows | 1,235 over 27.8 h |
| Window ROC-AUC / PR-AUC | 0.80 / 0.39 |

---

## 6. Summary

| Question | Answer |
|---|---|
| Is the pipeline the same detector? | Yes — configuration matches on every axis except two library minor versions. |
| Does the TUH window AUC reproduce? | **Yes — 0.89 [0.83, 0.94] vs 0.84 published; indistinguishable, not better.** |
| Do the false positives replicate? | **No published TUH false-positive figure exists to replicate.** |
| Does the TUH sensitivity replicate? | **No published TUH sensitivity figure exists to replicate.** |
| Do the RPAH headline figures replicate? | **No, and they never can** — private data, 20-channel model. |
| Does the two-channel SPMB result replicate? | No — that method was never run; its weights are absent. |

**One-line claim the thesis is entitled to make:** *the 19-channel continental-generalization
detector was reconstructed from its published description and reproduces its only publicly
checkable result: window-level AUC 0.89 (95 % CI [0.83, 0.94]) against 0.84 reported, on
27.8 hours of public TUSZ, a difference not statistically distinguishable from zero.*

---

## Regenerating every number above

```bash
python experiments/build_full_manifest.py  --out artifacts/zuna_thesis/manifest_full.csv
python experiments/replicate_paper_auc.py  --manifest artifacts/zuna_thesis/manifest_full.csv
python experiments/evaluate_baseline.py    --manifest artifacts/zuna_thesis/manifest_full.csv --name full_scorable
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest_full.csv
```
