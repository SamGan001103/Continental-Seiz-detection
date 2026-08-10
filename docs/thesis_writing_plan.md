# Thesis writing plan — every number, figure and claim, ready to write from

*BMET4111 — Sam Gan. Supervisor: Prof. Omid Kavehei.*
Companion to [`RESULTS.md`](RESULTS.md), which remains the single source of truth for numbers.

**How to use this.** Each section states *what to say*, *which numbers*, *where they come from*,
and *what must be caveated*. Sections marked **[DRAFTED]** are factual reporting of what the code
does and can be lifted with light editing. Sections marked **[YOURS]** are argument — write them
yourself, in your voice; the bullets are the skeleton, not the prose.

> **Update your Generative AI Use Statement.** The Progress Report declared Claude for research,
> grammar/concision, and code/figure generation. This project also used it for adversarial
> auditing of results and for drafting technical documentation. Declare that. It is entirely
> permissible and the audits are a strength — but the statement has to match reality.

---

## 0. What changed since the Progress Report — read this first

Six things in the report are now **wrong** and must not be carried forward.

| Progress Report said | Now |
|---|---|
| "not a reproduction of the paper" | **Reproduces it.** Window AUC 0.89 vs 0.84 published |
| Sensitivity 26.3 %, FP/24 h 328.7 | 49.4 % and 204.4 (26-file set was misleading in both directions) |
| ZUNA 26.3 % → 31.6 %, 328.7 → 205.4 | 21.1 % → 26.3 %, 41.1 → 0.0 |
| 26 files, 1.7 h | **206 annotated files, 27.8 h**, 28 patients |
| Window AUC 0.723 | 0.80 (project protocol) / **0.89** (paper protocol) |
| "TUSZ splits are patient-disjoint" | **False for v1.5.x** — see §4.6 |

---

## 1. Introduction  **[YOURS — mostly reusable]**

The Progress Report §1 is sound and can be carried over with three changes:

1. **Aim 1 is now achieved, not attempted.** Say the pipeline was reconstructed *and validated*
   against the source paper.
2. **Add the reproducibility thread.** Two adversarial audits and a scoring-convention finding are
   a genuine contribution, not just hygiene. Foreshadow it.
3. **Sharpen the contribution claim** to three things: (i) a validated reconstruction, (ii) a
   reviewer interface that lets a human *add* what the detector missed, (iii) an honest
   characterisation — calibration, ICA fragility, scoring-convention sensitivity.

---

## 2. Literature Review  **[YOURS — carry over, add three]**

The Progress Report §2 is strong. Add:

- **§2.5 Evaluation metrics** — you now have your own evidence that scoring convention moves a
  headline number more than most methodological differences do (§4.4). Cite Ziyabari and then your
  own result.
- **§2.x Calibration** — Guo et al. 2017 is already cited; add Naeini et al. 2015 and Bröcker 2009
  for the ECE form you actually use, and note that Guo's confidence-vs-accuracy form is
  inappropriate at 5 % prevalence.
- **§2.x ICA data requirements** — Winkler et al. 2015 (1–2 Hz high-pass before ICA), EEGLAB's kN²
  sample-count heuristic. These underpin §5.3.

---

## 3. Methods  **[DRAFTED]**

### 3.1 Detector

19-channel ConvLSTM, pretrained weights `convlstm_ICA_12_train.h5`, **inference only — no
retraining at any point**. Per-window pipeline: EDF → resample 250 Hz → 12-second windows at
6-second stride → per-window ICA (19 components, Fp1/Fp2 EOG rejection) → STFT (250-sample window,
50 % overlap, DC bin removed) → `(23, 19, 125)` tensor → ConvLSTM → per-window score.

Architecture: 3 ConvLSTM blocks (16/32/64 filters) + FC 256 + FC 2, **384,846 parameters**.
Environment: Python 3.6.15, Keras 2.2.5, TF 1.15, MNE 0.19.2.

> **Table 3.1 — configuration match.** Source: `docs/paper_comparison.md` §1. Every row matches
> the paper's §2.3–2.4 except MNE (0.19.2 vs v0.20) and Keras/TF (2.2.5/1.15 vs 2.0/1.4.0).
> **Disclose both** — MNE's ICA is the numerically sensitive stage.

### 3.2 Decision stage

The source method does not threshold raw windows. It (i) collapses overlapping windows to a
per-second mean, (ii) concatenates positive runs < 10 s apart, (iii) discards runs < 5 s. The 5 s
and 10 s constants are the paper's own, from its training corpus.

**None of this was running.** `post_process_code/{overlap,discard,clean}.py` implement parts of it
as standalone scripts on hardcoded paths, imported by nothing. Now `gui/postprocess.py`,
configured from `eval_config.py`, shared by the GUI and the evaluation so they cannot drift.

### 3.3 Data

TUSZ v2.0.0 `eval`. **206 recordings, 28 patients, 27.8 h, 85 seizures** (44 seizure-bearing,
162 background-only).

**99 further cached recordings have no `.csv_bi` and are excluded.** `read_csv_bi` returns `[]`
for a missing file and a seizure-free one alike, so scoring them as background would assert
ground truth nobody has — and ≥6 contain seizures per their per-channel `.csv`. State this; it is
the kind of thing an examiner asks about.

### 3.4 Evaluation protocols

Report **both**, always:

- **Project protocol** — sliding window every 6 s, positive on *any* overlap with a seizure.
- **Paper protocol** — only windows lying entirely inside one annotated interval, reproducing the
  loader at `utils/ICA_load_data_elec.py:115`.

The difference is worth **~0.09 AUC**. It excludes 1.8 % of windows but **36.2 % of positives**
(285 of 788) — state the second figure, not just the first.

Faithfulness evidence: `txt_file/ref_dev.txt` tiles all 1,013 dev recordings with contiguous
intervals and zero gaps, so the source loader could not physically emit a boundary window.

### 3.5 Statistics

Confidence intervals by **cluster bootstrap**, 2,000 resamples. Report the **patient-level**
interval as primary: 206 recordings come from 28 patients and only 13 contribute a positive
window, three supplying 68 %.

Calibration: grouped 5-fold CV **by patient**, out-of-fold. ECE = positive-class reliability form
over 15 equal-mass bins (Naeini 2015; Bröcker 2009) — *not* Guo's confidence-vs-accuracy form,
which is near-meaningless at 5 % prevalence.

---

## 4. Results  **[DRAFTED — tables ready, prose yours]**

### 4.1 Reproduction — the headline

> **Table 4.1** · source `artifacts/.../paper_protocol_auc_full.json`

| protocol | AUC | 95 % CI (patient) | windows | positive |
|---|---|---|---|---|
| project — any-overlap | 0.80 | [0.65, 0.88] | 15,496 | 788 |
| **paper — pure windows** | **0.89** | **[0.72, 0.95]** | 15,211 | 503 |
| paper — non-overlapping | 0.89 | [0.70, 0.95] | 7,678 | 249 |
| *source paper, TUH v1.5.1 dev* | *0.84* | — | — | — |

**Claim:** statistically indistinguishable from the published value. **Not** "better than."

**Two traps to avoid:**
1. Do not call it "the paper's own protocol" — it is the paper's *labelling rule* applied to your
   6 s grid. Their loader steps `i += 12` and anchors at each interval's start; **zero** window
   offsets coincide.
2. Do not quote 15,211 windows beside a 12-second non-overlapping protocol — 15,211 / 27.8 h =
   547/h when the maximum is 300/h. A co-author spots that instantly. Quote 7,678 / 249, or name
   the stride.

**Lead with the robustness**, it is the real result: 0.84 sits inside the interval under
non-overlapping stride, patient clustering, montage reweighting, and dropping the most influential
recording.

### 4.2 Event level

> **Table 4.2 — decision-stage ablation**, threshold 0.5 · `postproc_ablation_full.json`

| configuration | sensitivity | hits | FP/24 h |
|---|---|---|---|
| raw windows | 0.565 | 48/85 | 478.0 |
| \+ event shaping | 0.553 | 47/85 | 314.6 |
| \+ per-second averaging | 0.494 | 42/85 | 237.8 |
| **source method** | **0.494** | **42/85** | **222.9** |

The decision stage — *not the model* — cuts false alarms **2.1×** for 7 points of sensitivity.

> **Table 4.3 — comparison with published TUH results** · `comparable_scoring.json`

| system | scoring | sensitivity | FA/24 h |
|---|---|---|---|
| Shah et al. 2017 (TUH v1.1.0) | OVLP | 39.15 % | 22.83 |
| Golmohammadi et al. 2020 (TUH v1.4.0) | OVLP | 30.83 % | 6.75 |
| **This work** | OVLP | **50.6 %** | 205 |
| **This work** | OVLP + SDR merge | **50.6 %** | **148** |

**You are ahead of both published TUH systems on sensitivity and far behind on false alarms.**
That is the honest headline for this table, and the FA gap has a specific cause (§5.2).

### 4.3 Calibration

> **Figure 4.1** — `reliability.svg` (log x-axis; 13 of 15 bins sit below 0.08)

| | ECE | 95 % CI |
|---|---|---|
| raw | **0.072** | [0.052, 0.092] |
| temperature | 0.069 | [0.049, 0.094] |
| **Platt** | **0.011** | [0.007, 0.028] |
| isotonic | 0.015 | [0.010, 0.031] |

**A raw score of 0.5 means a 29 % chance of ictal, not 50 %.** Platt cuts ECE ~85 %.

Two findings worth prose:
- **Temperature scaling barely helps** despite a stable T ≈ 2.1, because a logit rescale through
  the origin has no intercept to pin the mean to a 5 % base rate.
- **Patient grouping is not optional.** By recording, isotonic looks best (0.0085); by patient it
  degrades to 0.0149 while Platt is flat at 0.011. The apparent advantage was leakage.

Calibration is **reported, not adopted** — thresholding calibrated scores at 0.5 collapses
sensitivity to 0.14, and the fit is prevalence-specific.

### 4.4 Scoring conventions move headline numbers

A result in its own right, and it lands directly on §2.5 of your literature review: the same
probabilities give **0.80 or 0.89** depending on the labelling rule, and **205 or 148** FA/24 h
depending on whether alarms within 30 s are merged. Cite Ziyabari's OVLP-vs-TAES gap and then your
own.

### 4.5 Usability

> **Table 4.4** — cognitive walkthrough, before/after (`cognitive_walkthrough_results.md`)

| | Q1 | Q2 | Q3 | Q4 | total |
|---|:--:|:--:|:--:|:--:|:--:|
| first pass | 3 | 4 | 5 | 5 | **17** |
| second pass | 2 | 1 | 1 | 3 | **7** |

**P1 issues 5 → 0.** The Q2/Q3 collapse is the result: failures were *discoverability*, and the
fixes targeted discoverability. Caveat: single evaluator, code inspection, perceptual rows
unconfirmed.

### 4.6 Corpus integrity  **[important — this is a strength]**

TUSZ v1.5.x splits were **not** patient-disjoint. Verified from the reference lists in this
repository: patients `00001027`, `00001981`, `00004671`, `00006546`, `00009842` appear in **both**
train and dev, carrying **22.4 % of dev seizure events**. NEDC report the same plus 13 subjects
shared between eval and train (Buckwalter et al., IEEE SPMB 2021).

**The 0.84 you are compared against is itself optimistically biased**, making your comparison
conservative. State that disjointness between your eval set and v1.5.1 train is *probable but
unverified*, and that recording-level fingerprinting found no reuse but has no power against
patient-level leakage.

---

## 5. Discussion  **[YOURS — these are the arguments]**

### 5.1 What the reproduction does and does not establish
Validated at the window level, the only level the paper makes a public-data claim. The RPAH
figures are not reproducible — private data, 20-channel model. Corpus version differs.

### 5.2 Why the false-alarm gap exists
Four causes, in order: (i) **the PWA/PEI lens** — their 76.68 % is a *two-stage* system whose
second stage sets thresholds from the **last two hours** of signal, and **none of your 206
recordings exceeds 58 minutes**, so it cannot run; (ii) 20 channels including ECG; (iii) your
corpus is **83× more seizure-dense** (3.05/h vs 0.037/h) — *and the authors themselves write that
TUH "do[es] not provide a realistic specificity test venue"*; (iv) SDR merging.

### 5.3 The detector's limits are structural, not tunable
22 % of seizures produce peak scores below 0.01 — no threshold recovers them. Separation, not
threshold. Plus the ICA analysis: 58 % non-convergence, an unrealisable 0.1 Hz pre-filter, 2.4×
too few samples for 19 components. **Report, do not fix** — training went through the same
function.

### 5.4 What the interface adds, and the honest limit
The reviewer can now originate events (`added` status), which is what makes recovery measurable.
**But be honest:** a human can only recover what they are shown or what they scan. For the 22 %
silent misses, triage does not help unless the reviewer reads unflagged regions — which is what
triage exists to avoid. The paper's 76.68 % → 92.19 % arbiter gain worked because the raw detector
was already at 77 %, not 49 %.

### 5.5 Reproducibility as a contribution
Two adversarial audits caught: a false-positive counting bug (4.5× → 2.1× inflation), a
threshold line drawn on a curve that did not produce the events (42/305 recordings), patient
leakage inverting the calibration recommendation, 99 unannotated files scored as background, and a
false claim about corpus splits. **This is a legitimate methodological contribution** — write it as
one.

### 5.6 Limitations
Inference is not bit-reproducible (`random_state=13` insufficient); single-institution corpus;
seizure-enriched selection; 28 patients; one usability evaluator; no clinician evaluation yet;
calibration is prevalence-specific.

---

## 6. Conclusion  **[YOURS]**
Reconstructed and validated; built and evaluated the interface; characterised honestly; showed the
scoring-convention sensitivity. Future: PWA/PEI lens for false alarms, C++ ICA front end
(BMET4112 — profiling says ICA is ~90 % of runtime, the network ~2 %), clinician evaluation.

---

## 7. Figures to produce

| # | Figure | Source | Status |
|---|---|---|---|
| 3.1 | Pipeline diagram | Progress Report Fig. 2 + ZUNA removed | reuse |
| 4.1 | Reliability diagram | `reliability.svg` | **done** |
| 4.2 | ROC curve | `paper_protocol_auc_full.json` → `roc_curve` | needs plotting |
| 4.3 | Threshold sweep, sens + FA/24 h | `full_scorable.json` | needs plotting |
| 4.4 | Peak-score histogram per seizure | shows the 22 % separation failure | **make this** |
| 4.5 | GUI screenshot, annotated | `aaaaajru_s031_t002` | needs capture |
| 4.6 | Recovery figure | after a review session | **needs a session** |

`seiz36` has no matplotlib. Either `pip install matplotlib` into it, or emit data and plot
elsewhere. Figure 4.4 is the most valuable new one — it makes §5.3 visual.

---

## 8. Suggested order

1. **Methods** (§3) — mostly drafted, gets you moving
2. **Results** (§4) — tables exist; write the connecting prose
3. **Discussion** (§5) — the real intellectual work
4. Introduction and Literature Review updates
5. Conclusion and Abstract last

Do not wait for more code. Everything in §4 is measured and committed.
