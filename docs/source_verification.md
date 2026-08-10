# Source verification — every external claim traced to the published paper

*BMET4111 Thesis — Sam Gan. Verified 2026-08-10 against the published PDF.*

Source: Yang, Truong, Maher, Nikpour, **Kavehei**, *Continental generalization of a human-in-the-loop
AI system for clinical seizure recognition*, **Expert Systems With Applications 207 (2022) 118083**.

> **This pass was done against the PDF, not an online rendering.** An earlier pass used automated
> retrieval of the arXiv HTML and produced **two errors of its own**, both recorded below. The
> lesson is in the record: automated extraction of tables and appendices from a rendered preprint
> is not a substitute for reading the paper. **Commit the PDF to the repository.**

---

## 1. Architecture — verified against §2.4.1 and Fig. 4

Every layer, checked by building the model and printing shapes.

| paper (§2.4.1 / Fig. 4) | our model | |
|---|---|---|
| input `23 × 19 × 125` | `(None, 23, 19, 125, 1)` | **match** |
| batch normalisation ("BN" in Fig. 4) | `normal1` | **match** |
| ConvLSTM 1: **16** kernels, `(n×19×3)`, stride `(1×2)` | `filters=16, kernel_size=(19,3), strides=(1,2)` | **match** |
| ConvLSTM 2: **32** kernels, `(1×3)`, stride `(1×2)` | `filters=32, kernel_size=(1,3), strides=(1,2)` | **match** |
| ConvLSTM 3: **64** kernels, `(1×3)`, stride `(1×2)` | `filters=64, kernel_size=(1,3), strides=(1,2)` | **match** |
| Fig. 4 flatten width **896** | `flatten_1 → (None, 896)` | **match** — a very specific number to hit by accident |
| FC **256**, sigmoid | `Dense(256, activation='sigmoid')` | **match** |
| FC **2** (output) | `Dense(2)` → softmax | **match** |
| dropout **0.5** on all FC layers | `Dropout(0.5)` ×2 | **match** |
| Adam, lr **5 × 10⁻⁴** | `Adam(lr=5e-4)` | **match** |

Total parameters **384,846**. The architecture is a faithful implementation.

**One environment deviation, now measured:** the paper states **Keras 2.0 and TensorFlow 1.4.0**;
this project pins **Keras 2.2.5 / TF 1.15**. Tested — max difference **6.8 × 10⁻⁹**. See §8.

## 2. Numbers — verified from Table 3, Table 1, Table 7 and body text

| claim in this repo | paper | |
|---|---|---|
| TUH v1.5.1, this work, AUC **0.84** | Table 3 **and** Table 2; body: "achieving a 0.84 AUROC score (see Table 3)" | **verified** |
| RPAH 1,006 sessions: **0.82**, SDR, **76.68 %**, **56.55** | Table 3; Table 7 "Overall" row | **verified** |
| RPAH 66-session pilot + arbiter: **92.19 %**, **0** FA | Table 3 | **verified** |
| EPILEPSIAE **0.81** | Table 3 | **verified** |
| TUH **dev** split = **170.3 h** | Table 1 | **verified** |
| TUH is ~**83×** more seizure-dense than RPAH | body: "0.038" vs "3.16" seizures/hour → 3.16/0.038 = **83.2** | **verified** — our figure is right |
| SDR "combines the false alarms within 30 s into one" | Table 3, footnote e | **verified** |
| OVLP = "Any Overlap Metric" (Ziyabari et al. 2017) | Table 3 note + footnote 1 | **verified** |
| Fig. 5 TUH-TUH = trained TUH **train**, tested TUH **development** | Fig. 5 caption | **verified** — licenses calling 0.84 a dev-split number |

### Comparator attribution — settled, and our code was right

| dataset | sensitivity | FA/24 h | reference **per Table 3** |
|---|---|---|---|
| TUH v1.1.0 | 39.15 % | 22.83 | **Shah et al. (2017)** |
| TUH v1.4.0 | 30.83 % | 6.75 | **Golmohammadi et al. (2020)** |

This confirms `experiments/comparable_scoring.py`'s **code** and `thesis_writing_plan.md`, and
refutes that file's own **docstring**, which said "Golmohammadi" for both and gave v1.4.**1**.
Fixed. The earlier automated retrieval had reported Table 3's Reference column as "Golmohammadi
et al." for both rows — it was wrong.

## 3. Two errors the automated pass introduced, now corrected

**(a) "The paper provides no ablation that ICA improves performance."** **False.** Appendix B.1:

> "We have trained a model without applying ICA and test on a small scale of the RPAH patient
> (2011), the AUC score for non-ICA VS ICA is **0.8089 vs. 0.8993**"

The retrieval never surfaced the appendix. See `RESULTS.md` §4 for why this does **not** conflict
with our own ICA on/off result — theirs compares *models trained* with and without ICA, ours
removes ICA from *inference only* using ICA-trained weights. Different questions.

**(b) "The lens is PWI/PEI, not PWA/PEI."** **False, and the repo was right before I changed it.**
§2.5:

> "The lens is a real-time signal processing method called **periodic waveform analysis (PWA)**…
> **Periodic energy index (PEI)** and **periodic waveform index (PWI)** values … were calculated"

PWA is the *method*; PEI and PWI are the two *indices*. The paper itself writes "deterministic
methods PWA and PEI". Seven documents were renamed on the strength of a bad retrieval and have
been reverted.

## 4. The lens — full specification, now available (§2.5, Appendix B.3)

More detail than was previously known, and it changes what a reimplementation would have to do:

- It is a **second stage** applied to regions the network already flags at **probability ≥ 10 %**
  — not at 0.5.
- PWI = *E<sub>τ</sub> / N<sub>τ</sub>*, the ratio of total harmonic energy to signal energy;
  PEI = max *E<sub>τ</sub>* over the period. Both computed on the **raw signal**, per band.
- Bands: **0–3, 4–7, 8–12, 13–30, > 31 Hz**.
- Threshold: "the **85-percentile** of PWI and PEI values for each frequency band over the **last
  two hours**".
- **Firing rule: "If the PWI and PEI values are higher than the corresponding adaptive thresholds
  in *all* frequency bands, the period will be reported."** An AND across all five bands and both
  indices — far stricter than a single adaptive threshold, and the detail most likely to be missed
  by a casual reimplementation.

This confirms the `RESULTS.md` §3b analysis: the lens is a *confirmation filter*, and our
`gui/adaptive.py` is explicitly **not** a reimplementation of it.

## 5. Internal inconsistencies in the paper itself

Worth knowing before quoting, since an examiner may hit them:

| | |
|---|---|
| **FA/24 h: 56.22 vs 56.55** | Body §3 says "56.22 false alarms per 24 hours"; the abstract, Table 3 and Table 7 all say **56.55**. Quote **56.55**. |
| **RPAH seizure count: 565 vs 536** | Body §1.3 says "The RPAH dataset has 565 seizures"; Fig. 1(a) says **536**. |
| **"three fully connected layers"** | §2.4.1 says three, then describes **two** ("output sizes of 256 and 2"), which is what Fig. 4 shows and what the code implements. |

## 6. Claims resting on our code, not the paper

| claim | status |
|---|---|
| The RPAH model is **20-channel** (19 EEG + ECG) | **The paper does not state this.** Fig. 3(a) shows a 19-electrode 10–20 layout; Table 4's caption calls EPILEPSIAE a "scalp-EEG (ECG)" dataset. `RESULTS.md` cites `utils/ICA_load_data_elec.py:285` — keep it attributed to the code, never to the publication. |
| Training used the same `ica_arti_remove` as inference | **verified** by test |
| `threshold=2.0` is a z-score, not a Pearson *r* | MNE introspection; paper says only "Pearson correlation" |
| MNE 0.19.2 ≡ 0.20.0, bit-identical | `experiments/diag_mne_version.py`, 25/25 exact |

## 7. Method references — all now checked

Verified 2026-08-10. These justify *our* analytical choices, so an examiner may well check them.

| reference | claim we make | status |
|---|---|---|
| **Bröcker (2009)**, *Reliability, sufficiency, and the decomposition of proper scores*, Q. J. R. Meteorol. Soc. **135**(643):1512–1519, doi:10.1002/qj.456 | cited for the reliability form of ECE | **verified** — exists as cited |
| **Murphy (1973)**, *A new vector partition of the probability score* | Brier decomposition **REL − RES + UNC** | **verified** — the decomposition is exactly Br = REL − RES + UNC |
| **Guo et al.**, *On Calibration of Modern Neural Networks*, arXiv:1706.04599 | the confidence-vs-accuracy ECE form we **reject** | **verified** — this is indeed that form |
| **Winkler et al. (2015)**, *On the influence of high-pass filtering on ICA-based artifact reduction in EEG-ERP*, Proc. IEEE EMBC, pp. 4101–4105 | 1–2 Hz "consistently produced good results in terms of signal-to-noise ratio, single-trial classification accuracy and the percentage of near-dipolar ICA components" | **verified** — wording matches |
| **Wharton et al. (1994)**, *The cognitive walkthrough method: a practitioner's guide* | the four-question instrument used in `docs/usability/` | **verified** — Q1–Q4 match (right goal / action visible / action associated / feedback understood) |
| **Naeini, Cooper & Hauskrecht (2015)**, *Obtaining Well Calibrated Probabilities Using Bayesian Binning*, AAAI | ECE in the **positive-class reliability** form | **partially verified** — the paper exists as cited, but secondary descriptions of its ECE use *confidence-vs-accuracy* language. For a **binary** problem the positive-class form and Naeini's binary definition coincide; the distinction we draw against Guo is defensible but should be checked against Naeini's own equation, not a summary. |
| **kN², k ≥ 20** samples-per-component heuristic | ICA window-length adequacy (`ica_implementation_review.md` §3.4) | **formula verified, the multiplier was not** — see below |

### The one claim that did not survive

The kN² formula is EEGLAB's and is quoted correctly. But **"k ≥ 20" was asserted as though the
source stated it, and the source does not.** EEGLAB says only that *k* "increases with higher
channel counts", and its own worked example implies *k* ≈ 30.

Corrected in §3.4 of the ICA review. **The conclusion is unaffected and if anything strengthened**:
at *k* = 20 the 12-second window is 2.4× short, at *k* ≈ 30 it is 3.6× short. The review now
states the assumed multiplier explicitly rather than hiding it inside a citation.

## 8. Environment deviations from the paper — both now measured

| | paper | this project | measured difference |
|---|---|---|---|
| MNE | v0.20 | 0.19.2 | **bit-identical**, 25/25 exact (`diag_mne_version.py`) |
| Keras / TensorFlow | Keras 2.0, TF 1.4.0 | Keras 2.2.5, TF 1.15 | **max 6.8 × 10⁻⁹**, 13/20 bit-identical, no window crossing the 0.5 threshold (`diag_tf_version.py`) |
| Python | 3.6 | 3.6.15 | match |

Both were installed into isolated `--target` directories, so `seiz36` was never modified. The
Keras/TF test isolates the forward pass: the STFT input tensors are computed **once** in the
current environment and saved, then the identical tensors are loaded and predicted under each
framework version, so nothing but Keras/TF differs between the two runs.

**Neither deviation from the published environment affects any reported number.** This closes
both rows that had been carried as disclosed-but-untested.

---

## Also confirmed

The paper's **code availability** statement points at
`https://github.com/NeuroSyd/Continental-Seiz-detection` — the upstream of this repository. The
implementation being reproduced here is the authors' own released code, which is why the ICA
deviations in §2b of `ica_implementation_review.md` are *the authors' code disagreeing with the
authors' prose*, not a transcription error by this project.
