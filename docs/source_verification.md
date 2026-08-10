# Source verification — what is checked against the papers, and what is not

*BMET4111 Thesis — Sam Gan. Verification pass 2026-08-10.*

Every externally-sourced number in this repository, traced to its source and marked
**VERIFIED**, **UNVERIFIED**, or **CONFLICTING**. The purpose is that no claim reaches the
thesis whose provenance nobody has checked.

> ### The single biggest gap: no papers are in the repository
>
> The repo contains only the two two-channel SPMB PDFs and the ZUNA preprint. It does **not**
> contain the source paper — Yang et al. 2022 — on which the entire reproduction argument rests.
> Every check below was therefore done against the arXiv rendering, over the network, and cannot
> be repeated offline or audited by an examiner working from this repository alone.
>
> **Action: commit a PDF of arXiv:2103.10900 and of each comparator paper.** Until that is done,
> the rows marked CONFLICTING below cannot be closed.

---

## 1. Verified — consistent across independent retrievals

These were extracted twice, in separate queries, and agreed both times.

| claim in this repo | source | status |
|---|---|---|
| ICA: "12-second segments", "19 independent components", BSS approach | Yang §2.3 | **VERIFIED** verbatim |
| ICA: eye movement "detected from two EEG channels, namely 'FP1' and 'FP2'" | Yang §2.3 | **VERIFIED** verbatim |
| ICA: "We remove **those** independent sources" (plural) | Yang §2.3 | **VERIFIED** verbatim — the basis of deviation row 5 |
| ICA: "implemented in Python 3.6 with the use of library **MNE v0.20**" | Yang §2.3 | **VERIFIED** verbatim |
| STFT: "window length of 250 (or 1 second) and 50 % overlapping" | Yang §2.3 | **VERIFIED** verbatim |
| STFT: "remove the DC component", shape "(n×23×125)" | Yang §2.3 | **VERIFIED** verbatim |
| The paper offers **no ablation or evidence** that ICA improves performance | Yang, whole text | **VERIFIED** — only a general motivating sentence exists |
| RPAH 1,006 sessions: **76.68 %**, **56.55** FA/24 h | Yang Table 2 | **VERIFIED** |
| RPAH 66-session pilot + human arbiter: **92.19 %**, **0** FA | Yang Table 2 | **VERIFIED** |
| RPAH AUC **0.82** | Yang, body text + Fig. 5 | **VERIFIED** — "An inference of the method on 14,590 hours of RPAH set achieved an AUC of 0.82 (Fig. 5)" |
| Fig. 5 "TUH-TUH" = trained on TUH **train**, tested on TUH **development** | Yang Fig. 5 caption | **VERIFIED** verbatim — this is what licenses calling 0.84 a *dev-split* number |
| SDR "combines the false alarms within 30 seconds into one" | Yang, footnote | **VERIFIED** — the basis for refusing to compare our FA/24 h to their 56.55 |
| PWI/PEI lens: "the 85-percentile of PWI and PEI values for each frequency band over the **last two hours** as adaptive thresholds" | Yang §2.5 | **VERIFIED** verbatim |
| The lens is **PWI**/PEI, not "PWA"/PEI | Yang §2.5 | **VERIFIED** — six documents in this repo had the wrong name; corrected |

## 2. Conflicting — do not cite until resolved from a PDF

Two independent retrievals of the *same* arXiv rendering returned **contradictory** answers.
Automated extraction is not reliable here, and no correction has been made on the strength of it.

| question | retrieval A | retrieval B | consequence |
|---|---|---|---|
| Does Table 2 have an **AUC column**? | column list ends `… Reference, Sensitivity, FA/24 hours` — **no AUC** | column list includes `… Seizure length, AUC, Evaluation method, …` — **has AUC** | `RESULTS.md` §1 presents its table as "Table 2" *including* an AUC column. If retrieval A is right, that citation is wrong and 0.84 belongs to Fig. 5 / body text. |
| Where does **0.84** appear? | body text, Fig. 5 | body text **and Table 2**; one quoted sentence says "achieving a 0.84 AUROC score (**see Table 3**)" | The paper may itself point at Table 3, not Table 2. |

**Resolution required:** open the PDF and read Table 2's header row. This is a two-minute check for
someone holding the paper, and it cannot be done reliably any other way. **Omid Kavehei is an
author** (Yang, Truong, Maher, Nikpour, Kavehei) — the fastest route is to ask.

Until then, `RESULTS.md` §1 should say "Table 2 and Fig. 5" rather than "Table 2" alone, and the
0.84 should be attributed to the text.

## 3. Attribution inconsistency inside this repository

`experiments/comparable_scoring.py` disagrees with **itself**:

| | TUH v1.1.0 · 39.15 % · 22.83 | TUH v1.4.x · 30.83 % · 6.75 |
|---|---|---|
| its own docstring (lines 9–10) | "Golmohammadi et al." | "Golmohammadi et al." |
| its own code (lines 56–57) | **"Shah et al. 2017"** | "Golmohammadi et al. 2020" |
| `docs/thesis_writing_plan.md` | "Shah et al. 2017" | "Golmohammadi et al. 2020 (TUH **v1.4.0**)" |
| Yang et al. Table 2 | "Golmohammadi et al." | "Golmohammadi et al." (TUH **v1.4.1**) |

External evidence is genuinely split. A literature search surfaced *"Shah et al. … reported the
best results of **39 %** on using all 22 channels"* — consistent with 39.15 % being Shah — and
*"Golmohammadi … delivers **30 %** sensitivity at **7** false alarms per 24 hours"* — consistent
with 30.83 % / 6.75 being Golmohammadi. So the code's labelling may well be more accurate than
the source paper's own table.

**Not silently "fixed" in either direction.** Both are plausible and the thesis will be cited
against whichever is written. Resolve by reading the two comparator papers directly, then make
the docstring, the code and the writing plan agree. Note also the **v1.4.0 vs v1.4.1**
discrepancy, which is a separate error in `thesis_writing_plan.md`.

## 4. Verified from this repository's own code, not from the paper

Claims where the *evidence is our code*, and which the paper does not state. These are honest as
long as they are attributed to the code and not to the publication.

| claim | evidence | note |
|---|---|---|
| The RPAH model is **20-channel** (19 EEG + ECG) | `utils/ICA_load_data_elec.py:285` | The paper does **not** state the RPAH channel count. `RESULTS.md` §1 already cites the code line rather than the paper — keep it that way. |
| Training features came from the same `ica_arti_remove` as inference | `utils/ICA_load_data_elec.py:15` imports it; `None` → skip handled identically | **VERIFIED** by test (`tests/test_paper_conformance.py`) |
| `threshold=2.0` is a **z-score**, not a Pearson *r* | MNE 0.19.2 `find_bads_eog` introspection | The paper says only "Pearson correlation" |
| `find_bads_eog` band-passes the channel **1–10 Hz** first | MNE defaults | Not mentioned in the paper |
| MNE **0.19.2 and 0.20.0 are bit-identical** on this pipeline | `experiments/diag_mne_version.py`, 25/25 exact | So the version deviation from the paper is numerically immaterial |

## 5. Method references cited in this repository

| reference | used for | status |
|---|---|---|
| Naeini et al. 2015; Bröcker 2009 | ECE in positive-class reliability form | cited in `experiments/calibration.py`; **not independently re-checked** |
| Guo et al. | the confidence-vs-accuracy ECE form we *reject* | **not independently re-checked** |
| Murphy | Brier REL − RES + UNC decomposition | **not independently re-checked** |
| Winkler et al. 2015 | 1–2 Hz as the standard ICA pre-filter | quoted in `ica_implementation_review.md`; **not independently re-checked** |
| Wharton et al. 1994 | cognitive walkthrough method | **not independently re-checked** |
| kN², k ≥ 20 samples-per-component heuristic | ICA window-length adequacy | **not independently re-checked** — this one underpins §3.4 and should be |

**These are the next verification target.** They are used to *justify method choices*, so an
examiner may well check them, and none has been read in this pass.

---

## What this pass changed

Nothing in the numbers. It changed what is *claimed to be known*: the §1 rows are now backed by
verbatim quotes retrieved twice, the §2 rows are flagged as unresolved rather than presented as
fact, and §3 documents an internal contradiction that had been sitting in the code and the
writing plan simultaneously.
