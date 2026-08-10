# Progress Log — 5–9 August 2026

**Reproduction established · scoring corrected · GUI hardened to clinician-demo readiness**

*BMET4111 Thesis — Sam Gan, University of Sydney. Supervisor: Prof. Omid Kavehei.*
Covers commits `35c3d3f` … `f6d6293` (17 commits). Previous log:
[`progress_2026-05-19_zuna_gui.md`](progress_2026-05-19_zuna_gui.md) — **whose numbers are all
superseded by this work.**

> **Numbers in this log are a snapshot.** [`RESULTS.md`](RESULTS.md) is the single source of truth
> and wins on any disagreement.

---

## 1. Headline outcomes

| | Before this period | After |
|---|---|---|
| Reproduction claim | "not a reproduction of the paper" | **AUC 0.89 [0.83, 0.94] vs 0.84 published — statistically indistinguishable** |
| Evaluation set | 26 seizure-enriched files, 1.7 h | 206 annotated files, 27.8 h (99 unannotated excluded) |
| Event FP/24 h @ 0.5 | 328.7 (wrong) | 204.4 |
| Clinician-demo blockers | 5 open | **0 open** |
| Usability evidence | 5 instruments, 0 results | **CW executed, 10 issues, 5 P1** |
| Fresh clone runnable | no | yes |
| Tests | 11 | 65 |

---

## 2. What was done, in order

### 2.1 The reproduction question (5 Aug)

**We were comparing against the wrong paper.** The pretrained weights
`convlstm_ICA_12_train.h5` are 19-channel / 12-second / ICA — the configuration of the **2022
continental-generalization** paper (ESWA 207:118083), not the two-channel SPMB 2020 paper the
docs were measuring against. Read from arXiv:2103.10900v2 Table 2 and Fig. 5, that paper's one
publicly checkable claim is **AUC 0.84 on TUH v1.5.1 dev**, before the PWI/PEI lens. Its
sensitivity and FA/24 h columns for TUH are **blank**, so no event-level number in this project
has a published counterpart. Its RPAH figures need private data and a 20-channel model.

**The apparent shortfall was an evaluation-protocol artefact.** The paper's feature loader
(`utils/ICA_load_data_elec.py:115`) tiles each annotated interval with windows lying *entirely
inside* it, so a boundary-straddling window is never generated. This project labelled any
overlapping window positive. Rescoring the same cached probabilities under the paper's protocol
moved the AUC by ~0.09.

**Sample size did the rest.** On the 26-file manifest the file-level bootstrap CI was
[0.67, 0.93] — too wide to distinguish anything. Precomputing the remaining 266 recordings
(8 parallel shards, ~45 min) narrowed it by more than half.

Deliverables: `experiments/replicate_paper_auc.py`, `experiments/build_full_manifest.py`,
`precompute_probs.py --shard`, `docs/reproduction_status.md`, `docs/paper_comparison.md`.

### 2.2 Scoring defects (5 Aug)

Three defects, all inflating the reported false-alarm rate, none requiring a model change:

1. **The source method's decision stage never ran.** Per-second averaging, concatenate <10 s,
   discard <5 s — implemented in `post_process_code/` as standalone scripts on hardcoded
   `/Users/yikai/…` paths, imported by nothing. Now `gui/postprocess.py`, shared by the GUI,
   `run_inference.py` and the evaluation so they cannot drift.
2. **Fragments of a detected seizure were charged as false alarms.** `match_events` now returns a
   fourth class, `duplicate_indexes`, reported but excluded from FP/24 h.
3. **`run_inference.py` reported event times in window-index units** — 6× compressed, then
   compared against reference seconds.

### 2.3 Purging false numbers (5 Aug)

Created **`docs/RESULTS.md`** as the single source of truth. Removed superseded figures rather
than annotating them; every remaining mention is an explicit correction. Re-scored the ZUNA
comparison from stored caches (`experiments/rescore_zuna_compare.py`): the previously reported
"328.7 → 205.4 FP/24 h" was badly wrong; corrected to 41.1 → 0.0. The window-level AUCs
(0.6878 / 0.6466) were never affected, so **the ZUNA conclusion stands at corrected magnitudes:
inconclusive, and mildly negative on the more rigorous measure.**

Historical documents were bannered rather than deleted, because they back a submitted assessment.

### 2.4 Portability (8 Aug)

A `git clone` was missing everything needed to start. Tracked the weights (4.46 MiB, unmodified
so its sha256 still matches `eval_config.WEIGHTS_SHA256`), `launch_gui.bat` (now discovers the
interpreter instead of hardcoding one machine's path), and the environment files (Windows-only
MSVC pins and the `prefix:` line removed, so they resolve on Linux/macOS). Verified by cloning to
a temp directory and loading the model — 384,846 params, matching the paper.

### 2.5 ICA audit (8 Aug)

Measured, not assumed (`experiments/diag_ica_behaviour.py`). **It faithfully reproduces the
original authors' code, and that code is unsound as ICA:** FastICA fails to converge on 58 % of
windows (20–100 % per recording); the 0.1 Hz pre-filter is unrealisable at this window length
(MNE: `filter_length 8251 > signal 3000`) and below the 1–2 Hz the literature recommends; 19
components from 3000 samples is 2.4× short of the kN² heuristic; half the flagged ocular
components are discarded before removal; and the two return paths preprocess differently.

**Recommendation: report, do not fix.** The training features were generated through this same
function, so the non-convergence is part of the operating point — capping `max_iter` already
flips detections across threshold (one 0.902 → 0.0014).

### 2.6 GUI honesty and safety, B1–B5 (8–9 Aug)

- **B1** — the app never said what it was. Disclaimer now on four surfaces: title, a permanent
  status-bar banner, `# tool =` / `# status =` headers in every exported `.csv_bi`, and
  `not_for_clinical_use: true` in provenance.
- **B2** — two places the UI stated something false. Windows the pipeline *refused* to score
  rendered as confident zeros (one recording is 49/49 skipped and contains a real 27 s seizure);
  they now carry a persisted `skip_code` and render as grey "not assessed" bands. Raw softmax was
  labelled `p(seiz)` and written into the `confidence` column of *human-confirmed* annotations;
  now "model score", with confidence 1.0 and the model score moved to provenance.
- **B3** — opening another file or closing the window silently destroyed a review. Dirty flag,
  `*` in the title, Export/Discard/Cancel guard, and autosave to `.review.autosave.json`.
- **B5** — export could write an empty file asserting "no seizures", and could overwrite the
  gitignored ground truth. Pre-flight dialog plus a hard refusal on the reference path.
- **B4** — resolved incidentally: demo-ready files went from 6 to 25 when the corpus was scored.

### 2.7 Correctness audit and its fallout (8 Aug)

A six-dimension adversarial audit of the above, with every serious finding sent to an independent
verifier. It confirmed the AUC arithmetic (independently re-implemented, matched to 4 dp) and
confirmed the boundary-window exclusion is **faithful, not cherry-picking** — `txt_file/ref_dev.txt`
tiles all 1,013 dev recordings with no gaps, so the source loader could not emit such a window.

It also found real problems, all since fixed:

- **The threshold line was drawn on a curve that does not produce the events** — strip showed
  per-second *max*, events came from per-second *mean*. On 42 of 305 recordings the drawn curve
  crossed the line, up to 0.987, with a necessarily empty worklist.
- **99 of 305 recordings have no `.csv_bi`** and were scored as background — asserting ground
  truth nobody has. At least 6 contain seizures. Now an explicit excluded cohort.
- **"4.5× false-alarm reduction" was the deprecated 26-file figure** and had reached the
  supervisor email draft. The decision stage is **2.1×**.
- **"77 % ICA non-convergence" came from 2 recordings**; over 8 it is 58 %.
- **Inference is not bit-reproducible** despite `random_state=13` — which is why every derived
  number is now quoted to two significant figures.

### 2.8 Cognitive walkthrough (9 Aug)

First usability results: **10 issues, 5 P1, no task blocked**. Failures concentrate in
learnability — controls that work but cannot be found (`J`/`K` invisible, ChannelInspector
double-click-only, no drag affordance, slider direction uncued) — plus one feedback defect with
wide reach (event status is text-only in the EventList). Also found: accepted events share the
ground-truth band's green hue.

Three assumed problems were checked and found **already fixed**, and both instruments were
corrected — the CW claimed 39 cached demo files when only 6 validated, and the heuristic
evaluation's seed rows are now flagged as stale.

---

## 3. Decisions taken, with rationale

Recorded because these are the judgement calls a thesis has to defend and the ones hardest to
reconstruct from a diff later.

| # | Decision | Rationale | Alternative rejected |
|---|---|---|---|
| D-1 | Target the **2022** paper, not the 2020 SPMB one | The weights are 19 ch / 12 s / ICA — that paper's exact configuration | Continue reporting "not a reproduction", which was measuring against a method never run |
| D-2 | Report **both** window protocols, always | The choice is worth ~0.09 AUC, more than most methodological differences | Report only the favourable one |
| D-3 | Claim **indistinguishable from**, not **better than** | 0.84 sits inside the CI; one-sided bootstrap p ≈ 0.06 does not reject it | "Replicates and then some" — an overclaim caught by the audit |
| D-4 | **Exclude** the 99 unannotated recordings | An absent `.csv_bi` is not evidence of a seizure-free recording, and 6 of them contain seizures | Score them as background, inflating the specificity denominator |
| D-5 | **Report** the ICA defects, do not fix them | Training features came through the same function, so the flaws are part of the operating point; changing them flips detections | "Improve" the ICA and silently change the operating point mid-thesis |
| D-6 | Keep **Python 3.6 / TF 1.15** this semester | MNE's ICA is numerically load-bearing; a modern-MNE move changes probabilities by up to 0.90 | Port the runtime now, forcing regeneration of every number during write-up |
| D-7 | Quote **two significant figures** | The CI is ±0.06 wide and inference is not bit-reproducible | Four decimals, inviting a reproducibility challenge that would fail |
| D-8 | Ship the weights **unmodified** (4.46 MiB, not stripped to 1.47) | Preserves the sha256 that ties every result to a specific file | Strip optimizer state, breaking the provenance chain to save 3 MiB |
| D-9 | Scope **ZUNA down** to a documented negative | Event and window metrics disagree; 6× real time, ~42 GiB | Expand it, or headline the single favourable event metric |
| D-10 | C/C++ target is the **ICA front end**, not the ConvLSTM | Profiling: ICA ~90 % of runtime, network ~7 % | Port the network — a ~3 % end-to-end ceiling an examiner would compute in the viva |

---

## 4. Dead ends and false starts

Worth recording — a thesis that only reports what worked is less credible, and re-litigating
these later wastes time.

- **"The pipeline doesn't reproduce the paper."** Wrong target paper. ~11 weeks of framing built
  on it.
- **A dict-unpack "bug" in `compare_zuna.py`** that wasn't — that module defines its own
  tuple-returning `load_probability_file` at line 66. My fix broke it; reverted.
- **`build_full_manifest` duration accounting** — used the `.csv_bi` duration with a 0.0 fallback,
  undercounting the corpus as 28.2 h where the evaluation saw 41.9 h.
- **Capping FastICA `max_iter`** for speed — tested, flips detections, abandoned.
- **The 26-file manifest as the reporting set** — misleading in *both* directions (pessimistic on
  sensitivity, optimistic on false alarms).

---

## 5. State of the workstreams

| WS | Status |
|---|---|
| WS1 Baseline evaluation & rigour | **Largely complete.** Three evaluation views, decision-stage ablation, threshold sweep, bootstrap CIs, ICA study done. Remaining: calibration analysis. |
| WS2 GUI MVP hardening | **B1–B5 closed.** Remaining: the 5 P1 items from the walkthrough (~1 day). |
| WS3 ZUNA | **Closed as a documented negative.** No further compute planned. |
| WS4 Usability evaluation | **CW pass done.** Remaining: heuristic evaluation with 2–3 evaluators; decide the blind-mode question (U-09) before any session used as evidence. |
| WS5 Write-up | Not started. All source material now exists in `docs/`. |
| BMET4112 | Scope set: C++ FastICA front end behind pybind11, with the profile as the opening figure. |

---

## 6. Next actions

1. **The 5 P1 usability items** (~1 day): Help ▸ Keyboard shortcuts dialog; status colour map in
   `event_list.py`; recolour accepted vs reference at `signal_view.py:426-429`; context menu for
   the ChannelInspector; decide blind-mode.
2. **Re-run the CW** afterwards — the before/after collapse in Q2/Q3 failures is a stronger
   thesis result than the issue list alone.
3. **Heuristic evaluation** with 2–3 evaluators, starting from the CW issue list rather than the
   stale seed rows.
4. **Calibration analysis** — the last WS1 gap, and load-bearing for the human-AI trust argument.
5. **"Show your work" features** — spectrogram tab and leave-one-channel-out occlusion. Under a
   supervised-medical framing these stop being optional: *the reviewer must be able to
   independently review the basis* is the criterion the whole design leans on.

---

## 7. Keeping this record

**Convention:** one `docs/progress_YYYY-MM-DD_<topic>.md` per work period, following this shape —
outcomes table, chronological narrative, decisions with rationale, dead ends, workstream status,
next actions. Commit messages carry the detail; this file carries the story.

**Do not** put numbers here that are not in `RESULTS.md`. If they diverge later, `RESULTS.md`
wins and this log is a historical snapshot — the same status the 19 May log now has.
