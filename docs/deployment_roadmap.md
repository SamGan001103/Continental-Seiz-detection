# ROADMAP — from "code that works on Sam's laptop" to "prototype a clinician can be shown"

> **Note on the numbers quoted below.** This roadmap was written from an audit run *before* the
> scoring fixes landed, so figures such as "26.3 % event sensitivity", "257.6 fp/24h", "171.7",
> and "window AUC of 0.723" are the pre-fix values. Findings **R1** and **R2** in §3 are now
> **fixed** (commit `35c3d3f`); the AUC is **0.881** under the source paper's window protocol (305 files).
> The engineering recommendations and the priority ordering are unaffected — only the illustrative
> numbers. Current values: `docs/RESULTS.md`.

Five audits ran; roughly a third of the raised findings collapsed into six root causes, and several headline claims were refuted under verification. What follows is deduplicated, sequenced by dependency, and pruned to items that are either *a clinician cannot use it without this* or *a thesis number depends on this*.

The good news up front, because it changes the plan: **the review loop works.** `merge_review_state` (gui/events.py:102-144), the cache round-trip, source switching, and threshold-rebuild state preservation were all traced and found correct. The GUI is not architecturally broken. What stands between today and a clinician session is roughly two weeks of small, specific changes — not a rewrite, not a port, and definitely not C++.

---

## 1. The single most important thing

**Send the ethics pathway email this week.** `docs/usability/ethics_pathway_email.md:3` is still stamped "DRAFT for review. Replace placeholder fields before sending", with `<STUDENT_EMAIL>`/`<SUPERVISOR_EMAIL>` unfilled at :8-9 and :45.

This is the only item on the entire board with **external latency**. Every other task — code, docs, evaluation — completes on your own clock. The ethics reply does not, and `peer_review_protocol.md:8` and `:185` hard-gate *all* data collection behind it. If the reply takes three weeks and you send the email in week four, you have lost three weeks that cost you one hour of typing.

Add one paragraph the current draft lacks: ask explicitly whether a **clinician engaged as an expert consultant giving feedback on an artefact** is treated differently from a **participant**. Your own documents already contain the argument (`heuristic_evaluation.md:39-46`: experts are "analysts, not research participants"; `cognitive_walkthrough.md:12`: "no human-research ethics burden") — you just have not asked whether it extends to clinicians. That single question determines whether a clinician session can be quoted in the thesis at all.

**In the same week, run the cognitive walkthrough solo.** It needs no ethics (`cognitive_walkthrough.md:12`), needs one evaluator (`:32`), and every scoring table in it is blank (`:60-66`, `:85-91`, `:110-117`, `:136-142`, `:161-168`, `:189-197`). You have five well-written instruments and zero results. The CW is the ungated one, it generates the real pre-clinician fix list, and it is itself a thesis chapter. Commit it as `docs/usability/cognitive_walkthrough_results.md`.

One caution while you run it: the pre-filled example rows in `heuristic_evaluation.md:70-79` are seeds, not findings, and at least two are already **factually wrong against current code** — `:75` claims the threshold value is not shown as text, but `app.py:254` creates `thr_lbl` and `app.py:1002` updates it; `:74` claims export is silent, but `app.py:1139-1141` shows a status message. Do not carry those seeds forward as results.

---

## 2. Clinician-demo blockers

These five must land before any external person sits in front of the app. All are hours-to-a-day.

**B1 — The app never says what it is.** The only identity string is `setWindowTitle('Seizure Review — Continental Human-AI')` (app.py:78), replaced by the filename at `:339`. There is no menu bar, no About, no disclaimer: a grep of the whole `gui/` package for `disclaim|not for diagnostic|research|prototype` returns two hits, neither a disclaimer. Meanwhile the tool has 26.3% event sensitivity at threshold 0.5 (`docs/methodology_statements.md:26`). Fix: suffix both `setWindowTitle` calls; add a permanent `statusBar().addPermanentWidget` label reading *RESEARCH PROTOTYPE — NOT FOR DIAGNOSTIC USE* next to the existing hover label (app.py:151); add `# tool = ...` / `# status = research prototype, not for clinical use` header lines in `write_csv_bi` (csv_bi.py:37-43); add a constant `not_for_clinical_use: true` field to the provenance payload (app.py:1149). Note `read_csv_bi:15` already skips `#` lines, so the header addition is format-safe.

**B2 — Two places the UI states something that is false.** These are the same defect wearing two hats, and both are squarely on the thesis's own subject matter.

*(a) Windows the pipeline refused to score render as confident negatives.* `gui/io/infer.py:134-138` and `:139-145` write `probs[i] = 0.0` for interrupted data and ICA failure; `cache.py:87-92` stores no status; `prob_strip.py:76-78` draws them at zero with `fillLevel=0.0`. Empirical: 136 of 3352 cached windows are exactly 0.0, **concentrated** — `aaaaaqtw_s002_t012.probs.npz` is 49/49 skipped, and its reference file contains `TERM,271.8395,299.0000,seiz,1.0000`, a real 27-second seizure. That recording renders as a perfectly flat zero strip with an empty event list. Fix: return a `scored` mask from `compute_probs_from_data`, persist it in the npz, bump `cache_version` (cache.py:38, 83), render unscored spans as grey hatch with a "not assessed" tooltip, and report the skip count in the status bar and provenance. Note `evaluate_baseline.py:26-31` **already documents this exact sentinel** and excludes those windows from AUC — the GUI is the only consumer that reads 0.0 as a real score.

*(b) Raw softmax is presented as probability and exported as "confidence".* `prob_strip.py:51` labels the axis `p(seiz)`; `event_list.py:9` heads the column `p`; `app.py:1131` writes `'confidence': ev['prob']` into the csv_bi. There is no calibration code anywhere in `gui/` (the `temp = 1.0` Lambda in `deep_conv_lstm.py:84` is the identity). Worse: export only emits **accepted and edited** events (`app.py:1126`), so the model's score is being written into the confidence field of a *human-confirmed* annotation, where TUSZ references carry 1.0000 — and `csv_bi.py:25` reads it back. Fix: relabel to "model score (uncalibrated)" / "score" with a tooltip; write `1.0` in the confidence column for human-confirmed events and move the model score into the provenance ledger. Your own `cognitive_walkthrough.md:258` already pre-wrote this fix.

**B3 — A whole session is destroyed by a routine click.** `load_edf` does `self._events = []` / `self._events_by_source = {}` (app.py:332-334) with no dirty check; `closeEvent` (:283-295) prompts only about ZUNA. Nothing reaches disk until the manual export. The machinery to fix this already exists — `_activate_prob_source` stashes and restores per source (app.py:462, 473) and threshold rebuilds pass `preserve_review=True` (`:1003`) — file-open and close are the only paths that bypass it. Fix: dirty flag set in `_on_accept`/`_on_reject`/`_on_region_edited`, three-button Export/Discard/Cancel guard at `app.py:310` and `:284`, plus `<edf>.review.autosave.json` on each status change.

**B4 — The frozen demo set does not exist.** `peer_review_protocol.md:159` says "There are 39 such cached sample EDFs" and warns "**Do not** use non-cached files during a session." Running the repo's own `load_probs` over all 306 sample EDFs returns **6 usable caches**; 34 of the 40 are silently rejected at `cache.py:61-62` for missing `edf_size`/`edf_mtime`. A facilitator following the protocol will most likely pick a file that silently re-runs full inference (~91 s of ICA alone on a 554-window file, with MNE `ConvergenceWarning` and "Reconstructing data from ICA components" spraying the console). Fix: `precompute_probs.py --overwrite` over five chosen files; assert `load_probs()` is non-None after every write so it cannot regress; create `demo/manifest.json` and point `app.py:301` at it. Choose files that (i) have a validated cache, (ii) produce events at threshold 0.5, (iii) contain a reference seizure, and (iv) are **not** all-skipped.

**B5 — Export has no guardrails.** `_export_reviewed` writes unconditionally (app.py:1124-1136), so zero accepted events produces a valid header-only `.csv_bi` asserting "no seizures", indistinguishable downstream from a careful negative read. Worse, the save filter is `*.csv_bi` (app.py:1119-1121) and the ground truth read on every open is `<stem>.csv_bi` in the same directory (app.py:342) — the reference file is *listed and clickable* in the save dialog, and `sample_data/` is gitignored so there is no recovery. Fix: pre-flight dialog with counts ("N accepted + M edited; K proposed unreviewed will be omitted"); hard-refuse a target path equal to the reference; show the written path.

### 2b. Thesis-result blockers (same urgency, different reason)

**R1 — The headline false-alarm number is wrong.** `compare_zuna.match_events` (`:133`, `:144-147`) allows one prediction per reference and charges every further prediction landing on an already-claimed reference as a false positive. Measured on the committed manifest: 18 FPs → 257.6 fp/24h; excluding within-seizure fragments, 171.7. Re-scoring the ten files behind the quoted "328.7 → 205.4 false positives per 24 hours" (`methodology_statements.md:26`, `:61`) gives roughly **164.3 → 123.3, and the baseline-vs-ZUNA delta collapses from 123.3 to ~41**. Fix: keep one-pred-per-ref for hits/sensitivity; define false positives as predictions matching *no* reference at all; report `n_duplicate_detections` separately.

**R2 — The root cause of R1.** `build_proposed_events` (gui/events.py:75) closes an event on the first sub-threshold window — no gap tolerance, no minimum duration. Verified: probs `[0.9, 0.1, 0.9, 0.1]` at step 6 yields two *abutting* events. The paper's own concatenate/discard stage exists (`post_process_code/discard.py:31-35`, merge if gap ≤10 s, drop if <5 s) and never runs. Fix R1 and R2 together, add `merge_gap_s`/`min_duration_s` threaded from `eval_config`, re-run the evaluation, and regenerate every event-level number in `methodology_statements.md`.

**R3 — `run_inference.py` prints times that are 6× wrong.** `merge_events` (`:85-107`) emits raw window indices, `:153` feeds it window-indexed probs, and `:181` prints them as seconds. Windows 20-23 at step 6 (true span 120-150 s) print as `20 - 24s`. The tally at `:235-236` compares indices against reference seconds, so a true hit is counted as *both* a miss and a false positive. Do not paste a single number from this script into the thesis until it is fixed (scale by `step_s`, or delete `merge_events` and call `gui.events.build_proposed_events`).

**R4/R5 — The operating point is not actually enforced.** `evaluate_baseline.py:128` bypasses `load_probs` (skipping the staleness check) and never reads `loaded['meta']`, then stamps `cfg.as_dict()` at `:344` — so a cache made with `--no-ica` pools into the reported AUC and the output JSON asserts a config it was not produced under. Ironically `compare_zuna` has a `validate_step` guard and `evaluate_baseline` imports from that very module without using it. Separately, `eval_config` is imported by only two of the five scripts the README says read it. Nothing is numerically wrong *today* — every hardcoded value happens to match — but this is exactly the drift the file was created to prevent. Fix both; correct the README sentence.

**R6 — Disclose the AUC's denominator.** `evaluate_baseline.py:140` reconstructs the skip set as `probs == 0.0` and excludes it. Those are disproportionately artefact-heavy windows, which are also disproportionately ictal, so the reported window AUC of 0.723 is measured on a cleaned subset. Once B2(a) gives you a real mask, use it instead of the value test, report `n_excluded`, and say so in the methodology.

---

## 3. Portability plan

Sequenced; the whole thing is about three days.

1. **`git add` the untracked files.** `launch_gui.bat`, `environment-seiz36.yml`, `requirements-seiz36.txt`, `eval_config.py` and all of `docs/usability/` are untracked. A literal `git clone` today receives neither the launcher the README tells you to double-click (README.md:20) nor the manifests it points at (`:35`). One commit.
2. **Ship the weights.** `.gitignore:136` is `*.h5` — and `git blame` shows Sam added it in the same commit that made `convlstm_ICA_12_train.h5` the runtime dependency. `gui/io/infer.py:14`, `run_inference.py:57`, `precompute_probs.py:30`, `eval_config.py:18` all hardcode the filename with no env override and no fetch script. The file is 4.5 MiB, well under GitHub's limit; two-thirds is discardable Adam state (`optimizer_weights/m_*,v_*,vhat_*`). Add `!convlstm_ICA_12_train.h5`, or strip to ~1.5 MB. Add its sha256 to `eval_config.py` and stamp it into provenance.
3. **Launcher discovery.** `launch_gui.bat:8` hardcodes `C:\Users\User\miniconda3\...`; README.md:23 and :25 repeat it. Fix: honour `%SEIZ36_PYTHON%`, then `%CONDA_PREFIX%`, then `%USERPROFILE%\miniconda3\envs\seiz36`, then PATH, keeping the existing (good) error message at `:10-18`. Replace README:23-25 with `conda env create -f environment-seiz36.yml` → `conda activate seiz36` → `python -m gui.main`.
4. **Env file.** Drop `environment-seiz36.yml:10-14` (five win-64-only MSVC packages) and the `prefix:` line at `:52`. Document `requirements-seiz36.txt` as the non-Windows route — it was verified to be an identical 35-package set, so a Linux/Intel-Mac install already works today, it is just undocumented.
5. **Cache portability.** `load_probs` validates on size + mtime with a 1.0 s tolerance (cache.py:59-69). Explorer copy and xcopy preserve mtime; `shutil.copy` and Windows zip do not — zip round-trip measured a 1.56 s delta because DOS timestamps are 2-second granular, so USB/zip handover to a reviewer fails non-deterministically. Fix: validate on `edf_size` + `sha256_file` (already written at cache.py:12, used only for provenance), treat mtime as advisory, and widen the tolerance to ≥2 s. While you are there: actually compare `edf_basename` (written at `:43`, never read) and compare `meta` against `cfg.as_dict()`.
6. **Kill the chdirs.** Root cause is `utils/pyst.py:606`'s bare relative `parameters="params_common_electrodes.txt"`. Give it an absolute default, then delete `os.chdir` at `gui/io/edf.py:29`, `gui/io/infer.py:39` (that one is dead weight — `models/` is at repo root and REPO is already on `sys.path`; the comment about "keras import paths" is false), and `run_inference.py:201` — which also fixes the relative `--file` bug at `:209`.

**Test that closes the workstream:** clone to a second Windows account (or a VM) and run the frozen demo start to finish without editing a file. That test *is* the deliverable; write it up.

**On the Python-3.10 port — be honest and defer it.** The forward pass ports cleanly: 384,846 parameters, 12 standard layers, an independently-rebuilt numpy implementation matched TF 1.15 to 2.7e-7 on real ICA'd windows, and `stft` 0.5.2 needs only `np.lib.pad = np.pad`. But the *pipeline* does not port: running the real `compute_probs_from_data` on 400 s of TUSZ in both environments gave max abs diff 5.4e-7 with `use_ica=False` and **0.90 with `use_ica=True`** — 21 of 65 windows moved by >0.01 and one detection fell from p=0.902 to p=0.0014, because MNE 1.12's ICA unmixing differs from 0.19.2's. Migrating the runtime this semester means regenerating every cached prob and re-validating every reported number mid-write-up. **Build the numpy forward pass as a validated reference implementation** — it is the right C++ target and a genuine artefact — but leave `seiz36` as the runtime and write the port up as future work.

---

## 4. GUI improvements, ranked

Only after §2. Ranked by value per hour, and deliberately short.

1. **Sensitivity default.** `signal_view.py:376` hard-clips at ±`_sensitivity_uv` with a 30 µV/div default (`:198`). Ictal discharges and sharp waves flat-top. (Contested detail: the audit's "wastes 2/3 of the row" is wrong — pitch is 3×sens and the clip is 1×, so headroom to the collision limit is 1.5×; and an unclipped view already exists in ChannelInspector.) So this is a **one-line default change**, not architecture: raise to 70 µV/div, or clip at `1.4 * sensitivity`. It changes every thesis screenshot. Do it first.
2. **Help ▸ About + F1 shortcut dialog + colour legend strip.** ~40 lines. There is no menu bar at all, and 13 of ~14 accelerators are documented only in the module docstring at `app.py:7-15`. Reuse that text verbatim.
3. **Spectrogram tab in ChannelInspector.** `channel_inspector.py:159` plots a time-domain trace and `:238-241` reports RMS/peak — while the model consumes a 125-bin log-STFT (`infer.py:23-34`) the reviewer never sees. `scipy.signal.spectrogram` over the visible slice is hours of work and is the single cheapest thing that moves the tool from "AI gives a number" toward "AI shows its evidence".
4. **Leave-one-channel-out occlusion attribution.** Re-score a flagged window 19 times with one channel zeroed, render Δp as a bar or head-map. Inference-only, no retraining, ~19 × 5.6 ms ≈ 0.1 s per event. This directly answers the clinician's first question ("which channels?") and is the strongest available *thesis* contribution in this list. Build it if week 7 has room; do not let it displace §2.
5. **`ChannelInspector.set_data()`** instead of `insp.close()` at `app.py:1069`, and delete the dead `x0, x1 = current_span()` at `:1068` whose result is never used. Today every open detail window vanishes on any filter or montage change — precisely when a reviewer is checking whether something is muscle artefact.
6. **Reviewer reason field.** `review_note` is threaded through rebuilds at `events.py:129` and *never written by anything*. Add reason chips (artefact / muscle / wrong extent / unsure) plus free text, and persist per-event into the provenance ledger. This is what makes a clinician session quotable rather than a binary vector — but it is not needed for the non-clinical peer study, so schedule it just before clinician contact.
7. **Provenance completeness.** `app.py:1149-1169` records no UTC timestamp, no git SHA, no reviewer id, no weights hash, and no ledger of rejected/unreviewed events — while `peer_review_protocol.md:74` asks the facilitator to log a build/commit the system gives them no way to obtain. Ten lines: `git rev-parse HEAD` in a try/except, `datetime.utcnow().isoformat()`, a reviewer string prompted once per session, `weights_sha256`, `references_visible`, and an all-four-statuses event list.
8. **Reference bands.** Accepted events are `(40,170,90,80)` and references `(50,160,70,40)` — same hue (`signal_view.py:427` vs `:400`). Recolour accepted to a non-green. On defaulting `chk_refs` off: contested — the walkthrough's own Tasks 2 and 3 *require* the reference to be locatable, and the peer protocol scripts the accept/reject actions, so contamination is prospective rather than actual. Recommended compromise: keep the default, add an explicit blind-mode toggle, and record `references_visible` in provenance so any session run with truth on screen is permanently flagged.

---

## 5. C/C++ scope for BMET4112

**Say the uncomfortable thing plainly: the expected target is the wrong one.** Measured independently three times in the real `seiz36` environment on real TUSZ windows:

| stage | per window | share |
|---|---|---|
| `ica_arti_remove` (utils/preprocessing.py:64-110) | 174–332 ms | **95–97%** |
| `model.predict` (infer.py:150) | 4.0–6.9 ms | ~2% |
| `_calc_stft` (infer.py:23-34) | 3.0–3.4 ms | ~2% |

A perfect, zero-execution-time C++ ConvLSTM and STFT yields a **2.7–4.1% end-to-end improvement**. An examiner will do that arithmetic in the viva. Do not build a semester project around it and call it a speedup.

The defensible scope, in order:

**Step 0 — before any C++ (days, belongs to 4111's tail).** Parallelise the window loop in `gui/io/infer.py:130`. Iterations are independent apart from the `probs` slot. Refactor the body into a module-level picklable `_score_window(args)` doing `detect_interupted_data` + `ica_arti_remove` + `_calc_stft` — **numpy/mne only, no TF, because a Keras model is unpicklable under Windows spawn semantics** — then `multiprocessing.Pool(min(cpu_count()-1, 8))`, and a single batched `model.predict` in the parent. Near-linear on an ICA-bound workload across the 24 available cores. Add `multiprocessing.freeze_support()` if the app is ever frozen. This is the honest biggest-win-per-hour and it must precede the port, or the port's benchmark baseline is dishonest.

**Step 1 — the 4112 core.** A C++ reimplementation of the per-window preprocessing front end behind a pybind11 binding: covariance estimation and whitening (eigendecomposition), the FastICA parallel fixed-point iteration with the logcosh nonlinearity, EOG-correlation component selection, and back-projection. Optionally fold in the framed STFT since it is adjacent and trivial. Deliverables: a benchmark table against `mne` 0.19's FastICA on the sample corpus, **a numerical-agreement study proving identical component selection, not just identical speed**, and a section explaining why the profile made this the only stage worth porting.

**Step 2 — stretch, and frame it correctly.** The validated numpy forward pass becomes the reference for a small C++ ConvLSTM inference kernel. Present it as a **portability / embedded-target contribution** and state the 4% ceiling out loud. That framing is far more defensible than pretending it is an optimisation, and it earns credit for having profiled before porting.

**One contested caution to carry into 4112.** Do not assume ICA can be "fixed" algorithmically first. Capping `max_iter` at 50 was tested: it changed 7 of 25 window probabilities and flipped 2 across the 0.5 threshold, including a confident detection going 0.902 → 0.0014. The reason is that `utils/ICA_load_data_elec.py:131,247` generated the *training* features through this same non-converged per-window ICA with no `max_iter` override — the non-convergence is baked into the operating point. Block-level ICA fitting carries the identical train/test-mismatch risk and is entirely unvalidated. So any ICA change — cap, warm start, block fit, or C++ rewrite — must ship with an AUC/sensitivity delta from `evaluate_baseline.py` on `sample_data` and a paragraph in `methodology_statements.md`. The C++ port is actually the *safest* ICA change available, because it can be made numerically comparable while the algorithmic "fixes" cannot.

---

## 6. What NOT to do

- **Do not port the ConvLSTM or the STFT to C++ as a performance claim.** 4% ceiling, measured.
- **Do not cap FastICA `max_iter` for speed.** Tested; it flips detections; the model was trained on the non-converged decomposition. Log the non-convergence rate instead (18/33 windows on one real file) and disclose it — that is a legitimate thesis observation.
- **Do not migrate the runtime to Python 3.10 / modern MNE this semester.** The model ports exactly; the ICA does not. You would be regenerating every cached prob and every reported number during write-up.
- **Do not engineer for 24-hour ambulatory files.** The corpus is 306 EDFs, median 7.7 min, max 58.3 min. The 18 GB / 30 GB / 6.4 GB memory findings all extrapolate to data that does not exist and could not be scored anyway (~14,400 windows ≈ 1.5 h single-threaded). Take only the two free wins: `float32` instead of `float64` in `processing.py:87`, and move the `np.clip` *after* `decimate_for_display` in `signal_view.py:376` (bitwise-identical output, O(N) → O(pixels)). Then stop.
- **Do not thread inference to fix a "freeze".** Measured max event-loop gap during scoring is 334 ms, with a live Cancel button and a modal parent that blocks re-entrant `File▸Open`. The only genuinely unpumped stretch is `_build_model` (1.58 s, rebuilt on every uncached open) plus a **redundant second full EDF read** at `infer.py:159` re-reading the file `app.py:314` just read. Cache the model and pass the already-loaded array — one hour — instead of a QThread refactor.
- **Do not build the whole-file overview minimap, an undo stack, or a QTableView migration** before the CW results justify them. Zoom-out already reaches whole-file (`signal_view.py:498-503` is uncapped), the median demo file is 300 s and the timebase presets top out at exactly 300 s, and every verdict is one keypress from correct. The one sharp sub-case worth a targeted fix: an accidental region drag flips a *proposed* event to `edited`, and `edited` **is** exported (`app.py:1126`) — so add a "revert to AI extent" action restoring `source_start`/`source_stop` (already preserved at `events.py:81-82`, never read) rather than a general undo stack.
- **Do not do viewport-local filtering.** Measured 0.13 s median, 0.97 s worst on the actual corpus.
- **Do not invest further in ZUNA.** `methodology_statements.md:30` already concludes it is "inconclusive at best and mildly negative". Keep it as a documented negative result, add the permanent on-screen "reconstructed signal" banner and a `# ai_source = zuna (reconstructed)` csv_bi header line, and consider hiding the control entirely for clinician sessions.
- **Do not add `pyproject.toml` / `pip install -e .`.** Every entry point already self-bootstraps `sys.path` from `__file__` (`gui/main.py:12-13` and seven others) and `launch_gui.bat:20` forces cwd. It buys nothing this semester.
- **Do not chase** `.DS_Store` cleanup, CRLF in exports, high-DPI attributes, wall-clock time formatting, or event-ID stability before the demo.
- **Do not run the timed peer study before the ethics reply** (`peer_review_protocol.md:8`, `:185`). The cognitive walkthrough and the heuristic evaluation are ungated. Run those.

---

## 7. Suggested week-by-week ordering

**Week 1 — unblock everything else (all hours-scale).**
Ethics email out with the consultation-vs-participation question. `git add` the four untracked files + the weights (`!convlstm_ICA_12_train.h5`). Rebuild the demo caches (`precompute_probs.py --overwrite`) over five chosen files and add the post-write `load_probs` assertion. Ship the disclaimer layer (B1): title suffix, permanent status-bar banner, csv_bi header lines, provenance field.

**Week 2 — the honesty layer + your first real result.**
`scored` mask end-to-end and hatched "not assessed" rendering (B2a). Relabel score and fix the exported confidence column (B2b). Dirty flag + guarded `load_edf`/`closeEvent` + autosave (B3). Export guardrails (B5). Then run the cognitive walkthrough solo against the frozen set and commit the results document.

**Week 3 — make the numbers defensible.**
`match_events` false-positive fix (R1) + event merge/min-duration in `build_proposed_events` (R2), then re-run `evaluate_baseline.py` across the manifest and regenerate **every** event-level number in `methodology_statements.md`. Fix `run_inference.py` time units (R3). Bind `eval_config` everywhere and add the meta-vs-config check in `evaluate_baseline.load_file` (R4/R5). Disclose the AUC exclusion (R6). This is the week the results chapter stops being wrong.

**Week 4 — portability.**
Launcher discovery, README install section, yml cleanup, cache validation on sha256 + basename + meta, remove the three `os.chdir` calls via `pyst.py:606`. Finish by cloning onto a second machine and running the demo cold. Write that up.

**Week 5 — GUI, driven by the CW findings.**
Sensitivity default, Help/About + F1 shortcut dialog + legend, `ChannelInspector.set_data`, spectrogram tab. Recolour accepted vs reference; add the blind-mode toggle and `references_visible` provenance flag.

**Week 6 — clinician-facing package.**
`docs/usability/clinician_briefing.md` (prototype status, public TUSZ only, no patient data, no diagnostic task, 26.3% sensitivity at 0.5, exactly what is and is not recorded) plus a clinician consent variant if comments are to be quoted. Reviewer reason field. Full provenance ledger (git SHA, UTC, reviewer, weights hash, all four statuses). **Clinician session, assuming ethics has replied.**

**Weeks 7-8 — write-up.**
Occlusion attribution only if there is genuine slack. Prototype the `multiprocessing` window-loop parallelisation and benchmark it — that becomes the honest baseline and the bridge into BMET4112.

**BMET4112 —** numpy reference forward pass → C++ FastICA front end with pybind11 → benchmark table + numerical-agreement study, with the profile that justified the choice as the opening figure.