# Cognitive Walkthrough Instrument — EEG Reviewer GUI

**Project:** BMET4111 Thesis (Sam Gan, University of Sydney) — A GUI-based Human-AI teaming system for reviewing ambulatory/outpatient EEG.
**Supervisor:** Prof. Omid Kavehei.
**System under evaluation:** The reviewer GUI (PyQt5 desktop application), launched with `C:\Users\User\miniconda3\envs\seiz36\python.exe -m gui.main`.
**Validation data:** Public TUH/TUSZ EEG only. No patient or RPAH data is used in this instrument.

---

## 1. About the cognitive walkthrough

A *cognitive walkthrough* (CW) is an expert-driven, task-based usability inspection method. Rather than recruiting end users, one or more evaluators (here, the thesis author and, where available, the supervisor or a peer reviewer acting as a domain proxy) step through a set of realistic tasks exactly as a first-time or occasional user would, and ask a fixed set of questions at each action. Because no participants are involved, the method carries **no human-research ethics burden** and can be run repeatedly and cheaply during development. Its purpose is to find *learnability* problems — points where the interface fails to lead a reasonable user to the correct next action — before any participant-based usability study is undertaken.

The walkthrough is appropriate for this thesis because the deliverable is the reviewer interface itself (an MVP) and its usability evaluation, not detector accuracy. The CW lets us surface interaction-design defects in a structured, defensible way and feed them into a severity-ranked issue list (Section 3).

### The four standard CW questions

At **each action** in a task, the evaluator records a pass/fail judgement and a note against the following four questions (Wharton et al., as adapted):

1. **Q1 — Goal/effect.** Will the user try to achieve the right effect? (Does the user understand that *this* action is the next thing to do to make progress towards their goal?)
2. **Q2 — Availability.** Will the user notice that the correct action is available? (Is the control, key, or affordance visible and identifiable at the moment it is needed?)
3. **Q3 — Association.** Will the user associate the correct action with the effect they are trying to achieve? (Does the label, icon, or position make it clear that this control produces the desired result, rather than some other control?)
4. **Q4 — Feedback.** If the correct action is performed, will the user see that progress is being made towards the goal? (Does the system give visible, timely, interpretable feedback that the action succeeded?)

A "fail" on any question is a candidate usability issue. The evaluator should write *why* it failed (e.g., "control present but no label", "feedback delayed with no indicator"), because the wording of that note drives the severity scoring in Section 3.

### How to run this instrument

- Launch the app: `C:\Users\User\miniconda3\envs\seiz36\python.exe -m gui.main`.
- Work through the tasks in order. Each task names a **demo EDF** via a placeholder (e.g. `<DEMO_SEIZURE_FILE_1>`); fill these from the 39 cached sample EDFs that already have `.probs.npz` files so they load instantly and warning-free. Suggested candidates from the cached set are listed under each task.
- For each action, tick Q1–Q4 (pass `P` / fail `F`) and write a short note. One row of the table is one action.
- A single evaluator can complete the whole instrument; two evaluators scoring independently and then reconciling will produce stronger evidence.

> **Note on demo file selection.** For the ZUNA comparison task you must choose a file that has **both** a baseline `.probs.npz` and a ZUNA `.zuna.probs.npz` cached, otherwise switching the AI-source will trigger slow on-demand inference rather than an instant load. Ten files satisfy this (e.g. `aaaaaarq_s016_t003`, `aaaaaghb_s010_t000`, `aaaaatao_s003_t000`, `aaaaahsi_s014_t000`, `aaaaaraf_s004_t000`); pick one of these for `<DEMO_ZUNA_FILE>`.

---

## 2. Reviewer tasks

Each task states: the **goal**, the **demo file** placeholder (with suggested cached candidates), the **correct action sequence**, and a **CW scoring table** with the four questions per action.

Legend for the scoring tables: **P** = pass, **F** = fail. Record one judgement per cell and a free-text note.

---

### Task 1 — Open a known-seizure file and obtain proposed events

**Goal:** Load a public seizure EDF, run on-demand inference, and arrive at a state where the probability strip and proposed events are visible.

**Demo file:** `<DEMO_SEIZURE_FILE_1>`
*Suggested cached candidates (load instantly):* `aaaaaarq_s016_t003`, `aaaaatao_s003_t000`, `aaaaaghb_s010_t000`.

**Correct action sequence:**
1. From the menu/toolbar, choose **Open** and browse to `<DEMO_SEIZURE_FILE_1>.edf`.
2. Confirm the file selection.
3. On-demand inference begins; a **cancellable progress dialog** appears. (On the very first open of the session the model is also loaded, so this is slower.)
4. Wait for the dialog to complete (do **not** cancel).
5. Observe the **SignalView** populated with EEG traces, the **ProbStrip** showing per-window probability aligned to the signal, and the **EventList** populated with candidate seizures at the default threshold (0.5).

| # | Action | Q1 Goal | Q2 Availability | Q3 Association | Q4 Feedback | Notes |
|---|--------|:---:|:---:|:---:|:---:|-------|
| 1.1 | Locate and trigger **Open** |  |  |  |  |  |
| 1.2 | Select the correct `.edf` in the file dialog |  |  |  |  |  |
| 1.3 | Recognise the progress dialog as "inference running, please wait" |  |  |  |  |  |
| 1.4 | Decide to wait rather than cancel |  |  |  |  |  |
| 1.5 | Recognise that traces + ProbStrip + EventList = file successfully loaded |  |  |  |  |  |

> **Known friction to watch for at Q4:** at the default threshold of 0.5 some files yield **few or zero proposed events**. A reasonable user may interpret an empty EventList as "load failed" rather than "no events above threshold". Note this explicitly if observed — it motivates Task 2.

---

### Task 2 — Lower the threshold until the reference seizure becomes a proposed event

**Goal:** Use the detection **threshold slider** to bring the known seizure (which the reviewer can see in the ProbStrip as an elevated region) above threshold so that it appears as a proposed event in the EventList.

**Demo file:** `<DEMO_SEIZURE_FILE_1>` (continue from Task 1).

**Correct action sequence:**
1. Inspect the **ProbStrip** and identify the time region where seizure probability is elevated.
2. Locate the **detection THRESHOLD slider** on the toolbar.
3. Drag the threshold **down** (e.g. from 0.5 towards 0.3) until the elevated region crosses the line.
4. Observe a new proposed event appear in the **EventList** and a corresponding marked region appear on the ProbStrip/SignalView.
5. Confirm the new event corresponds to the elevated probability region (the reference seizure).

| # | Action | Q1 Goal | Q2 Availability | Q3 Association | Q4 Feedback | Notes |
|---|--------|:---:|:---:|:---:|:---:|-------|
| 2.1 | Read the ProbStrip to locate the candidate seizure |  |  |  |  |  |
| 2.2 | Find the THRESHOLD slider |  |  |  |  |  |
| 2.3 | Understand that *lowering* threshold yields *more* events |  |  |  |  |  |
| 2.4 | Drag the slider to the correct value |  |  |  |  |  |
| 2.5 | See the new proposed event appear in the EventList |  |  |  |  |  |

> **Watch for at Q3:** the slider's direction-of-effect (lower = more sensitive) is a classic association failure. Note whether any on-screen cue (tick labels, a live count, the threshold line on the strip) makes the relationship clear.

---

### Task 3 — Jump to and accept the reference seizure event

**Goal:** Navigate to the proposed reference event and mark it **accepted**.

**Demo file:** `<DEMO_SEIZURE_FILE_1>` (continue from Task 2).

**Correct action sequence:**
1. In the **EventList**, select the proposed reference event (click the row, or use **J/K** to move the selection between events).
2. Press **Enter** to jump the SignalView to that event (or use the row's jump button).
3. Confirm the SignalView and ProbStrip have centred on the event.
4. Press **Space** to **Accept** the event (or use the row's accept button).
5. Confirm the event's state changes to **accepted** (state indication in the EventList row).

| # | Action | Q1 Goal | Q2 Availability | Q3 Association | Q4 Feedback | Notes |
|---|--------|:---:|:---:|:---:|:---:|-------|
| 3.1 | Select the target event in the EventList |  |  |  |  |  |
| 3.2 | Use J/K (or buttons) to move between events |  |  |  |  |  |
| 3.3 | Press Enter (or jump button) to navigate to the event |  |  |  |  |  |
| 3.4 | Recognise the view has jumped to the event |  |  |  |  |  |
| 3.5 | Press Space (or accept button) to accept |  |  |  |  |  |
| 3.6 | See the event state become "accepted" |  |  |  |  |  |

> **Watch for at Q2/Q3:** the keyboard shortcuts (J, K, Enter, Space, X) are powerful but invisible — there is no on-screen legend by default. Note whether a first-time user could discover them, and whether the per-row buttons provide an adequate visible alternative.

---

### Task 4 — Reject a false-positive event

**Goal:** Identify a proposed event that is not a genuine seizure and mark it **rejected**, so it is excluded from the export.

**Demo file:** `<DEMO_FP_FILE>`
*Suggested approach:* either continue on `<DEMO_SEIZURE_FILE_1>` after lowering the threshold (Task 2), which typically introduces spurious events, or use a cached non-seizure-dominant file such as `aaaaahsi_s014_t000` or `aaaaaraf_s004_t000`.

**Correct action sequence:**
1. In the **EventList**, select a proposed event that does not correspond to the reference seizure region.
2. Press **Enter** (or jump button) to inspect it in the SignalView; use the **ChannelInspector** if needed to confirm it is not seizure morphology.
3. Press **X** to **Reject** the event (or use the row's reject button).
4. Confirm the event's state changes to **rejected**.

| # | Action | Q1 Goal | Q2 Availability | Q3 Association | Q4 Feedback | Notes |
|---|--------|:---:|:---:|:---:|:---:|-------|
| 4.1 | Identify a candidate as a likely false positive |  |  |  |  |  |
| 4.2 | Jump to it to inspect the traces |  |  |  |  |  |
| 4.3 | Use the ChannelInspector to confirm (optional) |  |  |  |  |  |
| 4.4 | Press X (or reject button) to reject |  |  |  |  |  |
| 4.5 | See the event state become "rejected" |  |  |  |  |  |

> **Watch for at Q4:** confirm that accepted, rejected, proposed, and edited states are *visually distinguishable* in the EventList. If accept and reject produce the same visual change, that is a feedback failure.

---

### Task 5 — Edit an event's extent, then export the reviewed `.csv_bi`

**Goal:** Adjust the temporal extent of an accepted event by dragging its region, then export a reviewed annotation file containing only accepted and edited events.

**Demo file:** `<DEMO_SEIZURE_FILE_1>` (continue from Task 3).

**Correct action sequence:**
1. With the accepted reference event in view, **drag the edge of its region** on the SignalView/ProbStrip to adjust its start or end.
2. Confirm the event's state changes to **edited** and the new extent is reflected.
3. Locate the **Export** control on the toolbar.
4. Trigger **Export**.
5. Confirm a reviewed `.csv_bi` is written and contains only **accepted + edited** events (rejected and untouched-proposed events are excluded).

| # | Action | Q1 Goal | Q2 Availability | Q3 Association | Q4 Feedback | Notes |
|---|--------|:---:|:---:|:---:|:---:|-------|
| 5.1 | Recognise that the region edge is draggable |  |  |  |  |  |
| 5.2 | Drag the edge to the desired extent |  |  |  |  |  |
| 5.3 | See the state change to "edited" and the extent update |  |  |  |  |  |
| 5.4 | Find the Export control |  |  |  |  |  |
| 5.5 | Trigger Export |  |  |  |  |  |
| 5.6 | Confirm the `.csv_bi` was written (path/confirmation feedback) |  |  |  |  |  |

> **Watch for at Q1/Q2:** the draggable region edge has no obvious handle. Note whether the cursor changes on hover, or whether a user would even attempt the drag. At Q4, note whether the export gives a confirmation (path, toast, dialog) or completes silently — silent success is a feedback failure.

---

### Task 6 — Switch AI-source to ZUNA on a cached file and compare

**Goal:** Switch the AI-source selector from Baseline to **ZUNA full** on a file that has cached ZUNA probabilities, and compare the resulting probability strip and proposed events against the baseline.

**Demo file:** `<DEMO_ZUNA_FILE>` — **must** have both baseline and ZUNA cached probs.
*Suggested cached candidates (both sources cached):* `aaaaaarq_s016_t003`, `aaaaaghb_s010_t000`, `aaaaatao_s003_t000`, `aaaaahsi_s014_t000`, `aaaaaraf_s004_t000`.

**Correct action sequence:**
1. Open `<DEMO_ZUNA_FILE>.edf` (loads instantly from cache; AI-source defaults to **Baseline**).
2. Note the baseline ProbStrip shape and the baseline EventList at the current threshold.
3. Locate the **AI-source selector** on the toolbar and switch it to **ZUNA full**.
4. Confirm the ProbStrip and EventList update to the ZUNA probabilities (instant, because ZUNA probs are cached for this file).
5. Compare: note where ZUNA raises or lowers probability relative to baseline, and how the proposed-event set differs at the same threshold.
6. (If exporting under ZUNA) confirm the export is written as a `.zuna.reviewed.csv_bi`.

| # | Action | Q1 Goal | Q2 Availability | Q3 Association | Q4 Feedback | Notes |
|---|--------|:---:|:---:|:---:|:---:|-------|
| 6.1 | Open a both-sources-cached file |  |  |  |  |  |
| 6.2 | Read and remember the baseline ProbStrip/events |  |  |  |  |  |
| 6.3 | Find the AI-source selector |  |  |  |  |  |
| 6.4 | Recognise it switches the model that produced the strip |  |  |  |  |  |
| 6.5 | Switch to ZUNA full |  |  |  |  |  |
| 6.6 | See the strip/events update (and that it was instant) |  |  |  |  |  |
| 6.7 | Compare baseline vs ZUNA meaningfully |  |  |  |  |  |

> **Honesty note to record alongside this task.** The probability strip currently displays **raw, uncalibrated softmax** (no temperature/Platt/isotonic calibration). A reviewer comparing two strips is therefore comparing two *uncalibrated* probability fields; the visual "height" is not a calibrated likelihood. Record whether the GUI communicates this caveat at all — its absence is itself a usability/trust finding, given that the literature review treats calibration as a trust requirement. Do **not** present ZUNA as a validated seizure front-end in any walkthrough note; it is an exploratory repurposing of a general EEG super-resolution model.

---

## 3. From walkthrough failures to severity-ranked usability issues

Each **fail** recorded against Q1–Q4 becomes a candidate usability issue. Convert candidates into a ranked issue list using the procedure below.

### Step 1 — Consolidate

Group fails that describe the same underlying defect (e.g. "no shortcut legend" may surface as a Q2 fail in Tasks 3 and 4 — log it once). Each consolidated issue gets: an ID, a short title, the task(s)/action(s) where it appeared, and which CW question(s) it failed.

### Step 2 — Classify by CW question (diagnostic, not severity)

The failing question hints at the *type* of fix:

- **Q1 (Goal) fail** → conceptual-model / labelling / onboarding problem.
- **Q2 (Availability) fail** → visibility / discoverability problem (hidden control, invisible shortcut).
- **Q3 (Association) fail** → mapping / labelling problem (control present but its effect unclear, e.g. slider direction).
- **Q4 (Feedback) fail** → feedback problem (silent success, delayed or missing indicator).

### Step 3 — Score severity

Rate each issue on two axes, then combine.

**Impact** — how badly it blocks or misleads the reviewer:

| Impact | Meaning |
|---|---|
| 3 — High | User cannot complete the task, or completes it *incorrectly* (e.g. exports the wrong events, mis-reads an uncalibrated strip as calibrated). |
| 2 — Medium | User completes the task but with confusion, a wrong turn, or a workaround. |
| 1 — Low | Cosmetic or minor annoyance; task completes smoothly. |

**Frequency** — how often a reviewer hits it in normal use:

| Frequency | Meaning |
|---|---|
| 3 — Often | Occurs on every file / every review session (e.g. default threshold yielding empty EventList). |
| 2 — Sometimes | Occurs on some files or some paths. |
| 1 — Rare | Edge case only. |

**Severity = Impact × Frequency** (range 1–9). Map to a priority band:

| Severity score | Priority band | Action |
|---|---|---|
| 6–9 | **P1 — Critical** | Must fix before any participant-based usability study. |
| 3–4 | **P2 — Major** | Fix in the MVP if time permits; otherwise document as a known limitation. |
| 1–2 | **P3 — Minor** | Backlog / note in the thesis limitations section. |

### Step 4 — Record

Log each consolidated issue in a table of the form:

| ID | Title | Task/Action | CW question(s) failed | Impact (1–3) | Frequency (1–3) | Severity (I×F) | Priority | Recommended fix |
|----|-------|-------------|-----------------------|:---:|:---:|:---:|:---:|-----------------|
| U-01 | *e.g. Empty EventList at default threshold reads as "load failed"* | T1.5 / T2 | Q4 / Q1 |  |  |  |  | *e.g. show live event count + "0 events above 0.50" hint* |
| U-02 | *e.g. Keyboard shortcuts (J/K/Enter/Space/X) undiscoverable* | T3.2 / T4.4 | Q2 |  |  |  |  | *e.g. add a shortcut legend / tooltips on row buttons* |
| U-03 | *e.g. Threshold slider direction-of-effect unclear* | T2.3 | Q3 |  |  |  |  | *e.g. label "more / fewer events"; draw threshold line on strip* |
| U-04 | *e.g. Export completes silently* | T5.6 | Q4 |  |  |  |  | *e.g. confirmation toast with written file path* |
| U-05 | *e.g. Uncalibrated probability strip not flagged* | T6 | Q1 / Q4 |  |  |  |  | *e.g. label strip "raw softmax — uncalibrated"* |

The example rows above are illustrative seeds drawn from the "watch for" notes in Section 2; the evaluator should replace, extend, and score them from the actual walkthrough results.

### Step 5 — Report

In the thesis, report: the number of issues found, their distribution across the four CW questions, the count in each priority band, and the P1 items in full. This gives a defensible, expert-driven usability result for the MVP without invoking participant ethics, and cleanly sets up any subsequent participant-based evaluation.

---

*Reference: Wharton, C., Rieman, J., Lewis, C., & Polson, P. (1994). The Cognitive Walkthrough Method: A Practitioner's Guide. In J. Nielsen & R. L. Mack (Eds.), Usability Inspection Methods. Wiley.*
