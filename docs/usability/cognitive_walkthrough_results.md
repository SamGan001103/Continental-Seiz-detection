# Cognitive Walkthrough — Results

**Project:** BMET4111 Thesis (Sam Gan, University of Sydney) — GUI-based Human-AI teaming system
for reviewing ambulatory/outpatient EEG. **Supervisor:** Prof. Omid Kavehei.
**Instrument:** [`cognitive_walkthrough.md`](cognitive_walkthrough.md) (Wharton et al., 1994).
**Date of pass:** 2026-08-09 · **Evaluators:** 1 · **Commit under evaluation:** `29d4a34`
**Data:** public TUH/TUSZ only.

---

## 0. How this pass was conducted, and what it therefore cannot tell you

A cognitive walkthrough is an **expert inspection**, not a user test, so a single evaluator
working without participants is a valid execution of the method. This pass was performed by
stepping through each task's action sequence against the **running application** — the widget
tree, toolbar contents, labels, tooltips, colours, event states and status-bar messages were read
out of a live `MainWindow` instance driven offscreen, so every judgement below reflects the code
as committed rather than an assumption about it.

**Three limits follow, and they should be stated in the thesis:**

1. **Perceptual judgements are weaker than a human's.** Whether a colour difference is *noticeable*
   at a glance, whether a drag handle *looks* draggable, and whether a 140 px strip is legible at
   a clinical display size cannot be settled by reading widget state. Rows judged on perception
   are marked **(P?)** and need a human pass to confirm.
2. **Timing and flow are not assessed.** Real friction — hesitation, backtracking, mis-clicks —
   needs a person at the screen.
3. **One evaluator.** Nielsen's own guidance is that a single evaluator finds roughly a third of
   problems. Treat this as a floor, not a census.

A second pass by Prof. Kavehei or a peer, scoring independently and reconciling, would
materially strengthen the evidence and costs about an hour.

### Demo files used

Chosen from the cached set so every task loads instantly, verified against `load_probs`,
`.csv_bi` and the current decision stage:

| Placeholder | File | Why |
|---|---|---|
| `<DEMO_SEIZURE_FILE_1>` | `aaaaatao_s003_t000` | 1 reference seizure; **1 event at 0.5 → 2 at 0.3**, so Task 2's threshold drop genuinely changes the worklist |
| `<DEMO_FP_FILE>` | `aaaaajru_s031_t001` | 6 events at 0.5 against 3 references, so false positives exist to reject |
| `<DEMO_ZUNA_FILE>` | `aaaaaraf_s004_t000` | both baseline and ZUNA probabilities cached |

> The instrument says "39 cached sample EDFs". That was never true — only 6 validated at the
> time it was written. After this session's full-corpus precompute, **305 recordings have a
> validated cache** and 25 are demo-ready (validated + seizure-bearing + fires at 0.5). The
> instrument's file-selection note should be updated.

---

## 1. Task-by-task scoring

**P** = pass · **F** = fail · **(P?)** = pass, but perceptual and needs human confirmation.

### Task 1 — Open a known-seizure file and obtain proposed events

| # | Action | Q1 | Q2 | Q3 | Q4 | Notes |
|---|---|:--:|:--:|:--:|:--:|---|
| 1.1 | Locate and trigger **Open** | P | P | P | P | `Open EDF…` is the first toolbar item and carries the standard `Ctrl+O`. |
| 1.2 | Select the correct `.edf` | P | P | P | P | Dialog opens in `sample_data/` and filters to `*.edf`. |
| 1.3 | Recognise the progress dialog | P | P | P | **F** | Cached files skip inference entirely, so nothing appears — correct, but the *first uncached* open shows "Loading model (first open only)…" then per-window counts. The two phases are distinct; the dialog does not say the model load is a one-off cost in wall-clock terms. **U-06.** |
| 1.4 | Decide to wait rather than cancel | P | P | P | P | Cancel button present and functional; cancelling leaves an explanatory status message. |
| 1.5 | Recognise load succeeded | P | P | P | P | Status bar reports duration, window count and reference count; the event-list summary shows a live tally. |

**On the instrument's "known friction" note:** an empty worklist at threshold 0.5 no longer reads
as a failed load. The summary label renders **"No events proposed at this threshold"**, which
names the cause. Seed issue **U-01 is resolved** — do not carry it forward.
(`aaaaaghb_s010_t000` is a genuine zero-event file at both 0.5 and 0.3 if you want to exercise it.)

### Task 2 — Lower the threshold until the reference seizure becomes an event

| # | Action | Q1 | Q2 | Q3 | Q4 | Notes |
|---|---|:--:|:--:|:--:|:--:|---|
| 2.1 | Read the ProbStrip to locate the candidate | P | P | P | (P?) | The strip now draws the **decision curve** — the series the threshold actually acts on — with peak evidence as a dotted overlay. Legibility at 140 px is a perceptual question. |
| 2.2 | Find the THRESHOLD slider | P | P | P | P | Labelled `Threshold`, with the numeric value shown beside it (` 0.50 `). |
| 2.3 | Understand that *lowering* yields *more* events | P | P | **F** | P | Nothing states the direction of effect. The slider carries **no tooltip**, and there are no end labels. A reviewer must infer it or discover it by trial. **U-03.** |
| 2.4 | Drag the slider to the correct value | P | P | P | P | Value label updates live. |
| 2.5 | See the new event appear | P | P | P | P | Event list, summary tally and strip update together; on this file 1 → 2 events. |

> The instrument's "watch for" asks whether any cue makes the relationship clear. **The threshold
> line on the strip does** — it moves visibly against the curve — so the association is
> discoverable *after* the first drag. The failure is that it is not discoverable *before* it.

### Task 3 — Jump to and accept the reference seizure

| # | Action | Q1 | Q2 | Q3 | Q4 | Notes |
|---|---|:--:|:--:|:--:|:--:|---|
| 3.1 | Select the target event | P | P | P | P | Row click selects; selection auto-centres the view. |
| 3.2 | Use **J/K** to move between events | P | **F** | **F** | P | **J/K appear nowhere in the UI.** They exist only in the `gui/app.py` module docstring. Unlike Enter/Space/X they are not surfaced on any button tooltip, so they are undiscoverable. **U-02.** |
| 3.3 | Press **Enter** / jump button | P | P | P | P | The row's `→` button is tooltipped **"Jump (Enter)"**, which teaches the shortcut at the point of use. |
| 3.4 | Recognise the view jumped | P | P | P | (P?) | View re-centres with ±10 s context; no transient highlight marks *why* it moved. |
| 3.5 | Press **Space** / accept button | P | P | P | P | `✓` button tooltipped **"Accept (Space)"**. |
| 3.6 | See the state become "accepted" | P | P | **F** | **F** | Two defects. (a) In the **EventList** the status is **text only** — no colour, icon or weight change; `proposed` → `accepted` is a word swap. (b) In the **SignalView** accepted is `(40,170,90)` green while the ground-truth reference band is `(50,160,70)` green — **the same hue**, differing only in alpha, so an accepted event visually resembles the answer key. **U-07**, **U-08.** |

### Task 4 — Reject a false-positive event

| # | Action | Q1 | Q2 | Q3 | Q4 | Notes |
|---|---|:--:|:--:|:--:|:--:|---|
| 4.1 | Identify a likely false positive | P | P | P | P | With references visible the distinction is easy — *too* easy; see U-09. |
| 4.2 | Jump to it to inspect traces | P | P | P | P | As 3.3. |
| 4.3 | Use the **ChannelInspector** | **F** | **F** | **F** | P | Opened only by **double-clicking a channel label**. There is no menu item, no button, no tooltip and no hint anywhere in the UI. A first-time reviewer will not find it. Once open it works well. **U-04.** |
| 4.4 | Press **X** / reject button | P | P | P | P | `✗` button tooltipped **"Reject (X)"**. |
| 4.5 | See the state become "rejected" | P | P | **F** | **F** | Same as 3.6(a): text-only in the list. In the SignalView rejected greys out, which *is* distinct. **U-07.** |

> The instrument's "watch for" asks whether accept and reject are visually distinguishable.
> **In the SignalView, yes** (green vs grey). **In the EventList, no** — and the EventList is
> where the reviewer is working. That split is the finding.

### Task 5 — Edit an event's extent, then export

| # | Action | Q1 | Q2 | Q3 | Q4 | Notes |
|---|---|:--:|:--:|:--:|:--:|---|
| 5.1 | Recognise the region edge is draggable | **F** | **F** | P | P | Event regions are created `movable=True`, so the drag works, but there is no handle, no label and no documented affordance — only pyqtgraph's default line hover. **U-05.** |
| 5.2 | Drag the edge | P | P | P | P | Extent updates live. |
| 5.3 | See state → "edited" and extent update | P | P | P | **F** | Same text-only limitation as 3.6(a). Worse here: an **accidental** drag silently flips `proposed` → `edited`, and edited events **are exported**. There is no "revert to AI extent" despite `source_start`/`source_stop` being preserved in `gui/events.py`. **U-10.** |
| 5.4 | Find the Export control | P | P | P | P | `Export reviewed…` is the last toolbar item. |
| 5.5 | Trigger Export | P | P | P | P | Pre-flight dialog states how many events will be written and how many rejected/unreviewed will be omitted. |
| 5.6 | Confirm the file was written | P | P | P | P | Status bar shows the **full written path** for 10 s. |

> The instrument's "watch for" flags silent export as a feedback failure. **Both halves are now
> addressed** — a pre-flight before the write and a path afterwards. Seed issues **U-04 (silent
> export) is resolved**; the pre-flight additionally warns when zero events would be written and
> refuses to overwrite the reference file.

### Task 6 — Switch AI-source to ZUNA and compare

| # | Action | Q1 | Q2 | Q3 | Q4 | Notes |
|---|---|:--:|:--:|:--:|:--:|---|
| 6.1 | Open a both-sources-cached file | P | P | P | P | Loads instantly; cached ZUNA output is detected and the selector gains its entry. |
| 6.2 | Read/remember the baseline strip | P | P | P | (P?) | Nothing supports comparison across a switch — no overlay, no side-by-side, no memory aid. **U-11.** |
| 6.3 | Find the AI-source selector | P | P | P | P | Labelled `AI source`, tooltipped. |
| 6.4 | Recognise what it switches | P | P | P | P | Tooltip explains it selects between original-EDF and ZUNA probabilities. |
| 6.5 | Switch to ZUNA full | P | P | P | P | Instant from cache. |
| 6.6 | See strip/events update | P | P | P | P | Status bar names the active source and adds "reconstructed signal displayed"; channel labels gain a `(ZUNA)` suffix. |
| 6.7 | Compare baseline vs ZUNA meaningfully | **F** | P | P | (P?) | The reviewer is asked to compare two strips from memory alone. Nothing quantifies the difference, and nothing warns that a lower ZUNA curve is not evidence of a better detector. **U-11.** |

> **Honesty note (required by the instrument).** The strip is raw uncalibrated softmax.
> This **is** now communicated: the axis reads **"model score"** rather than `p(seiz)`, the event
> column header reads **"score"**, and the strip tooltip states that values are
> *"raw, UNCALIBRATED network outputs, not calibrated probabilities of seizure"*. Seed issue
> **U-05 (uncalibrated strip unflagged) is resolved.** The remaining gap is that the caveat lives
> in a tooltip, so it is available on demand rather than always visible.

---

## 2. Consolidated issue list

Severity = Impact × Frequency, per the instrument. Two seeds from the instrument
(**U-01** empty-list-reads-as-failure, **U-04** silent export) and one from the heuristic
instrument (threshold value not shown) were checked and found **already resolved**; they are
recorded as closed rather than carried forward.

| ID | Title | Task/Action | CW Q | I | F | Sev | Priority | Recommended fix |
|---|---|---|:--:|:-:|:-:|:--:|:--:|---|
| **U-07** | Event states are text-only in the EventList | T3.6, T4.5, T5.3 | Q3, Q4 | 2 | 3 | **6** | **P1** | Colour the status cell (or whole row) using the SignalView's existing palette; add an icon so status is not colour-alone. |
| **U-08** | Accepted events share the reference band's hue | T3.6 | Q3, Q4 | 3 | 2 | **6** | **P1** | Recolour accepted to a non-green (e.g. blue), keeping green exclusively for ground truth. Also add a blind-mode toggle and record `references_visible` in provenance. |
| **U-04** | ChannelInspector is undiscoverable (double-click only) | T4.3 | Q1, Q2, Q3 | 2 | 3 | **6** | **P1** | Add a per-channel context menu and a toolbar/menu entry; hint "double-click a channel" in the status bar on first load. |
| **U-02** | J/K shortcuts invisible | T3.2 | Q2, Q3 | 2 | 3 | **6** | **P1** | Add **Help ▸ Keyboard shortcuts** (F1) reusing the module docstring verbatim, plus next/prev buttons in the EventList header. |
| **U-03** | Threshold direction-of-effect uncued | T2.3 | Q3 | 2 | 2 | 4 | P2 | Tooltip on the slider ("lower = more candidates"); end labels *fewer / more*; live "N events at 0.50". |
| **U-05** | Region edge has no drag affordance | T5.1 | Q1, Q2 | 2 | 2 | 4 | P2 | Widen the hover zone, change cursor on hover, and state it in the help dialog. |
| **U-10** | Accidental drag silently marks an event "edited", and edited events export | T5.3 | Q4 | 3 | 1 | 3 | P2 | Add **revert to AI extent** restoring the already-preserved `source_start`/`source_stop`; distinguish edited from accepted in the list. |
| **U-11** | No support for comparing baseline vs ZUNA | T6.2, T6.7 | Q1 | 1 | 2 | 2 | P3 | Out of scope — ZUNA is a documented negative result. Consider hiding the selector for clinician sessions. |
| **U-06** | Model-load latency not distinguished from inference | T1.3 | Q4 | 1 | 1 | 1 | P3 | One-line dialog text: "loading model (one-off, ~2 s)". |
| **U-09** | Ground truth is visible by default during review | T4.1 | — | 2 | 3 | **6** | **P1*** | *Method*, not usability. Any session used as evidence must run with references hidden, or be permanently flagged. Add the blind-mode toggle with U-08. |

### Distribution

| CW question | Fails | Reading |
|---|---|---|
| Q1 — Goal | 3 | conceptual/onboarding: ChannelInspector, drag affordance, ZUNA comparison |
| Q2 — Availability | 4 | **the dominant failure mode — discoverability of controls that exist and work** |
| Q3 — Association | 5 | labelling/mapping: states, hues, slider direction |
| Q4 — Feedback | 5 | mostly the single text-only-status defect recurring across three tasks |

| Priority | Count | Items |
|---|---|---|
| **P1 — Critical** | 5 | U-07, U-08, U-04, U-02, U-09 |
| P2 — Major | 3 | U-03, U-05, U-10 |
| P3 — Minor | 2 | U-11, U-06 |

---

## 3. Findings in one paragraph

Ten issues survived consolidation, of which five are P1. **No task was blocked** — every action
sequence completed — so the interface is functionally sound and the failures are concentrated in
*learnability*, which is exactly what the method is designed to expose. The dominant pattern is
**Q2/Q3: controls that exist and work correctly but are not discoverable or not clearly mapped**
(J/K, the ChannelInspector, the draggable region edge, the slider's direction). The second
pattern is a **single feedback defect with wide reach**: event status is text-only in the
EventList, which alone produced three Q4 fails across Tasks 3, 4 and 5. Both are cheap to fix —
a Help dialog, a colour map already defined elsewhere in the codebase, and three tooltips would
close six of the ten.

The one finding that is not a usability issue is **U-09**: the ground-truth reference band is
visible by default, so any session used as evidence is contaminated unless references are hidden.
That needs deciding before, not after, the next evaluation.

Three previously assumed problems were checked and found **already fixed**: the empty worklist
now explains itself, export gives both a pre-flight and a written path, and the uncalibrated
nature of the score is stated in the axis label, the column header and the tooltip.

## 4. Feed-forward

U-07, U-08, U-02 and U-04 are the pre-clinician backlog and total roughly a day: a Help ▸ About /
Keyboard-shortcuts dialog, a status colour map in `event_list.py`, a recolour in
`signal_view.py:426-429`, and a context menu for the ChannelInspector. U-09 needs a decision
rather than code. U-03, U-05 and U-10 follow if time allows.

Re-running this instrument after those fixes should show the Q2/Q3 fails collapse, which is a
reportable before/after result for the thesis rather than a bare list of complaints.

---

*Reference: Wharton, C., Rieman, J., Lewis, C., & Polson, P. (1994). The Cognitive Walkthrough
Method: A Practitioner's Guide. In J. Nielsen & R. L. Mack (Eds.), Usability Inspection Methods.
Wiley.*
