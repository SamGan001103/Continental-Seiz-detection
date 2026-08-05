# Heuristic Evaluation Instrument — EEG Reviewer GUI

**Project:** BMET4111 Thesis — A Human–AI Teaming GUI for Reviewing Ambulatory/Outpatient EEG
**Author:** Sam Gan (University of Sydney) · **Supervisor:** Prof. Omid Kavehei
**Artefact under evaluation:** PyQt5 desktop reviewer GUI (`python -m gui.main`; Python 3.6 / TF 1.15)
**Method:** Nielsen's heuristic evaluation · **Scope:** WS4 usability evaluation feeding WS2 GUI hardening

---

## 1. Introduction

### What heuristic evaluation is
Heuristic evaluation is a discount usability-inspection method in which a small number of usability
experts independently examine an interface and judge its conformance to a short list of recognised
usability principles (the "heuristics"). Unlike a controlled user study, it does not measure task
performance with end users; instead, each expert walks through the interface, flags places where it
violates a heuristic, and records the location and a severity judgement for every problem found. The
independent findings are then pooled and ranked. It is fast, cheap, and well suited to evaluating an
MVP such as this reviewer GUI before it is exposed to clinical reviewers.

### Why Nielsen's 10 heuristics
Nielsen's ten usability heuristics (Nielsen & Molich, 1990; Nielsen, 1994) are the de facto standard
set for this method. They are general enough to apply to a specialised clinical tool, they are
compact and memorable, they are extensively cited and therefore defensible in a thesis, and they have
an established 0–4 severity scale that travels with them. Using a recognised standard set (rather than
an ad-hoc home-grown checklist) makes the evaluation reproducible and comparable to the wider HCI
literature, which is important given that the thesis deliverable is the review interface and its
usability evaluation rather than detector accuracy.

### Why 2–3 expert evaluators suffice
Nielsen's empirical work shows that a single evaluator finds only about a third of usability problems,
but that the proportion found rises steeply as evaluators are added, with diminishing returns beyond
roughly five. The widely used guidance is that three to five evaluators reach a favourable
cost–benefit point; for a single-window MVP of this size, **2–3 experts** is a pragmatic and
defensible choice that captures the majority of the high-severity issues while remaining feasible
within the thesis timeline. Each expert inspects the GUI **independently** first (to avoid anchoring),
and findings are merged afterwards.

### Why no human-subjects ethics approval is required
Heuristic evaluation engages **expert evaluators as analysts, not as research participants**. The
experts are inspecting an artefact and reporting professional judgements about its design; no personal
data are collected about them, no patient or RPAH data are involved (the GUI is exercised only on
public TUH/TUSZ sample EDFs that already ship with cached probabilities), and there is no
intervention on or measurement of human subjects. The method therefore falls outside the scope of
human-research ethics review. (The later usability study with reviewers acting as participants — WS4 —
is where an ethics pathway will be required and is documented separately.)

---

## 2. Heuristic Evaluation Table

**Severity scale (Nielsen 0–4).** Every observed issue is rated on the standard scale:

| Rating | Meaning |
|---|---|
| **0** | Not a usability problem at all. |
| **1** | Cosmetic problem only — fix if spare time is available. |
| **2** | Minor usability problem — low priority to fix. |
| **3** | Major usability problem — important to fix, high priority. |
| **4** | Usability catastrophe — imperative to fix before release. |

Severity is judged from three factors: frequency (how often it is encountered), impact (how hard it
is to overcome), and persistence (whether it recurs or is a one-off). The *Observed issue*,
*Severity*, and *Screen/control* cells below are **worked examples / candidate findings** to be
confirmed, amended, or replaced by each evaluator during the actual inspection; the *What to check*
column is the fixed instrument.

| # | Heuristic (and definition) | What to check in THIS EEG-reviewer GUI | Observed issue (example) | Severity | Screen/control |
|---|---|---|---|---|---|
| **1** | **Visibility of system status** — the system should keep users informed about what is going on, through timely feedback. | On opening an EDF, is the on-demand inference progress clearly shown (cancellable progress dialog, and the extra "first open also loads the model" delay called out)? Does the **ProbStrip** update live and stay visibly aligned to the **SignalView** as you scroll/zoom? Does moving the **detection THRESHOLD slider** give immediate, visible feedback (events appearing/disappearing on the ProbStrip and **EventList**)? Does switching the **AI-source selector** (Baseline vs ZUNA full) clearly indicate which source is currently active and shown? | First-open model-load latency is not distinguished from per-window inference, so the user cannot tell whether the app has stalled; AI-source label does not persist in view after selection. | 3 | Open/progress dialog; AI-source selector |
| **2** | **Match between system and the real world** — speak the users' language; follow real-world conventions. | Do montage names, **High-pass / Low-pass / Notch** filter labels, **sensitivity** (µV/mm) and **timebase** (mm/sec or sec/page) use conventional clinical EEG terminology and units that a reviewer expects? Is the seizure **probability** on the ProbStrip expressed in terms a clinician reads naturally (0–1 / %), and are event times shown in real recording time? Does "Accept/Reject" map to how a reviewer thinks about confirming candidate seizures? | "Sensitivity" and "timebase" presented as raw numbers without conventional clinical units; probability strip y-axis unlabelled. | 2 | Toolbar (sensitivity, timebase); ProbStrip |
| **3** | **User control and freedom** — provide clearly marked "emergency exits"; support undo and redo. | Can the reviewer **cancel** an in-progress inference cleanly via the progress dialog? After Accept (Space) / Reject (X), can a state be **reversed** (undo a mis-press)? After dragging to **edit an event's extent**, can the edit be undone or reverted to the proposed extent? Can the reviewer back out of a montage/filter change without re-running inference? | No undo for an accidental Reject (X); the only recovery is to re-find the event and re-accept; region-edit drags are not revertible to the original proposed bounds. | 3 | EventList accept/reject; SignalView region edit |
| **4** | **Consistency and standards** — users should not wonder whether different words, situations, or actions mean the same thing. | Are the four **event states** (proposed / accepted / rejected / edited) shown with consistent colour/iconography across the **EventList**, **ProbStrip**, and **SignalView** regions? Do toolbar controls behave consistently (e.g. all filters applied the same way)? Are keyboard shortcuts (**J/K** to move, **Enter** to jump, **Space** accept, **X** reject) consistent with their on-screen buttons and labelled the same way everywhere? Do Baseline and ZUNA sources present results in the same visual language? | Event-state colour coding differs between the EventList rows and the ProbStrip overlay; J/K shortcuts not surfaced on the per-row jump buttons. | 2 | EventList; ProbStrip; SignalView |
| **5** | **Error prevention** — even better than good error messages is a careful design that prevents problems occurring. | Does **Export** guard against writing an empty or unintended `.csv_bi` (e.g. when default threshold 0.5 yields few/zero events, or when nothing has been accepted)? Is the user warned before exporting under the *wrong* AI source (Baseline overwriting vs ZUNA writing `.zuna.reviewed.csv_bi`)? Are non-cached files (slow, sometimes-warning per-window ICA) flagged *before* a long run starts? Does the threshold slider prevent nonsensical values? | Export proceeds silently with zero accepted events on default-threshold files, producing an empty reviewed file; no pre-flight warning that a file lacks cached `.probs.npz` and will trigger slow ICA. | 3 | Export; threshold slider; open dialog |
| **6** | **Recognition rather than recall** — minimise memory load; make actions and options visible. | Are the keyboard shortcuts (J/K/Enter/Space/X) **visible** in the UI rather than memorised? Is the **currently active threshold** value shown numerically next to the slider, not just as a handle position? Is the **active montage / filter / AI source** always readable on screen so the reviewer need not remember what they set? Does each **EventList** row show its probability and current state so the reviewer recognises status without re-deriving it? | Shortcut keys are not displayed anywhere in the GUI; the reviewer must recall them. Active threshold value is not shown as text beside the slider. | 3 | Toolbar (threshold, montage, filters, AI source) |
| **7** | **Flexibility and efficiency of use** — accelerators (unseen by novices) speed up expert interaction. | Do the **J/K/Enter/Space/X** accelerators let an expert review events without leaving the keyboard? Can a reviewer jump directly between proposed events efficiently? Can frequently used configurations (montage + filter + threshold + AI source) be reused across files, or must they be reset each open? Does threshold change re-rank/re-list events without forcing re-inference (since review state survives threshold changes)? | Filter/montage/threshold settings reset to defaults on each new EDF, forcing repeated reconfiguration for a batch of files. | 2 | Toolbar; EventList navigation |
| **8** | **Aesthetic and minimalist design** — dialogues should not contain irrelevant or rarely needed information. | Is the single-window layout (**SignalView + ProbStrip + EventList + ChannelInspector**) legible without crowding at typical clinical display sizes? Does the **ProbStrip** present probability cleanly without chart-junk? Are toolbar controls grouped logically rather than as one long undifferentiated row? Is the **ChannelInspector** showing only what is relevant to the selected channel/event? | Toolbar packs montage, three filters, sensitivity, timebase, threshold, AI source, and Export into one dense row with no grouping, increasing visual search time. | 2 | Toolbar; overall layout |
| **9** | **Help users recognise, diagnose, and recover from errors** — express errors in plain language, indicate the problem, suggest a solution. | When per-window ICA on a non-cached file emits warnings, are these surfaced as **plain-language, actionable messages** (or do they leak as raw console/stack warnings the reviewer cannot interpret)? If inference fails or is cancelled, does the GUI explain the resulting state of the ProbStrip/EventList? If Export fails (path/permissions), is the reason and remedy stated? | ICA warnings on non-cached files surface as unfiltered technical text with no guidance; a cancelled inference leaves the ProbStrip blank with no explanatory message. | 3 | Inference path; ProbStrip; Export |
| **10** | **Help and documentation** — provide searchable, task-focused help even though the system should be usable without it. | Is there any in-app guidance for the reviewer loop (open → infer → step through events → accept/reject/edit → export)? Is the meaning of the **raw uncalibrated** probability on the ProbStrip explained (so reviewers do not over-trust it)? Is the distinction between **Baseline** and **ZUNA full** sources, and which output file each writes, documented where the reviewer needs it? | No in-app help; the ProbStrip gives no indication that probabilities are uncalibrated raw softmax, risking misplaced trust; AI-source semantics undocumented in the UI. | 3 | ProbStrip; AI-source selector; Help (absent) |

---

## 3. Scoring and Aggregation Method

**1. Independent collection.** Each of the 2–3 experts completes the table independently on the same
fixed set of cached sample EDFs, recording for every issue: the heuristic violated, a free-text
description, the screen/control, and a severity (0–4). Working independently first prevents one
evaluator anchoring the others.

**2. Merge into a master issue list.** After the independent passes, the evaluator findings are
collated into a single de-duplicated list. Two findings are treated as the **same issue** when they
concern the same control/behaviour and the same heuristic; otherwise they are kept separate. Record
how many evaluators independently flagged each issue (its *detection count*) — issues found by more
than one evaluator are typically the most robust.

**3. Aggregate severity.** For each merged issue, compute the **mean severity** across the evaluators
who rated it (rounding to one decimal place); also retain the **maximum** severity any evaluator
assigned, since a catastrophe seen by one expert still warrants attention. Use the mean to rank and
the max as a tie-breaker / escalation flag.

**4. Rank and prioritise.** Sort the master list by mean severity descending. The priority bands are:
**P1 = mean ≥ 3.5 or any max = 4** (fix before any reviewer trial), **P2 = mean 2.5–3.4** (fix in the
hardening sprint), **P3 = mean 1.5–2.4** (fix opportunistically), **P4 = mean < 1.5** (cosmetic /
backlog). Where two issues tie, the one with the higher detection count ranks first.

**5. Report.** Present (a) the ranked master table, (b) per-heuristic counts of issues (to show which
principles the GUI most often violates), and (c) the count of issues in each priority band. This gives
both an actionable fix list and a defensible summary statistic for the thesis.

---

## 4. Feed-forward into WS2 GUI Hardening

The ranked master issue list is the direct backlog for WS2. Each P1 and P2 finding becomes a tracked
hardening task scoped against the specific widget or control it implicates — for example, surfacing
keyboard shortcuts and the numeric threshold value (Heuristics 6/7), adding undo for an
accidental Reject and revertible region edits (Heuristic 3), guarding Export against empty/wrong-source
writes and pre-flagging non-cached files (Heuristic 5), distinguishing model-load from inference
latency and persisting the active AI-source label (Heuristic 1), and adding plain-language messaging
plus a note that the probability strip is uncalibrated raw softmax (Heuristics 9/10). Because these
fixes target the inspected controls directly and require no model retraining, they keep the work
strictly within the inference-only, GUI-first scope of the thesis; the re-inspected GUI then carries a
measurably reduced high-severity issue count into the WS4 reviewer usability study.

---

### References
- Nielsen, J., & Molich, R. (1990). Heuristic evaluation of user interfaces. *Proc. CHI '90*, 249–256.
- Nielsen, J. (1994). Heuristic evaluation. In *Usability Inspection Methods*. Wiley.
- Nielsen, J. (1994). *10 Usability Heuristics for User Interface Design.* Nielsen Norman Group.
