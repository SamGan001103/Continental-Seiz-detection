# Timed Peer-Review Usability Protocol

**Project:** BMET4111 Thesis — A Human–AI teaming GUI for reviewing ambulatory / outpatient EEG
**Author:** Sam Gan (University of Sydney)
**Supervisor:** Prof. Omid Kavehei
**Document:** `docs/usability/peer_review_protocol.md`
**Companion document:** `docs/usability/cognitive_walkthrough.md` (task definitions; see §2)
**Status:** Protocol specification. **No data may be collected under this protocol until the relevant ethics pathway is cleared (see §5.6).**

---

## 1. Study aim, design and measures

### 1.1 Aim

The thesis deliverable is the reviewer graphical user interface (GUI) — an early minimum-viable product (MVP) — together with its usability evaluation. The detector behind the GUI is *not* the object of study; detector accuracy is out of scope for this protocol. The aim of the timed peer review is to obtain an early, quantified indication of whether a non-clinical reviewer can learn and operate the core reviewer loop (open an EEG record, inspect proposed seizure events against the signal and probability strip, accept/reject/edit those events, and export a reviewed annotation file) efficiently and with few errors, and to capture a standardised subjective usability score for the MVP.

The task is deliberately framed to participants as **outpatient / ambulatory-EEG triage usability**: the participant plays the role of a technologist or reviewer who is triaging candidate seizure events that an AI has proposed, deciding which to keep, and exporting a corrected record. Participants are *not* asked to make any clinical judgement and are *not* told that their accept/reject decisions are clinically correct or incorrect; the dependent variables concern *operating the interface*, not diagnostic performance.

### 1.2 Design

- **Type:** Within-subjects (every participant performs the same fixed set of tasks, in the same fixed order, on the same frozen demo set). There is no between-groups manipulation; the AI-source selector (Baseline vs ZUNA) is exercised *as a task step*, not as an experimental factor.
- **Participants:** 3–5 non-clinical engineering-student peers (convenience sample). This is an informal expert-adjacent usability check, not a powered clinical trial; the small *n* is appropriate for early formative MVP feedback and is reported honestly as such. Participants are peers of the author and are **not** clinicians, neurologists or EEG technologists, and **no patient or clinical data is used** (see §1.4 and §5.2).
- **Order:** Fixed task order for all participants (no counterbalancing), because the tasks are sequential steps of a single reviewer workflow rather than independent conditions.
- **Sessions:** One session per participant, approximately 30–40 minutes, conducted one-to-one with the facilitator (the author). Sessions are run on the same machine and the same frozen, cached demo set so timings are comparable across participants (see §5).

### 1.3 What is measured

For each participant, per task:

1. **Per-task completion time** — wall-clock seconds from the facilitator saying "begin" for the task to the participant signalling completion (or to the facilitator calling a time-out / assist). Timed by stopwatch and logged in the table in §2.
2. **Errors / wrong turns** — count of discrete deviations from an efficient path: e.g. activating the wrong toolbar control, opening the wrong widget, mis-reading the probability strip, accepting an event the task asked to reject (or vice versa), or needing the same hint twice. Each is tallied and briefly noted.
3. **Task success** — a 3-level rating: **Success** (completed unaided), **Assisted** (completed only after a facilitator hint), **Fail** (not completed / timed out). Time-out threshold is **3 minutes per task** unless noted otherwise.

After all tasks, per participant:

4. **Subjective usability** — the standard 10-item System Usability Scale (SUS), 1–5 Likert (§3), yielding one 0–100 SUS score per participant. Group SUS is reported as mean and range (not as a significance test, given the small *n*).

### 1.4 Scope and honesty notes (must be reflected when results are written up)

- This is a **formative, small-*n*, non-clinical** usability check. Results indicate learnability and interaction friction of the MVP; they do **not** establish clinical usability, safety, or that the underlying detector is fit for purpose.
- The live detector is a single 12-second, 19-channel, ICA-denoised ConvLSTM and is **not** a reproduction of the source paper's blended multi-time, 2-channel method. The probability strip shows **raw, uncalibrated** softmax. None of this is the subject of the usability study, but participants' comments about trust/confidence should be interpreted with these limitations in mind and must not be reported as endorsements of detector accuracy.
- All evaluation uses **public data only** (TUH/TUSZ-derived demo files); **no patient or RPAH data** is involved.

---

## 2. Fixed task script and timing / error log

The task definitions below are the same workflow steps used in the **cognitive walkthrough** (`docs/usability/cognitive_walkthrough.md`). They are reproduced here in condensed, *participant-facing* form so this protocol is self-contained; if the two documents ever diverge, the cognitive-walkthrough document is authoritative for the canonical action sequence and the present document governs timing/scoring.

All tasks are performed on the **frozen demo set** (cached `.probs.npz` files) so inference is instant and warning-free (see §5.2). Files referenced as `DEMO_FILE_A`, `DEMO_FILE_B` are fixed members of that set, recorded in the session log so every participant uses the identical files.

### 2.1 Tasks (read each task aloud; start the stopwatch on "begin")

| # | Task (read to participant) | Efficient path / what "done" looks like |
|---|---|---|
| T1 | "Open the EEG recording `DEMO_FILE_A`." | Use Open; wait for on-demand inference progress dialog to complete (cached, so near-instant); SignalView, ProbStrip and EventList populate. |
| T2 | "The traces look noisy. Apply a 1 Hz high-pass, a 70 Hz low-pass and a 50 Hz notch filter." | Set High-pass, Low-pass and Notch in the toolbar; traces visibly change. |
| T3 | "Find the first proposed seizure event and jump to it in the signal view." | Use EventList (or J/K to move) and Enter to jump; SignalView centres on the first proposed event. |
| T4 | "Using the probability strip to guide you, decide whether this first event is a real-looking candidate and **accept** it." | Read ProbStrip alignment under the event; press Space (Accept); event state becomes *accepted*. |
| T5 | "Move to the next proposed event and **reject** it." | J/K to next event, Enter to jump, press X (Reject); state becomes *rejected*. |
| T6 | "The next event's box is too short — it starts too late. **Edit its extent** by dragging the region so it covers the whole burst." | Drag the event region handle in SignalView; state becomes *edited*. |
| T7 | "Lower the detection **threshold** until more candidate events appear, then raise it back. Confirm your earlier accept/reject decisions are still there." | Move THRESHOLD slider down (more events), back up; verify prior review states survive the threshold change. |
| T8 | "Switch the AI source from **Baseline** to **ZUNA full** and observe what changes in the event list / probability strip." | Use AI-source selector; ProbStrip / EventList update for the ZUNA source. |
| T9 | "**Export** your reviewed events to a file." | Use Export; a reviewed `.csv_bi` is written (ZUNA source writes `.zuna.reviewed.csv_bi`); only accepted + edited events are included. |
| T10 | "Open a second recording, `DEMO_FILE_B`, and tell me how many events are proposed at the default threshold." | Open `DEMO_FILE_B`; read event count from EventList (note: default threshold 0.5 yields few/zero events on some files — a correct answer may be "none"). |

> Facilitator note: if a task stalls past the 3-minute time-out, give one standardised hint (record as **Assisted**); if still not done, mark **Fail**, set up the correct end-state yourself, and move on so later tasks remain comparable.

### 2.2 Per-task timing + error log (one table per participant)

Copy this table for each participant. Record participant as **P1 … Pn** only (see §6).

**Participant ID:** ____   **Date:** ____   **DEMO_FILE_A:** ____   **DEMO_FILE_B:** ____   **Build / commit:** ____

| Task | Completion time (s) | Errors / wrong turns (count) | Error notes (brief) | Success (S / A / F) |
|------|--------------------:|-----------------------------:|---------------------|:-------------------:|
| T1   |                     |                              |                     |                     |
| T2   |                     |                              |                     |                     |
| T3   |                     |                              |                     |                     |
| T4   |                     |                              |                     |                     |
| T5   |                     |                              |                     |                     |
| T6   |                     |                              |                     |                     |
| T7   |                     |                              |                     |                     |
| T8   |                     |                              |                     |                     |
| T9   |                     |                              |                     |                     |
| T10  |                     |                              |                     |                     |
| **Total** | **Σ time** | **Σ errors** |  | **#S / #A / #F** |

**Facilitator observations (free text):**

---

## 3. System Usability Scale (SUS)

Administer immediately after T10. Read the instruction once: *"For each statement, mark the box that best reflects how you feel about the interface right now. Respond to every item; if you are unsure, give the response that feels closest. Don't dwell on any one item."*

### 3.1 The 10 items (verbatim), 1–5 Likert

Scale anchors: **1 = Strongly disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly agree.**

| # | Statement | 1 | 2 | 3 | 4 | 5 |
|---|-----------|:-:|:-:|:-:|:-:|:-:|
| 1 | I think that I would like to use this system frequently. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | I found the system unnecessarily complex. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | I thought the system was easy to use. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | I think that I would need the support of a technical person to be able to use this system. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | I found the various functions in this system were well integrated. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | I thought there was too much inconsistency in this system. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 7 | I would imagine that most people would learn to use this system very quickly. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8 | I found the system very cumbersome to use. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 9 | I felt very confident using the system. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | I needed to learn a lot of things before I could get going with this system. | ☐ | ☐ | ☐ | ☐ | ☐ |

*(SUS, Brooke 1996. The instrument is reproduced as standardly worded; "system" refers to the reviewer GUI.)*

### 3.2 SUS scoring formula

Let `xᵢ` be the raw 1–5 response to item *i*.

1. **Odd-numbered items (1, 3, 5, 7, 9):** score contribution = `xᵢ − 1`.
2. **Even-numbered items (2, 4, 6, 8, 10):** score contribution = `5 − xᵢ`.
3. Each item's contribution is now in the range 0–4.
4. **Sum** all ten contributions (range 0–40).
5. **Multiply by 2.5** to obtain the final SUS score on a **0–100** scale.

> Reporting note: a SUS score is *not* a percentage and *not* an accuracy. For context only, ~68 is the common reference "average". Report the per-participant SUS and the group **mean and range**; with 3–5 participants do **not** report inferential statistics or imply generalisability.

---

## 4. Open debrief questions

Ask verbally after the SUS; record short notes (and audio only if the cleared ethics pathway permits — otherwise notes only). Probe but do not lead.

1. **"Walk me through any moment where you were unsure what the interface was telling you or what to do next — especially anything to do with the probability strip or the proposed events."**
2. **"If you could change one thing about the accept / reject / edit / export workflow to make reviewing faster or less error-prone, what would it be?"**
3. **"How much did you trust the AI's proposed events, and what — if anything — in the interface affected that trust?"** *(Note for analysis: trust comments reflect the MVP's presentation only; the probability strip is currently uncalibrated, so these must not be read as statements about real detector accuracy.)*

---

## 5. Session logistics

### 5.1 Environment setup

- **Machine:** the development laptop, mains-powered, with other applications and notifications closed; single external display optional but, if used, identical for all participants.
- **Launch the GUI** from the `seiz36` environment:

  ```
  C:\Users\User\miniconda3\envs\seiz36\python.exe -m gui.main
  ```

  (PyQt5 desktop app; Python 3.6 / TF 1.15.)
- **Warm-up before the participant arrives:** the facilitator opens one demo file once so the model is loaded into memory, then closes/reopens to the clean start state. This avoids the participant's T1 timing being inflated by the one-time model load.
- Confirm the four widgets are visible and correctly laid out: **SignalView**, **ProbStrip**, **EventList**, **ChannelInspector**; and that the toolbar exposes montage, High-pass / Low-pass / Notch, sensitivity, timebase, the detection **THRESHOLD** slider, the **AI-source** selector (Baseline / ZUNA full) and **Export**.
- Have a stopwatch (or phone timer), the per-participant timing table (§2.2), and the SUS sheet (§3) ready.

### 5.2 Frozen demo set on cached files

- Sessions use the **frozen demo set only** — files that already have cached probabilities (`.probs.npz`) so they **load instantly and warning-free**. There are 39 such cached sample EDFs; a fixed subset (`DEMO_FILE_A`, `DEMO_FILE_B`, and any spares) is selected once, recorded in the session log, and used **unchanged for every participant**.
- **Do not** use non-cached files during a session: those trigger slow, sometimes warning-y per-window ICA and would make timings non-comparable and the experience unrepresentative of the intended demo.
- All demo files are **public (TUH/TUSZ-derived)**. No patient or RPAH data is loaded at any point.

### 5.3 Pre-session script (read verbatim to each participant)

> "Thanks for helping. You're going to try an early version of a desktop tool for **reviewing outpatient-style EEG recordings**. The tool runs an AI that proposes possible seizure events; your job is to play the role of a reviewer who opens a recording, looks at the AI's proposed events against the brain-wave traces and a probability strip, and decides which to **keep, reject, or adjust**, then exports the corrected record.
>
> A few things to keep in mind: **we are testing the software, not you** — there are no right or wrong answers about your performance, and if something is confusing that is exactly the kind of finding we need. I'll read you one short task at a time. Please **think aloud** as you go — tell me what you're looking at and what you expect to happen. I'll be timing each task and taking notes just so I can see where the interface helps or gets in the way; I won't help unless you're truly stuck. This is **not a clinical task** and nothing here involves real patient data — all recordings are from public research datasets. You can stop or take a break at any time, for any reason. Do you have any questions before we start?"

### 5.4 Session flow

1. Read the pre-session script (§5.3); answer questions; confirm willingness to continue.
2. Run T1–T10 (§2.1), reading one task at a time, timing and logging each (§2.2).
3. Administer the SUS (§3).
4. Ask the three debrief questions (§4).
5. Thank the participant; confirm what was recorded; close out.

### 5.5 Facilitator conduct

- Read tasks and the pre-session script as written; do not paraphrase the tasks differently for different participants.
- Offer **one standardised hint** at the time-out only; record any hint as **Assisted**.
- Stay neutral — no praise or correction that signals "right/wrong" clinical decisions.

### 5.6 Ethics gate (mandatory)

**Data collection under this protocol must not begin until the relevant University of Sydney human-research ethics pathway is cleared** (e.g. confirmation that the activity qualifies as low-/negligible-risk usability testing with non-clinical peers, or formal HREC approval as advised). Until that clearance is in writing, this document is a **specification only**; pilot rehearsals to refine wording must use the facilitator/author alone and must **not** record any third party's data.

---

## 6. Data-handling and anonymisation note

- **No identifying information is collected.** Do not record participants' names, email addresses, student IDs, faces, or voices (unless audio is explicitly permitted under the cleared ethics pathway and consented to in writing).
- Participants are referred to **only** by sequential codes **P1, P2, … Pn**. The mapping from code to person is **not** written down; if a contact list is unavoidable for scheduling, it is kept separately from the data and destroyed once sessions are complete.
- Stored artefacts (timing tables, SUS sheets, debrief notes) contain **only** the participant code, date, the demo file names, and the build/commit — no personal data.
- Any reviewed export files produced during a session are demo artefacts derived from **public** data and contain no participant identity.
- Data are stored on the University-managed device, reported only in **aggregate** (group SUS mean/range; per-task time and error distributions), and retained/disposed of in line with the conditions of the cleared ethics pathway.

---

*End of protocol.*
