# Intended use, limitations, and what this software must not be used for

This document travels with the application. It is written for the clinician who
opens it, the supervisor who signs off on it, and the ethics reviewer who has
to decide whether it may be used on patient data at all.

Every number here comes from `docs/RESULTS.md`, which is the single source of
truth for this project. If the two ever disagree, `RESULTS.md` is correct and
this file is stale.

---

## 1. What this software is

A **review assistant for human-in-the-loop EEG seizure annotation.**

It reads a 19-channel EEG recording, runs a pretrained convolutional-LSTM
detector over 12-second windows, and presents the resulting candidate events to
a qualified reviewer, who accepts, rejects, edits, or adds events. The output is
**the reviewer's annotation**, not the model's.

It is a **research prototype** built for a Master's thesis
(BMET4111, University of Sydney). It is used under supervision, on recordings
the reviewer would otherwise read manually.

## 2. What this software is not

- **It is not a medical device.** It is not registered with the TGA or any
  other regulator, has not undergone clinical validation, and carries no
  conformity assessment.
- **It is not a diagnostic tool.** It does not diagnose epilepsy, classify
  seizure type, or localise onset.
- **It is not an alarm or a monitoring system.** It does not run in real time
  and must not be relied upon to raise an alert.
- **It must not be used unsupervised.** No output may reach a patient record
  or influence a clinical decision without a qualified reviewer having read the
  underlying EEG.

## 3. Intended users

A clinical neurophysiologist, epileptologist, or EEG technologist **already
competent to annotate EEG unaided**. The software reduces the time spent
scanning quiet background; it does not substitute for the expertise needed to
judge what it proposes.

## 4. Intended data

- 19-channel scalp EEG in the standard 10–20 montage, EDF format
- Recordings from adult and paediatric ambulatory or outpatient monitoring
- Sampling rate of 250 Hz or higher

All 19 electrodes are required. Recordings missing any channel are rejected
rather than scored on a partial montage.

Anything not already at 250 Hz is resampled to it, **including rates below
250 Hz**. The software does not refuse a low-rate recording, but upsampling
invents no information: the detector was trained at 250 Hz and its behaviour on
a recording that was originally sampled lower has not been evaluated. Treat
such results with suspicion.

## 5. Measured performance, and how to read it

Measured on 206 annotated recordings from the TUH Seizure Corpus
(27.8 h, 85 reference seizures, 28 patients). Full method in
`docs/RESULTS.md`.

| | Value |
|---|---|
| Window-level AUC | **0.89**, 95 % CI [0.73, 0.95] by patient |
| Published AUC (source paper) | 0.84 — inside the interval |
| Event sensitivity @ threshold 0.5 | **49.4 %** (42 of 85 seizures) |
| False alarms | **204.4 per 24 h** |

**The two numbers that matter to a reviewer:**

> **About half of seizures are missed.** At the default threshold the detector
> finds 42 of 85. A recording with no proposed event has **not** been shown to
> be seizure-free. The software cannot be used to skip recordings.

> **Most proposals are wrong.** At ~204 false alarms per 24 h, the large
> majority of what is presented is not a seizure. This is expected; the
> reviewer's job is to reject them. It is why the interface is built around
> fast rejection.

**19 of the 85 reference seizures (22 %) produce no model response at all** —
their peak window score never exceeds 0.01. That is over half of everything the
detector misses. Lowering the threshold does not recover them: reaching them
would require a setting that also floods the reviewer with background. This is
a limitation of the model's discrimination, not of the chosen operating point,
and no setting in the interface can work around it.

### Calibration

The model's raw output is **not** a probability. A score of 0.5 corresponds to
roughly a 29 % chance the window is ictal. The interface therefore presents the
score as a **relative ranking**, never as "the AI is 50 % confident". Post-hoc
Platt scaling reduces calibration error by ~85 % (ECE 0.072 → 0.011) but is
**not applied** in the shipped application — see `docs/RESULTS.md` §8 for why.

## 6. Populations and settings not evaluated

Performance is unknown, and may be substantially worse, for:

- Neonatal EEG
- Intracranial or subdural recordings
- Intensive-care or status-epilepticus monitoring
- Recordings under 60 seconds
- Montages other than standard 10–20 scalp placement
- Any population not represented in the TUH corpus

The source paper's headline Australian result — 76.68 % sensitivity at ~56
false alarms per 24 h — **does not describe this software** and must not be
quoted as if it did. That figure is from 14,590 hours of private Australian
clinical data, using a **20-channel** model (19 EEG + ECG), an additional
post-processing stage, and the SDR metric, which merges false alarms occurring
within 30 seconds of each other into one. This application is the 19-channel
detector alone, measured on 27.8 hours of public data with per-event matching.
Different data, different model, different metric. See `docs/RESULTS.md` §1,
which sets out why the two numbers resemble each other only by coincidence.

For TUH — the only public data the source paper reports on — the paper
publishes **an AUC and nothing else.** There is no published TUH sensitivity or
false-alarm figure to compare against.

## 7. Failure modes the reviewer should expect

| What you see | What it means |
|---|---|
| No events proposed | Either genuinely quiet, or a missed seizure. Not evidence of absence. |
| A dense run of proposals | Usually movement or electrode artefact, not a seizure cluster. |
| "not assessed" in the probability strip | The window was **never scored** — interrupted signal, flat channels, or ICA failure. It is not a confident negative. |
| A proposal with no score | A human-added event. The model never saw it as a candidate. |

The distinction in row three matters: before it was made explicit, a refused
window and a confident "no seizure here" looked identical in the interface.

## 8. Provenance and auditability

Every exported annotation records the model weights' SHA-256, the software
commit, the reviewer's identifier, the detection threshold, and the per-event
status history. An annotation whose provenance cannot be reconstructed should
not be trusted, and the export is designed so that this can always be checked.

## 9. Privacy

The application performs **all computation locally**. No recording, annotation,
or derived data is transmitted anywhere. There is no network access, no
telemetry, and no cloud component. This is a deliberate architectural choice —
see `docs/DEPLOYMENT.md` §1.

Recordings are read from wherever the user opens them. The application writes
only a probability cache (next to the recording, or under
`%LOCALAPPDATA%\SeizureReview\` if that location is read-only) and a crash log.
De-identification of the source recordings is the responsibility of whoever
supplies them.

## 10. Reporting a problem

Failures are logged to `%LOCALAPPDATA%\SeizureReview\logs\seizure_review.log`.
That file plus the recording identifier is enough to diagnose most issues.
