# Draft email to Omid — progress update + deployment plan

**Before sending:** fill in your email address, and delete this header block.

This email is designed to close the three things Omid raised on the progress report: the missing
updates, the "does the student really have a plan to deploy" concern, and the "specific
requirements before external review" request.

---

**To:** omid.kavehei@sydney.edu.au
**From:** `<STUDENT_EMAIL>`
**Subject:** Thesis update: 19-channel detector replicates (AUC 0.881 vs 0.84) + plan to a clinician-ready prototype

Dear Prof. Kavehei,

Apologies for the gap in updates since the progress report — that was a fair criticism and I've
fixed the cadence. I'll send a short update every fortnight from here.

Three things: a result, the deployment plan you asked for, and a finding on the ICA stage you flagged.

## 1. The detector replicates the 2022 paper

I had reported that the work was "not a reproduction" of the source paper. **That was wrong, and
it was wrong for an interesting reason.** I had been comparing against the two-channel SPMB 2020
paper, but the pretrained weights (`convlstm_ICA_12_train.h5`) are 19-channel, 12-second and
ICA-denoised — which is the *continental generalization* configuration (ESWA 207:118083), not the
two-channel one.

Scored against that paper instead, using its own window-labelling protocol:

| | AUC | 95 % CI |
|---|---|---|
| **This reconstruction** | **0.89** | [0.83, 0.94] |
| Published (Table 2, TUH v1.5.1) | 0.84 | — |

Measured across the 206 annotated TUSZ v2.0.0 files available locally — 27.8 hours, 85 seizures.
(99 further recordings have no annotation file at all and are excluded; an absent annotation is
not evidence of a seizure-free recording.) The published value sits inside my confidence
interval, so the honest claim is that the reproduction is **statistically indistinguishable**
from the published result — not that it beats it.

Two things had been hiding this. First, an evaluation-protocol mismatch worth ~0.09 AUC: the
paper's feature loader only ever emitted windows lying *entirely* inside one annotated interval,
whereas I had been labelling any window that overlapped a seizure as positive. Second, sample
size — on my original 26-file subset the confidence interval was [0.673, 0.925], too wide to
distinguish anything.

I also found and fixed three scoring defects that had inflated the reported false-alarm rate by
2.1× (478 to 223 per 24 h), and I've since audited the ICA stage in detail (below).

## 2. Requirements the system must meet before any external review

You asked for these specifically. My list, all of which are days rather than weeks:

1. **An on-screen statement of what it is** — "research prototype, not for diagnostic use", in
   the window title, the status bar, and the header of every exported annotation. (Done.)
   Event sensitivity at the default threshold is 49 %.
2. **No screen that states something false.** Two exist today: windows the pipeline *refused* to
   score render as confident zeros (one recording is 49/49 skipped and contains a real 27-second
   seizure, but draws as a flat zero strip), and the raw softmax is labelled "p(seiz)" and written
   into the `confidence` column of human-confirmed annotations.
3. **No silent loss of reviewer work** — opening a second file currently discards a review with
   no prompt.
4. **Export guardrails** — export can currently overwrite the ground-truth file it read from.
5. **A frozen demo set** that cannot trigger a slow inference mid-session. (This one is now done —
   25 files qualify.)
6. **Full provenance on every export** — recording hash, model hash, threshold, git commit, so any
   output is traceable to what produced it.

I'd propose the clinician session happens once 1–4 are done — that is roughly two weeks of
small, specific changes, not a rewrite. I'll run the cognitive walkthrough myself first, since it
needs one evaluator and generates the real pre-clinician fix list.

## 3. On the ICA stage — a finding worth discussing

You flagged per-window ICA in your feedback and you were right to. Measured over 98 real windows:
FastICA **fails to converge on 77 %** of them; the 0.1 Hz pre-filter is not realisable on a
12-second window (MNE reports `filter_length 8251 > signal 3000` on essentially every call) and is
below the 1–2 Hz that the ICA literature recommends; 19 components from 3000 samples is 2.4×
short of the standard kN² requirement; and half the components flagged as ocular are silently
discarded before removal.

My inclination is to **report this rather than fix it**, because the training features were
generated through this same function, so the non-convergence is part of the operating point —
capping `max_iter` already flips detections across threshold (one goes 0.902 → 0.0014). Turning
ICA off entirely slightly *improves* window AUC (0.742 vs 0.716) and runs 30× faster. Happy to be
argued out of this if you think the fix is worth the re-validation.

## 4. What this means for BMET4112

Profiling says the expected C++ target is the wrong one: `ica_arti_remove` is **~95 %** of runtime
and the ConvLSTM is **~2 %**, so a perfect C++ network buys about 3 % end-to-end. The defensible
scope is a **C++ FastICA front end** behind a pybind11 binding, with the profile that justified
the choice as the opening figure — and a numerical-agreement study, not just a speed table.

Could we meet briefly in the next week or two to confirm the priority order before I start the
clinician-facing work?

Kind regards,
Sam Gan
SID 520478644 · BMET4111
`<STUDENT_EMAIL>`
