# EDFbrowser feature comparison

*Requested by Omid, 2026-08-31: "Check features available in these tools" —
[EDFbrowser](https://www.teuniz.net/edfbrowser/), a free general-purpose
biosignal viewer, annotator and toolbox.*

The two programs are not competitors and the comparison is not a scorecard.
EDFbrowser is a general viewer for any time-series file — EEG, ECG, EMG,
bio-impedance — with a large toolbox for repairing, converting and measuring
files. This application does one thing: put a detector's seizure proposals in
front of a clinician and let them accept, reject or add. Most of EDFbrowser's
surface is therefore correctly absent here, and a few things are missing that
should not be.

Everything below is judged against one question: **does it help a clinician
review a day of ambulatory EEG for seizures?**

---

## What both have

| capability | EDFbrowser | here |
|---|---|---|
| montages | user-definable | longitudinal bipolar, transverse bipolar, referential, common average |
| high-pass / low-pass / notch | 1st–8th order Butterworth, Chebyshev, Bessel, moving average; adjustable-Q notch | 2nd-order HP, 4th-order LP, fixed-Q notch, zero-phase |
| power spectrum | FFT, selectable windows and lengths | Welch, Hann, 50 % overlap, per-channel, with band powers |
| annotations in / out | import and export events | reads and writes TUSZ `.csv_bi`, with a provenance sidecar |
| zoom and time cursor | rectangle zoom, crosshairs | sweep presets, drag-zoom, hover time cursor |
| sensitivity | per-signal amplitude | µV/mm, the clinical convention |

Two differences inside that table are deliberate. Our filters are **zero-phase**
(`sosfiltfilt`, forward and back), because a phase shift moves a spike relative
to the trace a clinician is localising from; EDFbrowser offers more filter
families, we offer the one property that matters for morphology. And our
filters are **display-only** — the detector always scores the raw referential
signal, so nothing a reviewer does to the display can move a probability. In a
general viewer that distinction does not arise; in a tool built around a model
it is the central safety property, and it is enforced by tests.

---

## Worth adding, in priority order

### 1. A whole-recording spectrogram (EDFbrowser calls it CDSA)

**Colour Density Spectral Array** — power by frequency by time, the whole
recording in one strip. This is the standard way long EEG is navigated in ICU
and ambulatory practice: seizures show as vertical bands, and a reviewer scans
hours in seconds.

It is the single most valuable thing on this list, because it addresses the
problem day-long recordings actually create. We have a probability strip, which
shows *what the model thinks*. A CDSA shows *what the signal is doing*,
independently — so a reviewer can find events the detector missed, which is
precisely the failure demonstrated by demo recording 5 (three seizures, nothing
proposed).

Display-only, so it cannot affect scoring. The spectrum machinery already
exists in the channel inspector; the work is computing it in bands over the
whole recording and drawing it as an image. Bounded like everything else:
compute once per recording, cache beside the probabilities.

### 2. Embedded EDF+ annotations

Real recordings carry them and we ignore them. The Nihon Kohden recording Omid
circulated has **41**: photic stimulation frequencies, hyperventilation start
and end, technician notes ("hoest" — cough), electrode events. We display none
of it, because the app reads TUSZ `.csv_bi` reference files only.

That matters clinically and not only cosmetically. A run of rhythmic activity
during photic stimulation or hyperventilation is an *activation procedure*, not
necessarily a seizure — and a reviewer judging a proposal without knowing the
technician was flashing a strobe at the patient is missing the context that
decides the call. It is also the most likely reason for a false positive in
that exact recording.

Small: pyedflib already returns them (`readAnnotations`), and the reference
overlay that draws `.csv_bi` intervals can draw them in a second colour. They
must be visually distinct from ground truth and from reviewer events — three
kinds of mark with different authority.

### 3. Amplitude-integrated EEG (aEEG / CFM)

A compressed amplitude trend over hours, standard in neonatal and ICU
monitoring. Cheaper than a spectrogram and answers a different question: where
did the background change. Useful for the same day-long navigation problem, and
a natural companion to item 1 rather than a substitute.

### 4. Print / export to PDF

EDFbrowser prints to printer, image or PDF. We export annotations but produce
no report. A review that ends in a file another clinician can read is the
deliverable in a clinical workflow, and a PDF of the accepted events with their
traces is what a consultant would expect to receive.

### 5. Spike / glitch filter

EDFbrowser has a filter for spikes, glitches and pacemaker impulses. Ambulatory
recordings collect electrode pops and movement artefact, and both drive false
positives. Display-only, and it would help a reviewer decide whether a proposal
is cerebral — though note the detector will still have scored the unfiltered
signal, so this informs the reviewer rather than changing the answer.

---

## Deliberately not adopted

Recorded so the omissions read as decisions rather than gaps.

| feature | why not |
|---|---|
| **Video playback sync** | Would be genuinely decisive for seizure review — semiology often settles what the EEG cannot. Not adopted because the corpora available here have no video, not because it lacks value. The strongest candidate on this list if video data ever arrives. |
| **Hypnogram / sleep staging** | A different clinical question. Sleep staging matters for seizure work only indirectly (state-dependent activation) and would be a project of its own. |
| **Pan-Tompkins QRS detection, RR intervals** | Cardiology. The one EEG-adjacent use is recognising ECG artefact in the trace, which the channel inspector's spectrum already supports. |
| **Waveform averaging on triggers** | Evoked potentials. Nothing to average in spontaneous seizure review. |
| **Multiple files side by side** | A review tool should show one patient. Two recordings on screen is how the wrong patient's EEG gets read as this one's — a failure this app already guards against elsewhere by naming both files when a load fails. |
| **Header editor, format validator, EDF+D→C, BDF→EDF converters** | File repair and conversion. Correct in a general toolbox; in a clinical review tool an editable header is a way to corrupt provenance. |
| **Z-EEG** | Niche, and not part of the seizure-review workflow this tool is scoped to. |
| **File splitter / resampler** | Tempting as a workaround for long-file memory limits, but the right fix is reading recordings span by span (`gui/io/edf_source.py`), not asking a clinician to cut files up first. |
| **Streaming / live monitor** | This is a review tool for completed recordings. Live monitoring is a different regulatory and engineering problem entirely. |

---

## What this application has that EDFbrowser does not

Worth stating, because the comparison otherwise reads one way only.

- **A detector, and a workflow around its proposals.** Accept, reject, adjust
  extent, add an event the detector missed, with the review state preserved
  across filter and montage changes.
- **Model-input transparency.** The "Detector input (ICA)" view re-runs the
  detector's own artefact removal so a reviewer can see literally what the
  model was fed — not an approximation of it.
- **Blind mode.** Hides ground truth so a reviewer's decisions can be collected
  without the answer key on screen, which is what makes a usability study
  measurable.
- **Provenance on export.** A reviewed file records who reviewed it, what the
  model scored, and which events were human-originated, stamped so it cannot
  later be mistaken for clinical ground truth.
- **A refusal when a recording will not fit in memory**, with the numbers, and
  an electrode map tied to the montage.

---

## Suggested order

1 (spectrogram) and 2 (embedded annotations) before any others: one solves
day-long navigation, the other is a correctness gap on real recordings we can
already demonstrate. Both are display-only, so neither can touch the numbers.
