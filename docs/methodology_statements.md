# Methodology Statements: Honest Framing of the Detector, the ZUNA Side-Study, and the Reviewer-in-the-Loop Contribution

*BMET4111 Thesis — Sam Gan, University of Sydney. Supervisor: Prof. Omid Kavehei.*
*Scope note: All quantitative results below were obtained on public TUH/TUSZ data only. No patient or RPAH data were used. The thesis deliverable is the reviewer GUI (a minimum viable product) and its usability evaluation, not detector accuracy.*

This document records the methodological positions the thesis takes. Its purpose is to state plainly what was implemented, what was measured, and what may and may not be claimed, so that the evaluation chapter can be read as a rigorous and defensible account rather than as an overclaimed reproduction. The guiding policy throughout is rigorous honesty: no fabricated results, no overclaiming, and an inference-only methodology that never proposes retraining.

---

> **This document was revised after two corrections.** It previously (a) compared the
> implementation against the *two-channel* SPMB 2020 paper and concluded it was "not a
> reproduction", and (b) quoted event-level figures computed before three scoring defects were
> fixed. Both have been corrected in place — the false statements are gone, not merely annotated.
>
> **Numbers live in `docs/RESULTS.md`**, which is the single source of truth. The evidence behind
> the replication claim is in `docs/reproduction_status.md`. If a figure here ever disagrees with
> `RESULTS.md`, `RESULTS.md` wins.

## 1. Relationship to the source method

The detector that runs live inside the reviewer GUI is a single 12-second, 19-channel, ICA-denoised ConvLSTM, loaded from the pretrained weights `convlstm_ICA_12_train.h5`. Each inference window is a short-time Fourier transform of fixed shape `(23, 19, 125)`, which matches the input tensor of the saved model exactly. The model therefore loads cleanly and runs as trained: there is no loading error, no shape mismatch, and no silent reshaping or padding workaround in the inference path. When the GUI proposes events, it is reporting the genuine output of this network on the supplied EEG, not the output of a partially-wired or misconfigured pipeline. This point is stated up front because a reader who knows the source paper might otherwise assume that any divergence in numbers reflects a broken implementation; it does not.

**The repository descends from two NeuroSyd papers, and these weights belong to the second one.**
The 2020 SPMB paper describes a *two-channel* detector on a *blended multi-time* (3 s / 5 s / 7 s)
spectrogram. The 2022 *Expert Systems with Applications* paper, "Continental generalization of a
human-in-the-loop AI system for clinical seizure recognition", describes a **19-channel,
12-second, ICA-denoised ConvLSTM** — which is precisely what `convlstm_ICA_12_train.h5` is, and
precisely what §2.3 of that paper specifies (12-second segments, ICA into 19 components, Fp1/Fp2
correlation for EOG rejection, STFT with a 250-sample window at 50 % overlap, DC removed,
`(n × 23 × 125)`, MNE v0.20, Python 3.6).

The two-channel scaffolding in the repository (`utils/load_data_elec_{3,5,7}s.py` and their
`_dev_` counterparts) is unused dead code from the upstream project, and the two-channel weights
the SPMB method needs are absent. That method is genuinely not runnable here — but it was never
the right target.

**Against the correct target, the detector replicates.** Scored under the source paper's own
window protocol, the pooled window-level AUC over the 206 annotated TUSZ files (27.8 h) is
**0.89**, 95 % CI [0.83, 0.94], against the **0.84** that paper reports for TUH v1.5.1 in its
Table 2. The published value lies inside the interval. The thesis is therefore entitled to
claim a validated reconstruction of the 19-channel detector at the window level — the only level
at which that paper makes a checkable public-data claim.

Two limits on that claim, both of which must be stated wherever it is made. First, the comparison
is TUSZ v2.0.0 `eval` here against v1.5.1 `dev` there — the same corpus family and detector
configuration, but not the same files. Second, the paper's **RPAH** figures (76.68 %
sensitivity, 56.55 false alarms per 24 h, 92.19 % with a human arbiter) are **not reproducible
here and never will be**: they require private clinical data under hospital ethics approval and,
per `utils/ICA_load_data_elec.py:285`, a 20-channel model (19 EEG + ECG).

For the current value of every number, see **`docs/RESULTS.md`**, which is the single source of
truth; `docs/reproduction_status.md` records the evidence behind the replication claim.

---

## 2. Two metrics, two conclusions: reporting ZUNA honestly

ZUNA is reported as an exploratory side-study, and it is reported with both of the metrics that were computed for it, because those two metrics disagree about whether ZUNA helps.

At the **event level**, on ten public TUSZ seizure files scored event-wise at the default threshold of 0.5, ZUNA improves the headline detector behaviour: sensitivity rises from 21.1 % for the baseline to 26.3 % with ZUNA (4 → 5 of 19 reference seizures), and the false-alarm rate falls from 41.1 to 0.0 per 24 hours. Read in isolation, this is a favourable result.

At the **window level**, the picture reverses. Pooling all windows and computing the area under the ROC curve on the per-window probabilities gives a baseline AUC of 0.6878 against a ZUNA AUC of 0.6466. By this discrimination-quality measure ZUNA is *worse*: it degrades the model's ability to rank seizure windows above non-seizure windows. The event-level gain therefore appears to come from a change in where events land relative to the threshold rather than from a genuine improvement in the underlying probability ranking, and a single operating-point event metric can move favourably even as the threshold-independent measure deteriorates.

> The earlier version of this section quoted "26.3 % → 31.6 %" and "328.7 → 205.4 false positives per 24 hours". **Those event-level figures were wrong** — they predate the scoring fixes recorded in `docs/reproduction_status.md` §3, and the false-alarm rates in particular were badly inflated. The numbers above are the re-scored values from the same stored probability caches (`experiments/rescore_zuna_compare.py`). The window-level AUCs were never affected, because AUC does not depend on event scoring.

Presenting only the event-level numbers would be misleading, so the thesis presents both and states the disagreement explicitly. Given that ZUNA is, as deployed here, an exploratory repurposing of a general EEG super-resolution model into a seizure front-end — and given that it costs roughly six times real-time and peaks at approximately 42 GiB of RAM, which is why only 10 of the 26 candidate files could be processed at all — the honest conclusion is that the ZUNA side-study is **inconclusive at best and mildly negative on the more rigorous metric**. The recommendation that follows is to scope ZUNA *down*, presenting it as a documented negative or inconclusive exploration with its costs and its window-AUC regression stated, rather than to expand it or to promote the single favourable event metric into a headline claim. ZUNA is explicitly not a validated seizure detection front-end, and the thesis will not describe it as one.

---

## 3. The "money figure": reviewer-in-the-loop event recovery

The single primary figure around which the thesis should be built is **reviewer-in-the-loop event recovery**: the number of true seizures missed by the AI acting alone, compared against the number recovered once a human reviewer has stepped through the proposed events and made accept, reject, and extent-editing decisions, as captured in the exported reviewed `.csv_bi`.

Concretely, the figure plots, per file (and pooled across the public TUSZ subset), the ground-truth seizure events against three derived counts: (i) the events the detector proposes and that survive at the operating threshold — the AI-alone yield; (ii) the events present in the exported reviewed `.csv_bi` after the reviewer has accepted, rejected, and edited candidates; and (iii) the difference between them, which is the set of true seizures *recovered* through the human-in-the-loop interaction that the AI alone would have missed or mis-bounded. The recovered set includes both events the reviewer accepted that a naive threshold reading would have buried, and events whose temporal extent the reviewer corrected by dragging the region so that they now match the reference. The exported reviewed file is the natural data source because it already encodes exactly the accepted and edited events and nothing else, so the figure is a direct read-out of the reviewer's decisions rather than a reconstruction.

This figure is the right thesis centrepiece precisely because it binds together the three parts of the work into one argument. The detector is deliberately weak and is acknowledged as such; the GUI presents that weak detector's output as inspectable, navigable candidates through the SignalView, ProbStrip, and EventList; and the usability story is what lets a reviewer convert raw, low-sensitivity model output into a clinically more complete event set. The contribution is the human-in-the-loop interface, and event recovery is the metric that makes that contribution visible: it shows the interface adding value *on top of* the detector, which is the claim the thesis is actually entitled to make.

> **Implementation note (2026-08-10).** This figure was unmeasurable until now. The GUI derived
> its entire worklist from suprathreshold runs and exported only `accepted`/`edited`, so a
> reviewer could only ever *subtract* from the detector's list — "recovered" could never mean more
> than an extent correction on something the AI had already found. With 49 % event sensitivity and
> 22 % of seizures scoring essentially zero, the most important half of the figure was
> unreachable. A reviewer can now originate an event (**N**), it carries the distinct status
> `added` with no model score, it survives threshold rebuilds, and it is exported and flagged
> `human_originated: true` in the provenance ledger. The figure's three counts are therefore:
> (i) `accepted` + `edited` — the AI proposed it and the human confirmed it; (ii) `added` — **the
> human found it and the AI missed it entirely**, which is the recovery term; (iii) `rejected` —
> the human removed a false alarm. Undo is **Ctrl+Z**; **Ctrl+R** reverts an extent to the
> detector's.

The figure is defensible and consistent with the inference-only policy. It requires no retraining and no change to the model; the detector weights are frozen and the probabilities are produced exactly as in normal operation. It is computed entirely on public TUSZ data and ground truth. It does not depend on the detector being good — indeed it is most informative when the detector is mediocre, because the recovery gap is what the interface is for. And because it is derived from the exported reviewed `.csv_bi`, it measures a real reviewer's real decisions through the real GUI, which is the artefact under evaluation, rather than a simulated or idealised oracle.

---

## 4. Claims we will and will not make

**Claims the thesis will make:**

- The reviewer GUI is the contribution: a working minimum viable product for human-AI teamed review of ambulatory and outpatient EEG, evaluated for usability on public TUSZ data.
- The live detector is a single 12-second, 19-channel ICA ConvLSTM (`convlstm_ICA_12_train.h5`) that loads and runs correctly as trained, with the STFT input shape `(23, 19, 125)` matching the saved model.
- **The 19-channel detector of the 2022 continental-generalization paper is validly reconstructed**, and reproduces that paper's one publicly checkable claim: pooled window AUC **0.89** (95 % CI [0.83, 0.94], 206 annotated files, 27.8 h) against 0.84 published for TUH, when scored under the paper's own window protocol. The published value lies inside the interval, so the honest claim is *statistically indistinguishable from*, not *better than*.
- Reproducing the source method's *decision stage* rather than its model lowers the false-alarm rate **2.1×** (478.0 → 222.9 per 24 h at threshold 0.5). Separately, fixing three event-scoring defects lowered the previously reported figure again; the two must not be combined into one multiplier. A scoring convention moved a headline number by more than most methodological differences do, which is a result in its own right.
- All reported detector numbers describe this model's behaviour on a public TUSZ subset.
- A human reviewer, working through the GUI, recovers true seizure events that the AI alone misses, as evidenced by the exported reviewed `.csv_bi`; this reviewer-in-the-loop recovery is the primary result.
- ZUNA is an exploratory side-study whose event-level and window-level metrics disagree, and which is inconclusive-to-negative once its window-AUC regression and its roughly 6x real-time and ~42 GiB cost are accounted for.

**Claims the thesis will not make:**

- That the implemented system reproduces the **two-channel** SPMB 2020 blended multi-time 3s/5s/7s voting method, or its published performance — that method was never run, and its two-channel weights are not in the repository.
- That the 2022 paper's **RPAH** figures (76.68 % sensitivity, 56.55 false alarms per 24 h, 92.19 % with a human arbiter) are reproduced or reproducible. They require private clinical data under hospital ethics approval and a 20-channel model (19 EEG + ECG).
- That **any event-level number here has a published counterpart**. The 2022 paper's Table 2 reports an AUC for TUH and leaves the sensitivity and false-alarm columns blank, so there is no published TUH false-positive or sensitivity figure to compare against. In particular, the resemblance between this project's FP/24 h and the paper's RPAH figure of 56.55 is **coincidence**, across different data, model, metric, and about two and a half orders of magnitude of recording time.
- That 0.89 and 0.84 are measured on the same data — it is TUSZ v2.0.0 `eval` versus v1.5.1 `dev`.
- That the reproduction *beats* the paper. 0.84 sits inside the confidence interval (one-sided bootstrap p ~ 0.06); the claim is indistinguishability, not improvement.
- That any number here covers all 305 cached recordings. 99 have no `.csv_bi` and are excluded.
- That ZUNA is a validated seizure-detection front-end, or that its event-level improvement establishes that it helps, given the contradicting window-level AUC.
- That the reported false-positive rate of roughly 328 per 24 hours is a clinical false-alarm rate; it is derived from ten short TUSZ seizure clips and is not a deployment figure.
- That the displayed seizure probabilities are calibrated; the probability strip currently shows raw, uncalibrated softmax with no temperature, Platt, or isotonic calibration, even though the literature review argues calibration is a trust requirement, and this is acknowledged as a stated limitation rather than glossed over.
- Any recommendation to retrain, fine-tune, or otherwise modify model weights; the work is inference-only throughout.
