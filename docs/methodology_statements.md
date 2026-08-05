# Methodology Statements: Honest Framing of the Detector, the ZUNA Side-Study, and the Reviewer-in-the-Loop Contribution

*BMET4111 Thesis — Sam Gan, University of Sydney. Supervisor: Prof. Omid Kavehei.*
*Scope note: All quantitative results below were obtained on public TUH/TUSZ data only. No patient or RPAH data were used. The thesis deliverable is the reviewer GUI (a minimum viable product) and its usability evaluation, not detector accuracy.*

This document records the methodological positions the thesis takes. Its purpose is to state plainly what was implemented, what was measured, and what may and may not be claimed, so that the evaluation chapter can be read as a rigorous and defensible account rather than as an overclaimed reproduction. The guiding policy throughout is rigorous honesty: no fabricated results, no overclaiming, and an inference-only methodology that never proposes retraining.

---

> **Correction (superseded by `docs/reproduction_status.md`).** Section 1 below compares the
> implementation against the *two-channel* SPMB 2020 paper and concludes it is "not a
> reproduction". That comparison targets the wrong paper. The pretrained weights
> `convlstm_ICA_12_train.h5` are 19-channel, 12-second and ICA-denoised — every property of the
> **2022 continental-generalization** detector (Yang et al., *Expert Syst. Appl.* 207:118083),
> and none of the two-channel method. Against the correct target the work is a genuine partial
> reproduction: the STFT, montage, window geometry and ICA procedure are faithful, and the
> remaining gap is quantified (pooled window ROC-AUC 0.723 here vs ~0.84 reported).
>
> Section 2's event-level figures ("328.7 to 205.4 false positives per 24 hours") also predate
> three scoring fixes: the source method's decision stage was not implemented, detection
> fragments inside a true seizure were counted as false alarms, and `run_inference.py` reported
> event times in window-index units. Correcting those lowers the reported false-alarm rate 4.5×
> without touching the model. **Regenerate every event-level number in this document** with
> `experiments/evaluate_baseline.py` and `experiments/ablate_postprocessing.py` before the
> results chapter is written. The paragraphs below are retained as a record of the earlier
> position.

## 1. Relationship to the source method

The detector that runs live inside the reviewer GUI is a single 12-second, 19-channel, ICA-denoised ConvLSTM, loaded from the pretrained weights `convlstm_ICA_12_train.h5`. Each inference window is a short-time Fourier transform of fixed shape `(23, 19, 125)`, which matches the input tensor of the saved model exactly. The model therefore loads cleanly and runs as trained: there is no loading error, no shape mismatch, and no silent reshaping or padding workaround in the inference path. When the GUI proposes events, it is reporting the genuine output of this network on the supplied EEG, not the output of a partially-wired or misconfigured pipeline. This point is stated up front because a reader who knows the source paper might otherwise assume that any divergence in numbers reflects a broken implementation; it does not.

The source paper proposes a materially different architecture. Its method is a *two-channel* detector built on a *blended multi-time* spectrogram representation, in which 3-second, 5-second, and 7-second segments are computed in parallel and combined through a voting lens to reach a per-segment decision. The repository does contain the data-loading scaffolding for that approach — `utils/load_data_elec_3s.py`, `utils/load_data_elec_5s.py`, and `utils/load_data_elec_7s.py`, together with their `_dev_` counterparts — but these modules are two-channel, they are not invoked anywhere in the live GUI inference path, and they are best described as dead or unused code inherited from the upstream project. Critically, the two-channel weights that the paper's multi-time voting method would require are absent from the repository. There is consequently no way to run the paper's headline method from this codebase, and no attempt is made to do so.

It follows that the implemented detector is **not** a reproduction of the source paper. It is a single-scale, 12-second, 19-channel ICA ConvLSTM that happens to live in the same repository lineage. This thesis treats the single-window choice as a deliberate, inference-only simplification rather than as an incomplete reproduction. The simplification is appropriate to a GUI-focused thesis for three reasons. First, the deliverable is the reviewer interface and its usability, so the detector needs only to be a faithful, reproducible source of candidate events, which the single 12-second model provides. Second, an inference-only posture forbids retraining, so reconstructing the absent two-channel voting front-end — which would require training or sourcing weights that the repository does not hold — is out of scope by policy, not merely by convenience. Third, a single, well-characterised detector gives a stable and interpretable substrate against which to measure the human-in-the-loop interaction, which is the actual object of study.

The consequence for how results are described is therefore explicit and is carried through the rest of the thesis: every detector number reported here characterises *this specific model's behaviour on a public TUSZ subset*. None of these numbers should be read as, compared against, or presented as a replication of the source paper's published performance. The paper's blended multi-time voting method was never executed in this work, and any quantitative comparison to it would be unsupported.

---

## 2. Two metrics, two conclusions: reporting ZUNA honestly

ZUNA is reported as an exploratory side-study, and it is reported with both of the metrics that were computed for it, because those two metrics disagree about whether ZUNA helps.

At the **event level**, on ten public TUSZ seizure files scored event-wise at the default threshold of 0.5, ZUNA improves the headline detector behaviour. Sensitivity rises from 26.3% for the baseline to 31.6% with ZUNA, and the false-positive rate falls from 328.7 to 205.4 false positives per 24 hours. Read in isolation, this is a favourable result and it is the result that the current headline tables emphasise.

At the **window level**, the picture reverses. Pooling all windows and computing the area under the ROC curve on the calibrated per-window probabilities gives a baseline AUC of 0.689 against a ZUNA AUC of 0.647. By this discrimination-quality measure ZUNA is *worse*: it degrades the model's ability to rank seizure windows above non-seizure windows. The event-level gain therefore appears to come from a change in where events land relative to the threshold rather than from a genuine improvement in the underlying probability ranking, and a single operating-point event metric can move favourably even as the calibrated, threshold-independent measure deteriorates.

Presenting only the event-level numbers would be misleading, so the thesis presents both and states the disagreement explicitly. Given that ZUNA is, as deployed here, an exploratory repurposing of a general EEG super-resolution model into a seizure front-end — and given that it costs roughly six times real-time and peaks at approximately 42 GiB of RAM, which is why only 10 of the 26 candidate files could be processed at all — the honest conclusion is that the ZUNA side-study is **inconclusive at best and mildly negative on the more rigorous metric**. The recommendation that follows is to scope ZUNA *down*, presenting it as a documented negative or inconclusive exploration with its costs and its window-AUC regression stated, rather than to expand it or to promote the single favourable event metric into a headline claim. ZUNA is explicitly not a validated seizure detection front-end, and the thesis will not describe it as one.

---

## 3. The "money figure": reviewer-in-the-loop event recovery

The single primary figure around which the thesis should be built is **reviewer-in-the-loop event recovery**: the number of true seizures missed by the AI acting alone, compared against the number recovered once a human reviewer has stepped through the proposed events and made accept, reject, and extent-editing decisions, as captured in the exported reviewed `.csv_bi`.

Concretely, the figure plots, per file (and pooled across the public TUSZ subset), the ground-truth seizure events against three derived counts: (i) the events the detector proposes and that survive at the operating threshold — the AI-alone yield; (ii) the events present in the exported reviewed `.csv_bi` after the reviewer has accepted, rejected, and edited candidates; and (iii) the difference between them, which is the set of true seizures *recovered* through the human-in-the-loop interaction that the AI alone would have missed or mis-bounded. The recovered set includes both events the reviewer accepted that a naive threshold reading would have buried, and events whose temporal extent the reviewer corrected by dragging the region so that they now match the reference. The exported reviewed file is the natural data source because it already encodes exactly the accepted and edited events and nothing else, so the figure is a direct read-out of the reviewer's decisions rather than a reconstruction.

This figure is the right thesis centrepiece precisely because it binds together the three parts of the work into one argument. The detector is deliberately weak and is acknowledged as such; the GUI presents that weak detector's output as inspectable, navigable candidates through the SignalView, ProbStrip, and EventList; and the usability story is what lets a reviewer convert raw, low-sensitivity model output into a clinically more complete event set. The contribution is the human-in-the-loop interface, and event recovery is the metric that makes that contribution visible: it shows the interface adding value *on top of* the detector, which is the claim the thesis is actually entitled to make.

The figure is defensible and consistent with the inference-only policy. It requires no retraining and no change to the model; the detector weights are frozen and the probabilities are produced exactly as in normal operation. It is computed entirely on public TUSZ data and ground truth. It does not depend on the detector being good — indeed it is most informative when the detector is mediocre, because the recovery gap is what the interface is for. And because it is derived from the exported reviewed `.csv_bi`, it measures a real reviewer's real decisions through the real GUI, which is the artefact under evaluation, rather than a simulated or idealised oracle.

---

## 4. Claims we will and will not make

**Claims the thesis will make:**

- The reviewer GUI is the contribution: a working minimum viable product for human-AI teamed review of ambulatory and outpatient EEG, evaluated for usability on public TUSZ data.
- The live detector is a single 12-second, 19-channel ICA ConvLSTM (`convlstm_ICA_12_train.h5`) that loads and runs correctly as trained, with the STFT input shape `(23, 19, 125)` matching the saved model.
- All reported detector numbers describe this model's behaviour on a public TUSZ subset.
- A human reviewer, working through the GUI, recovers true seizure events that the AI alone misses, as evidenced by the exported reviewed `.csv_bi`; this reviewer-in-the-loop recovery is the primary result.
- ZUNA is an exploratory side-study whose event-level and window-level metrics disagree, and which is inconclusive-to-negative once its window-AUC regression and its roughly 6x real-time and ~42 GiB cost are accounted for.

**Claims the thesis will not make:**

- That the implemented system reproduces the source paper's blended multi-time 3s/5s/7s two-channel voting method, or its published performance — that method was never run, and its two-channel weights are not in the repository.
- That any detector number here is comparable to the source paper's headline numbers.
- That ZUNA is a validated seizure-detection front-end, or that its event-level improvement establishes that it helps, given the contradicting window-level AUC.
- That the reported false-positive rate of roughly 328 per 24 hours is a clinical false-alarm rate; it is derived from ten short TUSZ seizure clips and is not a deployment figure.
- That the displayed seizure probabilities are calibrated; the probability strip currently shows raw, uncalibrated softmax with no temperature, Platt, or isotonic calibration, even though the literature review argues calibration is a trust requirement, and this is acknowledged as a stated limitation rather than glossed over.
- Any recommendation to retrain, fine-tune, or otherwise modify model weights; the work is inference-only throughout.
