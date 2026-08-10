# RESULTS — the single source of truth for every number in this project

*BMET4111 Thesis — Sam Gan, University of Sydney. Supervisor: Prof. Omid Kavehei.*

**Every quantitative claim in the thesis should be traceable to this file.** If a number appears
anywhere else in the repository and disagrees with this file, this file wins and the other place
is stale. Each figure below names the command that regenerates it.

All measurements: public TUSZ **v2.0.0** `eval` data, 19 channels, 12-second windows, 6-second
stride, per-window ICA, pretrained `convlstm_ICA_12_train.h5`. Inference only — no retraining.

**Headline: the 19-channel detector reproduces its source paper.** Window-level AUC **0.89**
(95 % CI [0.82, 0.94] by recording, **[0.73, 0.95] by patient** — quote the latter) over 206
annotated recordings / 28 patients / 27.8 h, against the **0.84** published for TUH. The published value lies inside the interval: this is a result **statistically
indistinguishable from the published one**, not a demonstration of improvement.

**Quote two significant figures.** The interval is ±0.06 wide; four decimal places imply a
precision this evidence does not carry, and invite a reproducibility challenge that will fail
(see §9).

Use **`artifacts/zuna_thesis/manifest_full.csv`** for anything reported, and note that
**99 of the 305 cached recordings have no `.csv_bi` annotation and are excluded** — an absent
annotation is not evidence of a seizure-free recording. The scorable set is **206 files**.
The older `manifest.csv` is 26 seizure-enriched files and is misleading in both directions —
see the box in §2.

Last regenerated: 2026-08-08, in a single quiesced pass with no cache writers running.

---

## 1. What the source paper actually published

Yang et al., *Continental generalization of a human-in-the-loop AI system for clinical seizure
recognition*, **Expert Syst. Appl. 207:118083 (2022)** — preprint arXiv:2103.10900v2, **Table 2**.
This is the paper whose detector these weights implement (19 ch / 12 s / ICA).

| dataset | AUC | eval method | sensitivity | FA/24 h |
|---|---|---|---|---|
| **TUH EEG Corpus v1.5.1** (this work) | **0.84** | — | **—** | **—** |
| RPAH, 1,006 sessions | 0.82 | SDR | 76.68 % | 56.55 |
| RPAH, 66-session pilot + human arbiter | — | SDR | 92.19 % | 0 |

**Read the TUH row carefully: the paper publishes an AUC and nothing else.** The sensitivity and
false-alarm columns are blank for TUH. There is therefore **no published TUH false-positive
statistic to replicate**, and no published TUH sensitivity either. The only public-data claim the
paper makes for the 19-channel detector is the single number 0.84, and Fig. 5 identifies it as
the TUH **development** split, measured before the PWI/PEI lens.

> ### ⚠ Do not compare our FP/24 h to their 56.55
>
> Our event-level false-alarm figure and their 56.55 look similar. **The resemblance is a
> coincidence and the two are not comparable.** Theirs is RPAH: 14,590 hours, 1,006 sessions,
> **private** clinical data under hospital ethics, a **20-channel** model (19 EEG + ECG, see
> `utils/ICA_load_data_elec.py:285`), the PWI/PEI lens, and the **SDR** metric, which by the
> paper's own footnote "combines the false alarms within 30 seconds into one". Ours is 27.8 hours
> of public TUSZ across 206 files, 19 channels, concatenate/discard shaping, and per-event
> matching with a 5 s tolerance. Different data, different model, different metric, and about
> two and a half orders of magnitude apart in recording hours.

The two-channel SPMB 2020 paper asserts performance "improves dramatically when all 19
electrodes" but publishes **no 19-channel number**, so there is nothing to replicate there either.

---

## 2. Window level — the reproduction result

**Evaluation set: the 206 locally cached TUSZ v2.0.0 recordings that have a `.csv_bi`
annotation** — 44 seizure-bearing (85 seizures, 1.16 h ictal) and 162 background-only,
**27.8 h**. A further **99 cached recordings carry no annotation at all and are excluded**:
`read_csv_bi` returns `[]` for a missing file and for a seizure-free one alike, so scoring them
as background would assert a ground truth nobody has — and at least 6 of them do contain
seizures according to their per-channel `.csv`. For scale, the paper's TUH dev split is 170.3 h.

```
python experiments/build_full_manifest.py --out artifacts/zuna_thesis/manifest_full.csv
python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest_full.csv
```

| protocol | pooled AUC | 95 % CI (by file) | windows | positive |
|---|---|---|---|---|
| project — any-overlap labelling, 6 s stride | 0.80 | [0.74, 0.85] | 15,496 | 788 |
| **paper — pure windows only** | **0.89** | **[0.83, 0.94]** | 15,211 | 503 |
| paper — pure + non-overlapping | 0.89 | [0.82, 0.94] | 7,678 | 249 |
| *source paper, TUH v1.5.1 dev* | *0.84* | — | — | — |

**The reproduction is statistically indistinguishable from the published result.** 0.84 lies
inside the interval. **Do not phrase this as beating the paper.** The defensible claim is
reproduction, not improvement.

> ### Two things to get right when writing this up
>
> **1. It is the paper's *labelling rule*, not its *sampling*.** The paper's loader steps
> `i += 12` (non-overlapping) and anchors each tile at the labelled interval's own start
> (`utils/ICA_load_data_elec.py:115-147`). This project applies the same in-interval labelling
> rule to its own global 6-second grid. Calling it "the paper's own protocol" overstates it —
> **zero** of the window offsets the paper's loop would emit coincide with the ones scored here.
>
> **2. Do not quote 15,211 windows beside a 12-second non-overlapping protocol.** 15,211 windows
> over 27.8 h is 547/h; a 12-second non-overlapping tiling cannot exceed 300/h. A co-author
> spots that with one division. Quote **7,678 windows / 249 positive** for the non-overlapping
> row, or state plainly that the 15,211 figure is the 6-second-stride variant.
>
> **The robustness is the real result, so lead with it.** The same probabilities give 0.89 at the
> 6 s stride (15,211 / 503) and 0.89 non-overlapping (7,678 / 249). 0.84 lies inside the interval
> under every variant tried: non-overlapping stride, patient-level clustering, montage
> reweighting to the paper's own mix, and dropping the single most influential recording.

The confidence interval resamples whole **files**, not windows — windows within one recording
share patient, montage and artifact regime, so a window-level interval treats correlated windows
as independent evidence and comes out far too narrow to quote honestly.

**Quote the patient-level interval as well, and prefer it.** The 206 recordings come from 28
patients, and only **13 contribute a positive window** — three of which supply 68 % of the
positives. Resampling patients rather than files gives **[0.73, 0.95]**, considerably wider than
the file-level [0.82, 0.94]. 0.84 sits inside both. The patient interval is the honest one, for
the same reason patient grouping was necessary in the calibration analysis (§8).

### Why the two protocols differ, and the disclosure that must accompany it

The paper's feature loader (`utils/ICA_load_data_elec.py:115`) tiles each labelled interval with
windows that fit **entirely inside** it, so a window straddling a seizure boundary is never
generated. This project instead slides a window every 6 s and calls it a seizure on **any**
overlap.

The exclusion removes **1.8 % of all windows — but 36.2 % of the positive class** (285 of 788),
and those are the hardest positives, being mostly background by duration. Quoting only the 1.8 %
makes a 0.09 AUC difference look like rounding. **State the 36 % figure whenever the paper
protocol is used.**

It is nonetheless *faithful rather than selective*, and the evidence for that is checkable:
`txt_file/ref_dev.txt` tiles all 1,013 dev recordings with contiguous `bckg`/`seiz` intervals —
every recording starts at 0.0, with zero gaps and zero overlaps — so the source loop could not
physically have emitted a boundary window. Cite this when the protocol is challenged.

PR-AUC under the project protocol is **0.39**.

**Caveat to state whenever the comparison is made:** TUSZ v2.0.0 `eval` here versus v1.5.1 `dev`
there. Same corpus family and same detector configuration, but not the same files. The paper's
0.84 is a *development-split* number and ours is *eval*, which makes the comparison conservative.

> ### The 26-file subset was misleading in both directions
>
> The earlier `manifest.csv` (26 seizure-enriched files, 1.7 h) gave 0.82 [0.67, 0.93] under the
> paper protocol — an interval too wide to distinguish anything. It was also **pessimistic on
> sensitivity and optimistic on false alarms**: 25.0 % and 57.2 FP/24 h, against 49.4 % and
> 204.4 FP/24 h on the annotated corpus. A seizure-enriched subset has almost no background time
> in which to raise a false alarm, so its false-alarm rate is not interpretable.
> **Use `manifest_full.csv` for every reported number.**

---

## 3. Event level

```
python experiments/evaluate_baseline.py --manifest artifacts/zuna_thesis/manifest_full.csv --name full_scorable
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest_full.csv
```

**No published counterpart exists** (see §1) — these characterise this system, they do not
reproduce anything. 206 annotated files, 85 reference seizures, 27.8 h.

### Decision-stage ablation, threshold 0.5

| configuration | sensitivity | hits | FP/24 h | duplicates |
|---|---|---|---|---|
| raw windows | 0.565 | 48/85 | 478.0 | 32 |
| \+ event shaping (concatenate <10 s, discard <5 s) | 0.553 | 47/85 | 314.6 | 11 |
| per-second averaging only | 0.494 | 42/85 | 237.8 | 13 |
| **source method (averaging + shaping)** | **0.494** | **42/85** | **222.9** | **10** |

Reproducing the source method's decision stage — not its model — cuts the false-alarm rate
**2.1×** (478.0 → 222.9) for 7 percentage points of sensitivity. Note this is the *decision
stage* alone; the separate FP-counting bug fix is described in `reproduction_status.md` §3 and
must not be folded into the same multiplier.

With this sample, **shaping alone is arguably the better triage operating point**: 55.3 % at
314.6 FP/24 h against 49.4 % at 222.9 — five more seizures caught for an alarm burden a reviewer
can dismiss. Switch with `USE_PER_SECOND_AVERAGING` in `eval_config.py`.

### Threshold sweep, source method

| threshold | sensitivity | hits | FP | FP/24 h | duplicates |
|---|---|---|---|---|---|
| 0.50 | 0.494 | 42/85 | 237 | 204.4 | 10 |
| 0.30 | 0.565 | 48/85 | 352 | 303.6 | 10 |
| 0.10 | 0.588 | 50/85 | 457 | 394.2 | 10 |
| 0.05 | 0.647 | 55/85 | 522 | 450.2 | 9 |
| 0.01 | 0.718 | 61/85 | 650 | 560.6 | 5 |

(The sweep runs at the configured operating point, so its 0.50 row differs slightly from the
ablation table above, which re-scores each configuration independently.)

### The PWI/PEI lens — why it is deferred, and why it should eventually be built

```
python experiments/evaluate_baseline.py --manifest artifacts/zuna_thesis/manifest_full.csv \
    --thresholds 0.5 0.7 0.8 0.9 0.95 0.99 --name fa_cost
```

The paper's second stage is a **PWI/PEI lens** (Periodic Waveform Index / Periodic Energy Index),
not "PWA" — an error corrected throughout this repository on 2026-08-10. Verified against the
arXiv source, the mechanism is:

> "We use the 85-percentile of PWI and PEI values for each frequency band over the **last two
> hours** as adaptive thresholds."

Three considerations, in order of how decisive they are. The first two are reasons it is **deferred**; the third was initially written as a reason to reject it outright and is corrected below — the evidence does not support that.

**1. It cannot run on this corpus, let alone be validated on it.** The adaptive thresholds need
two hours of preceding signal. Of **306 recordings, zero reach two hours**; the longest is
**58.3 minutes — 49 % of the required window**. Implementing it would produce a component that
can never be exercised, and shipping an untestable stage into a clinical review tool is worse
than not having it.

**2. Plain thresholding already buys the false-alarm reduction, and the price is measurable.**

| threshold | sensitivity | hits | FP/24 h |
|---|---|---|---|
| 0.50 | 49.4 % | 42/85 | 204.4 |
| 0.70 | 44.7 % | 38/85 | 117.3 |
| 0.80 | 42.4 % | 36/85 | **67.3** |
| 0.90 | 36.5 % | 31/85 | 33.6 |
| 0.95 | 34.1 % | 29/85 | 18.1 |
| 0.99 | 27.1 % | 23/85 | 4.3 |

The paper's 56.55 FA/24 h sits between the 0.80 and 0.90 rows — reachable here at roughly
**41 % sensitivity**, i.e. a **3× false-alarm reduction for about 8 percentage points**. Any lens
has to beat *that* curve to be worth its complexity, and there is no way to show it does without
two-hour recordings.

**3. As a *filter* it is wrong for a human-in-the-loop tool — but that is not the only way to
use it, and the underlying mechanism addresses a real, measured defect here.**

The naive objection first, because it is valid as far as it goes: the lens *removes* detections.
A false positive costs the reviewer seconds to reject; a false negative is a seizure they are
never shown, by a detector that already misses about half of them. Threshold 0.50 → 0.80 is 159
fewer false alarms at the cost of **6 of the 42 seizures the system would otherwise surface**.

**But the lens is not a global threshold, and treating it as one understates it.** Its mechanism
is a *per-recording, per-band adaptive percentile*. That targets a failure a global threshold
cannot touch, and the failure is present in this corpus:

| background score across the 177 recordings with ≥ 20 scored background windows | |
|---|---|
| 85th percentile — median | 0.0032 |
| 85th percentile — maximum | **0.9885** (**308× the median**) |
| recordings whose 85th-pct *background* already exceeds the 0.5 threshold | **19 / 177 (11 %)** |

And the resulting false alarms are extremely concentrated:

| | |
|---|---|
| **5 of 204 recordings produce 44 %** of all false-positive windows | |
| 10 recordings produce 57 % | 20 recordings produce 74 % |
| **111 of 204 recordings (54 %) produce none at all** | |

The worst case, `aaaaaqek_s011_t001`, has **82 of its 99 background windows above threshold** and
a background 85th percentile of 0.989 — the model reads essentially the whole recording as
ictal. No global threshold can fix that without destroying sensitivity everywhere else. A
per-recording adaptive percentile fixes exactly it.

> **So the mechanism is worth having, and the paper's own use of it is not the only option.**
> The paper *filters* with it because its system screens 14,590 hours near-autonomously. A review
> assistant can take the same statistic and use it to **normalise and rank rather than suppress**:
> score each window against its own recording's background distribution, order the worklist by
> that, and show the reviewer everything. The attention burden falls — the wall of alarms on a
> bad recording is what actually exhausts a reviewer — while **nothing is hidden, so the
> false-negative risk added is zero.**
>
> That is the form to build: adapting an autonomous-system component into a human-in-the-loop-safe
> one, which is a contribution in itself rather than a reimplementation.

**Verdict: build it, in the ranking form, once ≥ 2 h recordings exist.** The deployment target is
ambulatory monitoring at 24–72 hours per study, where the two-hour window is comfortably
available and where 204 FA/24 h compounds into hundreds of alarms per study. The evidence above
says the mechanism has a genuine target in this data; the blocker is the evaluation corpus, not
the idea. The falsifiable question to answer when that data exists is narrow and well-posed:
**does per-recording adaptive normalisation beat a global threshold at matched sensitivity?**
§3b's sweep is the curve it has to beat.

### Reviewer-triage view, threshold 0.5

**Seizure-file recall 0.750 (33/44)** · background false-flag rate 0.451 (73/162) · 1,235
candidate windows over 27.8 h · window ROC-AUC 0.80, PR-AUC 0.39.

At the default threshold the system routes **three quarters of seizure-bearing recordings** to a
reviewer while also flagging about **45 % of background recordings** — it reduces the reading
burden without removing it. Data reduction, not sensitivity alone, is the honest figure of merit.

**Event matching is 5 s proximity, not overlap** (`compare_zuna.py:126-128`), and greedy, so a
"detected" seizure can be a prediction landing just outside it and sensitivity is a lower bound.

---

## 4. ICA on vs off

```
python experiments/diag_ica.py
```

12 seizure files with a usable pair, pooled window ROC-AUC. Both arms are restricted to the
**intersection of scored windows, matched by window start** — see the note below on why that
matters and why these numbers differ slightly from an earlier version of this table.

| | pooled window AUC | 95 % CI (cluster bootstrap by recording) |
|---|---|---|
| ICA on (as trained, as deployed) | 0.7214 | — |
| ICA off | 0.7417 | — |
| **delta (off − on)** | **+0.0203** | **[−0.030, +0.057]** |

`P(ICA-off better) = 0.81` over 10,000 resamples, 12 recordings.

> **The interval crosses zero.** An earlier version of this section said "turning ICA off
> **improves** discrimination slightly". It does not support that. The defensible statement is:
> **no detectable difference in window AUC over 12 recordings; the point estimate favours
> ICA-off and the evidence leans that way (P ≈ 0.81), but the interval spans −0.03 to +0.06.**
>
> What *is* solid and unaffected by this: ICA-off runs **~30× faster**, and it is ~90 % of the
> inference cost (§6). The speed argument stands on its own and needs no accuracy claim.

**Still reported as a finding, not acted on.** The model's training features were generated by
this same per-window ICA — now **verified**, not assumed: `utils/ICA_load_data_elec.py:15`
imports the same `ica_arti_remove` from the same module, and handles its `None` return by
skipping the segment exactly as inference does. The defects in that function are therefore baked
into the operating point, and removing them would invalidate the weights rather than improve
them. See `docs/ica_implementation_review.md` for the defect list and
`docs/reproduction_status.md` §4.

### Why these numbers moved

The first version of this comparison masked each arm independently with `probs != 0.0` and
pooled the arms without aligning them. That is the inference `skip_code` was added to the cache
to replace: a refused window is stored as 0.0 and is indistinguishable from a confident negative
by probability alone. It also allowed the two arms to be scored on *different* window sets — a
window where ICA fails is refused in the ICA-on arm and scored normally in the ICA-off arm.

**Measured, that particular bias does not occur on this manifest**: all 129 refusals across 974
windows are interrupted-signal rejections, which `detect_interupted_data` raises *before* the
`use_ica` branch and which therefore apply identically to both arms. Zero windows failed ICA. The
comparison was not actually confounded — but it should not depend on that remaining true, so
`experiments/diag_ica.py` now uses `skip_code` and intersects the arms on window start. One file
lost its pair under the stricter matching, hence 12 files rather than 13.

---

## 4b. The resolution floor — how large an effect this corpus can even detect

```
python experiments/diag_ica_paper_variant.py --variant as_trained     # the control
python experiments/diag_ica_paper_variant.py --variant all_components
```

§9 records that inference is **not bit-reproducible**: `random_state=13` does not pin FastICA,
so re-running an unchanged file moves individual window probabilities. That has a consequence
nobody had measured — **it puts a floor under every comparison in this document.**

Running the *identical* configuration twice and comparing it to its own cache:

| comparison | delta pooled AUC | 95 % CI (cluster bootstrap by recording) | vs floor |
|---|---|---|---|
| **noise floor — identical config re-run** | **−0.0057** | **[−0.019, +0.003]** | — |
| paper-literal: remove **all** flagged components | −0.0200 | [−0.049, +0.004] | 3.5× |
| ICA off vs on | +0.0203 | [−0.030, +0.057] | 3.5× |
| ZUNA vs baseline | −0.0428 | [−0.152, +0.040] | 7.5× |

All on 12 recordings except ZUNA (10), pooled window AUC, arms matched window-by-window and
masked on the union of each pair's refusals.

The floor itself is independently corroborated: re-running one file and diffing against its own
cache gives an identical window grid and a **maximum per-window divergence of 0.1067**, matching
the 0.107 recorded in §9 from a separate measurement.

> **Read this table before quoting any secondary result.** Two of the three effects are ~3.5×
> the noise *point estimate*, but the noise **interval** is itself ±0.02 — comparable to the
> effects being measured. **This corpus cannot resolve differences of this size.** Every interval
> crosses zero, and that is a statement about the sample, not about the interventions.
>
> The fix is more recordings, not more analysis. The full corpus has 206 annotated files against
> the 10–13 used here, and the headline replication (§2) is on all 206 — which is exactly why
> that result is solid and these are not. **Do not put §4, §4b and §5 numbers in the same table
> as the §2 AUC without saying they come from a 20× smaller sample.**

### What the paper-literal variant shows

The paper says "we remove **those** independent sources" — every source correlated with Fp1/Fp2.
The code removes only the top-scoring component per channel. Forcing the paper's literal reading
(907 components flagged, 907 removed, against roughly half that under top-1) moves pooled AUC
from 0.7214 to **0.7014**.

So the deviation is, if anything, *load-bearing in the code's favour*: being more literally
faithful to the paper's wording makes the point estimate **worse**. At 3.5× the noise floor with
an interval that grazes zero this is **suggestive, not established** — but it is a concrete
argument against "fixing" the top-1 restriction, and it is measured rather than assumed.

**This is the correct use of these experiments.** They are not a route to a better detector;
they are evidence about how much the documented deviations matter, produced without touching the
function that generated the training features.

```
python experiments/rescore_zuna_compare.py --dir artifacts/zuna_thesis/compare_first10
```

10 files, threshold 0.5, re-scored from the stored probability caches with the fixed scoring:

| arm | hits/refs | sensitivity | FP/24 h | duplicates | window AUC |
|---|---|---|---|---|---|
| baseline | 4/19 | 21.1 % | 41.1 | 0 | 0.6878 |
| ZUNA | 5/19 | 26.3 % | 0.0 | 1 | 0.6466 |
| **delta** | +1 | **+5.3 pp** | **−41.1** | | **−0.0412 (worse)** |

**The two metrics still disagree, and the direction of that disagreement is the finding.** ZUNA
gains one seizure and removes the false alarms at this operating point, but the point estimate of
the threshold-independent window AUC is *lower* — it does not rank seizure windows above
background better, it just moves events relative to the threshold.

### How uncertain is that AUC delta? (quantified, not asserted)

```
python experiments/zuna_auc_interval.py
```

The two arms score the same recordings, so the comparison is **paired** and the bootstrap must
preserve that: resample recordings, and recompute both arms' pooled AUC from the same drawn
files. Windows are matched on **start time**, and a window is dropped if **either** arm failed to
score it.

| | pooled AUC | 95 % CI (cluster bootstrap by recording) |
|---|---|---|
| baseline | 0.6878 | [0.526, 0.835] |
| ZUNA | 0.6450 | [0.498, 0.792] |
| **delta (ZUNA − baseline)** | **−0.0428** | **[−0.152, +0.040]** |

`P(ZUNA worse on AUC) = 0.84` over 10,000 resamples. 330 windows, 112 ictal, 10 recordings.

#### The paired mask is not a detail — the table above it is mismatched

`experiments/rescore_zuna_compare.py:105` masks each arm separately with `keep = probs != 0.0`.
Measured on these 10 files, **6 windows are refused in the baseline arm and scored in the ZUNA
arm**, so they enter one arm's AUC and not the other's. The 0.6878 / 0.6466 pair in the table
above is therefore computed on **different window sets**, which is exactly the defect this
document criticises `diag_ica.py` for in §4.

Masking *both* arms on the union of refusals gives baseline **0.6878** — the published figure
reproduces exactly — and ZUNA **0.6450**, slightly below the table's 0.6466 because the ZUNA arm
now also drops those 6 windows.

Keeping them instead is not a fix either: their stored `0.0` is a **sentinel for "never scored"**,
and handing it to an AUC ranks a refusal as a confident negative. An earlier draft of this
section did exactly that, reported baseline 0.6887, and attributed the gap to window-start
intersection. That explanation was wrong — the intersection is empty of effect here, since all
336 starts are shared. The cause was always the unscored-window mask.

**The conclusion is unchanged**: −0.0428 against −0.0412, and the interval still crosses zero.
The defect moved the third decimal, not the finding. It is recorded because the same masking
error, left in place, would silently bias a larger comparison.

> **The interval crosses zero, so say so.** An earlier version of this section called the result
> "inconclusive at best and **mildly negative** on the more rigorous measure". The second half of
> that is more than 10 recordings can support. The defensible statement is: **no detectable
> difference in window AUC; the point estimate is negative and the evidence leans that way
> (P ≈ 0.84), but the interval spans −0.15 to +0.04.**
>
> This does not rescue ZUNA. It moves the finding from "mildly harmful" to "unproven either way
> at this sample size", and the *reason* to leave it out of the deployed application is unchanged
> and independent: ~6× real time, ~42 GiB peak RAM, and no demonstrated benefit. An unproven
> gain does not justify that cost.

ZUNA costs ~6× real time and ~42 GiB peak RAM, which is why only 10 of 26 files were processed.

> **Superseded:** "sensitivity 26.3 % → 31.6 %, false positives 328.7 → 205.4 per 24 h". Those
> event-level figures were computed before the scoring fixes and are wrong — the false-alarm
> rates especially. The window AUCs (0.689 vs 0.647) were never affected, because AUC does not
> depend on event scoring, and they still hold at 0.6878 / 0.6466.

---

## 6. Runtime

Profiled in `seiz36` on real TUSZ windows.

| stage | per window | share of loop |
|---|---|---|
| `ica_arti_remove` | ~141 ms | **~90 %** |
| `model.predict` (batch 1) | ~10.8 ms | ~7 % |
| `_calc_stft` | ~4.3 ms | ~3 % |
| `detect_interupted_data` | ~0.9 ms | <1 % |

End to end the baseline runs at **~0.043× real time** (a 300 s recording scores in ~12.6 s).
Measured alternatives on the same file: no ICA **30.6×** faster, one ICA fit per recording
**9.6×** faster, batching the model **1.13×** faster (the network is not the bottleneck).

**Consequence for BMET4112:** a C++ ConvLSTM addresses ~7 % of runtime. The defensible port
target is the ICA/preprocessing front end. See `docs/deployment_roadmap.md` §5.

---

## 7. Known limitations that bound all of the above

- **99 of 305 cached recordings have no `.csv_bi`** and are excluded from every number here.
  At least 6 of them contain seizures per their per-channel `.csv`. They are not evidence of
  anything and must not be counted as background.
- **6.3 % of all windows (1,560 of 24,770) were never assessed** by the pipeline — rejected by
  the artifact check or by a failed ICA. They are excluded from the window AUC (which measures
  model quality, not preprocessing drop-out) and count as non-detections at event level. One
  recording, `aaaaaqtw_s002_t012`, has all 49 windows rejected and contains a real 27-second
  seizure.
- **Probabilities are uncalibrated raw softmax**, and §8 quantifies it: ECE 0.072, and a score
  of 0.5 corresponds to a 29 % chance of ictal, not 50 %. The `temp = 1.0` Lambda in
  `models/deep_conv_lstm.py:84` is the identity. Post-hoc Platt scaling would cut ECE ~85 % but
  is deliberately **not** adopted in the GUI (see §8).
- **Corpus version mismatch** throughout: weights trained on v1.5.1, evaluated on v2.0.0.
- **Train/test disjointness is probable but unverified — and the comparator is worse off.**
  TUSZ v1.5.x splits were **not** patient-disjoint. NEDC report 13 subjects common to eval and
  train and 5 common to dev and train (Buckwalter, Chhin, Rahman, Obeid & Picone, *IEEE SPMB*
  2021). We confirm the latter directly from the reference lists in this repository: patients
  `00001027`, `00001981`, `00004671`, `00006546`, `00009842` appear in **both** `ref_train2.txt`
  and `ref_dev.txt`, carrying **22.4 % of the development split's seizure events** (151 of 673).

  **The 0.84 we are compared against is therefore itself optimistically biased**, which makes
  this comparison *conservative* rather than flattering. NEDC rebuilt the partitions for v2.0.0
  to remove the overlap but do not state whether the shared subjects were dropped from eval or
  retained in train, and v2.0.0 IDs are anonymised — so disjointness between our evaluation set
  and the v1.5.1 training data is **probable but unverified**.

  Recording-level fingerprinting of all local `.csv_bi` against the 4,597 training recordings
  finds no reuse (0 exact seizure-boundary matches, 0 within 50 ms), **but that test has no power
  against patient-level leakage**, which is the mode that matters. Do not claim more than this.
- **Event matching uses 5 s proximity, greedily** — sensitivity is a lower bound.

---

## 8. Calibration — the score is not a probability, and can be made one

```
python experiments/calibration.py --manifest artifacts/zuna_thesis/manifest_full.csv     --out artifacts/zuna_thesis/baseline_eval/calibration.json     --svg artifacts/zuna_thesis/baseline_eval/reliability.svg
```

Out-of-fold under 5-fold cross-validation **grouped by patient**, 15,496 windows from 203
recordings / **28 patients**, 5.09 % ictal.

| | ECE | 95 % CI | Brier | log loss | ROC-AUC |
|---|---|---|---|---|---|
| **raw** | **0.072** | [0.052, 0.092] | 0.0647 | 0.283 | 0.801 |
| temperature | 0.069 | [0.049, 0.094] | 0.0606 | 0.212 | 0.801 |
| **Platt** | **0.011** | [0.007, 0.028] | 0.0438 | 0.172 | 0.787 |
| isotonic | 0.015 | [0.010, 0.031] | 0.0417 | 0.172 | 0.777 |

**The detector is substantially over-confident** — mean score 0.087 against a base rate of 0.051.
Platt scaling cuts ECE by ~85 %. Platt and isotonic are **not separable** at this sample size
(paired 95 % CI [−0.011, +0.003]); **report them as a tie and prefer Platt**, because it has two
parameters rather than a free monotone map, is stable across seeds and splits, and its intercept
can be re-shifted under label shift — which isotonic has no parameter to do.

### Grouping by patient is not optional

The 203 recordings come from only **28 patients**, and just 14 contribute a positive window.
Grouping folds by *recording* puts other recordings of the same patient in training for ~95 % of
test windows, and the free monotone map exploits it:

| | by recording | by patient | leave-one-patient-out |
|---|---|---|---|
| Platt | 0.0106 | 0.0109 | 0.0107 |
| isotonic | **0.0085** | 0.0149 | 0.0168 |

Isotonic's apparent advantage is entirely leakage. Any calibration result quoted from this project
must state the grouping unit.

### Why temperature scaling fails here

Temperature barely moves ECE (0.072 → 0.069) despite fitting a stable T ≈ 2.1 across all folds,
because a pure logit rescale through the origin has **no intercept**, so nothing pins the mean
prediction to the base rate. At 5 % prevalence, dividing predominantly negative logits by T > 1
pushes the mean score *away* from the base rate. Fitting the offset alone at temperature's own
slope already recovers most of the gap. Temperature is not useless — it cuts log loss 25 % and
Brier 6 % — but calibration error specifically is what it cannot fix.

### Three units, three different numbers

Calibration is fitted per **window** (what the network emits), but the GUI thresholds the
per-**second** mean and presents **events**. `mean(f(p)) ≠ f(mean(p))`, so these are different
objects. Say which one a number describes.

| unit | raw ECE | P(ictal \| score ≥ 0.5) | 95 % CI |
|---|---|---|---|
| window (12 s) | 0.072 | 0.287 | [0.166, 0.430] |
| second (what the GUI thresholds) | 0.059 | 0.350 | [0.199, 0.524] |
| event (what the reviewer sees) | — | 42 of 289 proposals = 0.15 | see §3 |

Per-second averaging is itself a partial calibrator, removing ~18 % of the miscalibration before
any fit. Per-second Platt reaches ECE 0.009.

**A raw score of 0.5 is not a 50 % chance of seizure** — at window level it is 29 %, and the
interval is wide. The GUI's existing labelling (axis "model score", `scores_are_calibrated: false`
in provenance, "raw, UNCALIBRATED" in the tooltip) is therefore **correct, and this analysis
validates it** rather than exposing a defect.

### Calibration is NOT adopted in the GUI

Reported as analysis only, for three reasons: the calibrator is fitted on windows while the GUI
thresholds their per-second mean; thresholding calibrated scores at 0.5 collapses event
sensitivity (0.494 → 0.32 isotonic / 0.14 Platt) so every event-level number in §3 would need
re-measuring; and the fitted map is specific to this corpus's prevalence.

### Caveats

- **Prevalence.** Every calibrated probability is conditional on π = 0.0509 on a seizure-enriched
  corpus. At ambulatory prevalence (20–200× lower) the same score means far less. Transportable
  only by re-shifting Platt's intercept under a label-shift assumption.
- **ECE definition.** Positive-class reliability form over 15 equal-mass bins (Naeini et al. 2015;
  Bröcker 2009), *not* Guo's confidence-vs-accuracy form, which is near-meaningless at a 5 % base
  rate where predicting background everywhere scores 95 % accuracy. Stable across 5–200 bins.
- **MCE is partition-dependent** (0.27 at 5 bins to 0.69 at 200) — diagnostic only, never a
  headline, and never quote a reduction in it.
- **Calibration buys no discrimination.** Within-fold ROC-AUC is *bit-identical* for temperature
  and Platt (recorded per fold in the JSON). Isotonic perturbs it by up to 0.003 in either
  direction through tie creation.
- **Coverage.** 943 windows (5.7 %), carrying 54 positives, are unscored and excluded — no
  calibrator here can touch them.

---

## 9. Inference is not bit-reproducible

`random_state=13` at `utils/preprocessing.py:71` is **not sufficient** to make the pipeline
deterministic. Re-running `compute_probs` on an unchanged EDF reproduces its own cache for some
windows and then diverges — one measured window moved by 0.107. Regenerating the caches
therefore moves per-file AUCs slightly and shifts pooled figures in the third decimal.

Consequences to respect:

1. **Quote two significant figures** for every derived number. The digits beyond that are not
   stable across regenerations.
2. **Regenerate all artefacts in one pass** with no cache writers running, and commit them
   together. Numbers from different passes must never be mixed in one table.
3. This also corrects `docs/ica_implementation_review.md`, which described the non-converged ICA
   as "at least deterministic". It is not.

The cause is not established. FastICA fails to converge on most windows (§4 and the ICA review),
so the returned unmixing matrix is wherever the iteration stopped, and that appears sensitive to
floating-point details that `random_state` does not control.
