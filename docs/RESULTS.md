# RESULTS — the single source of truth for every number in this project

*BMET4111 Thesis — Sam Gan, University of Sydney. Supervisor: Prof. Omid Kavehei.*

**Every quantitative claim in the thesis should be traceable to this file.** If a number appears
anywhere else in the repository and disagrees with this file, this file wins and the other place
is stale. Each figure below names the command that regenerates it.

All measurements: public TUSZ **v2.0.0** `eval` data, 19 channels, 12-second windows, 6-second
stride, per-window ICA, pretrained `convlstm_ICA_12_train.h5`. Inference only — no retraining.

**Headline: the 19-channel detector replicates its source paper.** Window-level AUC **0.881**
(95 % CI [0.820, 0.932]) across 305 files / 42 h, against the **0.84** published for TUH.

Use **`artifacts/zuna_thesis/manifest_full.csv`** (305 files, 42.24 h) for anything reported.
The older `manifest.csv` is 26 seizure-enriched files and is misleading in both directions —
see the box in §2.

Last regenerated: after the scoring fixes in `docs/reproduction_status.md` §3, on the full corpus.

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
the TUH **development** split, measured before the PWA/PEI lens.

> ### ⚠ Do not compare our FP/24 h to their 56.55
>
> Our event-level false-alarm figure and their 56.55 look similar. **The resemblance is a
> coincidence and the two are not comparable.** Theirs is RPAH: 14,590 hours, 1,006 sessions,
> **private** clinical data under hospital ethics, a **20-channel** model (19 EEG + ECG, see
> `utils/ICA_load_data_elec.py:285`), the PWA/PEI lens, and the **SDR** metric, which by the
> paper's own footnote "combines the false alarms within 30 seconds into one". Ours is 1.7 hours
> of public TUSZ across 26 files, 19 channels, concatenate/discard shaping, and per-event
> matching with a 5 s tolerance. Different data, different model, different metric, different
> scale, four orders of magnitude apart in recording hours.

The two-channel SPMB 2020 paper asserts performance "improves dramatically when all 19
electrodes" but publishes **no 19-channel number**, so there is nothing to replicate there either.

---

## 2. Window level — the replication result

**Evaluation set: all 305 locally available TUSZ v2.0.0 EDFs with a probability cache** —
44 seizure-bearing (85 seizures, 1.16 h ictal), 261 background-only, **42.24 h**, background to
seizure duration **35.4 : 1**. For scale, the paper's TUH dev split is 170.3 h at 9.5 : 1, so
this set is about a quarter the size and considerably *more* background-heavy — a harder
specificity venue, not an easier one.

```
python experiments/build_full_manifest.py --out artifacts/zuna_thesis/manifest_full.csv
python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest_full.csv
```

| protocol | pooled AUC | 95 % CI (by file) | windows | positive | P(AUC ≥ 0.84) |
|---|---|---|---|---|---|
| project — any-overlap labelling, 6 s stride | 0.7941 | [0.738, 0.843] | 23,088 | 788 | 0.03 |
| **paper — pure windows only** | **0.8811** | **[0.820, 0.932]** | 22,803 | 503 | **0.92** |
| paper — pure + non-overlapping | 0.8776 | [0.814, 0.930] | 11,507 | 249 | 0.90 |
| *source paper, TUH v1.5.1 dev* | *0.84* | — | — | — | — |

**The 19-channel detector replicates and then some: 0.881 against a published 0.84**, with 0.84
inside the confidence interval. This is the headline reproduction result.

The confidence interval resamples whole **files**, not windows — windows within one recording
share patient, montage and artifact regime, so a window-level interval treats correlated windows
as independent evidence and comes out far too narrow to quote honestly.

The gap between the first two rows is labelling alone: the paper's feature loader only emitted
windows lying entirely inside one annotated interval, while this project slides a window every
6 s and calls it a seizure on any overlap. **Always state which protocol a window-level number
uses** — the choice is worth ~0.09 AUC here, more than most methodological differences.

PR-AUC under the project protocol is **0.201** on this realistic mix (it was 0.430 on the
seizure-enriched 26-file subset — the inflation there is exactly what a 35:1 background ratio
corrects).

**Caveat to state whenever the comparison is made:** TUSZ v2.0.0 `eval` here versus v1.5.1 `dev`
there. Same corpus family and same detector configuration, but not the same files.

> ### The 26-file subset was misleading in both directions
>
> The earlier `manifest.csv` (26 seizure-enriched files, 1.7 h) gave 0.822 [0.673, 0.925] under
> the paper protocol — a much wider interval that could not distinguish 0.822 from 0.84
> (P = 0.39). It was also **pessimistic on sensitivity and optimistic on false alarms**: 25.0 %
> and 57.2 FP/24 h, against 48.2 % and 223.5 FP/24 h on the full corpus. A seizure-enriched
> subset has almost no background time in which to raise a false alarm, so its false-alarm rate
> is not interpretable. **Use `manifest_full.csv` for every reported number.**

---

## 3. Event level

```
python experiments/evaluate_baseline.py --manifest artifacts/zuna_thesis/manifest_full.csv --name full303
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest_full.csv
```

**No published counterpart exists** (see §1) — these characterise this system, they do not
reproduce anything. 305 files, 85 reference seizures, 42.24 h.

### Decision-stage ablation, full corpus, threshold 0.5

| configuration | sensitivity | hits | FP/24 h | duplicates |
|---|---|---|---|---|
| raw windows | 0.553 | 47/85 | 485.4 | 33 |
| \+ event shaping (concatenate <10 s, discard <5 s) | 0.541 | 46/85 | 319.8 | 11 |
| per-second averaging only | 0.482 | 41/85 | 237.8 | 12 |
| **source method (averaging + shaping)** | **0.482** | **41/85** | **223.5** | **9** |

The decision stage more than halves the false-alarm rate (485.4 → 223.5) for 7 percentage points
of sensitivity. With the larger sample, **shaping alone now looks like the better operating point
for a triage tool**: it keeps 54.1 % sensitivity at 319.8 FP/24 h, where averaging buys a further
FP reduction at the cost of 6 more missed seizures. Switch with
`USE_PER_SECOND_AVERAGING` in `eval_config.py`.

### Threshold sweep, source method, full corpus

| threshold | sensitivity | hits | FP | FP/24 h | duplicates |
|---|---|---|---|---|---|
| 0.50 | 0.482 | 41/85 | 390 | 223.5 | 9 |
| 0.30 | 0.565 | 48/85 | 570 | 326.7 | 10 |
| 0.10 | 0.588 | 50/85 | 695 | 398.3 | 8 |
| 0.05 | 0.635 | 54/85 | 781 | 447.6 | 9 |
| 0.01 | 0.741 | 63/85 | 967 | 554.2 | 6 |

### Reviewer-triage simulation, threshold 0.5

**Seizure-file recall 0.750 (33/44)** · background false-flag rate 0.487 (127/261) · 1,985
candidate windows over 41.9 h.

Read this as the triage claim the thesis can actually make: at the default threshold the system
routes **three quarters of seizure-bearing recordings** to a reviewer, while also flagging about
half the background recordings — so it reduces, but does not remove, the reading burden. The
data-reduction framing (how much raw EEG a reviewer can safely skip) is the honest figure of
merit here, not sensitivity alone.

---

## 4. ICA on vs off

```
python experiments/diag_ica.py
```

13 seizure files, pooled window ROC-AUC:

| | pooled window AUC |
|---|---|
| ICA on (as trained, as deployed) | 0.7157 |
| ICA off | **0.7417** |

Turning ICA off improves discrimination slightly and runs ~30× faster. **This is reported as a
finding, not acted on** — the model's training features were generated by this same
non-converged per-window ICA, so the operating point depends on it. See
`docs/reproduction_status.md` §4 for why any ICA change needs an AUC delta first.

---

## 5. ZUNA side-study — corrected

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
gains one seizure and removes the false alarms at this operating point, but it *degrades* the
threshold-independent window AUC — it does not rank seizure windows above background better, it
just moves events relative to the threshold. With 10 files and 19 reference seizures this is
**inconclusive at best and mildly negative on the more rigorous measure**, which is the same
conclusion as before, at corrected magnitudes.

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

- **One file scores nothing.** `aaaaaqtw_s002_t012` has all 49 windows rejected by
  `detect_interupted_data` and scores 0.0 throughout, despite containing a 27-second reference
  seizure. Those windows are excluded from the AUC (they measure preprocessing drop-out, not
  model quality) but are guaranteed misses at event level. Report the skip rate beside
  sensitivity.
- **Probabilities are uncalibrated raw softmax.** No temperature, Platt, or isotonic scaling.
  The `temp = 1.0` Lambda in `models/deep_conv_lstm.py:84` is the identity.
- **The manifest is seizure-enriched**, so it is not a realistic specificity test. The paper
  makes the same point about TUH generally ("TUH dataset do not provide a realistic specificity
  test venue").
- **Corpus version mismatch** with the weights' training data throughout (v2.0.0 vs v1.5.1).
