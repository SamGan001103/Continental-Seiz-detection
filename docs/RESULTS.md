# RESULTS — the single source of truth for every number in this project

*BMET4111 Thesis — Sam Gan, University of Sydney. Supervisor: Prof. Omid Kavehei.*

**Every quantitative claim in the thesis should be traceable to this file.** If a number appears
anywhere else in the repository and disagrees with this file, this file wins and the other place
is stale. Each figure below names the command that regenerates it.

All measurements: public TUSZ **v2.0.0** `eval` data, 19 channels, 12-second windows, 6-second
stride, per-window ICA, pretrained `convlstm_ICA_12_train.h5`. Inference only — no retraining.

Last regenerated: after the scoring fixes in `docs/reproduction_status.md` §3.

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

```
python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest.csv
```

| protocol | pooled AUC | windows | positive | mean per-file AUC |
|---|---|---|---|---|
| project — any-overlap labelling, 6 s stride | 0.7226 | 857 | 134 | 0.7590 |
| **paper — pure windows only** | **0.8220** | 778 | 55 | 0.7849 |
| paper — pure + non-overlapping | 0.8174 | 391 | 27 | 0.7866 |
| *source paper, TUH v1.5.1 dev* | *0.84* | — | — | — |

**0.822 against 0.84 is a replication.** The difference between the first two rows is labelling
alone: the paper's feature loader only ever emitted windows lying entirely inside one annotated
interval, while this project slides a window every 6 s and calls it a seizure on any overlap.
8.4 % of windows straddle a boundary and every one of them is labelled positive by the project
protocol.

PR-AUC under the project protocol is **0.430**. PR-AUC is the more honest summary of the
26-file set than ROC-AUC, because the file mix is seizure-enriched rather than realistic.

**Caveat to state whenever this comparison is made:** TUSZ v2.0.0 `eval` here vs v1.5.1 `dev`
there, 26 seizure-enriched files, 55 positive windows under the paper protocol.

---

## 3. Event level

```
python experiments/evaluate_baseline.py --manifest artifacts/zuna_thesis/manifest.csv --name baseline26
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest.csv
```

**No published counterpart exists** (see §1) — these characterise this system, they do not
reproduce anything.

### Decision-stage ablation, 26 files, threshold 0.5

| configuration | sensitivity | hits | FP | FP/24 h | duplicates |
|---|---|---|---|---|---|
| *as reported before the fixes (fragments counted as FPs)* | *0.292* | *7/24* | *18* | *257.6* | — |
| raw windows, FP counting fixed | 0.292 | 7/24 | 12 | 171.7 | 6 |
| \+ event shaping (concatenate <10 s, discard <5 s) | 0.292 | 7/24 | 10 | 143.1 | 2 |
| **source method (averaging + shaping)** | **0.250** | **6/24** | **4** | **57.2** | **0** |

### Threshold sweep, source method, 26 files

| threshold | sensitivity | hits | FP | FP/24 h | duplicates |
|---|---|---|---|---|---|
| 0.50 | 0.250 | 6/24 | 4 | 57.2 | 0 |
| 0.30 | 0.333 | 8/24 | 11 | 157.4 | 2 |
| 0.10 | 0.333 | 8/24 | 17 | 243.3 | 2 |
| 0.05 | 0.375 | 9/24 | 23 | 329.1 | 0 |
| 0.01 | 0.542 | 13/24 | 33 | 472.2 | 0 |

### Reviewer-triage simulation, threshold 0.5

Seizure-file recall 0.538 (7/13) · background false-flag rate 0.308 (4/13) · 51 candidate
windows over 1.7 h of recording.

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
