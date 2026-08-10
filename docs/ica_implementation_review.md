# Is the ICA stage implemented correctly?

*BMET4111 Thesis — Sam Gan. Evidence for the ICA fragility study (WS1).*

Short answer: **it faithfully reproduces the original authors' code, and that code has real
methodological defects.** Those two facts are both true and they pull in opposite directions,
which is what makes this stage delicate to touch.

Measured on real TUSZ windows with `experiments/diag_ica_behaviour.py`. The convergence figure
is pooled over **8 recordings / 243 windows**; the return-path and component-count figures come
from a closer pass over **98 windows from 2 recordings**.

| observation | measured |
|---|---|
| FastICA **did not converge** | **142 / 243 windows (58 % pooled; 20–100 % per file)** |
| No EOG component found → returns **raw, unfiltered** data | 10 / 98 (10 %) |
| Components flagged by `find_bads_eog` | 212 |
| Components actually removed | 106 — **half the flagged components are discarded** |
| Per-window peak difference between the two return paths | median **36.6 µV**, max 73 µV |
| Samples per window / per component | 3000 / 158 |
| Samples needed for a stable 19-component decomposition (kN², k≥20) | **≥ 7220 — we have 3000, 2.4× short** |

---

## 1. What the paper specifies

> **Provenance of this quote.** The repository contains **no copy of this paper** — only the two
> two-channel SPMB PDFs and the ZUNA preprint. The specification below was therefore re-verified
> on **2026-08-10 against the arXiv source itself** (arXiv:2103.10900, rendered full text), and
> the wording reproduced here matches that source verbatim. Before this check, the single
> most load-bearing statement in the whole reproduction argument could not be confirmed from
> anything in the repo. **Keep a copy of the paper in the repository.**
>
> Note also that **Omid Kavehei is an author** (Yang, Truong, Maher, Nikpour, Kavehei), so the
> questions this section cannot resolve from the text — chiefly the MNE version used to generate
> the *training* features, and whether the top-1 component restriction was intended — are
> answerable by asking the supervisor rather than by inference.

Yang et al. 2022, §2.3 (arXiv:2103.10900v2):

> "First, we split EEG signals into 12-second segments and applied the ICA algorithm to
> decompose the signal into 19 independent components using the Blind Source Separation (BSS)
> approach. […] We use Pearson correlation to identify which independent sources are highly
> related to eye movement that is detected from two EEG channels, namely 'FP1' and 'FP2'. **We
> remove those independent sources** and reconstruct the EEG signals to obtain the eye movement
> artifact-free signals. […] Our artifact removal is implemented in Python 3.6 with the use of
> library MNE v0.20."

That is the entire specification **in §2.3**. Note what it does not mention: any high-pass filter
before ICA, any correlation threshold, and any limit on how many components to remove.

> **But Appendix B.1 does contain an ICA ablation**, verified from the PDF on 2026-08-10:
> *"We have trained a model without applying ICA and test on a small scale of the RPAH patient
> (2011), the AUC score for non-ICA VS ICA is **0.8089 vs. 0.8993**."* An earlier version of this
> document said the paper offered no evidence for the ICA step. That was wrong — it came from an
> automated retrieval that never surfaced the appendix. The ablation compares *models trained*
> with and without ICA; this project's ICA on/off experiment removes ICA from *inference only*,
> which is a different question. See `docs/RESULTS.md` §4.

## 2. What the code does

`utils/preprocessing.py:64-110`, `ica_arti_remove()` — the same function that generated the
**training** features (`utils/ICA_load_data_elec.py:131`), so this is the operating point the
weights were fitted to.

```python
filt_raw.load_data().filter(l_freq=0.1, h_freq=None)   # 0.1 Hz high-pass  <- not in the paper
ica = ICA(n_components=19, random_state=13)            # full rank, no PCA reduction
ica.fit(filt_raw)                                      # bare `except: return None`
e1, _ = ica.find_bads_eog(filt_raw, threshold=2.0, ch_name='Fp1')
e2, _ = ica.find_bads_eog(filt_raw, threshold=2.0, ch_name='Fp2')
if e1: ica.exclude.append(e1[0])                       # ONLY the top-1 component
if e2: ica.exclude.append(e2[0])
if ica.exclude:
    ica.apply(reconst_raw); return reconst_raw.get_data()*1e6   # filtered + cleaned
return data                                            # <- RAW, UNFILTERED
```

## 2b. Point-by-point conformance to the published pipeline

Every row verified against running code on 2026-08-10. "Spec" is the paper's own wording.

| # | Paper specifies | Code does | |
|---|---|---|---|
| 1 | "12-second segments" | `SEGMENT_S = 12`, 3000 samples @ 250 Hz | **match** |
| 2 | "decompose the signal into 19 independent components" | `ICA(n_components=19)`; measured `n_components_ == 19`, no PCA reduction | **match** |
| 3 | eye movement "detected from two EEG channels, namely 'FP1' and 'FP2'" | `find_bads_eog(..., ch_name='Fp1')` and `'Fp2'` | **match** |
| 4 | "Pearson correlation to identify which independent sources are highly related" | `find_bads_eog` correlates each source with the channel, then **z-scores** the correlations and thresholds the z-score. It also **band-passes the channel 1–10 Hz first** (`l_freq=1, h_freq=10` defaults), which needs a 4096-sample filter over the 3000-sample window — a *second* unrealisable filter, on top of row 7 | **match in kind**, but with two undocumented steps |
| 5 | "**We remove those independent sources**" (plural — every correlated source) | appends only `e1[0]` and `e2[0]` — the **single top** component per channel. Measured: **212 flagged, 106 removed** | **DEVIATION** |
| 6 | *no threshold given* | `threshold=2.0`, against MNE's default of 3.0 — flags more aggressively, then removes conservatively (see 5) | **unspecified + non-default** |
| 7 | *no filter mentioned anywhere* | `filter(l_freq=0.1, h_freq=None)` before ICA. MNE designs an **8251-sample (33.0 s) filter for a 3000-sample (12 s) window**, warns `distortion is likely`, applies it anyway | **DEVIATION** |
| 8 | *silent on the no-EOG-found case* | returns the **raw, unfiltered** input — so ~10 % of windows reach the STFT with different preprocessing from their neighbours | **DEVIATION (bug)** |
| 9 | "Python 3.6" | 3.6.15 | **match** |
| 10 | "library **MNE v0.20**" | **0.19.2**, pinned in `environment-seiz36.yml` and `requirements-seiz36.txt` | **VERSION MISMATCH** |
| 11 | STFT "window length of 250 (or 1 second) and 50 % overlapping" | `framelength=250`; produces **23 time frames**, which is only possible at hop 125 = 50 % | **match** |
| 12 | "remove the DC component" | `d[:, :, 1:]` — 126 bins → **125** | **match** |
| 13 | "data shape will become (n×23×125)" | `(23, 19, 125)` — same content, axes ordered (time, electrode, freq) for the ConvLSTM input | **match** |

### The MNE defaults the paper does not mention

Verified by introspection on the installed MNE 0.19.2, because "Pearson correlation" in the paper
becomes several concrete choices in practice, and a methods section should state them:

| | value | note |
|---|---|---|
| `ICA(method=...)` | `'fastica'` | the default; consistent with the paper's "BSS approach" |
| `ICA(max_iter=...)` | `200` | never raised, and FastICA hits it on most windows (§3.5) |
| `find_bads_eog(l_freq, h_freq)` | `1, 10` | the EOG channel is **band-passed 1–10 Hz** before correlating — not mentioned in the paper |
| `find_bads_eog(threshold)` | code passes `2.0` | this is a **z-score of the correlations**, not a Pearson *r*. An *r* of 2.0 is impossible, so quoting "threshold 2.0" without saying "z-score" is misleading |

The `threshold=2.0` point matters for writing up: it is 2 standard deviations above the mean
correlation across components, not a correlation of 2.

### What to conclude from this table

**The signal path either side of ICA is a faithful implementation.** Segmentation, component
count, EOG channels, and every STFT parameter match the paper exactly, and rows 11–13 were
confirmed by running the code rather than by reading it.

**Four deviations are all inside `ica_arti_remove`**, and rows 5–8 are the same four defects
already ranked in §3 — this table just re-frames them as *departures from the published method*
rather than as *code smells*. Row 5 is the sharpest: the paper says remove **those** sources,
plural; the code removes one per channel, discarding half of what it flagged.

**Row 10 is new and needs disclosure.** The paper names MNE v0.20; this project pins 0.19.2.

### The trap in "make the pipeline the same"

Rows 5–8 cannot simply be "fixed", and this is the central tension of the whole reproduction:

> The **paper** describes what the authors intended. The **code** describes what they did. The
> **weights encode what they did.**

`utils/ICA_load_data_elec.py` imports this same `ica_arti_remove`, so the training features were
generated *with* the top-1 restriction, *with* the 0.1 Hz filter, and *with* the raw-passthrough
branch. Editing the function to match the paper's prose would produce a preprocessing chain the
released weights were never fitted to — closer to the paper, further from a working system. The
existing evidence for how violently that can move is §3.5's max_iter experiment, where a single
convergence change flipped one window from p = 0.902 to p = 0.0014.

So the correct action is **not** to change the code. It is to (a) keep inference bit-faithful to
the training-time function, (b) disclose every row above, and (c) *measure* what each deviation
is worth, which is a thesis result rather than a bug fix.

---

## 3. The defects, ranked

### 3.1 The two return paths give differently-preprocessed data — this is a bug, not a choice

When at least one EOG component is found, the function returns data that has been **0.1 Hz
high-passed and ICA-cleaned**. When none is found, it returns `data` — the **original,
unfiltered input**. The high-pass is silently confounded with the artifact removal.

This fires on **10 % of windows**, and it is not cosmetic. On those windows the *largest
sample-wise* difference between the filtered and raw versions has a median of **36.6 µV** across
windows and a maximum of 73 µV — the scale of the EEG signal itself. (That is a peak statistic,
not a typical sample difference; the typical difference is smaller. Quote it as a peak.) So one in ten windows reaches the STFT having had a materially different preprocessing
from its neighbours, determined by whether an eye-blink happened to be detected.

Nothing in the paper justifies this. It reads as an oversight in the original code.

### 3.2 Half the flagged components are silently thrown away

The paper says "remove **those** independent sources" — all sources correlated with Fp1/Fp2.
`find_bads_eog` returns every component above threshold, but the code appends only `e1[0]` and
`e2[0]`. Measured: **212 components flagged, 106 removed.** The implementation is strictly less
aggressive than the paper describes.

Compounding this, `threshold=2.0` is *more* permissive than MNE's default of 3.0 — so the code
flags aggressively and then removes conservatively, which is an odd pairing that no
documentation explains.

### 3.3 The 0.1 Hz high-pass cannot be realised on a 12-second window, and is the wrong cutoff

Two separate problems.

**It is not physically realisable.** A 0.1 Hz high-pass needs a filter of 8251 samples (33 s at
250 Hz);
the window is 3000 samples (12 s). MNE says so out loud on every call:

```
RuntimeWarning: filter_length (8251) is longer than the signal (3000),
distortion is likely. Reduce filter length or filter a longer signal.
```

`find_bads_eog` raises the same warning at filter_length 4096. These warnings appear on
essentially every window and are currently ignored.

**It is the wrong cutoff even if it were realisable.** The standard recommendation for
ICA pre-filtering is **1–2 Hz**: Winkler et al. (2015) found 1–2 Hz "consistently produced good
results in terms of signal-to-noise ratio, single-trial classification accuracy and the
percentage of near-dipolar ICA components", and EEGLAB's own guidance is that decompositions
are "notably higher quality … when the data is high-pass filtered above 1 Hz". 0.1 Hz leaves
exactly the slow drift that destabilises ICA.

### 3.4 The window is too short for a 19-component decomposition

The standard heuristic, from the EEGLAB documentation, is that

> "finding *N* stable components (from N-channel data) typically requires *more than* *kN²* data
> sample points (at each channel), where N² is the number of weights in the unmixing matrix that
> ICA is trying to learn and *k* is a multiplier."

**The source does not fix a value for *k*** — it says only that *k* "increases with higher
channel counts", and its own worked example (32 channels, 30,800 points) works out at about
**30** points per weight, while still warning that 30 is not enough at 256 components. An earlier
version of this document asserted "k ≥ 20" as though the source stated it; verified 2026-08-10,
it does not.

Taking *k* = 20 as the commonly quoted value, N = 19 needs **≥ 7220** samples; a 12-second window
at 250 Hz gives **3000**, a factor of **2.4** short. At the *k* ≈ 30 implied by EEGLAB's own
example the requirement is 10,830 and the shortfall is **3.6×**. **The conclusion does not depend
on which multiplier is used** — either way the window is short by a wide margin, at 158 samples
per component. Quote the shortfall as "at least 2.4×" and name the *k* you assumed.

This is the most likely driver of §3.5 — the relationship is a strong correlation, not a
demonstrated cause — and it is structural: it cannot be fixed without either
lengthening the window (which changes the model input) or reducing the component count via PCA
(which the paper does not do).

### 3.5 FastICA does not converge on most windows

Plausibly a consequence of §3.3 and §3.4, though that link is a correlation rather than a
demonstrated cause. On 142 of 243 windows (58 % pooled, 20–100 % depending on the recording)
the fixed-point iteration hits `max_iter` without converging, so the returned unmixing matrix is
wherever the iteration happened to stop.

**It *is* deterministic — an earlier claim here that it is not was wrong.** Two fresh runs of
`compute_probs` on the same EDF, in separate processes, are **bit-identical** (8/8 and 25/25
exact). The 0.107 divergence previously cited is between a fresh run and the *stored cache*, and
reflects caches written under conditions that were never recorded — not `random_state=13` failing
to pin FastICA. Corrected 2026-08-10; see `docs/RESULTS.md` §9 and
`experiments/diag_mne_version.py`.

**The rate is strongly recording-dependent, so do not quote a single figure without the range.**
A spot check on 2026-08-10 over the first 8 windows of `aaaaatao_s003_t001` produced **zero**
non-convergence warnings — which is not a contradiction of the 58 % pooled figure but an
instance of the 20–100 % spread, and a caution against generalising from any one recording. The
same check confirmed `ica.n_components_ == 19` (no rank reduction is applied), giving the 158
samples per component quoted in §3.4.

### 3.6 `except:` swallows every failure — *the downstream half is now fixed*

`ica.fit` is still wrapped in a bare `except: return None`, which catches `KeyboardInterrupt`
and `MemoryError` along with everything else and discards the reason. That part stands.

**The consequence described here no longer holds.** This section originally said `None` becomes
a probability of exactly 0.0, "indistinguishable downstream from a confident *no seizure*". That
was true when written and is not true now: `gui/io/infer.py` records
`skip_code[i] = SKIP_ICA_FAILED` alongside the 0.0, the cache persists the array
(`CACHE_VERSION = 2`), and the GUI renders those windows as *"not assessed — ICA decomposition
failed"* rather than as a confident negative. The 0.0 is retained only for backward
compatibility with v1 caches, and consumers are expected to test `skip_code`, not the
probability.

Verified 2026-08-10 against `gui/io/infer.py` and `gui/io/cache.py`. What remains open is the
bare `except` itself, not the sentinel.

---

## 4. So is it "correct"?

Two different questions, two different answers.

| question | answer |
|---|---|
| Does it faithfully reproduce the original authors' pipeline? | **Yes** — it is the same function that generated the training features, called identically. |
| Does it match the paper's *written* description? | **Partly.** 19 components ✅, Pearson correlation vs Fp1/Fp2 ✅, reconstruct ✅. But it removes only the top-1 source where the paper says "those sources" (plural), and adds an undocumented 0.1 Hz high-pass. |
| Is it methodologically sound as ICA? | **No** — §3.1 through §3.5. |

**The critical constraint: the model was trained on the output of this exact function.** The
58 % non-convergence, the top-1 truncation and the inconsistent return paths are all baked into
the operating point the weights expect. This is why "fixing" the ICA is not a free improvement —
it changes the input distribution at inference time relative to training.

That is not speculation. Measured effects of plausible "fixes":

- Capping `max_iter` at 50 changed 7 of 25 window probabilities and **flipped 2 across the 0.5
  threshold**, including a confident detection going 0.902 → 0.0014.
- Moving to modern MNE (1.12) changed probabilities by up to **0.90** with ICA on, versus 5.4e-7
  with ICA off — the entire discrepancy is the ICA.
- Turning ICA off entirely **improves** pooled window AUC (0.7417 vs 0.7157) and runs **30×**
  faster (`docs/RESULTS.md` §4).

---

## 5. Recommendation

**Do not change the ICA stage to "fix" it.** Report it. The thesis position should be:

> The ICA stage was reproduced faithfully from the source implementation, and auditing it
> revealed that per-window ICA on 12-second segments is statistically under-determined: the
> decomposition needs ≥7220 samples for 19 components and receives 3000, FastICA fails to
> converge on 58 % of windows (20-100 % per recording), the 0.1 Hz pre-filter is both
> unrealisable at this window length
> and below the 1–2 Hz recommended for ICA, and half the components flagged as ocular are
> discarded before removal. Because the pretrained weights were fitted to the output of this same
> procedure, these properties are part of the operating point rather than removable defects, and
> correcting them is shown to move individual detections across the decision threshold.

That is a stronger and more honest contribution than silently improving it — it is exactly the
kind of reproducibility finding the literature review already argues for (§2.3 of the report
flags per-window ICA fragility as a known risk), now backed by measurement.

**The one exception worth considering:** §3.1, the inconsistent return path, is a genuine bug
rather than a design decision — no reading of the paper produces it. If you change anything,
change that one, and ship it with an AUC/sensitivity delta from `evaluate_baseline.py` on the
full manifest plus a paragraph in `methodology_statements.md`. Everything else should be
documented, not touched.

**For BMET4112:** this analysis is also the justification for the C++ FastICA port. Porting the
convergence-limited fixed-point iteration to C++ is defensible precisely *because* it can be made
numerically comparable to MNE 0.19 while running far faster — where the algorithmic "fixes" above
cannot be made numerically comparable at all.

---

## References

- Winkler, I., Debener, S., Müller, K.-R., Tangermann, M. (2015). *On the influence of high-pass
  filtering on ICA-based artifact reduction in EEG-ERP.* Proc. IEEE EMBC.
  [PubMed 26737196](https://pubmed.ncbi.nlm.nih.gov/26737196/)
- EEGLAB, *Independent Component Analysis for artifact removal* and *ICA background* —
  the kN² sample-count heuristic and the >1 Hz high-pass guidance.
  [eeglab.org](https://eeglab.org/tutorials/06_RejectArtifacts/RunICA.html)
- Klug, M., Gramann, K., et al. (2024). *Optimizing EEG ICA decomposition with data cleaning in
  stationary and mobile experiments.* Sci. Rep.
  [nature.com/articles/s41598-024-64919-3](https://www.nature.com/articles/s41598-024-64919-3)
- Winkler, I., Haufe, S., Tangermann, M. (2011). *Automatic classification of artifactual ICA
  components for artifact removal in EEG signals.* Behav. Brain Funct. 7:30. — already cited as
  [10] in the progress report.
- `mne.preprocessing.ICA` documentation.
  [mne.tools](https://mne.tools/stable/generated/mne.preprocessing.ICA.html)
