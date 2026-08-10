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

Yang et al. 2022, §2.3 (arXiv:2103.10900v2):

> "First, we split EEG signals into 12-second segments and applied the ICA algorithm to
> decompose the signal into 19 independent components using the Blind Source Separation (BSS)
> approach. […] We use Pearson correlation to identify which independent sources are highly
> related to eye movement that is detected from two EEG channels, namely 'FP1' and 'FP2'. **We
> remove those independent sources** and reconstruct the EEG signals to obtain the eye movement
> artifact-free signals. […] Our artifact removal is implemented in Python 3.6 with the use of
> library MNE v0.20."

That is the entire specification. Note what it does **not** mention: any high-pass filter before
ICA, any correlation threshold, and any limit on how many components to remove.

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

The standard heuristic is that resolving N components from N channels needs **more than kN²
samples per channel, with k ≥ 20**. For N = 19 that is **≥ 7220 samples**. A 12-second window at
250 Hz gives **3000** — a factor of 2.4 short, or 158 samples per component.

This is the most likely driver of §3.5 — the relationship is a strong correlation, not a
demonstrated cause — and it is structural: it cannot be fixed without either
lengthening the window (which changes the model input) or reducing the component count via PCA
(which the paper does not do).

### 3.5 FastICA does not converge on most windows

Plausibly a consequence of §3.3 and §3.4, though that link is a correlation rather than a
demonstrated cause. On 142 of 243 windows (58 % pooled, 20–100 % depending on the recording)
the fixed-point iteration hits `max_iter` without converging, so the returned unmixing matrix is
wherever the iteration happened to stop.

**And it is not even deterministic.** `random_state=13` is insufficient: re-running
`compute_probs` on an unchanged EDF reproduces the cache for some windows and then diverges, one
measured window moving by 0.107. An earlier version of this document claimed the non-convergence
was "at least deterministic"; that was wrong. See `docs/RESULTS.md` §8.

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
