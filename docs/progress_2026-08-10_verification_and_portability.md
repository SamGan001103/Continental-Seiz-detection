# Progress — 2026-08-10 — verification, regeneration, and portability

*BMET4111 Thesis — Sam Gan. Everything measured this day, with the commands that reproduce it.*

Three pieces of work, in order: verify every claim against sources, regenerate the numbers from
scratch, and make the application run on Windows, macOS Apple Silicon and Linux.

---

## Part 1 — Source verification

Full record: `docs/source_verification.md`.

The source paper (Yang et al., *Expert Syst. Appl.* 207:118083) was obtained as a PDF. **Until
then no copy existed in the repository**, and every claim about it had been checked against an
online rendering. That rendering produced two errors of its own, both corrected.

### Verified against the PDF

- **Architecture matches Fig. 4 and §2.4.1 exactly**, checked by building the model and printing
  shapes: input 23×19×125, BN, ConvLSTM 16/(19×3), 32/(1×3), 64/(1×3) at stride (1×2), flatten to
  **896** — the specific width printed in the figure — FC 256 sigmoid, FC 2, dropout 0.5, Adam
  5 × 10⁻⁴. **384,846 parameters.**
- Numbers: TUH dev 170.3 h (Table 1); RPAH 0.82 / 76.68 % / 56.55 (Tables 3, 7); EPILEPSIAE 0.81;
  92.19 % / 0 for the arbiter pilot; the SDR 30-second merge footnote; Fig. 5's TUH-TUH definition.
- The "83× more seizure-dense" figure is confirmed: 3.16 vs 0.038 seizures/hour.

### Corrected

| was | is |
|---|---|
| results table cited as "Table 2" | it is **Table 3**; Table 2 is a baseline-model comparison |
| 39.15 % / 22.83 attributed to Golmohammadi in `comparable_scoring.py`'s docstring | **Shah et al. (2017)**, TUH v1.1.0 — the *code* was right, the docstring wrong |
| "the paper provides no ICA ablation" | **it does**, Appendix B.1: non-ICA vs ICA = **0.8089 vs 0.8993** |
| "the lens is PWI/PEI, not PWA/PEI" | **PWA is correct** — it is the *method*, PWI and PEI are the *indices*. Seven documents were wrongly renamed and reverted. |

The paper's ICA ablation does **not** contradict ours: theirs compares *models trained* with and
without ICA; ours removes ICA from *inference* using ICA-trained weights. Different questions.

### Method references — checked

Verified as cited: **Bröcker (2009)** QJRMS 135(643):1512–1519; **Murphy (1973)** (REL − RES + UNC
exactly as stated); **Guo et al.** arXiv:1706.04599; **Winkler et al. (2015)** EMBC 4101–4105
(1–2 Hz wording matches); **Wharton et al. (1994)** (four questions match).

One did not survive: the ICA sample-size heuristic was stated as "*kN²*, **k ≥ 20**" as though
EEGLAB fixed the multiplier. **It does not** — it says *k* "increases with higher channel counts"
and its own example implies ≈ 30. Corrected to state the assumption: **2.4× short at k = 20,
3.6× at k = 30**. Conclusion unaffected.

---

## Part 2 — Cache regeneration

`docs/RESULTS.md` §9 said "inference is not bit-reproducible" and blamed `random_state`. **That
diagnosis was wrong.** Measured:

| comparison | result |
|---|---|
| fresh run vs fresh run, same version | **bit-identical** (8/8, 25/25) |
| fresh run vs **stored cache** | **2/8 exact, max 0.1067** |

Inference is deterministic; the **caches** were stale, written under conditions nobody recorded.
All 80 caches inspected had `mne=None, commit=None, host=None` — no environment provenance at
all. `gui/io/cache.environment_stamp()` now records MNE, numpy, scipy, Python, host and platform.

**All 305 readable caches regenerated** in one 8-shard pass. One EDF is corrupt
(`aaaaaqxr_s003_t000`, `filesize 47333376 != 56006*1320+8192`).

### What moved

| | before | after |
|---|---|---|
| window AUC, pure windows | 0.8901 | **0.8873** |
| window AUC, non-overlapping | 0.8856 | **0.8838** → rounds to **0.88**, not 0.89 |
| event sensitivity @ 0.5 | 49.4 % (42/85) | **48.2 % (41/85)** |
| false alarms | 204.4 | 204.4 |
| decision-stage ablation | 478.0 → 222.9 (2.1×) | **484.3 → 219.5 (2.2×)** |
| seizures with no model response | 19 of 85 (22 %) | **16 of 85 (19 %)** |
| — as share of missed | 44 % | **36 %** |

**The headline is unchanged**: AUC **0.89**, CI [0.82, 0.94] by recording, **[0.73, 0.95] by
patient**, against 0.84 published. The old caches overstated the separation failure.

Propagated to every document and figure; `tests/test_doc_numbers.py` pins them.

---

## Part 3 — Portability

Full record: `docs/portability.md`.

### The measurement that mattered

One library swapped at a time, in isolation, 25 real windows each:

| change | max Δ p(seizure) |
|---|---|
| MNE 0.19.2 → 0.20.0 (the paper's version) | **0.000000** |
| MNE 0.19.2 → 0.23.4 | **0.000000** |
| Keras 2.2.5 / TF 1.15 → Keras 2.0.8 / TF 1.4.0 (the paper's) | 6.8 × 10⁻⁹ |
| Keras 2.2.5 / TF 1.15 → tf.keras / TF 2.6.2 | 6.8 × 10⁻⁹ |
| **scikit-learn 0.22.2 → 0.24.2** | **2.1 × 10⁻³** |

**MNE was never the fragile part.** `ICA(method='fastica')` delegates to
`sklearn.decomposition.FastICA`. And scikit-learn 0.22.2 has no macOS arm64 wheel — that single
dependency was the whole obstacle to Apple Silicon.

### The fix

`utils/fastica_pinned.py` transcribes scikit-learn 0.22.2's FastICA (the `whiten=False` path MNE
uses), depending only on numpy and scipy. Installed automatically on importing
`utils.preprocessing`. **Bit-identical** to what it replaces, and bit-identical under sklearn
0.24.2 where unpinned code diverges. Nine tests, including that *non-convergence* reproduces —
that is baked into the trained operating point.

TensorFlow 1.15 was never required: the released weights load into a `tf.keras` rebuild of the
same graph, same 384,846 parameters, agreeing to 6.8 × 10⁻⁹.

### Validated stack

`Python 3.11.15 / numpy 1.26.4 / scipy 1.17.1 / scikit-learn 1.9.0 / MNE 1.12.1 /
TensorFlow 2.21.0` runs the pipeline with the released weights.

| | value |
|---|---|
| median Δ p(seizure) vs the Windows baseline | **0.0001** |
| max Δ | 0.136 |
| windows changing a detection decision | **0 / 74** |
| previously recorded for "MNE 1.12" | 0.90 |

### Why it is not zero — and why that is inherent

| stage, MNE 0.19.2 vs 1.12.1 | max difference |
|---|---|
| filtered signal (0.1 Hz high-pass) | 6.4 × 10⁻²⁰ |
| PCA whitening components | 7.6 × 10⁻¹⁵ |
| **FastICA unmixing matrix** | **6.76** |

**Identical input, completely different output, same pinned algorithm.** FastICA does not
converge on this corpus, so a 10⁻¹⁵ perturbation in the whitening compounds over 200 iterations
into a different stopping point. No pinning can fix this; it would need bit-identical linear
algebra across platforms.

This unifies three previously separate observations — the 0.107 fresh-vs-cached drift, the 0.90
attributed to MNE 1.12 (one unlucky window, not a systematic shift), and this 0.136 residual.
**One sentence for the thesis: the pipeline is statistically reproducible, not bit-reproducible,
because of the non-converged ICA the weights were trained with.**

---

## Commands that reproduce everything here

```bash
# verification
python experiments/diag_mne_version.py --n 25 --out a.npz
python experiments/diag_mne_version.py --n 25 --out b.npz --mne-path /path/to/mne020
python experiments/diag_mne_version.py --compare a.npz b.npz
python experiments/diag_mne_version.py --n 25 --pin-fastica --out pinned.npz
python experiments/diag_tf_version.py  --make-inputs --n 20 --out tfx.npz
python experiments/diag_tf2_port.py    --inputs tfx.npz --out tf2.npz --lib-path /path/to/tf26

# regeneration and re-evaluation
python precompute_probs.py "sample_data/**/*.edf" --overwrite --shard i/8
python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest_full.csv
python experiments/evaluate_baseline.py   --manifest artifacts/zuna_thesis/manifest_full.csv
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest_full.csv
python experiments/calibration.py         --manifest artifacts/zuna_thesis/manifest_full.csv
python experiments/thesis_figures.py      --out docs/figures

# secondary studies
python experiments/zuna_auc_interval.py
python experiments/diag_ica.py
python experiments/diag_ica_paper_variant.py --variant all_components
python experiments/evaluate_adaptive.py
```

## Corrections made to my own earlier work, this day

Recorded because a thesis should show its error-correction, not hide it.

1. Fabricated explanation for a 0.6887/0.6878 AUC gap — the real cause was per-arm masking.
2. Denominator error in the clinician-facing document (22 % of *missed* vs of *all*).
3. ZUNA arms compared on different window sets; the fix reproduced the published 0.6878 exactly.
4. An argparse default that silently overrode the module default, producing a run labelled p50
   that was actually p85.
5. Oracle-dependent evidence used to motivate adaptive normalisation.
6. "The paper provides no ICA ablation" — it does, in Appendix B.1.
7. "PWA is wrong, it's PWI" — PWA was right; reverted across seven documents.
8. "k ≥ 20" asserted as sourced when EEGLAB does not fix the multiplier.
9. "Apple Silicon is blocked" — it is not; TF 1.15 was never required.

## Test suite

**91 → 176 tests**, all passing. New this day: `test_deployment_paths.py` (22),
`test_paper_conformance.py` (20), `test_doc_numbers.py` (16), `test_fastica_pinned.py` (9).
