# Running on Windows, macOS (Apple Silicon) and Linux

*Measured 2026-08-10. Every figure here was produced by running the pipeline, not by reading
release notes.*

## Result

**A modern stack runs the pipeline on all three platforms and agrees with the frozen Windows
build closely enough that no detection decision changes.** It does **not** agree bit-for-bit, and
this document explains why that is inherent rather than fixable.

| | current (Windows only) | portable target |
|---|---|---|
| Python | 3.6.15 | **3.11** |
| numpy | 1.16.4 | **1.26** (`<2`) |
| scipy | 1.4.1 | **1.17** |
| scikit-learn | 0.22.2 | **any** — the decomposition is now vendored |
| MNE | 0.19.2 | **1.12** |
| TensorFlow | 1.15 | **2.21** |
| Keras | 2.2.5 | `tf.keras` |
| **Apple Silicon** | **impossible** | **works** |

Validated end-to-end: `python 3.11.15 / mne 1.12.1 / scikit-learn 1.9.0 / tensorflow 2.21.0`
loads the released `convlstm_ICA_12_train.h5` and scores real recordings.

---

## What had to change, and why it was only one thing

### The decomposition was scikit-learn's, not MNE's

`mne.preprocessing.ICA(method='fastica')` does not implement ICA. It whitens, calls
`sklearn.decomposition.FastICA`, and keeps `components_`. Measured by swapping one library at a
time in isolation, 25 real windows each:

| change | max Δ p(seizure) |
|---|---|
| MNE 0.19.2 → 0.20.0 | 0.000000 |
| MNE 0.19.2 → 0.23.4 | 0.000000 |
| Keras 2.2.5 / TF 1.15 → Keras 2.0.8 / TF 1.4.0 | 6.8 × 10⁻⁹ |
| Keras 2.2.5 / TF 1.15 → tf.keras / TF 2.6.2 | 6.8 × 10⁻⁹ |
| **scikit-learn 0.22.2 → 0.24.2** | **2.1 × 10⁻³** |

So MNE and TensorFlow were never the obstacle. `utils/fastica_pinned.py` transcribes scikit-learn
0.22.2's FastICA, depends only on numpy and scipy, and is installed automatically when
`utils.preprocessing` is imported. With it in place, sklearn 0.24.2 becomes **bit-identical** to
the 0.22.2 baseline — the 0.002 sensitivity is gone.

### TensorFlow 1.15 was never required

The released weights load into a `tf.keras` rebuild of the same graph — same 384,846 parameters —
and predict to within **6.8 × 10⁻⁹** of TF 1.15. TF 1.15 has no arm64 wheel and never will;
TF 2.x does. `experiments/diag_tf2_port.py` is the rebuild.

---

## Why bit-identity is impossible, and why that is acceptable

Running the full pipeline on the modern stack against the Windows baseline, 74 windows over two
recordings:

| | value |
|---|---|
| median Δ p(seizure) | **0.0001** |
| mean Δ | 0.003 |
| max Δ | **0.136** |
| windows moved > 0.01 | 4 / 74 |
| **windows changing a detection decision** | **0 / 74** |

Against the **0.90** previously recorded for "MNE 1.12", that is more than a sixfold reduction —
but it is not zero, and the reason is worth stating precisely.

Probing the ICA stage directly, MNE 0.19.2 vs 1.12.1 on the same window:

| stage | max difference |
|---|---|
| filtered signal (0.1 Hz high-pass) | 6.4 × 10⁻²⁰ |
| PCA whitening components | 7.6 × 10⁻¹⁵ |
| PCA mean | 6.7 × 10⁻¹⁶ |
| **FastICA unmixing matrix** | **6.76** |

> **The input to FastICA is identical to machine precision. The output is completely different —
> with the same pinned algorithm.**
>
> That is chaotic amplification. FastICA **does not converge** on most windows of this corpus
> (`docs/ica_implementation_review.md` §3.5), so the returned unmixing is wherever the fixed-point
> iteration happened to stop after 200 steps. A 10⁻¹⁵ perturbation in the whitening — the
> unavoidable difference between two scipy `eigh`/`svd` builds — compounds into a different
> stopping point.

**No amount of pinning fixes this.** It would require bit-identical linear algebra across
platforms, which is not achievable. It is a property of running a non-convergent iterative
algorithm, not a defect in any library.

This also unifies three previously separate observations, all the same mechanism:

- the **0.107** drift between a fresh run and the stored caches (`RESULTS.md` §9),
- the **0.90** attributed to "MNE 1.12" — one unlucky window, not a systematic shift,
- the residual **0.136** here.

The consequence is a statement the thesis should make once, clearly: **this pipeline is
statistically reproducible, not bit-reproducible, and the cause is the non-converged ICA the
weights were trained with.** Numbers must be quoted to two significant figures and caches
regenerated in a single pass per platform — which `RESULTS.md` already requires for other reasons.

---

## What migrating actually costs

1. **Regenerate the caches on the target platform.** ~70 minutes for the 305-file corpus, one
   command, already sharded. Reported figures move in the third decimal, exactly as they did in
   the 2026-08-10 regeneration on Windows.
2. **Re-verify.** `experiments/replicate_paper_auc.py` and `evaluate_baseline.py` reproduce every
   headline number, and `tests/test_doc_numbers.py` pins the documents to them.
3. **Build per platform.** PyInstaller does not cross-compile: Windows, macOS and Linux each need
   a build on that machine.
4. **macOS notarisation** if you want to avoid right-click → Open.

## Recommendation

- **Do not switch the reported numbers mid-write-up.** The Windows build and its caches are
  verified and internally consistent. Freeze them for the thesis.
- **Do use the modern stack for development on the MacBook.** It works, and being unable to work
  on your own laptop is a larger cost than a third-decimal difference in a number nobody has
  quoted yet.
- **If the thesis figures are ever regenerated, do it once, on one machine, and say which.**
