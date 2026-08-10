# Running on Windows, macOS (Apple Silicon) and Linux

*Assessed 2026-08-10 by measurement, not recollection. Every number below was produced by running
the pipeline, not by reading release notes.*

## The answer changed

An earlier version of this document said Apple Silicon was **blocked**. That was based on the
correct observation that TensorFlow 1.15 has no arm64 wheel, and the incorrect assumption that
this project needs TensorFlow 1.15. **It does not.**

> **The released weights run unchanged under TensorFlow 2**, which has arm64 wheels.
> Measured over 20 real ICA'd windows: **max difference 6.75 × 10⁻⁹**, no window crossing the
> detection threshold. Same 384,846 parameters, same `.h5` file, loaded into `tf.keras`.

Apple Silicon is **not blocked**. There is one real obstacle left, and it is not the one everyone
assumes.

---

## What was measured

Each library was swapped in isolation (`pip install --no-deps --target`), so `seiz36` was never
modified and exactly one thing changed per run. Same 25 windows, same weights, same recording.

| change | max Δ p(seizure) | verdict |
|---|---|---|
| MNE 0.19.2 → **0.20.0** (the paper's version) | **0.000000** | bit-identical |
| MNE 0.19.2 → **0.23.4** (newest supporting Py 3.6) | **0.000000** | bit-identical, 25/25 exact |
| Keras 2.2.5 / TF 1.15 → **Keras 2.0.8 / TF 1.4.0** (the paper's) | 6.8 × 10⁻⁹ | float32 noise |
| Keras 2.2.5 / TF 1.15 → **tf.keras / TF 2.6.2** | **6.8 × 10⁻⁹** | float32 noise |
| scikit-learn 0.22.2 → **0.24.2** | **2.1 × 10⁻³** | **the sensitive component** |

Reproduce with `experiments/diag_mne_version.py`, `diag_tf_version.py`, `diag_tf2_port.py`.

### The one that matters

**MNE is not the fragile part — scikit-learn is.** `mne.preprocessing.ICA(method='fastica')`
delegates straight to `sklearn.decomposition.FastICA`:

```python
from sklearn.decomposition import FastICA
ica = FastICA(whiten=False, random_state=random_state, ...)
```

MNE contributes the whitening and the component selection; **sklearn does the decomposition**.
That is why MNE can move four minor versions with zero change while two sklearn minor versions
move probabilities by 0.002. It also explains the previously recorded 0.90 divergence under
"MNE 1.12": that measurement changed MNE, sklearn, numpy and scipy simultaneously, and attributed
the result to the wrong one.

It is consistent with the other half of that measurement, recorded in
`docs/deployment_roadmap.md`: with `use_ica=False` the whole pipeline ported to Python 3.10 with
a max difference of **5.4 × 10⁻⁷**. Everything except the ICA already ports.

---

## So what actually blocks Apple Silicon

**Nothing structural. One component needs pinning.**

Every dependency has a macOS arm64 wheel — numpy, scipy, scikit-learn, PyQt5, pyEDFlib — and MNE
and pyqtgraph are pure Python, so their lack of an arm64-specific wheel is meaningless.

| | status on Apple Silicon |
|---|---|
| Python 3.10–3.12 | native |
| numpy, scipy, scikit-learn, PyQt5, pyEDFlib | arm64 wheels |
| MNE, pyqtgraph | pure Python |
| **TensorFlow 2.x** | **arm64 wheels — and gives identical numbers** |
| TensorFlow 1.15 | no arm64 wheel, and **not needed** |
| **`ica_arti_remove` reproducibility** | **the one open item** |

### On "I couldn't install old Python on the MacBook"

That is correct and it does not matter. CPython gained macOS arm64 support in **3.9.1**; the 3.6
branch ended in December 2021 without ever producing an arm64 build. **Python 3.6 cannot run
natively on Apple Silicon and never will.**

The point is that the migration does not want old Python. It wants **new** Python — 3.10–3.12 —
which installs natively from python.org, Homebrew or conda-forge. The version wall you hit was
real, and it disappears once the target moves forwards instead of backwards.

---

## The remaining work, concretely

**1. Pin the ICA. This is the whole problem.**

`sklearn.decomposition.FastICA` is a self-contained fixed-point algorithm of a few hundred lines
of numpy. Two options:

- **Vendor it** — copy sklearn 0.22.2's `_fastica.py` into the repository as
  `utils/fastica_pinned.py` and call it directly, so the decomposition stops depending on the
  installed sklearn version at all.
- **Pin the version** — require `scikit-learn==0.22.2`. Simpler, but 0.22.2 has no arm64 wheel,
  so it does not solve the Mac problem. **Vendoring is the option that actually helps.**

Either way the acceptance test already exists: run `diag_mne_version.py` against the current
caches and require bit-identity, or a difference below the 0.107 fresh-vs-cached floor from
`RESULTS.md` §9.

**2. Port `models/deep_conv_lstm.py` to `tf.keras`.** Already done, as a by-product — see
`experiments/diag_tf2_port.py`, which rebuilds the graph in `tf.keras` and loads the released
weights. It needed only the Keras 1 idioms replaced (`Model(input=…)`, `keras.layers.normalization`).

**3. Build on each OS.** PyInstaller does not cross-compile, so Mac and Linux each need a build
*on* that machine. You have the MacBook, which is the machine that matters.

**4. Notarisation for macOS.** Gatekeeper is stricter than SmartScreen; distributing outside the
App Store without notarisation means the user must right-click → Open. A paid Apple Developer
account removes it.

---

## Recommendation

The migration is **substantially smaller than previously believed**, and the evidence is in the
table above rather than in an estimate.

1. **Do not migrate mid-write-up.** The Windows build is verified and the numbers are settled.
   Everything below is the next increment, not this week's work.
2. **The first experiment is small and decisive**: vendor sklearn 0.22.2's FastICA, run it under
   the current environment, and require bit-identity with the existing caches. If that passes, the
   ICA stops being version-sensitive and the rest is packaging.
3. **Then build on the MacBook.** Python 3.11 + TF 2.x + vendored FastICA + PyQt5, one
   `build_app.sh` mirroring the Windows script and its four gates.
4. **Treat "runs on the student's own laptop" as the real requirement it is.** Being unable to
   develop on your own machine is a bigger cost than any of the deployment questions this document
   started with.
