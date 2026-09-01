# Running on Windows, macOS (Apple Silicon) and Linux

*Measured 2026-08-10. Every figure here was produced by running the pipeline, not by reading
release notes.*

## Result

> **CORRECTION, 2026-08-11.** Everything below that attributes cross-machine drift to the ICA is
> wrong, and the error survived three rewrites of this document because the mechanism was never
> tested end to end. Feeding **byte-identical input** to the two stacks isolates it completely:
>
> ```
> input sha256 2f606aea42c9a296   (the same array on both machines)
> TF 1.15.0  (Windows legacy)   p(seizure) = 0.9779213071
> TF 2.21.0  (Linux modern)     p(seizure) = 0.9663650393
> ```
>
> Those are the *same two numbers* the full pipeline produces for that window, to the last digit.
> Checksums of every earlier stage agree: the EDF read is identical, the ICA output agrees to 14
> significant figures, the STFT to 13. **The whole of the drift is the TensorFlow version.**
>
> This explains what the ICA story never could: macOS on arm64 and Linux on x86-64 share no CPU,
> no BLAS and no operating system, yet agree to four decimals — because they share TensorFlow
> 2.21. Both differ from Windows, which runs 1.15.
>
> **It is therefore fixable, and the fix is to run one TensorFlow version everywhere.** The
> obstacle is `docs/known_issues.md` §1, the modern stack not freezing on Windows, which this
> promotes from a tidiness issue to the critical path for cross-machine consistency.
>
> The measurements of drift *magnitude* in this document stand — they were measured, not inferred.
> Only the attribution was wrong. The sections below are kept, marked, rather than deleted,
> because the reasoning that produced the wrong answer is worth seeing.

**A modern stack runs the pipeline on all three platforms. It does not agree with the frozen
Windows build closely enough to leave the reviewer's event list unchanged.**

An earlier version of this document claimed the opposite — "closely enough that no detection
decision changes" — on the strength of 74 windows from two recordings. Measured over 1 723
windows from 50 recordings, with Keras held constant so that the comparison is of platforms and
not of library versions, that claim is false: **11 windows change a detection decision and one
proposed event of 31 is lost.**

Two corrections were needed to get there, and both are worth stating because each looked like a
result. The 74-window sample characterised the bulk of the distribution and never reached the
tail, so the magnitude was wrong. And every modern-stack figure measured before 2026-08-11 10:11
— including the 3 870-window Linux run that first replaced the 74-window one — compared Keras 3
against a Keras 2 reference, so it measured a library version and called it a platform. About
four fifths of the drift reported at that stage was the Keras mismatch.

The *mechanism* described here has been right throughout. Only the magnitude has moved, twice,
in opposite directions.

The disagreement is inherent rather than fixable, for reasons this document explains. What
follows from it is a handling rule, not a repair: **regenerate caches on the machine you evaluate
on, and never mix figures from two machines in one table.**

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

Measured by `experiments/platform_drift.py`, which re-scores recordings the Windows machine has
already cached and compares window-for-window. It refuses any recording whose window grid does
not align exactly, and any whose file fingerprint does not match the reference, so a data
difference cannot be reported as a platform difference. Run against the machine that wrote the
reference it returns 0.000000 over 710 windows, so a non-zero result is real.

> **Correction, 2026-09-01: the figures in this section measured the Keras version, not the
> platform.** `tf-keras` was not pinned, and nothing anywhere set `TF_USE_LEGACY_KERAS`, until
> 36a56b6 on 2026-08-11 10:11. Every modern-stack run before that timestamp therefore used
> Keras 3, whose reimplemented `ConvLSTM2D` loads the same weights and returns different
> probabilities with no error. The Linux arm was measured at 02:24 that morning and the
> 74-window figure the day before, so both compared Keras 3 against a Keras 2.2.5 reference and
> reported the difference as platform drift.
>
> Established by re-scoring one recording both ways. Forcing Keras 3 reproduces the published run
> exactly — 0.9664 / 0.4602 / 0.4367 on the three windows that decide `aaaaatao_s003_t000` — and
> forcing Keras 2 reproduces the Windows reference exactly — 0.9779 / 0.5990 / 0.6511 —
> recovering the event that run had lost. Under matched Keras those two platforms agree on that
> recording to four decimal places on every window.
>
> The mismatched figures are kept below, marked, because they are what a Keras 3 build really
> does and the reasoning around them is sound. They are not platform drift.

### Keras held constant

**Windows (Python 3.6 / MNE 0.19.2 / TF 1.15 / Keras 2.2.5) against macOS arm64 (Python 3.12 /
MNE 1.12.1 / TF 2.21 / tf-keras 2.21), 50 recordings, 1 723 windows.** Same recordings, same
harness, re-run with `TF_USE_LEGACY_KERAS=1`:

| | Keras mismatched | **Keras matched** |
|---|---|---|
| median Δ p(seizure) | 0.000124 | **0.000000** |
| p95 Δ | 0.190 | 0.022 |
| p99 Δ | 0.400 | 0.235 |
| max Δ | 0.956 | **0.929** |
| **windows changing a detection decision** | 50 (2.9 %) | **11 / 1 723 (0.64 %)** |
| recordings whose maximum exceeds 0.136 | 27 / 50 | 10 / 50 |

About four fifths of what was reported as platform drift was the Keras mismatch. What remains
still contradicts the original claims, and by less:

* the **median is now exactly zero** — more than half of all windows agree bit for bit across two
  operating systems, two CPU architectures and two Python stacks. That is better than the 0.0001
  originally published, not worse.
* the **maximum is 0.929, 6.8× the published 0.136**. The tail claim was wrong independently of
  Keras: 74 windows characterise the bulk and never reach the tail.
* **11 of 1 723 decisions change**, so "no detection decision changes" remains false.

The shape is the documented mechanism rather than a systematic offset. On the worst recording,
`aaaaarpv_s002_t007`, the median window difference is exactly **zero** and one window moves
**0.93** — most of the pipeline is deterministic across platforms and a few windows are not,
which is what a non-converged FastICA amplifying a 10⁻¹⁵ whitening difference looks like.

**Events, over the same 50 recordings:**

| | value |
|---|---|
| events proposed on Windows | 31 |
| events proposed on macOS | 33 |
| matched (any temporal overlap) | 30 |
| **present on Windows, absent on macOS** | **1** |
| present on macOS, absent on Windows | 3 |
| recordings whose event list changed | 4 / 50 |
| largest boundary shift on a surviving event | 6.0 s |

One event of 31 is lost, not 41 %. The window and event figures still must not be used as
proxies for one another — 0.64 % of windows against 4 of 50 recordings changing — but the
clinical picture is an event list that is nearly, not substantially, stable.

### The Keras-mismatched figures, kept for the record

Everything below compares Keras 3 against Keras 2 and is **not** a platform measurement. The
Linux arm has not yet been re-taken with Keras held constant; until it is, no Linux drift figure
should be quoted.

**Windows (Python 3.6 / MNE 0.19.2 / TF 1.15) against Linux x86-64 (Python 3.12 / MNE 1.12.1 /
TF 2.21), 111 recordings, 3 870 windows — Keras 3 against Keras 2:**

| | value |
|---|---|
| median Δ p(seizure) | 0.000058 |
| p95 Δ | 0.087 |
| p99 Δ | 0.303 |
| max Δ | 0.788 |
| windows changing a detection decision | 57 / 3 870 (1.5 %) |

| | value |
|---|---|
| events proposed on Windows | 44 |
| events proposed on Linux | 33 |
| matched (any temporal overlap) | 26 |
| present on Windows, absent on Linux | 18 |
| present on Linux, absent on Windows | 7 |
| recordings whose event list changed | 14 / 111 |
| largest boundary shift on a surviving event | 12.0 s |

Two observations from that run survive the correction, because neither depends on the magnitude:

Window flips and event changes are **not related monotonically**. `aaaaaarq_s014_t003` lost an
event with **zero** window flips — per-second averaging means two windows can each stay on the
same side of the threshold while their average crosses it. So a window-level figure cannot be
used as a proxy for what a reviewer sees, in either direction.

And on `aaaaatao_s003_t000` the highest-scoring window in the recording — 0.9779 — produced **no
event on either platform**, because averaging it with quiet neighbours gives
(0.9779 + 0.0011)/2 = 0.489. The event that exists there comes from two adjacent moderate
windows, 0.599 and 0.651. That is a property of the shipped decision stage, not of drift: a
window scoring 0.98 can be invisible to a reviewer while two scoring 0.6 are not.

### Scope, stated plainly

* This compares **stack and platform together**. The Windows arm is Python 3.6 / TF 1.15; both
  others are Python 3.12 / TF 2.21. It is not a like-for-like replacement for the 74-window
  figure, which varied only the stack on one machine.
* Recordings were selected **shortest-first**, to cover more recordings per hour of compute. That
  is not a random sample of the corpus, and the event counts are small in absolute terms.
* Three recordings were dropped because their stem names two different files; five stems appear
  twice in `manifest_full.csv`, the same TUH session under the `01_tcp_ar` and `03_tcp_ar_a`
  montages.

Against the **0.90** previously recorded for "MNE 1.12", the typical case is far better — but the
tail is not, and the reason is worth stating precisely.

Probing the ICA stage directly, MNE 0.19.2 vs 1.12.1 on the same window:

| stage | max difference |
|---|---|
| filtered signal (0.1 Hz high-pass) | 6.4 × 10⁻²⁰ |
| PCA whitening components | 7.6 × 10⁻¹⁵ |
| PCA mean | 6.7 × 10⁻¹⁶ |
| **FastICA unmixing matrix** | **6.76** |

> **The input to FastICA is identical to machine precision. The output is completely different —
> with the same pinned algorithm and the same fixed seed (`random_state=13`).**

### The amplification, measured directly

Inferring this from two stacks disagreeing conflates the perturbation with everything else that
differs between them. Perturbing the whitened input by hand, at the scale two BLAS builds differ
by, on three windows of one recording:

| window | FastICA converged? | input Δ | unmixing Δ |
|---|---|---|---|
| t=18 | yes, 42 iterations | 9.8 × 10⁻¹⁵ | **0.0000** |
| t=30 | **no**, still running at 20 000 | 1.2 × 10⁻¹⁴ | **0.9546** |
| t=36 | yes, 124 iterations | 1.2 × 10⁻¹⁴ | **0.7337** |

Roughly fourteen orders of magnitude of amplification, on two windows out of three.

**An earlier version of this section blamed non-convergence, and that is not what the measurement
shows.** `t=36` converged in 124 iterations and amplified 10⁻¹⁴ into 0.73 regardless; `t=18`
converged in 42 and was perfectly stable. Non-convergence and instability are both symptoms of
the same thing — the ICA solution is **ill-conditioned on most windows of this data** — and
convergence does not confer stability.

Nor is more compute a remedy. `t=30` returns a unmixing **identical to three decimals** at
`max_iter` of 200, 1 000 and 20 000. The iteration is not wandering; it is parked, changing by
slightly more than `tol=1e-4` per step. Raising the budget buys nothing.

**No amount of pinning fixes this.** With fourteen orders of magnitude of amplification, agreement
would require the whitening to be bit-identical, which means no BLAS anywhere in the chain. It is
a property of the conditioning of this decomposition on this data, not a defect in any library.

### Three routes to a fix, and why none is taken

| route | verdict |
|---|---|
| raise `max_iter` so it converges | **Measured, no.** The unmixing is unchanged at 20 000 iterations, and a window that *did* converge amplified anyway. |
| drop the ICA | **No.** It is worth ~2.2× on detection, and the released weights were fitted to features generated with it (`utils/ICA_load_data_elec.py` imports the same `ica_arti_remove`). |
| bit-identical linear algebra | **Not achievable.** A research project, not a fix. |

The remaining option is the one already in force: **treat the machine as part of the provenance.**
Each machine is exactly self-consistent — the drift harness returns 0.000000 comparing a machine
to itself — so a recording analysed once, on one machine, by one reviewer is reproducible. What is
not reproducible is the same recording analysed on two machines, and no clinical workflow requires
that. It is the *thesis tables* that require it, which is why the handling rule is a reporting
rule.

This also unifies previously separate observations, all the same mechanism:

- the **0.107** drift between a fresh run and the stored caches (`RESULTS.md` §9),
- the **0.90** attributed to "MNE 1.12" — which now reads as an ordinary draw from this tail
  rather than one unlucky window,
- the **0.929** maximum measured here with Keras held constant. The 0.788 and 0.956
  recorded earlier were Keras-mismatched and measure a library version, not a platform.

Two facts rule out the explanations that would have made this fixable. The ICA *implementation*
is bit-identical on every platform: `utils/fastica_pinned` is verified installed on both stacks,
and the legacy Windows stack runs scikit-learn **0.22.2** — the exact version the pin transcribes.
And the fault is not TensorFlow's: swapping TF alone moves p(seizure) by 6.8 × 10⁻⁹.

The consequence is a statement the thesis should make once, clearly: **this pipeline is
statistically reproducible in the bulk of its distribution and not reproducible in its tail, and
the cause is the non-converged ICA the weights were trained with.** For per-window probabilities
that means quoting two significant figures. For **events** it means something weaker than the 41 %
once recorded here, and still not nothing — the list of candidates a reviewer is shown is
machine-dependent, and 1 of 31 changed between two correct builds of the same application, on
4 of 50 recordings. Caches must be regenerated in a single pass per platform,
which `RESULTS.md` already requires for other reasons.

### What this does and does not say about the tool

It does **not** say either machine is wrong. Neither decomposition is the true one, because
FastICA stops at `max_iter` without converging — verified on real data, `n_iter_` reaches 200
every window. The drift is the *width of the operating point*, not an error.

It does say that a reviewer must not be told the tool is deterministic across machines, and that
an audit trail should record the environment that produced any annotation. `gui/io/cache.py`
stamps `sklearn`, `fastica_pinned`, `machine`, `tensorflow` and the oneDNN flag into every cache
for exactly this reason.

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
  on your own laptop is a larger cost than the drift, provided no figure crosses machines.
- **If the thesis figures are ever regenerated, do it once, on one machine, and say which.** This
  was previously advice about the third decimal. It is now a correctness requirement: a table
  assembled from two machines can disagree with itself about whether a seizure was proposed.
- **Say this in the limitations, rather than leaving it for a reader to find.** The honest
  sentence is that the detector's *proposals* are reproducible in distribution but not
  individually across machines, and that the human-in-the-loop design is what makes that
  tolerable — a reviewer adjudicates every proposal, and unproposed time was never claimed to be
  seizure-free (`docs/INTENDED_USE.md`).
