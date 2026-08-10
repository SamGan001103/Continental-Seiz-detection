# Running on Windows, macOS and Linux

*Assessed 2026-08-10 against actual wheel availability, not from memory.*

## The short answer

| target | status | why |
|---|---|---|
| **Windows x86-64** | **works today** | built, smoke-tested, shipping |
| **Linux x86-64** | **buildable** — needs a Linux machine | `tensorflow-1.15-cp36-manylinux2010_x86_64.whl` exists |
| **macOS Intel (x86-64)** | **buildable** — needs an Intel Mac | `tensorflow-1.15-cp36-macosx_10_9_x86_64.whl` exists |
| **macOS Apple Silicon** | **blocked** | **no arm64 wheel for TensorFlow 1.15 at any Python version** |

Two facts drive everything below.

**1. PyInstaller does not cross-compile.** There is no way to produce a Mac or Linux app from
Windows. Each target needs a build *on* that operating system — a machine, a VM, or a CI runner.

**2. Apple Silicon is a hard block, and that is what "a Mac" means now.** Every Mac sold since
late 2020 is ARM. Checked directly against the PyPI release files for TensorFlow 1.15.0 and
1.15.5, the complete wheel list is:

```
cp36-cp36m-win_amd64.whl
cp36-cp36m-manylinux2010_x86_64.whl
cp36-cp36m-macosx_10_9_x86_64.whl        (+ cp27/cp35/cp37 variants)
```

**There is no `macosx_*_arm64` wheel.** This is not a packaging inconvenience; TF 1.15 predates
Apple Silicon entirely.

---

## What the code already does right

The application itself is close to portable. Audited 2026-08-10:

- **No hardcoded Windows paths.** Everything goes through `os.path.join`.
- **`gui/io/zuna.py` already branches** on `os.name` for process-group handling.
- **`gui/paths.py` now follows each platform's convention** for its data directory:
  `%LOCALAPPDATA%\SeizureReview` on Windows, `~/Library/Application Support/SeizureReview` on
  macOS, `$XDG_DATA_HOME/SeizureReview` (falling back to `~/.local/share`) on Linux. Pinned by
  tests. Before this it relied on `%LOCALAPPDATA%` simply being unset off Windows, which "works"
  by accident and would have left a stray `~/SeizureReview` where no Mac user would look.
- **PyQt5 and pyqtgraph are cross-platform**; the offscreen self-test already runs headless,
  which is what a Linux CI runner needs.

**What is Windows-only is the *packaging*, not the application**: `setup.bat`, `launch_gui.bat`
and `packaging/build_app.bat`. Those need shell equivalents, which is ordinary work.

---

## The realistic options, in order of cost

### A. Linux build — cheapest, and probably what you actually want

A `manylinux2010` wheel exists, so the environment resolves. Needs:

1. A Linux machine or VM with conda (the repo already has `run_zuna_vm_batch.sh`, so a Linux VM
   is not new territory here).
2. A `build_app.sh` mirroring `build_app.bat` — same four gates: weights hash, tests, freeze,
   smoke-test the frozen binary.
3. One-folder output, same as Windows.

**Worth doing if** the hospital has Linux workstations, or you want a CI runner that rebuilds and
smoke-tests on every commit. The second reason is the better one.

### B. macOS Intel build — possible, shrinking audience

Same shape as Linux, but the audience is Macs from 2020 or earlier. Also needs code signing and
notarisation to run without a Gatekeeper warning — a paid Apple Developer account, and a stricter
gate than Windows SmartScreen. **Low value for the effort.**

### C. Apple Silicon — only via Rosetta, and not recommended

An x86-64 Python 3.6 under Rosetta 2 could in principle run TF 1.15. Against it: Rosetta on an
unsupported EOL Python, translating an EOL TensorFlow, with an ICA stage that is already ~90 % of
runtime. It would be slow, fragile, and impossible to support at a distance.

**The honest answer for Apple Silicon is: not without migrating off TF 1.15** — and that
migration is exactly what `docs/deployment_roadmap.md` argues against doing this semester,
because MNE 1.x's ICA changes probabilities by up to 0.90 and would invalidate every cached
number mid-write-up.

### D. Do nothing — defensible

The deployment target is **hospital review workstations**, which are overwhelmingly Windows.
"Runs on Mac" is worth having only if a specific reviewer has a Mac. Ask before building it.

---

## If you want cross-platform properly, the real answer is the migration

Everything above is downstream of Python 3.6 + TF 1.15. Migrating the runtime would unlock ARM
Macs, modern Linux, and current Python — but it is a *separate project* with a known, measured
cost:

- The forward pass ports cleanly — an independent numpy reimplementation matched TF to 2.7 × 10⁻⁷.
- **The ICA does not.** MNE 1.12 changes probabilities by up to **0.90** with ICA on.
- So migrating means regenerating every cached probability and re-validating every reported
  number.

Two findings from 2026-08-10 make this *less* frightening than it was, and are worth knowing
before anyone decides:

- **MNE 0.19.2 and 0.20.0 are bit-identical** on this pipeline (25/25 exact).
- **Keras 2.0.8 / TF 1.4.0 and Keras 2.2.5 / TF 1.15 agree to 6.8 × 10⁻⁹.**

The framework is *not* the fragile part — **MNE's ICA is**, and specifically the jump to MNE 1.x.
A migration that pinned an older MNE, or replaced `ica_arti_remove` with a frozen reference
implementation, might well port cleanly. That is the experiment to run before committing to a
migration, and it is a much smaller experiment than the migration itself.

---

## Recommendation

1. **Ship Windows.** It is done, verified, and matches the deployment target.
2. **Add a Linux build only if you want CI**, which is a good reason on its own.
3. **Tell anyone asking about Mac that Apple Silicon needs the runtime migration**, and that the
   migration is scoped, costed and deliberately deferred — not overlooked.
4. **Before any migration**, run the one experiment that matters: does a modern MNE with a frozen
   `ica_arti_remove` reproduce the current probabilities? If yes, cross-platform becomes cheap.
