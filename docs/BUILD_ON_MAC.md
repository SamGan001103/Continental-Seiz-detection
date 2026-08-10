# Building on the MacBook (Apple Silicon)

Follow this on the Mac. It assumes you copied the `source/` folder off the USB.
Roughly 30–40 minutes, most of it downloads.

---

## 1. Get a modern Python — a venv, not conda

**Do not try to install Python 3.6.** It cannot run natively on Apple Silicon — CPython added
arm64 support in 3.9.1 and the 3.6 branch ended before that. This is the wall you hit before, and
the way past it is to go *forwards*, not backwards.

**Use `python -m venv`, not conda.** Every DLL problem in the Windows modern-stack build came from
conda's layout (`known_issues.md` §1), and `BUILD_ON_LINUX.md` already takes the venv route for the
same reason. The verified macOS build was made this way.

**Check the interpreter is arm64 before anything else.** An existing Intel Anaconda is the trap
here: it runs happily under Rosetta and `conda create` inherits its architecture silently, so the
environment is x86_64 and nothing complains until the numbers are slow and the binary is wrong.
Ask the *interpreter you are about to use*, not the one on `PATH`:

```bash
/path/to/python3 -c "import platform; print(platform.machine())"     # must print: arm64
```

Any arm64 CPython ≥ 3.9 works. The build below was verified on the python.org 3.12 framework
build; `portability.md` records 3.11 as the validated reference. Then:

```bash
/path/to/arm64/python3 -m venv ~/venvs/seizmodern
source ~/venvs/seizmodern/bin/activate
```

## 2. Install the stack

```bash
cd /path/to/source
pip install -r requirements-modern.txt
pip install pyinstaller==6.22.0
```

Confirm the environment itself is arm64 — if this prints `x86_64` you are under translation, and
the build will be an Intel binary:

```bash
python -c "import platform; print(platform.machine())"     # expect: arm64
```

## 3. Confirm it works before building anything

```bash
python -m unittest discover -s tests -q
```

176 tests. Then check the model loads:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from gui.io.infer import _build_model
print('parameters:', _build_model().count_params())     # expect 384846
"
```

384,846 is the number to see. It means the released weights loaded into the `tf.keras` graph.

Then run the GUI itself:

```bash
python -m gui.main --gui-self-test      # builds every widget offscreen, exits
python -m gui.main                      # the real thing
```

## 4. Build the app

```bash
SEIZ_PYTHON=~/venvs/seizmodern/bin/python bash packaging/build_app.sh
```

`SEIZ_PYTHON` matters if the venv is not the activated environment — the script otherwise takes
whatever `python` resolves to, which on a machine with Anaconda installed is usually the Intel one.

Four gates, same as Windows: weights hash → tests → freeze → **launch the frozen binary and score
a real recording**. It refuses to finish if any of them fails. The result is
`dist/SeizureReview/`.

**Gate 4 needs a recording, and skips itself without one.** `smoke_test.py` looks for the first
`manifest_full.csv` entry that exists on disk, then falls back to walking `sample_data/`. If it
finds neither it prints `no local EDF found`, checks only the bundle layout, and still exits 0 —
so the build reports success having never loaded TensorFlow, which is the one thing gate 4 exists
to prove. Point `sample_data/` at the corpus before building; a symlink is enough:

```bash
ln -sfn /path/to/tuh_dataset sample_data
```

Prefer a short recording: the whole corpus resolves too, but the first manifest row is 3337 s
(~278 windows, roughly 21 minutes at ~4.6 s/window) and `smoke_test.py` times out at 900 s.

## 5. Put it on the USB

```bash
python packaging/make_usb.py --dest /Volumes/YOUR_USB/SeizureReview
```

Additive — it fills the `macos/` slot and leaves the Windows build alone.

---

## Expected problems, and what they mean

| symptom | cause and fix |
|---|---|
| `platform.machine()` prints `x86_64` | you are under Rosetta. Reinstall Miniconda from the **arm64** installer. |
| PyInstaller: "Hidden import not found" for keras/sklearn names | expected and harmless — the spec declares TF1-era modules only when they exist. |
| App refuses to open: "cannot be opened because the developer cannot be verified" | **right-click → Open**, not double-click. Un-notarised apps give a dead end on double-click. |
| Still refuses after right-click → Open | `xattr -dr com.apple.quarantine /path/to/SeizureReview` — anything from a USB is quarantined. |
| Qt complains about a platform plugin | `export QT_QPA_PLATFORM=cocoa`, or run the GUI self-test first to isolate it. |
| `stft` import fails on numpy | `requirements-modern.txt` pins `numpy<2` because `stft` 0.5.2 uses `np.lib.pad`, removed in numpy 2.0. |
| **Any `DLL load failed` / `ImportError` naming a stdlib module** (`_ctypes`, `pyexpat`, `_ssl`, `_sqlite3`) | conda keeps shared libraries where PyInstaller does not look. On Windows the spec now bundles **every** DLL from `<env>/Library/bin` — about 19 MB, which buys out the whole class of failure rather than discovering the next missing one when a clinician opens a recording. macOS conda uses `<env>/lib/*.dylib`, and PyInstaller resolves those through `otool`, so it usually does not arise. If it does, add the directory to `binaries` in the spec the same way. |
| **"The scipy install you are using seems to be broken"** | **Do not believe it — scipy is fine.** The real error is above it in the traceback: `_ctypes` failed to load because its `libffi` was not bundled. conda keeps shared libraries somewhere PyInstaller does not scan. The spec already handles the Windows case (`Library/bin`); if it happens on macOS, find the library with `find $CONDA_PREFIX -name "libffi*"` and add it to the `binaries` list in `packaging/SeizureReview.spec`. This cost an hour on Windows by pointing at the wrong package. |
| `ModuleNotFoundError: No module named 'matplotlib'` at run time | MNE 1.x imports `mne.viz.topomap` from `mne.preprocessing.ica`, so matplotlib is a hard dependency of the ICA stage — unlike MNE 0.19. It is in `requirements-modern.txt` and the spec bundles it whenever MNE 1.x is installed. If you see this, matplotlib is missing from the environment. |
| `ModuleNotFoundError: No module named 'tensorflow.python.debug'`, raised from `tensorflow/__init__.py` in the frozen app | The spec used to exclude that module as dead weight, which is true under TF 1.15 and false under TF 2.x — `tf.compat.v1.debugging.experimental` imports it *during* `import tensorflow`, so the bundled TensorFlow cannot be imported at all. **Fixed** — the exclusion is now conditional on the TF major version. The spec prints which way it resolved. |
| Bundled files "missing" but the build succeeded | PyInstaller 6 puts data in `dist/SeizureReview/_internal/` rather than beside the executable. The smoke test accepts either layout. Copy the **whole** `SeizureReview` folder — the `_internal` directory is part of the application. |

## What will differ from the Windows build, and why

Scores will differ slightly — median about **0.0001**, worst case seen **0.136** over 74 windows,
with **no detection decision changing**.

This is expected and cannot be removed. FastICA does not converge on this data, so a 10⁻¹⁵
difference in linear algebra between two platforms compounds over 200 iterations into a different
decomposition. `docs/portability.md` has the measurements.

**Consequence for the thesis: do not mix figures produced on two machines in one table.** Pick one
platform for reported numbers — the Windows build and its caches are the verified set — and treat
the Mac as a development and demonstration machine.

## If you need the numbers on the Mac too

Regenerate the caches there rather than copying them:

```bash
python precompute_probs.py "sample_data/**/*.edf" --overwrite --shard 0/8   # ×8, in parallel
python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest_full.csv
```

About 70 minutes. State which machine produced any figure you report.
