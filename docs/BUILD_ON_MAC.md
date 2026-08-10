# Building on the MacBook (Apple Silicon)

Follow this on the Mac. It assumes you copied the `source/` folder off the USB.
Roughly 30–40 minutes, most of it downloads.

---

## 1. Get a modern Python

**Do not try to install Python 3.6.** It cannot run natively on Apple Silicon — CPython added
arm64 support in 3.9.1 and the 3.6 branch ended before that. This is the wall you hit before, and
the way past it is to go *forwards*, not backwards.

Miniconda for Apple Silicon, if you do not already have it:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

Then:

```bash
conda create -n seizmodern python=3.11
conda activate seizmodern
```

## 2. Install the stack

```bash
cd /path/to/source
pip install -r requirements-modern.txt
pip install pyinstaller
```

Check you got arm64 and not Rosetta x86-64 — if this prints `x86_64` you are running an Intel
Python under translation, which will be slow and is not what you want:

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
bash packaging/build_app.sh
```

Four gates, same as Windows: weights hash → tests → freeze → **launch the frozen binary and score
a real recording**. It refuses to finish if any of them fails. The result is
`dist/SeizureReview/`.

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
| **"The scipy install you are using seems to be broken"** | **Do not believe it — scipy is fine.** The real error is above it in the traceback: `_ctypes` failed to load because its `libffi` was not bundled. conda keeps shared libraries somewhere PyInstaller does not scan. The spec already handles the Windows case (`Library/bin`); if it happens on macOS, find the library with `find $CONDA_PREFIX -name "libffi*"` and add it to the `binaries` list in `packaging/SeizureReview.spec`. This cost an hour on Windows by pointing at the wrong package. |
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
