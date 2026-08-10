# Known issues

*Open problems, with what was tried. Kept so nobody repeats the same six attempts.*

---

## 1. The modern stack does not freeze on Windows (open)

**Status: open. Does not affect anything that ships today.**

The Windows application is built from the **Python 3.6 stack** (`seiz36`), which freezes and runs
correctly. Building the *same* application from the modern stack
(`requirements-modern.txt`, Python 3.11 / TF 2.21) fails at run time.

This matters only for the goal of having byte-identical library versions on Windows and macOS.
It does not block the Mac build, and it does not affect the verified Windows build.

### Where it fails

```
File "tensorflow\python\pywrap_tensorflow.py", line 74, in <module>
ImportError: DLL load failed while importing _pywrap_tensorflow_internal:
             A dynamic link library (DLL) initialization routine failed.
Failed to load the native TensorFlow runtime.
```

The freeze itself succeeds; the frozen binary dies loading TensorFlow's native library.

### What was tried, and what each attempt taught

Six builds. The first four failures were real and are **fixed** — they would have hit the macOS
build too, so the work was not wasted:

| # | failure | resolution |
|---|---|---|
| 1 | `_ctypes` → missing `ffi-8.dll` | conda keeps shared libraries in `<env>/Library/bin`, which PyInstaller does not scan. **Fixed** — the spec bundles them. **Reported by scipy as "your scipy install seems to be broken", naming entirely the wrong package.** |
| 2 | weights/montage "missing" | PyInstaller 6 moved bundled data into `dist/SeizureReview/_internal/`. **Fixed** — the smoke test accepts either layout. |
| 3 | `ModuleNotFoundError: matplotlib` | MNE 1.x imports `mne.viz.topomap` from `mne.preprocessing.ica`, so matplotlib is a hard dependency of the ICA stage. The spec excluded it as "GUI never uses it" — true under MNE 0.19, silently false under 1.x. **Fixed** — bundled whenever MNE 1.x is present. |
| 4 | `pyexpat` → missing expat DLL | same root cause as 1. **Fixed.** |
| 5 | TensorFlow native runtime | Introduced by over-correcting: bundling **all 68** conda DLLs shadowed one of TensorFlow's own. |
| 6 | TensorFlow native runtime | Narrowed to the **10** DLLs CPython itself needs. **Still fails.** |

The lesson from 5 → 6: the TF failure was **latent all along**, not caused by the DLL bundling.
It was simply never reached, because attempts 1–4 died earlier. Narrowing the DLL list was the
right change on its own merits and did not fix this.

### What to try next

TensorFlow 2.x with PyInstaller on Windows is known-awkward. In rough order of expected value:

1. **A pip-only virtualenv instead of conda.** Every DLL problem above came from conda's layout.
   `python -m venv` with wheels from PyPI keeps DLLs next to the extension modules, where
   PyInstaller looks.
2. **An older TensorFlow 2.x** — 2.10 was the last Windows-native release before the split to
   `tensorflow-intel`, and is much more widely reported working under PyInstaller.
3. `--collect-all tensorflow` rather than the current selective hidden imports.
4. Check whether `_pywrap_tensorflow_internal.pyd` needs an MSVC redistributable DLL that is
   present on the build machine and absent from the bundle.

### The same stack DOES freeze on macOS arm64 (2026-08-10)

Built on Apple Silicon from `requirements-modern.txt` — the same pins, TF 2.21 — with
**`python -m venv`, not conda**. All four gates pass, and the frozen binary loads TensorFlow and
scores a real recording (`aaaaatao_s003_t000`, 17/17 windows). So *TF 2 + PyInstaller is not
inherently broken*, which narrows the Windows problem to Windows.

Two things came out of it, one of which applies directly to Windows:

**a) A second, independent packaging bug — found only because macOS got far enough to hit it.**
The spec excluded `tensorflow.python.debug` as size reduction "verified unused by the GUI". True
under TF 1.15; under TF 2.x `tf.compat.v1.debugging.experimental` imports it *while `import
tensorflow` is still running*, so the bundled TensorFlow is unimportable:

```
ModuleNotFoundError: No module named 'tensorflow.python.debug'
```

raised from `tensorflow/__init__.py`, not from anything the GUI calls. This is the identical shape
to trap 3 (matplotlib under MNE 1.x): a module the application never references, pulled in by a
dependency whose behaviour changed with its major version, excluded on evidence that expired.
**Fixed** — conditioned on the TF major version, as matplotlib is on the MNE major version.

This is almost certainly *latent on Windows too*, sitting behind the DLL failure exactly as the
DLL failure sat behind traps 1–4. Expect it as the next error there once the native runtime loads.

**b) The venv hypothesis is supported but NOT proven.** macOS never produced a dylib equivalent of
the DLL error at any point, so there was nothing for the venv to fix — this is evidence that a
pip-only environment freezes TF 2 cleanly, not a demonstration that it cures the Windows fault.
Attempt 1 on the list above remains the right next move, now with a working reference build to
compare against.

### Why it is not urgent

The only thing lost is *identical library versions across platforms*. Measured, the practical
difference between the two stacks is a median **0.0001** in `p(seizure)` and **no detection
decision changing** over 74 windows (`docs/portability.md`). Both are correct applications; they
differ less than the ICA differs from itself between runs.

---

## 2. Not code-signed on any platform (open, needs money)

Windows shows SmartScreen; macOS requires right-click → Open on first launch. Both are expected
and documented in `START_HERE.txt`. Removing them needs a purchased certificate — an Apple
Developer account for notarisation, and an EV certificate for SmartScreen.

---

## 3. One corrupt EDF in the corpus (known, benign)

`aaaaaqxr_s003_t000.edf` fails to read: `filesize 47333376 != 56006*1320+8192`. It is excluded
from every evaluation, which is why the scorable set is 305 of 306 cached recordings.

---

## 4. Reported figures are platform-specific (inherent, not fixable)

The ICA does not converge, so a 10⁻¹⁵ difference in linear algebra between two machines produces
a different decomposition. Numbers move in the third decimal between platforms and between
regenerations. **Do not mix figures from two machines in one table**; regenerate the caches in a
single pass on one machine and say which. See `docs/portability.md`.
