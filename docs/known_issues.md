# Known issues

*Open problems, with what was tried. Kept so nobody repeats the same six attempts.*

---

## 1. The modern stack does not freeze on Windows — RESOLVED 2026-08-17

**Status: fixed.** It was an import order, not PyInstaller, not conda's DLL
layout, and not bundled-DLL shadowing. Six build attempts were spent on those
three theories.

On Windows with TensorFlow 2, loading Qt's DLLs first makes TensorFlow's native
library fail to initialise. Reversing the order fixes it. Measured from source,
with no PyInstaller involved:

```
import tensorflow  ->  import PyQt5        works
import PyQt5       ->  import tensorflow   DLL load failed
import PyQt5.QtCore only, then tensorflow  DLL load failed
```

Merely loading Qt is enough — not QApplication, not the widgets. The legacy
Python 3.6 stack does not have the conflict (Qt then TF 1.15 loads fine), which
is why the build that shipped all along worked and why this stayed hidden.

**Two fixes were needed, and the first alone was not enough.**

`gui/tf_preload.py`, imported as the first statement of `gui/main.py`, fixes it
from source. It does **not** fix the frozen build: PyInstaller ships
`pyi_rth_pyqt5.py`, a runtime hook that imports PyQt5 to set the Qt plugin path,
and runtime hooks execute *before* the entry script. No ordering inside
`gui/main.py` can be early enough.

`packaging/rthook_tf_before_qt.py` is registered through `runtime_hooks=` in the
spec and runs ahead of the automatic hooks. With it, the modern stack freezes
and scores a real recording on Windows for the first time:

```
self-test: frozen True · python 3.11.9 · 19 ch, 250 Hz, 31.0 s
self-test: windows 4 (4 scored, 0 skipped) · PASS
```

**Why it hid for so long.** The failure flatters the build. The freeze succeeds,
the executable runs, the window opens, all 137 widgets construct, and the GUI
self-test passes. Only scoring fails. Any check that stops at "it launches"
reports success — which is the reason gate 4 insists on scoring a real
recording rather than merely starting the binary.

Both fixes are gated to Windows on Python 3.9+, decided without importing
TensorFlow, so the legacy build keeps deferring it and pays nothing.

The original investigation is kept below, because the three wrong theories each
produced a real fix that the macOS and Linux builds needed.

---

### Original report (the six attempts)

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

### Why it is not urgent — and what that judgement now rests on

The only thing lost is *identical library versions across platforms*.

This section previously argued the point was academic, citing a median **0.0001** in `p(seizure)`
and **no detection decision changing** over 74 windows. Measured over 3 870 windows from 111
recordings, the median holds at 0.00006 but the rest does not: 1.5 % of windows change decision,
the maximum is 0.788 against the 0.136 quoted, and **41 % of proposed events present on Windows
are absent on Linux** (`docs/portability.md`).

That does **not** make the Windows freeze urgent, and it is worth being precise about why. Closing
this issue would give Windows the *modern* stack — the same one macOS and Linux run — which would
put all three platforms on one side of the divide. But the divide measured above is between the
**legacy** Windows build that ships today and the modern stack, and the shipped Windows build is
the one every reported figure was produced on. Fixing this would not remove the disagreement; it
would move Windows to the other side of it and invalidate the caches.

So the priority is unchanged, but the reasoning is inverted: this is not urgent because the
difference is small — it is not small — but because the shipped application is self-consistent and
the handling rule (regenerate caches per machine, never mix figures) covers it.

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

The ICA does not converge — verified on real data, `n_iter_` reaches `max_iter` = 200 on every
window — so a 10⁻¹⁵ difference in linear algebra between two machines produces a different
decomposition.

**"Third decimal" was wrong.** That was measured over 74 windows and describes only the bulk of
the distribution. Over 3 870 windows the median is 0.00006 but p99 is 0.30 and the maximum 0.79,
and at the level a reviewer actually sees, 41 % of proposed events differ between two correct
builds. See `docs/portability.md` for the distribution and for why neither machine is "right".

**Do not mix figures from two machines in one table.** Previously advice about precision, this is
now a correctness requirement: such a table can disagree with itself about whether a seizure was
proposed at all. Regenerate the caches in a single pass on one machine and say which.
