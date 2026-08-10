# Progress — 2026-08-10 — packaging the GUI for hospital deployment

**Goal for this session:** make the review GUI something a clinician can run on
a fresh Windows PC with no Python, no conda, no internet, and no administrator
rights.

**Outcome:** done. `packaging\build_app.bat` produces `dist\SeizureReview\`, a
1.2 GB folder that is copied to the target machine and double-clicked. The
frozen executable has been verified to build its main window *and* to run a
real recording end-to-end through EDF reading, ICA, STFT and TensorFlow.

---

## 1. The architectural decision, and why

The first question was whether to build a desktop application or a server/web
service. Recorded in full in [DEPLOYMENT.md §1](DEPLOYMENT.md); the short form:

**Desktop, because the binding constraint is governance, not engineering.**
A server holding patient EEG requires a data-transfer justification and a
hospital infosec review — months of calendar time on the thesis's critical
path. A folder on a USB stick requires a conversation. The desktop build also
preserves the strongest privacy claim available: the recording never leaves the
machine it was opened on.

The one genuine advantage of a server — central updates and a GPU — was not
worth that cost for a single reviewer at a single workstation.

---

## 2. What had to be fixed before freezing was possible

Freezing exposed a class of bug that is invisible when running from the
repository, because the repository *is* the working directory. Each of these
would have failed only on the target machine.

### 2.1 The working directory was load-bearing

`gui/io/edf.py`, `gui/io/infer.py` and `run_inference.py` each did
`os.chdir(REPO/utils)` before loading, because `pyst.nedc_load_parameters()`
opened a bare relative filename. A frozen build has no `utils/` to change into.

Fixed by making `pyst` resolve a bare parameter filename against its own
package directory, and deleting all three `chdir` calls.

A subtlety worth recording: the obvious implementation,
`os.path.dirname(os.path.abspath(__file__))` *inside* the function, is wrong.
In Python 3.6 `__file__` may be relative, so `abspath` resolves it against
whatever the current directory happens to be — which, in code whose entire
purpose is to survive a `chdir`, is exactly the wrong answer. It is now
captured once at import time into `_PKG_DIR`. This was caught by a test, not by
reading the code.

The comment on the `infer.py` chdir claimed "keras import paths expect this
cwd". That was untrue — `models/` sits at the repository root and resolves
through `sys.path`. The chdir was doing nothing but mutating global state.

### 2.2 Resources were located by walking up from `__file__`

Added `gui/paths.py`. `resource()` resolves against `sys._MEIPASS` when frozen
and the repository root otherwise; `weights_path()` and `params_path()` go
through it.

`writable_root()` is deliberately *not* the bundle, for two independent
reasons. In a one-file build `sys._MEIPASS` is a temporary directory deleted
when the process exits, so a review saved there is destroyed on close. In the
one-folder build actually shipped it is the application directory — which on a
clinical workstation is frequently a read-only share. Either way the bundle is
wrong, so `writable_root()` resolves to `%LOCALAPPDATA%\SeizureReview\`
regardless of build shape, falling back through `%APPDATA%`, the home
directory and the temp directory if that cannot be created.

Every candidate stays *inside* a `SeizureReview\` directory. A first version
fell back to the home directory itself, which would have scattered bare
`logs\`, `cache\` and `autosave\` folders into the user's home — litter with
names generic enough that nobody could tell what had created them.

### 2.3 The probability cache assumed a writable recording folder

`save_probs()` wrote a `.probs.npz` sidecar next to the EDF, with no fallback.
On a clinical workstation recordings normally live on a read-only network share
or a PACS export directory, so the write would have raised — **discarding a
multi-minute inference run** at the moment it completed. That is the single most
infuriating failure available in this application.

It now falls back to `%LOCALAPPDATA%\SeizureReview\cache\`, keyed by a digest of
the EDF's absolute path so identically-named recordings in different folders do
not collide. Reads check both locations. The sidecar is still preferred when it
works, because it travels with the recording.

### 2.4 The autosave had the same flaw, and it was worse

`_autosave_review()` wrote a crash-recovery JSON beside the EDF and swallowed
`OSError`. On a read-only share it therefore did **nothing, silently**, while
the "unexported review" dialog went on telling the reviewer their decisions
were safe in a file that did not exist.

Same fallback as the cache, plus the application now tracks *where the autosave
actually landed* and the dialog names that path — or warns explicitly that the
decisions exist only in the open window. Claiming safety that does not exist is
worse than claiming nothing.

### 2.5 Provenance silently disappeared when frozen

`_git_commit()` shells out to `git` in the repository. A frozen build has no
repository and a hospital PC has no git, so it returned `None` — stripping the
build identifier from exactly the annotations most likely to be relied on. The
commit is now stamped into `build_info.json` at build time by the spec and read
back from the bundle. `_weights_sha256()` had the same shape and now goes
through `gui.paths.weights_path()`.

### 2.6 The file dialog opened inside the bundle

`_open_edf_dialog()` started at `REPO/sample_data`, which when frozen is
PyInstaller's own extraction directory — showing the reviewer the application's
internals. It now prefers the last folder they opened, then `Documents`.

### 2.7 A research-only feature was still on the clinical toolbar

The toolbar carried a **Run full ZUNA** button and an *AI source* selector. A
full-ZUNA run needs a second Python interpreter with a modern stack,
`utils/zuna_bridge.py`, and a writable `artifacts/` tree — **none of which
exist in the frozen build**, which ships one interpreter and may sit on a
read-only share. A clinician pressing that button would have got a subprocess
error naming a script they do not have.

`gui.io.zuna.zuna_available()` now gates both controls, and they are not put on
the toolbar at all when a run is impossible. This also closes walkthrough item
**U-11**, whose recommendation was to "consider hiding the selector for
clinician sessions" — ZUNA is a documented *negative* result and has no place
in a clinical deployment regardless of whether it could run.

The widgets are still constructed, so every code path that references them
stays simple; they are simply never shown. Four tests pin both directions, so
the research build keeps the capability it was written for.

### 2.8 There was nowhere for a crash to go

A windowed build has no console; a traceback printed to stdout vanishes.
`gui/main.py` now installs an exception hook that writes to
`%LOCALAPPDATA%\SeizureReview\logs\seizure_review.log` and shows a dialog
containing the log path, so a user can read it out over the phone.

The hook itself has to survive a windowed build, where `sys.stderr` is `None` —
writing to it would have raised *inside the handler whose job is to survive a
crash*, producing neither a dialog nor a log entry.

---

## 3. How the build is verified

A PyInstaller build routinely compiles cleanly and then dies on the target
machine at the first module that was imported inside a function. Building is
therefore not evidence of anything; `packaging/build_app.bat` refuses to declare
success until the frozen executable has actually run.

1. **Weights hash** — checked against `eval_config.WEIGHTS_SHA256` *before*
   building. Fatal, not a warning: a build carrying the wrong weights is
   indistinguishable from a correct one at run time.
2. **Test suite** — 130 tests. A release is not built from a failing tree.
3. **`SeizureReview.exe --gui-self-test`** — constructs the real `MainWindow`
   offscreen and pumps the event loop, catching a missing PyQt5 or pyqtgraph
   submodule.
4. **`SeizureReview.exe --self-test <edf>`** — reads a real recording and scores
   two minutes of it, exercising the montage loader, MNE's ICA, the STFT helper
   and TensorFlow.

Result of step 4 on the first build:

```
self-test: frozen      True
self-test: signal      19 ch, 250 Hz, 3337.0 s
self-test: windows     19 (12 scored, 7 skipped)
self-test: scores      min 0.0001  max 0.9695  mean 0.3216
self-test: PASS
```

Twelve of nineteen windows scored, seven refused by the ICA/interrupted-signal
guards — consistent with the ICA non-convergence rate measured on this corpus,
so the frozen build is behaving like the source build rather than merely
starting.

---

## 4. Deliberate build choices

Recorded because each looks like an obvious thing to "optimise" later, and each
would make deployment worse.

- **One-folder, not one-file.** A one-file build unpacks 1.2 GB into `%TEMP%` on
  *every* launch, and a large self-extracting executable is precisely the shape
  enterprise endpoint protection blocks.
- **UPX disabled.** UPX-packed executables are heuristically classified as
  packed malware by most enterprise antivirus. Saving disk is not worth failing
  the scan on the one machine that matters.
- **`matplotlib`, `pandas`, `tkinter` excluded.** Verified unused by the GUI
  (`matplotlib` is used only by `experiments/thesis_figures.py`).

---

## 5. Making the limitations reachable

`docs/INTENDED_USE.md` is bundled into the application and wired to
**Help → Intended use and limitations…**. A limitations document that exists
only in the repository is not available to the person who needs it — a
clinician on an offline workstation. The menu item falls back to the repository
copy when running from source, and says plainly that the installation is
incomplete if neither is present.

The first implementation handed the file to `QDesktopServices.openUrl()`, which
was a defect: **a fresh Windows install has no default handler for `.md`**, so
on a hospital PC the menu item would either do nothing or raise a "how do you
want to open this file?" prompt — on the one machine where the document most
needs to be readable, in front of an offline user with no way to fix the
association. It now renders in-window through `QTextBrowser.setMarkdown()`
(Qt 5.15), falling back to plain text on older Qt rather than displaying raw
markup as broken HTML.

This is the general shape of the whole session: the failure was not in the
logic, it was in an assumption about the environment that holds on a
development machine and does not hold on the target.

---

## 6. Documentation written

- **`docs/DEPLOYMENT.md`** — the desktop-vs-server decision, how to build, how
  to install on a hospital PC, SmartScreen and antivirus expectations, where the
  app writes, performance expectations, troubleshooting, known gaps.
- **`docs/INTENDED_USE.md`** — what the software is and is not, intended users
  and data, measured performance stated plainly (49.4 % sensitivity, 204 FA/24 h,
  and that 19 of 85 seizures produce no model response at all), populations not
  evaluated, failure modes, provenance, privacy. Written for the clinician,
  the supervisor and an ethics reviewer. Bundled into the application.
- **`INSTALL.md`** — restructured so the frozen build is Route A and Miniconda
  is Route B (development only).

Two corrections made while writing, both about denominators. A first draft said
"22 % of *missed* seizures produce no model response"; recomputed from the
caches it is 22 % of *all* 85 reference seizures — 19 of them. The replacement
then said those 19 were "over half of everything the detector misses", which
was also wrong: that used the 36 seizures whose *peak window score* falls below
0.5, while the reader of that section has just been told the detector "finds 42
of 85", i.e. misses **43**. The correct figure against the event-level
denominator the surrounding text establishes is **44 %**. Stating a proportion
without naming its denominator is how both errors happened. Corrected
before it went anywhere.

---

## 7. What is not done

- **Code signing.** SmartScreen will warn on first launch and antivirus may
  quarantine the folder. Needs a certificate the project does not have. This is
  the largest remaining friction for a real hospital install.
- **An installer.** Copying a folder is fine and needs no admin rights; an MSI
  would be nicer and would need one.
- **Auto-update.** A new version is a new folder copy.
- **A GPU build.** CPU-only. Inference runs at ~0.043× real time (a 60-minute
  recording scores in ~2.5 min on the development machine), dominated by
  per-window ICA. `precompute_probs.py` remains the answer for a planned review
  session, and the timing should be re-measured on the actual target PC.
- **The bundle is bigger than it needs to be.** `collect_submodules('mne')`
  drags in `mne.simulation.tests`, `mne.commands` and similar. Pruning them
  would cut both size and build time, but every exclusion is a chance to break
  ICA on the target machine only, so it is not worth doing until the build is
  otherwise stable.

---

## 8. Files changed

```
new   gui/paths.py                       resource resolution, frozen and source
new   packaging/SeizureReview.spec       the build definition + build stamp
new   packaging/build_app.bat            verify -> test -> freeze -> smoke test
new   packaging/smoke_test.py            runs the frozen exe before shipping it
new   tests/test_deployment_paths.py     20 tests for target-machine failures
new   tests/test_doc_numbers.py          16 tests pinning the shipped docs to
                                         RESULTS.md, and the derived percentages
                                         to the counts they come from
new   docs/DEPLOYMENT.md
new   docs/INTENDED_USE.md
new   docs/progress_2026-08-10_packaging.md

mod   utils/pyst.py                      cwd-independent parameter resolution
mod   gui/io/edf.py                      chdir removed
mod   gui/io/infer.py                    chdir removed, weights via gui.paths
mod   gui/io/cache.py                    read-only-share fallback
mod   gui/main.py                        crash log, excepthook, self-tests, HiDPI
mod   gui/io/zuna.py                     zuna_available() gate
mod   gui/app.py                         autosave fallback, frozen provenance,
                                         file-dialog start directory,
                                         Help > intended use (rendered in-app)
mod   docs/usability/cognitive_walkthrough_results.md
                                         3d: U-10 closed after the 3c measurement
mod   run_inference.py                   chdir removed
mod   tests/test_review_guards.py        +3 autosave-fallback cases
mod   README.md, INSTALL.md              frozen build promoted to Route A
```

Tests: 91 → 130, all passing.
