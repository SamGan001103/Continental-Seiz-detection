# Setting this up on another PC

Two routes. Pick by who is doing it.

| | Route A — frozen `.exe` | Route B — Miniconda |
|---|---|---|
| Who | anyone at all | anyone who can double-click and paste one line |
| Time | ~2 min (copy a folder) | ~15 min, mostly download |
| Needs internet | no | yes |
| Needs admin rights | no | no, but it installs software |
| Coding knowledge | none | none, but you see a terminal |
| Status | **works** | **works** |
| Use it for | hospital PCs, clinician demos | development, running experiments |

**For a hospital PC, use Route A.** Full deployment guidance — antivirus,
SmartScreen, where the app writes, performance expectations — is in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## 1. Route A — the frozen application (no Python needed)

### If someone has already built it for you

1. Copy the whole **`SeizureReview`** folder to the PC. Anywhere the logged-in
   user can write is fine — `Documents` is a good choice. Copy the *entire*
   folder; the `.exe` on its own will not run.
2. Double-click **`SeizureReview.exe`**.
3. On first launch Windows will say *"Windows protected your PC"* because the
   app is not code-signed. Click **More info → Run anyway**. Tell the clinician
   this will happen *before* they see it.

That is the whole install. No Python, no conda, no internet, no admin rights.

### Building it yourself

On a machine that already has the `seiz36` environment (Route B):

```
packaging\build_app.bat
```

~10 minutes. It verifies the model weights hash, runs the test suite, freezes
the app, then launches the frozen executable and scores a real recording before
declaring success. The result is `dist\SeizureReview\` — about 1.2 GB.

Rebuild whenever the code changes; the folder does not update itself.

---

## 2. Route B — Miniconda (for development)

### Step 1 — install Miniconda (once per machine, ~5 min)

Download the Windows 64-bit installer from
<https://www.anaconda.com/download/success> (Miniconda section) and run it.
Accept every default. **Tick "Add Miniconda to PATH"** if offered.

### Step 2 — get the project

Either `git clone` it, or download the repository as a ZIP and extract it.
Everything needed is inside — including the trained model, so there is no
separate download.

### Step 3 — one command

Open **Anaconda Prompt** from the Start menu, `cd` to the project folder, and run:

```
setup.bat
```

That creates the environment and checks it. It takes ~10 minutes and prints
`SETUP OK` when it works.

### Step 4 — run it

Double-click **`launch_gui.bat`**. That's it, from then on.

> The GUI needs EEG files to open. They are **not** included — TUSZ requires its
> own (free) registration at <https://isip.piconepress.com/projects/nedc/>.
> For a demo, copy a folder of `.edf` files plus their `.probs.npz` caches from a
> machine that already has them; cached files open instantly.

---

## 3. If something goes wrong

**Route A (frozen app):** every crash is written to
`%LOCALAPPDATA%\SeizureReview\logs\seizure_review.log`. Start there. Common
cases are in [docs/DEPLOYMENT.md §5](docs/DEPLOYMENT.md).

**Route B (Miniconda):**

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'PyQt5'` | you ran the system Python, not the env | use `launch_gui.bat`, or `conda activate seiz36` first |
| `conda: command not found` | Miniconda not on PATH | use **Anaconda Prompt**, not plain PowerShell |
| `setup.bat` fails resolving packages | Python 3.6 is end-of-life; some mirrors have dropped it | see §3 — this is the real risk |
| GUI opens but a file takes minutes | that recording has no `.probs.npz` cache | copy the cache alongside the `.edf`, or wait |
| `Could not find the required 19 EEG channels` | the EDF is not a TUSZ 19-channel montage, **or the file path is wrong** | check the path first — a missing file reports this same message |

---

## 4. Honest limitations, and what "portable" really costs

**The environment is the fragile part, not the code.** This runs on **Python 3.6
with TensorFlow 1.15**, both long past end-of-life. That is not a preference —
the pretrained weights load under it, and MNE 0.19's ICA is numerically
load-bearing for the operating point (moving to a modern MNE changes individual
window probabilities by up to 0.90). Migrating mid-thesis would mean regenerating
every cached probability and every reported number.

The practical consequences:

- **Python 3.6 packages are increasingly hard to resolve.** `setup.bat` works
  today. It may not in a year, and it will not work on Apple Silicon at all.
- **No Apple Silicon, no ARM.** TF 1.15 has no wheels for either.
- **Route B needs internet and ~2 GB of download.**

**Route A insulates the target machine from all of this** — the frozen folder
carries its own Python 3.6 and TF 1.15, so a hospital PC never has to resolve a
package. The fragility moves to the *build* machine, where it can be dealt with
by someone who knows what a conda channel is. That is the main argument for
freezing, over and above convenience.

What Route A does **not** fix:

- **Not code-signed.** SmartScreen warns on first run; antivirus may quarantine.
  A signing certificate is the fix and the project does not have one.
- **No auto-update.** A new version is a new folder copy.
- **1.2 GB.** Too big to email; use a USB stick or a network share.

### For a clinician demo specifically

**Bring your own laptop.** A clinician session is 20 minutes; spending any of it
on their machine wastes the scarcest resource in the room. Have the frozen
folder on a USB stick in your pocket for the case where they ask to keep it.
