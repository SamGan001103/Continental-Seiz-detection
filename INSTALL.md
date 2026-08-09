# Setting this up on another PC

Two routes. Pick by who is doing it.

| | Route A — Miniconda | Route B — frozen `.exe` |
|---|---|---|
| Who | anyone who can double-click and paste one line | anyone at all |
| Time | ~15 min, mostly download | ~2 min |
| Needs internet | yes | no |
| Coding knowledge | none, but you see a terminal | none |
| Status | **works today** | **not built yet** — see §3 |

---

## 1. Route A — Miniconda (works today)

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

## 2. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'PyQt5'` | you ran the system Python, not the env | use `launch_gui.bat`, or `conda activate seiz36` first |
| `conda: command not found` | Miniconda not on PATH | use **Anaconda Prompt**, not plain PowerShell |
| `setup.bat` fails resolving packages | Python 3.6 is end-of-life; some mirrors have dropped it | see §3 — this is the real risk |
| GUI opens but a file takes minutes | that recording has no `.probs.npz` cache | copy the cache alongside the `.edf`, or wait |
| `Could not find the required 19 EEG channels` | the EDF is not a TUSZ 19-channel montage, **or the file path is wrong** | check the path first — a missing file reports this same message |

---

## 3. Honest limitations, and what "portable" really costs

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
- **Route A needs internet and ~2 GB of download.**

### The real fix for a no-knowledge install: freeze it

A **PyInstaller** build would produce a folder you copy to any Windows PC and
double-click — no Python, no conda, no internet. That is the honest answer to
"set it up fast on a fresh PC without coding knowledge".

It is not built yet because it is a day of fiddly work (TF 1.15 hidden imports,
Qt plugin paths, the ~700 MB–1 GB result) and it has to be redone whenever the
code changes. **Worth doing once, immediately before the clinician session** —
not before, or you will rebuild it repeatedly.

### For a demo specifically — don't install anything

The lowest-risk option by far: **bring your own laptop**. A clinician session is
20 minutes; spending the first 15 installing Miniconda on their machine wastes
the scarcest resource in the room. Install on their PC only if they ask to keep
it.
