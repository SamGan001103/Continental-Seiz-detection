# Thesis Report Bundle — Seizure-Detection Human-AI EEG Review GUI

> ## ⚠ ARCHIVED SNAPSHOT (June 2026) — STALE CODE AND STALE NUMBERS
>
> This folder is a **frozen copy** made to support the 20% Progress Report. It has not been
> regenerated since, so:
>
> - The `03_code__*.py` files are **out-of-date copies** of the live modules. In particular
>   `03_code__experiments__compare_zuna.py` still contains the false-positive counting bug that
>   charged detection fragments inside an already-detected seizure as false alarms.
> - Every event-level result file here predates the scoring fixes and is **wrong**.
> - Any statement that the work "is not a reproduction of the paper" compared against the wrong
>   paper; the 19-channel detector does replicate the 2022 result (window AUC 0.881 vs 0.84).
>
> **Do not read code or numbers from this folder.** Use the live repository and
> `docs/RESULTS.md`. Regenerate the bundle before the final thesis submission, or delete it.

**Owner:** Sam Gan · **Supervisor:** Prof. Omid Kavehei (University of Sydney)
**Purpose:** a self-contained snapshot of the project — papers, code, results, and the pretrained
model — assembled so it can be uploaded to Claude (or read by a collaborator) to help write the
**20% Progress Report** (and later the full thesis) without digging through the live repo.

> **This is one flat folder — no subfolders.** Every file keeps its original location in its *name*:
> `/` was replaced with `__`. So `04_results__zuna_thesis__compare_first10__zuna_manifest_summary.json`
> originally lived at `artifacts/zuna_thesis/compare_first10/zuna_manifest_summary.json`.
> The numeric prefixes (`01_`, `02_`, …) just keep related files grouped when sorted by name.
>
> Raw EEG data, multi-GB tensors, and a 1.6 GB archive were deliberately **excluded** (see "What's NOT here").

---

## 1. Project in one paragraph

The project builds a **GUI-based Human-AI teaming system for reviewing ambulatory / outpatient EEG**,
using **public datasets only** (TUH/TUSZ — no RPA clinical data). It extends prior NeuroSyd work
(Yikai Yang et al., *"Continental generalization of a human-in-the-loop AI system for clinical seizure
recognition"*, ESWA 2022, and the two-channel SPMB 2020 paper). The existing pipeline is a 19-channel
**ConvLSTM** detector: EDF → resample 250 Hz → 12 s windows / 6 s stride → ICA artifact removal →
STFT → ConvLSTM → per-window seizure probability → thresholded events scored against `.csv_bi`
references. Current work: (a) reconstruct & run that pipeline on public TUH files, (b) rebuild it behind
a reviewer GUI, and (c) pilot whether **ZUNA** (an external diffusion-based EEG signal-enhancement model)
improves detection if run before the detector. The thesis deliverable is the **review interface (MVP)**,
not maximising model recall.

## 2. How to use this bundle with Claude

Select **all files in this folder** and upload them to a Claude Project (since you can't upload folders,
the flat layout is intentional). Then ask for the section you need. Suggested prompts:
- *"Using `02_progress_and_context__progress_2026-05-19_zuna_gui.md` and the `04_results__*` files, draft
  the Progress Summary section of my progress report."*
- *"Using the `03_code__*` files, describe the reconstructed ConvLSTM pipeline for my Methods/Progress section."*
- *"Here are the two NeuroSyd papers (`01_papers__*.pdf`) — help me write the critical Literature Review around them."*

**Integrity note:** the lit review needs *real* citations (the papers here + the curated list Sam already has);
do not invent references. The progress numbers below are the actual measured results — use them as-is.

---

## 3. What's in the folder (by filename prefix)

| Filename prefix | Contents | Use it for |
|---|---|---|
| `01_papers__` | Two two-channel NeuroSyd PDFs (v1 + EEG version) + a text extract of the ZUNA paper | Literature Review grounding; the direct lineage of this project |
| `02_progress_and_context__` | `progress_2026-05-19_zuna_gui.md` (key progress log), `zuna_bridge.md` (integration design), `REPO_README.md` | Progress Summary; how the system works |
| `03_code__` | All project Python: pipeline, GUI, utils, models, post-processing, experiments, tests (50 `.py` files) | Methods / Progress; evidence of software development |
| `04_results__` | All metrics & comparison outputs (JSON/CSV/HTML/SVG), 68 files, no big binaries | Progress Summary results, tables, figures |
| `05_model__` | `convlstm_ICA_12_train.h5` — pretrained 19-channel ConvLSTM weights | Reference only (binary; Claude can't read it) |

### Code highlights (`03_code__*`)
- `run_inference.py`, `diag_sweep.py`, `precompute_probs.py` — reconstructed inference / threshold-sweep scripts.
- `gui__*` — the reviewer GUI: `app.py`, `main.py`, `io__*` (EDF load, `csv_bi` refs, inference, probability cache,
  `zuna.py` bridge), `widgets__*` (signal view, probability strip, event list, channel inspector).
- `models__deep_conv_lstm.py` — the ConvLSTM architecture; `shallow_model.py`, `CNN_grad.py` supporting models.
- `utils__*` — preprocessing, STFT (`pyst.py`), ICA loading, channel ranking, data generation, parameter files.
- `experiments__*` — ZUNA manifest prep & comparison (`compare_zuna_manifest.py`, `prepare_zuna_manifest.py`,
  `run_zuna_vm_batch.sh`).
- `post_process_code__*` — `overlap.py`, `discard.py`, `clean.py` (the paper's vote/discard post-processing).
- `tests__*` — 3 test modules (GUI events, ZUNA integration, compare); the suite passes (11 tests).

---

## 4. Headline results (already measured — safe to cite)

All on **public TUH/TUSZ seizure files**, event-level scoring against `.csv_bi`, 12 s windows, 6 s stride,
ICA on, threshold 0.5 unless noted. "ZUNA" = full diffusion enhancement (`diffusion_steps=50`) run on a GCP GPU VM.

**First-7 pilot** (`04_results__zuna_thesis__compare_first7__*`):

| Metric | Baseline | Full ZUNA |
|---|---:|---:|
| Files | 7 | 7 |
| Reference seizures | 9 | 9 |
| Hits | 5 | 6 |
| Sensitivity | 55.6% | 66.7% |
| False positives | 6 | 4 |

**First-10 pilot** (`04_results__zuna_thesis__compare_first10__zuna_manifest_summary.json`):

| Metric | Baseline | Full ZUNA |
|---|---:|---:|
| Files | 10 | 10 |
| Reference seizures | 19 | 19 |
| Hits | 5 | 6 |
| Sensitivity | 26.3% | 31.6% |
| False positives | 8 | 5 |
| FP / 24 h | 328.7 | 205.4 |

**Threshold sweep, first-10** (`04_results__zuna_thesis__compare_first10__threshold_sweep_first10.json`):

| Threshold | Baseline sens / FP-24h | ZUNA sens / FP-24h |
|---:|---|---|
| 0.50 | 26.3% / 328.7 | 31.6% / 205.4 |
| 0.10 | 26.3% / 451.9 | 47.4% / 575.2 |
| 0.05 | 36.8% / 616.3 | 52.6% / 657.3 |
| 0.01 | 57.9% / 821.7 | 63.2% / 862.8 |

**Runtime / cost:** full ZUNA ran ≈ **6× slower than real time** (first-10: 35 m of EEG → ≈3 h30 m inference),
peak RAM ≈ 42 GiB → too heavy for interactive GUI use; only viable offline/cached.

**Other result sets:** `04_results__zuna_ablation__*` (diffusion steps 5 vs 50), `04_results__zuna_fast__*`
(threshold sweeps 0.1–0.9, ICA vs no-ICA, candidate selection — the now-deprecated "fast selective ZUNA" line),
`04_results__zuna_batch__*` and `04_results__zuna_compare__*` (earlier batch & single-file comparisons).

## 5. Honest framing (important caveats — state these, don't overclaim)

From the progress log's own conclusions:
- These are **pilot comparisons on ≤10 TUH files**, **not** a reproduction of the paper's headline numbers.
- The paper reports TUH mainly as **window-level AUC** + an RPAH clinical workflow with a human arbiter
  (≈76.7% sens / 56.6 FP-24h raw; 92.2% with human-in-the-loop) — *not* the same metric as this event-level pilot.
  So the low pilot baseline sensitivity is **not** a failed reproduction.
- The current GUI baseline omits the paper's PWA/PEI post-processing "lens" and the human arbiter step.
- ZUNA slightly improved sensitivity and reduced false positives on this small pilot — **cannot** be called
  clinically useful from 10 files.
- Known technical risks: per-window ICA is fragile on 12 s windows; `run_inference.py` defaults (step 1,
  thr 0.95) differ from the GUI compare path (step 6, thr 0.5); some ZUNA NPZ are 1–3 s shorter than source.

## 6. What's NOT here (and why)

Excluded to keep the bundle small and uploadable (kept in the live repo at the paths shown):
- **Raw EEG**: `sample_data/` (~2.2 GB of TUH EDF + `.csv_bi`).
- **Large tensors / caches**: `*.fif`, `*.pt`, `*.npz` probability & ZUNA outputs inside `artifacts/`.
- **Archive**: `v2.0.0-...zip` (~1.6 GB).
- **Per-file run/export logs**: the `artifacts/**/logs/` noise (high-level summaries are kept instead).

The pretrained model `convlstm_ICA_12_train.h5` **is** included (`05_model__...`) for completeness, though it's
a binary weight file Claude can't read.

---

*Snapshot assembled 2026-06-02. Source repo branch: `master`. Progress log dated 2026-05-19.*
