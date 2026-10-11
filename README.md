# Clinician-in-the-Loop Seizure Detection System

> **Research prototype. NOT for diagnostic or clinical use.** Every detection
> this system produces is a *candidate* for a clinician to confirm or reject,
> never a finding, and no threshold in it has been clinically validated.

A neurologist reviewing continuous EEG has to find events lasting seconds in
hours of recording. Automated detection can propose where to look, but is not
reliable enough to be trusted unsupervised. This system is built around that
limitation rather than against it: **the detector proposes, the clinician
decides**, and the record shows which decisions were whose.

It is a BMET4111 honours thesis project (University of Sydney, supervised by
Prof. Omid Kavehei), extending the lab's published detector — see
[Upstream work](#upstream-work). **Inference only: nothing here is trained.**

---

## The system is in two parts

```
   ┌─────────────────────── PART 1 ── hospital compute server ───┐
   │  patient recording (EDF)                                    │
   │           │                                                 │
   │           ├── ConvLSTM  ────────► seizure probabilities     │
   │           └── SpikeNet1 ────────► interictal discharges     │
   │                                                             │
   │          server/prepare_study.py  ─────►  STUDY BUNDLE      │
   └─────────────────────────────────────────────┬───────────────┘
                                                 │  copied
          ┌──────────────────────────────────────┴───────────┐
          ▼                     ▼                            ▼
   ┌─── PART 2 ─────┐   ┌─── PART 2 ─────┐   ┌─── PART 2 ─────┐
   │ review app     │   │ review app     │   │ review app     │
   │ clinical PC    │   │ clinical PC    │   │ clinical PC    │
   │ runs NO model  │   │ runs NO model  │   │ runs NO model  │
   └────────────────┘   └────────────────┘   └────────────────┘
```

**Part 1 runs every model**, on the machine that already holds the patient
records and the computing power. **Part 2 reads what Part 1 produced** and
scores nothing — a clinical workstation typically has no GPU and often no
TensorFlow, and a job measured in hours must not begin while a reviewer waits
for a file to open. The split is enforced in code, not merely intended: the
review application has no inference path, and a test asserts it stays that way.

A **study bundle** is a recording plus:

| file | what it holds |
|---|---|
| `<stem>.probs.npz` | per-window seizure probability, from the ConvLSTM |
| `<stem>.ied.csv` | interictal discharge candidates, from SpikeNet1 |
| `<stem>.study.json` | the manifest — what is here, what built it, and which display settings it is valid for |

The manifest records the recording's size, mtime and digest, so a bundle
cannot be read against an EEG it was not built from.

---

## Quick start

### Part 1 — prepare a study (server)

```
python server/prepare_study.py "D:/studies/*.edf"
```

Drives both detectors and writes the bundle beside each recording. Copy each
recording **with its bundle** to the review workstations.

### Part 2 — review it (clinical workstation)

Double-click **`launch_gui.bat`**, or:

```
conda activate seiz36
python -m gui.main path/to/recording.edf
```

An unprepared recording still opens — a reviewer may want to read one — but
with no proposals, and the status bar names the command that prepares it.

**Reviewer loop:** open an EDF → adjust the **threshold** slider → step through
events with **J/K**, **Enter** to jump → **Space** accept, **X** reject, **D**
mark an interictal discharge, drag a region edge to edit its extent →
**Export** writes the reviewed annotations plus a provenance sidecar.

### Build a standalone application

```
packaging\build_app.bat
```

Produces `dist\SeizureReview\` — no Python, no conda, no internet, no admin
rights. The build refuses to complete unless the weights hash matches, the
tests pass, and the frozen executable has been launched and made to score a
real recording. See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for hospital
deployment and **[docs/INTENDED_USE.md](docs/INTENDED_USE.md)** for what the
software may and may not be used for.

### Development install

From an **Anaconda Prompt** in this folder: `setup.bat` (~10 min, prints
`SETUP OK`). See **[INSTALL.md](INSTALL.md)** for other machines and for the
honest limits of a Python 3.6 / TF 1.15 stack. Manual equivalent:
`conda env create -f environment-seiz36.yml`.

The pretrained weights (`convlstm_ICA_12_train.h5`, 4.5 MiB) are committed, so
a clone is runnable without any extra download. Their sha256 is pinned in
`eval_config.WEIGHTS_SHA256` and stamped into export provenance.

---

## Repository map

| path | what it is |
|---|---|
| `server/` | **Part 1** — everything that runs on the compute machine |
| `server/prepare_study.py` | the Part 1 entry point — prepares a study bundle |
| `server/precompute_probs.py`, `server/precompute_ieds.py` | the individual detector passes it drives |
| `gui/` | **Part 2** — the review application |
| `gui/io/` | readers and writers: EDF, caches, annotations, detectors |
| `eval_config.py` | the canonical operating point (threshold, stride, ICA, weights) — every script reads it |
| `experiments/` | evaluation and comparison scripts; not part of the shipped system |
| `tests/` | 727 tests; `python -m unittest discover -s tests` |
| `docs/` | results, deployment, intended use, progress logs |
| `packaging/` | frozen-build scripts |
| `utils/`, `models/`, `main.py`, `post_process_code/`, `txt_file/` | **upstream** — the original project's training and preprocessing code, unmodified |
| `artifacts/` | evaluation outputs (large runs are gitignored) |
| `thesis_report_bundle/` | archived June 2026 snapshot — **stale code and numbers** |

---

## Where the numbers live

**[`docs/RESULTS.md`](docs/RESULTS.md) is the single source of truth for every
quantitative claim.** If a figure anywhere else disagrees with it, that other
place is stale.

| file | what it is |
|---|---|
| `docs/RESULTS.md` | every current number, with the command that regenerates it |
| `docs/paper_comparison.md` | side-by-side tables: this work vs the source paper |
| `docs/reproduction_status.md` | what matches the source paper, what does not, and why |
| `docs/methodology_statements.md` | the claims the thesis will and will not make |
| `docs/deployment_roadmap.md` | path to a clinician-ready prototype |
| `docs/usability/cognitive_walkthrough_results.md` | first usability results (10 issues, 5 P1) |
| `docs/ica_implementation_review.md` | is the ICA stage implemented correctly? (measured) |

**Reproducible evaluation** — no GUI and no TensorFlow, scored from the cached
probabilities:

```
# build the evaluation manifest (206 annotated files scorable)
python experiments/build_full_manifest.py --out artifacts/zuna_thesis/manifest_full.csv

# window AUC, event threshold sweep, reviewer-triage simulation
python experiments/evaluate_baseline.py \
  --manifest artifacts/zuna_thesis/manifest_full.csv \
  --name full_scorable --out artifacts/zuna_thesis/baseline_eval/full_scorable.json

# window AUC under the source paper's own protocol (the replication result)
python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest_full.csv

# what the source method's decision stage is worth (4-way ablation)
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest_full.csv
```

Use `manifest_full.csv`, not the older 26-file `manifest.csv` — that subset is
seizure-enriched and misleads in both directions.

---

## Upstream work

The detector is the 19-channel / 12-second / ICA ConvLSTM of Yang et al.,
*Continental generalization of a human-in-the-loop AI system for clinical
seizure recognition*, **Expert Systems with Applications 207:118083 (2022)**.

Please cite: <https://www.sciencedirect.com/science/article/abs/pii/S0957417422012817>
· [paper v1 on ResearchGate](https://www.researchgate.net/publication/350387463_Two-Channel_Epileptic_Seizure_Detection_with_Blended_Multi-Time_Segments_Electroencephalography_Spectrogram)

Scored under that paper's own window protocol, this reconstruction reaches
**AUC 0.89** (95 % CI [0.83, 0.94], 206 annotated files, 27.8 h) against the
**0.84** it reports for TUH — statistically indistinguishable from the
published value. Its RPAH figures are not reproducible here (private clinical
data, 20-channel model), and it publishes no TUH sensitivity or false-alarm
rate, so no event-level number here has a published counterpart.

Interictal discharge detection uses **SpikeNet1** — Jing J, Sun H, … Westover
MB, *Development of Expert-Level Automated Detection of Epileptiform
Discharges During Electroencephalogram Interpretation*, JAMA Neurol
2020;77(1):103-108. Its weights are **not** vendored; run
`python experiments/fetch_spikenet.py` to download them. The upstream licence
is self-contradictory (CC BY-NC in `LICENSE.txt`, CC BY-ND in `README.md`) —
see `models/spikenet1/NOTICE.txt`.

### The original project's own instructions

This repository began as
*Neureka-2020-Epilepsy-Challenge && Continental generalization of an AI system
for clinical seizure recognition*. That code is retained unmodified in `utils/`,
`models/`, `main.py`, `post_process_code/` and `txt_file/`, and its workflow is:

```
# preprocess: STFT
cd utils/ && python load_data_elec_3s.py   # also 5s, 7s
# preprocess: ICA
cd utils/ && python ICA_load_data_elec.py
# train / test / vote
python main.py --mode=train
python main.py --mode=test
python main.py --mode=vote
# post-process
cd post_process_code/ && python overlap.py && python discard.py && python clean.py
```

Conv-LSTM pretrained model:
<https://drive.google.com/file/d/1Tj2JZ_B5OqZrVILg15L_lPR2DKYiBoDS/view?usp=sharing>

**This thesis trains nothing.** The training path above is upstream's and is
kept for reference and reproducibility only.

---

## Contributing

Pull requests welcome. For major changes please open an issue first — and for
anything touching the upstream detector, contact the original contributors.
Please keep the tests passing.

## License

[MIT](https://choosealicense.com/licenses/mit/)
