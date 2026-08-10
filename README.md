# Neureka-2020-Epilepsy-Challenge && Continental generalization of an AI system for clinical seizure recognition

The code implemented for 2020 Neureka-Epilepsy-Challenge paper and Continental generalization of an AI system for clinical seizure recognition.

Seizure Event Detection using minimum electrodes.

[Paper version 1 release here](https://www.researchgate.net/publication/350387463_Two-Channel_Epileptic_Seizure_Detection_with_Blended_Multi-Time_Segments_Electroencephalography_Spectrogram)

Please cite: https://www.sciencedirect.com/science/article/abs/pii/S0957417422012817

"Continental generalization of an AI system for clinical seizure recognition"

## Reviewer GUI (thesis MVP)

A PyQt5 desktop tool for human-in-the-loop EEG review: it runs the pretrained
ConvLSTM detector, shows per-window seizure probability, and lets a reviewer step
through proposed events and accept / reject / edit them, exporting a reviewed
`.csv_bi` plus a `.provenance.json` sidecar.

### Running it on a machine that is not a development machine

Build a standalone Windows application — no Python, no conda, no internet, no
administrator rights:

```
packaging\build_app.bat
```

The result is `dist\SeizureReview\`. Copy that folder to the target PC and
double-click `SeizureReview.exe`. The build refuses to complete unless the
weights hash matches, the tests pass, and the *frozen executable* has been
launched and made to score a real recording.

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for hospital deployment
(antivirus, SmartScreen, where the app writes, performance) and
**[docs/INTENDED_USE.md](docs/INTENDED_USE.md)** for what the software may and
may not be used for.

### Development install

From an **Anaconda Prompt** in this folder:

```
setup.bat
```

Creates the environment and verifies it, printing `SETUP OK`. Takes ~10 min.
See **[INSTALL.md](INSTALL.md)** for setting this up on another PC, what to do
when it fails, and the honest limits of a Python 3.6 / TF 1.15 stack.

Manual equivalent: `conda env create -f environment-seiz36.yml` then
`conda activate seiz36`. No conda? `requirements-seiz36.txt` is the same
package set via pip into a Python 3.6 virtualenv.

**Launch:** double-click **`launch_gui.bat`** — it finds the interpreter itself.

From a terminal, the env must be active first (a bare `python` will be your
system Python and fail on `PyQt5`):

```
conda activate seiz36
python -m gui.main                    # or: python -m gui.main path/to/file.edf
```

On Windows you can instead double-click **`launch_gui.bat`**, which finds the
interpreter itself — it checks `%SEIZ36_PYTHON%`, then an activated
`%CONDA_PREFIX%`, then the usual miniconda/anaconda locations, then `PATH`. Set
`SEIZ36_PYTHON` to override:

```
set SEIZ36_PYTHON=C:\path\to\envs\seiz36\python.exe
```

The pretrained weights (`convlstm_ICA_12_train.h5`, 4.5 MiB) are committed, so a
clone is runnable without any extra download. Its sha256 is pinned in
`eval_config.WEIGHTS_SHA256` and stamped into export provenance.

**Reviewer loop:** open an EDF → wait for inference (cancellable; the first open
also loads the model) → adjust the **threshold** slider → step through events with
**J/K**, **Enter** to jump → **Space** accept, **X** reject, drag a region edge to
edit extent → **Export** writes the reviewed annotations. Files with a cached
`<edf>.probs.npz` load instantly; uncached files run per-window ICA first (slower).

The detector environment is `seiz36` (Python 3.6 / TF 1.15 / PyQt5); see
`environment-seiz36.yml` / `requirements-seiz36.txt`.

**Reproducible evaluation** (no GUI, no TensorFlow — scores the cached probs):

```
# 0. build the evaluation manifest (206 annotated files scorable; 99 unannotated excluded)
python experiments/build_full_manifest.py --out artifacts/zuna_thesis/manifest_full.csv

# score any files that lack a cache — shard across cores, e.g. 8 processes:
#   for i in 0 1 2 3 4 5 6 7; do
#     python precompute_probs.py "sample_data/**/*.edf" --shard $i/8 &
#   done

# all three views: window AUC, event threshold sweep, reviewer-triage simulation
python experiments/evaluate_baseline.py \
  --manifest artifacts/zuna_thesis/manifest_full.csv \
  --name full_scorable --out artifacts/zuna_thesis/baseline_eval/full_scorable.json

# window AUC under the source paper's own protocol (the replication result)
python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest_full.csv

# what the source method's decision stage is worth (4-way ablation)
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest_full.csv

# calibration: reliability curves, ECE with CIs, three post-hoc calibrators
python experiments/calibration.py --manifest artifacts/zuna_thesis/manifest_full.csv \n  --out artifacts/zuna_thesis/baseline_eval/calibration.json \n  --svg artifacts/zuna_thesis/baseline_eval/reliability.svg

# ICA on vs off (needs TensorFlow)     # baseline vs ZUNA, re-scored from caches
python experiments/diag_ica.py         # python experiments/rescore_zuna_compare.py \
                                       #     --dir artifacts/zuna_thesis/compare_first10
```

Use `manifest_full.csv`, not the older 26-file `manifest.csv` — that subset is
seizure-enriched and misleads in both directions (it understates sensitivity and
overstates specificity, because it contains almost no background recording).

The canonical operating point (threshold, stride, ICA, weights, post-processing)
lives in `eval_config.py`; `run_inference.py`, `precompute_probs.py`, the GUI and
the eval scripts all read from it.

### Where the numbers live

**[`docs/RESULTS.md`](docs/RESULTS.md) is the single source of truth for every
quantitative claim.** If a figure anywhere else disagrees with it, that other place
is stale. Supporting documents:

| file | what it is |
|---|---|
| `docs/RESULTS.md` | every current number, with the command that regenerates it |
| `docs/paper_comparison.md` | **side-by-side tables: this work vs the source paper** |
| `docs/reproduction_status.md` | what matches the source paper, what does not, and why |
| `docs/methodology_statements.md` | the claims the thesis will and will not make |
| `docs/deployment_roadmap.md` | path to a clinician-ready prototype; C/C++ scope |
| `docs/progress_2026-08-09_…md` | **latest progress log** — chronology, decisions, dead ends |
| `docs/usability/cognitive_walkthrough_results.md` | first usability results (10 issues, 5 P1) |
| `docs/ica_implementation_review.md` | is the ICA stage implemented correctly? (measured) |
| `experiments/calibration.py` | calibration analysis — reliability, ECE, Platt/isotonic/temperature |
| `docs/progress_2026-05-19_zuna_gui.md` | **historical** — numbers superseded |
| `thesis_report_bundle/` | **archived June 2026 snapshot** — stale code and numbers |

The detector is the 19-channel / 12-second / ICA ConvLSTM of Yang et al.,
*Continental generalization of a human-in-the-loop AI system for clinical seizure
recognition* (Expert Syst. Appl. 207:118083, 2022). Scored under that paper's own
window protocol this reconstruction reaches **AUC 0.89** (95 % CI [0.83, 0.94],
206 annotated files, 27.8 h) against the **0.84** it reports for TUH — statistically
indistinguishable from the published value. Its RPAH figures are not reproducible here (private
clinical data, 20-channel model), and it publishes no TUH sensitivity or
false-alarm rate, so no event-level number here has a published counterpart.

## Preprocessing
Load raw eeg data using STFT
```python
cd utils/
python load_data_elec_3s.py
python load_data_elec_5s.py
python load_data_elec_7s.py
```
Preprocessing the data with ICA
```python
cd utils/
python ICA_load_data_elec.py
```
## Model Training
```python
python main.py --mode=train
```
## Pretrained Model
Conv-LSTM pretrained model:
https://drive.google.com/file/d/1Tj2JZ_B5OqZrVILg15L_lPR2DKYiBoDS/view?usp=sharing
## Post Processing
Get raw results
```python
python main.py --mode=test
```
Get results based on threhold and apply average method.
```python
python main.py --mode=vote
```
Vote and discard short prediction
```python
cd post_process_code/
python overlap.py 
python discard.py 
python clean.py
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Better contact original contributor.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
