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

**Launch** (Windows): double-click `launch_gui.bat`, or from the repo root:

```
C:\Users\User\miniconda3\envs\seiz36\python.exe -m gui.main
# optionally auto-open a file:
C:\Users\User\miniconda3\envs\seiz36\python.exe -m gui.main path\to\file.edf
```

**Reviewer loop:** open an EDF → wait for inference (cancellable; the first open
also loads the model) → adjust the **threshold** slider → step through events with
**J/K**, **Enter** to jump → **Space** accept, **X** reject, drag a region edge to
edit extent → **Export** writes the reviewed annotations. Files with a cached
`<edf>.probs.npz` load instantly; uncached files run per-window ICA first (slower).

The detector environment is `seiz36` (Python 3.6 / TF 1.15 / PyQt5); see
`environment-seiz36.yml` / `requirements-seiz36.txt`.

**Reproducible evaluation** (no GUI, no TensorFlow — scores the cached probs):

```
# all three views: window AUC, event threshold sweep, reviewer-triage simulation
python experiments/evaluate_baseline.py \
  --manifest artifacts/zuna_thesis/manifest.csv \
  --name baseline26 --out artifacts/zuna_thesis/baseline_eval/baseline26.json

# window AUC under the source paper's own protocol (the replication result)
python experiments/replicate_paper_auc.py --manifest artifacts/zuna_thesis/manifest.csv

# what the source method's decision stage is worth (4-way ablation)
python experiments/ablate_postprocessing.py --manifest artifacts/zuna_thesis/manifest.csv

# ICA on vs off (needs TensorFlow)     # baseline vs ZUNA, re-scored from caches
python experiments/diag_ica.py         # python experiments/rescore_zuna_compare.py \
                                       #     --dir artifacts/zuna_thesis/compare_first10
```

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
| `docs/reproduction_status.md` | what matches the source paper, what does not, and why |
| `docs/methodology_statements.md` | the claims the thesis will and will not make |
| `docs/deployment_roadmap.md` | path to a clinician-ready prototype; C/C++ scope |
| `docs/progress_2026-05-19_zuna_gui.md` | **historical** — numbers superseded |
| `thesis_report_bundle/` | **archived June 2026 snapshot** — stale code and numbers |

The detector is the 19-channel / 12-second / ICA ConvLSTM of Yang et al.,
*Continental generalization of a human-in-the-loop AI system for clinical seizure
recognition* (Expert Syst. Appl. 207:118083, 2022). Scored under that paper's own
window protocol this reconstruction reaches **AUC 0.822** against the **0.84** it
reports for TUH — a replication. Its RPAH figures are not reproducible here (private
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
