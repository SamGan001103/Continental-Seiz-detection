# ZUNA Bridge

ZUNA is a reconstruction/superresolution step, not a seizure detector. In this
repo it should run offline before the existing ConvLSTM detector, then the GUI
can review detector probabilities and clearly mark any imputed signals.

## Environment

The existing detector environment is `seiz36` with Python 3.6, Keras 2.2, and
TensorFlow 1.15. ZUNA requires a modern Python stack, so keep it separate:

```powershell
cd C:\Users\User\Continental-Seiz-detection
C:\Users\User\miniconda3\Scripts\conda.exe create -n zuna-eeg python=3.10 -y
C:\Users\User\miniconda3\Scripts\conda.exe run -n zuna-eeg pip install zuna mne
```

## Prepare EDF For ZUNA

ZUNA expects MNE `.fif` input with valid 3D electrode positions. The bridge
renames TUH/legacy labels such as `T3/T4/T5/T6` to MNE's modern
`T7/T8/P7/P8` names and applies `standard_1005` coordinates.

```powershell
C:\Users\User\miniconda3\Scripts\conda.exe run -n zuna-eeg python utils\zuna_bridge.py prepare `
  --edf sample_data\v2.0.0\edf\eval\aaaaaaaq\s006_2014_08_18\01_tcp_ar\aaaaaaaq_s006_t000.edf `
  --out artifacts\zuna\fif\aaaaaaaq_s006_t000.fif `
  --overwrite
```

## Run ZUNA

```powershell
C:\Users\User\miniconda3\Scripts\conda.exe run -n zuna-eeg python utils\zuna_bridge.py run `
  --fif-dir artifacts\zuna\fif `
  --work-dir artifacts\zuna\aaaaaaaq_s006_t000 `
  --gpu-device 0
```

Use `--gpu-device ""` for CPU. The first run downloads ZUNA weights from
Hugging Face. CPU mode is memory-capped by default (`tokens_per_batch <= 512`)
to avoid OOM on Windows. Use GPU/CUDA for full-file comparison when possible.

Safe CPU smoke run:

```powershell
C:\Users\User\miniconda3\Scripts\conda.exe run -n zuna-eeg python utils\zuna_bridge.py run `
  --fif-dir artifacts\zuna\fif `
  --work-dir artifacts\zuna\aaaaaaaq_s006_t000 `
  --gpu-device "" `
  --diffusion-steps 2 `
  --tokens-per-batch 512
```

## Export Back To Repo Format

The bridge can convert a reconstructed ZUNA `.fif` file back to a compressed
19-channel array using this repo's canonical channel order. MNE stores EEG in
volts, but the legacy detector expects EDF physical values in microvolts, so
the bridge exports `data` in microvolts. This is the handoff format for later
ConvLSTM experiments.

```powershell
C:\Users\User\miniconda3\Scripts\conda.exe run -n zuna-eeg python utils\zuna_bridge.py export-npz `
  --fif artifacts\zuna\aaaaaaaq_s006_t000\4_fif_output\aaaaaaaq_s006_t000.fif `
  --out artifacts\zuna\aaaaaaaq_s006_t000\aaaaaaaq_s006_t000.zuna_19ch.npz `
  --overwrite
```

## Compare Baseline vs ZUNA

The comparison keeps the seizure detector fixed and changes only the signal
source. The baseline row uses the original EDF probabilities; the ZUNA row
uses a ZUNA-exported 19-channel NPZ and then runs the same ConvLSTM detector.

Run the current detector/cache only:

```powershell
C:\Users\User\miniconda3\envs\seiz36\python.exe experiments\compare_zuna.py `
  --edf sample_data\v2.0.0\edf\eval\aaaaaaaq\s006_2014_08_18\01_tcp_ar\aaaaaaaq_s006_t000.edf `
  --threshold 0.5 `
  --out artifacts\zuna_compare\aaaaaaaq_s006_t000.baseline.json
```

Run the full comparison after exporting a ZUNA NPZ:

```powershell
C:\Users\User\miniconda3\envs\seiz36\python.exe experiments\compare_zuna.py `
  --edf sample_data\v2.0.0\edf\eval\aaaaaaaq\s006_2014_08_18\01_tcp_ar\aaaaaaaq_s006_t000.edf `
  --zuna-npz artifacts\zuna\aaaaaaaq_s006_t000\aaaaaaaq_s006_t000.zuna_19ch.npz `
  --zuna-probs-out artifacts\zuna_compare\aaaaaaaq_s006_t000.zuna.probs.npz `
  --threshold 0.5 `
  --out artifacts\zuna_compare\aaaaaaaq_s006_t000.zuna_compare.json
```

The output JSON includes reference seizures, predicted events, one-to-one
matches, sensitivity, misses, false positives, false alarms per 24 hours,
mean onset/offset error, and probability summary statistics.

## VM Execution Notes

The current GCP VM `instance-20260513-221429` is useful for safe background
CPU runs because it has much more RAM than the Windows desktop, but it does not
currently expose CUDA (`nvidia-smi` is not installed). Treat it as a long-run
CPU fallback, not a low-latency ZUNA machine.

Check the current VM IP through `gcloud`; the SSH config can become stale:

```powershell
gcloud compute instances list --format="table(name,zone,status,machineType.basename(),networkInterfaces[0].accessConfigs[0].natIP)"
```

Run remote commands through `gcloud compute ssh` or refresh SSH config before
using an alias. For long ZUNA jobs, use `tmux` and a log file:

```bash
tmux new -s zuna_safe
cd ~/Continental-Seiz-detection
python utils/zuna_bridge.py run \
  --fif-dir artifacts/zuna/fif \
  --work-dir artifacts/zuna/aaaaaaaq_s006_t000 \
  --gpu-device "" \
  --diffusion-steps 2 \
  --tokens-per-batch 512 2>&1 | tee artifacts/zuna/zuna_safe.log
```

Do not raise `--tokens-per-batch` on CPU unless RAM has been checked and
`--allow-high-memory` is intentionally used.

## GUI Full-ZUNA Mode

The GUI keeps the original EDF baseline as the default review path. Opening a
file loads or computes the baseline `.probs.npz` first, so threshold adjustment
and event review remain fast. Full ZUNA is optional for the current session.

When an EDF is open, press **Run full ZUNA**. The GUI starts
`utils/zuna_bridge.py` in a separate modern Python interpreter, writes full-ZUNA
artifacts under `artifacts/gui_zuna/<edf>_<hash>/`, then scores the exported
ZUNA 19-channel NPZ with the same ConvLSTM detector. Once the ZUNA probability
cache exists, the **AI source** selector switches between:

- `Baseline`: original EDF signal and baseline ConvLSTM probabilities.
- `ZUNA full`: full-ZUNA reconstructed signal and ConvLSTM probabilities on
  that reconstructed signal.

Set `ZUNA_PYTHON` if the GUI cannot find the ZUNA environment automatically:

```powershell
$env:ZUNA_PYTHON='C:\Users\User\miniconda3\envs\zuna-gpu\python.exe'
```

Optional runtime controls:

```powershell
$env:ZUNA_DIFFUSION_STEPS='50'
$env:ZUNA_TOKENS_PER_BATCH='128'
$env:ZUNA_GPU_DEVICE='0'
```

The GUI defaults to `ZUNA_TOKENS_PER_BATCH=128` when the variable is not set.
This keeps full-ZUNA safer on a 12 GB desktop GPU, but it can be slow.

The full-ZUNA progress dialog reports the current bridge stage, elapsed time,
settings, log path, and artifact directory. The percentage is artifact-based:
it advances when prepared FIF, ZUNA input tensors, ZUNA output tensors, the
reconstructed FIF, or the exported NPZ appear on disk. During the long upstream
ZUNA inference call, the percentage may hold steady if ZUNA has not written a
new artifact yet.

## Integration Plan

1. Run the current 19-channel detector as the baseline.
2. Optionally run full-file ZUNA for the current session.
3. Feed the full-ZUNA reconstructed signal into the existing STFT/ConvLSTM path.
4. Allow the reviewer to switch between baseline and full-ZUNA AI sources.
5. Artificially drop channels from selected EDF files for reconstruction
   benchmarking.
6. Reconstruct missing channels with MNE interpolation and with ZUNA.
7. Compare event sensitivity, onset/offset error, false alarms per 24h, and
   wall-clock runtime per EEG hour.

Do not display ZUNA output as original clinical signal. In the GUI, any channel
that came from ZUNA should be labelled as imputed and exported with provenance.
