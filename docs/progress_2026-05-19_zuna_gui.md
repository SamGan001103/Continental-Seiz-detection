# Progress Log - 2026-05-19

> ## ⚠ HISTORICAL SNAPSHOT — DO NOT QUOTE THE NUMBERS
>
> This is a dated log of what was known on 19 May 2026. **Every event-level figure below is
> superseded and wrong** (sensitivity 26.3 % / 31.6 %, FP/24 h 328.7 / 451.9 / 205.4, and the
> threshold-sweep tables). They were computed before three scoring defects were fixed: the source
> method's decision stage was not applied, detection fragments inside an already-detected seizure
> were charged as false positives, and `run_inference.py` reported event times in window-index
> units. See `docs/reproduction_status.md` §3.
>
> It also states that the work "is not a reproduction of the paper". That was based on comparing
> against the wrong paper. The 19-channel detector **does** replicate the 2022
> continental-generalization result (window AUC 0.822 vs 0.84 published).
>
> **For current numbers use `docs/RESULTS.md`.** Kept only as a record of the project's history.

## Thesis Context

Project direction: GUI-based Human-AI system for reviewing ambulatory/outpatient EEG files using public datasets only. Current repo work is focused on rebuilding and extending the existing ConvLSTM EEG seizure-detection GUI, then investigating whether ZUNA can improve the signal before the existing detector scores it.

## GUI And ZUNA Integration

Implemented the GUI design where baseline remains the default pipeline:

- Opening an EDF runs or loads the normal baseline ConvLSTM probability pipeline.
- Threshold adjustment still works immediately on the active probability source.
- Added an optional `Run full ZUNA` action for the current session.
- Added an AI source selector so the user can switch between `Baseline` and `ZUNA full` after ZUNA output is available.
- ZUNA runs as a separate external process in the modern ZUNA environment, not inside the old `seiz36` TensorFlow environment.
- ZUNA GUI artifacts are cached under `artifacts/gui_zuna/`.
- Cache validation now includes source EDF path/size/mtime so old ZUNA outputs are not silently reused for a different file.
- ZUNA export uses `.zuna.reviewed.csv_bi` and writes provenance metadata.
- Closing the GUI while ZUNA is running asks to cancel; changing EDF cancels stale ZUNA work.

Progress dialog update:

- Removed estimated remaining time because it can be misleading.
- Kept percentage, but made it artifact/stage-based instead of time-based.
- Current progress is based on known stages such as FIF prepared, ZUNA tensors present, output tensors written, reconstructed FIF present, and exported NPZ present.
- During upstream ZUNA inference the percentage may hold steady if ZUNA itself is not writing intermediate artifacts.

Validation performed:

- `python -m py_compile gui\app.py gui\io\zuna.py` passed.
- `python -m unittest discover tests` passed with 11 tests.
- Offscreen GUI smoke test loaded a cached EDF baseline successfully.
- ZUNA bridge import worked in the `zuna-gpu` environment.

## Removed Fast Selective ZUNA

Decision: remove the fast selective ZUNA idea for now and keep only full ZUNA.

Removed or cleaned:

- `experiments/prepare_zuna_candidates.py`
- `experiments/compare_zuna_candidates.py`
- `tests/test_zuna_candidates.py`
- Candidate/clip arguments from ZUNA bridge/manifest preparation.
- Documentation references to fast selective ZUNA.

Remaining ZUNA mode is full-session ZUNA only.

## VM ZUNA Run

The full-ZUNA thesis run was launched on the GCP VM:

- Instance: `instance-20260513-221429`
- Zone: `australia-southeast1-c`
- VM workdir: `/mnt/research-data/work/Continental-Seiz-detection`
- Main log: `artifacts/zuna_thesis/zuna_thesis_default.log`
- Status log: `artifacts/zuna_thesis/zuna_thesis_status.jsonl`
- ZUNA settings:
  - `diffusion_steps=50`
  - `tokens_per_batch=512`

The VM completed 10 ZUNA NPZ outputs before we stopped it.

Completed ZUNA files:

1. `aaaaatao_s003_t001`
2. `aaaaatao_s003_t000`
3. `aaaaaarq_s016_t007`
4. `aaaaaqtw_s002_t001`
5. `aaaaaghb_s010_t000`
6. `aaaaahsi_s014_t000`
7. `aaaaaraf_s004_t000`
8. `aaaaaarq_s017_t000`
9. `aaaaaarq_s016_t002`
10. `aaaaaarq_s016_t003`

The 10 completed NPZ outputs were copied locally into:

```text
artifacts/zuna_thesis/npz/
```

The VM was then stopped. GCP reported status:

```text
TERMINATED
```

This is the stopped state for a Compute Engine instance, so the VM is no longer burning compute/GPU credits.

## First 7 Comparison

Earlier partial comparison used the first 7 completed ZUNA files.

Artifacts:

- `artifacts/zuna_thesis/compare_first7/zuna_manifest_summary.json`
- `artifacts/zuna_thesis/compare_first7/zuna_first7_result_report.html`
- `artifacts/zuna_thesis/compare_first7/zuna_first7_result_diagram.svg`
- `artifacts/zuna_thesis/compare_first7/zuna_first7_runtime_digest.json`

Threshold:

```text
0.5
```

Results:

| Metric | Baseline | Full ZUNA |
|---|---:|---:|
| Files | 7 | 7 |
| Reference seizures | 9 | 9 |
| Hits | 5 | 6 |
| Misses | 4 | 3 |
| Sensitivity | 55.6% | 66.7% |
| False positives | 6 | 4 |

Runtime:

- Original EEG duration: `20m03s`
- Summed ZUNA VM inference runtime: `2h01m25s`
- Approximate runtime ratio: `6.1x` slower than real time
- Peak RAM: about `41.3 GiB`

## First 10 Comparison

After copying all 10 completed ZUNA outputs locally, a first-10 comparison was run.

Manifest:

```text
artifacts/zuna_thesis/manifest_first10_completed.csv
```

Main output:

```text
artifacts/zuna_thesis/compare_first10/zuna_manifest_summary.json
```

Additional outputs:

```text
artifacts/zuna_thesis/compare_first10/first10_file_metrics.csv
artifacts/zuna_thesis/compare_first10/first10_runtime_digest.json
artifacts/zuna_thesis/compare_first10/threshold_sweep_first10.json
artifacts/zuna_thesis/compare_first10/*.baseline.probs.npz
artifacts/zuna_thesis/compare_first10/*.zuna.probs.npz
```

Threshold:

```text
0.5
```

Results:

| Metric | Baseline | Full ZUNA |
|---|---:|---:|
| Files | 10 | 10 |
| Reference seizures | 19 | 19 |
| Hits | 5 | 6 |
| Misses | 14 | 13 |
| Sensitivity | 26.3% *(superseded: 21.1%)* | 31.6% *(superseded: 26.3%)* |
| False positives | 8 | 5 |
| Predicted events | 13 | 11 |
| FP / 24h | 328.7 *(superseded: 41.1)* | 205.4 *(superseded: 0.0)* |

Interpretation:

- ZUNA improved the first-10 pilot result slightly.
- ZUNA gained 1 additional seizure hit.
- ZUNA reduced false positives by 3.
- The strongest ZUNA improvement was `aaaaaarq_s016_t007`, where baseline missed all 3 reference seizures and ZUNA detected 2.
- ZUNA was not strictly better on every file; it missed `aaaaatao_s003_t001`, which baseline detected.

Runtime for first 10:

- EEG duration compared: `2103s` = `35m03s`
- Full ZUNA runtime sum: `12648s` = `3h30m48s`
- Runtime ratio: about `6.0x` slower than real time
- Peak RAM observed: about `42.2 GiB`

## Baseline Mismatch Investigation

Question investigated: why is baseline not hitting reference seizures, and why is this inconsistent with the paper?

Six investigation tracks were assigned to subagents:

1. Paper/evaluation protocol mismatch.
2. Repo baseline inference path.
3. Preprocessing/channel fidelity.
4. Threshold/AUC behavior.
5. Label/event matching validity.
6. ZUNA-vs-baseline artifact validity.

Main conclusion:

The first-10 baseline result is not directly comparable to the paper's headline result.

The paper reports:

- TUH result mainly as 12-second window-level AUC, not our event-level sensitivity.
- RPAH full clinical inference as about `76.68%` sensitivity / `56.55` false alarms per 24h.
- RPAH 66-session AI-assisted review as `92.19%`, but that includes the human arbiter workflow.
- High-risk regions at probability `>=10%`, followed by post-processing/lens logic.

Our first-10 comparison used:

- Only 10 TUH seizure files.
- Current GUI 19-channel ConvLSTM path.
- 12-second windows.
- 6-second stride.
- Direct event thresholding at `0.5`.
- No paper-style PWA/PEI lens post-processing.
- No human arbiter workflow.
- Event-level `.csv_bi` matching.

Therefore, the low first-10 baseline sensitivity should not be treated as a failed reproduction of the paper.

## Threshold Sweep Findings

Explicit baseline probability files were saved and a threshold sweep was run.

Baseline event-level sweep:

> **[SUPERSEDED — see docs/RESULTS.md]** The FP/24 h column below is wrong; it predates the scoring fixes.

| Threshold | Hits / 19 | Sensitivity | False Positives | FP / 24h |
|---:|---:|---:|---:|---:|
| 0.50 | 5 | 26.3% | 8 | 328.7 |
| 0.10 | 5 | 26.3% | 11 | 451.9 |
| 0.05 | 7 | 36.8% | 15 | 616.3 |
| 0.01 | 11 | 57.9% | 20 | 821.7 |

ZUNA event-level sweep:

> **[SUPERSEDED — see docs/RESULTS.md]** The FP/24 h column below is wrong; it predates the scoring fixes.

| Threshold | Hits / 19 | Sensitivity | False Positives | FP / 24h |
|---:|---:|---:|---:|---:|
| 0.50 | 6 | 31.6% | 5 | 205.4 |
| 0.10 | 9 | 47.4% | 14 | 575.2 |
| 0.05 | 10 | 52.6% | 16 | 657.3 |
| 0.01 | 12 | 63.2% | 21 | 862.8 |

Interpretation:

- Lowering the threshold recovers more seizures.
- However, false positives rise sharply.
- This means the low baseline result is not only a threshold issue.
- Several seizure files get very low baseline probabilities, so the current pipeline has weak separation on this first-10 subset.

Examples of low baseline confidence:

| File | Baseline Max Probability | Reference Seizures Missed |
|---|---:|---:|
| `aaaaaarq_s016_t007` | 0.0058 | 3 |
| `aaaaaghb_s010_t000` | 0.0058 | 1 |
| `aaaaaarq_s017_t000` | 0.0595 | 4 |
| `aaaaaarq_s016_t002` | 0.0346 | 3 |

## Technical Risks Identified

The investigation found several important risks:

- The GUI baseline is not the full paper clinical workflow.
- The older repo scripts are not fully consistent with the GUI path.
- `run_inference.py` uses default `step=1` and threshold `0.95`, while GUI compare uses `step=6` and threshold `0.5`.
- Current per-window ICA is fragile on 12-second windows.
- MNE warnings appeared during recomputation:
  - filter length longer than signal
  - FastICA convergence warnings
- Label parsing and event matching appear valid; they are not the main cause of the misses.
- Some high probabilities occur away from reference seizure intervals, indicating localization/calibration problems.
- ZUNA comparison is mostly comparable to baseline at the detector level, but some ZUNA NPZ files are slightly shorter than the source EDF by 1-3 seconds.
- Saved ZUNA probability files should be treated as run-local derived caches unless provenance is expanded.

## Current Position

What we can currently claim:

- Full ZUNA can be integrated into the GUI as an optional, heavy, cached session mode.
- On the first-10 pilot, full ZUNA slightly improved event sensitivity and reduced false positives compared with the current GUI baseline at threshold `0.5`.
- The first-10 result is a pilot comparison. *(Superseded: the 19-channel detector DOES replicate the 2022 continental-generalization paper at the window level, AUC 0.822 vs 0.84 published. See docs/reproduction_status.md.)*
- Full ZUNA is currently too slow for routine interactive GUI use unless run offline, cached, or selectively invoked by the user.

What we should not claim:

- We should not claim that the current baseline reproduces the paper result.
- We should not claim ZUNA is clinically useful from 10 files.
- We should not claim the first-10 sensitivity is representative of TUH, RPAH, or outpatient EEG broadly.

## Recommended Next Steps

1. Build a proper threshold-sweep report for baseline and ZUNA:
   - event sensitivity
   - false alarms per 24h
   - onset/offset error
   - window-level ROC/AUC
   - PR-AUC if possible

2. Separate evaluation modes clearly:
   - GUI pilot event-level comparison
   - paper-style window-level AUC comparison
   - clinical-review workflow simulation

3. Improve provenance:
   - record EDF path/size/mtime/hash
   - record ZUNA parameters
   - record model weights
   - record threshold/stride/ICA settings

4. Investigate ICA fragility:
   - compare ICA vs no-ICA baseline
   - compare 6-second stride vs 1-second stride
   - confirm whether MNE version differences affect probabilities

5. Create supervisor-facing plots:
   - threshold vs sensitivity
   - threshold vs false alarms
   - baseline vs ZUNA per-file hit/miss table
   - runtime vs EEG duration

## Useful Commands

Launch GUI:

```powershell
cd C:\Users\User\Continental-Seiz-detection
C:\Users\User\miniconda3\envs\seiz36\python.exe -m gui.main
```

Run first-10 comparison again:

```powershell
cd C:\Users\User\Continental-Seiz-detection
C:\Users\User\miniconda3\envs\seiz36\python.exe experiments\compare_zuna_manifest.py `
  --manifest artifacts\zuna_thesis\manifest_first10_completed.csv `
  --zuna-npz-dir artifacts\zuna_thesis\npz `
  --out-dir artifacts\zuna_thesis\compare_first10 `
  --threshold 0.5 `
  --step 6 `
  --python C:\Users\User\miniconda3\envs\seiz36\python.exe
```

Check VM status:

```powershell
gcloud compute instances describe instance-20260513-221429 `
  --zone australia-southeast1-c `
  --format="value(status)"
```

Expected current status:

```text
TERMINATED
```

