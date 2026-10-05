#!/usr/bin/env bash
set -euo pipefail

# Sequential ZUNA runner for VM use. Defaults match the GCP research VM layout,
# but every path/limit can be overridden through environment variables.
ROOT=${ROOT:-/mnt/research-data/work/Continental-Seiz-detection}
PY=${PY:-/mnt/research-data/venvs/zuna-eeg/bin/python}
BATCH=${BATCH:-$ROOT/artifacts/zuna_thesis}
MANIFEST=${MANIFEST:-$BATCH/manifest.csv}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-50}
TOKENS_PER_BATCH=${TOKENS_PER_BATCH:-512}
RSS_LIMIT_KB=${RSS_LIMIT_KB:-73400320}      # 70 GiB
AVAIL_LIMIT_KB=${AVAIL_LIMIT_KB:-41943040}  # 40 GiB

LOG=$BATCH/zuna_thesis_default.log
WATCH=$BATCH/zuna_thesis_watchdog.log
STATUS=$BATCH/zuna_thesis_status.jsonl

mkdir -p "$BATCH/input" "$BATCH/npz" "$BATCH/logs"
touch "$LOG" "$WATCH" "$STATUS"

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

sum_rss_for_work() {
  local work="$1"
  ps -eo rss=,args= | awk -v w="$work" 'index($0, w) {sum += $1} END {print sum + 0}'
}

json_status() {
  "$PY" - "$@" <<'PY'
import json
import sys
print(json.dumps(dict(arg.split("=", 1) for arg in sys.argv[1:]), sort_keys=True))
PY
}

run_one() {
  local manifest_index="$1"
  local cohort="$2"
  local stem="$3"
  local duration_s="$4"
  local seiz_count="$5"
  local input_dir="$BATCH/input/$stem"
  local work_dir="$BATCH/$stem"
  local fif="$BATCH/fif/$stem.fif"
  local one_log="$BATCH/logs/$stem.run.log"
  local out_npz="$BATCH/npz/$stem.zuna_19ch.npz"

  if [[ -s "$out_npz" ]]; then
    echo "[$(ts)] SKIP_EXISTING $stem $out_npz" | tee -a "$LOG"
    json_status event=skip_existing stem="$stem" cohort="$cohort" \
      manifest_index="$manifest_index" out_npz="$out_npz" >> "$STATUS"
    return 0
  fi

  echo "[$(ts)] START $stem cohort=$cohort duration_s=$duration_s seiz_count=$seiz_count diffusion_steps=$DIFFUSION_STEPS tokens_per_batch=$TOKENS_PER_BATCH" | tee -a "$LOG"
  json_status event=start stem="$stem" cohort="$cohort" \
    manifest_index="$manifest_index" duration_s="$duration_s" \
    seiz_count="$seiz_count" diffusion_steps="$DIFFUSION_STEPS" \
    tokens_per_batch="$TOKENS_PER_BATCH" timestamp="$(ts)" >> "$STATUS"

  if [[ ! -s "$fif" ]]; then
    echo "[$(ts)] MISSING_FIF $fif" | tee -a "$LOG"
    json_status event=missing_fif stem="$stem" cohort="$cohort" \
      manifest_index="$manifest_index" fif="$fif" timestamp="$(ts)" >> "$STATUS"
    return 2
  fi

  rm -rf "$input_dir" "$work_dir"
  mkdir -p "$input_dir" "$work_dir"
  ln -s "$fif" "$input_dir/$stem.fif"

  local started ended runtime_s
  started=$(date +%s)
  setsid "$PY" "$ROOT/utils/zuna_bridge.py" run \
    --fif-dir "$input_dir" \
    --work-dir "$work_dir" \
    --diffusion-steps "$DIFFUSION_STEPS" \
    --tokens-per-batch "$TOKENS_PER_BATCH" \
    > "$one_log" 2>&1 &
  local pid=$!
  echo "[$(ts)] PID $pid $stem" | tee -a "$LOG"

  local peak_rss=0
  while kill -0 "$pid" 2>/dev/null; do
    local avail rss
    avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    rss=$(sum_rss_for_work "$work_dir")
    if (( rss > peak_rss )); then
      peak_rss=$rss
    fi
    echo "[$(ts)] $stem pid=$pid rss_kb=$rss peak_rss_kb=$peak_rss mem_available_kb=$avail" >> "$WATCH"
    if (( rss > RSS_LIMIT_KB || avail < AVAIL_LIMIT_KB )); then
      echo "[$(ts)] KILL $stem rss_kb=$rss mem_available_kb=$avail" | tee -a "$LOG"
      kill -TERM -"$pid" 2>/dev/null || true
      sleep 10
      kill -KILL -"$pid" 2>/dev/null || true
      wait "$pid" || true
      json_status event=killed stem="$stem" cohort="$cohort" \
        manifest_index="$manifest_index" rss_kb="$rss" \
        mem_available_kb="$avail" timestamp="$(ts)" >> "$STATUS"
      return 99
    fi
    sleep 20
  done

  wait "$pid"
  ended=$(date +%s)
  runtime_s=$((ended - started))
  echo "[$(ts)] INFERENCE_DONE $stem runtime_s=$runtime_s peak_rss_kb=$peak_rss" | tee -a "$LOG"

  local out_fif
  out_fif=$(find "$work_dir/4_fif_output" -maxdepth 1 -type f -name '*.fif' | head -n 1 || true)
  if [[ -z "$out_fif" ]]; then
    echo "[$(ts)] MISSING_OUTPUT_FIF $stem" | tee -a "$LOG"
    json_status event=missing_output_fif stem="$stem" cohort="$cohort" \
      manifest_index="$manifest_index" runtime_s="$runtime_s" \
      peak_rss_kb="$peak_rss" timestamp="$(ts)" >> "$STATUS"
    return 3
  fi

  "$PY" "$ROOT/utils/zuna_bridge.py" export-npz \
    --fif "$out_fif" \
    --out "$out_npz" \
    --overwrite \
    >> "$one_log" 2>&1
  echo "[$(ts)] EXPORTED $out_npz" | tee -a "$LOG"
  json_status event=exported stem="$stem" cohort="$cohort" \
    manifest_index="$manifest_index" runtime_s="$runtime_s" \
    peak_rss_kb="$peak_rss" out_npz="$out_npz" timestamp="$(ts)" >> "$STATUS"
}

cd "$ROOT"
if [[ ! -s "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 2
fi

echo "[$(ts)] RUN_BEGIN manifest=$MANIFEST diffusion_steps=$DIFFUSION_STEPS tokens_per_batch=$TOKENS_PER_BATCH" | tee -a "$LOG"
tail -n +2 "$MANIFEST" | while IFS=, read -r manifest_index cohort stem duration_s _size_mb seiz_count _seiz_total_s _edf _zuna_quality; do
  run_one "$manifest_index" "$cohort" "$stem" "$duration_s" "$seiz_count"
done
echo "[$(ts)] ALL_DONE" | tee -a "$LOG"
