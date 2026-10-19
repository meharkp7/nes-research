#!/usr/bin/env bash
# Supervisor for the StableLM Gate E utility run.
# Keeps the resumable runner alive across crashes: if the process dies before
# all conditions complete, it is relaunched and resumes from per-batch
# checkpoints (nll/*.partial.json) and skips completed conditions (nll/*.json).
set -u
ROOT=/workspace/nes_run_20261016_anchor_ablation
QUAL="$ROOT/stablelm_qualification"
PY=/workspace/nes_env/bin/python
LOG="$QUAL/utility_supervisor.log"
HEART="$QUAL/utility_supervisor.heartbeat"
RUNNER="exp_stablelm_utility.py"
OUT="$QUAL/utility_v2"
ARGS=(--output "$OUT" --n-seq 580 --batch 4 --sizes 10000 50000 --seeds 20261016 20261017 20261018)

echo "[$(date -Is)] supervisor start (pid $$)" >> "$LOG"

# Single-instance guard: refuse to start if another supervisor holds the lock.
LOCK="$QUAL/utility_supervisor.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Is)] another supervisor holds the lock; exiting" >> "$LOG"
  exit 0
fi

while true; do
  date -Is > "$HEART"
  if ! pgrep -f "$RUNNER" >/dev/null 2>&1; then
    if [ -f "$OUT/comparisons.json" ]; then
      echo "[$(date -Is)] all conditions complete; supervisor exiting" >> "$LOG"
      break
    fi
    echo "[$(date -Is)] runner not found; relaunching" >> "$LOG"
    ( cd "$ROOT/code/nes-llm" && nohup "$PY" "$QUAL/$RUNNER" "${ARGS[@]}" >> "$QUAL/utility_v2.log" 2>&1 & )
    echo "[$(date -Is)] relaunch issued" >> "$LOG"
  fi
  sleep 60
done
