#!/usr/bin/env bash
# Long-running co-evolution worker for the VPS.
#
# Mirrors the pattern in ~/Claude Playground/Analysis Toys/scripts/run_worker.sh:
#   - nice'd Python so the box stays responsive
#   - unbuffered stdout (-u) so the log file fills line-by-line
#   - --resume so a crash is just a restart, not a from-scratch redo
#   - exponential-ish backoff on crashes (sleep 60s)
#   - clean exit when --resume reports "no_op" (all requested gens done)
#
# Knobs at the top — edit before launching, OR set them as env vars.
#
# Launch as detached tmux session:
#   tmux new-session -d -s evo './scripts/run_worker.sh'
# Watch it:
#   tail -F logs/coevolve.log
# Kill:
#   tmux kill-session -t evo

set -uo pipefail
cd "$(dirname "$0")/.."

# ---- knobs ----
SEED_CKPT="${SEED_CKPT:-runs/seed_ckpt/checkpoints/final}"   # initial PPO ckpt
OUT="${OUT:-runs/coevolve}"
GENERATIONS="${GENERATIONS:-50}"
N_MUTANTS="${N_MUTANTS:-20}"
SIGMA="${SIGMA:-0.2}"
EVAL_EPS="${EVAL_EPS:-2}"
N_PRED="${N_PRED:-2}"
N_PREY="${N_PREY:-2}"
N_OBSTACLES="${N_OBSTACLES:-2}"
MAX_CYCLES="${MAX_CYCLES:-100}"
WORKERS="${WORKERS:-2}"                                       # 2 vCPUs -> 2 workers
NUM_THREADS="${NUM_THREADS:-1}"                               # per-process thread cap
SEED="${SEED:-2024}"
LOG="${LOG:-logs/coevolve.log}"

# ---- helpers ----
mkdir -p logs runs
PY="${PY:-./venv/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi

# ---- step 0: ensure seed ckpt exists (bootstrap once) ----
if [ ! -f "$SEED_CKPT/manifest.json" ]; then
  echo "[$(date -Iseconds)] no seed ckpt at $SEED_CKPT — training a small PPO baseline first" | tee -a "$LOG"
  SDL_VIDEODRIVER=dummy nice -n 19 "$PY" -u run_cpu.py \
    --config configs/preview_coevolve.yaml \
    --save_dir runs/seed_ckpt_train --num_threads "$NUM_THREADS" >> "$LOG" 2>&1
  SEED_RUN=$(ls -td runs/seed_ckpt_train/run_* | head -1)
  rm -rf "$(dirname "$SEED_CKPT")"
  mkdir -p "$(dirname "$SEED_CKPT")"
  cp -r "$SEED_RUN/checkpoints/final" "$SEED_CKPT"
fi

# ---- step 1: coevolve loop with restart on crash ----
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date -Iseconds)] starting coevolve attempt #$attempt (resume=yes)" | tee -a "$LOG"

  SDL_VIDEODRIVER=dummy nice -n 19 "$PY" -u -m train.tools.coevolve \
    --seed_ckpt "$SEED_CKPT" --out "$OUT" \
    --generations "$GENERATIONS" --n_mutants "$N_MUTANTS" --sigma "$SIGMA" \
    --eval_eps "$EVAL_EPS" \
    --n_predators "$N_PRED" --n_prey "$N_PREY" --n_obstacles "$N_OBSTACLES" \
    --max_cycles "$MAX_CYCLES" --seed "$SEED" \
    --workers "$WORKERS" --num_threads "$NUM_THREADS" \
    --resume >> "$LOG" 2>&1
  rc=$?

  if [ $rc -eq 0 ]; then
    echo "[$(date -Iseconds)] coevolve exited cleanly (rc=0). Generating report and exiting." | tee -a "$LOG"
    "$PY" -m train.tools.report --dir "$OUT" --out "$OUT/report.html" >> "$LOG" 2>&1 || true
    exit 0
  fi

  echo "[$(date -Iseconds)] coevolve crashed (rc=$rc). Sleeping 60s then resuming." | tee -a "$LOG"
  sleep 60
done
