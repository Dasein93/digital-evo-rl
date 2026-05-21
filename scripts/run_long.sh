#!/usr/bin/env bash
# Long-running pure-PPO training entrypoint for the VPS.
#
# Wraps run_cpu.py in the same tmux + nice + unbuffered-log pattern as
# scripts/run_worker.sh, but for "let this train for hours" — not the
# coevolve loop.  Intermediate checkpoints are written every
# checkpoint.every episodes (see configs/big.yaml), so a crash leaves
# usable artifacts even if the run doesn't reach total_episodes.
#
# NOTE: run_cpu.py does not have --resume yet (each invocation makes a
# new run_YYYYMMDD_HHMMSS dir). If this process exits non-zero, the
# wrapper does NOT restart it — picking up the previous run cleanly
# would require run_cpu resume support, which is a separate piece of
# work. For now: if it crashes, inspect the last checkpoints under
# runs/long/run_*/checkpoints/ep_NNNNNN/ and re-launch manually.
#
# Launch:
#   tmux new-session -d -s long './scripts/run_long.sh'
# Watch:
#   tail -F logs/long.log
# Kill:
#   tmux kill-session -t long

set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/big.yaml}"
SAVE_DIR="${SAVE_DIR:-runs/long}"
NUM_THREADS="${NUM_THREADS:-2}"
EPISODES="${EPISODES:-}"               # empty = use config's total_episodes
SEED="${SEED:-}"                       # empty = use config's seed
LOG="${LOG:-logs/long.log}"

mkdir -p logs runs

PY="${PY:-./venv/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi

extra=()
if [ -n "$EPISODES" ]; then extra+=(--episodes "$EPISODES"); fi
if [ -n "$SEED"     ]; then extra+=(--seed     "$SEED");     fi

echo "[$(date -Iseconds)] launching run_cpu config=$CONFIG threads=$NUM_THREADS save_dir=$SAVE_DIR" | tee -a "$LOG"
SDL_VIDEODRIVER=dummy nice -n 19 "$PY" -u run_cpu.py \
    --config "$CONFIG" --save_dir "$SAVE_DIR" --num_threads "$NUM_THREADS" \
    "${extra[@]}" >> "$LOG" 2>&1
rc=$?
echo "[$(date -Iseconds)] run_cpu exited rc=$rc" | tee -a "$LOG"
exit $rc
