#!/usr/bin/env bash
# Bootstrap digital-evo-rl on the VPS from your Mac.
#
# Idempotent: re-running just rsyncs code + reinstalls only if the venv is missing.
#
# Usage:
#   ./scripts/vps_bootstrap.sh                  # defaults
#   VPS_HOST=stepan-vps ./scripts/vps_bootstrap.sh
#   PROJ=/opt/digital-evo-rl ./scripts/vps_bootstrap.sh
#
# Assumes the layout from docs/vps.md:
#   /opt/<proj>/{src,data,runs,logs,venv}
#   PyTorch CPU-only wheel (no GPU on box)
#   nice -n 19 + tmux for long runs (see run_worker.sh)

set -euo pipefail

VPS_HOST="${VPS_HOST:-stepan-vps}"
PROJ_REMOTE="${PROJ:-/opt/digital-evo-rl}"
PY="${PY:-python3}"

echo "==> Target: ${VPS_HOST}:${PROJ_REMOTE}"

# 1. Reachability + tool check.
ssh -o BatchMode=yes "$VPS_HOST" "uname -a && which $PY tmux rsync git" \
    || { echo "SSH/tooling check failed"; exit 1; }

# 2. Install OS-level pygame deps if missing (idempotent; needs sudo OR root).
ssh "$VPS_HOST" 'set -e
  if ! dpkg -s libsdl2-2.0-0 >/dev/null 2>&1; then
    echo "Installing SDL2 system libs..."
    apt-get update -y && apt-get install -y --no-install-recommends \
      libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0
  else
    echo "SDL2 libs already present."
  fi'

# 3. Create project tree.
ssh "$VPS_HOST" "mkdir -p $PROJ_REMOTE/{runs,logs}"

# 4. Rsync code (excludes generated stuff and venv).
# CRITICAL: runs/ and logs/ are VPS-side working data — never delete them from
# the remote even when they're absent locally. Same for the dormant
# /opt/transcribe/ scaffolding (which is outside $PROJ_REMOTE but be safe).
echo "==> Rsync code -> $VPS_HOST:$PROJ_REMOTE"
rsync -avh --delete \
  --exclude .git --exclude venv --exclude __pycache__ \
  --exclude artifacts --exclude .pytest_cache \
  --exclude '*.pyc' --exclude '.venv' \
  --exclude runs --exclude logs \
  ./ "$VPS_HOST:$PROJ_REMOTE/"

# 5. Create venv + install CPU-only PyTorch + the rest of requirements.txt.
ssh "$VPS_HOST" "set -e
  cd $PROJ_REMOTE
  if [ ! -d venv ]; then
    $PY -m venv venv
    ./venv/bin/pip install --upgrade pip wheel
    ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    ./venv/bin/pip install -r requirements.txt
    ./venv/bin/pip install pytest
  else
    # Ensure pytest is present even on re-runs of an existing venv.
    ./venv/bin/pip show pytest >/dev/null 2>&1 || ./venv/bin/pip install pytest
    echo 'venv exists; pytest verified (delete venv/ to force full reinstall).'
  fi"

# 6. Quick smoke test on the remote side.
echo "==> Running pytest -q on the VPS (full suite)"
ssh "$VPS_HOST" "cd $PROJ_REMOTE && SDL_VIDEODRIVER=dummy ./venv/bin/python -m pytest -q"

echo
echo "==> Done. Next steps:"
echo "    # interactive 1-line smoke training:"
echo "    ssh $VPS_HOST 'cd $PROJ_REMOTE && SDL_VIDEODRIVER=dummy ./venv/bin/python run_cpu.py --config configs/smoke.yaml --num_threads 2'"
echo
echo "    # detached multi-day co-evolution (see scripts/run_worker.sh on the box):"
echo "    ssh $VPS_HOST 'cd $PROJ_REMOTE && tmux new-session -d -s evo \"./scripts/run_worker.sh\"'"
echo "    ssh $VPS_HOST 'tail -F $PROJ_REMOTE/logs/coevolve.log'"
