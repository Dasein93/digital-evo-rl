# Deploying digital-evo-rl on the VPS

Short version: run `./scripts/vps_bootstrap.sh` from a Mac with SSH access
to `stepan-vps`. It rsyncs the repo to `/opt/digital-evo-rl/`, builds a
CPU-only venv, runs `pytest -q` to confirm, and prints the next-step
commands.

This document records the **project-specific** facts. The full VPS
reference (network, alternate projects, dormant transcription worker,
not-to-touch list) lives in the root VPS reference doc on your Mac.

## What we're running on

- **2 vCPUs, no GPU, 8 GB RAM, Ubuntu 24.04**, `python3.12` system-wide.
- **Network policy:** SSH on `:22` is open; bind anything else to
  `127.0.0.1` and tunnel from your Mac if you need it.
- **System libs already present:** ffmpeg, tmux, git, sqlite3.
- **System libs we need to add (idempotent):** SDL2 family for pygame
  (PyTorch CPU wheel installs cleanly from pip).
  `vps_bootstrap.sh` handles this.

## Tuning this project for 2 vCPUs

The MPE env step is the bottleneck, not PyTorch. The wins, in order:

1. **`--num_threads 1`** on each Python process — keeps PyTorch from
   grabbing both cores per process. Especially important when running
   with `--workers > 1`.
2. **`--workers 2`** on `coevolve` — each worker process scores one
   mutant in parallel. With 15 mutants/phase, that's ~7 sequential
   batches instead of 15 → roughly 2× wall-clock on a 2-core box.
3. **`nice -n 19`** on every long process — the box stays responsive
   to SSH and other work even when training pegs both cores.

Measured on a 2-core sandbox (representative of the VPS):

| config | `--workers 1` | `--workers 2` | speedup |
|---|---:|---:|---:|
| 2 gen × 15 mutants | 14.9s | 11.8s | 1.26× |
| 2 gen × 30 mutants | 25.3s | 15.9s | 1.59× |

Speedup grows with mutant count because process-spawn + state_dict
pickling is a fixed per-mutant overhead. At 30 mutants/gen × 50 gens
expect ~7 minutes wall clock on the VPS.

## File layout on the VPS

```
/opt/digital-evo-rl/
├── src code (rsync'd from Mac)
├── venv/                       # CPU-only PyTorch + requirements.txt
├── runs/                       # output artifacts (run dirs, ckpts, reports)
│   ├── seed_ckpt/              # initial PPO ckpt for coevolve (created by run_worker.sh)
│   └── coevolve/               # gen_NNN/ + champion_tournament/ + plots/ + report.html
├── logs/                       # tail -F logs/coevolve.log
└── scripts/run_worker.sh       # tmux entrypoint
```

## Detached multi-day runs

```bash
# launch detached
ssh stepan-vps "cd /opt/digital-evo-rl && \
  tmux new-session -d -s evo './scripts/run_worker.sh'"

# stream the log from your Mac
ssh stepan-vps "tail -F /opt/digital-evo-rl/logs/coevolve.log"

# reattach interactively
ssh -t stepan-vps "tmux attach -t evo"

# kill cleanly
ssh stepan-vps "tmux kill-session -t evo"
```

`run_worker.sh` uses `coevolve --resume` so a crash → 60s sleep → restart
loses at most the half-finished generation. When `coevolve` exits 0
(all requested generations done), the worker writes `report.html` and
exits — no infinite spin.

## Pulling results back

```bash
# the only thing you care about for sharing
rsync -avh stepan-vps:/opt/digital-evo-rl/runs/coevolve/report.html ./

# or the whole run dir
rsync -avh stepan-vps:/opt/digital-evo-rl/runs/coevolve/ ./local_coev/
```

## Knobs

`run_worker.sh` reads env vars:

| var | default | what it does |
|---|---|---|
| `GENERATIONS` | 50 | total generations to run |
| `N_MUTANTS` | 20 | mutants per (gen, team) phase |
| `SIGMA` | 0.2 | Gaussian weight perturbation scale |
| `EVAL_EPS` | 2 | episodes per mutant evaluation |
| `N_OBSTACLES` | 2 | obstacles in the env |
| `MAX_CYCLES` | 100 | episode length |
| `WORKERS` | 2 | parallel mutant evaluators |
| `NUM_THREADS` | 1 | PyTorch threads per worker process |
| `HOF_K` | 0 | Hall of Fame size (0 = off; 3 is a good default) |
| `HOF_EVAL_EPS` | 1 | episodes per HoF opponent |
| `HOF_CURRENT_WEIGHT` | (unset) | 0..1 weight on current opponent (unset = equal weights — original behavior; 0.7-0.8 prevents promotion of mutants that lose to current but beat weak historicals) |
| `KIND` | `mpe` | env backend; set to `grid` to coevolve on the grid env |
| `WIDTH` `HEIGHT` `N_FOOD` | 20 / 20 / 0 | grid-env only |
| `CATCH_REWARD` `FOOD_REWARD` `STEP_COST` | 10 / 5 / 0.05 | **must match the seed ckpt's training reward params** |
| `SEED` | 2024 | RNG seed |
| `OUT` | `runs/coevolve` | output dir |

### Grid co-evolution on top of a long-run PPO ckpt

Once a `runs/long_v2/run_*/checkpoints/final` exists (the v2 big.yaml
PPO baseline), kick a coevolve loop seeded from it:

```bash
ssh stepan-vps "cd /opt/digital-evo-rl && \
  tmux new-session -d -s evo_grid \
    -e SEED_CKPT=runs/long_v2/run_20260521_201956/checkpoints/final \
    -e OUT=runs/coev_grid \
    -e LOG=logs/coev_grid.log \
    -e KIND=grid -e WIDTH=80 -e HEIGHT=80 -e N_FOOD=30 \
    -e CATCH_REWARD=25 -e FOOD_REWARD=15 -e STEP_COST=0.01 \
    -e N_PRED=12 -e N_PREY=12 -e N_OBSTACLES=30 -e MAX_CYCLES=300 \
    -e GENERATIONS=20 -e N_MUTANTS=15 -e SIGMA=0.15 -e EVAL_EPS=2 \
    -e HOF_K=3 -e HOF_CURRENT_WEIGHT=0.7 \
    ./scripts/run_worker.sh"
```

This continues the arms race from the v2 PPO baseline using mutation
+ HoF. Expect ~3-5 hours wall clock (20 gens × 15 mutants × 8 evals
per mutant × 300 steps on 2 vCPU). Output `champion_tournament/` will
show whether the evolved policies beat the v2 baseline.

Example: a quick smoke run
```bash
ssh stepan-vps "cd /opt/digital-evo-rl && \
  GENERATIONS=4 N_MUTANTS=10 ./scripts/run_worker.sh"
```

### Hall of Fame: turning cycling into stable progress

Plain co-evolution shows non-monotone champion fitness — predators
forget how to beat older prey strategies once new prey appear. `HOF_K`
fixes this: each mutant is evaluated against the current opponent plus
`HOF_K` randomly-sampled historical opponents, and the **mean** fitness
across all of them is what selection uses (or weighted via
`HOF_CURRENT_WEIGHT` — see Phase 8b notes in the run log).

Cost: each mutant runs `eval_eps + hof_k * hof_eval_eps` episodes
instead of `eval_eps`. With defaults `eval_eps=2`, `hof_k=3`,
`hof_eval_eps=1` that's 5× the rollout work — but the per-mutant fitness
estimate is far less noisy, so generations stabilise.

```bash
# Side-by-side comparison run: HoF off vs HoF on, same seed.
# Use tmux -e per-session to avoid the env-var-propagation gotcha
# (see Phase 8 docs in run_log.md).
ssh stepan-vps "cd /opt/digital-evo-rl && \
  tmux new-session -d -s evo_nohof \
    -e HOF_K=0 -e OUT=runs/coev_nohof -e LOG=logs/coev_nohof.log \
    ./scripts/run_worker.sh"
ssh stepan-vps "cd /opt/digital-evo-rl && \
  tmux new-session -d -s evo_hof \
    -e HOF_K=3 -e OUT=runs/coev_hof -e LOG=logs/coev_hof.log \
    ./scripts/run_worker.sh"
```

### Long pure-PPO training (Phase 10 large-scale)

For hours-long training on the bigger grid env (12v12, 80×80, hidden=256
— see `configs/big.yaml`), there's a separate tmux entrypoint:

```bash
ssh stepan-vps "cd /opt/digital-evo-rl && \
  tmux new-session -d -s long ./scripts/run_long.sh"
ssh stepan-vps "tail -F /opt/digital-evo-rl/logs/long.log"
```

Knobs (env vars, override before `tmux new-session`):

| var | default | notes |
|---|---|---|
| `CONFIG` | `configs/big.yaml` | swap in any other config |
| `SAVE_DIR` | `runs/long` | parent dir for the `run_YYYYMMDD_HHMMSS/` dir |
| `NUM_THREADS` | 2 | per-process torch threads; set to vCPU count |
| `EPISODES` | (config) | override `total_episodes` from config |
| `SEED` | (config) | override seed from config |

Each run produces a fresh `run_<ts>/` with periodic checkpoints under
`checkpoints/ep_NNNNNN/` (every 250 episodes in the big config) plus
`metrics.csv` and `plots/return.png`. If the process crashes mid-run,
those intermediate checkpoints still work for eval — but **run_cpu
does not yet have `--resume`**, so re-launching restarts episode 1 in
a new run dir. Picking up cleanly mid-run would need a dedicated patch.

## When something goes wrong

- **`ModuleNotFoundError: pygame`** → re-run `./venv/bin/pip install pygame` (the bootstrap script does this via `requirements.txt`).
- **`mat1 and mat2 shapes cannot be multiplied`** → the ckpt was trained
  with a different `n_obstacles` than the resume env. Each obstacle adds
  2 to obs dim. Match the env to the ckpt.
- **No output in `logs/coevolve.log`** → forgot `-u` on python (worker
  script has it; if you ran by hand, add it).
- **Box pegged, SSH laggy** → `WORKERS` or `NUM_THREADS` is too high.
  Cap to (cores - 1) for headroom.
