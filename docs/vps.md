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
| `SEED` | 2024 | RNG seed |
| `OUT` | `runs/coevolve` | output dir |

Example: a quick smoke run
```bash
ssh stepan-vps "cd /opt/digital-evo-rl && \
  GENERATIONS=4 N_MUTANTS=10 ./scripts/run_worker.sh"
```

## When something goes wrong

- **`ModuleNotFoundError: pygame`** → re-run `./venv/bin/pip install pygame` (the bootstrap script does this via `requirements.txt`).
- **`mat1 and mat2 shapes cannot be multiplied`** → the ckpt was trained
  with a different `n_obstacles` than the resume env. Each obstacle adds
  2 to obs dim. Match the env to the ckpt.
- **No output in `logs/coevolve.log`** → forgot `-u` on python (worker
  script has it; if you ran by hand, add it).
- **Box pegged, SSH laggy** → `WORKERS` or `NUM_THREADS` is too high.
  Cap to (cores - 1) for headroom.
