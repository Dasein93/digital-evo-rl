# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multi-agent **predator–prey** RL using **PettingZoo MPE `simple_tag_v3`** (adversaries=predators, good=prey) with a **PPO** baseline in PyTorch. Designed to be authored locally / via Gemini and run on **Google Colab** (CPU smoke tests locally, GPU on Colab). The project is early-stage — see `README.md` for phase status.

## Commands

```bash
make venv         # create .venv and install requirements.txt
make test         # pytest -q (currently placeholder)
make run_cpu      # python run_cpu.py --config configs/base.yaml
make clean        # remove __pycache__ / .pytest_cache

# Run a short smoke (e.g. 5 episodes to a custom dir)
python run_cpu.py --config configs/base.yaml --episodes 5 --save_dir artifacts/

# Run a single test
python -m pytest tests/test_smoke.py::test_placeholder -q
```

## Architecture

- **Entrypoint is `run_cpu.py` at the repo root** — the file `train/run_cpu.py` is an empty stub (1 byte placeholder); do not edit it expecting to change behavior. The Makefile's `run_cpu` target invokes the root file.
- `run_cpu.py` builds a PettingZoo **parallel_env** from `simple_tag_v3` and drives an episode loop that: flattens per-agent obs into a batch, calls `ppo.ac.step(...)` to act for all agents jointly, stores a single shared trajectory (rewards are averaged across agents per step), then calls `ppo.update(...)` at episode end.
- **PettingZoo API compat:** `_reset()` and `_step()` in `run_cpu.py` normalize both old (4-tuple) and new (5-tuple) PettingZoo return shapes. Preserve these wrappers when modifying the loop — Colab and local envs may pin different PettingZoo versions.
- **`train/ppo.py` is currently incomplete** — it ships only a patch fragment containing `_compute_returns` and `update`, but `run_cpu.py` imports `PPO`, `PPOConfig`, `flatten_obs`, and `set_seed` from it. Restoring/adding the class scaffolding (`PPO` with an `ac` actor-critic exposing `.actor`, `.critic`, and `.step(obs)->(a, logp, v)`, a `PPOConfig` dataclass, `flatten_obs(obs_dict)->(np.ndarray, agent_list)`, and `set_seed(seed)`) is the first thing to do before `run_cpu.py` will execute.
- **Config-driven**: `configs/base.yaml` is the canonical config. Anything tunable (seed, env sizing, PPO hyperparams, logging cadence) should be added there rather than hard-coded.
- **Outputs**: each run writes to `artifacts/run_YYYYMMDD_HHMM/` containing `metrics.csv` and `plots/return.png`. The whole `artifacts/` tree is gitignored — link large outputs from `docs/run_log.md` or Google Drive instead of committing them.

## Conventions

- **Branches**: `main` (stable), `dev` (work), `feat/<topic>`. PR template at `.github/PULL_REQUEST_TEMPLATE.md` expects a Colab-style validation snippet (`!pip install -r requirements.txt` then `!python run_cpu.py ...`).
- **Run log**: append one row per experiment to `docs/run_log.md` (date, commit, config, seed, artifact link, notes).
- **CI**: the workflow lives at the unusual nested path `.github/ISSUE_TEMPLATE/.github/workflows/ci.yml` (Python 3.10, runs `pytest -q` on pushes to `main`/`dev` and on PRs). If you move it to the conventional `.github/workflows/ci.yml`, verify the move on GitHub before relying on it.
- **Headless plotting**: matplotlib is forced to the `Agg` backend in `run_cpu.py` for Colab/server use — keep new plotting code headless-safe.
