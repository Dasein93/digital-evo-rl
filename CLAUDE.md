# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multi-agent **predator–prey** RL on **PettingZoo MPE `simple_tag_v3`** (adversaries=predators, good=prey) with a shared-policy **PPO** baseline in PyTorch. Two policies are trained — one per team — because predators and prey have different observation dims (14 vs 12). Designed for CPU smoke locally and GPU sweeps on Google Colab.

## Commands

```bash
make venv         # create .venv and install requirements.txt
make test         # pytest -q (full Phase 1 loop runs under tests/)
make run_cpu      # python run_cpu.py --config configs/base.yaml
make clean        # remove __pycache__ / .pytest_cache

# Fast end-to-end smoke (2 episodes, ~2s) — exactly what CI runs
SDL_VIDEODRIVER=dummy python run_cpu.py --config configs/smoke.yaml --episodes 2 --save_dir /tmp/smoke

# Single test
SDL_VIDEODRIVER=dummy python -m pytest tests/test_smoke.py::test_run_cpu_end_to_end -q

# Replay a recorded trajectory to MP4
python -m train.tools.replay --npz <run>/trajectory.npz --out <run>/replays/ep1.mp4 \
    --episode 1 --n_predators 2 --n_prey 2 --max_cycles 200
```

`pygame` is a transitive dep (MPE renderer); `requirements.txt` does not pin it — install separately. On headless boxes (Colab/CI), export `SDL_VIDEODRIVER=dummy` before any env construction; `envs.predator_prey.make_env` sets it as a fallback via `setdefault`, but earlier pygame imports can lose the race.

## Architecture

### Entrypoint
`run_cpu.py` at the repo root. It builds **one `PPO` per team** (keyed by `TEAM_PREDATOR` / `TEAM_PREY` from `envs.predator_prey.team_of`), routes per-step observations through the matching policy, then calls `PPO.update` once per episode with per-agent trajectory buffers.

### Modules
- `envs/predator_prey.py` — `make_env`, `reset`, `step` compat wrappers (handle both 4- and 5-tuple PettingZoo `step` returns), and `team_of` to map agent names to teams. The MPE renderer needs pygame + an SDL driver, so the factory exports `SDL_VIDEODRIVER=dummy` if unset.
- `train/ppo.py` — `PPO`, `PPOConfig`, `ActorCritic`, plus the rollout helpers `flatten_obs`, `empty_rollout`, `append_step`, and `set_seed`. `flatten_obs` returns a **list of per-agent arrays** (not a stacked ndarray) because the two teams have different obs dims — do not change it to `np.stack`. `_gae` computes GAE-lambda per-agent and bootstraps with 0 at the trajectory end (fine for episodic rollouts; revisit if you add bootstrapped truncation). `PPO.__init__` accepts a `device` kwarg; `ActorCritic.step(obs, greedy=True)` uses argmax for eval.
- `agents/novelty.py` — 4D behavior characteristic `[mean_pos_x, mean_pos_y, mean_speed, action_entropy]` extracted from the first four entries of each obs (consistent across both teams). `NoveltyArchive` is a ring buffer; novelty is mean L2 distance to k nearest neighbors. Empty archive returns 0.0 by convention — the first episode is, by construction, not novel.
- `agents/checkpoint.py` — `save_checkpoint(dir, {team: PPO})` writes one `.pt` per team + `manifest.json` (obs/act dims, full `PPOConfig`). `load_checkpoint(dir)` reconstructs the PPOs without needing the original training script.
- `train/tools/recorder.py` — `TrajectoryRecorder` streams JSONL per step and dumps a compact NPZ on `close()` (or context exit). `sample_rate=k` keeps every k-th step.
- `train/tools/replay.py` — re-runs the env with the same seed + recorded actions, captures `rgb_array` frames, writes MP4 via `imageio[ffmpeg]`. Replay is deterministic only if env, seed, and action sequence match what was recorded.
- `train/tools/eval.py` — loads a checkpoint, runs greedy episodes (no PPO updates, no exploration), writes `eval.csv`/`eval_summary.json`/optional MP4. The function `evaluate(...)` is importable, the CLI is `python -m train.tools.eval`.
- `configs/base.yaml` — full training run with novelty/checkpoint blocks (novelty disabled by default). `configs/smoke.yaml` — tiny config used by tests. `configs/preview.yaml` (baseline) and `configs/preview_novelty.yaml` (`bonus_coef=0.05`) are paired demos used to render the README/PR comparison videos.

### Multi-agent rollout shape
Each transition is per-(step, agent). The per-team `PPO.update` receives `{agent_id: list_of_T_values}` dicts for `obs/acts/logps/rews/dones/vals`, computes GAE for each agent's contiguous trajectory, then concatenates across agents for shuffled minibatch SGD. If you add new buffer fields, add them to **both** `empty_rollout()` and `append_step()` in `train/ppo.py` and the per-agent slicing in `run_cpu.main`.

### Per-run artifacts
Each invocation writes to `artifacts/run_YYYYMMDD_HHMMSS/`:
- `manifest.json` — seed, device, env config, team obs dims, total_episodes, recording/novelty/checkpoint state
- `metrics.csv` — per-episode return (mean / predator / prey), step count, per-team `pg_loss`/`v_loss`/`entropy`, per-team novelty
- `plots/return.png`, `plots/novelty.png` (when novelty enabled)
- `trajectory.jsonl` + `trajectory.npz` — only when `recording.enabled: true`
- `checkpoints/final/{predator,prey}.pt` + `manifest.json` — written at end-of-training (or every N episodes via `checkpoint.every`)

The whole `artifacts/` tree is gitignored. Link large outputs from `docs/run_log.md` or Google Drive.

## Conventions
- Branches: `main` (stable), `dev` (work), `feat/<topic>`. PR template at `.github/PULL_REQUEST_TEMPLATE.md` expects a Colab-style validation snippet (`!pip install -r requirements.txt` then `!python run_cpu.py ...`).
- Run log: append one row per real experiment to `docs/run_log.md`.
- CI: `.github/workflows/ci.yml` — Python 3.11, installs SDL2 system libs + pygame, runs `pytest -q` (the full Phase 1 loop) on PRs and on `main`/`dev` pushes.
- Headless plotting: matplotlib is forced to `Agg` in `run_cpu.py` — keep new plotting code headless-safe.
