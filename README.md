# Digital Evolution + Multi-Agent RL

Python-first project to explore **predator–prey** dynamics with **multi-agent RL** (PPO, PyTorch) and **digital evolution** hooks (novelty/QD). Designed for **Gemini Advanced + Google Colab**, with **GitHub** as source of truth.

## Stack & Workflow
- **Authoring/Refactors:** Gemini Advanced
- **Runs & GPUs:** Google Colab (Pro/Pro+ optional)
- **Source of Truth:** GitHub (private), Issues/PRs/Releases
- **(Optional):** Jules for repo-wide PRs

## Project Status
- **Phase:** 1 (PPO baseline) + Phase 2 (evolution hooks) — both functional
- **Baseline PPO:** `run_cpu.py` — independent predator & prey policies on `simple_tag_v3`
- **Evolution:** `run_evolve.py` — co-evolving populations with Gaussian mutation,
  uniform/blend/layer crossover, tournament selection, optional novelty (k-NN) bonus
- **Tests:** `tests/test_smoke.py` (7 cases, run via `make test`)

## Structure
envs/ agents/ train/ tools/ configs/ artifacts/ tests/ docs/ .github/


## Conventions
- Branches: `main` (stable), `dev` (work), `feat/<topic>`
- Artifacts: `artifacts/run_YYYYMMDD_HHMM/` → {metrics.csv, plots/*.png, replays/*.mp4, manifest.json}
- Run Log: `docs/run_log.md` (one line per experiment)

## Quickstart (Colab)
```bash
!git clone https://github.com/<you>/digital-evo-rl.git
%cd digital-evo-rl
!pip install -r requirements.txt
!python run_cpu.py     --config configs/base.yaml --episodes 200    # PPO baseline
!python run_evolve.py  --config configs/base.yaml --generations 20   # GA / breeding

## Privacy

Repo stays private. Large artifacts live in Google Drive; link them from the Run Log or Releases.


### `configs/base.yaml`
```yaml
seed: 42
env:
  id: predator_prey_v0
  grid_size: 7
  n_predators: 2
  n_prey: 2
  max_steps: 200
train:
  algo: ppo
  total_episodes: 500
  gamma: 0.99
  lr: 3.0e-4
  batch_size: 2048
  update_epochs: 4
  clip_coef: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
logging:
  save_dir: artifacts/
  csv: true
  plots: true
  plot_every: 50
recording:
  enabled: false
  sample_rate: 1
