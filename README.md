# Digital Evolution + Multi-Agent RL

Python-first project to explore **predator–prey** dynamics with **multi-agent RL** (PPO, PyTorch) and **digital evolution** hooks (novelty/QD). Designed for **Gemini Advanced + Google Colab**, with **GitHub** as source of truth.

## Stack & Workflow
- **Authoring/Refactors:** Gemini Advanced
- **Runs & GPUs:** Google Colab (Pro/Pro+ optional)
- **Source of Truth:** GitHub (private), Issues/PRs/Releases
- **(Optional):** Jules for repo-wide PRs

## Project Status
- **Phase:** 0 — Foundation & Guardrails
- **Next Up:** Phase 1 — Baseline PPO (env + CPU smoke test)
- **Last Run:** _N/A_ (Phase 1 will populate)

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
# Phase 1 will add run_cpu.py

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
