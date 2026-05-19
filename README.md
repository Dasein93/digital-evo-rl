# Digital Evolution + Multi-Agent RL

Python-first project to explore **predator–prey** dynamics with **multi-agent RL** (PPO, PyTorch) and **digital evolution** hooks (novelty/QD). Designed for **Gemini Advanced + Google Colab**, with **GitHub** as source of truth.

## Stack & Workflow
- **Authoring/Refactors:** Gemini Advanced
- **Runs & GPUs:** Google Colab (Pro/Pro+ optional)
- **Source of Truth:** GitHub (private), Issues/PRs/Releases
- **(Optional):** Jules for repo-wide PRs

## Project Status
- **Phase:** 1 — Baseline PPO + trajectory recorder/replay
- **Next Up:** Phase 2 — Novelty/QD hooks; GPU sweeps; richer envs
- **Last Run:** see `docs/run_log.md`

## Structure
```
envs/                MPE simple_tag wrapper + team helpers
agents/              (reserved for evolved/QD agent variants)
train/               ppo.py (shared-policy PPO per team)
train/tools/         recorder.py, replay.py
configs/             base.yaml (training), smoke.yaml (CI/dev)
artifacts/           per-run output dirs (gitignored)
tests/               pytest smoke suite (runs full loop on smoke.yaml)
docs/                run_log.md, prompts.md
.github/             workflows/ci.yml, PR + issue templates
```

## Quickstart

### Local
```bash
make venv        # optional venv
pip install -r requirements.txt
pip install pygame                    # MPE dep
SDL_VIDEODRIVER=dummy pytest -q       # ~3s smoke
python run_cpu.py --config configs/base.yaml --episodes 500
```

### Colab
```bash
!git clone https://github.com/<you>/digital-evo-rl.git
%cd digital-evo-rl
!pip install -r requirements.txt pygame
!python run_cpu.py --config configs/base.yaml --episodes 500
```

### Replay a recorded trajectory
Set `recording.enabled: true` in the config (or use `configs/smoke.yaml`), then:
```bash
python -m train.tools.replay \
  --npz artifacts/<run>/trajectory.npz \
  --out  artifacts/<run>/replays/episode_1.mp4 \
  --episode 1 --n_predators 2 --n_prey 2 --max_cycles 200
```

## Conventions
- Branches: `main` (stable), `dev` (work), `feat/<topic>`
- Artifacts: `artifacts/run_YYYYMMDD_HHMMSS/` → `{metrics.csv, plots/*.png, replays/*.mp4, trajectory.{jsonl,npz}, manifest.json}`
- Run Log: `docs/run_log.md` (one line per experiment)

## Privacy
Repo stays private. Large artifacts live in Google Drive; link them from the Run Log or Releases.
