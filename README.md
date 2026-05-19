# Digital Evolution + Multi-Agent RL

Python-first project to explore **predator–prey** dynamics with **multi-agent RL** (PPO, PyTorch) and **digital evolution** hooks (novelty/QD). Designed for **Gemini Advanced + Google Colab**, with **GitHub** as source of truth.

## Stack & Workflow
- **Authoring/Refactors:** Gemini Advanced
- **Runs & GPUs:** Google Colab (Pro/Pro+ optional)
- **Source of Truth:** GitHub (private), Issues/PRs/Releases
- **(Optional):** Jules for repo-wide PRs

## Project Status
- **Phase:** 6 — Co-evolutionary arms race + obstacles in env + self-contained HTML report bundler + real seed override
- **Next Up:** Colab GPU run with bigger generations / populations; per-cell elites tournament; curriculum scheduling
- **Last Run:** see `docs/run_log.md`

## Structure
```
envs/                MPE simple_tag wrapper + team helpers
agents/              novelty.py, qd.py (MAP-Elites), checkpoint.py
train/               ppo.py (shared-policy PPO per team)
train/tools/         recorder.py, replay.py, eval.py, evolve.py,
                     tournament.py, sweep.py, plots.py
configs/             base / smoke / preview / preview_novelty / preview_qd
artifacts/           per-run output dirs (gitignored)
tests/               pytest suite — full loop, novelty, ckpt/eval, QD,
                     evolve, tournament
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

### Greedy evaluation from a saved checkpoint
```bash
python -m train.tools.eval \
  --ckpt artifacts/<run>/checkpoints/final \
  --out  artifacts/<run>/eval \
  --episodes 10 --n_predators 2 --n_prey 2 --max_cycles 200 \
  --record_first_n 1   # writes eval_episode_001.mp4
```

### Digital-evolution hooks (Phase 2)
Set `novelty.enabled: true` in the config to compute a per-team k-NN
novelty score (over a 4D behavior characteristic — mean position, mean
speed, action entropy) and optionally inject it as an intrinsic reward
via `novelty.bonus_coef`. See `configs/preview_novelty.yaml`.

### Quality-Diversity archive (Phase 3)
With `qd.enabled: true`, the training loop periodically snapshots the
current policy and tries to insert it into a per-team **MAP-Elites**
archive — a grid over two BC dimensions, keeping the best-fitness
policy per cell. Plots: `plots/qd_archive_{predator,prey}.png`.
See `configs/preview_qd.yaml`.

### Mutation-based evolution (Phase 4)
Spawn weight-perturbed mutants from any checkpoint, evaluate each
greedily against the opposing team, and fill a MAP-Elites archive.
The best elite is packaged as a tournament-ready checkpoint:
```bash
python -m train.tools.evolve \
  --ckpt artifacts/<run>/checkpoints/final \
  --out  artifacts/<run>/evolve --team predator \
  --n_mutants 30 --sigma 0.25 --eval_eps 2 \
  --n_predators 2 --n_prey 2 --max_cycles 100
```

### Cross-play tournament + sweep (Phase 5)
Pair every predator checkpoint with every prey checkpoint:
```bash
python -m train.tools.tournament \
  --pred ckptA:base ckptB:nov ckptC:evolved \
  --prey ckptA:base ckptB:nov ckptC:evolved \
  --out  artifacts/tour --episodes 3 \
  --n_predators 2 --n_prey 2 --n_obstacles 0 --max_cycles 100
```
Multi-config sweep with per-config mean ± std plot (`--seeds` actually
varies the training seed; pass `--workers W` for parallel subprocesses):
```bash
python -m train.tools.sweep \
  --configs configs/preview.yaml configs/preview_novelty.yaml \
  --seeds 0 1 2 --out artifacts/sweep --workers 2
```

### Co-evolutionary arms race (Phase 6)
Alternating predator/prey mutation + selection across N generations. The
incumbent is always in the candidate pool, so champion fitness is
monotone within a phase. Each generation snapshots both champions as a
tournament-ready ckpt and the run finishes with a cross-generation
heatmap (gen-i predator vs gen-j prey) so the arms race is visible:
```bash
python -m train.tools.coevolve \
  --seed_ckpt artifacts/<run>/checkpoints/final \
  --out artifacts/coev --generations 4 --n_mutants 20 --sigma 0.2 \
  --n_predators 2 --n_prey 2 --n_obstacles 2 --max_cycles 100
```

### Self-contained HTML report
Bundles every plot + every MP4 + summary JSONs into one shareable file
(base64-embedded), no server needed:
```bash
python -m train.tools.report --dir artifacts/<run> --out report.html
```

## Conventions
- Branches: `main` (stable), `dev` (work), `feat/<topic>`
- Artifacts: `artifacts/run_YYYYMMDD_HHMMSS/` → `{metrics.csv, plots/*.png, replays/*.mp4, trajectory.{jsonl,npz}, manifest.json}`
- Run Log: `docs/run_log.md` (one line per experiment)

## Privacy
Repo stays private. Large artifacts live in Google Drive; link them from the Run Log or Releases.
