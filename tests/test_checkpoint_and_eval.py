"""Checkpoint round-trip + eval entrypoint smoke tests."""
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from agents.checkpoint import load_checkpoint, save_checkpoint
from train.ppo import PPO, PPOConfig, set_seed


def test_checkpoint_round_trip_preserves_outputs(tmp_path: Path):
    set_seed(0)
    cfg = PPOConfig(hidden=16, update_epochs=1, minibatch_size=4)
    p1 = PPO(obs_dim=8, act_dim=5, cfg=cfg)
    obs = torch.randn(4, 8)
    with torch.no_grad():
        logits_before = p1.ac.actor(obs).clone()

    save_checkpoint(str(tmp_path / "ckpt"), {"predator": p1, "prey": p1})
    loaded, manifest = load_checkpoint(str(tmp_path / "ckpt"))

    assert set(loaded.keys()) == {"predator", "prey"}
    assert manifest["teams"]["predator"]["obs_dim"] == 8
    assert manifest["teams"]["predator"]["act_dim"] == 5

    with torch.no_grad():
        logits_after = loaded["predator"].ac.actor(obs)
    torch.testing.assert_close(logits_before, logits_after)


def test_eval_runs_from_trained_checkpoint(tmp_path: Path):
    """End-to-end: train via run_cpu, load the resulting ckpt, eval greedy."""
    import run_cpu
    from train.tools.eval import evaluate

    out_dir = Path(run_cpu.main("configs/smoke.yaml", override_eps=2, save_dir=str(tmp_path)))
    ckpt = out_dir / "checkpoints" / "final"
    assert (ckpt / "manifest.json").exists()
    assert (ckpt / "predator.pt").exists()
    assert (ckpt / "prey.pt").exists()

    summary = evaluate(
        ckpt_dir=str(ckpt),
        out_dir=str(out_dir / "eval"),
        episodes=2,
        n_predators=1, n_prey=1,
        max_cycles=20, seed=999,
        record_first_n=1, fps=10,
    )
    assert summary["episodes"] == 2
    assert (out_dir / "eval" / "eval.csv").exists()
    assert (out_dir / "eval" / "eval_summary.json").exists()
    assert (out_dir / "eval" / "eval_episode_001.mp4").exists()
    s = json.loads((out_dir / "eval" / "eval_summary.json").read_text())
    assert s["episodes"] == 2
