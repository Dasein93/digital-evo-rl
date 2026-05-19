"""End-to-end smoke tests for Phase 1.

Kept fast (~30s on CPU) so CI can run them on every PR. Anything heavier
should live under tests/integration/.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from envs.predator_prey import make_env, reset as env_reset, step as env_step, team_of, TEAM_PREDATOR, TEAM_PREY
from train.ppo import PPO, PPOConfig, set_seed


def test_set_seed_is_deterministic():
    set_seed(123)
    a = (torch.randn(3).tolist(), np.random.rand(3).tolist())
    set_seed(123)
    b = (torch.randn(3).tolist(), np.random.rand(3).tolist())
    assert a == b


def test_team_of_partitions_agents():
    assert team_of("adversary_0") == TEAM_PREDATOR
    assert team_of("agent_0") == TEAM_PREY


def test_env_reset_and_step_shapes():
    env = make_env(n_predators=1, n_prey=1, max_cycles=5, seed=0)
    obs = env_reset(env, seed=0)
    assert set(obs.keys()) == {"adversary_0", "agent_0"}
    acts = {a: 0 for a in obs}
    next_obs, rewards, done_any, _ = env_step(env, acts)
    assert set(next_obs.keys()) == set(obs.keys())
    assert set(rewards.keys()) == set(obs.keys())
    assert isinstance(done_any, bool)
    env.close()


def test_ppo_update_runs_on_tiny_rollout():
    set_seed(0)
    ppo = PPO(obs_dim=6, act_dim=5, cfg=PPOConfig(update_epochs=1, minibatch_size=8, hidden=16))
    n = 16
    obs = [np.random.randn(6).astype(np.float32) for _ in range(n)]
    acts = [int(np.random.randint(0, 5)) for _ in range(n)]
    logps = [-1.6 for _ in range(n)]
    rews = [float(np.random.randn()) for _ in range(n)]
    dones = [0.0] * (n - 1) + [1.0]
    vals = [0.0 for _ in range(n)]
    info = ppo.update(
        {"a0": obs}, {"a0": acts}, {"a0": logps},
        {"a0": rews}, {"a0": dones}, {"a0": vals},
    )
    assert info["n"] == n
    assert np.isfinite(info["pg_loss"])
    assert np.isfinite(info["v_loss"])
    assert np.isfinite(info["entropy"])


def test_ppo_gae_handles_terminal_correctly():
    """When lambda=1 and the final transition is terminal, returns are plain discounted G_t."""
    rews = [1.0, 1.0, 1.0]
    dones = [0.0, 0.0, 1.0]
    vals = [0.0, 0.0, 0.0]
    adv, ret = PPO._gae(rews, dones, vals, gamma=0.5, lam=1.0)
    # ret = G_t with gamma=0.5: [1 + 0.5*(1 + 0.5*1), 1 + 0.5*1, 1] = [1.75, 1.5, 1.0]
    np.testing.assert_allclose(ret, [1.75, 1.5, 1.0], rtol=1e-5)


def test_run_cpu_end_to_end(tmp_path: Path):
    """Full Phase 1 loop on the smoke config; verifies all artifacts land."""
    import run_cpu
    out_dir = run_cpu.main("configs/smoke.yaml", override_eps=2, save_dir=str(tmp_path))
    out = Path(out_dir)
    assert (out / "metrics.csv").exists()
    assert (out / "manifest.json").exists()
    assert (out / "plots" / "return.png").exists()
    # recording.enabled = true in smoke.yaml
    assert (out / "trajectory.jsonl").exists()
    assert (out / "trajectory.npz").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["total_episodes"] == 2
    rows = (out / "metrics.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 2  # header + 2 episodes


def test_replay_renders_frames(tmp_path: Path):
    """Recorder + replay pipeline produces non-empty rendered frames."""
    import run_cpu
    from train.tools.replay import _frames_for_episode

    out_dir = Path(run_cpu.main("configs/smoke.yaml", override_eps=2, save_dir=str(tmp_path)))
    npz = out_dir / "trajectory.npz"
    assert npz.exists()
    frames = _frames_for_episode(
        str(npz), episode=1, n_predators=1, n_prey=1, max_cycles=20,
    )
    assert len(frames) > 0
    assert frames[0].ndim == 3 and frames[0].shape[2] == 3
