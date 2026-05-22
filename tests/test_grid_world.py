"""Tests for the Phase 9 grid-world env."""
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from envs.grid_world import GridWorldEnv, N_ACTIONS, STAY, UP, DOWN, LEFT, RIGHT
from envs.predator_prey import make_env, reset as env_reset, step as env_step, team_of


def test_obs_dim_matches_formula():
    """obs_dim = 3 + 3*(n_agents-1) + 2*n_obstacles + 2*n_food, both teams equal."""
    env = GridWorldEnv(width=15, height=15, n_predators=3, n_prey=3,
                       n_obstacles=5, n_food=4, max_cycles=10, seed=0)
    obs = env.reset(seed=0)
    expected = 3 + 3 * (3 + 3 - 1) + 2 * 5 + 2 * 4
    for a, o in obs.items():
        assert o.shape == (expected,), f"{a}: {o.shape} != ({expected},)"


def test_agent_naming_matches_mpe():
    """team_of() must work without changes — names must follow MPE convention."""
    env = GridWorldEnv(width=10, height=10, n_predators=2, n_prey=2,
                       n_obstacles=0, n_food=0, max_cycles=5, seed=0)
    env.reset(seed=0)
    assert set(env.agents) == {"adversary_0", "adversary_1", "agent_0", "agent_1"}
    for a in env.agents:
        assert team_of(a) in ("predator", "prey")


def test_movement_clamps_to_bounds():
    """An agent trying to walk out-of-bounds stays at the boundary."""
    env = GridWorldEnv(width=5, height=5, n_predators=1, n_prey=1,
                       n_obstacles=0, n_food=0, max_cycles=20, seed=0)
    env.reset(seed=0)
    # Force a known position then try to walk left from x=0.
    env.positions["adversary_0"] = (0, 2)
    env.step({"adversary_0": LEFT, "agent_0": STAY})
    assert env.positions["adversary_0"] == (0, 2)  # blocked by wall, stays
    env.step({"adversary_0": RIGHT, "agent_0": STAY})
    assert env.positions["adversary_0"] == (1, 2)  # actually moves


def test_obstacles_are_impassable():
    env = GridWorldEnv(width=5, height=5, n_predators=1, n_prey=1,
                       n_obstacles=0, n_food=0, max_cycles=20, seed=0)
    env.reset(seed=0)
    env.obstacles = [(2, 2)]
    env.positions["adversary_0"] = (1, 2)
    env.step({"adversary_0": RIGHT, "agent_0": STAY})
    assert env.positions["adversary_0"] == (1, 2)   # blocked by obstacle


def test_catch_gives_reciprocal_rewards():
    """Predator on same cell as prey: +catch_reward / -catch_reward."""
    env = GridWorldEnv(width=10, height=10, n_predators=1, n_prey=1,
                       n_obstacles=0, n_food=0, max_cycles=10,
                       catch_reward=10.0, step_cost=0.0, seed=0)
    env.reset(seed=0)
    env.positions["adversary_0"] = (5, 5)
    env.positions["agent_0"]     = (5, 5)
    _obs, rew, _term, _trunc, _info = env.step({"adversary_0": STAY, "agent_0": STAY})
    assert rew["adversary_0"] == 10.0
    assert rew["agent_0"]     == -10.0


def test_prey_eats_food_and_food_respawns():
    """Eating food: +food_reward, food count unchanged after respawn."""
    env = GridWorldEnv(width=10, height=10, n_predators=1, n_prey=1,
                       n_obstacles=0, n_food=2, max_cycles=10,
                       food_reward=5.0, step_cost=0.0, seed=0)
    env.reset(seed=0)
    # Park prey on top of the first food.
    fx, fy = env.food[0]
    env.positions["agent_0"] = (fx, fy)
    initial_food = list(env.food)
    _obs, rew, _t, _tr, _i = env.step({"adversary_0": STAY, "agent_0": STAY})
    assert rew["agent_0"] == 5.0
    assert len(env.food) == 2, "food count must be conserved (respawn)"
    assert (fx, fy) not in env.food or env.food.count((fx, fy)) == 1, "old food slot freed"


def test_reset_with_same_seed_is_deterministic():
    env = GridWorldEnv(width=12, height=12, n_predators=2, n_prey=2,
                       n_obstacles=4, n_food=4, max_cycles=5, seed=0)
    obs1 = env.reset(seed=123)
    pos1 = dict(env.positions)
    food1 = list(env.food)
    obs2 = env.reset(seed=123)
    assert env.positions == pos1
    assert env.food == food1
    for a in obs1:
        np.testing.assert_array_equal(obs1[a], obs2[a])


def test_truncation_at_max_cycles():
    env = GridWorldEnv(width=10, height=10, n_predators=1, n_prey=1,
                       n_obstacles=0, n_food=0, max_cycles=3, seed=0)
    env.reset(seed=0)
    for step in range(3):
        _o, _r, term, trunc, _i = env.step({"adversary_0": STAY, "agent_0": STAY})
        if step < 2:
            assert not any(trunc.values())
        else:
            assert all(trunc.values())


def test_render_returns_rgb():
    env = GridWorldEnv(width=20, height=20, n_predators=2, n_prey=2,
                       n_obstacles=4, n_food=4, max_cycles=5, seed=0,
                       render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None and frame.ndim == 3 and frame.shape[2] == 3


def test_factory_dispatches_to_grid():
    env = make_env(kind="grid", width=10, height=10,
                   n_predators=1, n_prey=1, n_obstacles=2, n_food=2,
                   max_cycles=5, seed=0)
    obs = env_reset(env, seed=0)
    assert "adversary_0" in obs and "agent_0" in obs
    _obs, _rew, done, _info = env_step(env, {"adversary_0": STAY, "agent_0": STAY})
    assert isinstance(done, bool)


def test_factory_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown env kind"):
        make_env(kind="nope")


def test_run_cpu_end_to_end_on_grid(tmp_path: Path):
    """Full Phase 1-style loop must work with the new grid env."""
    import run_cpu
    # Write a tiny grid config inline so this test is self-contained.
    cfg = tmp_path / "grid_smoke.yaml"
    cfg.write_text("""
seed: 0
env:
  kind: grid
  width: 8
  height: 8
  n_predators: 1
  n_prey: 1
  n_obstacles: 2
  n_food: 2
  max_steps: 10
train:
  algo: ppo
  total_episodes: 2
  device: cpu
  hidden: 16
  batch_size: 128
  minibatch_size: 32
  update_epochs: 1
logging:
  save_dir: artifacts/
  plot_every: 1
recording: {enabled: true, sample_rate: 1}
checkpoint: {enabled: true, every: 0}
qd: {enabled: false}
novelty: {enabled: false}
""")
    out = Path(run_cpu.main(str(cfg), save_dir=str(tmp_path / "runs")))
    assert (out / "manifest.json").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "checkpoints" / "final" / "predator.pt").exists()
    import json
    m = json.loads((out / "manifest.json").read_text())
    assert m["env"]["kind"] == "grid"
    assert m["env"]["n_food"] == 2
