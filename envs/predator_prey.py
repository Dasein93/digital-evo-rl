"""Predator–prey env factory.

Supports two backends, selected via the ``kind`` argument to ``make_env``:
- ``"mpe"`` (default): PettingZoo MPE ``simple_tag_v3``. Continuous-ish
  positions, fixed arena, baked-in reward shaping. What Phase 1-8 used.
- ``"grid"``: our own grid world (``envs.grid_world.GridWorldEnv``).
  Arbitrary width/height, configurable obstacles, **respawning food**
  that prey collect for positive reward. The richer ecosystem that
  Phase 9 unlocked.

Both backends name predators ``adversary_*`` and prey ``agent_*``, so
the ``team_of`` helper below works without changes.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

TEAM_PREDATOR = "predator"
TEAM_PREY = "prey"


def team_of(agent_name: str) -> str:
    return TEAM_PREDATOR if agent_name.startswith("adversary") else TEAM_PREY


def make_env(
    n_predators: int = 2,
    n_prey: int = 2,
    n_obstacles: int = 0,
    max_cycles: int = 200,
    seed: int = 42,
    render_mode: str | None = None,
    *,
    kind: str = "mpe",
    width: int = 20,
    height: int = 20,
    n_food: int = 0,
    catch_reward: float = 10.0,
    food_reward: float = 5.0,
    step_cost: float = 0.05,
) -> Any:
    """Build a predator-prey env. Backend selected via ``kind``.

    The MPE-specific args (``n_predators``, ``n_prey``, ``n_obstacles``,
    ``max_cycles``, ``seed``, ``render_mode``) are accepted by both
    backends. The grid-only keyword-only args (``width``, ``height``,
    ``n_food``, reward knobs) are silently ignored by the MPE backend so
    that callers can pass them without conditional branching.
    """
    if kind == "grid":
        from envs.grid_world import GridWorldEnv
        env = GridWorldEnv(
            width=width, height=height,
            n_predators=n_predators, n_prey=n_prey,
            n_obstacles=n_obstacles, n_food=n_food,
            max_cycles=max_cycles,
            catch_reward=catch_reward, food_reward=food_reward, step_cost=step_cost,
            render_mode=render_mode, seed=seed,
        )
        env.reset(seed=seed)
        return env

    if kind != "mpe":
        raise ValueError(f"unknown env kind={kind!r}; expected 'mpe' or 'grid'")

    # Default: PettingZoo MPE simple_tag_v3. Force headless SDL on Colab / CI.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from pettingzoo.mpe import simple_tag_v3

    env = simple_tag_v3.parallel_env(
        num_adversaries=n_predators,
        num_good=n_prey,
        num_obstacles=n_obstacles,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env


def reset(env, seed: int | None = None) -> Dict[str, Any]:
    """Compat wrapper — parallel_env.reset may return ``(obs, info)``."""
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def step(env, actions: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
    """Compat wrapper — PettingZoo parallel ``step`` is 4-tuple (old) or 5-tuple (new)."""
    out = env.step(actions)
    if isinstance(out, tuple) and len(out) == 5:
        next_obs, rewards, terminations, truncations, infos = out
        done_any = bool(any(terminations.values()) or any(truncations.values()))
        return next_obs, rewards, done_any, infos
    if isinstance(out, tuple) and len(out) == 4:
        next_obs, rewards, dones, infos = out
        done_any = bool(any(dones.values()))
        return next_obs, rewards, done_any, infos
    raise RuntimeError(f"Unexpected step() return: {type(out).__name__} len={len(out) if hasattr(out, '__len__') else '?'}")
