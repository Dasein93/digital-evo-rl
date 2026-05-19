"""Predator–prey env wrapper around PettingZoo MPE ``simple_tag``.

The MPE env names adversaries (predators) ``adversary_*`` and good agents
(prey) ``agent_*`` and gives them different observation dims (14 vs 12),
so we expose a ``team_of`` helper to let callers route agents to the
right shared policy.
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
) -> Any:
    """Create a PettingZoo MPE ``simple_tag`` parallel env.

    Forces a headless SDL driver when no display is available so the env
    can be constructed on Colab / CI without an X server.
    """
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
