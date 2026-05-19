"""Replay a recorded trajectory to MP4.

Reads the NPZ written by :class:`TrajectoryRecorder`, re-runs the env with
the same seed and recorded actions, and captures rendered frames into a
video via ``imageio``.

Usage:

    python -m train.tools.replay --npz artifacts/<run>/trajectory.npz \\
        --out artifacts/<run>/replays/episode_1.mp4 --episode 1 \\
        --n_predators 2 --n_prey 2 --max_cycles 200
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

from envs.predator_prey import make_env, reset as env_reset, step as env_step


def _frames_for_episode(
    npz_path: str,
    episode: int,
    n_predators: int,
    n_prey: int,
    max_cycles: int,
    seed_override: int | None = None,
) -> List[np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    episodes = data["episodes"]
    actions = data["actions"]
    agents = [a.decode() if isinstance(a, bytes) else str(a) for a in data["agents"].tolist()]
    npz_seed = int(data["seed"][0])
    seed = seed_override if seed_override is not None else (42 if npz_seed < 0 else npz_seed)

    mask = episodes == episode
    if not mask.any():
        raise ValueError(f"Episode {episode} not found in {npz_path} (have {sorted(set(episodes.tolist()))})")
    ep_actions = actions[mask]

    env = make_env(
        n_predators=n_predators,
        n_prey=n_prey,
        max_cycles=max_cycles,
        seed=seed,
        render_mode="rgb_array",
    )
    obs = env_reset(env, seed=seed + episode)
    frames: List[np.ndarray] = []
    f0 = env.render()
    if f0 is not None:
        frames.append(np.asarray(f0))
    for row in ep_actions:
        if any(a not in obs for a in agents):
            break
        act = {a: int(row[j]) for j, a in enumerate(agents) if a in obs and int(row[j]) >= 0}
        obs, _rew, done_any, _info = env_step(env, act)
        f = env.render()
        if f is not None:
            frames.append(np.asarray(f))
        if done_any:
            break
    env.close()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to trajectory.npz")
    ap.add_argument("--out", required=True, help="Output MP4 path")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--n_predators", type=int, default=2)
    ap.add_argument("--n_prey", type=int, default=2)
    ap.add_argument("--max_cycles", type=int, default=200)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=None, help="Override env seed (otherwise read from NPZ)")
    args = ap.parse_args()

    frames = _frames_for_episode(
        args.npz,
        args.episode,
        n_predators=args.n_predators,
        n_prey=args.n_prey,
        max_cycles=args.max_cycles,
        seed_override=args.seed,
    )
    if not frames:
        raise SystemExit("No frames captured — env may not support rgb_array on this install.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
