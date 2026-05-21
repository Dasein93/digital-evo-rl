"""Evaluate a saved checkpoint with greedy policies.

No gradients, no exploration noise, no PPO updates — just runs ``N``
episodes of MPE ``simple_tag`` using the saved weights and writes:

    - ``eval.csv`` : per-episode return per team
    - ``eval_summary.json`` : aggregate stats
    - (optional) ``eval_episode_<k>.mp4`` for the first ``--record_first_n`` episodes

Usage:

    python -m train.tools.eval \\
        --ckpt artifacts/<run>/checkpoints \\
        --out  artifacts/<run>/eval \\
        --episodes 20 --n_predators 2 --n_prey 2 --max_cycles 200 \\
        --record_first_n 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch

from agents.checkpoint import load_checkpoint
from envs.predator_prey import make_env, reset as env_reset, step as env_step, team_of
from train.ppo import flatten_obs


def _split_by_team(agents, obs_list):
    by_team: Dict[str, List[int]] = {}
    for i, a in enumerate(agents):
        by_team.setdefault(team_of(a), []).append(i)
    return {
        team: (np.stack([obs_list[i] for i in idxs], axis=0), [agents[i] for i in idxs], idxs)
        for team, idxs in by_team.items()
    }


def evaluate(
    ckpt_dir: str,
    out_dir: str,
    episodes: int,
    n_predators: int,
    n_prey: int,
    max_cycles: int,
    seed: int,
    record_first_n: int = 0,
    fps: int = 15,
    device: str = "cpu",
    n_obstacles: int = 0,
    *,
    kind: str = "mpe",
    width: int = 20,
    height: int = 20,
    n_food: int = 0,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    ppos, manifest = load_checkpoint(ckpt_dir, device=device)
    csv_path = os.path.join(out_dir, "eval.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "return_mean", "predator_return", "prey_return", "steps"])

    render_mode = "rgb_array" if record_first_n > 0 else None
    env = make_env(
        n_predators=n_predators, n_prey=n_prey, n_obstacles=n_obstacles,
        max_cycles=max_cycles, seed=seed, render_mode=render_mode,
        kind=kind, width=width, height=height, n_food=n_food,
    )

    all_mean, all_pred, all_prey = [], [], []
    try:
        for ep in range(1, episodes + 1):
            obs = env_reset(env, seed=seed + ep)
            frames: List[np.ndarray] = []
            if ep <= record_first_n:
                f0 = env.render()
                if f0 is not None:
                    frames.append(np.asarray(f0))
            ep_ret = {a: 0.0 for a in obs}
            done_any = False
            steps = 0
            while not done_any:
                obs_list, step_agents = flatten_obs(obs)
                per_team = _split_by_team(step_agents, obs_list)
                acts: Dict[str, int] = {}
                with torch.no_grad():
                    for team, (team_obs, team_agents, _idxs) in per_team.items():
                        if team not in ppos:
                            for ag in team_agents:
                                acts[ag] = 0
                            continue
                        a_t, _logp, _v = ppos[team].ac.step(torch.from_numpy(team_obs).to(device), greedy=True)
                        for k, ag in enumerate(team_agents):
                            acts[ag] = int(a_t[k].item())
                obs, rewards, done_any, _info = env_step(env, acts)
                if ep <= record_first_n:
                    f = env.render()
                    if f is not None:
                        frames.append(np.asarray(f))
                for a in step_agents:
                    ep_ret[a] += float(rewards[a])
                steps += 1

            pred_ret = float(np.mean([ep_ret[a] for a in ep_ret if team_of(a) == "predator"] or [0.0]))
            prey_ret = float(np.mean([ep_ret[a] for a in ep_ret if team_of(a) == "prey"] or [0.0]))
            mean_ret = float(np.mean(list(ep_ret.values())))
            all_mean.append(mean_ret); all_pred.append(pred_ret); all_prey.append(prey_ret)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, mean_ret, pred_ret, prey_ret, steps])

            if ep <= record_first_n and frames:
                import imageio.v2 as imageio
                mp4_path = os.path.join(out_dir, f"eval_episode_{ep:03d}.mp4")
                imageio.mimsave(mp4_path, frames, fps=fps)
                print(f"Wrote {len(frames)} frames -> {mp4_path}")
    finally:
        env.close()

    summary = {
        "episodes": episodes,
        "return_mean": float(np.mean(all_mean)),
        "predator_return_mean": float(np.mean(all_pred)),
        "prey_return_mean": float(np.mean(all_prey)),
        "return_std": float(np.std(all_mean)),
        "manifest": manifest,
    }
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--n_predators", type=int, default=2)
    ap.add_argument("--n_prey", type=int, default=2)
    ap.add_argument("--max_cycles", type=int, default=200)
    ap.add_argument("--n_obstacles", type=int, default=0)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--record_first_n", type=int, default=0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kind", default="mpe", choices=["mpe", "grid"],
                    help="env backend; must match what the ckpt was trained on")
    ap.add_argument("--width", type=int, default=20, help="grid env only")
    ap.add_argument("--height", type=int, default=20, help="grid env only")
    ap.add_argument("--n_food", type=int, default=0, help="grid env only")
    args = ap.parse_args()
    summary = evaluate(
        ckpt_dir=args.ckpt, out_dir=args.out,
        episodes=args.episodes,
        n_predators=args.n_predators, n_prey=args.n_prey,
        max_cycles=args.max_cycles, seed=args.seed,
        record_first_n=args.record_first_n, fps=args.fps, device=args.device,
        n_obstacles=args.n_obstacles,
        kind=args.kind, width=args.width, height=args.height, n_food=args.n_food,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
