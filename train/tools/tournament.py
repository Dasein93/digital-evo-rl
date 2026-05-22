"""Cross-play tournament between checkpoints.

Pairs every predator checkpoint with every prey checkpoint, runs greedy
episodes, and writes a return matrix + heatmap. Useful for:
- comparing baseline vs novelty vs evolved policies
- finding "balanced" matchups (high predator return suggests the prey
  policy is exploitable, low suggests the predator is)

Usage:

    python -m train.tools.tournament \\
        --pred ckptA:label_a ckptB:label_b \\
        --prey ckptC:label_c ckptD:label_d \\
        --out  artifacts/tournament_demo \\
        --episodes 3 --n_predators 2 --n_prey 2 --max_cycles 100
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from agents.checkpoint import load_checkpoint
from envs.predator_prey import make_env, reset as env_reset, step as env_step, team_of, TEAM_PREDATOR, TEAM_PREY
from train.ppo import PPO, flatten_obs
from train.tools.plots import tournament_heatmap


def _split_by_team(agents, obs_list):
    by_team: Dict[str, List[int]] = {}
    for i, a in enumerate(agents):
        by_team.setdefault(team_of(a), []).append(i)
    return {
        team: (np.stack([obs_list[i] for i in idxs], axis=0), [agents[i] for i in idxs], idxs)
        for team, idxs in by_team.items()
    }


def _parse_entries(entries: List[str]) -> List[Tuple[str, str]]:
    out = []
    for e in entries:
        if ":" in e:
            path, label = e.rsplit(":", 1)
        else:
            path, label = e, os.path.basename(os.path.normpath(e))
        out.append((path, label))
    return out


def _match(
    pred_ppo: PPO,
    prey_ppo: PPO,
    n_predators: int,
    n_prey: int,
    max_cycles: int,
    episodes: int,
    seed: int,
    device: str,
    n_obstacles: int = 0,
    *,
    kind: str = "mpe",
    width: int = 20,
    height: int = 20,
    n_food: int = 0,
    catch_reward: float = 10.0,
    food_reward: float = 5.0,
    step_cost: float = 0.05,
) -> Tuple[float, float]:
    env = make_env(
        n_predators=n_predators, n_prey=n_prey, n_obstacles=n_obstacles,
        max_cycles=max_cycles, seed=seed,
        kind=kind, width=width, height=height, n_food=n_food,
        catch_reward=catch_reward, food_reward=food_reward, step_cost=step_cost,
    )
    ppos = {TEAM_PREDATOR: pred_ppo, TEAM_PREY: prey_ppo}
    pred_rets, prey_rets = [], []
    try:
        for ep in range(episodes):
            obs = env_reset(env, seed=seed + ep)
            per_agent_ret = {a: 0.0 for a in obs}
            done_any = False
            while not done_any:
                obs_list, step_agents = flatten_obs(obs)
                per_team = _split_by_team(step_agents, obs_list)
                acts: Dict[str, int] = {}
                with torch.no_grad():
                    for team, (team_obs, team_agents, _idxs) in per_team.items():
                        a_t, _logp, _v = ppos[team].ac.step(torch.from_numpy(team_obs).to(device), greedy=True)
                        for k, ag in enumerate(team_agents):
                            acts[ag] = int(a_t[k].item())
                obs, rewards, done_any, _ = env_step(env, acts)
                for ag in step_agents:
                    per_agent_ret[ag] += float(rewards[ag])
            pred_rets.append(float(np.mean([per_agent_ret[a] for a in per_agent_ret if team_of(a) == TEAM_PREDATOR])))
            prey_rets.append(float(np.mean([per_agent_ret[a] for a in per_agent_ret if team_of(a) == TEAM_PREY])))
    finally:
        env.close()
    return float(np.mean(pred_rets)), float(np.mean(prey_rets))


def tournament(
    pred_entries: List[Tuple[str, str]],
    prey_entries: List[Tuple[str, str]],
    out_dir: str,
    episodes: int,
    n_predators: int,
    n_prey: int,
    max_cycles: int,
    seed: int,
    device: str,
    n_obstacles: int = 0,
    *,
    kind: str = "mpe",
    width: int = 20,
    height: int = 20,
    n_food: int = 0,
    catch_reward: float = 10.0,
    food_reward: float = 5.0,
    step_cost: float = 0.05,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    # Pre-load every checkpoint just once.
    pred_ppos = {label: load_checkpoint(path, device=device)[0][TEAM_PREDATOR] for path, label in pred_entries}
    prey_ppos = {label: load_checkpoint(path, device=device)[0][TEAM_PREY] for path, label in prey_entries}

    rows, cols = list(pred_ppos.keys()), list(prey_ppos.keys())
    pred_mat = np.zeros((len(rows), len(cols)), dtype=np.float32)
    prey_mat = np.zeros((len(rows), len(cols)), dtype=np.float32)

    csv_path = os.path.join(out_dir, "tournament.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predator", "prey", "predator_return", "prey_return"])
        for i, pred_label in enumerate(rows):
            for j, prey_label in enumerate(cols):
                pr, py = _match(
                    pred_ppos[pred_label], prey_ppos[prey_label],
                    n_predators=n_predators, n_prey=n_prey, n_obstacles=n_obstacles,
                    max_cycles=max_cycles, episodes=episodes,
                    seed=seed + 1000 * i + j, device=device,
                    kind=kind, width=width, height=height, n_food=n_food,
                    catch_reward=catch_reward, food_reward=food_reward, step_cost=step_cost,
                )
                pred_mat[i, j] = pr
                prey_mat[i, j] = py
                w.writerow([pred_label, prey_label, pr, py])

    tournament_heatmap(
        pred_mat, rows, cols,
        out_path=os.path.join(out_dir, "tournament_predator.png"),
        title="Cross-play predator return (greedy)",
        cell_metric_name="predator return",
    )
    tournament_heatmap(
        prey_mat, rows, cols,
        out_path=os.path.join(out_dir, "tournament_prey.png"),
        title="Cross-play prey return (greedy)",
        cell_metric_name="prey return",
    )

    summary = {
        "predators": rows,
        "prey": cols,
        "episodes_per_match": episodes,
        "predator_return_matrix": pred_mat.tolist(),
        "prey_return_matrix": prey_mat.tolist(),
        "best_predator": rows[int(pred_mat.mean(axis=1).argmax())],
        "best_prey": cols[int(prey_mat.mean(axis=0).argmax())],
    }
    with open(os.path.join(out_dir, "tournament_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="ckpt_dir[:label] entries")
    ap.add_argument("--prey", nargs="+", required=True, help="ckpt_dir[:label] entries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--n_predators", type=int, default=2)
    ap.add_argument("--n_prey", type=int, default=2)
    ap.add_argument("--max_cycles", type=int, default=100)
    ap.add_argument("--n_obstacles", type=int, default=0)
    ap.add_argument("--seed", type=int, default=99999)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kind", default="mpe", choices=["mpe", "grid"],
                    help="env backend; must match what the ckpts were trained on")
    ap.add_argument("--width", type=int, default=20, help="grid env only")
    ap.add_argument("--height", type=int, default=20, help="grid env only")
    ap.add_argument("--n_food", type=int, default=0, help="grid env only")
    ap.add_argument("--catch_reward", type=float, default=10.0,
                    help="must match training; v2 big.yaml uses 25")
    ap.add_argument("--food_reward", type=float, default=5.0,
                    help="grid env only; v2 big.yaml uses 15")
    ap.add_argument("--step_cost", type=float, default=0.05,
                    help="must match training; v2 big.yaml uses 0.01")
    args = ap.parse_args()
    s = tournament(
        pred_entries=_parse_entries(args.pred),
        prey_entries=_parse_entries(args.prey),
        out_dir=args.out,
        episodes=args.episodes,
        n_predators=args.n_predators, n_prey=args.n_prey, n_obstacles=args.n_obstacles,
        max_cycles=args.max_cycles, seed=args.seed, device=args.device,
        kind=args.kind, width=args.width, height=args.height, n_food=args.n_food,
        catch_reward=args.catch_reward, food_reward=args.food_reward, step_cost=args.step_cost,
    )
    print(json.dumps({k: v for k, v in s.items() if "matrix" not in k}, indent=2))


if __name__ == "__main__":
    main()
