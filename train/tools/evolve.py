"""Mutation-based evolution that fills a MAP-Elites archive.

Loads a checkpoint as the seed. Spawns ``--n_mutants`` perturbed copies
(Gaussian weight noise, sigma controlled by ``--sigma``), evaluates each
greedily against the opponent team's existing policy for ``--eval_eps``
episodes, computes a behavior characteristic + mean fitness, and attempts
insertion into a per-team MAP-Elites archive.

Usage:

    python -m train.tools.evolve \\
        --ckpt artifacts/<run>/checkpoints/final \\
        --out  artifacts/<run>/evolve \\
        --team predator \\
        --n_mutants 30 --sigma 0.1 --eval_eps 2 \\
        --n_predators 2 --n_prey 2 --max_cycles 100

The opposing team's policy is held fixed (greedy) — first-cut "static
opponent" eval. The output is a populated MAP-Elites archive on disk
(plus heatmap PNG and a JSON summary), usable by ``tournament.py``.
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch

from agents.checkpoint import load_checkpoint, save_checkpoint
from agents.novelty import BehaviorCharacteristic
from agents.qd import MAPElitesArchive, MAPElitesConfig
from envs.predator_prey import make_env, reset as env_reset, step as env_step, team_of, TEAM_PREDATOR, TEAM_PREY
from train.ppo import PPO, flatten_obs, set_seed
from train.tools.plots import archive_heatmap


def _mutate_state_dict(sd: Dict[str, torch.Tensor], sigma: float, rng: torch.Generator) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if v.dtype.is_floating_point:
            noise = torch.empty_like(v).normal_(generator=rng) * sigma
            out[k] = v + noise
        else:
            out[k] = v.clone()
    return out


def _split_by_team(agents, obs_list):
    by_team: Dict[str, List[int]] = {}
    for i, a in enumerate(agents):
        by_team.setdefault(team_of(a), []).append(i)
    return {
        team: (np.stack([obs_list[i] for i in idxs], axis=0), [agents[i] for i in idxs], idxs)
        for team, idxs in by_team.items()
    }


def _rollout(
    env,
    ppos: Dict[str, PPO],
    n_episodes: int,
    seed: int,
    target_team: str,
    device: str,
) -> Tuple[float, np.ndarray]:
    """Returns (mean_fitness_for_target_team, mean_team_BC)."""
    fits: List[float] = []
    bcs: List[np.ndarray] = []
    for ep in range(n_episodes):
        obs = env_reset(env, seed=seed + ep)
        per_agent_obs: Dict[str, List[np.ndarray]] = {}
        per_agent_acts: Dict[str, List[int]] = {}
        per_agent_rews: Dict[str, float] = {}
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
            for i, ag in enumerate(step_agents):
                per_agent_obs.setdefault(ag, []).append(obs_list[i])
                per_agent_acts.setdefault(ag, []).append(int(acts[ag]))
            obs, rewards, done_any, _ = env_step(env, acts)
            for ag in step_agents:
                per_agent_rews[ag] = per_agent_rews.get(ag, 0.0) + float(rewards[ag])

        target_agents = [a for a in per_agent_rews if team_of(a) == target_team]
        if not target_agents:
            continue
        fits.append(float(np.mean([per_agent_rews[a] for a in target_agents])))
        per_agent_bc = {
            a: BehaviorCharacteristic.from_agent(per_agent_obs.get(a, []), per_agent_acts.get(a, []), n_actions=5)
            for a in target_agents
        }
        bcs.append(BehaviorCharacteristic.team(per_agent_bc))
    if not fits:
        return 0.0, np.zeros(4, dtype=np.float32)
    return float(np.mean(fits)), np.mean(np.stack(bcs, axis=0), axis=0).astype(np.float32)


def evolve(
    ckpt_dir: str,
    out_dir: str,
    target_team: str,
    n_mutants: int,
    sigma: float,
    eval_eps: int,
    n_predators: int,
    n_prey: int,
    max_cycles: int,
    seed: int,
    device: str,
    qd_cfg: MAPElitesConfig | None = None,
) -> Dict:
    set_seed(seed)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    seed_ppos, manifest = load_checkpoint(ckpt_dir, device=device)
    assert target_team in seed_ppos, f"checkpoint has no {target_team} team"

    os.makedirs(out_dir, exist_ok=True)
    archive_dir = os.path.join(out_dir, "qd", target_team)
    archive = MAPElitesArchive(qd_cfg or MAPElitesConfig(), save_dir=archive_dir)

    env = make_env(n_predators=n_predators, n_prey=n_prey, max_cycles=max_cycles, seed=seed)

    seed_sd = deepcopy(seed_ppos[target_team].ac.state_dict())
    summary_rows: List[Dict] = []
    try:
        # Always evaluate the unperturbed seed for reference.
        fit, bc = _rollout(env, seed_ppos, eval_eps, seed=seed * 13, target_team=target_team, device=device)
        inserted = archive.try_insert(bc, fit, deepcopy(seed_sd), label="seed")
        summary_rows.append({"label": "seed", "fitness": fit, "bc": bc.tolist(), "inserted": inserted})

        for k in range(n_mutants):
            sd_k = _mutate_state_dict(seed_sd, sigma=sigma, rng=rng)
            seed_ppos[target_team].ac.load_state_dict(sd_k)
            fit_k, bc_k = _rollout(env, seed_ppos, eval_eps, seed=seed * 13 + k + 1, target_team=target_team, device=device)
            inserted = archive.try_insert(bc_k, fit_k, deepcopy(sd_k), label=f"mut_{k:04d}")
            summary_rows.append({"label": f"mut_{k:04d}", "fitness": fit_k, "bc": bc_k.tolist(), "inserted": inserted})
    finally:
        env.close()

    archive.save_manifest()
    # Heatmap of the resulting archive.
    bc_names = {0: "mean_x", 1: "mean_y", 2: "mean_speed", 3: "action_entropy"}
    i_dim, j_dim = archive.cfg.bc_dims
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    archive_heatmap(
        archive.fitness_grid(),
        out_path=os.path.join(plots_dir, f"qd_archive_{target_team}.png"),
        title=f"MAP-Elites (evolve): {target_team} cov={archive.coverage():.0%} best={archive.best_fitness():.1f}",
        xlabel=bc_names.get(i_dim, f"BC[{i_dim}]"),
        ylabel=bc_names.get(j_dim, f"BC[{j_dim}]"),
        bc_bounds=archive.cfg.bc_bounds,
    )

    # Package the best elite as a tournament-ready checkpoint dir: the
    # evolved team gets the best mutant's weights, the opposing team is
    # copied unchanged from the seed checkpoint.
    best = archive.best_elite()
    best_ckpt_dir = None
    if best is not None:
        best_ckpt_dir = os.path.join(out_dir, "best_checkpoint")
        best_sd = torch.load(best["path"], map_location=device, weights_only=True)
        seed_ppos[target_team].ac.load_state_dict(best_sd)
        save_checkpoint(
            best_ckpt_dir,
            seed_ppos,
            extra={
                "from": ckpt_dir,
                "team_evolved": target_team,
                "best_label": best["label"],
                "best_fitness": best["fitness"],
                "best_bc": best["bc"],
            },
        )

    summary = {
        "seed_ckpt": ckpt_dir,
        "team": target_team,
        "n_mutants": n_mutants,
        "sigma": sigma,
        "eval_eps": eval_eps,
        "coverage": archive.coverage(),
        "best_fitness": archive.best_fitness(),
        "n_inserted": sum(1 for r in summary_rows if r["inserted"]),
        "best_checkpoint": best_ckpt_dir,
        "rows": summary_rows,
    }
    with open(os.path.join(out_dir, "evolve_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--team", default=TEAM_PREDATOR, choices=[TEAM_PREDATOR, TEAM_PREY])
    ap.add_argument("--n_mutants", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--eval_eps", type=int, default=2)
    ap.add_argument("--n_predators", type=int, default=2)
    ap.add_argument("--n_prey", type=int, default=2)
    ap.add_argument("--max_cycles", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    s = evolve(
        ckpt_dir=args.ckpt, out_dir=args.out, target_team=args.team,
        n_mutants=args.n_mutants, sigma=args.sigma, eval_eps=args.eval_eps,
        n_predators=args.n_predators, n_prey=args.n_prey, max_cycles=args.max_cycles,
        seed=args.seed, device=args.device,
    )
    print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
