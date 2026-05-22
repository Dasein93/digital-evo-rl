"""Co-evolving predator vs prey populations via genetic mutation + breeding.

Each team maintains a `Population` of policy genomes. Per generation:
  1) every individual rolls out a few episodes against a fixed opponent
     (defaults to opponent's current elite — Hall-of-Fame style),
  2) fitness = mean per-agent return, optionally + novelty bonus on final
     positions of own-team agents,
  3) survivors form the next generation by elitism + tournament selection
     + uniform crossover + Gaussian weight mutation.

This is a pure black-box ES/GA loop — no policy gradients. You can also seed
the populations from PPO-trained genomes (see `--warmstart`).
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, UTC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from run_cpu import (
    PREDATOR_PREFIX, PREY_PREFIX,
    is_predator, is_prey,
    make_env, _reset, _step, ensure_dir, _make_ppo,
)
from train.evolve import EvolveConfig, Population
from train.ppo import PPO, flatten_obs, set_seed


def _final_positions(obs: dict, agent_filter) -> np.ndarray:
    """Behavior characteristic: concatenated (vx, vy, x, y) of own-team agents
    at the final step. Compatible with simple_tag's standard obs layout
    [self_vel(2), self_pos(2), ...]."""
    pieces = []
    for name, vec in obs.items():
        if not agent_filter(name):
            continue
        v = np.asarray(vec, dtype=np.float32).ravel()
        pieces.append(v[:4] if v.size >= 4 else np.zeros(4, dtype=np.float32))
    if not pieces:
        return np.zeros(4, dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def _rollout(
    pred_ppo: PPO, prey_ppo: PPO,
    env, seed: int, max_steps: int,
    team: str,   # "predator" or "prey" — which side's reward/behavior we score
    episodes: int = 2,
):
    """Run `episodes` episodes; return (mean per-agent return for `team`, behavior)."""
    rets: list[float] = []
    bc: np.ndarray | None = None
    for k in range(episodes):
        obs = _reset(env, seed=seed + k)
        done = False
        team_ret = 0.0
        n_agents = 0
        last_obs = obs
        while not done:
            pred_arr, pred_now = flatten_obs(obs, agent_filter=is_predator)
            prey_arr, prey_now = flatten_obs(obs, agent_filter=is_prey)
            actions: dict = {}
            if pred_now:
                pa, _, _ = pred_ppo.act(pred_arr)
                for i, a in enumerate(pred_now):
                    actions[a] = int(pa[i])
            if prey_now:
                ea, _, _ = prey_ppo.act(prey_arr)
                for i, a in enumerate(prey_now):
                    actions[a] = int(ea[i])
            for a in obs.keys():
                actions.setdefault(a, 0)
            next_obs, rewards, done, _ = _step(env, actions)
            if team == "predator":
                team_ret += sum(rewards.get(a, 0.0) for a in pred_now)
                n_agents = max(n_agents, len(pred_now))
            else:
                team_ret += sum(rewards.get(a, 0.0) for a in prey_now)
                n_agents = max(n_agents, len(prey_now))
            last_obs = next_obs
            obs = next_obs
        rets.append(team_ret / max(1, n_agents))
        bc = _final_positions(last_obs, is_predator if team == "predator" else is_prey)
    return float(np.mean(rets)), (bc if bc is not None else np.zeros(4, dtype=np.float32))


def main(cfg_path: str, override_gens: int | None = None, save_dir: str | None = None,
         warmstart: str | None = None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    env_cfg = cfg.get("env", {})
    max_steps = int(env_cfg.get("max_steps", 200))
    n_pred = int(env_cfg.get("n_predators", 2))
    n_ev = int(env_cfg.get("n_prey", 2))

    evo = cfg.get("evolve", {}) or {}
    n_generations = int(override_gens or evo.get("generations", 20))
    eval_episodes = int(evo.get("eval_episodes", 2))
    ecfg = EvolveConfig(
        pop_size=int(evo.get("pop_size", 12)),
        elitism=int(evo.get("elitism", 2)),
        tournament_k=int(evo.get("tournament_k", 3)),
        mutation_sigma=float(evo.get("mutation_sigma", 0.05)),
        mutation_p=float(evo.get("mutation_p", 1.0)),
        crossover_mode=str(evo.get("crossover_mode", "uniform")),
        crossover_rate=float(evo.get("crossover_rate", 0.5)),
        novelty_coef=float(evo.get("novelty_coef", 0.0)),
        novelty_k=int(evo.get("novelty_k", 5)),
        novelty_capacity=int(evo.get("novelty_capacity", 500)),
    )

    run_id = datetime.now(UTC).strftime("evo_%Y%m%d_%H%M")
    out_dir = save_dir or cfg.get("logging", {}).get("save_dir", "artifacts/")
    out_dir = os.path.join(out_dir, run_id)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir); ensure_dir(plots_dir)

    env = make_env(n_predators=n_pred, n_prey=n_ev, max_cycles=max_steps, seed=seed)
    obs0 = _reset(env, seed=seed)
    pred_arr, pred_agents = flatten_obs(obs0, agent_filter=is_predator)
    prey_arr, prey_agents = flatten_obs(obs0, agent_filter=is_prey)

    pred_template = _make_ppo(pred_arr.shape[1], env.action_space(pred_agents[0]).n, cfg)
    prey_template = _make_ppo(prey_arr.shape[1], env.action_space(prey_agents[0]).n, cfg)

    # Optional warmstart from a PPO checkpoint (state_dict pair saved by run_cpu).
    seed_pred = seed_prey = None
    if warmstart and os.path.exists(warmstart):
        ck = torch.load(warmstart, map_location="cpu")
        seed_pred = ck.get("predator")
        seed_prey = ck.get("prey")
        if seed_pred is not None:
            pred_template.set_genome(seed_pred)
        if seed_prey is not None:
            prey_template.set_genome(seed_prey)
        print(f"Warmstarted from {warmstart}")

    pred_pop = Population(pred_template, cfg=ecfg, seed_genome=seed_pred)
    prey_pop = Population(prey_template, cfg=ecfg, seed_genome=seed_prey)

    metrics_path = os.path.join(out_dir, "evolve_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["gen",
                                "pred_fit_max", "pred_fit_mean", "pred_novelty",
                                "prey_fit_max", "prey_fit_mean", "prey_novelty"])

    pred_hist, prey_hist = [], []
    pred_elite = pred_template.get_genome()
    prey_elite = prey_template.get_genome()

    for gen in range(1, n_generations + 1):
        # Score predators against the current prey elite.
        prey_template.set_genome(prey_elite)
        def pred_rollout(ind_net: PPO):
            return _rollout(ind_net, prey_template, env, seed=seed + gen * 1000,
                            max_steps=max_steps, team="predator", episodes=eval_episodes)
        pred_stats = pred_pop.step_generation(pred_rollout)

        # Score prey against the current predator elite (use the just-updated pred elite).
        new_pred_elite, _ = pred_pop.best()
        pred_template.set_genome(new_pred_elite)
        def prey_rollout(ind_net: PPO):
            return _rollout(pred_template, ind_net, env, seed=seed + gen * 1000 + 1,
                            max_steps=max_steps, team="prey", episodes=eval_episodes)
        prey_stats = prey_pop.step_generation(prey_rollout)

        pred_elite, _ = pred_pop.best()
        prey_elite, _ = prey_pop.best()

        pred_hist.append(pred_stats["fit_mean"])
        prey_hist.append(prey_stats["fit_mean"])
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([gen,
                                    pred_stats["fit_max"], pred_stats["fit_mean"], pred_stats["novelty_mean"],
                                    prey_stats["fit_max"], prey_stats["fit_mean"], prey_stats["novelty_mean"]])

        print(f"gen {gen:3d} | pred fit max/mean: {pred_stats['fit_max']:+.3f}/{pred_stats['fit_mean']:+.3f} "
              f"| prey fit max/mean: {prey_stats['fit_max']:+.3f}/{prey_stats['fit_mean']:+.3f}")

    # Plot
    xs = np.arange(1, len(pred_hist) + 1)
    plt.figure()
    plt.plot(xs, pred_hist, label="predator mean fit")
    plt.plot(xs, prey_hist, label="prey mean fit")
    plt.xlabel("generation"); plt.ylabel("mean fitness"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "evolve.png")); plt.close()

    # Save best genomes
    torch.save({"predator": pred_elite, "prey": prey_elite},
               os.path.join(out_dir, "elites.pt"))
    print(f"Saved elites to: {os.path.join(out_dir, 'elites.pt')}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved plot to: {os.path.join(plots_dir, 'evolve.png')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--warmstart", type=str, default=None,
                    help="Path to a torch.save'd dict with keys {'predator','prey'} of state_dicts")
    args = ap.parse_args()
    main(args.config, args.generations, args.save_dir, args.warmstart)
