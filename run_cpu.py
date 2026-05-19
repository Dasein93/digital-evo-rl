"""Phase 1 CPU smoke test for the predator-prey baseline.

Trains two shared-policy PPO agents — one for predators, one for prey —
on PettingZoo MPE ``simple_tag_v3``, logs CSV metrics and a return plot
per run dir, and (optionally) records trajectories for later replay.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, UTC
from typing import Dict, List

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.predator_prey import make_env, reset as env_reset, step as env_step, team_of, TEAM_PREDATOR, TEAM_PREY
from train.ppo import PPO, PPOConfig, flatten_obs, set_seed, empty_rollout, append_step
from train.tools.recorder import TrajectoryRecorder


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_ppos(train_cfg: dict, obs_dims: Dict[str, int], act_dim: int) -> Dict[str, PPO]:
    pcfg = PPOConfig(
        lr=float(train_cfg.get("lr", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        clip_coef=float(train_cfg.get("clip_coef", 0.2)),
        ent_coef=float(train_cfg.get("ent_coef", 0.01)),
        vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        update_epochs=int(train_cfg.get("update_epochs", 4)),
        batch_size=int(train_cfg.get("batch_size", 2048)),
        minibatch_size=int(train_cfg.get("minibatch_size", 256)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        hidden=int(train_cfg.get("hidden", 128)),
    )
    return {team: PPO(obs_dims[team], act_dim, pcfg) for team in obs_dims}


def _split_by_team(agents: List[str], obs_list: List[np.ndarray]) -> Dict[str, tuple]:
    by_team: Dict[str, List[int]] = {TEAM_PREDATOR: [], TEAM_PREY: []}
    for i, a in enumerate(agents):
        by_team[team_of(a)].append(i)
    out = {}
    for team, idxs in by_team.items():
        if not idxs:
            continue
        out[team] = (
            np.stack([obs_list[i] for i in idxs], axis=0),
            [agents[i] for i in idxs],
            idxs,
        )
    return out


def main(cfg_path: str, override_eps: int | None = None, save_dir: str | None = None) -> str:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    env_cfg = cfg.get("env", {})
    max_steps = int(env_cfg.get("max_steps", 200))
    n_pred = int(env_cfg.get("n_predators", 2))
    n_prey = int(env_cfg.get("n_prey", 2))

    total_episodes = int(override_eps or cfg.get("train", {}).get("total_episodes", 500))

    rec_cfg = cfg.get("recording", {}) or {}
    rec_enabled = bool(rec_cfg.get("enabled", False))
    rec_sample_rate = int(rec_cfg.get("sample_rate", 1))

    run_id = datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    out_root = save_dir or cfg.get("logging", {}).get("save_dir", "artifacts/")
    out_dir = os.path.join(out_root, run_id)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    env = make_env(n_predators=n_pred, n_prey=n_prey, max_cycles=max_steps, seed=seed)
    obs0 = env_reset(env, seed=seed)
    obs_list, agents = flatten_obs(obs0)
    act_dim = int(env.action_space(agents[0]).n)

    team_dims: Dict[str, int] = {}
    for i, a in enumerate(agents):
        team_dims.setdefault(team_of(a), obs_list[i].shape[0])

    ppos = _build_ppos(cfg.get("train", {}) or {}, team_dims, act_dim)

    manifest = {
        "run_id": run_id,
        "seed": seed,
        "config_path": os.path.abspath(cfg_path),
        "env": {"n_predators": n_pred, "n_prey": n_prey, "max_steps": max_steps, "act_dim": act_dim},
        "team_obs_dims": team_dims,
        "total_episodes": total_episodes,
        "recording": {"enabled": rec_enabled, "sample_rate": rec_sample_rate},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "episode", "return_mean",
                "predator_return", "prey_return",
                "steps",
                "pred_pg_loss", "pred_v_loss", "pred_entropy",
                "prey_pg_loss", "prey_v_loss", "prey_entropy",
            ]
        )

    recorder = None
    if rec_enabled:
        ensure_dir(os.path.join(out_dir, "replays"))
        recorder = TrajectoryRecorder(
            out_dir=out_dir,
            sample_rate=rec_sample_rate,
            also_npz=True,
            seed=seed,
            env_meta={"n_predators": n_pred, "n_prey": n_prey, "max_steps": max_steps},
        )

    returns_mean: List[float] = []
    returns_pred: List[float] = []
    returns_prey: List[float] = []

    try:
        for ep in range(1, total_episodes + 1):
            obs = env_reset(env, seed=seed + ep)
            buf = empty_rollout()
            ep_ret_per_agent: Dict[str, float] = {a: 0.0 for a in agents}
            done_any = False
            steps = 0

            while not done_any:
                obs_list, step_agents = flatten_obs(obs)
                per_team = _split_by_team(step_agents, obs_list)

                logps_all = np.zeros(len(step_agents), dtype=np.float32)
                vals_all = np.zeros(len(step_agents), dtype=np.float32)
                acts_dict: Dict[str, int] = {}
                with torch.no_grad():
                    for team, (team_obs, team_agents, idxs) in per_team.items():
                        a_t, logp_t, v_t = ppos[team].ac.step(torch.from_numpy(team_obs))
                        for k, agent in enumerate(team_agents):
                            acts_dict[agent] = int(a_t[k].item())
                            logps_all[idxs[k]] = float(logp_t[k].item())
                            vals_all[idxs[k]] = float(v_t[k].item())

                next_obs, rewards, done_any, _info = env_step(env, acts_dict)
                append_step(buf, step_agents, obs_list, acts_dict, logps_all, vals_all, rewards, done_any)
                if recorder is not None:
                    recorder.record(ep, obs, acts_dict, rewards, done_any)

                for a in step_agents:
                    ep_ret_per_agent[a] += float(rewards[a])
                obs = next_obs
                steps += 1

            team_updates = {TEAM_PREDATOR: None, TEAM_PREY: None}
            for team in ppos:
                team_agents = [a for a in agents if team_of(a) == team]
                if not team_agents:
                    continue
                team_updates[team] = ppos[team].update(
                    {a: buf["obs"].get(a, []) for a in team_agents},
                    {a: buf["acts"].get(a, []) for a in team_agents},
                    {a: buf["logps"].get(a, []) for a in team_agents},
                    {a: buf["rews"].get(a, []) for a in team_agents},
                    {a: buf["dones"].get(a, []) for a in team_agents},
                    {a: buf["vals"].get(a, []) for a in team_agents},
                )

            pred_ret = float(np.mean([ep_ret_per_agent[a] for a in agents if team_of(a) == TEAM_PREDATOR] or [0.0]))
            prey_ret = float(np.mean([ep_ret_per_agent[a] for a in agents if team_of(a) == TEAM_PREY] or [0.0]))
            mean_ret = float(np.mean(list(ep_ret_per_agent.values())))
            returns_mean.append(mean_ret)
            returns_pred.append(pred_ret)
            returns_prey.append(prey_ret)

            pu = team_updates.get(TEAM_PREDATOR) or {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
            yu = team_updates.get(TEAM_PREY) or {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        ep, mean_ret, pred_ret, prey_ret, steps,
                        pu["pg_loss"], pu["v_loss"], pu["entropy"],
                        yu["pg_loss"], yu["v_loss"], yu["entropy"],
                    ]
                )

            plot_every = int(cfg.get("logging", {}).get("plot_every", 50))
            if ep % plot_every == 0 or ep == total_episodes:
                _plot(returns_mean, returns_pred, returns_prey, plots_dir)
                tail = returns_mean[-min(10, len(returns_mean)):]
                print(
                    f"[{ep}/{total_episodes}] mean={np.mean(tail):.3f} "
                    f"pred={np.mean(returns_pred[-10:]):.3f} prey={np.mean(returns_prey[-10:]):.3f} "
                    f"steps={steps}"
                )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved plot   -> {os.path.join(plots_dir, 'return.png')}")
    print(f"Run dir      -> {out_dir}")
    return out_dir


def _plot(returns_mean: List[float], returns_pred: List[float], returns_prey: List[float], plots_dir: str) -> None:
    xs = np.arange(1, len(returns_mean) + 1)
    window = min(50, len(returns_mean))
    plt.figure()
    plt.plot(xs, returns_mean, label="mean", alpha=0.6)
    plt.plot(xs, returns_pred, label="predator", alpha=0.6)
    plt.plot(xs, returns_prey, label="prey", alpha=0.6)
    if window > 1:
        ma = np.convolve(returns_mean, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window, len(returns_mean) + 1), ma, label=f"mean MA{window}", linewidth=2)
    plt.xlabel("episode")
    plt.ylabel("avg return per-agent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "return.png"))
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.episodes, args.save_dir)
