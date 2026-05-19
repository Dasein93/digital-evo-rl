"""CPU/GPU training entrypoint for the predator-prey baseline.

Trains two shared-policy PPO agents — one for predators, one for prey —
on PettingZoo MPE ``simple_tag_v3``. Logs CSV metrics + return plot per
run, optionally records trajectories for replay, computes per-team
behavior characteristics + novelty scores against a sliding archive, and
(optionally) adds a novelty bonus to that team's rewards before updating.

Checkpoints are written at the end of training (and optionally every
``checkpoint.every`` episodes) so :mod:`train.tools.eval` can later run
greedy evaluations.
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

from agents.checkpoint import save_checkpoint
from agents.novelty import BehaviorCharacteristic, NoveltyArchive
from envs.predator_prey import make_env, reset as env_reset, step as env_step, team_of, TEAM_PREDATOR, TEAM_PREY
from train.ppo import PPO, PPOConfig, flatten_obs, set_seed, empty_rollout, append_step
from train.tools.recorder import TrajectoryRecorder


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_ppos(train_cfg: dict, obs_dims: Dict[str, int], act_dim: int, device: str) -> Dict[str, PPO]:
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
    return {team: PPO(obs_dims[team], act_dim, pcfg, device=device) for team in obs_dims}


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


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def main(
    cfg_path: str,
    override_eps: int | None = None,
    save_dir: str | None = None,
    device_override: str | None = None,
) -> str:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    env_cfg = cfg.get("env", {})
    max_steps = int(env_cfg.get("max_steps", 200))
    n_pred = int(env_cfg.get("n_predators", 2))
    n_prey = int(env_cfg.get("n_prey", 2))

    total_episodes = int(override_eps or cfg.get("train", {}).get("total_episodes", 500))
    device = _resolve_device(device_override or cfg.get("train", {}).get("device", "cpu"))

    rec_cfg = cfg.get("recording", {}) or {}
    rec_enabled = bool(rec_cfg.get("enabled", False))
    rec_sample_rate = int(rec_cfg.get("sample_rate", 1))

    nov_cfg = cfg.get("novelty", {}) or {}
    nov_enabled = bool(nov_cfg.get("enabled", False))
    nov_capacity = int(nov_cfg.get("capacity", 500))
    nov_k = int(nov_cfg.get("k", 15))
    nov_bonus_coef = float(nov_cfg.get("bonus_coef", 0.0))

    ckpt_cfg = cfg.get("checkpoint", {}) or {}
    ckpt_enabled = bool(ckpt_cfg.get("enabled", True))
    ckpt_every = int(ckpt_cfg.get("every", 0))  # 0 = end-of-training only

    run_id = datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    out_root = save_dir or cfg.get("logging", {}).get("save_dir", "artifacts/")
    out_dir = os.path.join(out_root, run_id)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir); ensure_dir(plots_dir)

    env = make_env(n_predators=n_pred, n_prey=n_prey, max_cycles=max_steps, seed=seed)
    obs0 = env_reset(env, seed=seed)
    obs_list, agents = flatten_obs(obs0)
    act_dim = int(env.action_space(agents[0]).n)

    team_dims: Dict[str, int] = {}
    for i, a in enumerate(agents):
        team_dims.setdefault(team_of(a), obs_list[i].shape[0])

    ppos = _build_ppos(cfg.get("train", {}) or {}, team_dims, act_dim, device=device)

    archives = {team: NoveltyArchive(capacity=nov_capacity, k=nov_k) for team in ppos} if nov_enabled else {}

    manifest = {
        "run_id": run_id,
        "seed": seed,
        "device": device,
        "config_path": os.path.abspath(cfg_path),
        "env": {"n_predators": n_pred, "n_prey": n_prey, "max_steps": max_steps, "act_dim": act_dim},
        "team_obs_dims": team_dims,
        "total_episodes": total_episodes,
        "recording": {"enabled": rec_enabled, "sample_rate": rec_sample_rate},
        "novelty": {"enabled": nov_enabled, "capacity": nov_capacity, "k": nov_k, "bonus_coef": nov_bonus_coef},
        "checkpoint": {"enabled": ckpt_enabled, "every": ckpt_every},
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
                "pred_novelty", "prey_novelty",
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
    novelty_pred: List[float] = []
    novelty_prey: List[float] = []

    def _checkpoint(label: str) -> None:
        if not ckpt_enabled:
            return
        ckpt_dir = os.path.join(out_dir, "checkpoints", label)
        save_checkpoint(ckpt_dir, ppos, extra={"label": label, "manifest_ref": "../../manifest.json"})

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
                        a_t, logp_t, v_t = ppos[team].ac.step(torch.from_numpy(team_obs).to(device))
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

            team_novelty: Dict[str, float] = {TEAM_PREDATOR: 0.0, TEAM_PREY: 0.0}
            if nov_enabled:
                for team in ppos:
                    team_agents = [a for a in agents if team_of(a) == team]
                    per_agent_bc = {
                        a: BehaviorCharacteristic.from_agent(buf["obs"].get(a, []), buf["acts"].get(a, []), n_actions=act_dim)
                        for a in team_agents
                    }
                    tbc = BehaviorCharacteristic.team(per_agent_bc)
                    s = archives[team].score_and_add(tbc)
                    team_novelty[team] = s
                    if nov_bonus_coef > 0.0:
                        bonus = nov_bonus_coef * s
                        for a in team_agents:
                            r = buf["rews"].get(a)
                            if r:
                                buf["rews"][a] = [x + bonus for x in r]

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
            returns_mean.append(mean_ret); returns_pred.append(pred_ret); returns_prey.append(prey_ret)
            novelty_pred.append(team_novelty[TEAM_PREDATOR]); novelty_prey.append(team_novelty[TEAM_PREY])

            pu = team_updates.get(TEAM_PREDATOR) or {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
            yu = team_updates.get(TEAM_PREY) or {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        ep, mean_ret, pred_ret, prey_ret, steps,
                        pu["pg_loss"], pu["v_loss"], pu["entropy"],
                        yu["pg_loss"], yu["v_loss"], yu["entropy"],
                        team_novelty[TEAM_PREDATOR], team_novelty[TEAM_PREY],
                    ]
                )

            if ckpt_every > 0 and ep % ckpt_every == 0:
                _checkpoint(f"ep_{ep:06d}")

            plot_every = int(cfg.get("logging", {}).get("plot_every", 50))
            if ep % plot_every == 0 or ep == total_episodes:
                _plot(returns_mean, returns_pred, returns_prey, novelty_pred, novelty_prey, plots_dir, nov_enabled)
                tail = returns_mean[-min(10, len(returns_mean)):]
                novbit = (
                    f" nov(pred/prey)={np.mean(novelty_pred[-10:]):.3f}/{np.mean(novelty_prey[-10:]):.3f}"
                    if nov_enabled else ""
                )
                print(
                    f"[{ep}/{total_episodes}] mean={np.mean(tail):.3f} "
                    f"pred={np.mean(returns_pred[-10:]):.3f} prey={np.mean(returns_prey[-10:]):.3f} "
                    f"steps={steps}{novbit}"
                )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()

    _checkpoint("final")

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved plot   -> {os.path.join(plots_dir, 'return.png')}")
    if ckpt_enabled:
        print(f"Final ckpt   -> {os.path.join(out_dir, 'checkpoints', 'final')}")
    print(f"Run dir      -> {out_dir}")
    return out_dir


def _plot(
    returns_mean: List[float],
    returns_pred: List[float],
    returns_prey: List[float],
    novelty_pred: List[float],
    novelty_prey: List[float],
    plots_dir: str,
    nov_enabled: bool,
) -> None:
    xs = np.arange(1, len(returns_mean) + 1)
    window = min(50, len(returns_mean))
    plt.figure()
    plt.plot(xs, returns_mean, label="mean", alpha=0.6)
    plt.plot(xs, returns_pred, label="predator", alpha=0.6)
    plt.plot(xs, returns_prey, label="prey", alpha=0.6)
    if window > 1:
        ma = np.convolve(returns_mean, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window, len(returns_mean) + 1), ma, label=f"mean MA{window}", linewidth=2)
    plt.xlabel("episode"); plt.ylabel("avg return per-agent"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "return.png")); plt.close()

    if nov_enabled:
        plt.figure()
        plt.plot(xs, novelty_pred, label="predator novelty", alpha=0.7)
        plt.plot(xs, novelty_prey, label="prey novelty", alpha=0.7)
        plt.xlabel("episode"); plt.ylabel("k-NN novelty (BC space)"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "novelty.png")); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda | mps | auto (overrides config)")
    args = ap.parse_args()
    main(args.config, args.episodes, args.save_dir, args.device)
