"""Phase 1: multi-agent PPO smoke test on PettingZoo MPE simple_tag_v3.

Two independent PPO learners co-train:
  - predators (adversary_*) share one policy
  - prey      (agent_*)     share another policy

This makes the team incentives separable (predators get +reward for catching prey;
prey get -reward), which is what `simple_tag_v3` actually returns.
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
import yaml

from train.ppo import PPO, PPOConfig, flatten_obs, set_seed


PREDATOR_PREFIX = "adversary"
PREY_PREFIX = "agent"


def is_predator(name: str) -> bool:
    return name.startswith(PREDATOR_PREFIX)


def is_prey(name: str) -> bool:
    return name.startswith(PREY_PREFIX)


def make_env(n_predators=2, n_prey=2, max_cycles=200, seed=42):
    try:
        from mpe2 import simple_tag_v3
    except ImportError:
        from pettingzoo.mpe import simple_tag_v3
    env = simple_tag_v3.parallel_env(
        num_adversaries=n_predators,
        num_good=n_prey,
        num_obstacles=0,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=None,
    )
    env.reset(seed=seed)
    return env


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _reset(env, seed=None):
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _step(env, actions):
    out = env.step(actions)
    if isinstance(out, tuple) and len(out) == 5:
        next_obs, rewards, terminations, truncations, infos = out
        done_any = bool(any(terminations.values()) or any(truncations.values()))
        return next_obs, rewards, done_any, infos
    elif isinstance(out, tuple) and len(out) == 4:
        next_obs, rewards, dones, infos = out
        done_any = bool(any(dones.values()))
        return next_obs, rewards, done_any, infos
    else:
        raise RuntimeError("Unexpected step() return format from PettingZoo env.")


def _make_ppo(obs_dim: int, act_dim: int, cfg: dict) -> PPO:
    train = cfg.get("train", {})
    return PPO(
        obs_dim,
        act_dim,
        PPOConfig(
            lr=float(train.get("lr", 3e-4)),
            gamma=float(train.get("gamma", 0.99)),
            gae_lambda=float(train.get("gae_lambda", 0.95)),
            clip_coef=float(train.get("clip_coef", 0.2)),
            ent_coef=float(train.get("ent_coef", 0.01)),
            vf_coef=float(train.get("vf_coef", 0.5)),
            update_epochs=int(train.get("update_epochs", 4)),
            minibatch_size=int(train.get("minibatch_size", 1024)),
            hidden=int(train.get("hidden", 128)),
            use_gae=bool(train.get("use_gae", True)),
        ),
    )


def _empty_buf():
    return {"obs": [], "acts": [], "logps": [], "rews": [], "dones": [], "vals": []}


def main(cfg_path, override_eps=None, save_dir=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    env_cfg = cfg.get("env", {})
    max_steps = int(env_cfg.get("max_steps", 200))
    n_pred = int(env_cfg.get("n_predators", 2))
    n_ev = int(env_cfg.get("n_prey", 2))
    total_episodes = int(override_eps or cfg.get("train", {}).get("total_episodes", 500))

    run_id = datetime.now(UTC).strftime("run_%Y%m%d_%H%M")
    out_dir = save_dir or cfg.get("logging", {}).get("save_dir", "artifacts/")
    out_dir = os.path.join(out_dir, run_id)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir); ensure_dir(plots_dir)

    env = make_env(n_predators=n_pred, n_prey=n_ev, max_cycles=max_steps, seed=seed)

    # Probe observation shapes per team (they differ in simple_tag).
    obs0 = _reset(env, seed=seed)
    pred_arr, pred_agents = flatten_obs(obs0, agent_filter=is_predator)
    prey_arr, prey_agents = flatten_obs(obs0, agent_filter=is_prey)
    pred_obs_dim = pred_arr.shape[1] if pred_agents else 0
    prey_obs_dim = prey_arr.shape[1] if prey_agents else 0
    pred_act_dim = env.action_space(pred_agents[0]).n if pred_agents else 5
    prey_act_dim = env.action_space(prey_agents[0]).n if prey_agents else 5

    predator_ppo = _make_ppo(pred_obs_dim, pred_act_dim, cfg) if pred_agents else None
    prey_ppo = _make_ppo(prey_obs_dim, prey_act_dim, cfg) if prey_agents else None

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "pred_return", "prey_return"])

    pred_returns, prey_returns = [], []

    for ep in range(1, total_episodes + 1):
        obs = _reset(env, seed=seed + ep)
        pred_buf, prey_buf = _empty_buf(), _empty_buf()
        pred_ret_ep, prey_ret_ep = 0.0, 0.0
        done_any = False

        while not done_any:
            pred_arr, pred_now = flatten_obs(obs, agent_filter=is_predator)
            prey_arr, prey_now = flatten_obs(obs, agent_filter=is_prey)
            actions: dict = {}

            if pred_now and predator_ppo is not None:
                pa, plogp, pval = predator_ppo.act(pred_arr)
                for i, a in enumerate(pred_now):
                    actions[a] = int(pa[i])

            if prey_now and prey_ppo is not None:
                ea, elogp, eval_ = prey_ppo.act(prey_arr)
                for i, a in enumerate(prey_now):
                    actions[a] = int(ea[i])

            # Some MPE versions need an action for every agent that is alive.
            for a in obs.keys():
                actions.setdefault(a, 0)

            next_obs, rewards, done_any, infos = _step(env, actions)

            if pred_now and predator_ppo is not None:
                team_r = float(np.mean([rewards.get(a, 0.0) for a in pred_now]))
                pred_buf["obs"].extend(pred_arr)
                pred_buf["acts"].extend([actions[a] for a in pred_now])
                pred_buf["logps"].extend([float(plogp[i]) for i in range(len(pred_now))])
                pred_buf["vals"].extend([float(pval[i]) for i in range(len(pred_now))])
                pred_buf["rews"].extend([team_r] * len(pred_now))
                pred_buf["dones"].extend([float(done_any)] * len(pred_now))
                pred_ret_ep += sum(rewards.get(a, 0.0) for a in pred_now)

            if prey_now and prey_ppo is not None:
                team_r = float(np.mean([rewards.get(a, 0.0) for a in prey_now]))
                prey_buf["obs"].extend(prey_arr)
                prey_buf["acts"].extend([actions[a] for a in prey_now])
                prey_buf["logps"].extend([float(elogp[i]) for i in range(len(prey_now))])
                prey_buf["vals"].extend([float(eval_[i]) for i in range(len(prey_now))])
                prey_buf["rews"].extend([team_r] * len(prey_now))
                prey_buf["dones"].extend([float(done_any)] * len(prey_now))
                prey_ret_ep += sum(rewards.get(a, 0.0) for a in prey_now)

            obs = next_obs

        if predator_ppo is not None:
            predator_ppo.update(**pred_buf)
        if prey_ppo is not None:
            prey_ppo.update(**prey_buf)

        pred_returns.append(pred_ret_ep / max(1, n_pred))
        prey_returns.append(prey_ret_ep / max(1, n_ev))
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, pred_returns[-1], prey_returns[-1]])

        if ep % int(cfg["logging"].get("plot_every", 50)) == 0 or ep == total_episodes:
            xs = np.arange(1, len(pred_returns) + 1)
            window = min(50, len(pred_returns))
            plt.figure()
            plt.plot(xs, pred_returns, label="predator", alpha=0.5)
            plt.plot(xs, prey_returns, label="prey", alpha=0.5)
            if window > 1:
                pma = np.convolve(pred_returns, np.ones(window) / window, mode="valid")
                ema = np.convolve(prey_returns, np.ones(window) / window, mode="valid")
                plt.plot(np.arange(window, len(pred_returns) + 1), pma, label=f"pred MA{window}")
                plt.plot(np.arange(window, len(prey_returns) + 1), ema, label=f"prey MA{window}")
            plt.xlabel("episode"); plt.ylabel("avg return per-agent"); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "return.png"))
            plt.close()
            print(f"[{ep}/{total_episodes}] pred(last 10): {np.mean(pred_returns[-10:]):.3f}  "
                  f"prey(last 10): {np.mean(prey_returns[-10:]):.3f}")

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved plot to: {os.path.join(plots_dir, 'return.png')}")
    return predator_ppo, prey_ppo, pred_returns, prey_returns


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.episodes, args.save_dir)
