# Phase 1: CPU smoke test using PettingZoo MPE simple_tag_v3 (predator-prey style)
import os, csv, argparse, yaml, numpy as np
from datetime import datetime, UTC
import matplotlib
matplotlib.use("Agg")  # headless backend for Colab/servers
import matplotlib.pyplot as plt

from train.ppo import PPO, PPOConfig, flatten_obs, set_seed


def make_env(n_predators=2, n_prey=2, max_cycles=200, seed=42):
    """Adversaries = predators, Good agents = prey."""
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
    """Compat: parallel_env.reset may return (obs, info)."""
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _step(env, actions):
    """
    Compat wrapper:
      Newer PettingZoo parallel step -> (next_obs, rewards, terminations, truncations, infos)
      Older -> (next_obs, rewards, dones, infos)
    Returns: next_obs, rewards, done_any, infos
    """
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

    # Initial observation (handle tuple form)
    obs0 = _reset(env, seed=seed)
    obs_arr, agents = flatten_obs(obs0)
    obs_dim = obs_arr.shape[1]
    act_dim = env.action_space(agents[0]).n

    ppo = PPO(
        obs_dim,
        act_dim,
        PPOConfig(
            lr=float(cfg["train"].get("lr", 3e-4)),
            gamma=float(cfg["train"].get("gamma", 0.99)),
            clip_coef=float(cfg["train"].get("clip_coef", 0.2)),
            ent_coef=float(cfg["train"].get("ent_coef", 0.01)),
            vf_coef=float(cfg["train"].get("vf_coef", 0.5)),
            update_epochs=int(cfg["train"].get("update_epochs", 4)),
            batch_size=int(cfg["train"].get("batch_size", 2048)),
            hidden=128,
        ),
    )

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "return_mean"])

    returns = []
    storage = {"obs": [], "acts": [], "logps": [], "rews": [], "dones": [], "vals": []}

    for ep in range(1, total_episodes + 1):
        obs = _reset(env, seed=seed + ep)
        ep_ret = 0.0
        done_any = False

        while not done_any:
            obs_arr, agents = flatten_obs(obs)
            obs_t = np.asarray(obs_arr, dtype=np.float32)

            import torch
            with torch.no_grad():
                a, logp, v = ppo.ac.step(torch.from_numpy(obs_t))

            acts = {agent: int(a[i].item()) for i, agent in enumerate(agents)}
            next_obs, rewards, done_any, infos = _step(env, acts)

            # store per-agent, then average to single-trajectory view
            storage["obs"].extend(obs_t)
            storage["acts"].extend([acts[a] for a in agents])
            storage["logps"].extend([logp[i].item() for i in range(len(agents))])
            storage["vals"].extend([v[i].item() for i in range(len(agents))])
            storage["rews"].append(np.mean(list(rewards.values())))
            storage["dones"].append(float(done_any))

            ep_ret += np.sum(list(rewards.values()))
            obs = next_obs

        # PPO update on episode buffer
        ppo.update(storage["obs"], storage["acts"], storage["logps"],
                   storage["rews"], storage["dones"], storage["vals"])
        # clear storage
        for k in storage:
            storage[k] = []

        returns.append(ep_ret / max(1, len(agents)))
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, returns[-1]])

        if ep % int(cfg["logging"].get("plot_every", 50)) == 0 or ep == total_episodes:
            xs = np.arange(1, len(returns) + 1)
            window = min(50, len(returns))
            ma = np.convolve(returns, np.ones(window) / window, mode="valid")
            plt.figure()
            plt.plot(xs, returns, label="return")
            if len(ma) > 1:
                plt.plot(np.arange(window, len(returns) + 1), ma, label=f"MA{window}")
            plt.xlabel("episode"); plt.ylabel("avg return per-agent"); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "return.png"))
            plt.close()
            print(f"[{ep}/{total_episodes}] mean return (last 10): {np.mean(returns[-10:]):.3f}")

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved plot to: {os.path.join(plots_dir, 'return.png')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.episodes, args.save_dir)
