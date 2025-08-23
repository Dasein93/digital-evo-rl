# Phase 1: CPU smoke test training on PettingZoo pursuit_v4 (predator-prey grid)
import os, time, csv, argparse, yaml, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from train.ppo import PPO, PPOConfig, flatten_obs, set_seed

# We use a built-in predator–prey env for a fast first run.
# Your configs may say env.id=predator_prey_v0; we'll route to pursuit_v4 for Phase 1.
def make_env(grid_size=7, n_pursuers=5, n_evaders=5, max_cycles=200, seed=42):
    from pettingzoo.butterfly import pursuit_v4
    env = pursuit_v4.parallel_env(
        width=grid_size, height=grid_size,
        n_evaders=n_evaders, n_pursuers=n_pursuers,
        max_cycles=max_cycles, render_mode=None
    )
    env.reset(seed=seed)
    return env

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main(cfg_path, override_eps=None, save_dir=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 42)); set_seed(seed)

    env_cfg = cfg.get("env", {})
    grid = int(env_cfg.get("grid_size", 7))
    max_steps = int(env_cfg.get("max_steps", 200))
    n_pred = int(env_cfg.get("n_predators", 5))
    n_ev = int(env_cfg.get("n_prey", 5))

    total_episodes = int(override_eps or cfg.get("train", {}).get("total_episodes", 500))

    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M")
    out_dir = save_dir or cfg.get("logging", {}).get("save_dir", "artifacts/")
    out_dir = os.path.join(out_dir, run_id)
    plots_dir = os.path.join(out_dir, "plots"); ensure_dir(plots_dir)

    env = make_env(grid_size=grid, n_pursuers=n_pred, n_evaders=n_ev, max_cycles=max_steps, seed=seed)

    # Infer obs/action sizes from first observation
    obs0 = env.reset(seed=seed)
    obs_arr, agents = flatten_obs(obs0)
    obs_dim = obs_arr.shape[1]
    act_dim = env.action_space(agents[0]).n

    ppo = PPO(obs_dim, act_dim, PPOConfig(
        lr=float(cfg["train"].get("lr", 3e-4)),
        gamma=float(cfg["train"].get("gamma", 0.99)),
        clip_coef=float(cfg["train"].get("clip_coef", 0.2)),
        ent_coef=float(cfg["train"].get("ent_coef", 0.01)),
        vf_coef=float(cfg["train"].get("vf_coef", 0.5)),
        update_epochs=int(cfg["train"].get("update_epochs", 4)),
        batch_size=int(cfg["train"].get("batch_size", 2048)),
        hidden=128,
    ))

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "return_mean"])

    returns = []
    storage = {"obs": [], "acts": [], "logps": [], "rews": [], "dones": [], "vals": []}

    for ep in range(1, total_episodes + 1):
        obs = env.reset(seed=seed + ep)
        ep_ret = 0.0; done_any = False
        while not done_any:
            obs_arr, agents = flatten_obs(obs)
            obs_t = np.asarray(obs_arr, dtype=np.float32)
            import torch
            with torch.no_grad():
                a, logp, v = ppo.ac.step(torch.from_numpy(obs_t))
            acts = {agent: int(a[i].item()) for i, agent in enumerate(agents)}
            next_obs, rewards, dones, infos = env.step(acts)

            # store per-agent, then average to single-trajectory view
            storage["obs"].extend(obs_t)
            storage["acts"].extend([acts[a] for a in agents])
            storage["logps"].extend([logp[i].item() for i in range(len(agents))])
            storage["vals"].extend([v[i].item() for i in range(len(agents))])
            # average reward/done across agents so PPO update stays simple
            storage["rews"].append(np.mean(list(rewards.values())))
            storage["dones"].append(float(any(dones.values())))

            ep_ret += np.sum(list(rewards.values()))
            obs = next_obs
            done_any = any(dones.values())

        # One PPO update per episode using the collected steps
        ppo.update(storage["obs"], storage["acts"], storage["logps"],
                   storage["rews"], storage["dones"], storage["vals"])
        # clear storage each episode
        for k in storage: storage[k] = []

        returns.append(ep_ret / max(1, len(agents)))
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, returns[-1]])

        if ep % int(cfg["logging"].get("plot_every", 50)) == 0 or ep == total_episodes:
            plt.figure()
            xs = np.arange(1, len(returns)+1)
            window = min(50, len(returns))
            ma = np.convolve(returns, np.ones(window)/window, mode="valid")
            plt.plot(xs, returns, label="return")
            if len(ma) > 1: plt.plot(np.arange(window, len(returns)+1), ma, label=f"MA{window}")
            plt.xlabel("episode"); plt.ylabel("avg return per-agent"); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "return.png")); plt.close()

            print(f"[{ep}/{total_episodes}] mean return (last 10): {np.mean(returns[-10:]):.3f}")

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved plot to: {os.path.join(plots_dir, 'return.png')}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.episodes, args.save_dir)
