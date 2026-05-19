"""Shared-policy PPO for multi-agent discrete-action environments.

Designed to be driven by ``run_cpu.py``: a single policy is used for every
agent in a PettingZoo parallel env, and one transition is recorded per
(step, agent) pair. Trajectories are split per-agent before computing
returns / GAE-lambda advantages, then concatenated for minibatch updates.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_obs(obs_dict: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], List[str]]:
    """Normalize a ``{agent: obs}`` dict to a list of per-agent flat arrays.

    Returns a list rather than a stacked array because different teams
    (predators vs prey in MPE simple_tag) can have different obs dims.
    The returned agent order matches the list order — callers must use it
    when scattering actions back into the env.
    """
    agents = list(obs_dict.keys())
    arr = [np.asarray(obs_dict[a], dtype=np.float32).reshape(-1) for a in agents]
    return arr, agents


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    update_epochs: int = 4
    batch_size: int = 2048
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    hidden: int = 128


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def step(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        v = self.critic(obs).squeeze(-1)
        return a, logp, v


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ac = ActorCritic(obs_dim, act_dim, hidden=cfg.hidden)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)

    @staticmethod
    def _gae(
        rews: Sequence[float],
        dones: Sequence[float],
        vals: Sequence[float],
        gamma: float,
        lam: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (advantages, returns) for a single agent trajectory.

        Bootstraps with 0 at the end — fine for episodic rollouts where the
        final step is terminal.
        """
        n = len(rews)
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in range(n - 1, -1, -1):
            next_val = 0.0 if t == n - 1 else vals[t + 1]
            mask = 1.0 - float(dones[t])
            delta = float(rews[t]) + gamma * next_val * mask - float(vals[t])
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        ret = adv + np.asarray(vals, dtype=np.float32)
        return adv, ret

    def update(
        self,
        obs_per_agent: Dict[str, List[np.ndarray]],
        acts_per_agent: Dict[str, List[int]],
        logps_per_agent: Dict[str, List[float]],
        rews_per_agent: Dict[str, List[float]],
        dones_per_agent: Dict[str, List[float]],
        vals_per_agent: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """PPO update over a multi-agent rollout grouped by agent id.

        Each agent's trajectory contributes independent GAE estimates; the
        flattened transitions are then shuffled across agents for minibatch
        SGD.
        """
        cfg = self.cfg
        all_obs, all_acts, all_logps, all_adv, all_ret = [], [], [], [], []
        for agent in obs_per_agent:
            n = len(rews_per_agent[agent])
            if n == 0:
                continue
            assert len(obs_per_agent[agent]) == n
            assert len(acts_per_agent[agent]) == n
            assert len(logps_per_agent[agent]) == n
            assert len(vals_per_agent[agent]) == n
            assert len(dones_per_agent[agent]) == n
            adv, ret = self._gae(
                rews_per_agent[agent],
                dones_per_agent[agent],
                vals_per_agent[agent],
                cfg.gamma,
                cfg.gae_lambda,
            )
            all_obs.append(np.asarray(obs_per_agent[agent], dtype=np.float32))
            all_acts.append(np.asarray(acts_per_agent[agent], dtype=np.int64))
            all_logps.append(np.asarray(logps_per_agent[agent], dtype=np.float32))
            all_adv.append(adv)
            all_ret.append(ret)

        if not all_obs:
            return {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "n": 0}

        obs = torch.from_numpy(np.concatenate(all_obs, axis=0))
        acts = torch.from_numpy(np.concatenate(all_acts, axis=0))
        old_logps = torch.from_numpy(np.concatenate(all_logps, axis=0))
        adv = torch.from_numpy(np.concatenate(all_adv, axis=0))
        rets = torch.from_numpy(np.concatenate(all_ret, axis=0))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        mb = max(1, min(cfg.minibatch_size, n))
        pg_acc = v_acc = ent_acc = 0.0
        steps = 0
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb):
                b = idx[start:start + mb]
                o, a, ol, ad, rt = obs[b], acts[b], old_logps[b], adv[b], rets[b]
                logits = self.ac.actor(o)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a)
                ratio = (logp - ol).exp()
                clip_adv = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * ad
                pg_loss = -(torch.min(ratio * ad, clip_adv)).mean()
                v = self.ac.critic(o).squeeze(-1)
                v_loss = 0.5 * (rt - v).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.opt.step()
                pg_acc += float(pg_loss.item())
                v_acc += float(v_loss.item())
                ent_acc += float(ent.item())
                steps += 1
        return {
            "pg_loss": pg_acc / max(1, steps),
            "v_loss": v_acc / max(1, steps),
            "entropy": ent_acc / max(1, steps),
            "n": int(n),
        }


def empty_rollout() -> Dict[str, Dict[str, List]]:
    """A per-agent rollout buffer of empty lists, populated lazily."""
    return {
        "obs": {}, "acts": {}, "logps": {},
        "rews": {}, "dones": {}, "vals": {},
    }


def append_step(
    buf: Dict[str, Dict[str, List]],
    agents: Sequence[str],
    obs_arr: np.ndarray,
    acts: Dict[str, int],
    logps: Sequence[float],
    vals: Sequence[float],
    rewards: Dict[str, float],
    done_any: bool,
) -> None:
    """Append one env step into a per-agent rollout buffer."""
    for i, agent in enumerate(agents):
        for key in ("obs", "acts", "logps", "rews", "dones", "vals"):
            buf[key].setdefault(agent, [])
        buf["obs"][agent].append(np.asarray(obs_arr[i], dtype=np.float32))
        buf["acts"][agent].append(int(acts[agent]))
        buf["logps"][agent].append(float(logps[i]))
        buf["vals"][agent].append(float(vals[i]))
        buf["rews"][agent].append(float(rewards[agent]))
        buf["dones"][agent].append(float(done_any))
