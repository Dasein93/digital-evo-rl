"""PPO baseline for multi-agent predator-prey on PettingZoo MPE.

Provides:
- set_seed(seed)
- flatten_obs(obs_dict, agent_filter=None)
- PPOConfig dataclass
- ActorCritic(obs_dim, act_dim, hidden)
- PPO trainer (init, act, update, get/set genome)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_obs(obs_dict, agent_filter=None):
    """Turn `{agent_name: np.ndarray}` into `(stacked [N, obs_dim], agents_list)`.

    `agent_filter` optionally restricts to agents whose name passes the predicate
    (e.g. predators only). Order is stable in the dict's iteration order.
    """
    if agent_filter is None:
        agents = list(obs_dict.keys())
    else:
        agents = [a for a in obs_dict.keys() if agent_filter(a)]
    if not agents:
        return np.zeros((0, 0), dtype=np.float32), []
    arr = np.stack([np.asarray(obs_dict[a], dtype=np.float32) for a in agents], axis=0)
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
    minibatch_size: int = 1024
    hidden: int = 128
    max_grad_norm: float = 0.5
    use_gae: bool = True


class ActorCritic(nn.Module):
    """Shared-trunk MLP with separate actor (categorical) and critic heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)
        # Orthogonal init is standard for PPO stability.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.trunk(obs))

    def critic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.trunk(obs))

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.actor_head(h), self.critic_head(h)

    @torch.no_grad()
    def step(self, obs: torch.Tensor):
        """Sample an action, return (action, log_prob, value) as 1-D tensors over batch."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, value.squeeze(-1)


class PPO:
    """Single-team PPO trainer. One instance per population (e.g. predators or prey)."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig | None = None):
        self.cfg = cfg or PPOConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ac = ActorCritic(obs_dim, act_dim, hidden=self.cfg.hidden)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.cfg.lr)

    # ---- genome interface for evolution ----
    def get_genome(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.ac.state_dict().items()}

    def set_genome(self, genome: dict[str, torch.Tensor]) -> None:
        self.ac.load_state_dict(genome)

    # ---- inference ----
    @torch.no_grad()
    def act(self, obs_np: np.ndarray):
        """obs_np: [N, obs_dim] → (actions[N], logps[N], values[N]) as numpy."""
        if obs_np.size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        a, logp, v = self.ac.step(obs)
        return a.cpu().numpy().astype(np.int64), logp.cpu().numpy().astype(np.float32), v.cpu().numpy().astype(np.float32)

    # ---- advantages ----
    def _compute_returns(self, rews, dones, gamma):
        n = len(rews)
        out = np.zeros(n, dtype=np.float32)
        G = 0.0
        for i in range(n - 1, -1, -1):
            G = float(rews[i]) + gamma * G * (1.0 - float(dones[i]))
            out[i] = G
        return torch.from_numpy(out)

    def _compute_gae(self, rews, dones, vals, gamma, lam):
        n = len(rews)
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        # Bootstrap with 0 after final step (episode-buffer assumption).
        for i in range(n - 1, -1, -1):
            next_val = 0.0 if (i == n - 1 or dones[i]) else float(vals[i + 1])
            delta = float(rews[i]) + gamma * next_val * (1.0 - float(dones[i])) - float(vals[i])
            last_gae = delta + gamma * lam * (1.0 - float(dones[i])) * last_gae
            adv[i] = last_gae
        rets = adv + np.asarray(vals, dtype=np.float32)
        return torch.from_numpy(adv), torch.from_numpy(rets)

    # ---- update ----
    def update(self, obs, acts, logps, rews, dones, vals):
        n = len(rews)
        if n == 0:
            return {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "n": 0}
        assert len(dones) == n and len(vals) == n and len(obs) == n and len(acts) == n and len(logps) == n, (
            f"Buffer length mismatch: obs={len(obs)} acts={len(acts)} logps={len(logps)} "
            f"rews={len(rews)} dones={len(dones)} vals={len(vals)}"
        )

        cfg = self.cfg
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        acts_t = torch.as_tensor(np.asarray(acts), dtype=torch.int64)
        old_logps = torch.as_tensor(np.asarray(logps), dtype=torch.float32)
        vals_np = np.asarray(vals, dtype=np.float32)

        if cfg.use_gae:
            adv, rets = self._compute_gae(rews, dones, vals_np, cfg.gamma, cfg.gae_lambda)
        else:
            rets = self._compute_returns(rews, dones, cfg.gamma)
            adv = rets - torch.from_numpy(vals_np)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(n)
        mb = max(1, min(cfg.minibatch_size, n))
        last = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "n": n}
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb):
                b = idx[start:start + mb]
                o, a, ol, ad, rt = obs_t[b], acts_t[b], old_logps[b], adv[b], rets[b]
                logits, v = self.ac(o)
                v = v.squeeze(-1)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a)
                ratio = (logp - ol).exp()
                clip_adv = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * ad
                pg_loss = -(torch.min(ratio * ad, clip_adv)).mean()
                v_loss = 0.5 * (rt - v).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.opt.step()
                last = {"pg_loss": float(pg_loss.item()), "v_loss": float(v_loss.item()),
                        "entropy": float(ent.item()), "n": n}
        return last
