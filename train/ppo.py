# --- replace in train/ppo.py ---

def _compute_returns(self, rews, dones, values, gamma):
    """
    Compute discounted returns G_t with episode termination indicated by dones (0/1).
    Works with any buffer length; no reverse slicing tricks.
    """
    n = len(rews)
    out = [0.0] * n
    G = 0.0
    # iterate backward
    for i in range(n - 1, -1, -1):
        G = float(rews[i]) + gamma * G * (1.0 - float(dones[i]))
        out[i] = G
    return torch.tensor(out, dtype=torch.float32)

def update(self, obs, acts, logps, rews, dones, vals):
    # guard: all buffers must be same nonzero length
    n = len(rews)
    assert n > 0, "Empty rollout: no steps collected before update()"
    assert len(dones) == n and len(vals) == n and len(obs) == n and len(acts) == n and len(logps) == n, \
        f"Buffer length mismatch: {len(obs)=} {len(acts)=} {len(logps)=} {len(rews)=} {len(dones)=} {len(vals)=}"

    cfg = self.cfg
    obs = torch.tensor(np.array(obs), dtype=torch.float32)
    acts = torch.tensor(np.array(acts), dtype=torch.int64)
    old_logps = torch.tensor(np.array(logps), dtype=torch.float32)
    vals = torch.tensor(np.array(vals), dtype=torch.float32)
    rets = self._compute_returns(rews, dones, vals, cfg.gamma)
    adv = rets - vals
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idx = np.arange(n)
    for _ in range(cfg.update_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, 1024):
            b = idx[start:start+1024]
            o, a, ol, ad, rt = obs[b], acts[b], old_logps[b], adv[b], rets[b]
            logits = self.ac.actor(o)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(a)
            ratio = (logp - ol).exp()
            clip_adv = torch.clamp(ratio, 1-cfg.clip_coef, 1+cfg.clip_coef) * ad
            pg_loss = -(torch.min(ratio*ad, clip_adv)).mean()
            v = self.ac.critic(o).squeeze(-1)
            v_loss = 0.5 * (rt - v).pow(2).mean() * cfg.vf_coef
            ent = dist.entropy().mean() * cfg.ent_coef
            loss = pg_loss + v_loss - ent
            self.opt.zero_grad(); loss.backward(); self.opt.step()
# --- end patch ---
