"""Tiny smoke tests for the PPO + evolution stack.

These run on CPU in under ~10s. They don't assert learning, only that nothing
crashes end-to-end.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.ppo import PPO, PPOConfig, ActorCritic, flatten_obs, set_seed
from train.evolve import (
    EvolveConfig, Population, NoveltyArchive,
    mutate, crossover, tournament_select,
)


def test_actor_critic_shapes():
    set_seed(0)
    net = ActorCritic(obs_dim=8, act_dim=5, hidden=16)
    obs = torch.zeros(4, 8)
    logits, v = net(obs)
    assert logits.shape == (4, 5)
    assert v.shape == (4, 1)
    a, logp, val = net.step(obs)
    assert a.shape == (4,) and logp.shape == (4,) and val.shape == (4,)


def test_ppo_update_runs():
    set_seed(1)
    ppo = PPO(obs_dim=6, act_dim=4, cfg=PPOConfig(update_epochs=2, minibatch_size=8, hidden=16))
    n = 32
    obs = np.random.randn(n, 6).astype(np.float32)
    acts = np.random.randint(0, 4, size=n)
    logps = np.random.randn(n).astype(np.float32)
    vals = np.random.randn(n).astype(np.float32)
    rews = np.random.randn(n).astype(np.float32)
    dones = np.zeros(n, dtype=np.float32); dones[-1] = 1.0
    stats = ppo.update(list(obs), list(acts), list(logps), list(rews), list(dones), list(vals))
    assert stats["n"] == n


def test_flatten_obs_filter():
    obs = {"adversary_0": np.zeros(6), "adversary_1": np.zeros(6),
           "agent_0": np.zeros(8), "agent_1": np.zeros(8)}
    arr_pred, names_pred = flatten_obs(obs, agent_filter=lambda a: a.startswith("adversary"))
    arr_prey, names_prey = flatten_obs(obs, agent_filter=lambda a: a.startswith("agent"))
    assert arr_pred.shape == (2, 6) and names_pred == ["adversary_0", "adversary_1"]
    assert arr_prey.shape == (2, 8) and names_prey == ["agent_0", "agent_1"]


def test_mutate_changes_weights():
    set_seed(2)
    net = ActorCritic(4, 3, hidden=8)
    g0 = {k: v.clone() for k, v in net.state_dict().items()}
    g1 = mutate(g0, sigma=0.5, p=1.0)
    diffs = [float((g1[k] - g0[k]).abs().sum()) for k in g0 if torch.is_floating_point(g0[k])]
    assert sum(diffs) > 0.0


def test_crossover_blends_parents():
    set_seed(3)
    a = ActorCritic(4, 3, hidden=8).state_dict()
    b = ActorCritic(4, 3, hidden=8).state_dict()
    child = crossover(a, b, mode="blend")
    for k in a:
        if not torch.is_floating_point(a[k]):
            continue
        # blend should be exactly the midpoint
        assert torch.allclose(child[k], 0.5 * (a[k] + b[k]))


def test_novelty_archive_grows():
    arc = NoveltyArchive(k=2, capacity=4)
    for v in [np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.])]:
        arc.add(v)
    assert arc.novelty(np.array([2., 2.])) > 0.0


def test_population_step_runs():
    set_seed(4)
    template = PPO(obs_dim=4, act_dim=3, cfg=PPOConfig(hidden=8))
    pop = Population(template, cfg=EvolveConfig(pop_size=4, elitism=1, tournament_k=2,
                                                mutation_sigma=0.1))
    def rollout(net):
        # Trivial fitness: negative L2 of first weight tensor -> drives toward 0.
        sd = net.ac.state_dict()
        first = next(iter(sd.values())).float()
        bc = first.flatten()[:4].detach().cpu().numpy()
        return -float((first ** 2).sum()), bc
    stats = pop.step_generation(rollout)
    assert stats["pop_size"] == 4 and stats["gen"] == 1
