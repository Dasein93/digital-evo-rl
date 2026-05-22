"""Genetic / neuro-evolution layer over PPO policies.

A genome is a dict[str, torch.Tensor] — the state_dict of an `ActorCritic`.

Provides:
- mutate(genome, sigma, p)
- crossover(parent_a, parent_b, mode)
- tournament_select(pop, fitnesses, k)
- NoveltyArchive (k-NN behavior novelty)
- Population (per-team) with `step_generation`
- evaluate_genome(ppo, genome, env_fn, episodes, agent_filter, max_steps) -> (fitness, behavior)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .ppo import PPO, PPOConfig, ActorCritic, flatten_obs


Genome = dict[str, torch.Tensor]


# ----------------------------- operators -----------------------------------


def mutate(genome: Genome, sigma: float = 0.02, p: float = 1.0) -> Genome:
    """Gaussian perturbation. `p` is the per-tensor mutation probability."""
    out: Genome = {}
    for k, v in genome.items():
        if not torch.is_floating_point(v):
            out[k] = v.clone()
            continue
        if random.random() < p:
            out[k] = v + sigma * torch.randn_like(v)
        else:
            out[k] = v.clone()
    return out


def crossover(a: Genome, b: Genome, mode: str = "uniform") -> Genome:
    """Breed two parents into one child genome.

    mode:
      - "uniform": per-element coin flip between parents
      - "blend":   arithmetic mean of parent tensors
      - "layer":   pick whole layers from one parent or the other
    """
    assert a.keys() == b.keys(), "Parent genomes have different keys"
    out: Genome = {}
    if mode == "blend":
        for k in a:
            if torch.is_floating_point(a[k]):
                out[k] = 0.5 * (a[k] + b[k])
            else:
                out[k] = a[k].clone()
    elif mode == "layer":
        for k in a:
            out[k] = (a[k] if random.random() < 0.5 else b[k]).clone()
    else:  # uniform
        for k in a:
            if torch.is_floating_point(a[k]):
                mask = (torch.rand_like(a[k]) < 0.5).to(a[k].dtype)
                out[k] = mask * a[k] + (1.0 - mask) * b[k]
            else:
                out[k] = a[k].clone()
    return out


def tournament_select(population: list[Genome], fitnesses: list[float], k: int = 3) -> Genome:
    """Pick the best of `k` random contenders."""
    n = len(population)
    k = max(1, min(k, n))
    contenders = random.sample(range(n), k)
    winner = max(contenders, key=lambda i: fitnesses[i])
    return population[winner]


# ----------------------------- novelty -------------------------------------


class NoveltyArchive:
    """K-nearest-neighbour behavior novelty (Lehman & Stanley, 2011 - simplified)."""

    def __init__(self, k: int = 5, capacity: int = 500):
        self.k = k
        self.capacity = capacity
        self.items: list[np.ndarray] = []

    def add(self, bc: np.ndarray) -> None:
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(np.asarray(bc, dtype=np.float32))

    def novelty(self, bc: np.ndarray) -> float:
        if not self.items:
            return 0.0
        bc = np.asarray(bc, dtype=np.float32)
        dists = np.linalg.norm(np.stack(self.items, axis=0) - bc[None, :], axis=1)
        k = min(self.k, len(dists))
        return float(np.mean(np.sort(dists)[:k]))


# ----------------------------- evaluation ----------------------------------


def evaluate_genome(
    ppo: PPO,
    genome: Genome,
    rollout_fn: Callable[[PPO], tuple[float, np.ndarray]],
) -> tuple[float, np.ndarray]:
    """Load genome into ppo's network and run `rollout_fn`. Returns (fitness, behavior)."""
    ppo.set_genome(genome)
    return rollout_fn(ppo)


# ----------------------------- population ----------------------------------


@dataclass
class EvolveConfig:
    pop_size: int = 16
    elitism: int = 2
    tournament_k: int = 3
    mutation_sigma: float = 0.02
    mutation_p: float = 1.0
    crossover_mode: str = "uniform"
    crossover_rate: float = 0.5
    novelty_coef: float = 0.0     # 0.0 => pure fitness; >0 => novelty-blended
    novelty_k: int = 5
    novelty_capacity: int = 500


class Population:
    """Co-evolving population of policy genomes for a single team."""

    def __init__(
        self,
        template: PPO,
        cfg: EvolveConfig | None = None,
        seed_genome: Genome | None = None,
    ):
        self.template = template
        self.cfg = cfg or EvolveConfig()
        self.archive = NoveltyArchive(k=self.cfg.novelty_k, capacity=self.cfg.novelty_capacity)

        base = seed_genome if seed_genome is not None else template.get_genome()
        # Seed population: keep `elitism` exact copies, mutate the rest.
        self.individuals: list[Genome] = []
        for i in range(self.cfg.pop_size):
            if i < max(1, self.cfg.elitism):
                self.individuals.append({k: v.clone() for k, v in base.items()})
            else:
                self.individuals.append(mutate(base, sigma=self.cfg.mutation_sigma, p=self.cfg.mutation_p))

        self.fitnesses: list[float] = [float("-inf")] * self.cfg.pop_size
        self.behaviors: list[np.ndarray] = [np.zeros(1, dtype=np.float32)] * self.cfg.pop_size
        self.generation: int = 0

    # ---- one full GA step ----
    def step_generation(self, rollout_fn: Callable[[PPO], tuple[float, np.ndarray]]) -> dict:
        # 1) Evaluate everyone using the template's network in place.
        for i, g in enumerate(self.individuals):
            fit, bc = evaluate_genome(self.template, g, rollout_fn)
            self.fitnesses[i] = float(fit)
            self.behaviors[i] = np.asarray(bc, dtype=np.float32)
            self.archive.add(self.behaviors[i])

        # 2) Compute selection scores (fitness + novelty bonus).
        if self.cfg.novelty_coef > 0.0:
            novs = [self.archive.novelty(b) for b in self.behaviors]
            scores = [f + self.cfg.novelty_coef * n for f, n in zip(self.fitnesses, novs)]
        else:
            novs = [0.0] * len(self.individuals)
            scores = list(self.fitnesses)

        # 3) Rank, keep elites.
        order = sorted(range(len(self.individuals)), key=lambda i: scores[i], reverse=True)
        elites = [self.individuals[i] for i in order[: max(1, self.cfg.elitism)]]

        # 4) Fill the rest with tournament selection + breeding + mutation.
        children: list[Genome] = [{k: v.clone() for k, v in e.items()} for e in elites]
        while len(children) < self.cfg.pop_size:
            pa = tournament_select(self.individuals, scores, k=self.cfg.tournament_k)
            if random.random() < self.cfg.crossover_rate:
                pb = tournament_select(self.individuals, scores, k=self.cfg.tournament_k)
                child = crossover(pa, pb, mode=self.cfg.crossover_mode)
            else:
                child = {k: v.clone() for k, v in pa.items()}
            child = mutate(child, sigma=self.cfg.mutation_sigma, p=self.cfg.mutation_p)
            children.append(child)

        self.individuals = children
        self.generation += 1

        stats = {
            "gen": self.generation,
            "fit_max": float(np.max(self.fitnesses)),
            "fit_mean": float(np.mean(self.fitnesses)),
            "fit_min": float(np.min(self.fitnesses)),
            "novelty_mean": float(np.mean(novs)),
            "pop_size": len(self.individuals),
        }
        return stats

    def best(self) -> tuple[Genome, float]:
        i = int(np.argmax(self.fitnesses))
        return self.individuals[i], float(self.fitnesses[i])
