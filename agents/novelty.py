"""Behavior characteristics + novelty archive — minimal QD scaffolding.

The BC for one agent over one episode is a fixed-length vector summarizing
*how* it acted, independent of *how well*:

    [mean_pos_x, mean_pos_y, mean_speed, action_entropy]

Position and velocity are read from the first four entries of every MPE
``simple_tag`` observation (``[vx, vy, x, y, ...]``), which is consistent
across both teams despite their different total obs dims.

The team-level BC is the mean of its members' BCs. Each team owns an
independent :class:`NoveltyArchive`; novelty is the mean L2 distance to
the *k* nearest neighbors already in the archive. An empty archive scores
0.0 — the first episode is, by construction, not novel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

BC_DIM = 4


def _action_entropy(actions: Sequence[int], n_actions: int) -> float:
    if not actions:
        return 0.0
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=n_actions).astype(np.float64)
    p = counts / counts.sum()
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def bc_from_agent_trajectory(
    obs_seq: Sequence[np.ndarray],
    act_seq: Sequence[int],
    n_actions: int = 5,
) -> np.ndarray:
    """Compute the 4D behavior characteristic for one agent's episode."""
    if not obs_seq:
        return np.zeros(BC_DIM, dtype=np.float32)
    obs_arr = np.stack([np.asarray(o, dtype=np.float32) for o in obs_seq], axis=0)
    vel = obs_arr[:, 0:2]
    pos = obs_arr[:, 2:4]
    mean_pos = pos.mean(axis=0)
    mean_speed = float(np.linalg.norm(vel, axis=1).mean())
    H = _action_entropy(act_seq, n_actions)
    return np.asarray([mean_pos[0], mean_pos[1], mean_speed, H], dtype=np.float32)


def team_bc(per_agent_bc: Dict[str, np.ndarray]) -> np.ndarray:
    """Mean of per-agent BCs for one team. Returns zeros if the team is empty."""
    if not per_agent_bc:
        return np.zeros(BC_DIM, dtype=np.float32)
    stacked = np.stack(list(per_agent_bc.values()), axis=0)
    return stacked.mean(axis=0).astype(np.float32)


@dataclass
class BehaviorCharacteristic:
    """Convenience namespace so callers can write ``BehaviorCharacteristic.from_agent(...)``."""

    @staticmethod
    def from_agent(obs_seq, act_seq, n_actions: int = 5) -> np.ndarray:
        return bc_from_agent_trajectory(obs_seq, act_seq, n_actions)

    @staticmethod
    def team(per_agent_bc: Dict[str, np.ndarray]) -> np.ndarray:
        return team_bc(per_agent_bc)


class NoveltyArchive:
    """Ring buffer of past behavior characteristics with k-NN novelty scoring."""

    def __init__(self, capacity: int = 500, k: int = 15):
        assert capacity >= 1 and k >= 1
        self.capacity = int(capacity)
        self.k = int(k)
        self._buf: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, bc: np.ndarray) -> None:
        self._buf.append(np.asarray(bc, dtype=np.float32).copy())
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    def score(self, bc: np.ndarray) -> float:
        if not self._buf:
            return 0.0
        arr = np.stack(self._buf, axis=0)
        dists = np.linalg.norm(arr - np.asarray(bc, dtype=np.float32), axis=1)
        k = min(self.k, dists.shape[0])
        return float(np.sort(dists)[:k].mean())

    def score_and_add(self, bc: np.ndarray) -> float:
        s = self.score(bc)
        self.add(bc)
        return s
