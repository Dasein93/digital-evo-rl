"""MAP-Elites archive over policies.

A 2D grid is laid over two chosen dimensions of the behavior characteristic
(see :mod:`agents.novelty`). Each cell holds the highest-fitness policy
that *lands in that cell*, snapshotted to disk as a ``.pt`` state_dict so
it can be re-instantiated later. Coverage = fraction of cells occupied.

The archive is the actual "QD" piece of the project — novelty alone
(Phase 2) shaped the reward; this stores a diverse *portfolio* of solved
behaviors.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class MAPElitesConfig:
    bc_dims: Tuple[int, int] = (0, 3)
    bc_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(-2.0, 2.0), (0.0, 1.65)])
    grid_shape: Tuple[int, int] = (10, 10)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MAPElitesConfig":
        d = d or {}
        return cls(
            bc_dims=tuple(d.get("bc_dims", (0, 3))),
            bc_bounds=[tuple(b) for b in d.get("bc_bounds", [(-2.0, 2.0), (0.0, 1.65)])],
            grid_shape=tuple(d.get("grid_shape", (10, 10))),
        )


class MAPElitesArchive:
    def __init__(self, cfg: MAPElitesConfig, save_dir: str):
        self.cfg = cfg
        self.save_dir = save_dir
        self.cells: Dict[Tuple[int, int], Dict] = {}
        os.makedirs(save_dir, exist_ok=True)

    def coord(self, bc: np.ndarray) -> Tuple[int, int]:
        i_dim, j_dim = self.cfg.bc_dims
        (i_lo, i_hi), (j_lo, j_hi) = self.cfg.bc_bounds
        ni, nj = self.cfg.grid_shape
        bi = int(np.clip((bc[i_dim] - i_lo) / max(i_hi - i_lo, 1e-9) * ni, 0, ni - 1))
        bj = int(np.clip((bc[j_dim] - j_lo) / max(j_hi - j_lo, 1e-9) * nj, 0, nj - 1))
        return (bi, bj)

    def try_insert(self, bc: np.ndarray, fitness: float, state_dict: Dict[str, torch.Tensor], label: str) -> bool:
        c = self.coord(bc)
        cur = self.cells.get(c)
        if cur is not None and fitness <= cur["fitness"]:
            return False
        path = os.path.join(self.save_dir, f"cell_{c[0]:02d}_{c[1]:02d}.pt")
        torch.save(state_dict, path)
        self.cells[c] = {
            "fitness": float(fitness),
            "bc": [float(x) for x in bc],
            "path": path,
            "label": label,
        }
        return True

    def coverage(self) -> float:
        ni, nj = self.cfg.grid_shape
        return len(self.cells) / float(ni * nj)

    def best_fitness(self) -> float:
        if not self.cells:
            return float("-inf")
        return max(c["fitness"] for c in self.cells.values())

    def fitness_grid(self) -> np.ndarray:
        ni, nj = self.cfg.grid_shape
        g = np.full((ni, nj), np.nan, dtype=np.float32)
        for (i, j), c in self.cells.items():
            g[i, j] = c["fitness"]
        return g

    def elite_paths(self) -> List[str]:
        return [c["path"] for c in self.cells.values()]

    def best_elite(self) -> Optional[Dict]:
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda c: c["fitness"])

    def save_manifest(self) -> str:
        path = os.path.join(self.save_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "cfg": {
                        "bc_dims": list(self.cfg.bc_dims),
                        "bc_bounds": [list(b) for b in self.cfg.bc_bounds],
                        "grid_shape": list(self.cfg.grid_shape),
                    },
                    "coverage": self.coverage(),
                    "best_fitness": (None if not self.cells else self.best_fitness()),
                    "cells": {f"{i},{j}": v for (i, j), v in self.cells.items()},
                },
                f,
                indent=2,
            )
        return path

    @classmethod
    def load(cls, save_dir: str) -> "MAPElitesArchive":
        with open(os.path.join(save_dir, "manifest.json"), "r") as f:
            data = json.load(f)
        cfg = MAPElitesConfig(
            bc_dims=tuple(data["cfg"]["bc_dims"]),
            bc_bounds=[tuple(b) for b in data["cfg"]["bc_bounds"]],
            grid_shape=tuple(data["cfg"]["grid_shape"]),
        )
        arch = cls(cfg, save_dir)
        for key, v in data["cells"].items():
            i, j = (int(x) for x in key.split(","))
            arch.cells[(i, j)] = v
        return arch
