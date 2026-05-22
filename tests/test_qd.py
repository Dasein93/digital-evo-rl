"""MAP-Elites archive unit tests."""
import os
from pathlib import Path

import numpy as np
import torch

from agents.qd import MAPElitesArchive, MAPElitesConfig


def _dummy_sd():
    return {"w": torch.randn(3, 3)}


def test_coord_maps_inside_grid(tmp_path: Path):
    cfg = MAPElitesConfig(bc_dims=(0, 1), bc_bounds=[(0.0, 1.0), (0.0, 1.0)], grid_shape=(4, 4))
    arch = MAPElitesArchive(cfg, save_dir=str(tmp_path))
    assert arch.coord(np.array([0.0, 0.0, 0, 0], dtype=np.float32)) == (0, 0)
    assert arch.coord(np.array([0.99, 0.99, 0, 0], dtype=np.float32)) == (3, 3)
    # Outside bounds is clamped.
    assert arch.coord(np.array([-5.0, 5.0, 0, 0], dtype=np.float32)) == (0, 3)


def test_insert_keeps_best_in_cell(tmp_path: Path):
    cfg = MAPElitesConfig(bc_dims=(0, 1), bc_bounds=[(0, 1), (0, 1)], grid_shape=(2, 2))
    arch = MAPElitesArchive(cfg, save_dir=str(tmp_path))
    bc = np.array([0.1, 0.1, 0, 0], dtype=np.float32)
    assert arch.try_insert(bc, fitness=1.0, state_dict=_dummy_sd(), label="a") is True
    # Lower fitness in same cell — rejected.
    assert arch.try_insert(bc, fitness=0.5, state_dict=_dummy_sd(), label="b") is False
    # Higher fitness in same cell — accepted.
    assert arch.try_insert(bc, fitness=2.0, state_dict=_dummy_sd(), label="c") is True
    assert arch.cells[(0, 0)]["label"] == "c"
    assert arch.coverage() == 0.25
    assert arch.best_fitness() == 2.0


def test_save_load_round_trip(tmp_path: Path):
    cfg = MAPElitesConfig(bc_dims=(0, 1), bc_bounds=[(0, 1), (0, 1)], grid_shape=(2, 2))
    arch = MAPElitesArchive(cfg, save_dir=str(tmp_path / "a"))
    arch.try_insert(np.array([0.1, 0.1, 0, 0], dtype=np.float32), 1.0, _dummy_sd(), "x")
    arch.try_insert(np.array([0.9, 0.9, 0, 0], dtype=np.float32), 2.0, _dummy_sd(), "y")
    arch.save_manifest()
    arch2 = MAPElitesArchive.load(str(tmp_path / "a"))
    assert set(arch2.cells.keys()) == {(0, 0), (1, 1)}
    assert arch2.coverage() == 0.5
    assert arch2.best_fitness() == 2.0


def test_fitness_grid_has_nan_for_empty(tmp_path: Path):
    cfg = MAPElitesConfig(bc_dims=(0, 1), bc_bounds=[(0, 1), (0, 1)], grid_shape=(2, 2))
    arch = MAPElitesArchive(cfg, save_dir=str(tmp_path))
    arch.try_insert(np.array([0.1, 0.1, 0, 0], dtype=np.float32), 3.0, _dummy_sd(), "x")
    g = arch.fitness_grid()
    assert g[0, 0] == 3.0
    assert np.isnan(g[1, 1])
