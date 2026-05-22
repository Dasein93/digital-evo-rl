"""End-to-end tests for evolve.py and tournament.py — keep the smoke fast."""
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module")
def trained_ckpt(tmp_path_factory) -> Path:
    """Train once via the smoke config so multiple tests can share the ckpt."""
    import run_cpu
    out_dir = Path(run_cpu.main(
        "configs/smoke.yaml", override_eps=2, save_dir=str(tmp_path_factory.mktemp("train"))
    ))
    return out_dir / "checkpoints" / "final"


def test_run_cpu_writes_qd_artifacts(tmp_path: Path):
    """The smoke config has qd.enabled=true — run_cpu should leave an archive."""
    import run_cpu
    out_dir = Path(run_cpu.main("configs/smoke.yaml", override_eps=2, save_dir=str(tmp_path)))
    for team in ("predator", "prey"):
        assert (out_dir / "qd" / team / "manifest.json").exists()
        assert (out_dir / "plots" / f"qd_archive_{team}.png").exists()


def test_evolve_populates_archive(trained_ckpt: Path, tmp_path: Path):
    from train.tools.evolve import evolve
    from agents.qd import MAPElitesConfig
    summary = evolve(
        ckpt_dir=str(trained_ckpt),
        out_dir=str(tmp_path / "evolve"),
        target_team="predator",
        n_mutants=3, sigma=0.2, eval_eps=1,
        n_predators=1, n_prey=1, max_cycles=15, seed=7,
        device="cpu",
        qd_cfg=MAPElitesConfig(grid_shape=(3, 3)),
    )
    assert summary["n_mutants"] == 3
    assert summary["coverage"] > 0.0
    out = tmp_path / "evolve"
    assert (out / "qd" / "predator" / "manifest.json").exists()
    assert (out / "plots" / "qd_archive_predator.png").exists()
    assert (out / "evolve_summary.json").exists()


def test_tournament_writes_matrix(trained_ckpt: Path, tmp_path: Path):
    from train.tools.tournament import tournament
    summary = tournament(
        pred_entries=[(str(trained_ckpt), "A"), (str(trained_ckpt), "A2")],
        prey_entries=[(str(trained_ckpt), "B"), (str(trained_ckpt), "B2")],
        out_dir=str(tmp_path / "tour"),
        episodes=1,
        n_predators=1, n_prey=1, max_cycles=15,
        seed=42, device="cpu",
    )
    out = tmp_path / "tour"
    assert (out / "tournament.csv").exists()
    assert (out / "tournament_predator.png").exists()
    assert (out / "tournament_prey.png").exists()
    assert (out / "tournament_summary.json").exists()
    data = json.loads((out / "tournament_summary.json").read_text())
    assert len(data["predator_return_matrix"]) == 2
    assert len(data["predator_return_matrix"][0]) == 2
