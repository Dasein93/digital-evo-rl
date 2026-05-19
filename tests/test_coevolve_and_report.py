"""End-to-end tests for the co-evolution loop and the HTML report bundler."""
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module")
def trained_ckpt(tmp_path_factory) -> Path:
    import run_cpu
    out = Path(run_cpu.main(
        "configs/smoke.yaml", override_eps=2, save_dir=str(tmp_path_factory.mktemp("base"))
    ))
    return out / "checkpoints" / "final"


def test_seed_override_takes_effect(tmp_path: Path):
    """run_cpu --seed should override the config seed and land in the manifest."""
    import run_cpu
    out = Path(run_cpu.main(
        "configs/smoke.yaml", override_eps=1, save_dir=str(tmp_path), seed_override=999,
    ))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == 999


def test_coevolve_runs_two_generations(trained_ckpt: Path, tmp_path: Path):
    from train.tools.coevolve import coevolve
    from agents.qd import MAPElitesConfig
    s = coevolve(
        seed_ckpt_dir=str(trained_ckpt),
        out_dir=str(tmp_path / "coev"),
        generations=2, n_mutants=2, sigma=0.3, eval_eps=1,
        n_predators=1, n_prey=1, n_obstacles=0, max_cycles=15,
        seed=11, device="cpu",
        qd_cfg=MAPElitesConfig(grid_shape=(3, 3)),
    )
    assert s["generations"] == 2
    out = tmp_path / "coev"
    assert (out / "gen_001" / "champion_checkpoint" / "predator.pt").exists()
    assert (out / "gen_002" / "champion_checkpoint" / "prey.pt").exists()
    assert (out / "plots" / "coevolution_fitness.png").exists()
    assert (out / "plots" / "coevolution_coverage.png").exists()
    assert (out / "champion_tournament" / "tournament.csv").exists()
    assert (out / "coevolve_summary.json").exists()
    summary = json.loads((out / "coevolve_summary.json").read_text())
    assert len(summary["history"]) == 4  # 2 gens x 2 teams


def test_report_bundles_artifacts(trained_ckpt: Path, tmp_path: Path):
    """Report should be one self-contained HTML file with embedded images + JSON blocks."""
    from train.tools.report import build_report

    run_dir = trained_ckpt.parent.parent  # the run dir, not the ckpt dir
    out_path = tmp_path / "report.html"
    build_report(str(run_dir), str(out_path), title="Test report")
    txt = out_path.read_text()
    assert "<!doctype html>" in txt.lower()
    assert "Test report" in txt
    # smoke.yaml has qd enabled and writes plots — at least one image should embed.
    assert "data:image/png;base64," in txt
    assert "manifest.json" in txt
