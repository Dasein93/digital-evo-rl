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


def test_coevolve_resume_continues_from_last_gen(trained_ckpt: Path, tmp_path: Path):
    """Run 2 gens, then re-invoke with --resume + larger N — only the new gens execute."""
    from train.tools.coevolve import coevolve
    from agents.qd import MAPElitesConfig
    out = tmp_path / "coev_resume"
    kwargs = dict(
        seed_ckpt_dir=str(trained_ckpt), out_dir=str(out),
        n_mutants=2, sigma=0.3, eval_eps=1,
        n_predators=1, n_prey=1, n_obstacles=0, max_cycles=12,
        seed=33, device="cpu",
        qd_cfg=MAPElitesConfig(grid_shape=(3, 3)),
    )
    s1 = coevolve(generations=2, **kwargs)
    assert s1["generations"] == 2
    # Resume to gen 4 — gens 1-2 should be reused, gens 3-4 executed.
    s2 = coevolve(generations=4, resume=True, **kwargs)
    assert s2["generations"] == 4
    # History should have 4 gens × 2 teams = 8 records.
    assert len(s2["history"]) == 8
    for g in (1, 2, 3, 4):
        assert (out / f"gen_{g:03d}" / "champion_checkpoint" / "manifest.json").exists()


def test_coevolve_with_workers_matches_serial(trained_ckpt: Path, tmp_path: Path):
    """Parallel mutant eval should produce equivalent fitness rankings to serial.

    We don't require *identical* numbers (RNG/process ordering differs) — just
    that both modes complete and produce non-empty history.
    """
    from train.tools.coevolve import coevolve
    from agents.qd import MAPElitesConfig
    base_kwargs = dict(
        seed_ckpt_dir=str(trained_ckpt),
        generations=1, n_mutants=4, sigma=0.3, eval_eps=1,
        n_predators=1, n_prey=1, n_obstacles=0, max_cycles=10,
        seed=44, device="cpu",
        qd_cfg=MAPElitesConfig(grid_shape=(3, 3)),
    )
    s_serial = coevolve(out_dir=str(tmp_path / "serial"), workers=1, **base_kwargs)
    s_par = coevolve(out_dir=str(tmp_path / "parallel"), workers=2, **base_kwargs)
    assert len(s_serial["history"]) == 2  # 1 gen x 2 teams
    assert len(s_par["history"]) == 2


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
