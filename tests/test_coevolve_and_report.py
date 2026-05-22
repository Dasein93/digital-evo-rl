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


def test_weighted_hof_combine_math():
    """Direct unit test for the weighted fitness combiner.

    Catches regressions in the central formula independent of the env loop.
    """
    from train.tools.coevolve import _combine_fitnesses

    # No HoF opponents → current fitness returned verbatim, regardless of weight.
    assert _combine_fitnesses(10.0, [], None) == 10.0
    assert _combine_fitnesses(10.0, [], 0.5) == 10.0
    assert _combine_fitnesses(10.0, [], 0.9) == 10.0

    # w_current=None → equal weight across (current + HoF), as in original behavior.
    # mean of [10, 2, 4] = 16/3 ≈ 5.333
    assert abs(_combine_fitnesses(10.0, [2.0, 4.0], None) - 16 / 3) < 1e-6

    # w_current=0.5 → 0.5 * current + 0.5 * mean(hof). With hof=[2,4], mean=3.
    # 0.5*10 + 0.5*3 = 5 + 1.5 = 6.5
    assert abs(_combine_fitnesses(10.0, [2.0, 4.0], 0.5) - 6.5) < 1e-6

    # w_current=1.0 → HoF effectively ignored.
    assert abs(_combine_fitnesses(10.0, [-100.0, -100.0], 1.0) - 10.0) < 1e-6

    # w_current=0.0 → HoF mean only; current ignored.
    assert abs(_combine_fitnesses(10.0, [2.0, 4.0], 0.0) - 3.0) < 1e-6

    # The motivating case (the bug from the first HoF experiment): a mutant
    # smashes 3 historical opponents (+50 each) but loses to the current
    # strong opponent (-20). Under uniform weights it's promoted on average,
    # which is exactly the failure mode we're trying to fix. Under a strong
    # current-opponent weight (0.8) the same mutant is demoted.
    current = -20.0
    hof = [50.0, 50.0, 50.0]
    assert _combine_fitnesses(current, hof, None)  > 0   # uniform: promoted (the bug)
    assert _combine_fitnesses(current, hof, 0.8)   < 0   # weighted: demoted (the fix)


def test_coevolve_weighted_hof_runs(trained_ckpt: Path, tmp_path: Path):
    """End-to-end: weighted HoF runs cleanly and produces history records."""
    from train.tools.coevolve import coevolve
    from agents.qd import MAPElitesConfig
    s = coevolve(
        seed_ckpt_dir=str(trained_ckpt),
        out_dir=str(tmp_path / "coev_weighted"),
        generations=2, n_mutants=2, sigma=0.3, eval_eps=1,
        n_predators=1, n_prey=1, n_obstacles=0, max_cycles=10,
        seed=66, device="cpu",
        qd_cfg=MAPElitesConfig(grid_shape=(3, 3)),
        hof_k=2, hof_eval_eps=1, hof_current_weight=0.7,
    )
    assert s["generations"] == 2
    assert len(s["history"]) == 4


def test_hall_of_fame_grows_each_generation(trained_ckpt: Path, tmp_path: Path):
    """With hof_k>0 the HoF should be populated and the fitness selection should
    still complete cleanly. We can't easily assert behavior is better, only that
    the wiring works end-to-end and the HoF sample size matches what's available."""
    from train.tools.coevolve import coevolve
    from agents.qd import MAPElitesConfig
    s = coevolve(
        seed_ckpt_dir=str(trained_ckpt),
        out_dir=str(tmp_path / "coev_hof"),
        generations=3, n_mutants=2, sigma=0.3, eval_eps=1,
        n_predators=1, n_prey=1, n_obstacles=0, max_cycles=10,
        seed=55, device="cpu",
        qd_cfg=MAPElitesConfig(grid_shape=(3, 3)),
        hof_k=2, hof_eval_eps=1,
    )
    # 3 gens × 2 teams = 6 records. HoF starts empty; by gen 2 it has 1 entry,
    # by gen 3 it has 2 entries — so hof_size_used should be 0, 0 (gen 1),
    # then 1, 1 (gen 2), then 2, 2 (gen 3).
    by_gen = {}
    for r in s["history"]:
        by_gen.setdefault(r["generation"], []).append(r["hof_size_used"])
    assert by_gen[1] == [0, 0]                  # HoF empty at gen 1
    assert all(x >= 1 for x in by_gen[2])       # HoF has at least 1 by gen 2
    assert all(x >= 2 for x in by_gen[3])       # HoF has at least 2 by gen 3


def test_coevolve_runs_on_grid_env(tmp_path: Path):
    """End-to-end: coevolve on a tiny grid env with food.

    Trains a brief PPO seed on the grid env, then runs 2 generations of
    coevolution against it. Catches regressions in the env_kwargs
    plumbing (kind/width/height/n_food/reward args through coevolve,
    its parallel worker payload, _evolve_one_team, and the final
    cross-generation tournament).
    """
    import run_cpu
    from train.tools.coevolve import coevolve
    from agents.qd import MAPElitesConfig

    # Train a grid seed ckpt inline.
    cfg = tmp_path / "grid_seed.yaml"
    cfg.write_text("""\
seed: 0
env:
  kind: grid
  width: 8
  height: 8
  n_predators: 1
  n_prey: 1
  n_obstacles: 1
  n_food: 2
  max_steps: 10
  catch_reward: 25.0
  food_reward: 15.0
  step_cost: 0.01
train:
  algo: ppo
  total_episodes: 2
  device: cpu
  hidden: 16
  batch_size: 64
  minibatch_size: 16
  update_epochs: 1
logging: {save_dir: artifacts/, plot_every: 1}
recording: {enabled: false}
checkpoint: {enabled: true, every: 0}
qd: {enabled: false}
novelty: {enabled: false}
""")
    out = Path(run_cpu.main(str(cfg), save_dir=str(tmp_path / "seed_runs")))
    seed_ckpt = out / "checkpoints" / "final"
    assert (seed_ckpt / "manifest.json").exists()

    s = coevolve(
        seed_ckpt_dir=str(seed_ckpt),
        out_dir=str(tmp_path / "coev_grid"),
        generations=2, n_mutants=2, sigma=0.3, eval_eps=1,
        n_predators=1, n_prey=1, n_obstacles=1, max_cycles=10,
        seed=77, device="cpu",
        qd_cfg=MAPElitesConfig(grid_shape=(3, 3)),
        kind="grid", width=8, height=8, n_food=2,
        catch_reward=25.0, food_reward=15.0, step_cost=0.01,
    )
    assert s["generations"] == 2
    coev = tmp_path / "coev_grid"
    assert (coev / "gen_001" / "champion_checkpoint" / "manifest.json").exists()
    assert (coev / "gen_002" / "champion_checkpoint" / "manifest.json").exists()
    # The cross-gen tournament inside _wrap_up must also use grid env, else
    # it would shape-mismatch the obs_dim=29 grid ckpt against a 14-dim MPE env.
    assert (coev / "champion_tournament" / "tournament.csv").exists()


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
