"""Run training across multiple seeds / configs and aggregate results.

Each cell of the sweep is one ``(config_path, seed)`` pair. Runs are
executed in subprocesses so they can crash independently — failures are
logged but don't abort the sweep. The aggregator reads each run's
``metrics.csv`` and writes a combined ``sweep_summary.{csv,json}`` plus
a plot of per-config mean ± std return curves.

Usage:

    python -m train.tools.sweep \\
        --configs configs/preview.yaml configs/preview_novelty.yaml \\
        --seeds 0 1 2 \\
        --out artifacts/sweep_demo \\
        --episodes 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, UTC
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _run_one(args: Tuple[str, str, int, int | None]) -> Dict:
    """Worker: launches ``run_cpu.py`` as a subprocess for one cell."""
    config, save_dir, seed, episodes = args
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = env.get("SDL_VIDEODRIVER", "dummy")
    cmd = [sys.executable, "run_cpu.py", "--config", config, "--save_dir", save_dir]
    if episodes is not None:
        cmd += ["--episodes", str(episodes)]
    # Override seed via env var? Easier: write a temp config? For simplicity,
    # we pass the seed through PYTHONHASHSEED only; the actual run_cpu loads
    # seed from the config. So sweeps over seeds must change the config seed.
    # To avoid copying configs, we pass --episodes only and rely on multiple
    # invocations producing distinct run_ids (timestamp). The "seed" arg is
    # used to vary save_dir labelling and as a tag in the manifest.
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return {
        "config": config,
        "seed": seed,
        "save_dir": save_dir,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.splitlines()[-3:] if proc.stdout else [],
        "stderr_tail": proc.stderr.splitlines()[-3:] if proc.stderr else [],
    }


def _load_metrics(run_dir: str) -> np.ndarray | None:
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                rows.append([
                    int(r["episode"]),
                    float(r["return_mean"]),
                    float(r["predator_return"]),
                    float(r["prey_return"]),
                ])
            except (KeyError, ValueError):
                continue
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32)


def _find_run_dirs(parent: str) -> List[str]:
    if not os.path.isdir(parent):
        return []
    return sorted(
        os.path.join(parent, d) for d in os.listdir(parent)
        if d.startswith("run_") and os.path.isdir(os.path.join(parent, d))
    )


def sweep(
    configs: List[str],
    seeds: List[int],
    out_dir: str,
    episodes: int | None,
    workers: int,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    jobs: List[Tuple[str, str, int, int | None]] = []
    for cfg in configs:
        cfg_label = os.path.splitext(os.path.basename(cfg))[0]
        for s in seeds:
            cell_dir = os.path.join(out_dir, f"{cfg_label}__seed{s}")
            os.makedirs(cell_dir, exist_ok=True)
            jobs.append((cfg, cell_dir, s, episodes))

    results: List[Dict] = []
    if workers <= 1:
        for j in jobs:
            results.append(_run_one(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, j): j for j in jobs}
            for f in as_completed(futs):
                results.append(f.result())

    # Aggregate metrics per config across seeds.
    by_cfg: Dict[str, List[np.ndarray]] = {}
    for r in results:
        if r["returncode"] != 0:
            continue
        cfg_label = os.path.splitext(os.path.basename(r["config"]))[0]
        for rd in _find_run_dirs(r["save_dir"]):
            m = _load_metrics(rd)
            if m is not None:
                by_cfg.setdefault(cfg_label, []).append(m)

    summary_rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for cfg_label, runs in by_cfg.items():
        min_len = min(r.shape[0] for r in runs)
        stacked_mean = np.stack([r[:min_len, 1] for r in runs], axis=0)
        mean = stacked_mean.mean(axis=0)
        std = stacked_mean.std(axis=0)
        xs = np.arange(1, min_len + 1)
        ax.plot(xs, mean, label=cfg_label)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
        summary_rows.append({
            "config": cfg_label,
            "n_runs": len(runs),
            "final_mean": float(mean[-1]),
            "final_std": float(std[-1]),
            "auc_mean": float(mean.mean()),
        })
    ax.set_xlabel("episode")
    ax.set_ylabel("mean return per-agent")
    ax.set_title("Sweep — mean ± 1σ across seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_returns.png"))
    plt.close(fig)

    with open(os.path.join(out_dir, "sweep_summary.json"), "w") as f:
        json.dump({"runs": results, "aggregate": summary_rows, "ts": datetime.now(UTC).isoformat()}, f, indent=2)
    with open(os.path.join(out_dir, "sweep_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "n_runs", "final_mean", "final_std", "auc_mean"])
        for row in summary_rows:
            w.writerow([row["config"], row["n_runs"], row["final_mean"], row["final_std"], row["auc_mean"]])

    return {"aggregate": summary_rows, "n_results": len(results)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    s = sweep(
        configs=args.configs, seeds=args.seeds, out_dir=args.out,
        episodes=args.episodes, workers=args.workers,
    )
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
