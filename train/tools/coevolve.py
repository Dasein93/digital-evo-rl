"""Co-evolutionary arms race between predators and prey.

Each generation alternates two phases:

  1. Predator phase — spawn ``n_mutants`` Gaussian-perturbed copies of
     the current predator champion, greedy-evaluate each against the
     current prey champion, insert into a per-generation MAP-Elites
     archive. The new champion is the highest-fitness elite.
  2. Prey phase — symmetric, with prey mutating against the freshly
     promoted predator champion.

This is the actual digital-evolution loop the project's name promises:
each side is selection-pressured by the other side's *latest* policy, so
strategies should escalate (Red Queen dynamics) within the limits of the
mutation operator and the env. The first generation seeds from a saved
PPO checkpoint (or random init), and a champion-only cross-generation
tournament at the end visualises the arms race as a heatmap.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.checkpoint import load_checkpoint, save_checkpoint
from agents.novelty import BehaviorCharacteristic
from agents.qd import MAPElitesArchive, MAPElitesConfig
from envs.predator_prey import make_env, reset as env_reset, step as env_step, team_of, TEAM_PREDATOR, TEAM_PREY
from train.ppo import PPO, PPOConfig, flatten_obs, set_seed, set_num_threads
from train.tools.plots import archive_heatmap, tournament_heatmap


def _split_by_team(agents, obs_list):
    by_team: Dict[str, List[int]] = {}
    for i, a in enumerate(agents):
        by_team.setdefault(team_of(a), []).append(i)
    return {
        team: (np.stack([obs_list[i] for i in idxs], axis=0), [agents[i] for i in idxs], idxs)
        for team, idxs in by_team.items()
    }


def _mutate(sd: Dict[str, torch.Tensor], sigma: float, rng: torch.Generator) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if v.dtype.is_floating_point:
            out[k] = v + torch.empty_like(v).normal_(generator=rng) * sigma
        else:
            out[k] = v.clone()
    return out


def _eval_mutant_worker(payload: Dict) -> Tuple[float, List[float]]:
    """Worker entrypoint for parallel mutant evaluation.

    Each worker constructs its own env + PPOs from state_dicts so nothing
    is shared with the parent (avoids pickling the PPO objects directly,
    and keeps env state isolated).
    """
    set_num_threads(1)
    target_team = payload["target_team"]
    sd_target = {k: torch.tensor(v) for k, v in payload["sd_target"].items()}
    sd_opponent = {k: torch.tensor(v) for k, v in payload["sd_opponent"].items()}
    teams_meta = payload["teams_meta"]
    cfg_dict = teams_meta[target_team]["cfg"]
    opp_team = TEAM_PREY if target_team == TEAM_PREDATOR else TEAM_PREDATOR
    cfg_opp = teams_meta[opp_team]["cfg"]

    ppos = {
        target_team: PPO(int(teams_meta[target_team]["obs_dim"]),
                         int(teams_meta[target_team]["act_dim"]),
                         PPOConfig(**cfg_dict), device="cpu"),
        opp_team: PPO(int(teams_meta[opp_team]["obs_dim"]),
                      int(teams_meta[opp_team]["act_dim"]),
                      PPOConfig(**cfg_opp), device="cpu"),
    }
    ppos[target_team].ac.load_state_dict(sd_target)
    ppos[opp_team].ac.load_state_dict(sd_opponent)

    env = make_env(
        n_predators=payload["n_predators"], n_prey=payload["n_prey"],
        n_obstacles=payload["n_obstacles"], max_cycles=payload["max_cycles"],
        seed=payload["seed"],
    )
    try:
        fit, bc = _rollout(env, ppos, payload["eval_eps"], seed=payload["seed"],
                           target_team=target_team, device="cpu")
    finally:
        env.close()
    return float(fit), [float(x) for x in bc]


def _state_dict_to_lists(sd: Dict[str, torch.Tensor]) -> Dict[str, List]:
    return {k: v.detach().cpu().tolist() for k, v in sd.items()}


def _teams_meta_from_ppos(ppos: Dict[str, PPO]) -> Dict[str, Dict]:
    from dataclasses import asdict
    return {
        team: {"obs_dim": p.obs_dim, "act_dim": p.act_dim, "cfg": asdict(p.cfg)}
        for team, p in ppos.items()
    }


def _rollout(env, ppos, n_episodes, seed, target_team, device) -> Tuple[float, np.ndarray]:
    """Greedy rollout. Returns (mean fitness for target team, mean team BC)."""
    fits, bcs = [], []
    for ep in range(n_episodes):
        obs = env_reset(env, seed=seed + ep)
        per_agent_obs: Dict[str, List[np.ndarray]] = {}
        per_agent_acts: Dict[str, List[int]] = {}
        per_agent_rews: Dict[str, float] = {}
        done_any = False
        while not done_any:
            obs_list, step_agents = flatten_obs(obs)
            per_team = _split_by_team(step_agents, obs_list)
            acts: Dict[str, int] = {}
            with torch.no_grad():
                for team, (team_obs, team_agents, _idxs) in per_team.items():
                    a_t, _logp, _v = ppos[team].ac.step(torch.from_numpy(team_obs).to(device), greedy=True)
                    for k, ag in enumerate(team_agents):
                        acts[ag] = int(a_t[k].item())
            for i, ag in enumerate(step_agents):
                per_agent_obs.setdefault(ag, []).append(obs_list[i])
                per_agent_acts.setdefault(ag, []).append(int(acts[ag]))
            obs, rewards, done_any, _ = env_step(env, acts)
            for ag in step_agents:
                per_agent_rews[ag] = per_agent_rews.get(ag, 0.0) + float(rewards[ag])
        target_agents = [a for a in per_agent_rews if team_of(a) == target_team]
        if not target_agents:
            continue
        fits.append(float(np.mean([per_agent_rews[a] for a in target_agents])))
        per_agent_bc = {
            a: BehaviorCharacteristic.from_agent(per_agent_obs.get(a, []), per_agent_acts.get(a, []), n_actions=5)
            for a in target_agents
        }
        bcs.append(BehaviorCharacteristic.team(per_agent_bc))
    if not fits:
        return 0.0, np.zeros(4, dtype=np.float32)
    return float(np.mean(fits)), np.mean(np.stack(bcs, axis=0), axis=0).astype(np.float32)


def _evolve_one_team(
    env,
    ppos: Dict[str, PPO],
    target_team: str,
    n_mutants: int,
    sigma: float,
    eval_eps: int,
    seed: int,
    device: str,
    archive: MAPElitesArchive,
    rng: torch.Generator,
    workers: int = 1,
    n_predators: int = 2,
    n_prey: int = 2,
    n_obstacles: int = 0,
    max_cycles: int = 100,
) -> Dict:
    """One generation phase for one team. Returns the new champion's metadata."""
    base_sd = deepcopy(ppos[target_team].ac.state_dict())
    # Evaluate the incumbent so the champion stays competitive if all mutants regress.
    incumbent_fit, incumbent_bc = _rollout(env, ppos, eval_eps, seed=seed, target_team=target_team, device=device)
    archive.try_insert(incumbent_bc, incumbent_fit, deepcopy(base_sd), label="incumbent")
    best_fit = incumbent_fit
    best_sd = base_sd
    best_label = "incumbent"
    best_bc = incumbent_bc

    opp_team = TEAM_PREY if target_team == TEAM_PREDATOR else TEAM_PREDATOR
    opp_sd = ppos[opp_team].ac.state_dict()

    mutant_sds = [_mutate(base_sd, sigma=sigma, rng=rng) for _ in range(n_mutants)]
    if workers > 1 and n_mutants > 0:
        # Serialize state_dicts to plain Python so the worker pool can pickle them.
        teams_meta = _teams_meta_from_ppos(ppos)
        payloads = [
            {
                "target_team": target_team,
                "sd_target": _state_dict_to_lists(sd_k),
                "sd_opponent": _state_dict_to_lists(opp_sd),
                "teams_meta": teams_meta,
                "n_predators": n_predators, "n_prey": n_prey,
                "n_obstacles": n_obstacles, "max_cycles": max_cycles,
                "eval_eps": eval_eps, "seed": seed + k * 17 + 1,
            }
            for k, sd_k in enumerate(mutant_sds)
        ]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_eval_mutant_worker, payloads))
        for k, (fit_k, bc_k_list) in enumerate(results):
            bc_k = np.asarray(bc_k_list, dtype=np.float32)
            archive.try_insert(bc_k, fit_k, deepcopy(mutant_sds[k]), label=f"mut_{k:04d}")
            if fit_k > best_fit:
                best_fit = fit_k
                best_sd = mutant_sds[k]
                best_label = f"mut_{k:04d}"
                best_bc = bc_k
    else:
        for k, sd_k in enumerate(mutant_sds):
            ppos[target_team].ac.load_state_dict(sd_k)
            fit_k, bc_k = _rollout(env, ppos, eval_eps, seed=seed + k * 17 + 1, target_team=target_team, device=device)
            archive.try_insert(bc_k, fit_k, deepcopy(sd_k), label=f"mut_{k:04d}")
            if fit_k > best_fit:
                best_fit = fit_k
                best_sd = sd_k
                best_label = f"mut_{k:04d}"
                best_bc = bc_k
    ppos[target_team].ac.load_state_dict(best_sd)
    return {
        "team": target_team,
        "champion_fitness": float(best_fit),
        "champion_label": best_label,
        "champion_bc": [float(x) for x in best_bc],
        "incumbent_fitness": float(incumbent_fit),
        "coverage": archive.coverage(),
    }


def _find_resume_point(out_dir: str) -> Tuple[int, str | None, List[Dict]]:
    """Look for the highest finished gen_NNN/champion_checkpoint/ in ``out_dir``.

    Returns (next_gen_to_run, latest_champion_ckpt or None, history-so-far).
    A generation is considered "finished" iff its champion_checkpoint has a
    manifest.json. Partial generations (in-progress crashes) are restarted.
    """
    if not os.path.isdir(out_dir):
        return 1, None, []
    gens = []
    for path in glob.glob(os.path.join(out_dir, "gen_*")):
        m = re.match(r"gen_(\d+)$", os.path.basename(path))
        if not m:
            continue
        g = int(m.group(1))
        ck_manifest = os.path.join(path, "champion_checkpoint", "manifest.json")
        if os.path.exists(ck_manifest):
            gens.append((g, os.path.join(path, "champion_checkpoint")))
    if not gens:
        return 1, None, []
    gens.sort()
    last_g, last_ckpt = gens[-1]
    # Rebuild history from saved coevolve_summary.json if it exists.
    history: List[Dict] = []
    summary_path = os.path.join(out_dir, "coevolve_summary.json")
    if os.path.exists(summary_path):
        try:
            history = json.load(open(summary_path)).get("history", [])
        except Exception:
            history = []
    return last_g + 1, last_ckpt, history


def coevolve(
    seed_ckpt_dir: str,
    out_dir: str,
    generations: int,
    n_mutants: int,
    sigma: float,
    eval_eps: int,
    n_predators: int,
    n_prey: int,
    n_obstacles: int,
    max_cycles: int,
    seed: int,
    device: str,
    qd_cfg: MAPElitesConfig | None = None,
    workers: int = 1,
    resume: bool = False,
    num_threads: int | None = None,
) -> Dict:
    if num_threads is not None:
        set_num_threads(num_threads)
    set_seed(seed)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    start_gen = 1
    ckpt_to_load = seed_ckpt_dir
    history: List[Dict] = []
    if resume:
        start_gen, latest_ckpt, history = _find_resume_point(out_dir)
        if latest_ckpt is not None:
            ckpt_to_load = latest_ckpt
            print(f"[resume] continuing from gen {start_gen}, seed ckpt = {ckpt_to_load}")
            if start_gen > generations:
                print(f"[resume] already past requested generations ({start_gen-1} >= {generations}); nothing to do")
                return {"generations": start_gen - 1, "history": history, "resumed": True, "no_op": True}

    ppos, _ = load_checkpoint(ckpt_to_load, device=device)
    env = make_env(
        n_predators=n_predators, n_prey=n_prey, n_obstacles=n_obstacles,
        max_cycles=max_cycles, seed=seed,
    )

    archives: Dict[Tuple[int, str], MAPElitesArchive] = {}
    champion_ckpts: List[Tuple[int, str]] = []  # (gen, ckpt_dir)
    # Re-populate champion_ckpts list AND archives from any already-completed
    # generations on resume so the wrap-up plots include the full history.
    if resume and start_gen > 1:
        for g in range(1, start_gen):
            ck = os.path.join(out_dir, f"gen_{g:03d}", "champion_checkpoint")
            if os.path.exists(os.path.join(ck, "manifest.json")):
                champion_ckpts.append((g, ck))
            for team in (TEAM_PREDATOR, TEAM_PREY):
                qd_dir = os.path.join(out_dir, f"gen_{g:03d}", "qd", team)
                if os.path.exists(os.path.join(qd_dir, "manifest.json")):
                    archives[(g, team)] = MAPElitesArchive.load(qd_dir)

    base_qd = qd_cfg or MAPElitesConfig()
    try:
        for g in range(start_gen, generations + 1):
            gen_dir = os.path.join(out_dir, f"gen_{g:03d}")
            os.makedirs(gen_dir, exist_ok=True)
            phase_records: List[Dict] = []
            for team in (TEAM_PREDATOR, TEAM_PREY):
                arch = MAPElitesArchive(base_qd, save_dir=os.path.join(gen_dir, "qd", team))
                archives[(g, team)] = arch
                rec = _evolve_one_team(
                    env, ppos, target_team=team,
                    n_mutants=n_mutants, sigma=sigma, eval_eps=eval_eps,
                    seed=seed + g * 1000 + (0 if team == TEAM_PREDATOR else 500),
                    device=device, archive=arch, rng=rng,
                    workers=workers, n_predators=n_predators, n_prey=n_prey,
                    n_obstacles=n_obstacles, max_cycles=max_cycles,
                )
                arch.save_manifest()
                rec["generation"] = g
                phase_records.append(rec)
                print(
                    f"[gen {g}/{generations}] {team:8s} champion fitness "
                    f"{rec['champion_fitness']:+.2f} (incumbent {rec['incumbent_fitness']:+.2f}, "
                    f"coverage {rec['coverage']:.0%})"
                )
            # Snapshot both champions as a single tournament-ready ckpt.
            ck = os.path.join(gen_dir, "champion_checkpoint")
            save_checkpoint(ck, ppos, extra={"generation": g})
            champion_ckpts.append((g, ck))
            history.extend(phase_records)
            # Persist incremental progress so --resume can pick up after crash.
            with open(os.path.join(out_dir, "coevolve_summary.json"), "w") as f:
                json.dump({"history": history, "in_progress": True, "last_gen": g}, f, indent=2)
    finally:
        env.close()

    summary = _wrap_up(out_dir, archives, history, champion_ckpts, base_qd,
                       n_predators, n_prey, n_obstacles, max_cycles, seed, device)
    return summary


def _wrap_up(
    out_dir, archives, history, champion_ckpts, qd_cfg,
    n_predators, n_prey, n_obstacles, max_cycles, seed, device,
) -> Dict:
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Per-team fitness over generations.
    gens = sorted({h["generation"] for h in history})
    pred = [next(h["champion_fitness"] for h in history if h["generation"] == g and h["team"] == TEAM_PREDATOR) for g in gens]
    prey = [next(h["champion_fitness"] for h in history if h["generation"] == g and h["team"] == TEAM_PREY) for g in gens]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gens, pred, marker="o", label="predator champion")
    ax.plot(gens, prey, marker="o", label="prey champion")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
    ax.set_xlabel("generation"); ax.set_ylabel("champion fitness (vs current opponent)")
    ax.set_title("Co-evolutionary arms race")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "coevolution_fitness.png")); plt.close(fig)

    # Coverage over generations.
    cov_pred = [archives[(g, TEAM_PREDATOR)].coverage() for g in gens]
    cov_prey = [archives[(g, TEAM_PREY)].coverage() for g in gens]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gens, cov_pred, marker="s", label="predator archive")
    ax.plot(gens, cov_prey, marker="s", label="prey archive")
    ax.set_xlabel("generation"); ax.set_ylabel("MAP-Elites coverage")
    ax.set_title("Behavioral diversity per generation")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "coevolution_coverage.png")); plt.close(fig)

    # Cross-generation champion tournament (every champion's predator vs every champion's prey).
    from train.tools.tournament import tournament
    pred_entries = [(ck, f"g{g}") for (g, ck) in champion_ckpts]
    prey_entries = [(ck, f"g{g}") for (g, ck) in champion_ckpts]
    tour_summary = tournament(
        pred_entries=pred_entries, prey_entries=prey_entries,
        out_dir=os.path.join(out_dir, "champion_tournament"),
        episodes=3, n_predators=n_predators, n_prey=n_prey, n_obstacles=n_obstacles,
        max_cycles=max_cycles, seed=seed + 99_991, device=device,
    )

    summary = {
        "generations": len(gens),
        "history": history,
        "champion_ckpts": champion_ckpts,
        "qd_cfg": {
            "bc_dims": list(qd_cfg.bc_dims),
            "bc_bounds": [list(b) for b in qd_cfg.bc_bounds],
            "grid_shape": list(qd_cfg.grid_shape),
        },
        "tournament_summary": tour_summary,
        "env": {
            "n_predators": n_predators, "n_prey": n_prey,
            "n_obstacles": n_obstacles, "max_cycles": max_cycles,
        },
    }
    with open(os.path.join(out_dir, "coevolve_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_ckpt", required=True, help="Initial checkpoint for both teams")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--n_mutants", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--eval_eps", type=int, default=2)
    ap.add_argument("--n_predators", type=int, default=2)
    ap.add_argument("--n_prey", type=int, default=2)
    ap.add_argument("--n_obstacles", type=int, default=0)
    ap.add_argument("--max_cycles", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1, help="parallel mutant evaluators (set to n_cores on a low-core VPS)")
    ap.add_argument("--resume", action="store_true", help="continue from the highest completed gen_NNN in --out")
    ap.add_argument("--num_threads", type=int, default=None, help="cap PyTorch CPU threads (use 1-2 on a 2-vCPU VPS)")
    args = ap.parse_args()
    s = coevolve(
        seed_ckpt_dir=args.seed_ckpt, out_dir=args.out,
        generations=args.generations, n_mutants=args.n_mutants, sigma=args.sigma,
        eval_eps=args.eval_eps,
        n_predators=args.n_predators, n_prey=args.n_prey, n_obstacles=args.n_obstacles,
        max_cycles=args.max_cycles, seed=args.seed, device=args.device,
        workers=args.workers, resume=args.resume, num_threads=args.num_threads,
    )
    print(json.dumps({k: v for k, v in s.items() if k not in {"history", "tournament_summary"}}, indent=2))


if __name__ == "__main__":
    main()
