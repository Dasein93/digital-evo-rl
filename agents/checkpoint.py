"""Save/load PPO policies as a self-describing checkpoint bundle.

A checkpoint is one ``.pt`` per team plus a ``manifest.json`` that records
obs/act dims and the ``PPOConfig`` so the policies can be reconstructed
without the original training script.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Tuple

import torch

from train.ppo import PPO, PPOConfig


def save_checkpoint(
    ckpt_dir: str,
    ppos: Dict[str, PPO],
    extra: Dict | None = None,
) -> str:
    """Write a checkpoint bundle for the given team -> PPO mapping.

    Returns the path to ``manifest.json``.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    teams_meta = {}
    for team, ppo in ppos.items():
        torch.save(ppo.ac.state_dict(), os.path.join(ckpt_dir, f"{team}.pt"))
        teams_meta[team] = {
            "obs_dim": int(ppo.obs_dim),
            "act_dim": int(ppo.act_dim),
            "cfg": asdict(ppo.cfg),
        }
    manifest = {"teams": teams_meta}
    if extra:
        manifest["extra"] = extra
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_checkpoint(ckpt_dir: str, device: str = "cpu") -> Tuple[Dict[str, PPO], Dict]:
    """Reconstruct the team -> PPO mapping written by :func:`save_checkpoint`."""
    with open(os.path.join(ckpt_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
    ppos: Dict[str, PPO] = {}
    for team, meta in manifest["teams"].items():
        cfg = PPOConfig(**meta["cfg"])
        ppo = PPO(obs_dim=int(meta["obs_dim"]), act_dim=int(meta["act_dim"]), cfg=cfg, device=device)
        state = torch.load(os.path.join(ckpt_dir, f"{team}.pt"), map_location=device, weights_only=True)
        ppo.ac.load_state_dict(state)
        ppo.ac.eval()
        ppos[team] = ppo
    return ppos, manifest
