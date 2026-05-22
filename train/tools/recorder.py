"""Trajectory recorder.

Streams per-step transitions to JSONL (line-per-step, human-greppable) and
optionally a compact NPZ at the end of a run. ``sample_rate`` skips steps
(write every k-th step) to keep recordings small for long rollouts.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class TrajectoryRecorder:
    def __init__(
        self,
        out_dir: str,
        sample_rate: int = 1,
        also_npz: bool = True,
        seed: Optional[int] = None,
        env_meta: Optional[Dict[str, Any]] = None,
    ):
        assert sample_rate >= 1, "sample_rate must be >= 1"
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.sample_rate = sample_rate
        self.also_npz = also_npz
        self.seed = seed
        self.env_meta = env_meta or {}
        self.jsonl_path = os.path.join(out_dir, "trajectory.jsonl")
        self.npz_path = os.path.join(out_dir, "trajectory.npz")
        self._jsonl = open(self.jsonl_path, "w")
        self._step_idx = 0
        self._frames: List[Dict[str, Any]] = []
        header = {"type": "header", "seed": seed, "env": self.env_meta, "sample_rate": sample_rate}
        self._jsonl.write(json.dumps(header) + "\n")
        self._jsonl.flush()

    def record(
        self,
        episode: int,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        done: bool,
    ) -> None:
        if self._step_idx % self.sample_rate == 0:
            row = {
                "type": "step",
                "episode": int(episode),
                "step": int(self._step_idx),
                "actions": {a: int(v) for a, v in actions.items()},
                "rewards": {a: float(v) for a, v in rewards.items()},
                "done": bool(done),
                "obs": {a: np.asarray(o, dtype=np.float32).tolist() for a, o in obs.items()},
            }
            self._jsonl.write(json.dumps(row) + "\n")
            if self.also_npz:
                self._frames.append(
                    {
                        "episode": episode,
                        "step": self._step_idx,
                        "actions": {a: int(v) for a, v in actions.items()},
                        "done": bool(done),
                    }
                )
        self._step_idx += 1

    def close(self) -> None:
        if self._jsonl and not self._jsonl.closed:
            self._jsonl.flush()
            self._jsonl.close()
        if self.also_npz and self._frames:
            episodes = np.asarray([f["episode"] for f in self._frames], dtype=np.int32)
            steps = np.asarray([f["step"] for f in self._frames], dtype=np.int32)
            dones = np.asarray([f["done"] for f in self._frames], dtype=np.bool_)
            agents = sorted({a for f in self._frames for a in f["actions"]})
            actions = np.full((len(self._frames), len(agents)), -1, dtype=np.int32)
            for i, f in enumerate(self._frames):
                for j, a in enumerate(agents):
                    if a in f["actions"]:
                        actions[i, j] = f["actions"][a]
            np.savez_compressed(
                self.npz_path,
                episodes=episodes,
                steps=steps,
                dones=dones,
                actions=actions,
                agents=np.asarray(agents),
                seed=np.asarray([-1 if self.seed is None else self.seed], dtype=np.int64),
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
