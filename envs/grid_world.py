"""Grid-world predator-prey env with obstacles and respawning food.

This is the Phase 9 graduation from MPE ``simple_tag``: arbitrary arena
size, more obstacles, and a third entity type (food) that prey try to
eat for positive reward. The intent is a richer ecosystem where prey
have a goal of their own — not just running away — so the arms race
isn't purely "chase vs evade."

API: implements the subset of PettingZoo's parallel_env interface that
the rest of the project consumes (``reset``, ``step``, ``action_space``,
``render``, ``close``, ``agents``). Action space is the same 5-discrete
(stay / up / down / left / right) the MPE envs use, so existing PPO code
just works.

Naming matches MPE simple_tag (``adversary_*`` for predators,
``agent_*`` for prey) so ``envs.predator_prey.team_of`` works unchanged.

Observation per agent (same dim for both teams — both see the full
world state):
    [self_x_norm, self_y_norm, self_is_predator,
     for each other agent: dx_norm, dy_norm, is_predator,
     for each obstacle: dx_norm, dy_norm,
     for each food: dx_norm, dy_norm]
total = 3 + 3*(n_agents-1) + 2*n_obstacles + 2*n_food

Rewards per step (continuous, not sparse):
    - predator: +catch_reward per prey it is co-located with
    - prey:     -catch_reward per predator it is co-located with
    - prey:     +food_reward per food on its cell (food respawns elsewhere)
    - all:      -step_cost (small per-step pressure to act)

The env is fully deterministic given a seed; ``reset(seed=S)`` followed
by the same action sequence reproduces obs/rewards/food respawns
exactly, so trajectory replay still works.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Action constants — match the 5-discrete MPE convention so a PPO trained
# on MPE could (in principle) be loaded into a grid env without action-space
# remapping. The semantics differ (MPE is continuous velocity; we're
# grid-discrete) but the action_dim is the same.
STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
N_ACTIONS = 5

DELTAS = {
    STAY:  (0, 0),
    UP:    (0, -1),
    DOWN:  (0,  1),
    LEFT:  (-1, 0),
    RIGHT: (1,  0),
}


class _DiscreteSpace:
    """Minimal stand-in for ``gymnasium.spaces.Discrete`` — we only need ``.n``."""
    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return int(np.random.randint(0, self.n))


class _BoxSpace:
    """Minimal stand-in for ``gymnasium.spaces.Box`` — we only need ``.shape``."""
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = tuple(shape)


class GridWorldEnv:
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        n_predators: int = 2,
        n_prey: int = 2,
        n_obstacles: int = 4,
        n_food: int = 4,
        max_cycles: int = 200,
        catch_reward: float = 10.0,
        food_reward: float = 5.0,
        step_cost: float = 0.05,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if width * height < n_predators + n_prey + n_obstacles + n_food:
            raise ValueError(
                f"Grid {width}x{height} = {width*height} cells is too small "
                f"to hold {n_predators} predators + {n_prey} prey + {n_obstacles} "
                f"obstacles + {n_food} food."
            )
        self.width = int(width)
        self.height = int(height)
        self.n_predators = int(n_predators)
        self.n_prey = int(n_prey)
        self.n_obstacles = int(n_obstacles)
        self.n_food = int(n_food)
        self.max_cycles = int(max_cycles)
        self.catch_reward = float(catch_reward)
        self.food_reward = float(food_reward)
        self.step_cost = float(step_cost)
        self.render_mode = render_mode

        self.predator_names = [f"adversary_{i}" for i in range(self.n_predators)]
        self.prey_names = [f"agent_{i}" for i in range(self.n_prey)]
        self.agents: List[str] = self.predator_names + self.prey_names
        self.possible_agents = list(self.agents)

        self.positions: Dict[str, Tuple[int, int]] = {}
        self.obstacles: List[Tuple[int, int]] = []
        self.food: List[Tuple[int, int]] = []
        self.step_count = 0
        self._rng = np.random.default_rng(seed)

        n_others = len(self.agents) - 1
        self._obs_dim = 3 + 3 * n_others + 2 * self.n_obstacles + 2 * self.n_food
        self._act_space = _DiscreteSpace(N_ACTIONS)
        self._obs_space = _BoxSpace((self._obs_dim,))

    def action_space(self, agent: str):
        return self._act_space

    def observation_space(self, agent: str):
        return self._obs_space

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def _free_cells(self, exclude) -> List[Tuple[int, int]]:
        excl = set(exclude)
        return [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in excl]

    def _sample_cells(self, n: int, exclude) -> List[Tuple[int, int]]:
        free = self._free_cells(exclude)
        if len(free) < n:
            raise ValueError(f"need {n} free cells, only {len(free)} available")
        idxs = self._rng.choice(len(free), size=n, replace=False)
        return [free[int(i)] for i in idxs]

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0

        self.obstacles = self._sample_cells(self.n_obstacles, exclude=set())
        occupied = set(self.obstacles)

        # Place agents (predators first, then prey) — order matters for determinism.
        self.positions = {}
        for name in self.agents:
            pos = self._sample_cells(1, exclude=occupied)[0]
            self.positions[name] = pos
            occupied.add(pos)

        # Food on remaining cells.
        self.food = self._sample_cells(self.n_food, exclude=occupied)

        return {a: self._observe(a) for a in self.agents}

    def step(self, actions: Dict[str, int]):
        self.step_count += 1

        # Apply moves. Bounds-clamp; refuse to walk into obstacles (stay in place).
        # Agents *can* overlap (that's how catches happen). Move predators first,
        # then prey, so a prey can dodge into a freshly-vacated cell — but since
        # agents can overlap anyway this ordering only affects edge cases at walls.
        for name in self.agents:
            a = int(actions.get(name, STAY))
            dx, dy = DELTAS.get(a, (0, 0))
            x, y = self.positions[name]
            nx = max(0, min(self.width - 1, x + dx))
            ny = max(0, min(self.height - 1, y + dy))
            if (nx, ny) in self.obstacles:
                # Bumped a wall; stay put. Slightly punitive via the step_cost only.
                continue
            self.positions[name] = (nx, ny)

        rewards = {a: -self.step_cost for a in self.agents}

        # Catches: any predator on same cell as a prey triggers ±catch_reward.
        # We count multi-catches (two predators on one prey = 2x reward to predators,
        # 2x penalty to that prey) — matches MPE simple_tag's per-collision logic.
        for pred in self.predator_names:
            ppos = self.positions[pred]
            for prey in self.prey_names:
                if ppos == self.positions[prey]:
                    rewards[pred] += self.catch_reward
                    rewards[prey] -= self.catch_reward

        # Food: each prey on a food cell eats it; the food respawns elsewhere.
        # Iterate from the end so pop() doesn't invalidate indices.
        eaten_idx: List[int] = []
        for i, food_pos in enumerate(self.food):
            for prey in self.prey_names:
                if self.positions[prey] == food_pos:
                    rewards[prey] += self.food_reward
                    eaten_idx.append(i)
                    break
        if eaten_idx:
            for i in sorted(eaten_idx, reverse=True):
                self.food.pop(i)
            occupied = set(self.obstacles) | set(self.positions.values()) | set(self.food)
            new_food = self._sample_cells(len(eaten_idx), exclude=occupied)
            self.food.extend(new_food)

        truncated = self.step_count >= self.max_cycles
        terminations = {a: False for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {a: {} for a in self.agents}
        next_obs = {a: self._observe(a) for a in self.agents}
        return next_obs, rewards, terminations, truncations, infos

    def _observe(self, agent_name: str) -> np.ndarray:
        ax, ay = self.positions[agent_name]
        is_pred = 1.0 if agent_name in self.predator_names else 0.0
        nx, ny = float(self.width), float(self.height)
        out: List[float] = [ax / nx, ay / ny, is_pred]
        for other in self.agents:
            if other == agent_name:
                continue
            ox, oy = self.positions[other]
            out.extend([(ox - ax) / nx, (oy - ay) / ny,
                        1.0 if other in self.predator_names else 0.0])
        for ox, oy in self.obstacles:
            out.extend([(ox - ax) / nx, (oy - ay) / ny])
        for fx, fy in self.food:
            out.extend([(fx - ax) / nx, (fy - ay) / ny])
        return np.asarray(out, dtype=np.float32)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        # Matplotlib render — figure size scales with grid so a 40x40 arena
        # actually looks big in the output frame.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_w = max(4.0, self.width / 4.0)
        fig_h = max(4.0, self.height / 4.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=80)

        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # (0,0) top-left, like an image
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f5f5f0")

        # Light grid lines.
        for x in range(self.width + 1):
            ax.axvline(x - 0.5, color="#dddddd", linewidth=0.3, zorder=0)
        for y in range(self.height + 1):
            ax.axhline(y - 0.5, color="#dddddd", linewidth=0.3, zorder=0)

        for (ox, oy) in self.obstacles:
            ax.add_patch(plt.Rectangle((ox - 0.45, oy - 0.45), 0.9, 0.9,
                                       facecolor="#555555", edgecolor="black", zorder=1))
        if self.food:
            fx = [p[0] for p in self.food]; fy = [p[1] for p in self.food]
            ax.scatter(fx, fy, s=140, c="#5fcf3f", marker="*",
                       edgecolors="darkgreen", linewidths=0.8, zorder=2)

        for name in self.predator_names:
            x, y = self.positions[name]
            ax.scatter([x], [y], s=240, c="#e63946", marker="o",
                       edgecolors="black", linewidths=1.2, zorder=4)
        for name in self.prey_names:
            x, y = self.positions[name]
            ax.scatter([x], [y], s=240, c="#3a86ff", marker="o",
                       edgecolors="black", linewidths=1.2, zorder=4)

        ax.set_title(
            f"grid {self.width}x{self.height}  step {self.step_count}/{self.max_cycles}  "
            f"pred={self.n_predators} prey={self.n_prey} obs={self.n_obstacles} food={self.n_food}",
            fontsize=9,
        )
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        rgb = buf[:, :, :3].copy()
        plt.close(fig)
        return rgb

    def close(self):
        return None
