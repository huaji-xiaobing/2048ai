"""TD(0) trainer for the afterstate value function.

Learning target
---------------
``V`` estimates the value of an **afterstate** (the board right after a slide
and merge, before a new tile is spawned).  Transitions are

    afterstate --spawn--> board --a ~ pi--> afterstate'   with reward r

so the update is the SARSA-style on-policy bootstrap

    ``V(s') <- V(s') + alpha * (r + gamma * V(s'') - V(s'))``

where ``a`` is drawn from the same (pluggable) policy that produced ``s''``.
A terminal board contributes ``V(s') <- V(s') + alpha * (r - V(s'))``.

Parallelism
-----------
:class:`Trainer` runs ``config.boards`` boards simultaneously with numpy: at
every step all four afterstates of every board are computed and evaluated in a
single batch.  :mod:`ntuple2048.parallel` runs several such trainers in
separate processes over one shared weight buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .env import NUM_DIRECTIONS, batch_move_all, batch_spawn, new_boards
from .lookup import BOARD_CELLS
from .network import DEFAULT_BASE, NTupleNetwork
from .policies import LinearSchedule, Policy, make_policy
from .symmetry import NUM_SYMMETRIES, transform_boards

NEG_INF = np.float32(-1e30)


@dataclass
class TrainConfig:
    """Everything the rollout loop needs.  Picklable, so it can cross processes."""

    base: int = DEFAULT_BASE
    boards: int = 256
    lr: float = 0.0025
    gamma: float = 0.999
    reward: str = "log"  # "log" (sum of merged exponents) or "score" (real 2048 score)
    d4_augment: bool = True
    seed: int = 0
    policy: str = "epsilon_greedy"
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000_000

    def build_policy(self) -> Policy:
        if self.policy == "epsilon_greedy":
            return make_policy(
                "epsilon_greedy",
                schedule=LinearSchedule(
                    start=self.epsilon_start,
                    end=self.epsilon_end,
                    decay_steps=self.epsilon_decay_steps,
                ),
            )
        return make_policy(self.policy)


@dataclass
class Stats:
    """Aggregated rollout statistics (all fields are sums)."""

    steps: int = 0
    episodes: int = 0
    score_sum: float = 0.0
    length_sum: int = 0
    best_exponent: int = 0
    td_error_sum: float = 0.0
    td_count: int = 0
    tile_counts: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        episodes = max(self.episodes, 1)
        return {
            "steps": self.steps,
            "episodes": self.episodes,
            "mean_score": self.score_sum / episodes,
            "mean_length": self.length_sum / episodes,
            "best_tile": (1 << self.best_exponent) if self.best_exponent else 0,
            "best_exponent": self.best_exponent,
            "mean_abs_td": self.td_error_sum / max(self.td_count, 1),
            "tile_counts": dict(sorted(self.tile_counts.items())),
        }


def merge_stats(items: list[Stats]) -> Stats:
    out = Stats()
    for item in items:
        out.steps += item.steps
        out.episodes += item.episodes
        out.score_sum += item.score_sum
        out.length_sum += item.length_sum
        out.best_exponent = max(out.best_exponent, item.best_exponent)
        out.td_error_sum += item.td_error_sum
        out.td_count += item.td_count
        for exponent, count in item.tile_counts.items():
            out.tile_counts[exponent] = out.tile_counts.get(exponent, 0) + count
    return out


class Trainer:
    """Single-process, multi-board TD trainer."""

    def __init__(
        self,
        network: NTupleNetwork,
        config: TrainConfig | None = None,
        policy: Policy | None = None,
        seed: int | None = None,
    ) -> None:
        self.net = network
        self.config = config or TrainConfig(base=network.base)
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.policy = policy or self.config.build_policy()
        self.n = int(self.config.boards)
        self.use_log = self.config.reward == "log"

        self.boards = new_boards(self.n, self.rng)
        self.ep_score = np.zeros(self.n, dtype=np.float64)
        self.ep_len = np.zeros(self.n, dtype=np.int64)
        self.ep_max = np.zeros(self.n, dtype=np.int64)
        self.ep_max[:] = self.boards.max(axis=1)

        self.prev_c6 = np.zeros((self.n, self.net.n6), dtype=np.int64)
        self.prev_c4 = np.zeros((self.n, self.net.n4), dtype=np.int64)
        self.prev_v = np.zeros(self.n, dtype=np.float32)
        self.prev_exists = np.zeros(self.n, dtype=bool)
        self._started = False
        self._reset_chunk_stats()

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        self.boards = new_boards(self.n, self.rng)
        self.ep_score[:] = 0.0
        self.ep_len[:] = 0
        self.ep_max[:] = self.boards.max(axis=1)
        self.prev_exists[:] = False
        self._started = False

    def _reset_chunk_stats(self) -> None:
        self._chunk_episodes = 0
        self._chunk_score = 0.0
        self._chunk_length = 0
        self._chunk_best = 0
        self._chunk_tiles: dict[int, int] = {}
        self._chunk_td_error = 0.0
        self._chunk_td_count = 0

    # ------------------------------------------------------------------- step
    def step(self, global_step: int) -> None:
        cfg = self.config
        rng = self.rng
        n = self.n
        rows = np.arange(n)

        boards = self.boards
        after, gain_log, gain_real, valid = batch_move_all(boards)
        alive = valid.any(axis=1)

        codes6, codes4 = self.net.codes(after.reshape(-1, BOARD_CELLS))
        q = self.net.values_from_codes(codes6, codes4).reshape(n, NUM_DIRECTIONS)

        self.policy.on_step(global_step)
        q_masked = q.copy()
        q_masked[~valid] = NEG_INF
        actions = self.policy.select(q_masked, valid, rng)

        next_states = after[rows, actions]
        step_gain = (gain_log if self.use_log else gain_real)[rows, actions]
        step_real = gain_real[rows, actions]
        next_values = np.where(alive, q[rows, actions], 0.0).astype(np.float32)

        # --- TD(0) / SARSA-style update of the *previous* afterstate ---------
        # ``prev`` is the afterstate of the previous step; the board we are
        # acting on right now is exactly ``spawn(prev)``, so the reward of the
        # transition ``prev -> next`` is ``step_gain`` and a terminal board
        # contributes no future reward.
        if self._started and self.prev_exists.any():
            target = np.where(alive, step_gain.astype(np.float32) + np.float32(cfg.gamma) * next_values, np.float32(0.0))
            delta = np.where(self.prev_exists, target - self.prev_v, 0.0).astype(np.float32)
            self.net.update_from_codes(self.prev_c6, self.prev_c4, np.float32(cfg.lr) * delta)
            self._chunk_td_error += float(np.abs(delta[self.prev_exists]).sum())
            self._chunk_td_count += int(self.prev_exists.sum())
        self._started = True

        # --- remember the new afterstate for the next update -----------------
        self.prev_c6 = codes6.reshape(n, NUM_DIRECTIONS, self.net.n6)[rows, actions]
        self.prev_c4 = codes4.reshape(n, NUM_DIRECTIONS, self.net.n4)[rows, actions]
        self.prev_v = next_values
        self.prev_exists = alive.copy()

        # --- bookkeeping -----------------------------------------------------
        self.ep_score += np.where(alive, step_real, 0)
        self.ep_len += alive
        self.ep_max = np.maximum(self.ep_max, boards.max(axis=1))

        # --- spawn a new tile and restart finished games ---------------------
        next_boards = next_states.copy()
        batch_spawn(next_boards, rng, mask=alive)
        finished = ~alive
        if finished.any():
            self._finish(finished)
            next_boards[finished] = new_boards(int(finished.sum()), rng)
            self.ep_max[finished] = next_boards[finished].max(axis=1)
        self.boards = next_boards

    # ------------------------------------------------------------- statistics
    def _finish(self, mask: np.ndarray) -> None:
        count = int(mask.sum())
        self._chunk_episodes += count
        self._chunk_score += float(self.ep_score[mask].sum())
        self._chunk_length += int(self.ep_len[mask].sum())
        self._chunk_best = max(self._chunk_best, int(self.ep_max[mask].max()))
        for exponent, occurrences in zip(*np.unique(self.ep_max[mask], return_counts=True)):
            key = int(exponent)
            self._chunk_tiles[key] = self._chunk_tiles.get(key, 0) + int(occurrences)
        self.ep_score[mask] = 0.0
        self.ep_len[mask] = 0
        self.ep_max[mask] = 0

    # -------------------------------------------------------------------- run
    def run(self, steps: int, start_step: int = 0) -> Stats:
        if self.config.d4_augment and steps > 0:
            # Rotate the whole batch once per chunk.  Keeping one orientation
            # for many steps preserves TD consistency (``prev_v`` and the next
            # state are evaluated under the same transform).
            symmetry = int(self.rng.integers(NUM_SYMMETRIES))
            if symmetry:
                self.boards = transform_boards(self.boards, symmetry)
        for i in range(steps):
            self.step(start_step + i)
        result = Stats(
            steps=steps,
            episodes=self._chunk_episodes,
            score_sum=self._chunk_score,
            length_sum=self._chunk_length,
            best_exponent=self._chunk_best,
            td_error_sum=self._chunk_td_error,
            td_count=self._chunk_td_count,
            tile_counts=dict(self._chunk_tiles),
        )
        self._reset_chunk_stats()
        return result
