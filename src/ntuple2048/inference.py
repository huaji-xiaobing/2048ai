"""Inference / evaluation with a trained n-tuple value table.

The agent is purely greedy on the afterstate value: for a board it evaluates
the four afterstates and plays the direction with the highest value.  Because
the network is trained on afterstates, no search is needed.

``d4_average=True`` evaluates each afterstate under all eight symmetries and
averages the results -- this helps early / partially trained tables and is a
direct sanity check that the learned values respect the board symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .board import DIRECTIONS, DIRECTION_NAMES, Board
from .env import batch_move_all, batch_spawn, new_boards
from .network import NTupleNetwork
from .symmetry import NUM_SYMMETRIES, transform_boards

NEG_INF = np.float32(-1e30)


def symmetry_averaged_values(
    network: NTupleNetwork, boards: np.ndarray, d4_average: bool = False
) -> np.ndarray:
    """``V`` of every board, optionally averaged over the eight D4 symmetries."""
    if not d4_average:
        return network.values(boards)
    total = np.zeros(boards.shape[0], dtype=np.float32)
    for g in range(NUM_SYMMETRIES):
        total += network.values(transform_boards(boards, g))
    total /= NUM_SYMMETRIES
    return total


def best_action(
    network: NTupleNetwork, board: Board, d4_average: bool = False
) -> tuple[int | None, float]:
    """Return ``(direction, value)`` for the best legal move, or ``(None, 0)``."""
    directions: list[int] = []
    afterstates: list[tuple[int, ...]] = []
    for d in DIRECTIONS:
        moved, _, _ = board.move(d)
        if moved.cells != board.cells:
            directions.append(d)
            afterstates.append(moved.cells)
    if not directions:
        return None, 0.0
    values = symmetry_averaged_values(network, np.array(afterstates, dtype=np.uint8), d4_average)
    best = int(np.argmax(values))
    return directions[best], float(values[best])


def action_values(
    network: NTupleNetwork, board: Board, d4_average: bool = False
) -> dict[str, float]:
    """Afterstate value for every legal direction (debugging aid)."""
    result: dict[str, float] = {}
    cells = []
    names = []
    for d in DIRECTIONS:
        moved, _, _ = board.move(d)
        if moved.cells != board.cells:
            cells.append(moved.cells)
            names.append(DIRECTION_NAMES[d])
    if not cells:
        return result
    values = symmetry_averaged_values(network, np.array(cells, dtype=np.uint8), d4_average)
    for name, value in zip(names, values):
        result[name] = float(value)
    return result


@dataclass
class EpisodeResult:
    score: int
    length: int
    max_tile: int
    board: Board


def play_game(
    network: NTupleNetwork,
    seed: int | None = None,
    d4_average: bool = False,
    verbose: bool = False,
    max_steps: int = 1_000_000,
) -> EpisodeResult:
    """Play one full game greedily."""
    rng = np.random.default_rng(seed)
    board = Board.random_start(rng)
    score = 0
    steps = 0
    while steps < max_steps:
        direction, _ = best_action(network, board, d4_average)
        if direction is None:
            break
        board, _, gain_real = board.move(direction)
        score += gain_real
        board = board.with_spawn(rng)
        steps += 1
        if verbose:
            print(f"step {steps:5d}  move={DIRECTION_NAMES[direction]:5s}  score={score}")
            print(board)
            print()
    return EpisodeResult(score=score, length=steps, max_tile=board.max_tile, board=board)


def evaluate(
    network: NTupleNetwork,
    episodes: int = 100,
    seed: int = 0,
    d4_average: bool = False,
    max_steps: int = 100_000,
) -> dict[str, float | int | dict[int, int]]:
    """Play ``episodes`` games in parallel batches and summarise the results."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(seed)
    n = episodes
    boards = new_boards(n, rng)
    rows = np.arange(n)

    scores = np.zeros(n, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    best = boards.max(axis=1).astype(np.int64)
    active = np.ones(n, dtype=bool)

    for _ in range(max_steps):
        if not active.any():
            break
        after, _, gain_real, valid = batch_move_all(boards)
        alive = valid.any(axis=1)

        values = symmetry_averaged_values(network, after.reshape(-1, 16), d4_average)
        values = values.reshape(n, -1).copy()
        values[~valid] = NEG_INF
        actions = np.argmax(values, axis=1)

        scores += np.where(alive, gain_real[rows, actions], 0)
        lengths += alive
        best = np.maximum(best, boards.max(axis=1))

        next_boards = np.where(alive[:, None], after[rows, actions], boards)
        batch_spawn(next_boards, rng, mask=alive)
        boards = next_boards
        active = alive

    best_tiles = np.where(best > 0, 1 << best, 0)
    tile_hist: dict[int, int] = {}
    for exponent, occurrences in zip(*np.unique(best, return_counts=True)):
        tile_hist[int(exponent)] = int(occurrences)

    percentiles = np.percentile(scores, [10, 50, 90, 99]) if n else np.zeros(4)
    return {
        "episodes": n,
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "min_score": int(scores.min()),
        "max_score": int(scores.max()),
        "p10": float(percentiles[0]),
        "p90": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "mean_length": float(lengths.mean()),
        "best_tile": int(best_tiles.max()),
        "tile_hist": tile_hist,
    }
