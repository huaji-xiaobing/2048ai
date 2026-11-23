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
from .env import NUM_DIRECTIONS, batch_move_all, batch_spawn, new_boards
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


NEG_BIG = np.float32(-1e30)


def _afterstate_values(
    network: NTupleNetwork, boards: np.ndarray, d4_average: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """``(M, 4)`` afterstate values (illegal moves set to ``-1e30``) and validity."""
    after, _, _, valid = batch_move_all(boards)
    values = symmetry_averaged_values(network, after.reshape(-1, 16), d4_average)
    values = values.reshape(boards.shape[0], NUM_DIRECTIONS)
    values = np.where(valid, values, NEG_BIG)
    return values, valid


def best_action_deep(
    network: NTupleNetwork,
    board: Board,
    depth: int = 2,
    gamma: float = 0.999,
    d4_average: bool = False,
) -> tuple[int | None, float]:
    """Expectimax lookahead on top of the value table.

    ``depth=1`` is the plain greedy rule used everywhere else: pick the move
    whose afterstate the network likes best.

    ``depth=2`` expands one spawn layer explicitly::

        Q(s, a) = r_a + gamma * E_spawn[ max_b V(afterstate(spawn(s_a), b)) ]

    The spawn expectation is exact (0.9/0.1 over every empty cell), so this
    trades ~80x compute for replacing the network's guess of ``V(s_a)`` with a
    one-step bootstrap.  Only useful if ``V`` is biased enough that the
    correction pays for the extra variance.
    """
    actions: list[tuple[int, tuple[int, ...], int]] = []
    for direction in DIRECTIONS:
        moved, _, gain = board.move(direction)
        if moved.cells != board.cells:
            actions.append((direction, moved.cells, gain))
    if not actions:
        return None, 0.0

    if depth <= 1:
        values = symmetry_averaged_values(
            network, np.array([a[1] for a in actions], dtype=np.uint8), d4_average
        )
        best = int(np.argmax(values))
        return actions[best][0], float(values[best])

    spawned: list[list[int]] = []
    owner: list[int] = []
    weight: list[float] = []
    for index, (_, cells, _) in enumerate(actions):
        empties = [p for p in range(16) if cells[p] == 0]
        if not empties:
            continue
        scale = 1.0 / len(empties)
        for position in empties:
            for tile, probability in ((1, 0.9), (2, 0.1)):
                row = list(cells)
                row[position] = tile
                spawned.append(row)
                owner.append(index)
                weight.append(probability * scale)

    if not spawned:
        values = symmetry_averaged_values(
            network, np.array([a[1] for a in actions], dtype=np.uint8), d4_average
        )
        best = int(np.argmax(values))
        return actions[best][0], float(values[best])

    values, valid = _afterstate_values(network, np.array(spawned, dtype=np.uint8), d4_average)
    next_best = np.where(valid, values, np.float32(0.0)).max(axis=1)

    continuation = np.zeros(len(actions), dtype=np.float64)
    np.add.at(continuation, np.array(owner, dtype=np.intp), np.array(weight) * next_best)
    q = np.array([a[2] for a in actions], dtype=np.float64) + gamma * continuation

    best = int(np.argmax(q))
    return actions[best][0], float(q[best])


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
    depth: int = 1,
    gamma: float = 0.999,
) -> EpisodeResult:
    """Play one full game with ``depth``-ply expectimax lookahead."""
    rng = np.random.default_rng(seed)
    board = Board.random_start(rng)
    score = 0
    steps = 0
    while steps < max_steps:
        direction, _ = best_action_deep(network, board, depth=depth, gamma=gamma, d4_average=d4_average)
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
