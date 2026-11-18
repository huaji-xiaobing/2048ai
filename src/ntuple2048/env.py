"""Vectorised (many boards at once) 2048 dynamics.

Everything the trainer does per step lives here and works on ``(N, 16)``
``uint8`` arrays of *exponents*.  All motion is expressed through the
pre-computed line tables, so a step costs a few dozen numpy calls regardless of
how many boards are simulated in parallel.
"""

from __future__ import annotations

import numpy as np

from .lookup import BOARD_CELLS, BOARD_SIDE, line_tables

LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
NUM_DIRECTIONS = 4
DIRECTIONS = (LEFT, RIGHT, UP, DOWN)

#: probability of spawning a ``2`` (exponent 1) instead of a ``4`` (exponent 2)
SPAWN_2_PROB = 0.9


def _to_lines(boards: np.ndarray, direction: int) -> np.ndarray:
    """View ``(N, 16)`` boards as ``(N, 4, 4)`` lines ordered along the move."""
    grid = boards.reshape(-1, BOARD_SIDE, BOARD_SIDE)
    if direction == LEFT:
        return grid
    if direction == RIGHT:
        return grid[:, :, ::-1]
    if direction == UP:
        return grid.transpose(0, 2, 1)
    return grid.transpose(0, 2, 1)[:, :, ::-1]


def _from_lines(lines: np.ndarray, direction: int, n: int) -> np.ndarray:
    """Inverse of :func:`_to_lines`; returns a contiguous ``(N, 16)`` board."""
    if direction == LEFT:
        grid = lines
    elif direction == RIGHT:
        grid = lines[:, :, ::-1]
    elif direction == UP:
        grid = lines.transpose(0, 2, 1)
    else:  # DOWN: lines[n, k, r] = grid[n, 3-r, k]
        grid = lines.transpose(0, 2, 1)[:, ::-1, :]
    return np.ascontiguousarray(grid).reshape(n, BOARD_CELLS)


def batch_move(boards: np.ndarray, direction: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide every board in ``direction``.

    Returns ``(new_boards, gain_log, gain_real)`` where the gains are zero for
    boards on which the move was a no-op.
    """
    move, gain_log_table, gain_real_table = line_tables()
    n = boards.shape[0]
    lines = np.ascontiguousarray(_to_lines(boards, direction))

    codes = (
        lines[:, :, 0].astype(np.int32)
        | (lines[:, :, 1].astype(np.int32) << 4)
        | (lines[:, :, 2].astype(np.int32) << 8)
        | (lines[:, :, 3].astype(np.int32) << 12)
    )
    new_codes = move[codes]
    gain_log = gain_log_table[codes].sum(axis=1, dtype=np.int32)
    gain_real = gain_real_table[codes].sum(axis=1, dtype=np.int32)

    new_lines = np.empty_like(lines)
    new_lines[:, :, 0] = new_codes & 0xF
    new_lines[:, :, 1] = (new_codes >> 4) & 0xF
    new_lines[:, :, 2] = (new_codes >> 8) & 0xF
    new_lines[:, :, 3] = (new_codes >> 12) & 0xF

    new_boards = _from_lines(new_lines, direction, n)
    return new_boards, gain_log.astype(np.int32), gain_real.astype(np.int32)


def batch_move_all(
    boards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the four afterstates of every board.

    Returns
    -------
    after : ``(N, 4, 16)`` uint8
        Afterstate for every direction (equal to the input board when the move
        is illegal).
    gain_log, gain_real : ``(N, 4)`` int32
    valid : ``(N, 4)`` bool
        Whether the direction actually changes the board.
    """
    n = boards.shape[0]
    after = np.empty((n, NUM_DIRECTIONS, BOARD_CELLS), dtype=np.uint8)
    gain_log = np.empty((n, NUM_DIRECTIONS), dtype=np.int32)
    gain_real = np.empty((n, NUM_DIRECTIONS), dtype=np.int32)
    for d in DIRECTIONS:
        moved, gl, gr = batch_move(boards, d)
        after[:, d] = moved
        gain_log[:, d] = gl
        gain_real[:, d] = gr
    valid = np.any(after != boards[:, None, :], axis=2)
    return after, gain_log, gain_real, valid


def batch_spawn(boards: np.ndarray, rng: np.random.Generator, mask: np.ndarray | None = None) -> np.ndarray:
    """Add one random tile (2 with p=0.9, 4 with p=0.1) to every masked board.

    Boards without any empty cell are skipped.  ``boards`` is modified in place.
    """
    n = boards.shape[0]
    empty = boards == 0
    counts = empty.sum(axis=1)
    todo = counts > 0
    if mask is not None:
        todo &= mask
    if not todo.any():
        return boards

    idx = np.flatnonzero(todo)
    sub_counts = counts[idx]
    # index of the empty cell to fill: uniform among the empty cells
    pick = (rng.random(idx.size) * sub_counts).astype(np.int64)
    np.minimum(pick, sub_counts - 1, out=pick)
    cumulative = np.cumsum(empty[idx], axis=1)
    positions = np.argmax(cumulative > pick[:, None], axis=1)
    values = np.where(rng.random(idx.size) < SPAWN_2_PROB, 1, 2).astype(np.uint8)
    boards[idx, positions] = values
    return boards


def new_boards(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` fresh boards, each with two random starting tiles."""
    boards = np.zeros((n, BOARD_CELLS), dtype=np.uint8)
    batch_spawn(boards, rng)
    batch_spawn(boards, rng)
    return boards


def any_valid(valid: np.ndarray) -> np.ndarray:
    """Reduce the ``(N, 4)`` validity matrix to ``(N,)`` "is the board alive"."""
    return valid.any(axis=1)


def max_exponent(boards: np.ndarray) -> np.ndarray:
    return boards.max(axis=1)
