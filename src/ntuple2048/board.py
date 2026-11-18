"""Single-board 2048 engine (no GUI).

The board is an immutable 16-tuple of *exponents*: a cell holds ``log2(tile)``
so ``0`` is empty, ``1`` is tile 2, ``2`` is tile 4, ...

Moves are implemented with the pre-computed line tables from
:mod:`ntuple2048.lookup`, which makes a move four table lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from .lookup import BOARD_CELLS, BOARD_SIDE, MAX_EXPONENT, line_from_cells, scalar_line_tables

LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
DIRECTIONS = (LEFT, RIGHT, UP, DOWN)
DIRECTION_NAMES = {LEFT: "left", RIGHT: "right", UP: "up", DOWN: "down"}
DIRECTION_FROM_NAME = {name: d for d, name in DIRECTION_NAMES.items()}


@lru_cache(maxsize=1)
def direction_lines() -> dict[int, tuple[tuple[int, int, int, int], ...]]:
    """For each direction, the four cell-index lines ordered along the move.

    The first element of every line is where tiles pile up, i.e. the resulting
    line is always produced by :func:`ntuple2048.lookup._move_line_left`.
    """
    lines: dict[int, tuple[tuple[int, int, int, int], ...]] = {}
    for d in DIRECTIONS:
        rows = []
        for k in range(BOARD_SIDE):
            if d == LEFT:
                idx = (k * 4 + 0, k * 4 + 1, k * 4 + 2, k * 4 + 3)
            elif d == RIGHT:
                idx = (k * 4 + 3, k * 4 + 2, k * 4 + 1, k * 4 + 0)
            elif d == UP:
                idx = (0 * 4 + k, 1 * 4 + k, 2 * 4 + k, 3 * 4 + k)
            else:  # DOWN
                idx = (3 * 4 + k, 2 * 4 + k, 1 * 4 + k, 0 * 4 + k)
            rows.append(idx)
        lines[d] = tuple(rows)
    return lines


def move_cells(cells: Sequence[int], direction: int) -> tuple[tuple[int, ...], int, int]:
    """Apply ``direction`` to a raw 16-cell tuple.

    Returns ``(new_cells, gain_log, gain_real)``.
    """
    move, gain_log_t, gain_real_t = scalar_line_tables()
    out = [0] * BOARD_CELLS
    gain_log = 0
    gain_real = 0
    for idx in direction_lines()[direction]:
        code = line_from_cells(cells, idx)
        gain_log += gain_log_t[code]
        gain_real += gain_real_t[code]
        new_code = move[code]
        out[idx[0]] = new_code & 0xF
        out[idx[1]] = (new_code >> 4) & 0xF
        out[idx[2]] = (new_code >> 8) & 0xF
        out[idx[3]] = (new_code >> 12) & 0xF
    return tuple(out), gain_log, gain_real


@dataclass(frozen=True, slots=True)
class Board:
    """Immutable 2048 board."""

    cells: tuple[int, ...]

    # ------------------------------------------------------------------ build
    @classmethod
    def empty(cls) -> "Board":
        return cls((0,) * BOARD_CELLS)

    @classmethod
    def from_cells(cls, cells: Iterable[int]) -> "Board":
        values = tuple(int(v) for v in cells)
        if len(values) != BOARD_CELLS:
            raise ValueError(f"expected {BOARD_CELLS} cells, got {len(values)}")
        if any(v < 0 or v > MAX_EXPONENT for v in values):
            raise ValueError("cell values must be in [0, 15]")
        return cls(values)

    @classmethod
    def from_tiles(cls, rows: Sequence[Sequence[int]]) -> "Board":
        """Build from a 4x4 grid of real tile values (0 = empty, 2, 4, ...)."""
        cells = []
        for row in rows:
            for tile in row:
                cells.append(0 if tile == 0 else int(round(np.log2(tile))))
        return cls.from_cells(cells)

    @classmethod
    def random_start(cls, rng: np.random.Generator) -> "Board":
        cells = [0] * BOARD_CELLS
        for _ in range(2):
            _spawn_into(cells, rng)
        return cls(tuple(cells))

    # ------------------------------------------------------------------- view
    def tiles(self) -> list[list[int]]:
        """4x4 grid of real tile values."""
        return [
            [0 if v == 0 else 1 << v for v in self.cells[r * 4:(r + 1) * 4]]
            for r in range(BOARD_SIDE)
        ]

    @property
    def max_exponent(self) -> int:
        return max(self.cells)

    @property
    def max_tile(self) -> int:
        return 1 << self.max_exponent if self.max_exponent else 0

    @property
    def empty_count(self) -> int:
        return self.cells.count(0)

    def __str__(self) -> str:
        width = max(5, len(str(self.max_tile)))
        lines = []
        for r in range(BOARD_SIDE):
            row = self.tiles()[r]
            lines.append(" ".join(f"{v:>{width}d}" if v else "." * width for v in row))
        return "\n".join(lines)

    # ------------------------------------------------------------------ moves
    def move(self, direction: int) -> tuple["Board", int, int]:
        """Return ``(new_board, gain_log, gain_real)``.

        The board is returned unchanged when the move does nothing.
        """
        cells, gain_log, gain_real = move_cells(self.cells, direction)
        if cells == self.cells:
            return self, 0, 0
        return Board(cells), gain_log, gain_real

    def valid_directions(self) -> list[int]:
        cells = self.cells
        return [d for d in DIRECTIONS if move_cells(cells, d)[0] != cells]

    def is_terminal(self) -> bool:
        return not self.valid_directions()

    # ------------------------------------------------------------------ spawn
    def with_spawn(self, rng: np.random.Generator) -> "Board":
        cells = list(self.cells)
        _spawn_into(cells, rng)
        return Board(tuple(cells))

    # ------------------------------------------------------------------- misc
    def afterstates(self, directions: Sequence[int] = DIRECTIONS):
        """Yield ``(direction, board, gain_real)`` for every *valid* move."""
        for d in directions:
            new_board, _, gain_real = self.move(d)
            if new_board.cells != self.cells:
                yield d, new_board, gain_real


def _spawn_into(cells: list[int], rng: np.random.Generator) -> None:
    empty = [i for i, v in enumerate(cells) if v == 0]
    if not empty:
        return
    pos = empty[int(rng.integers(len(empty)))]
    cells[pos] = 1 if rng.random() < 0.9 else 2
