"""Pre-computed tables for a single 4-cell line.

A line is four cells packed into 16 bits (4 bits per cell).  A cell stores
``log2(tile)`` so ``0`` means *empty*, ``1`` means tile ``2``, ``2`` means
tile ``4`` and so on.

Every one of the ``16**4 == 65536`` possible lines is tabulated once:

``MOVE[code]``
    Code of the line after sliding it towards index 0 (and merging).
``GAIN_LOG[code]``
    Sum of ``log2`` of the tiles produced by merges.
``GAIN_REAL[code]``
    Sum of the numeric values of the tiles produced by merges (game score).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

LINE_LEN = 4
BOARD_SIDE = 4
BOARD_CELLS = BOARD_SIDE * BOARD_SIDE
MAX_EXPONENT = 15
NUM_LINE_CODES = 1 << (4 * LINE_LEN)


def _move_line_left(line: list[int]) -> tuple[list[int], int, int]:
    """Slide ``line`` towards index 0, merging each pair of equal cells once."""
    values = [v for v in line if v]
    out: list[int] = []
    gain_log = 0
    gain_real = 0
    i = 0
    n = len(values)
    while i < n:
        if i + 1 < n and values[i] == values[i + 1]:
            merged = values[i] + 1
            if merged > MAX_EXPONENT:
                merged = MAX_EXPONENT
            out.append(merged)
            gain_log += merged
            gain_real += 1 << merged
            i += 2
        else:
            out.append(values[i])
            i += 1
    out.extend([0] * (LINE_LEN - len(out)))
    return out, gain_log, gain_real


@lru_cache(maxsize=1)
def line_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(MOVE, GAIN_LOG, GAIN_REAL)`` numpy arrays keyed by line code."""
    move = np.zeros(NUM_LINE_CODES, dtype=np.uint16)
    gain_log = np.zeros(NUM_LINE_CODES, dtype=np.int32)
    gain_real = np.zeros(NUM_LINE_CODES, dtype=np.int32)

    for code in range(NUM_LINE_CODES):
        line = [(code >> (4 * i)) & 0xF for i in range(LINE_LEN)]
        out, gl, gr = _move_line_left(line)
        new_code = 0
        for i, v in enumerate(out):
            new_code |= v << (4 * i)
        move[code] = new_code
        gain_log[code] = gl
        gain_real[code] = gr

    move.setflags(write=False)
    gain_log.setflags(write=False)
    gain_real.setflags(write=False)
    return move, gain_log, gain_real


@lru_cache(maxsize=1)
def scalar_line_tables() -> tuple[list[int], list[int], list[int]]:
    """Same tables as plain Python lists (faster for per-cell scalar access)."""
    move, gain_log, gain_real = line_tables()
    return move.tolist(), gain_log.tolist(), gain_real.tolist()


def line_from_cells(cells, indices) -> int:
    """Pack four cell values into a line code."""
    return (
        int(cells[indices[0]])
        | (int(cells[indices[1]]) << 4)
        | (int(cells[indices[2]]) << 8)
        | (int(cells[indices[3]]) << 12)
    )
