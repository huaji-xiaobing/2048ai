"""N-tuple feature patterns.

Default configuration (the one used by every paper on 2048 n-tuple learning):

* **12 rectangular 6-tuples** -- every 2x3 and 3x2 window of the 4x4 grid
  (6 positions x 2 orientations = 12), read row-major.
* **8 straight 4-tuples** -- the 4 rows and the 4 columns.

The *order* of the cells inside a pattern matters: it defines the base-N digit
position of the pattern code (``code = sum_t cell[pattern[t]] * base**t``).
The ordering used here is always "row-major inside the window", which makes the
pattern set closed under the 8 elements of the D4 symmetry group.
"""

from __future__ import annotations

from .lookup import BOARD_SIDE

SIDE = BOARD_SIDE


def default_patterns_6() -> tuple[tuple[int, ...], ...]:
    """12 sliding 2x3 / 3x2 rectangles (6 cells each)."""
    patterns: list[tuple[int, ...]] = []
    for r in range(SIDE - 1):
        for c in range(SIDE - 2):
            patterns.append(
                tuple((r + dr) * SIDE + (c + dc) for dr in range(2) for dc in range(3))
            )
    for r in range(SIDE - 2):
        for c in range(SIDE - 1):
            patterns.append(
                tuple((r + dr) * SIDE + (c + dc) for dr in range(3) for dc in range(2))
            )
    return tuple(patterns)


def default_patterns_4() -> tuple[tuple[int, ...], ...]:
    """8 straight lines: 4 rows then 4 columns."""
    patterns: list[tuple[int, ...]] = []
    for r in range(SIDE):
        patterns.append(tuple(r * SIDE + c for c in range(SIDE)))
    for c in range(SIDE):
        patterns.append(tuple(r * SIDE + c for r in range(SIDE)))
    return tuple(patterns)


def pattern_code(flat_board, pattern, base: int = 16) -> int:
    """Python-int code of ``pattern`` for a flat board (convenience helper)."""
    code = 0
    for value in pattern:
        code = code * base + int(flat_board[value])
    return code
