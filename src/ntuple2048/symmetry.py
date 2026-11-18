"""D4 symmetry of the 4x4 board.

The eight elements of the dihedral group act on cells through the permutations
in :data:`PERMUTATIONS`.  A board transform is defined as

    ``(T_g b)[p] = b[PERMUTATIONS[g][p]]``

and the *ordered image* of a pattern ``P = (p_0, ..., p_{k-1})`` is

    ``Pi_g(P) = (g(p_0), ..., g(p_{k-1}))``.

The default n-tuple set (12 rectangles + 8 lines) is closed under D4 as a *set
of positions*, but **not** as a set of *ordered* tuples: transposing a 2x3
window that is read row-major yields a 3x2 window read column-major, whose cell
order differs from the row-major order used by the 3x2 patterns.  A strict
weight projection ``W_P <- W_{Pi_g(P)}`` would therefore need every window in
all four traversal orders, i.e. four times the tables, for no accuracy gain.

The package exploits D4 through the two mechanisms that are exact and cheap:

* ``TrainConfig.d4_augment`` -- every rollout step applies a random symmetry to
  the board.  Because the pattern set is closed as a *set*, this makes all
  pattern tables see statistically identical data (the "shared statistics"
  effect a weight projection would give), at the cost of one gather.
* ``d4_average`` at inference -- the afterstate value is averaged over the eight
  symmetries (see :func:`ntuple2048.inference.symmetry_averaged_values`).

:meth:`ntuple2048.network.NTupleNetwork.d4_consistency_error` measures how far a
trained table is from exact D4 invariance; a well trained network gets close to
zero, which is a direct check that the symmetry was learned.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

BOARD_SIDE = 4
NUM_CELLS = 16
NUM_SYMMETRIES = 8


def _transform_cell(g: int, r: int, c: int) -> tuple[int, int]:
    last = BOARD_SIDE - 1
    if g == 0:
        return r, c
    if g == 1:  # rotate 90 deg clockwise
        return c, last - r
    if g == 2:  # rotate 180 deg
        return last - r, last - c
    if g == 3:  # rotate 270 deg clockwise
        return last - c, r
    if g == 4:  # mirror left/right
        return r, last - c
    if g == 5:  # mirror up/down
        return last - r, c
    if g == 6:  # transpose (main diagonal)
        return c, r
    if g == 7:  # transpose (anti diagonal)
        return last - c, last - r
    raise ValueError(f"symmetry index out of range: {g}")


PERMUTATIONS: tuple[np.ndarray, ...] = tuple(
    np.array(
        [
            (lambda rc: rc[0] * BOARD_SIDE + rc[1])(_transform_cell(g, *divmod(p, BOARD_SIDE)))
            for p in range(NUM_CELLS)
        ],
        dtype=np.intp,
    )
    for g in range(NUM_SYMMETRIES)
)

_PERM_STACK = np.stack(PERMUTATIONS)
INVERSE: tuple[int, ...] = tuple(
    int(np.flatnonzero((_PERM_STACK == np.argsort(PERMUTATIONS[g])).all(axis=1))[0])
    for g in range(NUM_SYMMETRIES)
)


def transform_boards(boards: np.ndarray, g: int) -> np.ndarray:
    """Apply symmetry ``g`` to a ``(N, 16)`` batch of boards."""
    return boards[:, PERMUTATIONS[g]]


def transform_board_cells(cells: Sequence[int], g: int) -> tuple[int, ...]:
    """Apply symmetry ``g`` to a single 16-cell board tuple."""
    perm = PERMUTATIONS[g]
    return tuple(int(cells[int(i)]) for i in perm)


def transform_pattern(pattern: Sequence[int], g: int) -> tuple[int, ...]:
    """Ordered image of ``pattern`` under symmetry ``g``."""
    perm = PERMUTATIONS[g]
    return tuple(int(perm[int(p)]) for p in pattern)


def is_closed_under_d4(patterns: Iterable[Sequence[int]]) -> bool:
    """True when the *ordered* pattern set is mapped onto itself by D4."""
    pattern_set = {tuple(int(x) for x in p) for p in patterns}
    return all(
        transform_pattern(p, g) in pattern_set for p in pattern_set for g in range(NUM_SYMMETRIES)
    )


def is_closed_as_sets(patterns: Iterable[Sequence[int]]) -> bool:
    """True when the *position sets* of the patterns are mapped onto each other."""
    pattern_set = {frozenset(int(x) for x in p) for p in patterns}
    return all(
        frozenset(transform_pattern(p, g)) in pattern_set
        for p in pattern_set
        for g in range(NUM_SYMMETRIES)
    )


def pattern_orbits(patterns: Sequence[Sequence[int]]) -> list[list[int]]:
    """Group pattern indices by their D4 orbit (as position sets).

    The default 12 rectangles form two orbits (8 + 4) and the 8 lines form a
    single orbit of 8 -- a compact description of how symmetric the tiling is.
    """
    index = {frozenset(int(x) for x in p): i for i, p in enumerate(patterns)}
    seen = [False] * len(patterns)
    orbits: list[list[int]] = []
    for i, pattern in enumerate(patterns):
        if seen[i]:
            continue
        members: list[int] = []
        for g in range(NUM_SYMMETRIES):
            image = frozenset(transform_pattern(pattern, g))
            if image not in index:
                raise ValueError(f"pattern position sets are not closed under D4: {sorted(image)}")
            j = index[image]
            if not seen[j]:
                seen[j] = True
                members.append(j)
        orbits.append(members)
    return orbits


def orbit_report(patterns: Sequence[Sequence[int]]) -> list[list[int]]:
    """Human-readable orbit members (handy for ``python -m ntuple2048 patterns``)."""
    return [sorted(o) for o in pattern_orbits(patterns)]
