"""N-tuple value network.

The network approximates the value of an **afterstate** (the board right after a
slide/merge, before a new tile is spawned) as

    ``V(s) = sum_{P in patterns} W_P[ code(P, s) ]``

where ``code(P, s) = sum_t base ** t * s[P[t]]`` and ``s`` stores tile exponents.

Two table groups are kept because their code spaces differ:

* ``W6`` -- ``(12, base**6)`` for the rectangular 6-tuples
* ``W4`` -- ``(8,  base**4)`` for the straight 4-tuples

Both are flat lookup tables, which makes the forward pass two fancy-index
gathers and the TD update one scattered add.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .lookup import BOARD_CELLS
from .symmetry import NUM_SYMMETRIES, is_closed_under_d4, pattern_orbits, transform_boards
from .tuples import default_patterns_4, default_patterns_6

DEFAULT_BASE = 15  # exponents 0..14 -> tiles up to 16384


class NTupleNetwork:
    """Weight tables for a fixed n-tuple pattern set."""

    def __init__(
        self,
        base: int = DEFAULT_BASE,
        patterns6: Sequence[Sequence[int]] | None = None,
        patterns4: Sequence[Sequence[int]] | None = None,
        dtype: str | np.dtype = np.float32,
    ) -> None:
        if base < 3:
            raise ValueError("base must be >= 3")
        self.base = int(base)
        self.dtype = np.dtype(dtype)
        self.patterns6 = tuple(tuple(int(x) for x in p) for p in (patterns6 or default_patterns_6()))
        self.patterns4 = tuple(tuple(int(x) for x in p) for p in (patterns4 or default_patterns_4()))
        self.n6 = len(self.patterns6)
        self.n4 = len(self.patterns4)
        self.l6 = self.base**6
        self.l4 = self.base**4

        self.W6 = np.zeros((self.n6, self.l6), dtype=self.dtype)
        self.W4 = np.zeros((self.n4, self.l4), dtype=self.dtype)
        self._after_load()

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_arrays(
        cls,
        W6: np.ndarray,
        W4: np.ndarray,
        base: int = DEFAULT_BASE,
        patterns6: Sequence[Sequence[int]] | None = None,
        patterns4: Sequence[Sequence[int]] | None = None,
    ) -> "NTupleNetwork":
        """Wrap existing (possibly shared-memory) weight arrays."""
        net = cls.__new__(cls)
        net.base = int(base)
        net.dtype = np.dtype(W6.dtype)
        net.patterns6 = tuple(tuple(int(x) for x in p) for p in (patterns6 or default_patterns_6()))
        net.patterns4 = tuple(tuple(int(x) for x in p) for p in (patterns4 or default_patterns_4()))
        net.n6 = len(net.patterns6)
        net.n4 = len(net.patterns4)
        net.l6 = int(W6.shape[1])
        net.l4 = int(W4.shape[1])
        net.W6 = W6
        net.W4 = W4
        net._after_load()
        return net

    def _after_load(self) -> None:
        expected6 = self.base**6
        expected4 = self.base**4
        if self.l6 != expected6 or self.l4 != expected4:
            raise ValueError(
                f"weight shapes {self.W6.shape}/{self.W4.shape} do not match base {self.base}"
            )
        if self.W6.dtype != np.float32 or self.W4.dtype != np.float32:
            self.W6 = self.W6.astype(np.float32)
            self.W4 = self.W4.astype(np.float32)
            self.dtype = np.dtype(np.float32)
        self.W6_flat = self.W6.reshape(-1)
        self.W4_flat = self.W4.reshape(-1)
        self._row6 = np.arange(self.n6, dtype=np.intp)
        self._row4 = np.arange(self.n4, dtype=np.intp)
        self._off6 = (np.arange(self.n6, dtype=np.int64) * self.l6)
        self._off4 = (np.arange(self.n4, dtype=np.int64) * self.l4)
        self._code_matrix = self._build_code_matrix()

    def _build_code_matrix(self) -> np.ndarray:
        """``(16, n6+n4)`` matrix such that ``boards @ M`` yields all codes.

        ``M[P[t], i] = base**t``; because a cell appears at most once inside a
        pattern this reproduces ``code(P, board)`` exactly.  Everything stays
        below ``2**24`` so the float32 matmul is exact for integer inputs.
        """
        matrix = np.zeros((BOARD_CELLS, self.n6 + self.n4), dtype=np.float32)
        for i, pattern in enumerate(self.patterns6):
            for t, pos in enumerate(pattern):
                matrix[pos, i] += float(self.base) ** t
        offset = self.n6
        for i, pattern in enumerate(self.patterns4):
            for t, pos in enumerate(pattern):
                matrix[pos, offset + i] += float(self.base) ** t
        max_code = float(self.base**6 - 1)
        if max_code > 2.0**24:
            raise ValueError("base**6 exceeds the exact float32 integer range; use a smaller base")
        return matrix

    # ------------------------------------------------------------------ codes
    def codes(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map ``(N, 16)`` exponent boards to ``(c6, c4)`` integer index arrays."""
        clipped = np.minimum(boards, self.base - 1).astype(np.float32, copy=False)
        all_codes = clipped @ self._code_matrix
        c6 = all_codes[:, : self.n6].astype(np.int64)
        c4 = all_codes[:, self.n6 :].astype(np.int64)
        return c6, c4

    # --------------------------------------------------------------- forward
    def values_from_codes(self, c6: np.ndarray, c4: np.ndarray) -> np.ndarray:
        """Vectorised ``V`` from pre-computed codes."""
        total = self.W6[self._row6, c6].sum(axis=1, dtype=np.float32)
        if self.n4:
            total = total + self.W4[self._row4, c4].sum(axis=1, dtype=np.float32)
        return total

    def values(self, boards: np.ndarray) -> np.ndarray:
        """Value of every board in an ``(N, 16)`` batch."""
        c6, c4 = self.codes(boards)
        return self.values_from_codes(c6, c4)

    # ---------------------------------------------------------------- update
    def update_from_codes(self, c6: np.ndarray, c4: np.ndarray, deltas: np.ndarray) -> None:
        """Sparse TD update: ``W_P[code_P] += delta`` for every sample.

        ``deltas`` already contains the learning rate.  Scattered adds are not
        atomic; concurrent workers may drop an update on a rare index collision,
        which is the usual Hogwild! trade-off and is harmless for TD learning.
        """
        deltas = np.ascontiguousarray(deltas, dtype=np.float32)
        if not deltas.any():
            return
        if self.n6:
            self.W6_flat[c6 + self._off6] += deltas[:, None]
        if self.n4:
            self.W4_flat[c4 + self._off4] += deltas[:, None]

    # -------------------------------------------------------------- symmetry
    def orbit_sizes(self) -> dict[str, list[int]]:
        """Size of every D4 orbit of the pattern *position sets*."""
        return {
            "patterns6": [len(o) for o in pattern_orbits(self.patterns6)],
            "patterns4": [len(o) for o in pattern_orbits(self.patterns4)],
        }

    def check_d4_closed(self) -> bool:
        """Whether the ordered pattern set is invariant under D4 (False by default)."""
        return is_closed_under_d4(self.patterns6) and is_closed_under_d4(self.patterns4)

    def d4_consistency_error(self, boards: np.ndarray) -> float:
        """``max |V(b) - V(g b)|`` over the batch and the eight symmetries.

        A converged, D4-augmented model approaches zero here.
        """
        reference = self.values(boards)
        worst = 0.0
        for g in range(1, NUM_SYMMETRIES):
            deviation = np.abs(self.values(transform_boards(boards, g)) - reference)
            if deviation.size:
                worst = max(worst, float(deviation.max()))
        return worst

    # -------------------------------------------------------------- lifecycle
    def copy_weights_to(self, W6: np.ndarray, W4: np.ndarray) -> None:
        W6[...] = self.W6
        W4[...] = self.W4

    def mean_abs(self) -> float:
        total = float(np.abs(self.W6).mean()) if self.n6 else 0.0
        if self.n4:
            total += float(np.abs(self.W4).mean())
        return total

    def meta(self) -> dict:
        return {
            "format": "ntuple2048-v1",
            "base": self.base,
            "patterns6": [list(p) for p in self.patterns6],
            "patterns4": [list(p) for p in self.patterns4],
            "dtype": "float32",
        }

    def save(self, path: str | Path, compress: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = np.frombuffer(json.dumps(self.meta()).encode("utf-8"), dtype=np.uint8)
        saver = np.savez_compressed if compress else np.savez
        saver(path, W6=self.W6, W4=self.W4, meta=meta)

    @classmethod
    def load(cls, path: str | Path) -> "NTupleNetwork":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["meta"].tobytes().decode("utf-8"))
            if meta.get("format") != "ntuple2048-v1":
                raise ValueError(f"unsupported model format: {meta.get('format')!r}")
            return cls.from_arrays(
                np.array(data["W6"], dtype=np.float32),
                np.array(data["W4"], dtype=np.float32),
                base=int(meta["base"]),
                patterns6=[tuple(p) for p in meta["patterns6"]],
                patterns4=[tuple(p) for p in meta["patterns4"]],
            )
