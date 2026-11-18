"""Behaviour distillation of an expert policy into the n-tuple value table.

The expert returns an action for a board.  Our network scores *afterstates*, so
the natural objective is a classification over the four legal afterstates:

    ``p_i = softmax_i( V(afterstate_i) / T )``   target = one-hot(expert action)

Minimising ``-log p_a`` gives the sparse gradient

    ``dL/dV_i = p_i - [i == a]``

which plugs straight into :meth:`NTupleNetwork.update_from_codes` (n-tuple
networks learn by additive table updates, so "gradient" means "value delta").

Because the value scale of a trained table can be huge, ``V`` is normalised per
sample (z-score over the legal afterstates) before the softmax; that keeps the
effective learning rate independent of how big the table has grown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import NUM_DIRECTIONS, batch_move_all
from .lookup import BOARD_CELLS
from .network import NTupleNetwork

NEG_INF = np.float32(-1e30)


@dataclass
class DistillStats:
    samples: int
    epochs: int
    loss: float
    accuracy: float
    illegal: int

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "epochs": self.epochs,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "illegal": self.illegal,
        }


def _softmax(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    z = np.where(valid, z, NEG_INF)
    z = z - z.max(axis=1, keepdims=True)
    weights = np.exp(z) * valid
    total = weights.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return weights / total


def distill_batch(
    network: NTupleNetwork,
    boards: np.ndarray,
    actions: np.ndarray,
    lr: float,
) -> tuple[float, int, int]:
    """One gradient step. Returns ``(loss_sum, correct, illegal)``."""
    n = boards.shape[0]
    rows = np.arange(n)
    after, _, _, valid = batch_move_all(boards)

    legal = valid[rows, actions]
    if not legal.any():
        return 0.0, 0, n

    codes6, codes4 = network.codes(after.reshape(-1, BOARD_CELLS))
    values = network.values_from_codes(codes6, codes4).reshape(n, NUM_DIRECTIONS)

    masked = np.where(valid, values, np.nan)
    mean = np.nanmean(masked, axis=1, keepdims=True)
    std = np.nanstd(masked, axis=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)

    probabilities = _softmax((values - mean) / std, valid)
    target = np.zeros_like(probabilities)
    target[rows, actions] = 1.0

    gradient = (probabilities - target) * legal[:, None]
    gradient[~legal] = 0.0

    loss = float(-np.log(np.clip(probabilities[rows, actions][legal], 1e-12, 1.0)).sum())
    correct = int((np.argmax(probabilities, axis=1) == actions)[legal].sum())

    if gradient.any():
        network.update_from_codes(
            codes6,
            codes4,
            (np.float32(-lr) * gradient).reshape(-1),
        )
    return loss, correct, int((~legal).sum())


def distill(
    network: NTupleNetwork,
    boards: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 2048,
    seed: int = 0,
    progress=None,
) -> DistillStats:
    """Run ``epochs`` passes of behaviour cloning over ``(boards, actions)``."""
    if boards.shape[0] != actions.shape[0]:
        raise ValueError("boards and actions must have the same length")
    rng = np.random.default_rng(seed)
    total = boards.shape[0]
    loss_sum = 0.0
    correct = 0
    illegal = 0

    for epoch in range(epochs):
        order = rng.permutation(total)
        for start in range(0, total, batch_size):
            index = order[start : start + batch_size]
            loss, good, bad = distill_batch(
                network, boards[index], actions[index].astype(np.int64), lr
            )
            loss_sum += loss
            correct += good
            illegal += bad
        if progress is not None:
            progress(epoch + 1, epochs, loss_sum, correct, total)

    seen = max(total * epochs, 1)
    return DistillStats(
        samples=total,
        epochs=epochs,
        loss=loss_sum / seen,
        accuracy=correct / max(seen - illegal, 1),
        illegal=illegal,
    )
