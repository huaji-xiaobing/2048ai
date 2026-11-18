"""Pluggable action-selection policies.

The trainer only relies on the small :class:`Policy` interface, so a new
strategy (softmax, UCB, a learned network, ...) can be dropped in without
touching the training loop::

    @register("my-policy")
    class MyPolicy(Policy):
        def select(self, q, valid, rng):
            ...  # return an (N,) array of action indices

Policies operate on *batches*: ``q`` and ``valid`` are ``(N, 4)`` arrays and the
result is ``(N,)``.  ``q`` has already been masked with ``-inf`` on illegal
actions, but ``valid`` is passed as well so that exploration can be restricted
to legal moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

NEG_INF = -1e30


@dataclass(frozen=True)
class LinearSchedule:
    """Linear interpolation from ``start`` to ``end`` over ``decay_steps``."""

    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 2_000_000

    def value(self, step: int) -> float:
        if self.decay_steps <= 0 or step >= self.decay_steps:
            return float(self.end)
        frac = step / self.decay_steps
        return float(self.start + (self.end - self.start) * frac)


class Policy:
    """Base class. Subclasses implement :meth:`select`."""

    name = "base"

    def select(self, q: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def on_step(self, step: int) -> None:
        """Optional hook called once per training step (schedules, annealing...)."""


class GreedyPolicy(Policy):
    """Pure exploitation (used for inference)."""

    name = "greedy"

    def select(self, q: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.argmax(q, axis=1)


class EpsilonGreedyPolicy(Policy):
    """epsilon-greedy with a linear annealing schedule."""

    name = "epsilon_greedy"

    def __init__(self, epsilon: float = 0.1, schedule: LinearSchedule | None = None) -> None:
        self.epsilon = float(epsilon)
        self.schedule = schedule

    def epsilon_at(self, step: int) -> float:
        return self.schedule.value(step) if self.schedule is not None else self.epsilon

    def on_step(self, step: int) -> None:
        if self.schedule is not None:
            self.epsilon = self.schedule.value(step)

    def select(self, q: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        greedy = np.argmax(q, axis=1)
        if self.epsilon <= 0.0:
            return greedy
        n = q.shape[0]
        noise = rng.random(q.shape)
        noise[~valid] = -1.0
        random_action = np.argmax(noise, axis=1)
        explore = rng.random(n) < self.epsilon
        return np.where(explore, random_action, greedy)


class SoftmaxPolicy(Policy):
    """Boltzmann exploration over the afterstate values."""

    name = "softmax"

    def __init__(self, temperature: float = 1.0, schedule: LinearSchedule | None = None) -> None:
        self.temperature = float(temperature)
        self.schedule = schedule

    def on_step(self, step: int) -> None:
        if self.schedule is not None:
            self.temperature = max(self.schedule.value(step), 1e-6)

    def select(self, q: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        logits = np.where(valid, q / max(self.temperature, 1e-6), NEG_INF)
        logits = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights[~valid] = 0.0
        totals = weights.sum(axis=1, keepdims=True)
        totals[totals <= 0.0] = 1.0
        probabilities = weights / totals
        cumulative = np.cumsum(probabilities, axis=1)
        draws = rng.random((q.shape[0], 1))
        actions = np.sum(draws > cumulative, axis=1)
        return np.minimum(actions, q.shape[1] - 1)


_REGISTRY: dict[str, Callable[..., Policy]] = {}


def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


for _cls in (GreedyPolicy, EpsilonGreedyPolicy, SoftmaxPolicy):
    _REGISTRY[_cls.name] = _cls


def make_policy(name: str = "epsilon_greedy", **kwargs) -> Policy:
    """Instantiate a policy by registered name."""
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; available: {sorted(_REGISTRY)}") from exc
    return factory(**kwargs)


def available_policies() -> Iterable[str]:
    return sorted(_REGISTRY)
