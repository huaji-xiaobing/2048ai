"""Multi-core training driver.

One :class:`~ntuple2048.trainer.Trainer` instance runs per process; all of them
read and write a single weight buffer allocated with
``multiprocessing.RawArray``.  Updates are sparse scattered adds (Hogwild!
style), so concurrent workers occasionally drop an update -- harmless for TD
learning and far cheaper than locking on every step.

The main process owns the buffer too, which makes checkpointing and the
optional D4 projection a plain in-place operation between two chunks.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Callable

import numpy as np

from .network import NTupleNetwork
from .trainer import Stats, TrainConfig, Trainer, merge_stats

_CTX = mp.get_context("fork")
_WORKER: Trainer | None = None


def _worker_init(
    raw6,
    shape6,
    raw4,
    shape4,
    base: int,
    patterns6,
    patterns4,
    config: TrainConfig,
) -> None:
    global _WORKER
    W6 = np.ctypeslib.as_array(raw6).reshape(shape6)
    W4 = np.ctypeslib.as_array(raw4).reshape(shape4)
    network = NTupleNetwork.from_arrays(
        W6, W4, base=base, patterns6=patterns6, patterns4=patterns4
    )
    identity = mp.current_process()._identity
    worker_id = identity[0] if identity else 0
    _WORKER = Trainer(network, config, seed=config.seed + 7919 * worker_id)


def _worker_run(args: tuple[int, int]) -> Stats:
    steps, start_step = args
    assert _WORKER is not None, "worker was not initialised"
    return _WORKER.run(steps, start_step)


ChunkCallback = Callable[[int, Stats, float], None]


class ParallelTrainer:
    """Coordinates ``workers`` training processes over one shared weight buffer."""

    def __init__(
        self,
        config: TrainConfig,
        network: NTupleNetwork | None = None,
        workers: int | None = None,
    ) -> None:
        self.config = config
        self.workers = max(1, int(workers or os.cpu_count() or 1))

        base_network = network if network is not None else NTupleNetwork(config.base)
        self.config.base = base_network.base
        n6, l6 = base_network.W6.shape
        n4, l4 = base_network.W4.shape

        self._raw6 = _CTX.RawArray("f", n6 * l6)
        self._raw4 = _CTX.RawArray("f", n4 * l4)
        self.W6 = np.ctypeslib.as_array(self._raw6).reshape(n6, l6)
        self.W4 = np.ctypeslib.as_array(self._raw4).reshape(n4, l4)
        self.W6[...] = base_network.W6
        self.W4[...] = base_network.W4

        self.network = NTupleNetwork.from_arrays(
            self.W6,
            self.W4,
            base=base_network.base,
            patterns6=base_network.patterns6,
            patterns4=base_network.patterns4,
        )
        if network is None:
            # Drop the temporary tables before forking so the children do not
            # inherit (and keep alive through copy-on-write) a second copy.
            del base_network

        self._pool = _CTX.Pool(
            processes=self.workers,
            initializer=_worker_init,
            initargs=(
                self._raw6,
                (n6, l6),
                self._raw4,
                (n4, l4),
                self.network.base,
                self.network.patterns6,
                self.network.patterns4,
                self.config,
            ),
        )

    # ------------------------------------------------------------------ train
    def train(
        self,
        total_steps: int,
        chunk_steps: int = 5_000,
        save_path: str | None = None,
        save_every: int = 0,
        on_chunk: ChunkCallback | None = None,
        start_step: int = 0,
    ) -> Stats:
        """Run ``total_steps`` environment steps in total across all workers.

        ``chunk_steps`` is the number of steps each worker performs between two
        barriers; shorter chunks give finer monitoring and more frequent
        checkpointing, longer chunks waste less time in IPC.

        ``start_step`` is the number of environment steps already done by
        previous calls; it only feeds the epsilon schedule (and the ``on_chunk``
        counter), so a long run can be sliced into wall-clock sized pieces
        without restarting the exploration schedule.
        """
        if chunk_steps <= 0:
            raise ValueError("chunk_steps must be positive")

        total = Stats()
        step = 0
        round_index = 0
        started = time.monotonic()

        while step < total_steps:
            remaining = total_steps - step
            steps = max(1, min(chunk_steps, -(-remaining // self.workers)))
            results = self._pool.map(
                _worker_run, [(steps, start_step + step)] * self.workers
            )
            chunk = merge_stats(results)
            total.steps += chunk.steps
            total.episodes += chunk.episodes
            total.score_sum += chunk.score_sum
            total.length_sum += chunk.length_sum
            total.best_exponent = max(total.best_exponent, chunk.best_exponent)
            total.td_error_sum += chunk.td_error_sum
            total.td_count += chunk.td_count
            for exponent, count in chunk.tile_counts.items():
                total.tile_counts[exponent] = total.tile_counts.get(exponent, 0) + count

            step += chunk.steps
            round_index += 1

            if save_path and save_every and round_index % save_every == 0:
                self.network.save(save_path)
            if on_chunk is not None:
                on_chunk(start_step + step, chunk, time.monotonic() - started)

        return total

    # -------------------------------------------------------------- lifecycle
    def save(self, path: str, compress: bool = False) -> None:
        self.network.save(path, compress=compress)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def terminate(self) -> None:
        self._pool.terminate()
        self._pool.join()

    def __enter__(self) -> "ParallelTrainer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
