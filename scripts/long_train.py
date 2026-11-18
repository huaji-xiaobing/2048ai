#!/usr/bin/env python3
"""Wall-clock bounded training with periodic greedy evaluation.

The run is sliced into small pieces (``--slice-seconds``) so that an evaluation
can be triggered close to its scheduled wall-clock time instead of waiting for a
large fixed step budget to finish.  ``start_step`` is threaded through
:meth:`ParallelTrainer.train` so the epsilon schedule keeps annealing across
slices.

Outputs (under ``--out``)::

    model.npz        latest weights (also refreshed after every evaluation)
    eval_log.jsonl   one JSON object per evaluation
    train.log        stdout/stderr of the run (created by the launcher)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ntuple2048.inference import evaluate
from ntuple2048.parallel import ParallelTrainer
from ntuple2048.trainer import TrainConfig

SUMMARY_KEYS = (
    "mean_score",
    "median_score",
    "min_score",
    "max_score",
    "p10",
    "p90",
    "p99",
    "mean_length",
    "best_tile",
    "tile_hist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=60.0, help="total wall-clock budget")
    parser.add_argument("--eval-every", type=float, default=5.0, help="evaluation interval (minutes)")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--boards", type=int, default=128)
    parser.add_argument("--base", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--reward", choices=("log", "score"), default="log")
    parser.add_argument("--policy", default="epsilon_greedy")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=16_000_000)
    parser.add_argument("--no-d4-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="checkpoints/longrun")
    parser.add_argument("--slice-seconds", type=float, default=10.0,
                        help="granularity of the time slicing; smaller = tighter eval timing")
    parser.add_argument("--est-speed", type=float, default=13_000.0,
                        help="initial guess of global env-steps per second")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.npz"
    log_path = out_dir / "eval_log.jsonl"
    log_path.write_text("")

    config = TrainConfig(
        base=args.base,
        boards=args.boards,
        lr=args.lr,
        gamma=args.gamma,
        reward=args.reward,
        d4_augment=not args.no_d4_augment,
        seed=args.seed,
        policy=args.policy,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    total_planned = int(round(args.minutes / args.eval_every))
    interval = args.eval_every * 60.0

    start = time.monotonic()
    deadline = start + args.minutes * 60.0
    speed = args.est_speed
    total_steps = 0
    eval_index = 0

    print(
        f"=== long run: {args.minutes:.0f} min, eval every {args.eval_every:.0f} min "
        f"({total_planned} evaluations) ===",
        flush=True,
    )
    print(
        f"workers={args.workers} boards={args.boards} base={args.base} lr={args.lr} "
        f"gamma={args.gamma} reward={args.reward} d4_augment={not args.no_d4_augment} "
        f"epsilon_decay_steps={args.epsilon_decay_steps}",
        flush=True,
    )

    trainer = ParallelTrainer(config, workers=args.workers)
    try:
        while eval_index < total_planned:
            target = start + (eval_index + 1) * interval
            # train in small slices until the scheduled evaluation moment
            while True:
                now = time.monotonic()
                if now >= target or now >= deadline:
                    break
                budget = min(args.slice_seconds, target - now, deadline - now)
                if budget <= 0:
                    break
                slice_steps = max(1, int(speed * max(budget, 0.05)))
                t0 = time.monotonic()
                stats = trainer.train(
                    total_steps=slice_steps,
                    chunk_steps=slice_steps,
                    start_step=total_steps,
                )
                elapsed = max(time.monotonic() - t0, 1e-9)
                total_steps += stats.steps
                speed = 0.7 * speed + 0.3 * (stats.steps / elapsed)

            eval_index += 1
            t_eval = time.monotonic()
            result = evaluate(
                trainer.network,
                episodes=args.eval_episodes,
                seed=args.seed + 1000 * eval_index,
            )
            trainer.save(str(model_path))

            entry = {
                "eval": eval_index,
                "elapsed_min": round((t_eval - start) / 60.0, 2),
                "steps": total_steps,
                "board_steps": total_steps * args.boards,
                "sps": round(speed, 1),
                "epsilon": round(
                    config.epsilon_start
                    + (config.epsilon_end - config.epsilon_start)
                    * min(total_steps / max(config.epsilon_decay_steps, 1), 1.0),
                    4,
                ),
                **{key: result[key] for key in SUMMARY_KEYS},
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(
                f"[eval {eval_index:02d}/{total_planned}] t={entry['elapsed_min']:6.2f}min "
                f"steps={total_steps:,} eps={entry['epsilon']:.3f} | "
                f"mean={result['mean_score']:9.1f} median={result['median_score']:9.1f} "
                f"p10={result['p10']:8.1f} min={result['min_score']:6d} max={result['max_score']:6d} "
                f"| tile={result['best_tile']:5d} len={result['mean_length']:6.1f} "
                f"| {speed / 1000:.1f}k step/s",
                flush=True,
            )

            if time.monotonic() >= deadline and eval_index >= total_planned:
                break
    except KeyboardInterrupt:
        print("interrupted, saving...", flush=True)
    finally:
        trainer.save(str(model_path))
        trainer.close()
        print(f"=== done: {total_steps:,} env steps, model -> {model_path} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
