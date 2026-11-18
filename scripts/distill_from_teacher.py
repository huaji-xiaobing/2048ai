#!/usr/bin/env python3
"""Distil the WASM expert into an n-tuple value table.

    python scripts/distill_from_teacher.py --games 64 --depth 4 --epochs 5 \
        --lr 0.01 --save checkpoints/distilled.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ntuple2048.distill import distill
from ntuple2048.inference import evaluate
from ntuple2048.network import NTupleNetwork
from ntuple2048.teacher import Teacher, merge_trajectories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--base", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--work", type=str, default="/tmp/ntuple2048_teacher")
    p.add_argument("--save", type=str, default="checkpoints/distilled.npz")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--reuse", action="store_true", help="reuse cached trajectories in --work")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()

    teacher = Teacher(speed=args.speed)
    print(f"teacher: speed={args.speed} node={teacher.node}", flush=True)
    cached = sorted(Path(args.work).glob("traj_*.bin"))
    if args.reuse and cached:
        boards, actions = merge_trajectories(cached)
        print(f"reused {len(cached)} cached trajectories", flush=True)
    else:
        boards, actions = teacher.generate(
            args.games, args.work, workers=args.workers, seed=args.seed, keep_files=True
        )
    print(
        f"generated {boards.shape[0]:,} states from {args.games} games "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    counts = [int((actions == i).sum()) for i in range(4)]
    print("teacher action histogram (L/R/U/D):", counts, flush=True)

    network = NTupleNetwork.load(args.resume) if args.resume else NTupleNetwork(base=args.base)
    print(f"network: base={network.base}", flush=True)

    t0 = time.monotonic()
    stats = distill(
        network,
        boards,
        actions,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        seed=args.seed,
    )
    print(f"distilled in {time.monotonic() - t0:.1f}s -> {json.dumps(stats.as_dict())}", flush=True)

    result = evaluate(network, episodes=args.eval_episodes, seed=args.seed + 1)
    print(json.dumps(result, indent=2), flush=True)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    network.save(args.save)
    print(f"saved -> {args.save}  (total {time.monotonic() - started:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
