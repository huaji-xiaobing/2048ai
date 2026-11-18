"""Command line interface: ``python -m ntuple2048 {train,eval,bench,patterns}``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .inference import evaluate, play_game
from .network import DEFAULT_BASE, NTupleNetwork
from .parallel import ParallelTrainer
from .policies import LinearSchedule, available_policies
from .symmetry import orbit_report
from .trainer import Stats, TrainConfig, Trainer
from .tuples import default_patterns_4, default_patterns_6


def _realistic_boards(count: int, rng: np.random.Generator) -> np.ndarray:
    """Boards with 6..13 random tiles, close to a mid-game distribution.

    Freshly spawned boards are a bad probe for symmetry checks: their values are
    tiny and the measured deviation would look deceptively small.
    """
    boards = np.zeros((count, 16), dtype=np.uint8)
    order = rng.random((count, 16)).argsort(axis=1)
    n_tiles = rng.integers(6, 14, size=count)
    for k in range(16):
        selected = n_tiles > k
        if not selected.any():
            continue
        rows = np.flatnonzero(selected)
        boards[rows, order[rows, k]] = rng.integers(1, 11, size=rows.size).astype(np.uint8)
    return boards


def _fmt_stats(
    step: int, stats: Stats, elapsed: float, boards: int, epsilon: float
) -> str:
    info = stats.as_dict()
    # ``step`` counts environment steps summed over all workers; each step
    # advances ``boards`` boards inside its worker.
    sps = step * boards / max(elapsed, 1e-9)
    return (
        f"step {step:>10,d} | eps {epsilon:5.3f} | "
        f"ep {info['episodes']:>7,d} | score {info['mean_score']:>10.1f} | "
        f"len {info['mean_length']:>7.1f} | best {info['best_tile']:>6d} | "
        f"|td| {info['mean_abs_td']:.4f} | {sps / 1e3:8.1f}k sps | {elapsed:7.1f}s"
    )


# --------------------------------------------------------------------- train
def cmd_train(args: argparse.Namespace) -> int:
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

    network = None
    if args.resume:
        network = NTupleNetwork.load(args.resume)
        print(f"resumed from {args.resume} (base={network.base})")

    save_path = str(args.save) if args.save else None
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(
        f"training: workers={args.workers} boards/worker={args.boards} "
        f"total_boards={args.workers * args.boards} steps={args.steps:,} "
        f"base={args.base} lr={args.lr} gamma={args.gamma} reward={args.reward} "
        f"policy={args.policy} d4_augment={not args.no_d4_augment}"
    )

    started = time.monotonic()
    schedule = LinearSchedule(
        start=config.epsilon_start,
        end=config.epsilon_end,
        decay_steps=config.epsilon_decay_steps,
    )

    def on_chunk(step: int, stats: Stats, elapsed: float) -> None:
        print(
            _fmt_stats(step, stats, elapsed, args.boards, schedule.value(step)),
            flush=True,
        )

    if args.workers <= 1:
        trainer = Trainer(network or NTupleNetwork(args.base), config, seed=args.seed)
        step = 0
        while step < args.steps:
            chunk = min(args.chunk, args.steps - step)
            stats = trainer.run(chunk, start_step=step)
            step += chunk
            on_chunk(step, stats, time.monotonic() - started)
            if save_path and args.save_every and (step // args.chunk) % args.save_every == 0:
                trainer.net.save(save_path)
                print(f"saved -> {save_path}", flush=True)
        final_network = trainer.net
    else:
        with ParallelTrainer(config, network=network, workers=args.workers) as trainer:
            trainer.train(
                total_steps=args.steps,
                chunk_steps=args.chunk,
                save_path=save_path,
                save_every=args.save_every,
                on_chunk=on_chunk,
            )
            final_network = trainer.network
            if save_path:
                final_network.save(save_path)

    if save_path:
        final_network.save(save_path)
        print(f"saved -> {save_path}")

    probe = _realistic_boards(256, np.random.default_rng(args.seed + 12345))
    error = final_network.d4_consistency_error(probe)
    scale = float(np.abs(final_network.values(probe)).mean()) or 1.0
    print(
        f"D4 consistency: max |V(b) - V(g b)| = {error:.3f} "
        f"({error / scale:.2%} of mean |V|)"
    )

    if args.eval_episodes:
        result = evaluate(
            final_network,
            episodes=args.eval_episodes,
            seed=args.seed + 1,
            d4_average=args.eval_d4_average,
        )
        print(json.dumps(result, indent=2))
    return 0


# ---------------------------------------------------------------------- eval
def cmd_eval(args: argparse.Namespace) -> int:
    network = NTupleNetwork.load(args.model)
    print(f"loaded {args.model} (base={network.base})")
    if args.play:
        result = play_game(network, seed=args.seed, d4_average=args.d4_average, verbose=args.verbose)
        print(f"score={result.score} length={result.length} max_tile={result.max_tile}")
        print(result.board)
        return 0
    result = evaluate(
        network,
        episodes=args.episodes,
        seed=args.seed,
        d4_average=args.d4_average,
    )
    print(json.dumps(result, indent=2))
    return 0


# --------------------------------------------------------------------- bench
def cmd_bench(args: argparse.Namespace) -> int:
    network = NTupleNetwork(args.base)
    config = TrainConfig(
        base=args.base,
        boards=args.boards,
        d4_augment=not args.no_d4_augment,
        reward=args.reward,
    )
    trainer = Trainer(network, config, seed=0)
    trainer.run(50)  # warm-up
    started = time.monotonic()
    trainer.run(args.steps)
    elapsed = time.monotonic() - started
    board_steps = args.steps * args.boards
    print(
        f"{args.steps:,} steps x {args.boards} boards = {board_steps:,} board-steps "
        f"in {elapsed:.2f}s  ->  {board_steps / elapsed / 1e3:.1f}k board-steps/s"
    )
    return 0


# ------------------------------------------------------------------- distill
def cmd_distill(args: argparse.Namespace) -> int:
    from .distill import distill
    from .teacher import Teacher, merge_trajectories

    started = time.monotonic()
    cached = sorted(Path(args.work).glob("traj_*.bin")) if args.work else []
    if args.reuse and cached:
        boards, actions = merge_trajectories(cached)
        print(f"reused {len(cached)} cached trajectories ({boards.shape[0]:,} states)")
    else:
        teacher = Teacher(speed=args.speed)
        boards, actions = teacher.generate(
            args.games, args.work, workers=args.workers, seed=args.seed, keep_files=True
        )
    print(
        f"teacher data: {boards.shape[0]:,} states, "
        f"actions L/R/U/D = {[int((actions == i).sum()) for i in range(4)]}"
    )

    network = NTupleNetwork.load(args.resume) if args.resume else NTupleNetwork(base=args.base)
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
    print(f"distilled in {time.monotonic() - t0:.1f}s -> {json.dumps(stats.as_dict())}")

    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    network.save(str(path))
    print(f"saved -> {path}  (total {time.monotonic() - started:.1f}s)")

    if args.eval_episodes:
        print(json.dumps(evaluate(network, episodes=args.eval_episodes, seed=args.seed + 1), indent=2))
    return 0


# ------------------------------------------------------------------ patterns
def cmd_patterns(args: argparse.Namespace) -> int:
    p6 = default_patterns_6()
    p4 = default_patterns_4()
    print(f"6-tuples: {len(p6)}")
    for i, pattern in enumerate(p6):
        grid = np.zeros((4, 4), dtype=int)
        for t, pos in enumerate(pattern):
            grid[pos // 4, pos % 4] = t + 1
        print(f"  #{i:02d} cells={pattern} order={grid.tolist()}")
    print(f"4-tuples: {len(p4)}")
    for i, pattern in enumerate(p4):
        print(f"  #{i:02d} cells={pattern}")
    print("6-tuple orbits:", [len(o) for o in orbit_report(p6)])
    print("4-tuple orbits:", [len(o) for o in orbit_report(p4)])
    return 0


# ----------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ntuple2048",
        description="2048 simulation + n-tuple TD learning (no GUI)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a value table")
    p_train.add_argument("--workers", type=int, default=4, help="number of processes")
    p_train.add_argument("--boards", type=int, default=256, help="parallel boards per process")
    p_train.add_argument("--steps", type=int, default=1_000_000, help="total environment steps")
    p_train.add_argument("--chunk", type=int, default=2_000, help="steps per worker between barriers")
    p_train.add_argument("--base", type=int, default=DEFAULT_BASE, help="code base (15 -> tiles up to 16384)")
    p_train.add_argument("--lr", type=float, default=0.0025)
    p_train.add_argument("--gamma", type=float, default=0.999)
    p_train.add_argument("--reward", choices=("log", "score"), default="log")
    p_train.add_argument("--policy", choices=available_policies(), default="epsilon_greedy")
    p_train.add_argument("--epsilon-start", type=float, default=1.0)
    p_train.add_argument("--epsilon-end", type=float, default=0.05)
    p_train.add_argument("--epsilon-decay-steps", type=int, default=5_000_000)
    p_train.add_argument("--no-d4-augment", action="store_true",
                         help="disable random D4 board transforms during rollouts")
    p_train.add_argument("--save", type=str, default="checkpoints/model.npz")
    p_train.add_argument("--save-every", type=int, default=0, help="checkpoint every N chunks")
    p_train.add_argument("--resume", type=str, default=None)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--eval-episodes", type=int, default=0, help="evaluate after training")
    p_train.add_argument("--eval-d4-average", action="store_true",
                         help="use symmetry-averaged values for the post-training evaluation")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate a trained model")
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--episodes", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--d4-average", action="store_true",
                        help="average the value over the 8 board symmetries")
    p_eval.add_argument("--play", action="store_true", help="play a single game and print it")
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_bench = sub.add_parser("bench", help="measure rollout throughput")
    p_bench.add_argument("--boards", type=int, default=256)
    p_bench.add_argument("--steps", type=int, default=2_000)
    p_bench.add_argument("--base", type=int, default=DEFAULT_BASE)
    p_bench.add_argument("--reward", choices=("log", "score"), default="log")
    p_bench.add_argument("--no-d4-augment", action="store_true")
    p_bench.set_defaults(func=cmd_bench)

    p_dis = sub.add_parser("distill", help="distil the WASM expert web demo into a value table")
    p_dis.add_argument("--games", type=int, default=64, help="expert self-play games to generate")
    p_dis.add_argument("--workers", type=int, default=8, help="parallel Node processes")
    p_dis.add_argument("--speed", type=float, default=1.0,
                       help="expert speed slider: <1 faster/weaker, >1 slower/stronger")
    p_dis.add_argument("--epochs", type=int, default=5)
    p_dis.add_argument("--lr", type=float, default=0.01)
    p_dis.add_argument("--batch", type=int, default=2048)
    p_dis.add_argument("--base", type=int, default=DEFAULT_BASE)
    p_dis.add_argument("--seed", type=int, default=0)
    p_dis.add_argument("--work", type=str, default="/tmp/ntuple2048_teacher")
    p_dis.add_argument("--reuse", action="store_true", help="reuse cached trajectories in --work")
    p_dis.add_argument("--resume", type=str, default=None)
    p_dis.add_argument("--save", type=str, default="checkpoints/distilled.npz")
    p_dis.add_argument("--eval-episodes", type=int, default=50)
    p_dis.set_defaults(func=cmd_distill)

    p_pat = sub.add_parser("patterns", help="show the n-tuple patterns and their D4 orbits")
    p_pat.set_defaults(func=cmd_patterns)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
