import numpy as np
import pytest

from ntuple2048.inference import best_action, evaluate, play_game
from ntuple2048.board import Board
from ntuple2048.network import NTupleNetwork
from ntuple2048.policies import EpsilonGreedyPolicy, LinearSchedule, SoftmaxPolicy, make_policy
from ntuple2048.trainer import TrainConfig, Trainer, merge_stats


def test_policy_factory_and_schedules():
    policy = make_policy("epsilon_greedy", schedule=LinearSchedule(1.0, 0.1, 100))
    assert policy.epsilon_at(0) == pytest.approx(1.0)
    assert policy.epsilon_at(100) == pytest.approx(0.1)
    assert isinstance(make_policy("softmax"), SoftmaxPolicy)


def test_epsilon_greedy_only_picks_valid_actions():
    rng = np.random.default_rng(0)
    q = np.array([[1.0, 5.0, -1e30, 2.0], [3.0, -1e30, -1e30, 4.0]])
    valid = np.array([[True, True, False, True], [True, False, False, True]])
    policy = EpsilonGreedyPolicy(epsilon=0.5)
    for _ in range(200):
        actions = policy.select(q, valid, rng)
        assert valid[np.arange(2), actions].all()


def test_trainer_updates_tables():
    net = NTupleNetwork(base=6)
    config = TrainConfig(base=6, boards=32, lr=0.01, seed=1)
    trainer = Trainer(net, config)
    stats = trainer.run(200)
    assert stats.steps == 200
    assert np.abs(net.W6).sum() > 0
    assert stats.td_count > 0


def test_stats_merge():
    a = Trainer(NTupleNetwork(base=4), TrainConfig(base=4, boards=8, seed=0)).run(50)
    b = Trainer(NTupleNetwork(base=4), TrainConfig(base=4, boards=8, seed=1)).run(50)
    merged = merge_stats([a, b])
    assert merged.steps == a.steps + b.steps
    assert merged.episodes == a.episodes + b.episodes


def test_inference_runs():
    net = NTupleNetwork(base=6)
    board = Board.random_start(np.random.default_rng(0))
    direction, value = best_action(net, board)
    assert direction in (0, 1, 2, 3)
    result = play_game(net, seed=3)
    assert result.length > 0
    assert result.max_tile >= 2


def test_evaluate_returns_summary():
    net = NTupleNetwork(base=6)
    summary = evaluate(net, episodes=6, seed=0)
    assert summary["episodes"] == 6
    assert summary["mean_score"] >= 0
    assert summary["tile_hist"]


def test_d4_average_matches_for_symmetric_network():
    net = NTupleNetwork(base=6)
    summary_a = evaluate(net, episodes=4, seed=0, d4_average=False)
    summary_b = evaluate(net, episodes=4, seed=0, d4_average=True)
    # zero network -> identical (deterministic tie breaking)
    assert summary_a["mean_score"] == summary_b["mean_score"]


def test_parallel_trainer_smoke():
    from ntuple2048.parallel import ParallelTrainer

    config = TrainConfig(base=5, boards=16, lr=0.01, seed=0)
    trainer = ParallelTrainer(config, workers=2)
    try:
        stats = trainer.train(total_steps=64, chunk_steps=32)
        assert stats.steps == 64
        assert np.abs(trainer.network.W6).sum() > 0
        out = trainer.network.values(np.zeros((1, 16), dtype=np.uint8))
        assert out.shape == (1,)
    finally:
        trainer.close()
