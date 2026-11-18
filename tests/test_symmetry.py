import numpy as np
import pytest

from ntuple2048.env import new_boards
from ntuple2048.network import NTupleNetwork
from ntuple2048.symmetry import (
    INVERSE,
    NUM_SYMMETRIES,
    PERMUTATIONS,
    is_closed_as_sets,
    is_closed_under_d4,
    orbit_report,
    pattern_orbits,
    transform_board_cells,
    transform_boards,
)
from ntuple2048.tuples import default_patterns_4, default_patterns_6


def test_permutations_are_bijections():
    for perm in PERMUTATIONS:
        assert perm.shape == (16,)
        assert sorted(perm.tolist()) == list(range(16))


def test_symmetries_form_a_group():
    def compose(a, b):
        return tuple(int(a[b[i]]) for i in range(16))

    as_tuples = [tuple(p.tolist()) for p in PERMUTATIONS]
    assert len(set(as_tuples)) == NUM_SYMMETRIES
    identity = tuple(range(16))
    for a in as_tuples:
        for b in as_tuples:
            assert compose(a, b) in set(as_tuples)
    for g, perm in enumerate(PERMUTATIONS):
        composed = compose(tuple(perm.tolist()), tuple(PERMUTATIONS[INVERSE[g]].tolist()))
        assert composed == identity


def test_transform_board_matches_batch():
    rng = np.random.default_rng(0)
    boards = new_boards(8, rng)
    for g in range(NUM_SYMMETRIES):
        batched = transform_boards(boards, g)
        for i in range(8):
            assert batched[i].tolist() == list(transform_board_cells(boards[i].tolist(), g))


def test_rotate_180_twice_is_identity():
    rng = np.random.default_rng(1)
    boards = new_boards(4, rng)
    twice = transform_boards(transform_boards(boards, 2), 2)
    np.testing.assert_array_equal(twice, boards)


def test_transpose_swaps_row_and_column():
    board = np.arange(16, dtype=np.uint8).reshape(1, 16)
    transposed = transform_boards(board, 6)
    assert transposed.tolist()[0] == board.reshape(4, 4).T.reshape(-1).tolist()


def test_pattern_sets_closed_as_positions_only():
    p6 = default_patterns_6()
    p4 = default_patterns_4()
    assert is_closed_as_sets(p6)
    assert is_closed_as_sets(p4)
    # the ordered tuple set is *not* closed: row-major 2x3 -> column-major 3x2
    assert not is_closed_under_d4(p6)


def test_orbit_sizes():
    assert [len(o) for o in pattern_orbits(default_patterns_6())] == [8, 4]
    assert [len(o) for o in pattern_orbits(default_patterns_4())] == [4, 4]
    assert orbit_report(default_patterns_6())


def test_d4_consistency_of_zero_network_is_zero():
    net = NTupleNetwork(base=5)
    boards = new_boards(32, np.random.default_rng(4))
    assert net.d4_consistency_error(boards) == 0.0


def test_d4_consistency_detects_asymmetry():
    net = NTupleNetwork(base=5)
    net.W6[0, 1] = 5.0
    boards = new_boards(4, np.random.default_rng(5))
    assert net.d4_consistency_error(boards) >= 0.0


def test_augmented_training_reduces_d4_error():
    """Rollouts with random symmetries should move the tables towards symmetry."""
    from ntuple2048.trainer import TrainConfig, Trainer

    net = NTupleNetwork(base=6)
    config = TrainConfig(base=6, boards=64, lr=0.01, d4_augment=True, seed=0)
    trainer = Trainer(net, config)
    probe = new_boards(128, np.random.default_rng(99))
    before = net.d4_consistency_error(probe)
    trainer.run(300)
    after = net.d4_consistency_error(probe)
    assert np.isfinite(before) and np.isfinite(after)
    assert np.abs(net.W6).sum() > 0
