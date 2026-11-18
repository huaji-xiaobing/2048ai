import numpy as np
import pytest

from ntuple2048.env import new_boards
from ntuple2048.network import NTupleNetwork
from ntuple2048.tuples import default_patterns_4, default_patterns_6


@pytest.fixture(scope="module")
def small_net():
    return NTupleNetwork(base=5)


def test_pattern_counts():
    assert len(default_patterns_6()) == 12
    assert len(default_patterns_4()) == 8
    for pattern in default_patterns_6():
        assert len(pattern) == 6
        assert len(set(pattern)) == 6
    for pattern in default_patterns_4():
        assert len(pattern) == 4
        assert len(set(pattern)) == 4


def test_patterns_cover_every_cell():
    cells = set()
    for pattern in default_patterns_6():
        cells.update(pattern)
    assert cells == set(range(16))


def test_code_matrix_matches_naive(small_net):
    rng = np.random.default_rng(0)
    boards = new_boards(32, rng)
    c6, c4 = small_net.codes(boards)
    clipped = np.minimum(boards, small_net.base - 1)
    for i in range(32):
        for p, pattern in enumerate(small_net.patterns6):
            expected = 0
            for t, pos in enumerate(pattern):
                expected += int(clipped[i, pos]) * small_net.base**t
            assert c6[i, p] == expected
        for p, pattern in enumerate(small_net.patterns4):
            expected = 0
            for t, pos in enumerate(pattern):
                expected += int(clipped[i, pos]) * small_net.base**t
            assert c4[i, p] == expected


def test_forward_equals_manual_sum(small_net):
    rng = np.random.default_rng(1)
    boards = new_boards(16, rng)
    small_net.W6 = rng.normal(size=small_net.W6.shape).astype(np.float32)
    small_net.W4 = rng.normal(size=small_net.W4.shape).astype(np.float32)
    small_net._after_load()

    values = small_net.values(boards)
    c6, c4 = small_net.codes(boards)
    for i in range(16):
        expected = sum(small_net.W6[p, c6[i, p]] for p in range(small_net.n6))
        expected += sum(small_net.W4[p, c4[i, p]] for p in range(small_net.n4))
        assert values[i] == pytest.approx(expected, rel=1e-5)


def test_sparse_update_matches_dense(small_net):
    rng = np.random.default_rng(2)
    boards = new_boards(8, rng)
    c6, c4 = small_net.codes(boards)
    deltas = rng.normal(size=8).astype(np.float32) * 0.1

    # one sample at a time so that no two writes can collide on the same cell
    for i in range(8):
        net = NTupleNetwork(base=small_net.base)
        net.update_from_codes(c6[i : i + 1], c4[i : i + 1], deltas[i : i + 1])
        for p in range(net.n6):
            assert net.W6[p, c6[i, p]] == pytest.approx(deltas[i], rel=1e-6)
        for p in range(net.n4):
            assert net.W4[p, c4[i, p]] == pytest.approx(deltas[i], rel=1e-6)


def test_save_load_roundtrip(small_net, tmp_path):
    rng = np.random.default_rng(3)
    small_net.W6 = rng.normal(size=small_net.W6.shape).astype(np.float32)
    small_net.W4 = rng.normal(size=small_net.W4.shape).astype(np.float32)
    small_net._after_load()

    path = tmp_path / "model.npz"
    small_net.save(path)
    loaded = NTupleNetwork.load(path)
    assert loaded.base == small_net.base
    assert loaded.patterns6 == small_net.patterns6
    assert loaded.patterns4 == small_net.patterns4
    np.testing.assert_allclose(loaded.W6, small_net.W6)
    np.testing.assert_allclose(loaded.W4, small_net.W4)


def test_default_base_tables_are_large_but_valid():
    net = NTupleNetwork(base=15)
    assert net.W6.shape == (12, 15**6)
    assert net.W4.shape == (8, 15**4)
    assert net.values(new_boards(4, np.random.default_rng(0))).shape == (4,)
