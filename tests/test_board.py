import numpy as np
import pytest

from ntuple2048.board import DOWN, LEFT, RIGHT, UP, Board
from ntuple2048.env import batch_move, batch_move_all, batch_spawn, new_boards


def tiles(board: Board):
    return board.tiles()


def test_from_tiles_roundtrip():
    grid = [[2, 0, 4, 8], [16, 0, 0, 2], [0, 32, 0, 0], [64, 0, 0, 0]]
    board = Board.from_tiles(grid)
    assert tiles(board) == grid
    assert board.max_tile == 64


def test_simple_slide():
    board = Board.from_tiles([[0, 0, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    moved, gain_log, gain_real = board.move(LEFT)
    assert tiles(moved)[0] == [4, 0, 0, 0]
    assert gain_real == 4
    assert gain_log == 2


def test_no_double_merge():
    board = Board.from_tiles([[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    moved, _, gain_real = board.move(LEFT)
    assert tiles(moved)[0] == [4, 4, 0, 0]
    assert gain_real == 8


def test_merges_do_not_chain():
    board = Board.from_tiles([[4, 4, 8, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    moved, _, gain_real = board.move(LEFT)
    assert tiles(moved)[0] == [8, 8, 0, 0]
    assert gain_real == 8


@pytest.mark.parametrize(
    "direction,expected",
    [
        (LEFT, [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        (RIGHT, [[0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0]]),
        (UP, [[4, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        (DOWN, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 2, 0, 0]]),
    ],
)
def test_all_directions(direction, expected):
    board = Board.from_tiles([[2, 0, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    moved, _, _ = board.move(direction)
    assert tiles(moved) == expected


def test_invalid_move_is_identity():
    board = Board.from_tiles([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    moved, gain_log, gain_real = board.move(LEFT)
    assert moved is board
    assert gain_log == 0 and gain_real == 0
    assert board.valid_directions() == [DOWN]


def test_terminal_detection():
    full = Board.from_tiles(
        [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    )
    assert full.is_terminal()
    playable = Board.from_tiles(
        [[2, 2, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    )
    assert not playable.is_terminal()


def test_spawn_keeps_tile_count():
    rng = np.random.default_rng(0)
    board = Board.empty()
    for _ in range(10):
        board = board.with_spawn(rng)
    assert sum(1 for v in board.cells if v) == 10
    assert set(v for v in board.cells) <= {0, 1, 2}


def _random_boards(n, rng):
    boards = new_boards(n, rng)
    # add a few moves worth of tiles so boards are interesting
    for _ in range(6):
        boards = np.where(boards == 0, boards, boards)
        batch_spawn(boards, rng)
    return boards


@pytest.mark.parametrize("direction", [LEFT, RIGHT, UP, DOWN])
def test_batch_matches_scalar(direction):
    rng = np.random.default_rng(7)
    boards = _random_boards(64, rng)
    moved, gain_log, gain_real = batch_move(boards, direction)
    for i in range(boards.shape[0]):
        board = Board.from_cells(boards[i].tolist())
        expected_board, expected_log, expected_real = board.move(direction)
        assert moved[i].tolist() == list(expected_board.cells)
        assert int(gain_log[i]) == expected_log
        assert int(gain_real[i]) == expected_real


def test_batch_move_all_validity():
    rng = np.random.default_rng(3)
    boards = new_boards(16, rng)
    after, gain_log, gain_real, valid = batch_move_all(boards)
    assert after.shape == (16, 4, 16)
    assert valid.shape == (16, 4)
    for i in range(16):
        board = Board.from_cells(boards[i].tolist())
        for d in range(4):
            moved, gl, gr = board.move(d)
            assert valid[i, d] == (moved.cells != board.cells)
            assert after[i, d].tolist() == list(moved.cells)


def test_batch_spawn_respects_mask():
    rng = np.random.default_rng(11)
    boards = new_boards(8, rng)
    before = boards.copy()
    mask = np.array([True, False] * 4)
    batch_spawn(boards, rng, mask=mask)
    assert (boards[~mask] == before[~mask]).all()
    assert (boards[mask] != before[mask]).any(axis=1).all()
