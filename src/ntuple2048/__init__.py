"""2048 simulation, n-tuple TD training and greedy inference.

Public entry points::

    from ntuple2048 import Board, NTupleNetwork, Trainer, TrainConfig
    from ntuple2048 import evaluate, play_game, best_action
"""

from .board import DIRECTIONS, DIRECTION_NAMES, Board
from .distill import DistillStats, distill
from .env import batch_move, batch_move_all, new_boards
from .inference import (
    EpisodeResult,
    action_values,
    best_action,
    evaluate,
    play_game,
    symmetry_averaged_values,
)
from .network import DEFAULT_BASE, NTupleNetwork
from .policies import (
    EpsilonGreedyPolicy,
    GreedyPolicy,
    LinearSchedule,
    Policy,
    SoftmaxPolicy,
    make_policy,
    register,
)
from .teacher import Teacher, merge_trajectories, read_trajectory
from .trainer import Stats, TrainConfig, Trainer, merge_stats
from .tuples import default_patterns_4, default_patterns_6

__version__ = "0.1.0"

__all__ = [
    "DIRECTIONS",
    "DIRECTION_NAMES",
    "Board",
    "DEFAULT_BASE",
    "DistillStats",
    "EpsilonGreedyPolicy",
    "EpisodeResult",
    "GreedyPolicy",
    "Teacher",
    "distill",
    "merge_trajectories",
    "read_trajectory",
    "LinearSchedule",
    "NTupleNetwork",
    "Policy",
    "SoftmaxPolicy",
    "Stats",
    "TrainConfig",
    "Trainer",
    "action_values",
    "batch_move",
    "batch_move_all",
    "best_action",
    "default_patterns_4",
    "default_patterns_6",
    "evaluate",
    "make_policy",
    "merge_stats",
    "new_boards",
    "play_game",
    "register",
    "symmetry_averaged_values",
]
