"""Arena evaluation and Elo rating module for CoVayAI."""

from arena.elo import (
    expected_score,
    update_ratings,
    estimate_rating_diff,
    EloTracker,
)
from arena.match import (
    Arena,
    ArenaResult,
    PairedGameResult,
    play_arena_game,
    compute_wilson_score_interval,
    compute_paired_bootstrap_ci,
    compute_paired_permutation_test,
)
from arena.evaluation import (
    FixedBaselineEvaluator,
    BaselineEvaluationReport,
)

__all__ = [
    "expected_score",
    "update_ratings",
    "estimate_rating_diff",
    "EloTracker",
    "Arena",
    "ArenaResult",
    "PairedGameResult",
    "play_arena_game",
    "compute_wilson_score_interval",
    "compute_paired_bootstrap_ci",
    "compute_paired_permutation_test",
    "FixedBaselineEvaluator",
    "BaselineEvaluationReport",
]
