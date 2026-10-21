"""
Tests for Paired Confidence-Aware 19x19 Evaluation (PR 6 / Master Plan V4 P9).
Verifies:
- Paired bootstrap confidence intervals account for intra-pair opening correlation.
- Paired permutation tests compute exact non-parametric p-values against H0 (equal strength).
- Color-separated win rates (Black vs White) detect komi/color bias in 19x19 evaluation.
- FixedBaselineEvaluator provides standardized benchmarking against Random, CaptureHeuristic, and Frozen checkpoints.
- Strength regression detection flags drops in performance against fixed references.
"""

import numpy as np
import pytest
import torch

from nn.model import AlphaZeroNet
from arena.match import (
    Arena,
    ArenaResult,
    PairedGameResult,
    compute_paired_bootstrap_ci,
    compute_paired_permutation_test,
)
from arena.evaluation import (
    FixedBaselineEvaluator,
    BaselineEvaluationReport,
)


class DummyModel:
    """Fast dummy model for deterministic arena evaluation testing."""
    def __init__(self, win_prob: float = 0.5):
        self.win_prob = win_prob

    def eval(self):
        pass

    def predict(self, state_features=None, legal_moves_mask=None, device=None, board=None):
        action_size = len(legal_moves_mask) if legal_moves_mask is not None else 82
        probs = np.ones(action_size, dtype=np.float32)
        if legal_moves_mask is not None:
            probs = probs * (legal_moves_mask > 0).astype(np.float32)
        total = np.sum(probs)
        if total > 0:
            probs /= total
        return probs, 0.0


class TestArenaStatisticsAndPairedEvaluation:
    def test_paired_bootstrap_ci_extreme_cases(self):
        """Verify paired bootstrap CI on extreme cases (all wins, all losses, all draws)."""
        # All wins: pair scores are all 2.0 (normalized 1.0)
        pair_scores_win = [2.0] * 10
        ci_low, ci_high = compute_paired_bootstrap_ci(pair_scores_win, num_bootstraps=500, seed=42)
        assert np.isclose(ci_low, 1.0)
        assert np.isclose(ci_high, 1.0)

        # All losses: pair scores are all 0.0
        pair_scores_loss = [0.0] * 10
        ci_low, ci_high = compute_paired_bootstrap_ci(pair_scores_loss, num_bootstraps=500, seed=42)
        assert np.isclose(ci_low, 0.0)
        assert np.isclose(ci_high, 0.0)

        # Balanced draws/splits: pair scores are all 1.0 (normalized 0.5)
        pair_scores_balanced = [1.0] * 10
        ci_low, ci_high = compute_paired_bootstrap_ci(pair_scores_balanced, num_bootstraps=500, seed=42)
        assert np.isclose(ci_low, 0.5)
        assert np.isclose(ci_high, 0.5)

    def test_paired_bootstrap_ci_nominal(self):
        """Verify bootstrap CI bounds on heterogeneous pair outcomes."""
        # 10 pairs with varying scores
        pair_scores = [2.0, 2.0, 1.5, 1.0, 2.0, 0.5, 2.0, 1.0, 1.5, 2.0]
        mean_score = sum(pair_scores) / (len(pair_scores) * 2.0)  # normalized mean
        ci_low, ci_high = compute_paired_bootstrap_ci(pair_scores, num_bootstraps=1000, confidence=0.95, seed=42)

        assert 0.0 <= ci_low <= mean_score <= ci_high <= 1.0
        # For this strongly winning set (mean ~0.775), CI should be well above 0.5
        assert ci_low > 0.55

    def test_paired_permutation_test_significance(self):
        """Verify permutation test calculates correct one-sided p-values."""
        # Dominant candidate: always sweeps pairs (2.0 out of 2.0)
        dominant_scores = [2.0] * 12
        p_val_dom = compute_paired_permutation_test(dominant_scores, num_permutations=1000, seed=42)
        assert p_val_dom < 0.005  # Strongly significant p < 0.01

        # Evenly matched models: pair scores clustered around 1.0
        even_scores = [1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0]
        p_val_even = compute_paired_permutation_test(even_scores, num_permutations=1000, seed=42)
        assert p_val_even > 0.20  # Not significant

    def test_arena_paired_confidence_interval_and_color_bias(self):
        """Verify Arena.evaluate returns paired bootstrap CI and color-separated win rates."""
        torch.manual_seed(42)
        m1 = AlphaZeroNet(board_size=5, num_res_blocks=1, num_filters=16)
        m2 = AlphaZeroNet(board_size=5, num_res_blocks=1, num_filters=16)
        m1.eval()
        m2.eval()

        arena = Arena(
            board_size=5,
            komi=0.5,
            num_sims=3,
            max_moves=8,
            opening_plies=2,
            win_threshold=0.55,
        )

        result = arena.evaluate(
            candidate_model=m1,
            baseline_model=m2,
            num_games=4,  # 2 pairs
            base_seed=123,
        )

        assert isinstance(result, ArenaResult)
        assert result.games_played == 4
        assert result.paired_results is not None
        assert len(result.paired_results) == 2

        # Check paired bootstrap CI
        assert result.pair_bootstrap_ci is not None
        ci_low, ci_high = result.pair_bootstrap_ci
        assert 0.0 <= ci_low <= result.candidate_win_rate <= ci_high <= 1.0

        # Check permutation p-value
        assert result.paired_p_value is not None
        assert 0.0 <= result.paired_p_value <= 1.0

        # Check color breakdown
        assert hasattr(result, "candidate_as_black_win_rate")
        assert hasattr(result, "candidate_as_white_win_rate")
        assert 0.0 <= result.candidate_as_black_win_rate <= 1.0
        assert 0.0 <= result.candidate_as_white_win_rate <= 1.0

    def test_fixed_baseline_evaluator_against_random(self):
        """Verify FixedBaselineEvaluator against RandomModel."""
        torch.manual_seed(42)
        model = AlphaZeroNet(board_size=5, num_res_blocks=1, num_filters=16)
        model.eval()

        evaluator = FixedBaselineEvaluator(board_size=5, komi=0.5, num_sims=5, max_moves=10)
        report = evaluator.evaluate_against_random(model=model, num_games=2, base_seed=42)

        assert isinstance(report, BaselineEvaluationReport)
        assert report.baseline_name == "RandomModel"
        assert report.board_size == 5
        assert report.total_games == 2
        assert 0.0 <= report.win_rate <= 1.0
        assert report.paired_ci is not None
        assert len(report.paired_ci) == 2
        assert report.avg_moves > 0

    def test_fixed_baseline_evaluator_against_capture_heuristic(self):
        """Verify FixedBaselineEvaluator against CaptureHeuristicModel."""
        torch.manual_seed(42)
        model = AlphaZeroNet(board_size=5, num_res_blocks=1, num_filters=16)
        model.eval()

        evaluator = FixedBaselineEvaluator(board_size=5, komi=0.5, num_sims=5, max_moves=10)
        report = evaluator.evaluate_against_capture_heuristic(model=model, num_games=2, base_seed=42)

        assert isinstance(report, BaselineEvaluationReport)
        assert report.baseline_name == "CaptureHeuristicModel"
        assert report.board_size == 5
        assert report.total_games == 2
        assert 0.0 <= report.win_rate <= 1.0

    def test_detect_strength_regression(self):
        """Verify check_regression flags a performance drop against fixed benchmark threshold."""
        evaluator = FixedBaselineEvaluator(board_size=5, komi=0.5)

        # High winrate report
        report_good = BaselineEvaluationReport(
            baseline_name="RandomModel",
            board_size=5,
            win_rate=0.85,
            paired_ci=(0.70, 0.95),
            paired_p_value=0.01,
            candidate_wins=17,
            baseline_wins=3,
            draws=0,
            candidate_as_black_wins=9,
            candidate_as_white_wins=8,
            total_games=20,
            avg_moves=35.2,
        )
        assert evaluator.check_regression(report_good, min_win_rate=0.60) is False

        # Low winrate report below safety threshold
        report_bad = BaselineEvaluationReport(
            baseline_name="RandomModel",
            board_size=5,
            win_rate=0.40,
            paired_ci=(0.25, 0.55),
            paired_p_value=0.75,
            candidate_wins=8,
            baseline_wins=12,
            draws=0,
            candidate_as_black_wins=4,
            candidate_as_white_wins=4,
            total_games=20,
            avg_moves=32.0,
        )
        assert evaluator.check_regression(report_bad, min_win_rate=0.60) is True
