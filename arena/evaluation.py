"""
Fixed Baseline Evaluation and Regression Detection for CoVayAI (PR 6 / Master Plan V4 P9).
Provides standardized benchmarking of AlphaZero models against:
  1. RandomModel (sanity check against random play).
  2. CaptureHeuristicModel (tactical check against atari/capture logic).
  3. Frozen checkpoint models (historical strength reference).

Calculates paired bootstrap confidence intervals and exact permutation p-values.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
import os
import numpy as np
import torch

from go.board import Board
from go.zobrist import BLACK, WHITE
from nn.model import AlphaZeroNet
from arena.match import Arena, ArenaResult, compute_paired_bootstrap_ci, compute_paired_permutation_test
from arena.baseline import RandomModel, CaptureHeuristicModel


@dataclass
class BaselineEvaluationReport:
    """Standardized report summarizing evaluation against an immutable reference baseline."""

    baseline_name: str
    board_size: int
    win_rate: float
    paired_ci: Tuple[float, float]
    paired_p_value: float
    candidate_wins: int
    baseline_wins: int
    draws: int
    candidate_as_black_wins: int
    candidate_as_white_wins: int
    total_games: int
    avg_moves: float
    passed_sanity_check: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FixedBaselineEvaluator:
    """
    Evaluates candidate models against deterministic or frozen baselines under paired openings.
    Ensures that training progression can be measured against absolute external yardsticks,
    not merely relative to the preceding iteration's weights.
    """

    def __init__(
        self,
        board_size: int = 9,
        komi: Optional[float] = None,
        num_sims: int = 50,
        max_moves: Optional[int] = None,
        c_puct: float = 1.5,
        opening_plies: int = 8,
        device: Optional[torch.device] = None,
    ):
        self.board_size = board_size
        self.komi = komi if komi is not None else (0.5 if board_size <= 7 else 7.5)
        self.num_sims = num_sims
        self.max_moves = max_moves if max_moves is not None else (2 * board_size * board_size)
        self.c_puct = c_puct
        self.opening_plies = min(opening_plies, max(2, board_size - 1))
        self.device = device or torch.device("cpu")

    def _run_paired_arena(
        self,
        candidate_model: Any,
        baseline_model: Any,
        baseline_name: str,
        num_games: int = 20,
        base_seed: Optional[int] = None,
    ) -> BaselineEvaluationReport:
        """Run paired match between candidate and baseline, extracting standardized metrics."""
        arena = Arena(
            board_size=self.board_size,
            komi=self.komi,
            num_sims=self.num_sims,
            c_puct=self.c_puct,
            max_moves=self.max_moves,
            opening_plies=self.opening_plies,
            device=self.device,
        )

        res: ArenaResult = arena.evaluate(
            candidate_model=candidate_model,
            baseline_model=baseline_model,
            num_games=num_games,
            base_seed=base_seed,
        )

        paired_ci = res.pair_bootstrap_ci if res.pair_bootstrap_ci is not None else res.win_rate_ci
        p_val = res.paired_p_value if res.paired_p_value is not None else 1.0

        return BaselineEvaluationReport(
            baseline_name=baseline_name,
            board_size=self.board_size,
            win_rate=res.candidate_win_rate,
            paired_ci=paired_ci if paired_ci is not None else (0.0, 1.0),
            paired_p_value=p_val,
            candidate_wins=res.candidate_wins,
            baseline_wins=res.baseline_wins,
            draws=res.draws,
            candidate_as_black_wins=res.candidate_as_black_wins,
            candidate_as_white_wins=res.candidate_as_white_wins,
            total_games=res.games_played,
            avg_moves=res.avg_moves,
            passed_sanity_check=res.candidate_win_rate >= 0.50,
        )

    def evaluate_against_random(
        self,
        model: Any,
        num_games: int = 20,
        base_seed: Optional[int] = None,
    ) -> BaselineEvaluationReport:
        """Evaluate candidate against uniform Random baseline agent."""
        random_agent = RandomModel(allow_early_pass=False)
        report = self._run_paired_arena(
            candidate_model=model,
            baseline_model=random_agent,
            baseline_name="RandomModel",
            num_games=num_games,
            base_seed=base_seed,
        )
        report.passed_sanity_check = report.win_rate >= 0.70
        return report

    def evaluate_against_capture_heuristic(
        self,
        model: Any,
        num_games: int = 20,
        base_seed: Optional[int] = None,
    ) -> BaselineEvaluationReport:
        """Evaluate candidate against tactical CaptureHeuristic baseline agent."""
        heuristic_agent = CaptureHeuristicModel(board_size=self.board_size)
        report = self._run_paired_arena(
            candidate_model=model,
            baseline_model=heuristic_agent,
            baseline_name="CaptureHeuristicModel",
            num_games=num_games,
            base_seed=base_seed,
        )
        report.passed_sanity_check = report.win_rate >= 0.55
        return report

    def evaluate_against_frozen_checkpoint(
        self,
        model: Any,
        checkpoint_path: str,
        num_games: int = 20,
        base_seed: Optional[int] = None,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        aux_heads: bool = False,
    ) -> BaselineEvaluationReport:
        """Evaluate candidate against frozen checkpoint weights from previous iteration or release."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Frozen checkpoint not found at: {checkpoint_path}")

        baseline_net = AlphaZeroNet(
            board_size=self.board_size,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            aux_heads=aux_heads,
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        baseline_net.load_state_dict(state_dict)
        baseline_net.eval()

        baseline_name = os.path.basename(checkpoint_path)
        return self._run_paired_arena(
            candidate_model=model,
            baseline_model=baseline_net,
            baseline_name=f"FrozenCheckpoint({baseline_name})",
            num_games=num_games,
            base_seed=base_seed,
        )

    @staticmethod
    def check_regression(
        report: BaselineEvaluationReport,
        min_win_rate: float = 0.55,
    ) -> bool:
        """
        Return True if evaluation report indicates a detected regression in strength.
        A regression occurs when candidate win rate drops strictly below min_win_rate.
        """
        return bool(report.win_rate < min_win_rate)
