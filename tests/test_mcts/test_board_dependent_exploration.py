"""Tests for board-dependent exploration, Dirichlet alpha scaling, and playout cap randomization."""

import numpy as np
import pytest
import torch

from go.board import Board
from mcts.search import (
    MCTS,
    get_default_dirichlet_alpha,
    get_default_temp_moves,
)
from selfplay.worker import play_selfplay_game, generate_selfplay_games


class MockModel:
    """Fast dummy model for deterministic MCTS testing."""
    def predict(self, state_features: np.ndarray, legal_moves_mask: np.ndarray = None, device=None):
        action_size = len(legal_moves_mask)
        probs = np.ones(action_size, dtype=np.float32)
        if legal_moves_mask is not None:
            probs = probs * (legal_moves_mask > 0).astype(np.float32)
        total = np.sum(probs)
        if total > 0:
            probs /= total
        return probs, 0.0


def test_default_dirichlet_alpha_scaling():
    """Verify Dirichlet noise alpha scales inversely with average legal moves across board sizes."""
    assert np.isclose(get_default_dirichlet_alpha(5), 0.60)
    assert np.isclose(get_default_dirichlet_alpha(7), 0.35)
    assert np.isclose(get_default_dirichlet_alpha(9), 0.25)
    assert np.isclose(get_default_dirichlet_alpha(13), 0.08)
    assert np.isclose(get_default_dirichlet_alpha(19), 0.03)


def test_default_temp_moves_scaling():
    """Verify temperature exploration moves scale with board size to allow opening diversity."""
    assert get_default_temp_moves(5) <= 8
    assert get_default_temp_moves(9) == 12
    assert get_default_temp_moves(13) >= 16
    assert get_default_temp_moves(19) >= 25


def test_mcts_uses_board_dependent_dirichlet_alpha():
    """Verify MCTS resolves Dirichlet alpha dynamically when not explicitly set."""
    model = MockModel()

    # MCTS without explicit dirichlet_alpha should resolve dynamically based on board size
    mcts_default = MCTS(model=model, num_simulations=5, dirichlet_alpha=None, add_noise=True)

    b19 = Board(size=19)
    # Search should run without errors using board-dependent alpha (0.03)
    pi19 = mcts_default.search(b19)
    assert len(pi19) == 362
    assert np.isclose(np.sum(pi19), 1.0)

    b9 = Board(size=9)
    mcts_default.reset()
    pi9 = mcts_default.search(b9)
    assert len(pi9) == 82
    assert np.isclose(np.sum(pi9), 1.0)


def test_mcts_search_simulation_override():
    """Verify search() allows overriding simulation count on a per-move basis."""
    model = MockModel()
    mcts = MCTS(model=model, num_simulations=50, add_noise=False)
    board = Board(size=9)

    # Override to only 3 simulations
    pi = mcts.search(board, num_simulations=3)
    # Total visits in tree should equal 3
    total_visits = sum(c.visit_count for c in mcts.root.children.values())
    assert total_visits == 3


def test_playout_cap_randomization_in_selfplay():
    """Verify playout cap randomization runs fast vs full simulations and trains only full-search moves."""
    model = MockModel()
    np.random.seed(42)

    # Run selfplay with playout cap randomization enabled
    # 20 full sims, 3 fast sims, 50% full search probability
    result = play_selfplay_game(
        model=model,
        board_size=9,
        num_simulations=20,
        fast_sims=3,
        full_search_prob=0.5,
        playout_cap_randomization=True,
        train_only_full_search=True,
        max_moves=20,
        disable_resign=True,
    )

    assert result.moves > 0
    assert result.full_search_count + result.fast_search_count == result.moves
    assert result.full_search_count > 0
    assert result.fast_search_count > 0

    # Training samples should only contain full-search positions
    assert len(result.samples) == result.full_search_count
    assert len(result.samples) < result.moves
