"""Tests for distinguishing terminal and truncated games in Board, Self-play, and Arena."""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from go.board import Board, BLACK, WHITE
from go.scoring import compute_area_score
from selfplay.worker import play_selfplay_game, GameResult
from arena.match import play_arena_game


class MockPolicyModel:
    """Mock model that returns uniform policy and neutral value."""
    def __init__(self, board_size: int = 5):
        self.board_size = board_size
        self.action_size = board_size * board_size + 1

    def predict(self, state_features, legal_moves_mask=None, device=None):
        probs = np.ones(self.action_size, dtype=np.float32)
        if legal_moves_mask is not None:
            probs = probs * legal_moves_mask
            s = probs.sum()
            if s > 0:
                probs /= s
            else:
                probs = np.ones(self.action_size, dtype=np.float32) / self.action_size
        else:
            probs /= self.action_size
        return probs, 0.0


def test_board_distinguishes_terminal_and_truncated():
    """Verify that Board accurately separates natural terminal state from move-capped truncation."""
    b = Board(size=5, max_moves=10)
    assert not b.is_game_over()
    assert not b.is_terminal()
    assert not b.is_truncated()

    # Case 1: Terminal game via 2 consecutive passes
    pass_act = 5 * 5
    b.play(pass_act)
    assert not b.is_game_over()
    assert not b.is_terminal()
    b.play(pass_act)
    assert b.is_game_over()
    assert b.is_terminal()
    assert not b.is_truncated()

    # Case 2: Truncated game via max_moves limit without 2 passes
    b2 = Board(size=5, max_moves=4)
    # Play 4 legal stone moves (no passes)
    b2.play(0)  # (0, 0)
    b2.play(1)  # (0, 1)
    b2.play(2)  # (0, 2)
    b2.play(3)  # (0, 3)
    assert b2.move_count == 4
    assert b2.is_game_over()
    assert not b2.is_terminal()
    assert b2.is_truncated()


def test_selfplay_truncated_game_does_not_create_fake_z_by_default():
    """Verify that when a game is truncated by max_moves, no fake z outcome is created by default."""
    model = MockPolicyModel(board_size=5)
    result = play_selfplay_game(
        model=model,
        board_size=5,
        num_simulations=5,
        temp_moves=2,
        max_moves=4,  # Truncate after 4 moves
        disable_resign=True,
        adjudicate_truncated=False,  # Default: do NOT create fake z outcome
    )

    assert isinstance(result, GameResult)
    assert result.is_truncated is True
    assert result.end_reason in ("truncated", "max_moves")
    assert result.termination_reason == "truncated"
    assert result.winner == 0  # No winner assigned to incomplete game
    assert len(result.samples) == 0  # 0 fake outcome labels created!


def test_selfplay_truncated_game_with_explicit_adjudication():
    """Verify that truncated game can be adjudicated when explicitly requested."""
    model = MockPolicyModel(board_size=5)
    result = play_selfplay_game(
        model=model,
        board_size=5,
        num_simulations=5,
        temp_moves=2,
        max_moves=4,
        disable_resign=True,
        adjudicate_truncated=True,  # Explicit adjudication
    )

    assert result.is_truncated is True
    assert result.end_reason in ("truncated", "max_moves")
    assert result.termination_reason == "truncated"
    assert result.winner in (BLACK, WHITE)
    assert len(result.samples) == 4
    for s, pi, z in result.samples:
        assert z in (-1.0, 1.0)


def test_arena_truncated_game_returns_draw():
    """Verify that in Arena, a game cut off by max_moves without termination evaluates to a draw (0.0) when adjudicate_truncated=False."""
    m1 = MockPolicyModel(board_size=5)
    m2 = MockPolicyModel(board_size=5)

    winner_val, move_count = play_arena_game(
        black_model=m1,
        white_model=m2,
        board_size=5,
        num_sims=5,
        max_moves=4,  # Truncate game after 4 plies
        opening_plies=0,
        adjudicate_truncated=False,
    )
    assert move_count == 4
    assert winner_val == 0.0, "Truncated arena game must evaluate to Draw (0.0), not a fake win!"
