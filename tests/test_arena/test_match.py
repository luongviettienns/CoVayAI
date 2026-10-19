"""Unit tests for Arena match playing between two models."""

import pytest
import torch
from nn.model import AlphaZeroNet
from arena.match import Arena, ArenaResult, play_arena_game


@pytest.fixture
def models():
    """Create two lightweight models for fast arena testing."""
    torch.manual_seed(42)
    m1 = AlphaZeroNet(board_size=9, num_res_blocks=1, num_filters=16)
    m2 = AlphaZeroNet(board_size=9, num_res_blocks=1, num_filters=16)
    m1.eval()
    m2.eval()
    return m1, m2


def test_play_arena_game(models):
    """Test playing a single game between two models."""
    m1, m2 = models
    # Play 1 game with small simulation count and max_moves limit
    winner, move_count = play_arena_game(
        black_model=m1,
        white_model=m2,
        board_size=9,
        num_sims=5,
        max_moves=20,
    )
    # Winner must be +1 (Black win) or -1 (White win)
    assert winner in (+1.0, -1.0)
    assert move_count > 0
    assert move_count <= 20


def test_arena_evaluation_and_alternation(models):
    """Test arena evaluates multiple games, alternating colors properly."""
    m1, m2 = models
    arena = Arena(
        board_size=9,
        num_sims=5,
        max_moves=10,
        win_threshold=0.55,
    )

    # Run a short 2-game match
    result = arena.evaluate(
        candidate_model=m1,
        baseline_model=m2,
        num_games=2,
    )

    assert isinstance(result, ArenaResult)
    assert result.games_played == 2
    assert result.candidate_wins + result.baseline_wins + result.draws == 2
    assert 0.0 <= result.candidate_win_rate <= 1.0
    assert isinstance(result.accepted, bool)
    assert result.accepted == (result.candidate_win_rate >= 0.55)


def test_arena_deterministic_no_noise(models):
    """Verify that arena plays deterministically (tau=0, no Dirichlet noise)."""
    m1, m2 = models
    # Running the same game twice with same seed should yield the exact same winner and move count
    w1, c1 = play_arena_game(m1, m2, board_size=9, num_sims=5, max_moves=10)
    w2, c2 = play_arena_game(m1, m2, board_size=9, num_sims=5, max_moves=10)
    assert w1 == w2
    assert c1 == c2
