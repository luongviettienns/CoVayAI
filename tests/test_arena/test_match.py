"""Unit tests for Arena match playing between two models."""

import pytest
import numpy as np
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
    """Verify that arena plays deterministically when opening_plies=0 (tau=0, no Dirichlet noise)."""
    m1, m2 = models
    # Running the same game twice with opening_plies=0 should yield the exact same winner and move count
    w1, c1 = play_arena_game(m1, m2, board_size=9, num_sims=5, max_moves=10, opening_plies=0)
    w2, c2 = play_arena_game(m1, m2, board_size=9, num_sims=5, max_moves=10, opening_plies=0)
    assert w1 == w2
    assert c1 == c2


def test_evaluate_against_random():
    """Verify evaluation against random player baseline."""
    from arena.baseline import evaluate_against_random
    m_5x5 = AlphaZeroNet(board_size=5, num_res_blocks=1, num_filters=16)
    m_5x5.eval()
    win_rate, wins, losses, draws = evaluate_against_random(
        model=m_5x5,
        board_size=5,
        komi=0.5,
        num_games=2,
        num_sims=5,
        max_moves=10,
    )
    assert 0.0 <= win_rate <= 1.0
    assert wins + losses + draws == 2


def test_generate_opening_sequence():
    from arena.match import generate_opening_sequence
    seq1 = generate_opening_sequence(board_size=9, num_plies=4, seed=100)
    seq2 = generate_opening_sequence(board_size=9, num_plies=4, seed=100)
    assert len(seq1) == 4
    assert seq1 == seq2
    # Verify moves are valid on a 9x9 board
    for m in seq1:
        assert 0 <= m < 81  # No pass in opening plies


def test_color_swap_uses_same_opening(models):
    m1, m2 = models
    arena = Arena(
        board_size=9,
        num_sims=5,
        max_moves=10,
        opening_plies=2,
    )
    result = arena.evaluate(
        candidate_model=m1,
        baseline_model=m2,
        num_games=2,  # 1 pair
    )
    assert result.paired_results is not None
    assert len(result.paired_results) == 1
    pair = result.paired_results[0]
    assert len(pair.opening_moves) == 2
    assert pair.cand_as_black_winner in (-1.0, 0.0, 1.0)
    assert pair.cand_as_white_winner in (-1.0, 0.0, 1.0)


def test_arena_score_is_symmetrical_on_model_swap(models):
    m1, m2 = models
    arena = Arena(
        board_size=9,
        num_sims=5,
        max_moves=10,
        opening_plies=2,
    )
    # Match 1: m1 as candidate, m2 as baseline
    res1 = arena.evaluate(m1, m2, num_games=2, base_seed=42)
    # Match 2: m2 as candidate, m1 as baseline with same seed
    res2 = arena.evaluate(m2, m1, num_games=2, base_seed=42)

    # Wins for m1 in match 1 should equal baseline wins in match 2
    assert res1.candidate_wins == res2.baseline_wins
    assert res1.baseline_wins == res2.candidate_wins
    assert res1.draws == res2.draws


def test_arena_result_confidence_interval(models):
    m1, m2 = models
    arena = Arena(board_size=9, num_sims=5, max_moves=10)
    result = arena.evaluate(m1, m2, num_games=4)
    assert result.win_rate_ci is not None
    ci_low, ci_high = result.win_rate_ci
    assert 0.0 <= ci_low <= result.candidate_win_rate <= ci_high <= 1.0


def test_capture_heuristic_baseline():
    from arena.baseline import CaptureHeuristicModel
    from go.board import Board
    from go.zobrist import BLACK, WHITE

    b = Board(size=5)
    # Setup atari on (0, 0): White stone at (0, 0)
    # Black stones at (0, 1) and (1, 0) surrounding it
    b.play(1)   # Black plays (0, 1)
    b.play(0)   # White plays (0, 0) - has only liberty at (1, 0)
    # Black now needs to capture by playing (1, 0) -> index 5
    heuristic = CaptureHeuristicModel(board_size=5)
    legal = b.get_legal_moves()
    probs, _ = heuristic.predict(None, legal_moves_mask=legal, board=b)
    best_move = int(np.argmax(probs))
    # Capture move is index 5
    assert best_move == 5

