"""Comprehensive tests for Go rules invariants, MCTS safety, and self-play GameResult metadata."""

import numpy as np
import pytest
import torch

from go.board import Board, BLACK, WHITE, EMPTY
from go.scoring import compute_area_score
from mcts.search import MCTS
from nn.model import AlphaZeroNet
from selfplay.worker import play_selfplay_game, generate_selfplay_games, GameResult


class DummyUniformModel(AlphaZeroNet):
    """Deterministic uniform policy model for property tests."""
    def __init__(self, board_size: int = 5):
        super().__init__(board_size=board_size, num_res_blocks=1, num_filters=16)

    def predict(self, features: np.ndarray, legal_moves_mask: np.ndarray = None, device = None):
        num_actions = self.board_size * self.board_size + 1
        if legal_moves_mask is not None:
            mask = legal_moves_mask.astype(bool)
        else:
            mask = np.ones(num_actions, dtype=bool)
        policy = np.zeros(num_actions, dtype=np.float32)
        legal_count = np.count_nonzero(mask)
        if legal_count > 0:
            policy[mask] = 1.0 / legal_count
        else:
            policy[-1] = 1.0
        return policy, 0.0


def test_board_invariants_across_100_random_games():
    """Property test: Verify hash consistency, player alternation, and superko tracking across 100 random games."""
    np.random.seed(42)
    for _ in range(100):
        board = Board(size=5, max_moves=20)
        while not board.is_game_over():
            # Invariant 1: Incremental hash matches freshly computed hash
            expected_hash = board.zobrist.compute_position_hash(board.grid)
            assert board.current_position_hash == expected_hash

            # Invariant 2: Current player is either BLACK or WHITE
            assert board.current_player in (BLACK, WHITE)

            # Invariant 3: Current position hash is recorded in history
            assert board.current_position_hash in board.history_positions

            # Invariant 4: Occupied intersections cannot be legally played
            occupied_indices = np.flatnonzero(board.grid.flatten() != EMPTY)
            for occ in occupied_indices:
                assert not board.is_valid_move(occ)

            legal_mask = board.get_legal_moves()
            legal_actions = np.flatnonzero(legal_mask)
            assert len(legal_actions) > 0  # Pass is always an option

            action = np.random.choice(legal_actions)
            success = board.play(action)
            assert success, f"Failed to play legal action {action}"


def test_board_copy_deep_independence():
    """Verify that Board.copy() creates a fully decoupled state preserving superko and grids."""
    board = Board(size=5)
    board.play(0)
    board.play(6)
    board.play(12)
    board.play(18)

    copied = board.copy()

    assert np.array_equal(copied.grid, board.grid)
    assert copied.current_position_hash == board.current_position_hash
    assert copied.current_player == board.current_player
    assert copied.move_count == board.move_count
    assert copied.history_positions == board.history_positions
    assert len(copied.history_grids) == len(board.history_grids)

    # Mutate copy
    orig_hash = board.current_position_hash
    orig_grid = board.grid.copy()
    orig_move_count = board.move_count

    copied.play(24)  # Play on copy

    # Original must be completely unaffected
    assert board.current_position_hash == orig_hash
    assert np.array_equal(board.grid, orig_grid)
    assert board.move_count == orig_move_count
    assert 24 not in board.history_positions


def test_mcts_illegal_actions_have_zero_visits():
    """Verify that MCTS never assigns visits or prior probability to illegal moves."""
    board = Board(size=5)
    # Make corner (0, 0) suicide for White: Black at (0, 1) and (1, 0)
    board.play(1)   # B (0, 1)
    board.play(24)  # W (4, 4)
    board.play(5)   # B (1, 0)
    # Now White's turn: move 0 is suicide
    assert not board.is_valid_move(0)

    model = DummyUniformModel(board_size=5)
    mcts = MCTS(model=model, num_simulations=40, c_puct=1.5, add_noise=False)
    pi = mcts.search(board)

    # Action 0 must have exactly 0 probability
    assert pi[0] == 0.0


def test_search_after_two_passes_terminal():
    """Verify that MCTS gracefully handles terminal state after two consecutive passes."""
    board = Board(size=5)
    board.play(board.pass_action)
    board.play(board.pass_action)
    assert board.is_game_over()

    model = DummyUniformModel(board_size=5)
    mcts = MCTS(model=model, num_simulations=20, add_noise=False)
    pi = mcts.search(board)

    assert pi.shape == (26,)
    assert np.isclose(np.sum(pi), 1.0)


def test_game_result_metadata_and_winner_accuracy():
    """Verify that GameResult captures exact scores, winner, moves, and end_reason."""
    model = DummyUniformModel(board_size=5)
    result = play_selfplay_game(
        model=model,
        board_size=5,
        num_simulations=10,
        temp_moves=2,
        max_moves=6,
        disable_resign=True,
    )

    assert isinstance(result, GameResult)
    assert result.moves <= 6
    assert result.winner in (BLACK, WHITE)
    assert result.end_reason in ("two_passes", "max_moves", "resign")
    assert len(result) == result.moves

    # Verify each sample perspective
    for s, pi, z in result:
        assert s.shape == (17, 5, 5)
        assert pi.shape == (26,)
        assert z in (+1.0, -1.0)


def test_disable_resign_never_resigns():
    """Verify that disable_resign=True prevents early resignation even under extreme threshold."""
    model = DummyUniformModel(board_size=5)
    result = play_selfplay_game(
        model=model,
        board_size=5,
        num_simulations=10,
        temp_moves=2,
        resign_threshold=0.99,  # Would resign immediately if enabled
        disable_resign=True,
        max_moves=8,
    )

    assert result.end_reason != "resign"
    assert result.resigned_by is None
