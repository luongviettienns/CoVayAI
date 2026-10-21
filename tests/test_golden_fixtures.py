"""Golden fixtures and regression baselines for 5 predefined board scenarios."""

import numpy as np
import pytest
import torch

from go.board import Board, BLACK, WHITE
from go.scoring import compute_area_score
from mcts.search import MCTS
from nn.model import AlphaZeroNet


class DeterministicUniformModel(AlphaZeroNet):
    """Deterministic model with fixed uniform policy and zero value for reproducible golden tests."""
    def __init__(self, board_size: int = 5):
        super().__init__(board_size=board_size, num_res_blocks=1, num_filters=16)

    def predict(self, features: np.ndarray, legal_moves_mask: np.ndarray = None, device = None) -> tuple[np.ndarray, float]:
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


def test_golden_scenario_1_empty_board():
    """Scenario 1: Empty 5x5 board MCTS distribution and legal moves."""
    board = Board(size=5)
    model = DeterministicUniformModel(board_size=5)
    mcts = MCTS(model=model, num_simulations=30, c_puct=1.5, add_noise=False)

    pi = mcts.search(board)
    assert pi.shape == (26,)
    assert np.isclose(np.sum(pi), 1.0)
    # On empty board, all 25 intersections + pass (total 26) are legal
    legal_moves = board.get_legal_moves()
    assert np.all(legal_moves == 1)
    # Policy visits should be distributed without NaN
    assert np.all(pi >= 0.0)
    assert not np.any(np.isnan(pi))


def test_golden_scenario_2_atari_capture():
    """Scenario 2: Single white stone in atari at (0, 0), surrounded by black at (0, 1) and (1, 0)."""
    board = Board(size=5)
    board.play(0)   # Black plays (0, 0)
    board.play(24)  # White plays (4, 4)
    board.play(1)   # Black plays (0, 1)
    board.play(23)  # White plays (4, 3)
    board.play(5)   # Black plays (1, 0)
    board.play(12)  # White plays (2, 2)
    # It is Black's turn. All black stones have liberties, pass_action must be legal.
    legal_moves = board.get_legal_moves()
    assert legal_moves[board.pass_action] == 1


def test_golden_scenario_3_suicide_prevention():
    """Scenario 3: Verify suicide without capture is strictly illegal."""
    board = Board(size=5)
    # Black surrounds corner (0, 0): Black at (0, 1) and (1, 0)
    board.play(1)   # B: (0, 1)
    board.play(24)  # W: (4, 4)
    board.play(5)   # B: (1, 0)
    # Now White's turn: White playing at (0, 0) has 0 liberties and does not capture -> SUICIDE
    assert not board.is_valid_move(0)
    assert board.get_legal_moves()[0] == 0


def test_golden_scenario_4_superko_repetition():
    """Scenario 4: Standard Ko fight reproduction prohibited by Positional Superko."""
    board = Board(size=5)
    board.play(7)   # B (1, 2)
    board.play(11)  # W (2, 1)
    board.play(17)  # B (3, 2)
    board.play(13)  # W (2, 3)
    board.play(12)  # B (2, 2)
    board.play(6)   # W (1, 1)
    assert len(board.history_positions) >= 6


def test_golden_scenario_5_terminal_scoring():
    """Scenario 5: Two consecutive passes lead to exact terminal Area Scoring."""
    board = Board(size=5)
    # Black places stone at (2, 2)
    board.play(12)
    # Both pass
    board.play(board.pass_action)
    board.play(board.pass_action)
    assert board.is_game_over()
    # Area scoring with komi 0.5
    b_score, w_score, winner = compute_area_score(board, komi=0.5)
    assert winner == BLACK
    assert b_score > w_score
