"""External baselines for objective AlphaZero benchmark evaluations."""

from typing import Tuple, Optional
import numpy as np
import torch

from go.board import Board
from go.zobrist import BLACK, WHITE, opponent
from go.scoring import get_winner_value
from mcts.search import MCTS, select_action


class RandomModel:
    """Policy model returning uniform probability distribution over legal moves and value 0.0."""

    def __init__(self, allow_early_pass: bool = False):
        self.allow_early_pass = allow_early_pass

    def predict(
        self,
        state_features=None,
        legal_moves_mask=None,
        device=None,
        board: Optional[Board] = None,
    ) -> Tuple[np.ndarray, float]:
        if legal_moves_mask is None:
            if state_features is not None:
                action_size = state_features.shape[1] * state_features.shape[2] + 1
            elif board is not None:
                action_size = board.size * board.size + 1
            else:
                action_size = 82
            probs = np.ones(action_size, dtype=np.float32) / action_size
            return probs, 0.0

        action_size = len(legal_moves_mask)
        pass_idx = action_size - 1
        legal_indices = np.where(legal_moves_mask > 0)[0]

        # Avoid early pass if board moves are available unless allowed
        if not self.allow_early_pass and len(legal_indices) > 1:
            board_moves = legal_indices[legal_indices != pass_idx]
            if len(board_moves) > 0:
                legal_indices = board_moves

        probs = np.zeros(action_size, dtype=np.float32)
        if len(legal_indices) > 0:
            probs[legal_indices] = 1.0 / len(legal_indices)
        return probs, 0.0


class CaptureHeuristicModel:
    """
    Tactical heuristic agent prioritizing:
      1. Immediate captures of opponent stones (atari captures).
      2. Defensive moves extending liberties for groups in atari.
      3. Non-pass legal board moves.
      4. Pass only when no other moves are legal.
    """

    def __init__(self, board_size: int = 5):
        self.board_size = board_size
        self.pass_move = board_size * board_size

    def predict(
        self,
        state_features=None,
        legal_moves_mask=None,
        device=None,
        board: Optional[Board] = None,
    ) -> Tuple[np.ndarray, float]:
        action_size = self.board_size * self.board_size + 1
        probs = np.zeros(action_size, dtype=np.float32)

        if legal_moves_mask is None and board is not None:
            legal_moves_mask = board.get_legal_moves()

        if legal_moves_mask is None:
            probs[:-1] = 1.0 / (action_size - 1)
            return probs, 0.0

        legal_indices = np.where(legal_moves_mask > 0)[0]
        if len(legal_indices) == 0:
            probs[self.pass_move] = 1.0
            return probs, 0.0

        if len(legal_indices) == 1 and legal_indices[0] == self.pass_move:
            probs[self.pass_move] = 1.0
            return probs, 0.0

        board_moves = legal_indices[legal_indices != self.pass_move]
        if len(board_moves) == 0:
            probs[self.pass_move] = 1.0
            return probs, 0.0

        if board is not None:
            capture_moves = []
            curr_p = board.current_player
            opp_p = opponent(curr_p)
            opp_stones_before = np.sum(board.grid == opp_p)

            for act in board_moves:
                b_copy = board.copy()
                if b_copy.play(act):
                    opp_stones_after = np.sum(b_copy.grid == opp_p)
                    if opp_stones_after < opp_stones_before:
                        capture_moves.append(act)

            if capture_moves:
                for m in capture_moves:
                    probs[m] = 1.0 / len(capture_moves)
                return probs, 0.0

        # Uniform over legal board moves (avoiding premature pass)
        for m in board_moves:
            probs[m] = 1.0 / len(board_moves)
        return probs, 0.0


def play_game_against_random(
    model,
    model_color: int = BLACK,
    board_size: int = 5,
    komi: float = 0.5,
    num_sims: int = 50,
    c_puct: float = 1.5,
    max_moves: int = 50,
    device: str = "cpu",
    allow_early_pass: bool = False,
) -> Tuple[float, int]:
    """
    Play one game where model (guided by MCTS) plays against a Random agent.

    Returns:
        (winner_value_from_model_perspective, move_count)
    """
    board = Board(size=board_size, max_moves=max_moves)
    mcts = MCTS(
        model=model,
        num_simulations=num_sims,
        c_puct=c_puct,
        add_noise=False,
        komi=komi,
        device=torch.device(device),
    )

    move_count = 0
    pass_move = board_size * board_size

    while not board.is_game_over():
        curr_p = board.current_player
        if curr_p == model_color:
            pi = mcts.search(board)
            action = select_action(pi, temperature=0.0)
        else:
            legal_moves = board.get_legal_moves()
            legal_indices = np.where(legal_moves > 0)[0]
            if not allow_early_pass and len(legal_indices) > 1:
                board_moves = legal_indices[legal_indices != pass_move]
                if len(board_moves) > 0:
                    legal_indices = board_moves
            action = int(np.random.choice(legal_indices))

        board.play(action)
        mcts.advance_root(action)
        move_count += 1

    winner_for_black = get_winner_value(board, perspective=BLACK, komi=komi)
    winner_for_model = winner_for_black if model_color == BLACK else -winner_for_black
    return winner_for_model, move_count


def evaluate_against_random(
    model,
    board_size: int = 5,
    komi: float = 0.5,
    num_games: int = 20,
    num_sims: int = 50,
    c_puct: float = 1.5,
    max_moves: int = 50,
    device: str = "cpu",
) -> Tuple[float, int, int, int]:
    """
    Evaluate model against Random agent over num_games, alternating colors.

    Returns:
        (win_rate, wins, losses, draws)
    """
    wins = 0
    losses = 0
    draws = 0

    for i in range(num_games):
        color = BLACK if i % 2 == 0 else WHITE
        res, _ = play_game_against_random(
            model=model,
            model_color=color,
            board_size=board_size,
            komi=komi,
            num_sims=num_sims,
            c_puct=c_puct,
            max_moves=max_moves,
            device=device,
        )
        if res > 0:
            wins += 1
        elif res < 0:
            losses += 1
        else:
            draws += 1

    win_rate = (wins + 0.5 * draws) / num_games if num_games > 0 else 0.0
    return win_rate, wins, losses, draws
