"""External baselines for objective AlphaZero benchmark evaluations."""

from typing import Tuple
import numpy as np
import torch

from go.board import Board
from go.zobrist import BLACK, WHITE, opponent
from go.scoring import get_winner_value
from mcts.search import MCTS, select_action


class RandomModel:
    """Policy model returning uniform probability distribution over legal moves and value 0.0."""

    def predict(self, state_features, legal_moves_mask=None, device=None):
        if legal_moves_mask is None:
            action_size = state_features.shape[1] * state_features.shape[2] + 1
            probs = np.ones(action_size, dtype=np.float32) / action_size
            return probs, 0.0

        action_size = len(legal_moves_mask)
        legal_indices = np.where(legal_moves_mask > 0)[0]
        probs = np.zeros(action_size, dtype=np.float32)
        if len(legal_indices) > 0:
            probs[legal_indices] = 1.0 / len(legal_indices)
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
) -> Tuple[float, int]:
    """
    Play one game where model (guided by MCTS) plays against a pure Random agent.
    
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
    while not board.is_game_over():
        curr_p = board.current_player
        if curr_p == model_color:
            pi = mcts.search(board)
            action = select_action(pi, temperature=0.0)
        else:
            # Random agent selects uniformly from legal moves
            legal_moves = board.get_legal_moves()
            legal_indices = np.where(legal_moves > 0)[0]
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
