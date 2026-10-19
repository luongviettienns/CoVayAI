"""Head-to-head match playing and model evaluation for AlphaZero Go."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch

from go.board import Board
from go.zobrist import BLACK, WHITE, opponent
from go.scoring import get_winner_value
from mcts.search import MCTS


@dataclass
class ArenaResult:
    """Detailed summary of match outcomes between candidate and baseline models."""

    candidate_wins: int
    baseline_wins: int
    draws: int
    candidate_win_rate: float
    games_played: int
    accepted: bool
    candidate_as_black_wins: int = 0
    candidate_as_white_wins: int = 0


def play_arena_game(
    black_model,
    white_model,
    board_size: int = 9,
    komi: float = 7.5,
    num_sims: int = 200,
    c_puct: float = 1.5,
    max_moves: int = 162,
    resign_threshold: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, int]:
    """Play a single head-to-head game between black_model and white_model.

    Uses deterministic MCTS (temperature=0.0, no Dirichlet noise).

    Returns:
        winner: +1.0 if Black wins, -1.0 if White wins, 0.0 if draw.
        move_count: Number of moves played in the game.
    """
    board = Board(size=board_size, max_moves=max_moves)

    mcts_black = MCTS(
        model=black_model,
        num_simulations=num_sims,
        c_puct=c_puct,
        add_noise=False,
        device=device,
    )
    mcts_white = MCTS(
        model=white_model,
        num_simulations=num_sims,
        c_puct=c_puct,
        add_noise=False,
        device=device,
    )

    move_count = 0
    resigned_player: Optional[int] = None

    while not board.is_game_over():
        curr_mcts = mcts_black if board.current_player == BLACK else mcts_white
        pi = curr_mcts.search(board, temperature=0.0)

        # Check resignation if enabled (only after move 10 to avoid premature quits)
        if resign_threshold is not None and move_count >= 10:
            if curr_mcts.root.value_score < resign_threshold:
                resigned_player = board.current_player
                break

        action = int(np.argmax(pi))
        board.play(action)
        move_count += 1

        # Advance subtree roots for both players
        mcts_black.advance_root(action)
        mcts_white.advance_root(action)

    if resigned_player is not None:
        # Resigned player loses, opponent wins
        winner_color = opponent(resigned_player)
        return (+1.0 if winner_color == BLACK else -1.0), move_count

    winner_value = get_winner_value(board, perspective=BLACK, komi=komi)
    return winner_value, move_count


class Arena:
    """Evaluates candidate model against baseline model across multiple games."""

    def __init__(
        self,
        board_size: int = 9,
        komi: float = 7.5,
        num_sims: int = 200,
        c_puct: float = 1.5,
        max_moves: int = 162,
        win_threshold: float = 0.55,
        resign_threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.board_size = board_size
        self.komi = komi
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.max_moves = max_moves
        self.win_threshold = win_threshold
        self.resign_threshold = resign_threshold
        self.device = device

    def evaluate(
        self,
        candidate_model,
        baseline_model,
        num_games: int = 50,
        verbose: bool = False,
        progress_bar: bool = False,
    ) -> ArenaResult:
        """Play num_games alternating colors between candidate and baseline.

        Even game indices: candidate is Black, baseline is White.
        Odd game indices: baseline is Black, candidate is White.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        candidate_wins = 0
        baseline_wins = 0
        draws = 0
        cand_as_black_wins = 0
        cand_as_white_wins = 0

        # Ensure models are in evaluation mode
        if hasattr(candidate_model, "eval"):
            candidate_model.eval()
        if hasattr(baseline_model, "eval"):
            baseline_model.eval()

        iterator = range(num_games)
        pbar = None
        if progress_bar and tqdm is not None:
            pbar = tqdm(iterator, desc="Arena Eval", unit="game", dynamic_ncols=True)
            iterator = pbar

        for game_idx in iterator:
            if game_idx % 2 == 0:
                # Candidate plays Black
                winner, moves = play_arena_game(
                    black_model=candidate_model,
                    white_model=baseline_model,
                    board_size=self.board_size,
                    komi=self.komi,
                    num_sims=self.num_sims,
                    c_puct=self.c_puct,
                    max_moves=self.max_moves,
                    resign_threshold=self.resign_threshold,
                    device=self.device,
                )
                if winner > 0:
                    candidate_wins += 1
                    cand_as_black_wins += 1
                elif winner < 0:
                    baseline_wins += 1
                else:
                    draws += 1
            else:
                # Baseline plays Black, Candidate plays White
                winner, moves = play_arena_game(
                    black_model=baseline_model,
                    white_model=candidate_model,
                    board_size=self.board_size,
                    komi=self.komi,
                    num_sims=self.num_sims,
                    c_puct=self.c_puct,
                    max_moves=self.max_moves,
                    resign_threshold=self.resign_threshold,
                    device=self.device,
                )
                if winner < 0:  # White won -> candidate won
                    candidate_wins += 1
                    cand_as_white_wins += 1
                elif winner > 0:  # Black won -> baseline won
                    baseline_wins += 1
                else:
                    draws += 1

            if pbar is not None:
                curr_wr = (candidate_wins + 0.5 * draws) / (game_idx + 1)
                pbar.set_postfix({
                    "Cand_W": candidate_wins,
                    "Base_W": baseline_wins,
                    "Cand_WR": f"{curr_wr*100:.1f}%",
                })
            elif verbose:
                is_candidate_win = (
                    (winner > 0 and game_idx % 2 == 0)
                    or (winner < 0 and game_idx % 2 != 0)
                )
                print(
                    f"   -> [Arena {game_idx + 1}/{num_games}] "
                    f"Candidate {'B' if game_idx % 2 == 0 else 'W'}, "
                    f"Thắng: {'Candidate' if is_candidate_win else 'Baseline'}, "
                    f"Nước: {moves}"
                )

        candidate_win_rate = (
            (candidate_wins + 0.5 * draws) / num_games if num_games > 0 else 0.0
        )
        accepted = candidate_win_rate >= self.win_threshold

        return ArenaResult(
            candidate_wins=candidate_wins,
            baseline_wins=baseline_wins,
            draws=draws,
            candidate_win_rate=candidate_win_rate,
            games_played=num_games,
            accepted=accepted,
            candidate_as_black_wins=cand_as_black_wins,
            candidate_as_white_wins=cand_as_white_wins,
        )
