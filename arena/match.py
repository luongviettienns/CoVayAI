"""Head-to-head match playing and model evaluation for AlphaZero Go."""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, overload, Literal
import numpy as np
import torch

from go.board import Board
from go.zobrist import BLACK, WHITE, opponent
from go.scoring import get_winner_value
from mcts.search import MCTS, select_action


@dataclass
class PairedGameResult:
    """Outcomes of a paired game where both models play the identical opening as both Black and White."""

    pair_index: int
    opening_moves: List[int]
    cand_as_black_winner: float  # +1.0 if cand won, -1.0 if baseline won, 0.0 if draw
    cand_as_black_moves: int
    cand_as_white_winner: float  # +1.0 if cand won, -1.0 if baseline won, 0.0 if draw
    cand_as_white_moves: int
    cand_pair_score: float  # points in pair: 1.0 per win, 0.5 per draw (out of 2.0)


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
    paired_results: Optional[List[PairedGameResult]] = None
    win_rate_ci: Optional[Tuple[float, float]] = None


def compute_wilson_score_interval(
    p: float, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate 95% Wilson score confidence interval for a proportion."""
    if n <= 0:
        return 0.0, 1.0
    z = 1.95996  # 95% confidence z-score
    z_sq = z * z
    denom = 1.0 + z_sq / n
    centre = (p + z_sq / (2.0 * n)) / denom
    spread = (z * np.sqrt((p * (1.0 - p) / n) + (z_sq / (4.0 * n * n)))) / denom
    ci_low = max(0.0, float(centre - spread))
    ci_high = min(1.0, float(centre + spread))
    return ci_low, ci_high


def generate_opening_sequence(
    board_size: int,
    num_plies: int,
    seed: Optional[int] = None,
    allow_pass: bool = False,
) -> List[int]:
    """
    Generate a diverse, strictly legal opening sequence of num_plies moves.
    Ensures that opening moves do not include premature pass unless forced.
    """
    if num_plies <= 0:
        return []

    rng = np.random.RandomState(seed) if seed is not None else np.random
    board = Board(size=board_size)
    pass_move = board_size * board_size
    moves: List[int] = []

    for _ in range(num_plies):
        if board.is_game_over():
            break
        legal_mask = board.get_legal_moves()
        legal_indices = np.where(legal_mask > 0)[0]
        if not allow_pass and len(legal_indices) > 1:
            legal_indices = legal_indices[legal_indices != pass_move]
        if len(legal_indices) == 0:
            break
        chosen_action = int(rng.choice(legal_indices))
        if not board.play(chosen_action):
            break
        moves.append(chosen_action)

    return moves


@overload
def play_arena_game(
    black_model,
    white_model,
    board_size: int = 9,
    komi: float = 7.5,
    num_sims: int = 200,
    c_puct: float = 1.5,
    max_moves: int = 162,
    resign_threshold: Optional[float] = None,
    opening_plies: int = 8,
    return_moves: Literal[False] = False,
    device: Optional[torch.device] = None,
    initial_moves: Optional[List[int]] = None,
) -> Tuple[float, int]: ...


@overload
def play_arena_game(
    black_model,
    white_model,
    board_size: int = 9,
    komi: float = 7.5,
    num_sims: int = 200,
    c_puct: float = 1.5,
    max_moves: int = 162,
    resign_threshold: Optional[float] = None,
    opening_plies: int = 8,
    return_moves: Literal[True] = ...,
    device: Optional[torch.device] = None,
    initial_moves: Optional[List[int]] = None,
) -> Tuple[float, int, List[int]]: ...


def play_arena_game(
    black_model,
    white_model,
    board_size: int = 9,
    komi: float = 7.5,
    num_sims: int = 200,
    c_puct: float = 1.5,
    max_moves: int = 162,
    resign_threshold: Optional[float] = None,
    opening_plies: int = 8,
    return_moves: bool = False,
    device: Optional[torch.device] = None,
    initial_moves: Optional[List[int]] = None,
    adjudicate_truncated: bool = True,
) -> Union[Tuple[float, int], Tuple[float, int, List[int]]]:
    """Play a single head-to-head game between black_model and white_model.

    If initial_moves is provided, plays them verbatim at the start, advancing
    both MCTS trees, and then switches to greedy action selection (temp=0.0).

    Returns:
        If return_moves is False:
            winner: +1.0 if Black wins, -1.0 if White wins, 0.0 if draw.
            move_count: Number of moves played in the game.
        If return_moves is True:
            (winner, move_count, move_sequence)
    """
    board = Board(size=board_size, max_moves=max_moves)

    mcts_black = MCTS(
        model=black_model,
        num_simulations=num_sims,
        c_puct=c_puct,
        add_noise=False,
        komi=komi,
        device=device,
    )
    mcts_white = MCTS(
        model=white_model,
        num_simulations=num_sims,
        c_puct=c_puct,
        add_noise=False,
        komi=komi,
        device=device,
    )

    move_count = 0
    resigned_player: Optional[int] = None
    move_sequence: List[int] = []

    # Replay initial opening moves if specified
    if initial_moves:
        for action in initial_moves:
            if board.is_game_over():
                break
            if not board.play(action):
                raise RuntimeError(
                    f"Arena encountered illegal initial move {action} on board:\n{board.grid}"
                )
            move_sequence.append(action)
            move_count += 1
            mcts_black.advance_root(action)
            mcts_white.advance_root(action)

    while not board.is_game_over():
        curr_mcts = mcts_black if board.current_player == BLACK else mcts_white
        pi = curr_mcts.search(board)

        # Check resignation if enabled (only after move 10 to avoid premature quits)
        if resign_threshold is not None and move_count >= 10:
            if curr_mcts.root.value_score < resign_threshold:
                resigned_player = board.current_player
                break

        # Greedy play after initial opening
        if initial_moves is not None:
            temp = 0.0
        else:
            temp = 1.0 if move_count < opening_plies else 0.0

        action = select_action(pi, temperature=temp)

        if not board.play(action):
            raise RuntimeError(
                f"Arena encountered illegal action {action} for player {board.current_player} "
                f"at move {move_count} on board:\n{board.grid}"
            )
        move_sequence.append(action)
        move_count += 1

        # Advance subtree roots for both players
        mcts_black.advance_root(action)
        mcts_white.advance_root(action)

    if resigned_player is not None:
        winner_color = opponent(resigned_player)
        winner_val = +1.0 if winner_color == BLACK else -1.0
    elif board.consecutive_passes >= 2 or adjudicate_truncated:
        winner_val = get_winner_value(board, perspective=BLACK, komi=komi)
    else:
        # Game reached max_moves limit without terminating naturally and adjudication is disabled -> Draw (0.0)
        winner_val = 0.0

    if return_moves:
        return winner_val, move_count, move_sequence
    return winner_val, move_count


class Arena:
    """Evaluates candidate model against baseline model across paired games with identical openings."""

    def __init__(
        self,
        board_size: int = 9,
        komi: float = 7.5,
        num_sims: int = 200,
        c_puct: float = 1.5,
        max_moves: int = 162,
        win_threshold: float = 0.55,
        resign_threshold: Optional[float] = None,
        opening_plies: int = 8,
        device: Optional[torch.device] = None,
    ):
        self.board_size = board_size
        self.komi = komi
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.max_moves = max_moves
        self.win_threshold = win_threshold
        self.resign_threshold = resign_threshold
        self.opening_plies = opening_plies
        self.device = device

    def evaluate(
        self,
        candidate_model,
        baseline_model,
        num_games: int = 50,
        verbose: bool = False,
        progress_bar: bool = False,
        base_seed: Optional[int] = None,
    ) -> ArenaResult:
        """
        Evaluate candidate_model against baseline_model using paired openings.
        For each pair:
          1. Candidate plays Black, Baseline plays White from opening sequence.
          2. Baseline plays Black, Candidate plays White from the identical opening sequence.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        if hasattr(candidate_model, "eval"):
            candidate_model.eval()
        if hasattr(baseline_model, "eval"):
            baseline_model.eval()

        num_pairs = max(1, num_games // 2)
        total_games = num_pairs * 2

        candidate_wins = 0
        baseline_wins = 0
        draws = 0
        cand_as_black_wins = 0
        cand_as_white_wins = 0
        paired_results: List[PairedGameResult] = []

        iterator = range(num_pairs)
        pbar = None
        if progress_bar and tqdm is not None:
            pbar = tqdm(iterator, desc="Arena Eval (Pairs)", unit="pair", dynamic_ncols=True)
            iterator = pbar

        for pair_idx in iterator:
            pair_seed = (base_seed + pair_idx * 1000) if base_seed is not None else None
            opening_moves = generate_opening_sequence(
                board_size=self.board_size,
                num_plies=self.opening_plies,
                seed=pair_seed,
            )

            # Game 1: Candidate as Black, Baseline as White
            w1, m1 = play_arena_game(
                black_model=candidate_model,
                white_model=baseline_model,
                board_size=self.board_size,
                komi=self.komi,
                num_sims=self.num_sims,
                c_puct=self.c_puct,
                max_moves=self.max_moves,
                resign_threshold=self.resign_threshold,
                device=self.device,
                initial_moves=opening_moves,
            )
            cand_b_win = w1
            if w1 > 0:
                candidate_wins += 1
                cand_as_black_wins += 1
                score1 = 1.0
            elif w1 < 0:
                baseline_wins += 1
                score1 = 0.0
            else:
                draws += 1
                score1 = 0.5

            # Game 2: Baseline as Black, Candidate as White
            w2, m2 = play_arena_game(
                black_model=baseline_model,
                white_model=candidate_model,
                board_size=self.board_size,
                komi=self.komi,
                num_sims=self.num_sims,
                c_puct=self.c_puct,
                max_moves=self.max_moves,
                resign_threshold=self.resign_threshold,
                device=self.device,
                initial_moves=opening_moves,
            )
            cand_w_win = -w2
            if w2 < 0:
                candidate_wins += 1
                cand_as_white_wins += 1
                score2 = 1.0
            elif w2 > 0:
                baseline_wins += 1
                score2 = 0.0
            else:
                draws += 1
                score2 = 0.5

            pair_res = PairedGameResult(
                pair_index=pair_idx,
                opening_moves=opening_moves,
                cand_as_black_winner=cand_b_win,
                cand_as_black_moves=m1,
                cand_as_white_winner=cand_w_win,
                cand_as_white_moves=m2,
                cand_pair_score=score1 + score2,
            )
            paired_results.append(pair_res)

            if pbar is not None:
                curr_wr = (candidate_wins + 0.5 * draws) / ((pair_idx + 1) * 2)
                pbar.set_postfix({
                    "Cand_W": candidate_wins,
                    "Base_W": baseline_wins,
                    "Cand_WR": f"{curr_wr*100:.1f}%",
                })
            elif verbose:
                print(
                    f"   -> [Arena Pair {pair_idx + 1}/{num_pairs}] "
                    f"Cand Black: {'Win' if w1 > 0 else ('Loss' if w1 < 0 else 'Draw')}, "
                    f"Cand White: {'Win' if w2 < 0 else ('Loss' if w2 > 0 else 'Draw')}"
                )

        candidate_win_rate = (
            (candidate_wins + 0.5 * draws) / total_games if total_games > 0 else 0.0
        )
        accepted = candidate_win_rate >= self.win_threshold
        win_rate_ci = compute_wilson_score_interval(candidate_win_rate, total_games)

        return ArenaResult(
            candidate_wins=candidate_wins,
            baseline_wins=baseline_wins,
            draws=draws,
            candidate_win_rate=candidate_win_rate,
            games_played=total_games,
            accepted=accepted,
            candidate_as_black_wins=cand_as_black_wins,
            candidate_as_white_wins=cand_as_white_wins,
            paired_results=paired_results,
            win_rate_ci=win_rate_ci,
        )
