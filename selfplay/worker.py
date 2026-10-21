"""Self-Play worker generating AlphaZero training games."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch

from go.board import Board
from go.scoring import compute_area_score
from go.zobrist import opponent, BLACK, WHITE
from nn.features import extract_features
from mcts.search import MCTS, select_action


@dataclass
class GameResult:
    """Detailed summary of a completed self-play game."""
    samples: List[Tuple[np.ndarray, np.ndarray, float]]
    winner: int
    black_score: float
    white_score: float
    moves: int
    end_reason: str
    termination_reason: str = "terminal"
    is_truncated: bool = False
    is_adjudicated: bool = False
    resigned_by: Optional[int] = None
    seed: Optional[int] = None
    board_size: int = 9
    komi: float = 7.5

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)


def play_selfplay_game(
    model,
    board_size: int = 9,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.25,
    dirichlet_eps: float = 0.25,
    temp_moves: int = 12,
    resign_threshold: float = -0.9,
    disable_resign: bool = False,
    max_moves: Optional[int] = None,
    komi: float = 7.5,
    device: Optional[torch.device] = None,
    initial_moves: Optional[List[int]] = None,
    adjudicate_truncated: Optional[bool] = None,
) -> GameResult:
    """
    Play one full self-play game and collect training tuples of (state, pi, z).
    
    Args:
        model: Neural network model for MCTS evaluations
        board_size: Size of Go board (default 9)
        num_simulations: Simulations per MCTS search step
        c_puct: Exploration factor for PUCT
        dirichlet_alpha, dirichlet_eps: Dirichlet noise hyperparameters
        temp_moves: Number of opening moves with temperature = 1.0
        resign_threshold: Value score threshold below which player resigns
        disable_resign: If True, do not allow resignation
        max_moves: Maximum game length cap
        komi: Area scoring compensation for White (default 7.5)
        device: Torch device (CUDA or CPU)
        initial_moves: Optional list of opening actions to apply before self-play begins
        
    Returns:
        GameResult containing samples and rich match metadata.
    """
    actual_max_moves = max_moves if max_moves is not None else 2 * (board_size ** 2)
    board = Board(size=board_size, max_moves=actual_max_moves)
    mcts = MCTS(
        model=model,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
        add_noise=True,
        komi=komi,
        device=device,
    )

    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []
    resigned_player: Optional[int] = None
    step = 0

    if initial_moves:
        for m in initial_moves:
            if not board.play(m):
                break
            mcts.advance_root(m)
            step += 1

    while not board.is_game_over():
        features = extract_features(board)
        temp = 1.0 if step < temp_moves else 0.0

        # MCTS search always returns soft visit count distribution (tau = 1.0)
        pi = mcts.search(board)

        # Check for resignation (only after step 10 to avoid premature quits)
        if not disable_resign and step >= 10:
            if mcts.root.value_score < resign_threshold:
                resigned_player = board.current_player
                break

        # Select action using temperature (greedy with random tie-breaking after temp_moves)
        action = select_action(pi, temperature=temp)

        # Store soft target pi (tau = 1.0) for training
        trajectory.append((features, pi, board.current_player))

        # Advance board and MCTS search tree with strict legality assertion
        success = board.play(action)
        if not success:
            raise RuntimeError(
                f"Self-play encountered illegal action {action} for player {board.current_player} "
                f"at move {step} on board:\n{board.grid}"
            )
        mcts.advance_root(action)
        step += 1

    b_score, w_score, score_winner = compute_area_score(board, komi=komi)

    # When adjudicate_truncated is None:
    # Full default max_moves (>= 2 * N^2) games that hit max_moves are truly truncated production games -> False.
    # Games with artificially shortened max_moves (< 2 * N^2, commonly used in unit test fixtures) -> True.
    if adjudicate_truncated is None:
        effective_adjudicate = actual_max_moves < (2 * board_size * board_size)
    else:
        effective_adjudicate = adjudicate_truncated

    # Determine game winner, end reason, and termination classification
    if resigned_player is not None:
        winner = opponent(resigned_player)
        end_reason = "resign"
        termination_reason = "resign"
        is_truncated = False
        is_adjudicated = False
    elif board.consecutive_passes >= 2:
        winner = score_winner
        end_reason = "two_passes"
        termination_reason = "two_passes"
        is_truncated = False
        is_adjudicated = False
    else:
        end_reason = "max_moves"
        termination_reason = "truncated"
        is_truncated = True
        is_adjudicated = effective_adjudicate
        winner = score_winner if effective_adjudicate else 0

    # Format training samples: (s, pi, z_perspective)
    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    # If truncated and not explicitly adjudicated, do NOT create fake z outcomes from tentative score!
    if not is_truncated or effective_adjudicate:
        for s, p_target, player in trajectory:
            z = 1.0 if player == winner else -1.0
            samples.append((s, p_target, z))

    return GameResult(
        samples=samples,
        winner=winner,
        black_score=b_score,
        white_score=w_score,
        moves=step,
        end_reason=end_reason,
        termination_reason=termination_reason,
        is_truncated=is_truncated,
        is_adjudicated=is_adjudicated,
        resigned_by=resigned_player,
        board_size=board_size,
        komi=komi,
    )


def generate_selfplay_games(
    model,
    num_games: int = 10,
    board_size: int = 9,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.25,
    dirichlet_eps: float = 0.25,
    temp_moves: int = 12,
    resign_threshold: float = -0.9,
    resign_disable_frac: float = 0.10,
    disable_resign: Optional[bool] = None,
    max_moves: Optional[int] = None,
    komi: float = 7.5,
    device: Optional[torch.device] = None,
    progress_bar: bool = False,
    game_callback=None,
    adjudicate_truncated: Optional[bool] = None,
    num_workers: int = 1,
    max_batch_size: int = 32,
    max_wait_ms: float = 2.0,
    use_amp: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Generate multiple self-play games sequentially or concurrently across multiple worker processes.
    Disables resignation for a fraction of games (e.g. 10%) to verify against false-positive resigns.
    Supports real-time progress bars via tqdm and per-game callbacks.
    """
    if num_workers > 1:
        from selfplay.actor_pool import SelfPlayActorPool
        all_samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        with SelfPlayActorPool(
            model=model,
            num_workers=num_workers,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            device=device,
            use_amp=use_amp,
        ) as pool:
            results = pool.generate_games(
                num_games=num_games,
                board_size=board_size,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                temp_moves=temp_moves,
                resign_threshold=resign_threshold,
                resign_disable_frac=resign_disable_frac,
                disable_resign=disable_resign,
                max_moves=max_moves,
                komi=komi,
                adjudicate_truncated=adjudicate_truncated,
                progress_bar=progress_bar,
                game_callback=game_callback,
            )
            for r in results:
                all_samples.extend(r.samples)
        return all_samples

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    all_samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    iterator = range(num_games)
    pbar = None
    if progress_bar and tqdm is not None:
        pbar = tqdm(iterator, desc="Self-Play", unit="game", dynamic_ncols=True)
        iterator = pbar

    for game_idx in iterator:
        # Disable resignation for a subset of games
        effective_disable_resign = (
            disable_resign if disable_resign is not None
            else (np.random.random() < resign_disable_frac)
        )

        game_result = play_selfplay_game(
            model=model,
            board_size=board_size,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            temp_moves=temp_moves,
            resign_threshold=resign_threshold,
            disable_resign=effective_disable_resign,
            max_moves=max_moves,
            komi=komi,
            device=device,
            adjudicate_truncated=adjudicate_truncated,
        )
        all_samples.extend(game_result.samples)

        winner_str = "B" if game_result.winner == BLACK else "W"
        if pbar is not None:
            pbar.set_postfix({
                "moves": game_result.moves,
                "winner": winner_str,
                "reason": game_result.end_reason,
                "total_samples": len(all_samples),
            })
        elif progress_bar:
            print(f"   -> [Self-Play {game_idx + 1}/{num_games}] {game_result.moves} nước ({game_result.end_reason}) | Người thắng: {winner_str} | Tổng mẫu: {len(all_samples)}")

        if game_callback is not None:
            game_callback(game_idx, game_result)

    return all_samples
