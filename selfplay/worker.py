"""Self-Play worker generating AlphaZero training games."""

from typing import List, Tuple, Optional
import numpy as np
import torch

from go.board import Board
from go.scoring import compute_area_score
from go.zobrist import opponent, BLACK, WHITE
from nn.features import extract_features
from mcts.search import MCTS, select_action

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
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
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
        
    Returns:
        List of (state_features, pi, z) tuples for every step in the game.
    """
    board = Board(size=board_size, max_moves=max_moves)
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

        # Advance board and MCTS search tree
        board.play(action)
        mcts.advance_root(action)
        step += 1

    # Determine game winner
    if resigned_player is not None:
        winner = opponent(resigned_player)
    else:
        _, _, winner = compute_area_score(board, komi=komi)

    # Format training samples: (s, pi, z_perspective)
    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for s, p_target, player in trajectory:
        z = 1.0 if player == winner else -1.0
        samples.append((s, p_target, z))

    return samples

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
    max_moves: Optional[int] = None,
    komi: float = 7.5,
    device: Optional[torch.device] = None,
    progress_bar: bool = False,
    game_callback=None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Generate multiple self-play games sequentially.
    Disables resignation for a fraction of games (e.g. 10%) to verify against false-positive resigns.
    Supports real-time progress bars via tqdm and per-game callbacks.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    all_samples = []
    iterator = range(num_games)
    pbar = None
    if progress_bar and tqdm is not None:
        pbar = tqdm(iterator, desc="Self-Play", unit="game", dynamic_ncols=True)
        iterator = pbar

    for game_idx in iterator:
        # Disable resignation for a subset of games
        disable_resign = (np.random.random() < resign_disable_frac)

        game_samples = play_selfplay_game(
            model=model,
            board_size=board_size,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            temp_moves=temp_moves,
            resign_threshold=resign_threshold,
            disable_resign=disable_resign,
            max_moves=max_moves,
            komi=komi,
            device=device,
        )
        all_samples.extend(game_samples)

        winner_str = "B" if game_samples and game_samples[-1][2] > 0 else "W"
        if pbar is not None:
            pbar.set_postfix({
                "moves": len(game_samples),
                "winner": winner_str,
                "total_samples": len(all_samples),
            })
        elif progress_bar:
            print(f"   -> [Self-Play {game_idx + 1}/{num_games}] {len(game_samples)} nước | Người thắng: {winner_str} | Tổng mẫu: {len(all_samples)}")

        if game_callback is not None:
            game_callback(game_idx, game_samples)

    return all_samples
