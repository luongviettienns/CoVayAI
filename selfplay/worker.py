"""Self-Play worker generating AlphaZero training games."""

from typing import List, Tuple, Optional
import numpy as np
import torch

from go.board import Board
from go.scoring import compute_area_score
from go.zobrist import opponent, BLACK, WHITE
from nn.features import extract_features
from mcts.search import MCTS

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
        device=device,
    )

    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []
    resigned_player: Optional[int] = None
    step = 0

    while not board.is_game_over():
        features = extract_features(board)
        temp = 1.0 if step < temp_moves else 0.0

        pi = mcts.search(board, temperature=temp)

        # Check for resignation (only after step 10 to avoid premature quits)
        if not disable_resign and step >= 10:
            # mcts.root.value_score represents expected outcome for board.current_player
            if mcts.root.value_score < resign_threshold:
                resigned_player = board.current_player
                break

        # Select action
        if temp > 1e-2:
            action = int(np.random.choice(len(pi), p=pi))
        else:
            action = int(np.argmax(pi))

        trajectory.append((features, pi, board.current_player))

        # Advance board and MCTS search tree
        board.play(action)
        mcts.advance_root(action)
        step += 1

    # Determine game winner
    if resigned_player is not None:
        winner = opponent(resigned_player)
    else:
        _, _, winner = compute_area_score(board, komi=7.5)

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
    device: Optional[torch.device] = None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Generate multiple self-play games sequentially.
    Disables resignation for a fraction of games (e.g. 10%) to verify against false-positive resigns.
    """
    all_samples = []

    for game_idx in range(num_games):
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
            device=device,
        )
        all_samples.extend(game_samples)

    return all_samples
