"""Feature extraction and D4 data augmentation for AlphaZero Go."""

from typing import List, Tuple
import numpy as np
from go.zobrist import BLACK, opponent
from go.board import Board

def extract_features(board: Board, num_history: int = 8) -> np.ndarray:
    """
    Extract spatial features tensor of shape (C, N, N) where C = 2 * num_history + 1 (default 17).
    
    Planes 0..7: Friendly stones at times t, t-1, ..., t-7
    Planes 8..15: Opponent stones at times t, t-1, ..., t-7
    Plane 16: Turn indicator (all 1.0 if current_player is BLACK, else all 0.0)
    
    Always presented from the canonical perspective of board.current_player.
    """
    size = board.size
    total_channels = 2 * num_history + 1
    features = np.zeros((total_channels, size, size), dtype=np.float32)

    current_player = board.current_player
    opp = opponent(current_player)
    history = board.history_grids

    # Fill history planes for friendly and opponent stones
    for i in range(num_history):
        if len(history) > i:
            grid_past = history[-1 - i]
            features[i] = (grid_past == current_player).astype(np.float32)
            features[num_history + i] = (grid_past == opp).astype(np.float32)

    # Turn plane (Black = 1.0, White = 0.0)
    if current_player == BLACK:
        features[2 * num_history, :, :] = 1.0
    else:
        features[2 * num_history, :, :] = 0.0

    return features

def get_symmetries(features: np.ndarray, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate the 8 D4 dihedral group symmetries (4 rotations x 2 reflections)
    for both state features and policy probability vector pi.
    
    The pass probability pi[-1] remains completely invariant under all transformations.
    """
    _, h, w = features.shape
    board_size = h * w
    pi_board = pi[:board_size].reshape(h, w)
    pi_pass = pi[board_size]

    symmetries = []

    for flip in [False, True]:
        if flip:
            # Horizontal flip (along axis 2 / columns)
            feat_flip = np.flip(features, axis=2)
            pi_flip = np.fliplr(pi_board)
        else:
            feat_flip = features
            pi_flip = pi_board

        for rot in range(4):
            # Rotate by rot * 90 degrees in the plane (axis 1 and 2)
            feat_rot = np.rot90(feat_flip, k=rot, axes=(1, 2))
            pi_rot = np.rot90(pi_flip, k=rot)

            pi_sym = np.zeros_like(pi)
            pi_sym[:board_size] = pi_rot.flatten()
            pi_sym[board_size] = pi_pass

            symmetries.append((np.ascontiguousarray(feat_rot), np.ascontiguousarray(pi_sym)))

    return symmetries
