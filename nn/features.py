"""Neural network feature extraction and D4 data augmentation."""

from typing import List, Tuple, Optional
import numpy as np

from go.board import Board
from go.zobrist import EMPTY, BLACK, WHITE, opponent


def extract_features(board: Board) -> np.ndarray:
    """
    Extract 17-plane input tensor for AlphaZero Go:
      - Planes 0-7: Current player's stones across the last 8 moves (T_0 to T_-7)
      - Planes 8-15: Opponent player's stones across the last 8 moves
      - Plane 16: All 1s if current player is BLACK, all 0s if WHITE
      
    Returns:
        np.ndarray of shape (17, N, N) with dtype np.float32.
    """
    size = board.size
    planes = np.zeros((17, size, size), dtype=np.float32)

    curr_p = board.current_player
    opp_p = opponent(curr_p)

    history_len = len(board.history_grids)

    # Populate 8 historical planes for each player
    for t in range(8):
        hist_idx = history_len - 1 - t
        if hist_idx >= 0:
            grid = board.history_grids[hist_idx]
            planes[t, :, :] = (grid == curr_p).astype(np.float32)
            planes[t + 8, :, :] = (grid == opp_p).astype(np.float32)

    # Plane 16: turn indicator (all 1s if Black to play, all 0s if White)
    if curr_p == BLACK:
        planes[16, :, :] = 1.0

    return planes


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


def apply_random_symmetry(
    features: np.ndarray,
    pi: np.ndarray,
    flip: Optional[bool] = None,
    rot: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a single D4 symmetry transformation directly to features and policy vector.
    Bypasses computing all 8 transformations when only one is needed for sampling.
    """
    if flip is None:
        flip = bool(np.random.choice([False, True]))
    if rot is None:
        rot = int(np.random.randint(4))

    _, h, w = features.shape
    board_size = h * w
    pi_board = pi[:board_size].reshape(h, w)
    pi_pass = pi[board_size]

    if flip:
        feat_flip = np.flip(features, axis=2)
        pi_flip = np.fliplr(pi_board)
    else:
        feat_flip = features
        pi_flip = pi_board

    feat_rot = np.rot90(feat_flip, k=rot, axes=(1, 2))
    pi_rot = np.rot90(pi_flip, k=rot)

    pi_sym = np.zeros_like(pi)
    pi_sym[:board_size] = pi_rot.flatten()
    pi_sym[board_size] = pi_pass

    return np.ascontiguousarray(feat_rot), np.ascontiguousarray(pi_sym)
