"""Deterministic board fixtures for benchmark profiling across opening, mid-game, and late-game states."""

from typing import Dict
import random
import numpy as np
from go.board import Board


PHASE_MOVE_TARGETS: Dict[int, Dict[str, int]] = {
    9: {
        "opening": 6,
        "middle_game": 30,
        "late_game": 60,
    },
    13: {
        "opening": 12,
        "middle_game": 60,
        "late_game": 120,
    },
    19: {
        "opening": 20,
        "middle_game": 90,
        "late_game": 180,
    },
}


def create_deterministic_board(size: int, phase: str, seed: int = 42) -> Board:
    """Generate a valid, reproducible Board state for a given size and phase.
    
    Args:
        size: Board size (9, 13, or 19).
        phase: 'opening', 'middle_game', or 'late_game'.
        seed: Random seed for reproducibility.
        
    Returns:
        A cloned Board instance at the specified game phase.
    """
    if size not in PHASE_MOVE_TARGETS:
        raise ValueError(f"Unsupported board size: {size}. Supported: 9, 13, 19")
    if phase not in PHASE_MOVE_TARGETS[size]:
        raise ValueError(f"Unknown phase: {phase}. Supported: 'opening', 'middle_game', 'late_game'")

    target_moves = PHASE_MOVE_TARGETS[size][phase]
    rng = random.Random(seed + size * 1000 + target_moves)
    board = Board(size=size)

    pass_action = size * size
    consecutive_passes = 0

    for _ in range(target_moves):
        legal_mask = board.get_legal_moves()
        legal_actions = list(np.flatnonzero(legal_mask))
        if not legal_actions:
            break

        # Prefer non-pass moves to keep the board state rich and non-terminal
        non_pass_moves = [m for m in legal_actions if m != pass_action]
        if non_pass_moves and consecutive_passes == 0:
            move = rng.choice(non_pass_moves)
            consecutive_passes = 0
        else:
            move = rng.choice(legal_actions)
            if move == pass_action:
                consecutive_passes += 1
            else:
                consecutive_passes = 0

        if consecutive_passes >= 2:
            break

        board.play(move)

    return board.copy()
