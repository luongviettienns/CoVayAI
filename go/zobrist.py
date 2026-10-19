"""Zobrist Hashing implementation for Go board states and positional superko."""

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2

def opponent(player: int) -> int:
    """Return the opposing player."""
    return WHITE if player == BLACK else BLACK

class ZobristHash:
    """Manages 64-bit Zobrist keys for fast incremental hashing of Go board states."""

    def __init__(self, board_size: int = 9, seed: int = 42):
        self.board_size = board_size
        rng = np.random.default_rng(seed)
        
        # Table of random 64-bit integers: shape (board_size, board_size, 3)
        # index 0 is unused (EMPTY), 1 is BLACK, 2 is WHITE
        self.table = rng.integers(
            1, np.iinfo(np.uint64).max, size=(board_size, board_size, 3), dtype=np.uint64
        )
        # Ensure index 0 (EMPTY) is 0
        self.table[:, :, 0] = 0
        
        # Key for turn indicator (e.g., XORed when it's WHITE's turn)
        self.turn_key = int(rng.integers(1, np.iinfo(np.uint64).max, dtype=np.uint64))

    def toggle_stone(self, current_hash: int, row: int, col: int, color: int) -> int:
        """XOR the stone's zobrist key at (row, col) with current_hash."""
        if color == EMPTY:
            return current_hash
        return current_hash ^ int(self.table[row, col, color])

    def toggle_turn(self, current_hash: int) -> int:
        """Toggle player turn in the hash."""
        return current_hash ^ self.turn_key

    def compute_position_hash(self, board: np.ndarray) -> int:
        """Compute positional hash (stones only, ignoring player turn) for Positional Superko."""
        h = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                color = int(board[r, c])
                if color != EMPTY:
                    h ^= int(self.table[r, c, color])
        return h

    def compute_full_hash(self, board: np.ndarray, current_player: int = BLACK) -> int:
        """Compute full hash (stones + player turn)."""
        h = self.compute_position_hash(board)
        if current_player == WHITE:
            h ^= self.turn_key
        return h
