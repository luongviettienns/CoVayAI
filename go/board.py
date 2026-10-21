"""Go Board implementation with Chinese Rules (Area Scoring & Positional Superko)."""

from typing import List, Tuple, Set, Optional
import numpy as np
from go.zobrist import ZobristHash, EMPTY, BLACK, WHITE, opponent

class Board:
    """Represents a Go board supporting Chinese rules with Positional Superko."""

    def __init__(self, size: int = 9, max_moves: Optional[int] = None, seed: int = 42):
        self.size = size
        self.pass_action = size * size
        self.max_moves = max_moves if max_moves is not None else size * size * 2
        
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.current_player = BLACK
        self.consecutive_passes = 0
        self.move_count = 0
        
        self.zobrist = ZobristHash(board_size=size, seed=seed)
        self.current_position_hash = 0
        self.history_positions: Set[int] = {self.current_position_hash}
        self.history_grids: List[np.ndarray] = [self.grid.copy()]
        
        # Precompute neighbors for every coordinate (r, c)
        self._neighbors: List[List[List[Tuple[int, int]]]] = [
            [self._compute_neighbors(r, c) for c in range(size)] for r in range(size)
        ]

    def _compute_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append((nr, nc))
        return neighbors

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return the precomputed 4-way orthogonal neighbors of (r, c)."""
        return self._neighbors[r][c]

    def _find_chain_and_liberties(
        self, grid: np.ndarray, start_r: int, start_c: int
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Find the connected stone chain and its liberties using BFS."""
        color = grid[start_r, start_c]
        if color == EMPTY:
            return set(), set()

        chain = set()
        liberties = set()
        queue = [(start_r, start_c)]
        visited = {(start_r, start_c)}

        while queue:
            r, c = queue.pop(0)
            chain.add((r, c))

            for nr, nc in self.get_neighbors(r, c):
                n_color = grid[nr, nc]
                if n_color == EMPTY:
                    liberties.add((nr, nc))
                elif n_color == color and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return chain, liberties

    def _simulate_move(
        self, action: int, player: int
    ) -> Tuple[bool, Optional[np.ndarray], int, int]:
        """
        Simulate a move and check all validity criteria:
        1. Within board bounds and intersection is empty
        2. Capture any 0-liberty enemy chains
        3. Suicide check: friendly chain must have >0 liberties
        4. Positional Superko: resulting board position must not repeat any historical state.
        
        Returns:
            (is_valid, new_grid, captured_count, new_position_hash)
        """
        if action == self.pass_action:
            return True, self.grid.copy(), 0, self.current_position_hash

        r, c = divmod(action, self.size)
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False, None, 0, 0

        if self.grid[r, c] != EMPTY:
            return False, None, 0, 0

        new_grid = self.grid.copy()
        new_grid[r, c] = player
        new_pos_hash = self.zobrist.toggle_stone(self.current_position_hash, r, c, player)

        opp = opponent(player)
        captured_count = 0
        visited_opp_chains = set()

        # Check adjacent enemy chains to see if any are captured
        for nr, nc in self.get_neighbors(r, c):
            if new_grid[nr, nc] == opp and (nr, nc) not in visited_opp_chains:
                opp_chain, opp_liberties = self._find_chain_and_liberties(new_grid, nr, nc)
                visited_opp_chains.update(opp_chain)
                if len(opp_liberties) == 0:
                    # Capture this chain
                    for cr, cc in opp_chain:
                        new_grid[cr, cc] = EMPTY
                        new_pos_hash = self.zobrist.toggle_stone(new_pos_hash, cr, cc, opp)
                    captured_count += len(opp_chain)

        # Check friendly chain liberties (Suicide rule check)
        _, friendly_liberties = self._find_chain_and_liberties(new_grid, r, c)
        if len(friendly_liberties) == 0:
            # Suicide is strictly illegal in standard Chinese rules
            return False, None, 0, 0

        # Positional Superko (PSK) check:
        # A move is illegal if the resulting stone configuration has already appeared in the game.
        if new_pos_hash in self.history_positions:
            return False, None, 0, 0

        return True, new_grid, captured_count, new_pos_hash

    def is_valid_move(self, action: int, player: Optional[int] = None) -> bool:
        """Check if an action is legal for the given player (default: current player)."""
        p = player if player is not None else self.current_player
        if action == self.pass_action:
            return True

        r, c = divmod(action, self.size)
        if not (0 <= r < self.size and 0 <= c < self.size) or self.grid[r, c] != EMPTY:
            return False

        opp = opponent(p)
        has_empty = False
        has_opp = False

        for nr, nc in self.get_neighbors(r, c):
            color = self.grid[nr, nc]
            if color == EMPTY:
                has_empty = True
            elif color == opp:
                has_opp = True

        # Fast path: Empty point with an empty neighbor and no opponent neighbors
        # Cannot be suicide (has empty liberty), cannot capture (no opp neighbors),
        # only check positional superko via single XOR hash.
        if has_empty and not has_opp:
            new_hash = self.zobrist.toggle_stone(self.current_position_hash, r, c, p)
            if new_hash not in self.history_positions:
                return True

        # Fallback to full move simulation (BFS chain/liberties & superko)
        is_valid, _, _, _ = self._simulate_move(action, p)
        return is_valid

    def play(self, action: int) -> bool:
        """
        Execute an action on the board.
        Returns True if successful, False if illegal move.
        """
        is_valid, new_grid, captured_count, new_pos_hash = self._simulate_move(
            action, self.current_player
        )
        if not is_valid:
            return False

        if action == self.pass_action:
            self.consecutive_passes += 1
            self.history_grids.append(self.grid.copy())
        else:
            self.consecutive_passes = 0
            self.grid = new_grid
            self.current_position_hash = new_pos_hash
            self.history_positions.add(new_pos_hash)
            self.history_grids.append(self.grid.copy())

        self.current_player = opponent(self.current_player)
        self.move_count += 1
        return True

    def get_legal_moves(self) -> np.ndarray:
        """
        Return a binary 1D array of shape (size * size + 1,)
        where 1 indicates a legal move, 0 indicates an illegal move.
        """
        legal = np.zeros(self.size * self.size + 1, dtype=np.int8)
        # Pass is always legal
        legal[self.pass_action] = 1

        for action in range(self.size * self.size):
            if self.is_valid_move(action, self.current_player):
                legal[action] = 1

        return legal

    def is_terminal(self) -> bool:
        """Return True only if the game ended naturally via 2 consecutive passes."""
        return self.consecutive_passes >= 2

    def is_truncated(self) -> bool:
        """Return True if the game reached max_moves limit without terminating naturally."""
        return self.consecutive_passes < 2 and self.move_count >= self.max_moves

    def is_game_over(self) -> bool:
        """Check if the game has ended (either natural 2 passes or truncated by max moves)."""
        return self.consecutive_passes >= 2 or self.move_count >= self.max_moves

    def copy(self) -> "Board":
        """Create a fast deep copy of the board for MCTS simulation bypassing __init__."""
        b = Board.__new__(Board)
        b.size = self.size
        b.pass_action = self.pass_action
        b.max_moves = self.max_moves
        b.grid = self.grid.copy()
        b.current_player = self.current_player
        b.consecutive_passes = self.consecutive_passes
        b.move_count = self.move_count
        b.zobrist = self.zobrist
        b._neighbors = self._neighbors
        b.current_position_hash = self.current_position_hash
        b.history_positions = set(self.history_positions)
        b.history_grids = [g.copy() for g in self.history_grids[-8:]]
        return b
