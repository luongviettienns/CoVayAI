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
        """Find the connected stone chain and its liberties using DFS stack."""
        color = grid[start_r, start_c]
        if color == EMPTY:
            return set(), set()

        chain = set()
        liberties = set()
        stack = [(start_r, start_c)]
        visited = {(start_r, start_c)}

        while stack:
            r, c = stack.pop()
            chain.add((r, c))

            for nr, nc in self.get_neighbors(r, c):
                n_color = grid[nr, nc]
                if n_color == EMPTY:
                    liberties.add((nr, nc))
                elif n_color == color and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

        return chain, liberties

    def _get_chain_if_captured(
        self, grid: np.ndarray, start_r: int, start_c: int, opp_color: int, new_stone: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Check if the opponent chain at (start_r, start_c) has any liberty other than new_stone.
        If it has another liberty, early exits immediately and returns None (not captured).
        If it has NO other liberties, returns the full chain of stones to be captured.
        """
        stack = [(start_r, start_c)]
        visited = {(start_r, start_c)}
        chain = [(start_r, start_c)]

        while stack:
            r, c = stack.pop()
            for nr, nc in self.get_neighbors(r, c):
                if (nr, nc) == new_stone:
                    continue
                n_color = grid[nr, nc]
                if n_color == EMPTY:
                    return None  # Has another liberty: cannot be captured!
                elif n_color == opp_color and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    chain.append((nr, nc))
                    stack.append((nr, nc))

        return chain

    def _has_any_liberty(
        self, grid: np.ndarray, start_r: int, start_c: int, player: int
    ) -> bool:
        """
        Check if the friendly chain starting at (start_r, start_c) has at least 1 liberty.
        Early exits immediately upon encountering the first empty intersection.
        """
        stack = [(start_r, start_c)]
        visited = {(start_r, start_c)}

        while stack:
            r, c = stack.pop()
            for nr, nc in self.get_neighbors(r, c):
                n_color = grid[nr, nc]
                if n_color == EMPTY:
                    return True  # Found 1 liberty: alive!
                elif n_color == player and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

        return False  # Zero liberties: suicide

    def _simulate_move(
        self, action: int, player: int
    ) -> Tuple[bool, Optional[np.ndarray], int, int]:
        """
        Simulate a move and check all validity criteria without gratuitous allocations:
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

        opp = opponent(player)
        neighbors = self.get_neighbors(r, c)

        has_empty_neighbor = False
        has_opp_neighbor = False

        for nr, nc in neighbors:
            col = self.grid[nr, nc]
            if col == EMPTY:
                has_empty_neighbor = True
            elif col == opp:
                has_opp_neighbor = True

        captured_stones: List[Tuple[int, int]] = []
        visited_opp: Set[Tuple[int, int]] = set()

        if has_opp_neighbor:
            for nr, nc in neighbors:
                if self.grid[nr, nc] == opp and (nr, nc) not in visited_opp:
                    ch = self._get_chain_if_captured(self.grid, nr, nc, opp, new_stone=(r, c))
                    if ch is not None:
                        visited_opp.update(ch)
                        captured_stones.extend(ch)

        if not captured_stones and not has_empty_neighbor:
            self.grid[r, c] = player
            alive = self._has_any_liberty(self.grid, r, c, player)
            self.grid[r, c] = EMPTY
            if not alive:
                return False, None, 0, 0

        new_hash = self.zobrist.toggle_stone(self.current_position_hash, r, c, player)
        for cr, cc in captured_stones:
            new_hash = self.zobrist.toggle_stone(new_hash, cr, cc, opp)

        if new_hash in self.history_positions:
            return False, None, 0, 0

        new_grid = self.grid.copy()
        new_grid[r, c] = player
        for cr, cc in captured_stones:
            new_grid[cr, cc] = EMPTY

        return True, new_grid, len(captured_stones), new_hash

    def is_valid_move(self, action: int, player: Optional[int] = None) -> bool:
        """Fast legality check without allocating or copying board grid."""
        p = player if player is not None else self.current_player
        if action == self.pass_action:
            return True

        r, c = divmod(action, self.size)
        if not (0 <= r < self.size and 0 <= c < self.size) or self.grid[r, c] != EMPTY:
            return False

        opp = opponent(p)
        neighbors = self.get_neighbors(r, c)

        has_empty_neighbor = False
        has_opp_neighbor = False

        for nr, nc in neighbors:
            col = self.grid[nr, nc]
            if col == EMPTY:
                has_empty_neighbor = True
            elif col == opp:
                has_opp_neighbor = True

        # Fast path: Empty point with an empty neighbor and no opponent neighbor
        if has_empty_neighbor and not has_opp_neighbor:
            new_hash = self.zobrist.toggle_stone(self.current_position_hash, r, c, p)
            return new_hash not in self.history_positions

        # Check opponent captures
        captured_stones: List[Tuple[int, int]] = []
        visited_opp: Set[Tuple[int, int]] = set()

        if has_opp_neighbor:
            for nr, nc in neighbors:
                if self.grid[nr, nc] == opp and (nr, nc) not in visited_opp:
                    ch = self._get_chain_if_captured(self.grid, nr, nc, opp, new_stone=(r, c))
                    if ch is not None:
                        visited_opp.update(ch)
                        captured_stones.extend(ch)

        # Check suicide
        if not captured_stones and not has_empty_neighbor:
            self.grid[r, c] = p
            alive = self._has_any_liberty(self.grid, r, c, p)
            self.grid[r, c] = EMPTY
            if not alive:
                return False

        # Positional Superko check
        new_hash = self.zobrist.toggle_stone(self.current_position_hash, r, c, p)
        for cr, cc in captured_stones:
            new_hash = self.zobrist.toggle_stone(new_hash, cr, cc, opp)

        return new_hash not in self.history_positions

    def play(self, action: int) -> bool:
        """
        Execute an action on the board in-place.
        Returns True if successful, False if illegal move.
        """
        if action == self.pass_action:
            self.consecutive_passes += 1
            self.history_grids.append(self.grid.copy())
            self.current_player = opponent(self.current_player)
            self.move_count += 1
            return True

        r, c = divmod(action, self.size)
        if not (0 <= r < self.size and 0 <= c < self.size) or self.grid[r, c] != EMPTY:
            return False

        p = self.current_player
        opp = opponent(p)
        neighbors = self.get_neighbors(r, c)

        has_empty_neighbor = False
        has_opp_neighbor = False

        for nr, nc in neighbors:
            col = self.grid[nr, nc]
            if col == EMPTY:
                has_empty_neighbor = True
            elif col == opp:
                has_opp_neighbor = True

        captured_stones: List[Tuple[int, int]] = []
        visited_opp: Set[Tuple[int, int]] = set()

        if has_opp_neighbor:
            for nr, nc in neighbors:
                if self.grid[nr, nc] == opp and (nr, nc) not in visited_opp:
                    ch = self._get_chain_if_captured(self.grid, nr, nc, opp, new_stone=(r, c))
                    if ch is not None:
                        visited_opp.update(ch)
                        captured_stones.extend(ch)

        if not captured_stones and not has_empty_neighbor:
            self.grid[r, c] = p
            alive = self._has_any_liberty(self.grid, r, c, p)
            self.grid[r, c] = EMPTY
            if not alive:
                return False

        new_hash = self.zobrist.toggle_stone(self.current_position_hash, r, c, p)
        for cr, cc in captured_stones:
            new_hash = self.zobrist.toggle_stone(new_hash, cr, cc, opp)

        if new_hash in self.history_positions:
            return False

        # Apply move in-place
        self.grid[r, c] = p
        for cr, cc in captured_stones:
            self.grid[cr, cc] = EMPTY

        self.consecutive_passes = 0
        self.current_position_hash = new_hash
        self.history_positions.add(new_hash)
        self.history_grids.append(self.grid.copy())
        self.current_player = opp
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
