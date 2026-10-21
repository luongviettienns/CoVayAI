"""Chinese Area Scoring (Tromp-Taylor rules) for Go."""

from typing import Tuple, Set
import numpy as np
from go.zobrist import EMPTY, BLACK, WHITE
from go.board import Board

def compute_ownership_grid(board: Board) -> np.ndarray:
    """
    Calculate intersection-level Tromp-Taylor ownership grid:
      +1.0 for Black stones and Black territory (empty intersections strictly surrounded by Black)
      -1.0 for White stones and White territory (empty intersections strictly surrounded by White)
       0.0 for neutral / dame intersections (unclaimed empty spaces, seki dame)

    Returns:
        np.ndarray of shape (board.size, board.size) and dtype np.float32.
    """
    size = board.size
    grid = board.grid
    ownership = np.zeros((size, size), dtype=np.float32)

    ownership[grid == BLACK] = 1.0
    ownership[grid == WHITE] = -1.0

    visited_empty: Set[Tuple[int, int]] = set()

    for r in range(size):
        for c in range(size):
            if grid[r, c] == EMPTY and (r, c) not in visited_empty:
                component = []
                border_colors = set()
                stack = [(r, c)]
                visited_empty.add((r, c))

                while stack:
                    curr_r, curr_c = stack.pop()
                    component.append((curr_r, curr_c))

                    for nr, nc in board.get_neighbors(curr_r, curr_c):
                        color = grid[nr, nc]
                        if color == EMPTY:
                            if (nr, nc) not in visited_empty:
                                visited_empty.add((nr, nc))
                                stack.append((nr, nc))
                        else:
                            border_colors.add(color)

                if border_colors == {BLACK}:
                    for cr, cc in component:
                        ownership[cr, cc] = 1.0
                elif border_colors == {WHITE}:
                    for cr, cc in component:
                        ownership[cr, cc] = -1.0

    return ownership


def compute_area_score(board: Board, komi: float = 7.5) -> Tuple[float, float, int]:
    """
    Calculate Chinese Area Score (Tromp-Taylor rules):
    Score = (number of stones alive on board) + (empty intersections surrounded solely by one color).
    
    Returns:
        (black_total_score, white_total_score, winner)
        winner is either BLACK (1) or WHITE (2).
    """
    ownership = compute_ownership_grid(board)
    black_total = float(np.sum(ownership == 1.0))
    white_total = float(np.sum(ownership == -1.0) + komi)
    winner = BLACK if black_total > white_total else WHITE
    return black_total, white_total, winner


def get_winner_value(board: Board, perspective: int, komi: float = 7.5) -> float:
    """
    Return +1.0 if player with 'perspective' won, or -1.0 if they lost.
    """
    _, _, winner = compute_area_score(board, komi=komi)
    return 1.0 if winner == perspective else -1.0
