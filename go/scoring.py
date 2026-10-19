"""Chinese Area Scoring (Tromp-Taylor rules) for Go."""

from typing import Tuple, Set
import numpy as np
from go.zobrist import EMPTY, BLACK, WHITE
from go.board import Board

def compute_area_score(board: Board, komi: float = 7.5) -> Tuple[float, float, int]:
    """
    Calculate Chinese Area Score (Tromp-Taylor rules):
    Score = (number of stones alive on board) + (empty intersections surrounded solely by one color).
    
    Returns:
        (black_total_score, white_total_score, winner)
        winner is either BLACK (1) or WHITE (2).
    """
    size = board.size
    grid = board.grid

    black_stones = int(np.sum(grid == BLACK))
    white_stones = int(np.sum(grid == WHITE))

    black_territory = 0
    white_territory = 0

    visited_empty: Set[Tuple[int, int]] = set()

    for r in range(size):
        for c in range(size):
            if grid[r, c] == EMPTY and (r, c) not in visited_empty:
                # Find connected empty region
                component = []
                border_colors = set()
                queue = [(r, c)]
                visited_empty.add((r, c))

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    component.append((curr_r, curr_c))

                    for nr, nc in board.get_neighbors(curr_r, curr_c):
                        color = grid[nr, nc]
                        if color == EMPTY:
                            if (nr, nc) not in visited_empty:
                                visited_empty.add((nr, nc))
                                queue.append((nr, nc))
                        else:
                            border_colors.add(color)

                # Attribute territory if strictly surrounded by one color
                if border_colors == {BLACK}:
                    black_territory += len(component)
                elif border_colors == {WHITE}:
                    white_territory += len(component)

    black_total = float(black_stones + black_territory)
    white_total = float(white_stones + white_territory + komi)

    winner = BLACK if black_total > white_total else WHITE
    return black_total, white_total, winner

def get_winner_value(board: Board, perspective: int, komi: float = 7.5) -> float:
    """
    Return +1.0 if player with 'perspective' won, or -1.0 if they lost.
    """
    _, _, winner = compute_area_score(board, komi=komi)
    return 1.0 if winner == perspective else -1.0
