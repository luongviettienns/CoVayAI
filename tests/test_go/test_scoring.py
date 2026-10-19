import pytest
import numpy as np
from go.zobrist import BLACK, WHITE, EMPTY
from go.board import Board
from go.scoring import compute_area_score, get_winner_value

def test_empty_board_scoring():
    b = Board(size=9)
    # Empty board: 0 stones, 0 territory for both
    # White gets komi 7.5
    b_score, w_score, winner = compute_area_score(b, komi=7.5)
    assert b_score == 0.0
    assert w_score == 7.5
    assert winner == WHITE
    assert get_winner_value(b, perspective=BLACK, komi=7.5) == -1.0
    assert get_winner_value(b, perspective=WHITE, komi=7.5) == 1.0

def test_simple_territory_scoring():
    # Divide 9x9 board into Black (top half) and White (bottom half)
    b = Board(size=9)
    # Row 3 is all Black stones
    for c in range(9):
        b.grid[3, c] = BLACK
    # Row 5 is all White stones
    for c in range(9):
        b.grid[5, c] = WHITE
    
    # Rows 0, 1, 2 (3 rows * 9 = 27 empty points) reach only Black stones -> Black territory
    # Row 3 has 9 Black stones -> Black total = 27 + 9 = 36 points.
    
    # Rows 6, 7, 8 (3 rows * 9 = 27 empty points) reach only White stones -> White territory
    # Row 5 has 9 White stones -> White total stones + territory = 36 points.
    # Plus White komi 7.5 -> White total = 43.5 points.
    
    # Row 4 is neutral (touches both Black and White) -> 0 points.
    
    b_score, w_score, winner = compute_area_score(b, komi=7.5)
    assert b_score == 36.0
    assert w_score == 43.5
    assert winner == WHITE

def test_black_winning_board():
    b = Board(size=9)
    # Fill row 4 with Black stones.
    # Rows 0..3 reach only Black.
    # Total Black area = 9 * 5 = 45 points.
    for c in range(9):
        b.grid[4, c] = BLACK
    
    # White only has 1 stone at (8, 8)
    b.grid[8, 8] = WHITE
    # White area: 1 stone + 0 territory (touches nothing isolated or reaches only White)
    b_score, w_score, winner = compute_area_score(b, komi=7.5)
    assert b_score > w_score
    assert winner == BLACK
    assert get_winner_value(b, perspective=BLACK, komi=7.5) == 1.0
    assert get_winner_value(b, perspective=WHITE, komi=7.5) == -1.0
