import pytest
import numpy as np
from go.zobrist import BLACK, WHITE, EMPTY
from go.board import Board
from go.scoring import compute_area_score, get_winner_value

def test_cannot_play_on_occupied_spot():
    b = Board(size=9)
    assert b.play(4 * 9 + 4) # Black at (4,4)
    assert not b.play(4 * 9 + 4) # White tries to play on same spot -> illegal
    assert not b.is_valid_move(4 * 9 + 4, player=WHITE)

def test_out_of_bounds_move():
    b = Board(size=9)
    assert not b.is_valid_move(-1)
    assert not b.is_valid_move(82) # 9x9 board: actions are 0..80, 81 is pass
    assert not b.is_valid_move(100)

def test_multi_stone_chain_capture():
    """Capture a 3-stone snake."""
    b = Board(size=9)
    # White places 3 stones at (1,1), (1,2), (1,3)
    b.play(b.pass_action)
    b.play(1 * 9 + 1) # W
    b.play(b.pass_action)
    b.play(1 * 9 + 2) # W
    b.play(b.pass_action)
    b.play(1 * 9 + 3) # W
    
    # Black surrounds them:
    # Top: (0,1), (0,2), (0,3)
    # Bottom: (2,1), (2,2), (2,3)
    # Left: (1,0)
    # Right: (1,4)
    surround_coords = [
        (0, 1), (0, 2), (0, 3),
        (2, 1), (2, 2), (2, 3),
        (1, 0)
    ]
    for r, c in surround_coords:
        b.play(r * 9 + c)
        b.play(b.pass_action)
        
    # The 3 white stones have only 1 liberty left at (1, 4)
    assert b.grid[1, 1] == WHITE and b.grid[1, 2] == WHITE and b.grid[1, 3] == WHITE
    # Black plays (1, 4) to capture all 3 stones
    assert b.is_valid_move(1 * 9 + 4)
    b.play(1 * 9 + 4)
    
    # All 3 white stones should be captured
    assert b.grid[1, 1] == EMPTY
    assert b.grid[1, 2] == EMPTY
    assert b.grid[1, 3] == EMPTY
    assert b.grid[1, 4] == BLACK

def test_seki_scoring():
    """
    Seki (dual life): Two opposing groups share common liberties.
    Neither player can fill the shared liberty without self-atari.
    In Area scoring, shared liberties remain empty and border both colors -> neutral territory.
    """
    b = Board(size=9)
    # Put a stone of Black at (0, 1) and White at (0, 3)
    # They share an empty point at (0, 2)
    b.grid[0, 1] = BLACK
    b.grid[0, 3] = WHITE
    # Both groups also touch the rest of the board which touches both -> empty point at (0,2) touches both B and W
    b_score, w_score, winner = compute_area_score(b, komi=7.5)
    # Shared point (0, 2) touches both Black and White, so neither player gets territory from it
    # Black score: 1 stone + 0 territory
    # White score: 1 stone + 0 territory + 7.5 komi
    assert b_score == 1.0
    assert w_score == 8.5
