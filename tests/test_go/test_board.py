import pytest
import numpy as np
from go.zobrist import BLACK, WHITE, EMPTY
from go.board import Board

def test_initial_board():
    b = Board(size=9)
    assert b.size == 9
    assert b.current_player == BLACK
    assert b.consecutive_passes == 0
    assert not b.is_game_over()
    assert np.all(b.grid == EMPTY)
    assert len(b.history_positions) == 1  # empty board hash
    assert b.get_legal_moves()[-1] == 1   # Pass is always legal
    assert np.all(b.get_legal_moves()[:-1] == 1) # All empty points legal initially

def test_basic_moves():
    b = Board(size=9)
    # Black plays (4, 4)
    action = 4 * 9 + 4
    success = b.play(action)
    assert success
    assert b.grid[4, 4] == BLACK
    assert b.current_player == WHITE
    assert b.consecutive_passes == 0

    # White plays adjacent (4, 5)
    success = b.play(4 * 9 + 5)
    assert success
    assert b.grid[4, 5] == WHITE
    assert b.current_player == BLACK

def test_capture_single_stone():
    b = Board(size=9)
    # White plays in center (4, 4)
    b.play(b.pass_action) # Black pass
    b.play(4 * 9 + 4)      # White (4, 4)
    
    # Black surrounds White at (3,4), (5,4), (4,3), (4,5)
    b.play(3 * 9 + 4) # Black
    b.play(b.pass_action) # White pass
    b.play(5 * 9 + 4) # Black
    b.play(b.pass_action) # White pass
    b.play(4 * 9 + 3) # Black
    b.play(b.pass_action) # White pass
    
    # White stone at (4,4) has 1 liberty left at (4,5)
    assert b.grid[4, 4] == WHITE
    # Black captures
    b.play(4 * 9 + 5)
    assert b.grid[4, 4] == EMPTY # Captured!
    assert b.grid[4, 5] == BLACK

def test_suicide_prohibited():
    b = Board(size=9)
    # Black surrounds (0, 0) by occupying (0, 1) and (1, 0)
    b.play(0 * 9 + 1) # Black (0, 1)
    b.play(b.pass_action)
    b.play(1 * 9 + 0) # Black (1, 0)
    
    # Now it's White's turn. Playing (0, 0) would have 0 liberties and captures no stones.
    assert not b.is_valid_move(0 * 9 + 0)
    legal_moves = b.get_legal_moves()
    assert legal_moves[0] == 0 # (0, 0) is illegal

def test_suicide_exception_when_capturing():
    """A move that would otherwise have 0 liberties IS legal if it captures enemy stones and gains liberties."""
    b = Board(size=9)
    # White stone at (0, 0)
    b.play(b.pass_action)
    b.play(0 * 9 + 0) # White
    b.play(0 * 9 + 1) # Black (surrounds White's liberty to the right)
    b.play(b.pass_action)
    # Black plays (1, 0) capturing the white stone at (0,0)
    assert b.is_valid_move(1 * 9 + 0)
    b.play(1 * 9 + 0)
    assert b.grid[0, 0] == EMPTY # White captured
    assert b.grid[1, 0] == BLACK

def test_consecutive_passes_ends_game():
    b = Board(size=9)
    b.play(b.pass_action)
    assert not b.is_game_over()
    assert b.consecutive_passes == 1
    
    b.play(b.pass_action)
    assert b.is_game_over()
    assert b.consecutive_passes == 2

def test_positional_superko():
    """Test standard simple Ko / Positional Superko."""
    b = Board(size=9)
    # Construct standard Ko shape at center:
    # Black at (3,4), (5,4), (4,3)
    # White at (3,5), (5,5), (4,6)
    b.play(3 * 9 + 4) # B
    b.play(3 * 9 + 5) # W
    b.play(5 * 9 + 4) # B
    b.play(5 * 9 + 5) # W
    b.play(4 * 9 + 3) # B
    b.play(4 * 9 + 6) # W
    
    # Black plays (4, 5) -> now White stone at (4,5) is not there, Black plays it
    b.play(4 * 9 + 5) # B
    # White captures Black by playing (4, 4)
    assert b.is_valid_move(4 * 9 + 4)
    b.play(4 * 9 + 4) # W captures B at (4,5)
    assert b.grid[4, 5] == EMPTY
    
    # Now Black immediately trying to recapture at (4, 5) recreates the position before White's move!
    # Positional Superko must forbid this!
    assert not b.is_valid_move(4 * 9 + 5)
    legal = b.get_legal_moves()
    assert legal[4 * 9 + 5] == 0
