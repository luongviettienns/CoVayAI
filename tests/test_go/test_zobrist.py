import pytest
import numpy as np
from go.zobrist import ZobristHash, BLACK, WHITE

def test_zobrist_determinism():
    z1 = ZobristHash(board_size=9, seed=12345)
    z2 = ZobristHash(board_size=9, seed=12345)
    assert np.array_equal(z1.table, z2.table)
    assert z1.turn_key == z2.turn_key

def test_zobrist_uniqueness():
    z = ZobristHash(board_size=9, seed=12345)
    # Check that keys for BLACK and WHITE are 64-bit non-zero integers and unique
    keys = z.table[:, :, 1:].flatten()
    assert len(set(keys)) == len(keys)
    assert 0 not in keys
    assert z.turn_key not in keys

def test_zobrist_incremental_vs_full():
    z = ZobristHash(board_size=9, seed=12345)
    board = np.zeros((9, 9), dtype=np.int8)
    
    current_hash = z.compute_full_hash(board, current_player=BLACK)
    
    # Place BLACK at (2, 3)
    board[2, 3] = BLACK
    current_hash = z.toggle_stone(current_hash, 2, 3, BLACK)
    current_hash = z.toggle_turn(current_hash)
    
    full_hash = z.compute_full_hash(board, current_player=WHITE)
    assert current_hash == full_hash

    # Place WHITE at (4, 5)
    board[4, 5] = WHITE
    current_hash = z.toggle_stone(current_hash, 4, 5, WHITE)
    current_hash = z.toggle_turn(current_hash)
    
    full_hash = z.compute_full_hash(board, current_player=BLACK)
    assert current_hash == full_hash

    # Capture BLACK at (2, 3) -> remove it
    board[2, 3] = 0
    current_hash = z.toggle_stone(current_hash, 2, 3, BLACK)
    
    full_hash = z.compute_full_hash(board, current_player=BLACK)
    assert current_hash == full_hash

def test_zobrist_xor_reversibility():
    z = ZobristHash(board_size=9, seed=42)
    h = 0
    h1 = z.toggle_stone(h, 0, 0, BLACK)
    assert h1 != h
    h2 = z.toggle_stone(h1, 0, 0, BLACK)
    assert h2 == h
