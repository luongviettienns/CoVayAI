import pytest
import numpy as np
from go.board import Board
from go.zobrist import BLACK, WHITE
from nn.features import extract_features, get_symmetries

def test_extract_features_shape():
    b = Board(size=9)
    feat = extract_features(b)
    assert feat.shape == (17, 9, 9)
    assert feat.dtype == np.float32

def test_extract_features_empty_board():
    b = Board(size=9)
    feat = extract_features(b)
    # Planes 0..15 should be all 0s (no stones yet)
    assert np.all(feat[:16] == 0.0)
    # Plane 16 is Black's turn (1.0)
    assert np.all(feat[16] == 1.0)

def test_extract_features_canonical_perspective():
    b = Board(size=9)
    # Black plays (3, 3)
    b.play(3 * 9 + 3)
    # Now it's White's turn
    feat = extract_features(b)
    assert b.current_player == WHITE
    
    # Plane 0: Current player (White) stones at time t -> should be all 0
    assert np.all(feat[0] == 0.0)
    # Plane 8: Opponent (Black) stones at time t -> should have stone at (3, 3)
    assert feat[8, 3, 3] == 1.0
    assert np.sum(feat[8]) == 1.0
    # Plane 16: White's turn -> should be 0.0
    assert np.all(feat[16] == 0.0)

def test_d4_symmetries_count_and_invariants():
    size = 9
    feat = np.zeros((17, size, size), dtype=np.float32)
    # Put asymmetric marks
    feat[0, 1, 2] = 1.0
    feat[8, 5, 6] = 1.0
    feat[16, :, :] = 1.0

    pi = np.zeros(size * size + 1, dtype=np.float32)
    pi[1 * size + 2] = 0.7 # corresponding to (1, 2)
    pi[size * size] = 0.3  # pass move

    symmetries = get_symmetries(feat, pi)
    assert len(symmetries) == 8

    for s_sym, pi_sym in symmetries:
        assert s_sym.shape == (17, size, size)
        assert pi_sym.shape == (size * size + 1,)
        # Pass move probability must remain completely invariant in all 8 symmetries!
        assert np.isclose(pi_sym[-1], 0.3)
        # Sum of probabilities remains 1.0
        assert np.isclose(np.sum(pi_sym), 1.0)
