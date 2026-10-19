import pytest
import numpy as np
from selfplay.worker import play_selfplay_game

class FastDummyModel:
    """Mock model that returns uniform policy on legal moves and value 0.0."""
    def predict(self, state_features, legal_moves_mask=None, device=None):
        action_size = len(legal_moves_mask)
        legal_indices = np.where(legal_moves_mask > 0)[0]
        probs = np.zeros(action_size, dtype=np.float32)
        if len(legal_indices) > 0:
            probs[legal_indices] = 1.0 / len(legal_indices)
        return probs, 0.0

class ResignMockModel:
    """Mock model where Black is losing (-0.95) and White is winning (+0.95)."""
    def predict(self, state_features, legal_moves_mask=None, device=None):
        action_size = len(legal_moves_mask)
        legal_indices = np.where(legal_moves_mask > 0)[0]
        probs = np.zeros(action_size, dtype=np.float32)
        if len(legal_indices) > 0:
            probs[legal_indices] = 1.0 / len(legal_indices)
        # Plane 16 is 1.0 if Black's turn, 0.0 if White's turn
        is_black_turn = state_features[16, 0, 0] > 0.5
        v = -0.95 if is_black_turn else 0.95
        return probs, v

def test_play_selfplay_game():
    model = FastDummyModel()
    # Play a quick game with small max_moves to keep test fast
    samples = play_selfplay_game(
        model=model,
        board_size=9,
        num_simulations=5,
        temp_moves=2,
        max_moves=10,
        disable_resign=True,
    )
    assert len(samples) > 0
    assert len(samples) <= 10

    for s, pi, z in samples:
        assert s.shape == (17, 9, 9)
        assert pi.shape == (82,)
        assert np.isclose(np.sum(pi), 1.0)
        assert z in (-1.0, 1.0)

def test_selfplay_resignation():
    model = ResignMockModel()
    samples = play_selfplay_game(
        model=model,
        board_size=9,
        num_simulations=5,
        temp_moves=2,
        max_moves=40,
        resign_threshold=-0.9,
        disable_resign=False,
    )
    # Since model always evaluates -0.98, player should resign as soon as check triggers (step 10)
    assert len(samples) == 10
