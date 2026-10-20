import pytest
import numpy as np
from go.board import Board
from go.zobrist import BLACK, WHITE
from nn.model import AlphaZeroNet
from mcts.search import MCTS, select_action

class DummyModel:
    """Mock model that returns uniform policy on legal moves and value 0.0."""
    def predict(self, state_features, legal_moves_mask=None, device=None):
        action_size = len(legal_moves_mask)
        legal_indices = np.where(legal_moves_mask > 0)[0]
        probs = np.zeros(action_size, dtype=np.float32)
        if len(legal_indices) > 0:
            probs[legal_indices] = 1.0 / len(legal_indices)
        return probs, 0.0

def test_mcts_search_visits():
    model = DummyModel()
    mcts = MCTS(model=model, num_simulations=50, c_puct=1.5, add_noise=False)
    board = Board(size=9)

    pi = mcts.search(board, temperature=1.0)
    assert len(pi) == 82
    assert np.isclose(np.sum(pi), 1.0)
    assert mcts.root.visit_count == 50
    # Sum of visits of children of root equals total simulations
    child_visits = sum(child.visit_count for child in mcts.root.children.values())
    assert child_visits == 50

def test_mcts_temperature():
    model = DummyModel()
    mcts = MCTS(model=model, num_simulations=30, c_puct=1.5, add_noise=False)
    board = Board(size=9)

    # Search always returns soft policy distribution (tau = 1.0)
    pi_soft = mcts.search(board)
    assert np.isclose(np.sum(pi_soft), 1.0)

    # Action selection with temperature=0.0 selects greedy action
    best_action = select_action(pi_soft, temperature=0.0)
    assert pi_soft[best_action] == np.max(pi_soft)

def test_mcts_dirichlet_noise():
    model = DummyModel()
    # Search with noise
    mcts_noise = MCTS(model=model, num_simulations=10, add_noise=True, dirichlet_alpha=0.25, dirichlet_eps=0.5)
    board = Board(size=9)
    mcts_noise.search(board)

    # Children of root should have priors altered by noise
    priors = [child.prior for child in mcts_noise.root.children.values()]
    # At least some priors should not be exactly 1/82
    assert not all(np.isclose(p, 1.0 / 82) for p in priors)

def test_mcts_subtree_reuse():
    model = DummyModel()
    mcts = MCTS(model=model, num_simulations=40, add_noise=False)
    board = Board(size=9)
    pi = mcts.search(board, temperature=1.0)
    
    # Pick the most visited child action
    best_action = max(mcts.root.children.items(), key=lambda item: item[1].visit_count)[0]
    expected_visits = mcts.root.children[best_action].visit_count

    # Execute move on board and advance MCTS tree
    board.play(best_action)
    mcts.advance_root(best_action)

    assert mcts.root.visit_count == expected_visits
    assert mcts.root.parent is None # Root has no parent
    
    # Running another search continues from this subroot
    mcts.search(board, temperature=1.0)
    assert mcts.root.visit_count == expected_visits + 40

def test_mcts_with_alphazeronet():
    """Verify end-to-end integration of MCTS with AlphaZeroNet neural network."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32).to(device)
    
    mcts = MCTS(model=net, num_simulations=20, c_puct=1.5, add_noise=True, device=device)
    board = Board(size=9)
    pi = mcts.search(board, temperature=1.0)
    
    assert len(pi) == 82
    assert np.isclose(np.sum(pi), 1.0)
    assert mcts.root.visit_count == 20
    # Must only visit legal actions
    legal_mask = board.get_legal_moves()
    for a, child in mcts.root.children.items():
        assert legal_mask[a] > 0
        assert child.visit_count >= 0

