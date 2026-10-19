import pytest
import numpy as np
from mcts.node import MCTSNode

def test_node_init():
    node = MCTSNode(prior=0.5)
    assert node.visit_count == 0
    assert node.total_value == 0.0
    assert node.value_score == 0.0
    assert node.prior == 0.5
    assert node.parent is None
    assert len(node.children) == 0
    assert node.is_leaf()

def test_node_update():
    node = MCTSNode(prior=0.2)
    node.update(value=0.6)
    assert node.visit_count == 1
    assert np.isclose(node.total_value, 0.6)
    assert np.isclose(node.value_score, 0.6)

    node.update(value=-0.2)
    assert node.visit_count == 2
    assert np.isclose(node.total_value, 0.4)
    assert np.isclose(node.value_score, 0.2)

def test_puct_selection():
    parent = MCTSNode(prior=1.0)
    parent.visit_count = 10
    
    # Child 1: high Q, low P
    child1 = MCTSNode(prior=0.1, parent=parent)
    child1.visit_count = 5
    child1.total_value = 2.5 # Q = 0.5
    child1.value_score = 0.5
    
    # Child 2: low Q, high P
    child2 = MCTSNode(prior=0.9, parent=parent)
    child2.visit_count = 1
    child2.total_value = 0.1 # Q = 0.1
    child2.value_score = 0.1

    c_puct = 1.5
    score1 = child1.puct_score(c_puct=c_puct, parent_visit_count=10)
    score2 = child2.puct_score(c_puct=c_puct, parent_visit_count=10)
    
    # Verify both scores are finite and computable
    assert np.isfinite(score1)
    assert np.isfinite(score2)
