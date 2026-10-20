"""Monte Carlo Tree Search (MCTS) Node implementation."""

from typing import Dict, Optional, Tuple
import math

class MCTSNode:
    """Represents a single node in the MCTS search tree."""

    __slots__ = (
        "prior",
        "clean_prior",
        "parent",
        "children",
        "visit_count",
        "total_value",
        "value_score",
    )

    def __init__(self, prior: float = 0.0, parent: Optional["MCTSNode"] = None):
        self.prior = float(prior)
        self.clean_prior = float(prior)
        self.parent = parent
        self.children: Dict[int, "MCTSNode"] = {}
        
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.value_score: float = 0.0  # Q = total_value / visit_count

    def is_leaf(self) -> bool:
        """Return True if this node has not been expanded yet."""
        return len(self.children) == 0

    def update(self, value: float) -> None:
        """Update node visit count and accumulated value."""
        self.visit_count += 1
        self.total_value += value
        self.value_score = self.total_value / self.visit_count

    def puct_score(self, c_puct: float, parent_visit_count: int) -> float:
        """
        Compute PUCT upper confidence bound:
        Score = -Q_child(s,a) + c_puct * P(s,a) * sqrt(max(1, parent_visit_count)) / (1 + N(s,a))
        
        Note:
            self.value_score is from the child's perspective (next player to move = opponent of parent).
            Therefore, parent evaluating action a must invert the sign (-self.value_score).
            max(1, parent_visit_count) ensures the neural network prior P(s, a) guides exploration
            even on the very first simulation when parent_visit_count == 0.
        """
        u = c_puct * self.prior * math.sqrt(max(1, parent_visit_count)) / (1 + self.visit_count)
        return -self.value_score + u

    def best_child(self, c_puct: float) -> Tuple[int, "MCTSNode"]:
        """Select action and child node with highest PUCT score."""
        best_score = -float("inf")
        best_action = -1
        best_node = None

        parent_visits = self.visit_count
        for action, child in self.children.items():
            score = child.puct_score(c_puct, parent_visits)
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child

        return best_action, best_node
