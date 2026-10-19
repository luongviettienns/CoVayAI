"""Monte Carlo Tree Search with PUCT, Dirichlet Noise, and Subtree Reuse."""

from typing import Optional, Dict
import numpy as np
import torch

from go.board import Board
from go.scoring import get_winner_value
from go.zobrist import opponent
from nn.features import extract_features
from mcts.node import MCTSNode

class MCTS:
    """
    Monte Carlo Tree Search algorithm guided by dual-head neural network.
    Uses PUCT selection, Dirichlet root noise, and subtree reuse.
    """

    def __init__(
        self,
        model,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.25,
        dirichlet_eps: float = 0.25,
        add_noise: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise = add_noise
        self.device = device

        self.root = MCTSNode(prior=1.0)

    def reset(self) -> None:
        """Reset the search tree completely."""
        self.root = MCTSNode(prior=1.0)

    def advance_root(self, action: int) -> None:
        """
        Advance root to child corresponding to action taken (subtree reuse).
        If child doesn't exist, reset tree.
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.reset()

    def _expand_node(self, node: MCTSNode, board: Board) -> float:
        """
        Evaluate board state with the neural network and expand children.
        Returns value v from the perspective of board.current_player.
        """
        if board.is_game_over():
            return get_winner_value(board, perspective=board.current_player, komi=7.5)

        features = extract_features(board)
        mask = board.get_legal_moves()
        probs, v = self.model.predict(features, legal_moves_mask=mask, device=self.device)

        for action in range(len(mask)):
            if mask[action] > 0:
                node.children[action] = MCTSNode(prior=probs[action], parent=node)

        return v

    def _simulate_once(self, root_board: Board) -> None:
        """Execute one MCTS simulation: Select -> Expand/Evaluate -> Backup."""
        if root_board.is_game_over() or len(self.root.children) == 0:
            return

        board = root_board.copy()
        node = self.root

        # 1. Select: Traverse down tree until reaching an unexpanded leaf or terminal node
        while not node.is_leaf() and not board.is_game_over():
            action, node = node.best_child(self.c_puct)
            board.play(action)

        # 2. Evaluate & Expand leaf
        if board.is_game_over():
            v = get_winner_value(board, perspective=board.current_player, komi=7.5)
        else:
            v = self._expand_node(node, board)

        # 3. Backup: Backpropagate value v up to root, inverting sign at each alternating step
        curr = node
        curr_v = v
        while curr is not None:
            curr.update(curr_v)
            curr_v = -curr_v
            curr = curr.parent

    def search(self, board: Board, temperature: float = 1.0) -> np.ndarray:
        """
        Run MCTS simulations from the current board state.
        
        Args:
            board: Current Board state
            temperature: Exploration temperature tau for policy target pi.
                         tau -> 0 produces deterministic greedy action.
                         tau = 1.0 samples proportionally to visit counts.
                         
        Returns:
            pi: Normalized policy vector of shape (N*N + 1,)
        """
        if board.is_game_over():
            action_size = board.size * board.size + 1
            pi = np.zeros(action_size, dtype=np.float32)
            pi[-1] = 1.0
            return pi

        # If root is not yet expanded, expand it first
        if self.root.is_leaf():
            self._expand_node(self.root, board)
            # Apply Dirichlet noise on root children
            if self.add_noise and len(self.root.children) > 0:
                actions = list(self.root.children.keys())
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
                for a, n in zip(actions, noise):
                    self.root.children[a].prior = (
                        (1.0 - self.dirichlet_eps) * self.root.children[a].prior
                        + self.dirichlet_eps * n
                    )

        # Run simulations from root
        for _ in range(self.num_simulations):
            self._simulate_once(board)

        action_size = board.size * board.size + 1
        pi = np.zeros(action_size, dtype=np.float32)

        if len(self.root.children) == 0:
            pi[-1] = 1.0
            return pi

        if temperature <= 1e-2:
            # Greedy: select action with maximum visit count
            best_action = max(self.root.children.items(), key=lambda item: item[1].visit_count)[0]
            pi[best_action] = 1.0
        else:
            for a, child in self.root.children.items():
                pi[a] = child.visit_count ** (1.0 / temperature)
            total = np.sum(pi)
            if total > 0:
                pi /= total
            else:
                pi[-1] = 1.0

        return pi
