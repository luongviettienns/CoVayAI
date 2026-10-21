"""Monte Carlo Tree Search with PUCT, Dirichlet Noise, and Subtree Reuse."""

from typing import Optional, Dict
import numpy as np
import torch

from go.board import Board
from go.scoring import get_winner_value
from go.zobrist import opponent
from nn.features import extract_features
from mcts.node import MCTSNode


def select_action(pi: np.ndarray, temperature: float = 1.0) -> int:
    """Select action from policy distribution pi according to temperature.
    
    Args:
        pi: Soft policy distribution vector (probabilities summing to 1.0).
        temperature: If > 1e-2, sample action proportionally (optionally scaled by 1/T).
                     If <= 1e-2, greedily select action with maximum probability,
                     breaking ties uniformly at random.
                     
    Returns:
        Selected action integer.
    """
    if temperature > 1e-2:
        if abs(temperature - 1.0) > 1e-4:
            probs = pi ** (1.0 / temperature)
            total = np.sum(probs)
            if total > 0:
                probs /= total
            else:
                probs = pi
        else:
            probs = pi
        return int(np.random.choice(len(probs), p=probs))
    else:
        # Greedy selection with uniform random tie-breaking among max elements
        max_val = np.max(pi)
        candidates = np.flatnonzero(np.isclose(pi, max_val))
        return int(np.random.choice(candidates))


def get_default_dirichlet_alpha(board_size: int) -> float:
    """Calculate size-dependent Dirichlet noise concentration parameter.

    In AlphaZero and KataGo, alpha scales inversely with average legal move count:
      - 5x5: 0.60
      - 7x7: 0.35
      - 9x9: 0.25 (or 0.30)
      - 13x13: 0.08
      - 19x19: 0.03
    """
    if board_size <= 5:
        return 0.60
    elif board_size <= 7:
        return 0.35
    elif board_size <= 9:
        return 0.25
    elif board_size <= 13:
        return 0.08
    else:
        return 0.03


def get_default_temp_moves(board_size: int) -> int:
    """Calculate temperature exploration opening length scaled by board size."""
    if board_size <= 5:
        return 6
    elif board_size <= 7:
        return 8
    elif board_size <= 9:
        return 12
    elif board_size <= 13:
        return 20
    else:
        return 30


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
        dirichlet_alpha: Optional[float] = None,
        dirichlet_eps: float = 0.25,
        add_noise: bool = True,
        komi: float = 7.5,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise = add_noise
        self.komi = komi
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
            return get_winner_value(board, perspective=board.current_player, komi=self.komi)

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
            v = get_winner_value(board, perspective=board.current_player, komi=self.komi)
        else:
            v = self._expand_node(node, board)

        # 3. Backup: Backpropagate value v up to root, inverting sign at each alternating step
        curr = node
        curr_v = v
        while curr is not None:
            curr.update(curr_v)
            curr_v = -curr_v
            curr = curr.parent

    def search(
        self,
        board: Board,
        temperature: float = 1.0,
        num_simulations: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run MCTS simulations from the current board state and return normalized visit distribution.
        
        AlphaZero Target Rule:
            Target pi for neural network training is always the normalized visit distribution
            (tau = 1.0). Action selection should use `select_action(pi, temperature)`.
            
        Args:
            board: Current Board state
            temperature: Legacy exploration parameter. Note that search() returns soft
                         normalized visit count distribution pi regardless of temperature.
            num_simulations: Optional simulation override per move (e.g. for playout cap randomization).
                         
        Returns:
            pi: Soft normalized visit count vector of shape (N*N + 1,) summing to 1.0.
        """
        if board.is_game_over():
            action_size = board.size * board.size + 1
            pi = np.zeros(action_size, dtype=np.float32)
            pi[-1] = 1.0
            return pi

        # 1. If root is not yet expanded, expand it first
        if self.root.is_leaf():
            self._expand_node(self.root, board)

        # 2. Apply Dirichlet noise on root children whenever add_noise is True
        # Always compute noise relative to clean_prior to avoid cumulative distortion across searches
        if self.add_noise and len(self.root.children) > 0:
            effective_alpha = (
                self.dirichlet_alpha
                if self.dirichlet_alpha is not None
                else get_default_dirichlet_alpha(board.size)
            )
            actions = list(self.root.children.keys())
            noise = np.random.dirichlet([effective_alpha] * len(actions))
            for a, n in zip(actions, noise):
                child = self.root.children[a]
                child.prior = (
                    (1.0 - self.dirichlet_eps) * child.clean_prior
                    + self.dirichlet_eps * n
                )

        # 3. Run simulations from root
        sims_to_run = num_simulations if num_simulations is not None else self.num_simulations
        for _ in range(sims_to_run):
            self._simulate_once(board)

        # 4. Compute soft visit count distribution (tau = 1.0)
        action_size = board.size * board.size + 1
        pi = np.zeros(action_size, dtype=np.float32)

        if len(self.root.children) == 0:
            pi[-1] = 1.0
            return pi

        for a, child in self.root.children.items():
            pi[a] = child.visit_count

        total = np.sum(pi)
        if total > 0:
            pi /= total
        else:
            pi[-1] = 1.0

        return pi
