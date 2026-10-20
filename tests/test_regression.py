"""Regression tests for the 4 critical algorithmic bugs identified in CoVayAI.

1. MCTS PUCT sign bug: Q child is from next player's perspective, PUCT must subtract Q.
2. Dirichlet noise skipped on move 2+ after subtree reuse (advance_root).
3. Self-play policy target becomes one-hot after move 12 (search with temp=0).
4. Arena deterministic replay: identical move sequences across games due to lack of opening plies.
"""

import pytest
import numpy as np
import torch

from go.board import Board
from go.zobrist import BLACK, WHITE
from mcts.search import MCTS
from nn.model import AlphaZeroNet
from selfplay.worker import play_selfplay_game
from arena.match import play_arena_game


class UniformDummyModel:
    """Mock model returning uniform policy on legal moves and 0.0 value."""
    def predict(self, state_features, legal_moves_mask=None, device=None):
        action_size = len(legal_moves_mask)
        legal_indices = np.where(legal_moves_mask > 0)[0]
        probs = np.zeros(action_size, dtype=np.float32)
        if len(legal_indices) > 0:
            probs[legal_indices] = 1.0 / len(legal_indices)
        return probs, 0.0


def test_mcts_winning_and_losing_moves():
    """Bug 1: MCTS must pick winning moves and avoid losing moves.

    When Black is clearly winning and White has just passed, Black passing terminates
    the game and secures Black's win. With correct PUCT (-Q + u), Black should
    overwhelmingly choose to pass. With the bug (+Q + u), Black avoids passing.
    """
    model = UniformDummyModel()
    board = Board(size=9)
    # Fill substantial territory for Black so Black is winning decisively
    for r in range(1, 8):
        for c in range(1, 8):
            board.grid[r, c] = BLACK
    # White passes first: consecutive_passes = 1, current_player = BLACK
    board.consecutive_passes = 1
    board.current_player = BLACK

    mcts = MCTS(model=model, num_simulations=50, c_puct=1.5, add_noise=False)
    pi = mcts.search(board, temperature=1.0)
    pass_action = board.pass_action

    # Pass should have the highest visit count since passing wins immediately
    best_action = np.argmax(pi)
    assert best_action == pass_action, (
        f"Expected MCTS to choose pass (action {pass_action}) to seal victory, "
        f"but chose action {best_action} with visits {mcts.root.children.get(best_action, None).visit_count if best_action in mcts.root.children else 0}. "
        f"Pass visits: {mcts.root.children[pass_action].visit_count if pass_action in mcts.root.children else 0}."
    )


def test_dirichlet_noise_persists_on_move_2():
    """Bug 2: Dirichlet noise must be applied at move 2+ even after advance_root.

    After advance_root, the new root already has children. Dirichlet noise should still
    be applied to the root's children to ensure exploration in self-play.
    """
    model = UniformDummyModel()
    mcts = MCTS(
        model=model,
        num_simulations=20,
        c_puct=1.5,
        add_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.5,
    )
    board = Board(size=9)

    # Move 1 search
    mcts.search(board, temperature=1.0)
    # Choose action and advance root
    action = max(mcts.root.children.items(), key=lambda item: item[1].visit_count)[0]
    board.play(action)
    mcts.advance_root(action)

    # Move 2 search
    mcts.search(board, temperature=1.0)

    # In move 2, Dirichlet noise should have modified the priors of the children
    priors = [child.prior for child in mcts.root.children.values()]
    # If noise was applied, priors should vary across legal actions (std > 0)
    # If noise was skipped, priors remain uniform (std == 0)
    prior_std = np.std(priors)
    assert prior_std > 1e-4, f"Dirichlet noise was skipped on move 2: prior_std = {prior_std}"


def test_selfplay_target_entropy_after_move_12():
    """Bug 3: Policy target in selfplay should retain soft distribution after move 12.

    AlphaZero trains policy target pi = N^(1/1) / sum(N). Temperature tau -> 0 is only
    for move selection. Storing one-hot target damages learning.
    """
    model = UniformDummyModel()
    samples = play_selfplay_game(
        model=model,
        board_size=9,
        num_simulations=30,
        temp_moves=5,  # test threshold at move 5 for fast testing
        max_moves=10,
        disable_resign=True,
    )

    assert len(samples) > 5, "Game should have more than 5 moves"

    # Check samples collected after temp_moves (step >= 5) before terminal cutoff
    for step_idx in range(5, len(samples) - 1):
        state, pi_target, z = samples[step_idx]
        max_prob = np.max(pi_target)
        # Soft target should have probability distributed among multiple visited moves
        assert max_prob < 0.99, (
            f"Sample at step {step_idx} has one-hot target (max_prob={max_prob:.4f}). "
            f"AlphaZero targets must be soft visit distributions!"
        )


def test_arena_diversity_with_opening_plies():
    """Bug 4: Arena games should not be completely identical across matches.

    With opening_plies > 0 (tau=1 in opening), two matches between the same models
    should explore different game trajectories.
    """
    torch.manual_seed(42)
    m1 = AlphaZeroNet(board_size=9, num_res_blocks=1, num_filters=16)
    m2 = AlphaZeroNet(board_size=9, num_res_blocks=1, num_filters=16)
    m1.eval()
    m2.eval()

    # Play two arena games with opening_plies=8 and recording moves
    out1 = play_arena_game(
        black_model=m1,
        white_model=m2,
        board_size=9,
        num_sims=5,
        max_moves=30,
        opening_plies=8,
        return_moves=True,
    )
    assert len(out1) == 3
    seq1 = out1[2]

    out2 = play_arena_game(
        black_model=m1,
        white_model=m2,
        board_size=9,
        num_sims=5,
        max_moves=30,
        opening_plies=8,
        return_moves=True,
    )
    assert len(out2) == 3
    seq2 = out2[2]

    # Sequences should diverge due to opening randomization
    assert seq1 != seq2, f"Arena games are identical: {seq1} == {seq2}"
