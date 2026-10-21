"""Tests for PR 10: Ownership and Score Auxiliary Supervision (Phase P6)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from go.board import Board
from go.zobrist import BLACK, WHITE, EMPTY
from go.scoring import compute_area_score, compute_ownership_grid
from nn.model import AlphaZeroNet
from selfplay.replay_buffer import ReplayBuffer
from selfplay.worker import play_selfplay_game, GameResult
from train.trainer import Trainer


class TestAuxiliarySupervision:
    """Test suite verifying auxiliary heads, ownership ground truth, and multi-task loss."""

    def test_compute_ownership_grid_simple_board(self):
        """Verify ground truth ownership grid correctly assigns +1 to Black, -1 to White, 0 to dame."""
        board = Board(size=5)
        # Create a simple game state with known territory
        # Black plays (1, 1), White plays (3, 3)
        board.play(1 * 5 + 1)  # Black
        board.play(3 * 5 + 3)  # White
        board.play(board.pass_action)  # Black pass
        board.play(board.pass_action)  # White pass (game over)

        ownership = compute_ownership_grid(board)
        assert ownership.shape == (5, 5)
        assert ownership[1, 1] == 1.0  # Black stone
        assert ownership[3, 3] == -1.0  # White stone

        b_score, w_score, winner = compute_area_score(board, komi=0.5)
        # Ownership +1 count equals black territory + stones
        assert np.sum(ownership == 1.0) == (b_score)
        # Ownership -1 count equals white territory + stones (without komi)
        assert np.sum(ownership == -1.0) == (w_score - 0.5)

    def test_alphazero_net_with_aux_heads_size_independent(self):
        """Verify AlphaZeroNet with aux_heads works for both 9x9 and 19x19 with return_aux flag."""
        net = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32, aux_heads=True)

        # 1. Default forward should maintain strict backward compatibility: returns (pi, v)
        x_9 = torch.randn(2, 17, 9, 9)
        out = net(x_9)
        assert isinstance(out, tuple) and len(out) == 2
        pi_9, v_9 = out
        assert pi_9.shape == (2, 9 * 9 + 1)
        assert v_9.shape == (2, 1)

        # 2. Forward with return_aux=True returns (pi, v, ownership, score)
        pi_9, v_9, own_9, score_9 = net(x_9, return_aux=True)
        assert own_9.shape == (2, 9, 9)
        assert score_9.shape == (2, 1)
        assert (-1.01 <= own_9).all() and (own_9 <= 1.01).all()  # tanh output
        assert (-1.01 <= score_9).all() and (score_9 <= 1.01).all()  # tanh output

        # 3. Size-independent execution on 19x19 input without modifying model weights
        x_19 = torch.randn(2, 17, 19, 19)
        pi_19, v_19, own_19, score_19 = net(x_19, return_aux=True)
        assert pi_19.shape == (2, 19 * 19 + 1)
        assert v_19.shape == (2, 1)
        assert own_19.shape == (2, 19, 19)
        assert score_19.shape == (2, 1)

    def test_replay_buffer_stores_and_augments_auxiliary_targets(self):
        """Verify ReplayBuffer allocates compact aux buffers and applies D4 symmetry to ownership."""
        buf = ReplayBuffer(capacity=100, board_size=5)
        state = np.zeros((17, 5, 5), dtype=np.uint8)
        state[0, 1, 1] = 1
        pi = np.zeros(26, dtype=np.float32)
        pi[1 * 5 + 1] = 1.0
        z = 1.0

        ownership = np.zeros((5, 5), dtype=np.int8)
        ownership[1, 1] = 1
        score = 0.5

        # Push with aux
        buf.push(state, pi, z, ownership=ownership, score=score, has_aux=True)
        assert len(buf) == 1

        # Sample with return_aux=True
        states, pis, zs, owns, scores, masks = buf.sample_batch(batch_size=4, augment=True, return_aux=True)
        assert states.shape == (4, 17, 5, 5)
        assert pis.shape == (4, 26)
        assert zs.shape == (4, 1)
        assert owns.shape == (4, 5, 5)
        assert scores.shape == (4, 1)
        assert masks.shape == (4, 1)
        assert (masks == 1.0).all()

        # Check that when state rotates, ownership rotates congruently
        for i in range(4):
            # The stone at (1, 1) in state should match the ownership == 1 position
            stone_pos = np.argwhere(states[i, 0].numpy() > 0.5)
            own_pos = np.argwhere(owns[i].numpy() > 0.5)
            assert np.array_equal(stone_pos, own_pos)

    def test_selfplay_generates_auxiliary_labels_only_for_terminal_games(self):
        """Verify selfplay sets valid aux labels for natural 2-pass games and zeroes for truncated games."""
        net = AlphaZeroNet(board_size=5, num_res_blocks=1, num_filters=16, aux_heads=True)

        # 1. Natural terminal game (consecutive passes)
        game_res = play_selfplay_game(
            model=net,
            board_size=5,
            num_simulations=10,
            max_moves=20,
            adjudicate_truncated=False,
        )
        assert hasattr(game_res, "has_aux")
        if game_res.termination_reason == "two_passes":
            assert game_res.has_aux is True
            assert game_res.ownership is not None
            assert game_res.ownership.shape == (5, 5)
            assert game_res.score_diff is not None

        # 2. Truncated game without adjudication
        game_truncated = play_selfplay_game(
            model=net,
            board_size=5,
            num_simulations=5,
            max_moves=2,  # Artificially short
            adjudicate_truncated=False,
        )
        assert game_truncated.termination_reason == "truncated"
        assert game_truncated.has_aux is False
        assert len(game_truncated.samples) == 0

    def test_trainer_auxiliary_loss_optimization(self):
        """Verify Trainer computes ownership and score losses and updates model weights."""
        net = AlphaZeroNet(board_size=5, num_res_blocks=2, num_filters=32, aux_heads=True)
        trainer = Trainer(
            model=net,
            lr=1e-3,
            optimizer_type="adamw",
            use_amp=False,
            ownership_coef=0.25,
            score_coef=0.02,
        )

        states = torch.randn(4, 17, 5, 5)
        target_pis = torch.softmax(torch.randn(4, 26), dim=-1)
        target_zs = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
        target_owns = torch.randn(4, 5, 5).clamp(-1.0, 1.0)
        target_scores = torch.tensor([[0.2], [-0.5], [0.8], [-0.1]])
        aux_masks = torch.tensor([[1.0], [1.0], [0.0], [1.0]])

        # Train step with aux
        metrics = trainer.train_step(
            states=states,
            target_pis=target_pis,
            target_zs=target_zs,
            target_ownerships=target_owns,
            target_scores=target_scores,
            aux_masks=aux_masks,
        )

        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "ownership_loss" in metrics
        assert "score_loss" in metrics
        assert metrics["ownership_loss"] > 0.0
        assert metrics["score_loss"] > 0.0

        # Verify backward compatibility: train step without aux
        metrics_no_aux = trainer.train_step(
            states=states,
            target_pis=target_pis,
            target_zs=target_zs,
        )
        assert metrics_no_aux["ownership_loss"] == 0.0
        assert metrics_no_aux["score_loss"] == 0.0

    def test_checkpoint_compatibility_with_and_without_aux_heads(self):
        """Verify cross-compatibility when loading weights between models with and without aux heads."""
        base_net = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32, aux_heads=False)
        aux_net = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32, aux_heads=True)

        # 1. Aux net loads weights from base net
        loaded, reinit = aux_net.load_compatible_weights(base_net.state_dict())
        assert loaded is True
        assert reinit is True  # aux heads are newly initialized

        # Forward pass works after loading
        x = torch.randn(2, 17, 9, 9)
        p, v, own, sc = aux_net(x, return_aux=True)
        assert p.shape == (2, 82)
        assert v.shape == (2, 1)

        # 2. Base net loads weights from aux net (ignoring aux heads)
        loaded_base, reinit_base = base_net.load_compatible_weights(aux_net.state_dict())
        assert loaded_base is True
        p_base, v_base = base_net(x)
        # Outputs must match
        torch.testing.assert_close(p, p_base)
        torch.testing.assert_close(v, v_base)

    def test_replay_buffer_roundtrip_save_load_with_aux_targets(self, tmp_path):
        """Verify buffer saving and loading perfectly preserves ownership and score targets."""
        buf = ReplayBuffer(capacity=50, board_size=5)
        s = np.zeros((17, 5, 5), dtype=np.uint8)
        p = np.zeros(26, dtype=np.float32)
        z = 1.0
        own = np.ones((5, 5), dtype=np.int8)
        sc = 0.75

        buf.push(s, p, z, ownership=own, score=sc, has_aux=True)

        save_file = str(tmp_path / "test_buffer_aux.pkl")
        buf.save(save_file)

        buf2 = ReplayBuffer(capacity=50, board_size=5)
        buf2.load(save_file)

        assert len(buf2) == 1
        states, pis, zs, owns, scores, masks = buf2.sample_batch(batch_size=1, augment=False, return_aux=True)
        assert (owns[0].numpy() == 1.0).all()
        assert np.isclose(scores[0, 0].item(), 0.75, atol=1e-3)
        assert masks[0, 0].item() == 1.0
