"""Tests for size-independent neural network architecture (PR 4 - Phase P2).

Verifies:
1. Single model forward and backward on 9x9, 13x13, and 19x19 board sizes.
2. State dict has zero dependence on board size N.
3. Gradient flow is finite and nonzero across all parameters for all sizes.
4. Pass action is preserved at index N*N and invariant under D4 spatial transformations.
5. Predict and predict_batch work seamlessly across board sizes.
6. Backward compatibility / migration helper loads backbone from v1 checkpoints.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from nn.model import AlphaZeroNet


def test_single_model_forward_across_sizes():
    """Verify that a single model instance evaluates 9x9, 13x13, and 19x19 boards."""
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32)
    model.eval()

    sizes = [9, 13, 19]
    for n in sizes:
        x = torch.randn(2, 17, n, n)
        policy_logits, value = model(x)

        expected_action_size = n * n + 1
        assert policy_logits.shape == (2, expected_action_size), (
            f"Expected policy shape (2, {expected_action_size}) for board {n}x{n}, got {policy_logits.shape}"
        )
        assert value.shape == (2, 1), f"Expected value shape (2, 1) for board {n}x{n}, got {value.shape}"
        assert torch.all(value >= -1.0) and torch.all(value <= 1.0), "Value must be in [-1, 1]"


def test_single_model_backward_gradients_across_sizes():
    """Verify that gradients are finite and nonzero for all parameters on 9x9, 13x13, and 19x19."""
    for n in [9, 13, 19]:
        model = AlphaZeroNet(num_res_blocks=2, num_filters=32)
        model.train()

        x = torch.randn(2, 17, n, n)
        policy_logits, value = model(x)

        action_size = n * n + 1
        target_policy = torch.zeros(2, action_size)
        target_policy[:, 0] = 0.5
        target_policy[:, -1] = 0.5  # target includes pass move

        target_value = torch.tensor([[0.5], [-0.5]])

        loss_p = F.cross_entropy(policy_logits, target_policy)
        loss_v = F.mse_loss(value, target_value)
        loss = loss_p + loss_v
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has None grad on board size {n}"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf grad on board size {n}"
            assert param.grad.abs().sum() > 0, f"Parameter {name} has zero grad on board size {n}"


def test_state_dict_has_no_board_size_dependence():
    """Verify that no tensor in state_dict has shape dependent on board size N."""
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32)
    state_dict = model.state_dict()

    banned_dims = {81, 82, 169, 170, 361, 362}
    for key, tensor in state_dict.items():
        for dim in tensor.shape:
            assert dim not in banned_dims, (
                f"State dict parameter '{key}' has shape {tensor.shape} containing board-size dependent dimension {dim}!"
            )


def test_predict_and_predict_batch_across_sizes():
    """Verify inference API (predict and predict_batch) on different board sizes with legal masks."""
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32)
    model.eval()

    for n in [9, 13, 19]:
        action_size = n * n + 1

        # Single predict
        single_features = np.zeros((17, n, n), dtype=np.float32)
        mask = np.zeros(action_size, dtype=np.int8)
        mask[0] = 1
        mask[-1] = 1  # pass move
        probs, val = model.predict(single_features, legal_moves_mask=mask)

        assert probs.shape == (action_size,)
        assert np.isclose(probs.sum(), 1.0)
        assert probs[0] > 0.0
        assert probs[-1] > 0.0
        if action_size > 2:
            assert np.all(probs[1:-1] == 0.0)
        assert -1.0 <= val <= 1.0

        # Batch predict
        batch_size = 4
        batch_features = np.zeros((batch_size, 17, n, n), dtype=np.float32)
        batch_masks = np.ones((batch_size, action_size), dtype=np.int8)
        batch_masks[0, 5:] = 0  # only first 5 moves legal

        probs_b, vals_b = model.predict_batch(batch_features, legal_moves_masks=batch_masks)
        assert probs_b.shape == (batch_size, action_size)
        assert vals_b.shape == (batch_size, 1)

        for i in range(batch_size):
            assert np.isclose(probs_b[i].sum(), 1.0)
            assert -1.0 <= vals_b[i, 0] <= 1.0


def test_d4_symmetry_preserves_pass_action():
    """Verify that spatial rotation/reflection rotates spatial logits and preserves pass logit invariant."""
    model = AlphaZeroNet(num_res_blocks=2, num_filters=32)
    model.eval()

    torch.manual_seed(42)
    n = 9
    x = torch.randn(1, 17, n, n)
    # Rotate input 90 degrees clockwise (k=1 rot90 along (-2, -1))
    x_rot = torch.rot90(x, k=-1, dims=(-2, -1))

    with torch.no_grad():
        logits_orig, val_orig = model(x)
        logits_rot, val_rot = model(x_rot)

    pass_orig = logits_orig[0, -1].item()
    pass_rot = logits_rot[0, -1].item()

    # Pass logit and value are approximately invariant under spatial rotation (within boundary padding tolerance)
    np.testing.assert_allclose(pass_orig, pass_rot, atol=0.05, err_msg="Pass logit must be approximately invariant under rotation")
    np.testing.assert_allclose(val_orig.item(), val_rot.item(), atol=0.05, err_msg="Value must be approximately invariant under rotation")

    # Verify D4 data symmetry preserves pass action exactly across all board sizes
    from nn.features import apply_random_symmetry
    for size in [9, 13, 19]:
        feat = np.random.randn(17, size, size).astype(np.float32)
        pi = np.random.rand(size * size + 1).astype(np.float32)
        pi /= pi.sum()
        pass_prob_orig = pi[-1]

        feat_sym, pi_sym = apply_random_symmetry(feat, pi, flip=True, rot=3)
        assert feat_sym.shape == (17, size, size)
        assert pi_sym.shape == (size * size + 1,)
        assert np.isclose(pi_sym[-1], pass_prob_orig), f"D4 transformation must preserve pass probability on {size}x{size}"


def test_migration_loads_backbone_from_legacy_checkpoint():
    """Verify that load_compatible_weights cleanly extracts backbone weights when heads differ."""
    # Simulate legacy v1 model state dict with fixed linear heads
    legacy_state_dict = {}
    legacy_state_dict["conv_in.weight"] = torch.randn(32, 17, 3, 3)
    legacy_state_dict["bn_in.weight"] = torch.randn(32)
    legacy_state_dict["bn_in.bias"] = torch.randn(32)
    legacy_state_dict["bn_in.running_mean"] = torch.randn(32)
    legacy_state_dict["bn_in.running_var"] = torch.ones(32)
    legacy_state_dict["bn_in.num_batches_tracked"] = torch.tensor(10)

    # Legacy FC heads
    legacy_state_dict["policy_fc.weight"] = torch.randn(82, 162)
    legacy_state_dict["policy_fc.bias"] = torch.randn(82)
    legacy_state_dict["value_fc1.weight"] = torch.randn(128, 81)
    legacy_state_dict["value_fc1.bias"] = torch.randn(128)

    model = AlphaZeroNet(num_res_blocks=0, num_filters=32)
    loaded_backbone, reinit_heads = model.load_compatible_weights(legacy_state_dict)

    assert loaded_backbone is True
    assert reinit_heads is True
    # Backbone conv_in must match legacy weights exactly
    assert torch.equal(model.conv_in.weight, legacy_state_dict["conv_in.weight"])
