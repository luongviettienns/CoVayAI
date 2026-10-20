import pytest
import torch
import numpy as np
from nn.model import AlphaZeroNet

def test_model_forward_shapes():
    net = AlphaZeroNet(board_size=9, num_res_blocks=4, num_filters=64)
    x = torch.randn(2, 17, 9, 9)
    policy_logits, value = net(x)
    
    assert policy_logits.shape == (2, 82) # 9*9 + 1 = 82
    assert value.shape == (2, 1)
    # Check value bounds [-1, 1]
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

def test_model_backward_and_gradients():
    net = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32)
    x = torch.randn(2, 17, 9, 9)
    policy_logits, value = net(x)

    target_policy = torch.zeros(2, 82)
    target_policy[:, 0] = 1.0
    target_value = torch.ones(2, 1)

    loss_p = torch.nn.functional.cross_entropy(policy_logits, target_policy)
    loss_v = torch.nn.functional.mse_loss(value, target_value)
    loss = loss_p + loss_v
    loss.backward()

    for name, param in net.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"

def test_predict_with_mask():
    net = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32)
    features = np.zeros((17, 9, 9), dtype=np.float32)
    
    # Legal moves mask: only action 0 and pass (81) are legal
    mask = np.zeros(82, dtype=np.int8)
    mask[0] = 1
    mask[81] = 1

    probs, val = net.predict(features, legal_moves_mask=mask)
    assert len(probs) == 82
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > 0.0
    assert probs[81] > 0.0
    # Illegal moves must have 0.0 probability
    assert np.all(probs[1:81] == 0.0)
    assert -1.0 <= val <= 1.0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution():
    device = torch.device("cuda")
    net = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32).to(device)
    x = torch.randn(2, 17, 9, 9, device=device)
    policy_logits, value = net(x)
    assert policy_logits.is_cuda
    assert value.is_cuda


def test_predict_batch():
    net = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32)
    net.eval()
    batch_features = np.zeros((4, 17, 9, 9), dtype=np.float32)
    batch_features[0, 0, 0, 0] = 1.0
    batch_features[1, 1, 1, 1] = 1.0

    masks = np.ones((4, 82), dtype=np.int8)
    masks[0, 10:] = 0  # Only moves 0-9 legal for sample 0

    probs_b, vals_b = net.predict_batch(batch_features, legal_moves_masks=masks)

    assert probs_b.shape == (4, 82)
    assert vals_b.shape == (4, 1)

    for i in range(4):
        assert np.isclose(np.sum(probs_b[i]), 1.0)
        assert -1.0 <= vals_b[i, 0] <= 1.0

    # Sample 0 illegal moves must have probability 0
    assert np.all(probs_b[0, 10:] == 0.0)

    # Check match with single predict
    p0, v0 = net.predict(batch_features[0], legal_moves_mask=masks[0])
    np.testing.assert_allclose(probs_b[0], p0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vals_b[0, 0], v0, rtol=1e-5, atol=1e-5)

