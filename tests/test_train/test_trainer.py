import pytest
import os
import tempfile
import numpy as np
import torch
from nn.model import AlphaZeroNet
from selfplay.replay_buffer import ReplayBuffer
from train.trainer import Trainer

def test_trainer_single_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32)
    trainer = Trainer(model=model, lr=1e-3, weight_decay=1e-4, device=device)

    batch_size = 4
    states = torch.randn(batch_size, 17, 9, 9, device=device)
    target_pis = torch.zeros(batch_size, 82, device=device)
    target_pis[:, 0] = 1.0 # One-hot target
    target_zs = torch.ones(batch_size, 1, device=device)

    metrics = trainer.train_step(states, target_pis, target_zs)
    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert np.isfinite(metrics["total_loss"])
    assert metrics["total_loss"] > 0.0

def test_trainer_overfit_single_batch():
    """Verify gradient descent works by overfitting on a single batch for 20 steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32)
    trainer = Trainer(model=model, lr=5e-3, weight_decay=0.0, device=device)

    batch_size = 4
    states = torch.randn(batch_size, 17, 9, 9, device=device)
    target_pis = torch.zeros(batch_size, 82, device=device)
    target_pis[:, 5] = 1.0
    target_zs = torch.ones(batch_size, 1, device=device) * 0.5

    initial_loss = trainer.train_step(states, target_pis, target_zs)["total_loss"]
    for _ in range(25):
        final_loss = trainer.train_step(states, target_pis, target_zs)["total_loss"]

    assert final_loss < initial_loss

def test_trainer_iteration_with_replay_buffer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32)
    trainer = Trainer(model=model, lr=1e-3, device=device)

    buffer = ReplayBuffer(capacity=100)
    s = np.zeros((17, 9, 9), dtype=np.float32)
    pi = np.zeros(82, dtype=np.float32)
    pi[0] = 1.0
    for _ in range(20):
        buffer.push(s, pi, z=1.0)

    metrics = trainer.train_iteration(buffer, steps=5, batch_size=4, augment=True)
    assert "total_loss" in metrics
    assert metrics["total_loss"] > 0.0

def test_trainer_save_and_load_checkpoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32)
    trainer1 = Trainer(model=model1, device=device)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        trainer1.save_checkpoint(ckpt_path, iteration=5)

        model2 = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32)
        trainer2 = Trainer(model=model2, device=device)
        loaded_iter = trainer2.load_checkpoint(ckpt_path)

        assert loaded_iter == 5
        # Verify model parameters match exactly
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1.cpu(), p2.cpu())
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_trainer_compile_model_and_channels_last():
    """Verify Trainer handles compile_model and channels_last without NameError or unhandled exceptions."""
    device = torch.device("cpu")
    model = AlphaZeroNet(board_size=9, num_res_blocks=1, num_filters=16)
    trainer = Trainer(
        model=model,
        channels_last=True,
        compile_model=True,
        device=device,
    )
    assert trainer.channels_last is True
    # is_compiled can be True (if torch.compile supported) or False (graceful fallback)
    assert isinstance(trainer.is_compiled, bool)
