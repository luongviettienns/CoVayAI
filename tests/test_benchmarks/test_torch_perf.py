"""
Tests for PyTorch Performance Optimization: Memory Formats (Channels-Last) & Model Compilation (PR 4 / Master Plan V4).
Verifies:
- Output equivalence between standard contiguous (NCHW) and channels-last (NHWC) memory format.
- Compatibility with auxiliary heads (ownership & score) under channels-last layout.
- Trainer integration and gradient optimization under channels-last memory format.
- Safe execution and graceful fallback for model compilation (torch.compile).
- Reproducible benchmarking metrics generation via benchmarks/torch_perf.py.
"""

import os
import pytest
import torch
import numpy as np

from config import CHANNELS_LAST, COMPILE_MODEL
from main import PipelineConfig
from nn.model import AlphaZeroNet
from selfplay.replay_buffer import ReplayBuffer
from train.trainer import Trainer
from benchmarks.torch_perf import (
    benchmark_memory_format,
    benchmark_compile,
    benchmark_training_throughput,
)


class TestTorchPerf:
    def test_channels_last_output_equivalence_standard_heads(self):
        """Verifies forward pass predictions match between contiguous and channels-last format."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32, aux_heads=False).to(device)
        model.eval()

        # Input tensor
        x_contiguous = torch.randn(4, 17, 9, 9, device=device)
        x_channels_last = x_contiguous.clone().to(memory_format=torch.channels_last)

        with torch.no_grad():
            pi_cont, v_cont = model(x_contiguous)

            # Switch model to channels_last
            model_cl = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32, aux_heads=False).to(device)
            model_cl.load_state_dict(model.state_dict())
            model_cl.to_channels_last()
            model_cl.eval()

            pi_cl, v_cl = model_cl(x_channels_last)

        # Output predictions should be identical within numerical tolerance
        assert torch.allclose(pi_cont, pi_cl, atol=1e-5, rtol=1e-4)
        assert torch.allclose(v_cont, v_cl, atol=1e-5, rtol=1e-4)

    def test_channels_last_output_equivalence_aux_heads(self):
        """Verifies forward pass with auxiliary ownership and score heads under channels-last."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32, aux_heads=True).to(device)
        model.eval()

        x_contiguous = torch.randn(2, 17, 9, 9, device=device)
        x_channels_last = x_contiguous.clone().to(memory_format=torch.channels_last)

        with torch.no_grad():
            pi_c, v_c, own_c, sc_c = model(x_contiguous, return_aux=True)

            model_cl = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=32, aux_heads=True).to(device)
            model_cl.load_state_dict(model.state_dict())
            model_cl.to_channels_last()
            model_cl.eval()

            pi_cl, v_cl, own_cl, sc_cl = model_cl(x_channels_last, return_aux=True)

        assert torch.allclose(pi_c, pi_cl, atol=1e-5, rtol=1e-4)
        assert torch.allclose(v_c, v_cl, atol=1e-5, rtol=1e-4)
        assert torch.allclose(own_c, own_cl, atol=1e-5, rtol=1e-4)
        assert torch.allclose(sc_c, sc_cl, atol=1e-5, rtol=1e-4)

    def test_trainer_integration_with_channels_last(self):
        """Verifies Trainer executes training step and updates weights correctly in channels-last mode."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = AlphaZeroNet(board_size=9, num_res_blocks=1, num_filters=16, aux_heads=True).to(device)
        trainer = Trainer(model, device=device, use_amp=False, channels_last=True)

        assert trainer.channels_last is True

        buffer = ReplayBuffer(capacity=200, board_size=9)
        for _ in range(50):
            buffer.push(
                np.zeros((17, 9, 9), dtype=np.uint8),
                np.ones(82, dtype=np.float32) / 82,
                1.0,
                ownership=np.ones((9, 9), dtype=np.int8),
                score=5.0,
                has_aux=True,
            )

        # Initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Train 5 steps
        metrics = trainer.train_iteration(buffer, steps=5, batch_size=16)

        assert metrics["actual_steps"] == 5
        assert metrics["total_loss"] > 0.0

        # Weights must have updated
        updated = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, model.parameters()))
        assert updated

    def test_benchmark_memory_format_returns_valid_metrics(self):
        """Verifies benchmark_memory_format runs and returns full metrics structure."""
        report = benchmark_memory_format(
            board_size=9,
            num_res_blocks=1,
            num_filters=16,
            batch_size=8,
            num_warmup=2,
            num_iters=5,
            device="cpu",
        )

        assert "contiguous" in report
        assert "channels_last" in report
        assert "speedup_ratio" in report
        assert "output_equivalence_max_diff" in report

        assert report["contiguous"]["evals_per_sec"] > 0
        assert report["channels_last"]["evals_per_sec"] > 0
        assert report["output_equivalence_max_diff"] < 1e-4

    def test_benchmark_training_throughput_returns_valid_metrics(self):
        """Verifies training step throughput benchmark executes and reports steps/sec."""
        report = benchmark_training_throughput(
            board_size=9,
            num_res_blocks=1,
            num_filters=16,
            batch_size=8,
            steps=5,
            device="cpu",
        )

        assert "steps_per_sec" in report
        assert "avg_step_ms" in report
        assert report["steps_per_sec"] > 0

    def test_benchmark_compile_safe_fallback(self):
        """Verifies benchmark_compile handles supported and unsupported compilation environments safely."""
        report = benchmark_compile(
            board_size=9,
            num_res_blocks=1,
            num_filters=16,
            device="cpu",
        )

        assert "supported" in report
        if report["supported"]:
            assert "speedup_ratio" in report
            assert report["speedup_ratio"] > 0
        else:
            assert "reason" in report

    def test_pipeline_config_includes_perf_flags(self):
        """Verifies PipelineConfig supports channels_last and compile_model configuration."""
        cfg = PipelineConfig(
            channels_last=True,
            compile_model=True,
        )
        assert cfg.channels_last is True
        assert cfg.compile_model is True
