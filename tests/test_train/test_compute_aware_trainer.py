"""
Tests for Compute-Aware Training Steps and Game-Level Held-Out Validation (P8 & C3).
Verifies:
- Adaptive training steps scaling according to new incoming valid samples.
- Game-level partitioning ensuring zero data leakage between train and validation sets.
- Held-out validation metrics calculation (Val Loss, Policy KL divergence, Policy Entropy, Value Calibration).
"""

import numpy as np
import pytest
import torch

from main import PipelineConfig
from nn.model import AlphaZeroNet
from selfplay.replay_buffer import ReplayBuffer
from selfplay.worker import GameResult
from train.trainer import (
    Trainer,
    compute_adaptive_train_steps,
    split_games_train_val,
)


class TestComputeAwareTrainer:
    def test_compute_adaptive_train_steps_scaling(self):
        """Verifies adaptive step calculation scales with new samples and respects bounds."""
        # 100 new samples * 0.5 = 50 steps
        assert compute_adaptive_train_steps(new_samples=100, steps_per_sample=0.5, min_steps=20, max_steps=400) == 50

        # Small sample count should clamp to min_steps
        assert compute_adaptive_train_steps(new_samples=10, steps_per_sample=0.5, min_steps=20, max_steps=400) == 20

        # Very large sample count should clamp to max_steps
        assert compute_adaptive_train_steps(new_samples=2000, steps_per_sample=0.5, min_steps=20, max_steps=400) == 400

        # Zero or negative new samples falls back to min_steps
        assert compute_adaptive_train_steps(new_samples=0, min_steps=25) == 25
        assert compute_adaptive_train_steps(new_samples=-5, min_steps=25) == 25

    def test_split_games_train_val_no_data_leakage(self):
        """Verifies whole-game partitioning without sample-level leakage."""
        games = []
        for i in range(10):
            sample = (
                np.zeros((17, 9, 9), dtype=np.float32),
                np.ones(82, dtype=np.float32) / 82,
                1.0 if i % 2 == 0 else -1.0,
            )
            gr = GameResult(
                samples=[sample],
                winner=1,
                black_score=10.0,
                white_score=0.0,
                moves=1,
                end_reason="two_passes",
                seed=1000 + i,
            )
            games.append(gr)

        train_games, val_games = split_games_train_val(games, val_ratio=0.20, seed=42)

        assert len(train_games) == 8
        assert len(val_games) == 2

        train_seeds = {g.seed for g in train_games}
        val_seeds = {g.seed for g in val_games}

        # Sets must be completely disjoint
        assert len(train_seeds.intersection(val_seeds)) == 0

    def test_trainer_train_iteration_adaptive_steps(self):
        """Verifies train_iteration executes exact adaptive steps and reports effective epochs."""
        model = AlphaZeroNet(board_size=9, num_blocks=1, num_filters=16)
        trainer = Trainer(model, device=torch.device("cpu"), use_amp=False)
        buffer = ReplayBuffer(capacity=1000, board_size=9)

        # Add 100 samples
        for _ in range(100):
            buffer.push(
                np.zeros((17, 9, 9), dtype=np.uint8),
                np.ones(82, dtype=np.float32) / 82,
                1.0,
            )

        # Train with adaptive steps for 60 new samples: 60 * 0.5 = 30 steps
        metrics = trainer.train_iteration(
            buffer,
            batch_size=16,
            adaptive_steps=True,
            new_samples=60,
            steps_per_sample=0.5,
            min_steps=10,
            max_steps=200,
        )

        assert metrics["actual_steps"] == 30
        # effective_epochs = 30 * 16 / 100 = 4.8
        assert pytest.approx(metrics["effective_epochs"], 1e-2) == 4.8
        assert "total_loss" in metrics

    def test_trainer_evaluate_held_out_metrics(self):
        """Verifies held-out validation evaluation computes loss, KL, entropy, and calibration."""
        model = AlphaZeroNet(board_size=9, num_blocks=1, num_filters=16)
        trainer = Trainer(model, device=torch.device("cpu"), use_amp=False)
        val_buffer = ReplayBuffer(capacity=500, board_size=9)

        for _ in range(40):
            pi = np.zeros(82, dtype=np.float32)
            pi[0] = 0.5
            pi[1] = 0.5
            val_buffer.push(
                np.zeros((17, 9, 9), dtype=np.uint8),
                pi,
                1.0,
            )

        val_metrics = trainer.evaluate(val_buffer, batch_size=16)

        assert "val_loss" in val_metrics
        assert "val_policy_loss" in val_metrics
        assert "val_value_loss" in val_metrics
        assert "policy_kl" in val_metrics
        assert "policy_entropy" in val_metrics
        assert "value_calibration" in val_metrics

        assert val_metrics["val_loss"] > 0.0
        assert val_metrics["policy_kl"] >= 0.0
        assert val_metrics["policy_entropy"] >= 0.0

    def test_trainer_evaluate_empty_buffer_returns_zeros(self):
        """Verifies evaluate handles an empty validation buffer gracefully."""
        model = AlphaZeroNet(board_size=9, num_blocks=1, num_filters=16)
        trainer = Trainer(model, device=torch.device("cpu"), use_amp=False)
        empty_buffer = ReplayBuffer(capacity=100, board_size=9)

        metrics = trainer.evaluate(empty_buffer)
        assert metrics["val_loss"] == 0.0
        assert metrics["policy_kl"] == 0.0

    def test_pipeline_config_includes_adaptive_training_flags(self):
        """Verifies PipelineConfig supports compute-aware training parameters."""
        config = PipelineConfig(
            adaptive_train_steps=True,
            val_ratio=0.15,
            steps_per_sample=0.4,
            min_train_steps=25,
            max_train_steps=300,
        )
        assert config.adaptive_train_steps is True
        assert config.val_ratio == 0.15
        assert config.steps_per_sample == 0.4
        assert config.min_train_steps == 25
        assert config.max_train_steps == 300
