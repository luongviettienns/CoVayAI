"""End-to-end pipeline integration test for AlphaZero loop."""

import os
import tempfile
import pytest
import torch

from main import AlphaZeroPipeline, PipelineConfig


def test_pipeline_single_iteration():
    """Verify that a complete miniature iteration runs end-to-end without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            board_size=9,
            num_res_blocks=1,
            num_filters=16,
            num_sims=5,
            games_per_iter=1,
            train_steps=2,
            batch_size=8,
            arena_games=2,
            win_threshold=0.55,
            checkpoint_dir=tmpdir,
            device="cpu",
        )

        pipeline = AlphaZeroPipeline(config)
        metrics = pipeline.run_iteration(iteration=1)

        assert "iteration" in metrics
        assert metrics["iteration"] == 1
        assert "selfplay_samples" in metrics
        assert metrics["selfplay_samples"] > 0
        assert "train_loss" in metrics
        assert "arena_win_rate" in metrics
        assert "accepted" in metrics
        assert "best_elo" in metrics

        # Verify artifacts were created in checkpoint_dir
        assert os.path.exists(os.path.join(tmpdir, "ratings.json"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_iter_1.pt"))
        assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))
