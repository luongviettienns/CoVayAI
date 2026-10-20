"""End-to-end pipeline integration test for AlphaZero loop."""

import os
import tempfile
import pytest
import torch

from main import AlphaZeroPipeline, PipelineConfig


def test_pipeline_single_iteration_and_resume():
    """Verify that a complete iteration runs, saves buffer.pkl, and resumes seamlessly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            board_size=9,
            num_res_blocks=1,
            num_filters=16,
            num_sims=3,
            games_per_iter=1,
            train_steps=2,
            batch_size=8,
            arena_games=2,
            win_threshold=0.55,
            checkpoint_dir=tmpdir,
            device="cpu",
            progress_bar=False,
            resume=True,
            max_moves=4,
        )

        pipeline = AlphaZeroPipeline(config)
        assert pipeline.start_iteration == 1

        metrics = pipeline.run_iteration(iteration=1)

        assert "iteration" in metrics
        assert metrics["iteration"] == 1
        assert "selfplay_samples" in metrics
        assert metrics["selfplay_samples"] > 0
        assert "train_loss" in metrics
        assert "arena_win_rate" in metrics
        assert "accepted" in metrics
        assert "best_elo" in metrics

        # Verify all persistence artifacts exist
        assert os.path.exists(os.path.join(tmpdir, "ratings.json"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_iter_1.pt"))
        assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))
        assert os.path.exists(os.path.join(tmpdir, "buffer.pkl"))

        # Test resume functionality: create second pipeline instance pointing to same tmpdir
        resumed_pipeline = AlphaZeroPipeline(config)
        assert resumed_pipeline.start_iteration == 2
        assert len(resumed_pipeline.buffer) == metrics["selfplay_samples"]
        expected_best_id = "iter_1" if metrics["accepted"] else "iter_0"
        assert resumed_pipeline.current_best_id == expected_best_id
