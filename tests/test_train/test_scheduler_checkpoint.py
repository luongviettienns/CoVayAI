"""Tests for global LR scheduler progression and atomic checkpoint persistence."""

import os
import tempfile
from unittest.mock import MagicMock
import pytest
import torch

from main import AlphaZeroPipeline, PipelineConfig
from arena.match import ArenaResult


def test_scheduler_advances_after_multiple_rejections():
    """Verify that global scheduler advances its epoch and drops LR at milestones even when candidates are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            board_size=5,
            num_res_blocks=1,
            num_filters=16,
            num_sims=2,
            games_per_iter=1,
            train_steps=1,
            batch_size=4,
            arena_games=2,
            win_threshold=0.55,
            checkpoint_dir=tmpdir,
            device="cpu",
            progress_bar=False,
            resume=False,
            max_moves=2,
        )

        pipeline = AlphaZeroPipeline(config)
        # Override scheduler with milestones at [2, 4] and initial LR 0.001
        pipeline.trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            pipeline.trainer.optimizer, milestones=[2, 4], gamma=0.1
        )

        # Mock arena to ALWAYS reject candidate
        pipeline.arena.evaluate = MagicMock(return_value=ArenaResult(
            candidate_wins=0, baseline_wins=2, draws=0,
            candidate_win_rate=0.0, games_played=2, accepted=False
        ))

        # Iteration 1: Rejected
        metrics_1 = pipeline.run_iteration(iteration=1)
        assert not metrics_1["accepted"]
        assert pipeline.trainer.scheduler.last_epoch == 1
        assert pipeline.trainer.get_lr() == pytest.approx(0.001)

        # Iteration 2: Rejected again, but should reach milestone 2 and decay LR to 0.0001
        metrics_2 = pipeline.run_iteration(iteration=2)
        assert not metrics_2["accepted"]
        assert pipeline.trainer.scheduler.last_epoch == 2
        assert pipeline.trainer.get_lr() == pytest.approx(0.0001)

        # Iteration 3: Rejected again, last_epoch should be 3
        metrics_3 = pipeline.run_iteration(iteration=3)
        assert not metrics_3["accepted"]
        assert pipeline.trainer.scheduler.last_epoch == 3
        assert pipeline.trainer.get_lr() == pytest.approx(0.0001)


def test_scheduler_survives_accept_reject_accept():
    """Verify scheduler progression across alternating accept and reject decisions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            board_size=5,
            num_res_blocks=1,
            num_filters=16,
            num_sims=2,
            games_per_iter=1,
            train_steps=1,
            batch_size=4,
            arena_games=2,
            win_threshold=0.55,
            checkpoint_dir=tmpdir,
            device="cpu",
            progress_bar=False,
            resume=False,
            max_moves=2,
        )

        pipeline = AlphaZeroPipeline(config)
        pipeline.trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            pipeline.trainer.optimizer, milestones=[2], gamma=0.1
        )

        # Iteration 1: Accept
        pipeline.arena.evaluate = MagicMock(return_value=ArenaResult(
            candidate_wins=2, baseline_wins=0, draws=0, candidate_win_rate=1.0, games_played=2, accepted=True
        ))
        pipeline.run_iteration(iteration=1)
        assert pipeline.trainer.scheduler.last_epoch == 1
        assert pipeline.trainer.get_lr() == pytest.approx(0.001)

        # Iteration 2: Reject
        pipeline.arena.evaluate = MagicMock(return_value=ArenaResult(
            candidate_wins=0, baseline_wins=2, draws=0, candidate_win_rate=0.0, games_played=2, accepted=False
        ))
        pipeline.run_iteration(iteration=2)
        assert pipeline.trainer.scheduler.last_epoch == 2
        assert pipeline.trainer.get_lr() == pytest.approx(0.0001)

        # Iteration 3: Accept
        pipeline.arena.evaluate = MagicMock(return_value=ArenaResult(
            candidate_wins=2, baseline_wins=0, draws=0, candidate_win_rate=1.0, games_played=2, accepted=True
        ))
        pipeline.run_iteration(iteration=3)
        assert pipeline.trainer.scheduler.last_epoch == 3
        assert pipeline.trainer.get_lr() == pytest.approx(0.0001)


def test_resume_with_manifest_and_reject_architecture_mismatch():
    """Verify that latest_manifest.json is saved atomically and validates architectural compatibility on resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            board_size=5,
            num_res_blocks=1,
            num_filters=16,
            num_sims=2,
            games_per_iter=1,
            train_steps=1,
            batch_size=4,
            arena_games=2,
            win_threshold=0.55,
            checkpoint_dir=tmpdir,
            device="cpu",
            progress_bar=False,
            resume=False,
            max_moves=2,
        )

        pipeline = AlphaZeroPipeline(config)
        pipeline.run_iteration(iteration=1)

        manifest_path = os.path.join(tmpdir, "latest_manifest.json")
        assert os.path.exists(manifest_path)

        # 1. Compatible resume: should load iteration 1 and advance start_iteration to 2
        resume_cfg = PipelineConfig(
            board_size=5,
            num_res_blocks=1,
            num_filters=16,
            num_sims=2,
            games_per_iter=1,
            train_steps=1,
            batch_size=4,
            arena_games=2,
            checkpoint_dir=tmpdir,
            device="cpu",
            progress_bar=False,
            resume=True,
            max_moves=2,
        )
        resumed = AlphaZeroPipeline(resume_cfg)
        assert resumed.start_iteration == 2

        # 2. Incompatible architecture resume: different board size must raise ValueError
        incompatible_cfg = PipelineConfig(
            board_size=9,
            num_res_blocks=1,
            num_filters=16,
            checkpoint_dir=tmpdir,
            device="cpu",
            resume=True,
        )
        with pytest.raises(ValueError, match="architecture mismatch"):
            AlphaZeroPipeline(incompatible_cfg)


def test_no_resume_existing_dir_raises_without_force_new():
    """Verify that --no-resume on an existing directory refuses to run unless force_new=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an artifact in tmpdir
        with open(os.path.join(tmpdir, "best_model.pt"), "w") as f:
            f.write("dummy")

        cfg = PipelineConfig(
            board_size=5,
            checkpoint_dir=tmpdir,
            resume=False,
            force_new=False,
        )
        with pytest.raises(FileExistsError, match="already contains existing artifacts"):
            AlphaZeroPipeline(cfg)

        # With force_new=True, it should proceed without error
        cfg_force = PipelineConfig(
            board_size=5,
            checkpoint_dir=tmpdir,
            resume=False,
            force_new=True,
        )
        # Should initialize successfully
        pipeline = AlphaZeroPipeline(cfg_force)
        assert pipeline.start_iteration == 1


def test_config_validation_bounds():
    """Verify that invalid hyperparameters are rejected by validate()."""
    with pytest.raises(ValueError, match="board_size"):
        PipelineConfig(board_size=2).validate()

    with pytest.raises(ValueError, match="games_per_iter"):
        PipelineConfig(games_per_iter=0).validate()

    with pytest.raises(ValueError, match="num_sims"):
        PipelineConfig(num_sims=0).validate()

    with pytest.raises(ValueError, match="batch_size"):
        PipelineConfig(batch_size=0).validate()

    with pytest.raises(ValueError, match="train_steps"):
        PipelineConfig(train_steps=0).validate()

    with pytest.raises(ValueError, match="resign_disable_frac"):
        PipelineConfig(resign_disable_frac=1.5).validate()

    with pytest.raises(ValueError, match="arena_games"):
        PipelineConfig(arena_games=0).validate()

