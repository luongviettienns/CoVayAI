"""
Tests for Cost Gates, Budget Limits, and Safe Stop Mechanism (P11 & C6).
Verifies:
- Pipeline automatic termination when exceeding max_wallclock_hours or max_budget_usd.
- Safe Stop guarantees atomic checkpoint commitment and cost_report.json generation upon exit.
- Seamless resumption after a budget-induced safe stop.
- Strict input validation on budget parameters.
"""

import json
import os
import tempfile
import time
import pytest
import torch

from main import AlphaZeroPipeline, PipelineConfig


class TestSafeStopAndBudget:
    def test_pipeline_config_budget_validation(self):
        """Verifies validation rules for budget and time limits."""
        # Valid config
        cfg = PipelineConfig(
            max_budget_usd=50.0,
            max_wallclock_hours=5.0,
            gpu_price_per_hour=0.50,
        )
        assert cfg.max_budget_usd == 50.0
        assert cfg.max_wallclock_hours == 5.0
        assert cfg.gpu_price_per_hour == 0.50

        # Negative budget must raise ValueError
        with pytest.raises(ValueError, match="max_budget_usd"):
            PipelineConfig(max_budget_usd=-10.0).validate()

        # Negative wallclock limit must raise ValueError
        with pytest.raises(ValueError, match="max_wallclock_hours"):
            PipelineConfig(max_wallclock_hours=-1.0).validate()

        # Negative gpu price must raise ValueError
        with pytest.raises(ValueError, match="gpu_price_per_hour"):
            PipelineConfig(gpu_price_per_hour=-0.25).validate()

    def test_check_budget_limits_trigger_safe_stop(self):
        """Verifies check_budget_limits flags stop when cost or time limit is exceeded."""
        cfg = PipelineConfig(
            board_size=5,
            max_budget_usd=1.0,
            max_wallclock_hours=0.001,  # ~3.6s
            gpu_price_per_hour=10.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.checkpoint_dir = tmpdir
            pipeline = AlphaZeroPipeline(cfg)

            # Initially start time is now, elapsed ~0, cost ~0
            should_stop, reason = pipeline.check_budget_limits()
            assert not should_stop
            assert reason is None

            # Simulate passage of time past wallclock limit
            pipeline.start_time = time.time() - 10.0  # 10s elapsed > 3.6s
            should_stop, reason = pipeline.check_budget_limits()
            assert should_stop
            assert "max_wallclock_hours" in reason

            # Simulate cost limit exceeded
            pipeline.config.max_wallclock_hours = 100.0  # disable time trigger
            pipeline.config.max_budget_usd = 0.05
            # 10s at $10/h = 10 * (10/3600) = $0.0277...
            # 100s at $10/h = 100 * (10/3600) = $0.277 > $0.05
            pipeline.start_time = time.time() - 100.0
            should_stop, reason = pipeline.check_budget_limits()
            assert should_stop
            assert "max_budget_usd" in reason

    def test_safe_stop_saves_checkpoint_and_cost_report(self):
        """Verifies that when safe stop is triggered, checkpoint and cost_report.json are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PipelineConfig(
                board_size=5,
                num_res_blocks=1,
                num_filters=16,
                num_sims=2,
                games_per_iter=1,
                train_steps=1,
                arena_games=1,
                max_moves=4,
                num_iters=10,
                checkpoint_dir=tmpdir,
                progress_bar=False,
                max_wallclock_hours=0.00001,  # very small limit to trigger immediate safe stop after iter 1
                gpu_price_per_hour=0.50,
                device="cpu",
            )
            pipeline = AlphaZeroPipeline(cfg)
            pipeline.train()

            # Pipeline should have stopped early at iteration 1 instead of running to 10
            manifest_path = os.path.join(tmpdir, "latest_manifest.json")
            assert os.path.exists(manifest_path)
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["iteration"] == 1

            # Cost report JSON should be saved
            cost_report_path = os.path.join(tmpdir, "cost_report.json")
            assert os.path.exists(cost_report_path)
            with open(cost_report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            assert "metrics" in report
            assert report["metrics"]["board_size"] == 5
            assert "total_compute_cost" in report["metrics"]

    def test_resume_after_safe_stop(self):
        """Verifies resuming after a safe stop continues correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run 1: safe stop at iter 1
            cfg1 = PipelineConfig(
                board_size=5,
                num_res_blocks=1,
                num_filters=16,
                num_sims=2,
                games_per_iter=1,
                train_steps=1,
                arena_games=1,
                max_moves=4,
                num_iters=5,
                checkpoint_dir=tmpdir,
                progress_bar=False,
                max_wallclock_hours=0.00001,  # stops after iter 1
                device="cpu",
            )
            p1 = AlphaZeroPipeline(cfg1)
            p1.train()

            # Run 2: resume with higher limit up to 2 iterations
            cfg2 = PipelineConfig(
                board_size=5,
                num_res_blocks=1,
                num_filters=16,
                num_sims=2,
                games_per_iter=1,
                train_steps=1,
                arena_games=1,
                max_moves=4,
                num_iters=2,
                checkpoint_dir=tmpdir,
                progress_bar=False,
                max_wallclock_hours=1.0,  # 1 hour limit, allows iter 2 to run
                device="cpu",
                resume=True,
            )
            p2 = AlphaZeroPipeline(cfg2)
            assert p2.start_iteration == 2
            p2.train()

            manifest_path = os.path.join(tmpdir, "latest_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["iteration"] == 2
