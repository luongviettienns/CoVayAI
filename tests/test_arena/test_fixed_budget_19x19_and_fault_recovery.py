"""Tests for PR 11: Fixed-Budget 19x19 Evaluation and Fault Recovery (Phase P7)."""

import json
import os
import shutil
import numpy as np
import pytest
import torch

from arena.baseline import evaluate_against_random, RandomModel
from arena.match import Arena, ArenaResult, play_arena_game
from main import AlphaZeroPipeline, PipelineConfig
from nn.model import AlphaZeroNet
from selfplay.storage import save_shard, load_shard, BufferShardStorage


class TestFixedBudget19x19AndFaultRecovery:
    """Test suite covering 19x19 evaluation, budget scaling, and fault tolerance."""

    def test_arena_fixed_budget_19x19_match(self):
        """Verify Arena plays a 19x19 match with paired openings and fixed simulation budget."""
        m1 = AlphaZeroNet(board_size=19, num_res_blocks=1, num_filters=16)
        m2 = AlphaZeroNet(board_size=19, num_res_blocks=1, num_filters=16)
        m1.eval()
        m2.eval()

        arena = Arena(
            board_size=19,
            komi=7.5,
            num_sims=5,
            max_moves=8,
            opening_plies=2,
            win_threshold=0.55,
        )

        res = arena.evaluate(
            candidate_model=m1,
            baseline_model=m2,
            num_games=2,
            base_seed=123,
        )

        assert isinstance(res, ArenaResult)
        assert res.games_played == 2
        assert res.candidate_wins + res.baseline_wins + res.draws == 2
        assert 0.0 <= res.candidate_win_rate <= 1.0
        assert res.win_rate_ci is not None
        assert 0.0 <= res.win_rate_ci[0] <= res.win_rate_ci[1] <= 1.0
        assert res.paired_results is not None
        assert len(res.paired_results) == 1

        # Check opening moves on 19x19
        opening = res.paired_results[0].opening_moves
        assert len(opening) == 2
        for move in opening:
            assert 0 <= move < 361  # 19x19 board space

    def test_arena_asymmetric_simulation_budgets(self):
        """Verify play_arena_game and Arena support asymmetric simulation budgets (e.g. 10 vs 2 sims)."""
        m1 = AlphaZeroNet(board_size=9, num_res_blocks=1, num_filters=16)
        m2 = AlphaZeroNet(board_size=9, num_res_blocks=1, num_filters=16)
        m1.eval()
        m2.eval()

        arena = Arena(
            board_size=9,
            komi=7.5,
            cand_sims=10,
            base_sims=2,
            max_moves=6,
            opening_plies=2,
        )

        res = arena.evaluate(m1, m2, num_games=2, base_seed=42)
        assert res.games_played == 2
        assert res.candidate_wins + res.baseline_wins + res.draws == 2

    def test_evaluate_against_random_19x19(self):
        """Verify AlphaZeroNet can be benchmarked against RandomModel on 19x19."""
        net = AlphaZeroNet(board_size=19, num_res_blocks=1, num_filters=16)
        net.eval()

        win_rate, wins, losses, draws = evaluate_against_random(
            model=net,
            board_size=19,
            komi=7.5,
            num_games=2,
            num_sims=5,
            max_moves=6,
        )

        assert 0.0 <= win_rate <= 1.0
        assert wins + losses + draws == 2

    def test_fault_recovery_corrupt_manifest(self, tmp_path):
        """Verify pipeline handles corrupted latest_manifest.json cleanly."""
        ckpt_dir = str(tmp_path / "corrupt_ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)

        manifest_path = os.path.join(ckpt_dir, "latest_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("CORRUPTED_JSON_{{{invalid_bytes")

        cfg = PipelineConfig(
            board_size=5,
            checkpoint_dir=ckpt_dir,
            resume=True,
            force_new=False,
        )

        # Loading corrupted manifest must be caught with clean warning or error fallback
        pipeline = AlphaZeroPipeline(cfg)
        # Should gracefully fallback or start from iteration 1 without crashing
        assert pipeline.start_iteration >= 1

    def test_fault_recovery_corrupt_shard(self, tmp_path):
        """Verify load_shard detects corrupted payload or bad checksum and raises ValueError."""
        from selfplay.replay_buffer import ReplayBuffer

        shard_path = str(tmp_path / "shard_0001.npz")
        buf = ReplayBuffer(capacity=20, board_size=5)
        for _ in range(10):
            buf.push(np.zeros((17, 5, 5), dtype=np.uint8), np.zeros(26, dtype=np.float32), 1.0)

        checksum = save_shard(shard_path, buf)
        assert os.path.exists(shard_path)

        # Corrupt the file on disk by appending garbage bytes
        with open(shard_path, "ab") as f:
            f.write(b"CORRUPT_PAYLOAD_TAMPERING")

        target_buf = ReplayBuffer(capacity=20, board_size=5)
        with pytest.raises(ValueError, match="Checksum mismatch"):
            load_shard(shard_path, buffer=target_buf, expected_checksum=checksum)

    def test_fault_recovery_orphaned_tmp_files(self, tmp_path):
        """Verify orphaned .tmp files from interrupted writes are safely ignored."""
        ckpt_dir = str(tmp_path / "interrupted_ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Create valid baseline
        net = AlphaZeroNet(board_size=5, num_res_blocks=1, num_filters=16)
        valid_model_path = os.path.join(ckpt_dir, "best_model.pt")
        torch.save(
            {
                "iteration": 1,
                "model_state_dict": net.state_dict(),
                "board_size": 5,
                "num_res_blocks": 1,
                "num_filters": 16,
            },
            valid_model_path,
        )

        # Create orphaned .tmp files
        with open(os.path.join(ckpt_dir, "best_model.pt.tmp"), "wb") as f:
            f.write(b"interrupted_write_garbage")
        with open(os.path.join(ckpt_dir, "latest_manifest.json.tmp"), "w") as f:
            f.write("incomplete_json")

        cfg = PipelineConfig(
            board_size=5,
            num_res_blocks=1,
            num_filters=16,
            checkpoint_dir=ckpt_dir,
            resume=True,
        )

        pipeline = AlphaZeroPipeline(cfg)
        assert pipeline.best_model is not None
        assert pipeline.current_best_id == "iter_1"

    def test_resume_equivalence_19x19(self, tmp_path):
        """Verify 19x19 pipeline run saves valid checkpoints and resumes with equivalent state."""
        ckpt_dir = str(tmp_path / "resume_19x19")

        cfg = PipelineConfig(
            board_size=19,
            num_res_blocks=1,
            num_filters=16,
            checkpoint_dir=ckpt_dir,
            games_per_iter=1,
            num_sims=5,
            arena_games=2,
            gating=False,
            train_steps=2,
            batch_size=2,
            max_moves=6,
            progress_bar=False,
            resume=False,
            force_new=True,
        )

        pipeline1 = AlphaZeroPipeline(cfg)
        metrics = pipeline1.run_iteration(iteration=1)
        assert metrics["iteration"] == 1

        # Check latest_manifest.json
        manifest_path = os.path.join(ckpt_dir, "latest_manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["iteration"] == 1
        assert manifest["board_size"] == 19

        # Second pipeline resumes from checkpoint
        cfg_resume = PipelineConfig(
            board_size=19,
            num_res_blocks=1,
            num_filters=16,
            checkpoint_dir=ckpt_dir,
            resume=True,
            progress_bar=False,
        )

        pipeline2 = AlphaZeroPipeline(cfg_resume)
        assert pipeline2.start_iteration == 2
        assert len(pipeline2.buffer) == len(pipeline1.buffer)

        # Model weights must match exactly
        for p1, p2 in zip(pipeline1.best_model.parameters(), pipeline2.best_model.parameters()):
            assert torch.equal(p1, p2)
