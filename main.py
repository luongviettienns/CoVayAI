"""Main AlphaZero Training Pipeline Orchestrator for CoVayAI.

Coordinates the end-to-end AlphaZero loop:
  1. Self-Play: Generate training data using current best model (with Live Progress Bar)
  2. Dynamic Training: Train candidate model with dynamic step scaling (anti-overfitting on early buffer)
  3. Arena Evaluation: Head-to-head match between candidate and best models (with opening randomization)
  4. Model Promotion & Optimizer Sync: Sync optimizer momentum & scheduler on promotion/rollback (with deepcopy)
  5. Checkpoint & Resume: Auto-saves buffer.pkl once per iter, ratings.json, and supports seamless resuming
"""

import os
import re
import copy
import json
import time
import uuid
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np
import torch

from nn.model import AlphaZeroNet
from selfplay.replay_buffer import ReplayBuffer
from selfplay.worker import generate_selfplay_games
from train.trainer import Trainer
from arena.match import Arena, ArenaResult
from arena.elo import EloTracker
import config as default_cfg


def get_default_komi(board_size: int) -> float:
    """Return default komi compensation based on board size (0.5 for <=7x7, 7.5 for >=9x9)."""
    return 0.5 if board_size <= 7 else 7.5


def get_default_max_moves(board_size: int) -> int:
    """Return default maximum moves cap (2 * N^2) to prevent infinite games."""
    return 2 * board_size * board_size


def _atomic_torch_save(obj: Any, filepath: str) -> None:
    """Save PyTorch object to temporary file then atomically replace target."""
    tmp_path = filepath + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, filepath)


def _atomic_json_save(data: dict, filepath: str) -> None:
    """Save JSON dictionary to temporary file then atomically replace target."""
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, filepath)


def validate_checkpoint_compatibility(manifest: dict, current_cfg: "PipelineConfig") -> None:
    """Ensure that the checkpoint being resumed is structurally compatible with current config."""
    m_size = manifest.get("board_size")
    if m_size is not None and m_size != current_cfg.board_size:
        raise ValueError(
            f"Checkpoint architecture mismatch: checkpoint was created for board_size={m_size}, "
            f"but current run requested board_size={current_cfg.board_size}."
        )
    m_blocks = manifest.get("num_res_blocks")
    if m_blocks is not None and m_blocks != current_cfg.num_res_blocks:
        raise ValueError(
            f"Checkpoint architecture mismatch: checkpoint has num_res_blocks={m_blocks}, "
            f"current config has {current_cfg.num_res_blocks}."
        )
    m_filters = manifest.get("num_filters")
    if m_filters is not None and m_filters != current_cfg.num_filters:
        raise ValueError(
            f"Checkpoint architecture mismatch: checkpoint has num_filters={m_filters}, "
            f"current config has {current_cfg.num_filters}."
        )


@dataclass
class PipelineConfig:
    """Hyperparameters and runtime configuration for the AlphaZero pipeline."""

    board_size: int = default_cfg.BOARD_SIZE
    komi: Optional[float] = None
    max_moves: Optional[int] = None
    num_res_blocks: int = default_cfg.NUM_BLOCKS
    num_filters: int = default_cfg.NUM_FILTERS
    use_amp: bool = default_cfg.USE_AMP

    # MCTS & Self-Play
    num_sims: int = default_cfg.NUM_SIMS
    c_puct: float = default_cfg.C_PUCT
    dirichlet_alpha: float = default_cfg.DIRICHLET_ALPHA
    dirichlet_eps: float = default_cfg.DIRICHLET_EPS
    temp_moves: int = default_cfg.TEMP_MOVES
    resign_threshold: float = default_cfg.RESIGN_THRESHOLD
    resign_disable_frac: float = default_cfg.RESIGN_DISABLE_FRAC
    games_per_iter: int = default_cfg.GAMES_PER_ITER
    adjudicate_truncated: Optional[bool] = None
    num_workers: int = 1
    infer_batch_size: int = default_cfg.INFER_BATCH_SIZE
    infer_max_wait_ms: float = 2.0
    playout_cap_randomization: bool = default_cfg.PLAYOUT_CAP_RANDOMIZATION
    fast_sims: int = default_cfg.FAST_SIMS
    full_search_prob: float = default_cfg.FULL_SEARCH_PROB
    train_only_full_search: bool = default_cfg.TRAIN_ONLY_FULL_SEARCH

    # Training
    optimizer: str = default_cfg.OPTIMIZER
    batch_size: int = default_cfg.BATCH_SIZE
    train_steps: int = default_cfg.TRAIN_STEPS
    lr: float = default_cfg.LR
    weight_decay: float = default_cfg.WEIGHT_DECAY
    buffer_size: int = default_cfg.BUFFER_SIZE
    augment: bool = default_cfg.AUGMENT
    num_iters: int = default_cfg.NUM_ITERS

    # Arena & Gating
    arena_games: int = default_cfg.ARENA_GAMES
    arena_sims: int = default_cfg.ARENA_SIMS
    opening_plies: int = default_cfg.OPENING_PLIES
    win_threshold: float = default_cfg.WIN_THRESHOLD
    gating: bool = True

    # Runtime, Tracking & Checkpoints
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    progress_bar: bool = True
    resume: bool = True
    force_new: bool = False
    seed: Optional[int] = None
    run_id: Optional[str] = None

    def validate(self) -> None:
        """Validate pipeline configuration bounds."""
        if self.board_size < 3:
            raise ValueError(f"board_size must be >= 3, got {self.board_size}")
        if self.games_per_iter < 1:
            raise ValueError(f"games_per_iter must be >= 1, got {self.games_per_iter}")
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        if self.infer_batch_size < 1:
            raise ValueError(f"infer_batch_size must be >= 1, got {self.infer_batch_size}")
        if self.num_sims < 1:
            raise ValueError(f"num_sims must be >= 1, got {self.num_sims}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.train_steps < 1:
            raise ValueError(f"train_steps must be >= 1, got {self.train_steps}")
        if not (0.0 <= self.resign_disable_frac <= 1.0):
            raise ValueError(f"resign_disable_frac must be between 0 and 1, got {self.resign_disable_frac}")
        if self.arena_games < 1:
            raise ValueError(f"arena_games must be >= 1, got {self.arena_games}")


class AlphaZeroPipeline:
    """Orchestrates Self-Play, Training, Arena, and Model Promotion with live tracking and resume."""

    def __init__(self, cfg: PipelineConfig):
        self.config = cfg
        self.config.validate()

        # Deterministic seed setting if configured
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        self.run_id = self.config.run_id or f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.device = torch.device(cfg.device)

        # Dynamic defaults based on board size
        if self.config.komi is None:
            self.config.komi = get_default_komi(self.config.board_size)
        if self.config.max_moves is None:
            self.config.max_moves = get_default_max_moves(self.config.board_size)

        # Create checkpoint directory if not exists
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        manifest_path = os.path.join(self.config.checkpoint_dir, "latest_manifest.json")
        buffer_path = os.path.join(self.config.checkpoint_dir, "buffer.pkl")
        ratings_path = os.path.join(self.config.checkpoint_dir, "ratings.json")
        best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")

        # Refuse accidental overwrite on --no-resume unless --force-new is specified
        if not self.config.resume and not self.config.force_new:
            has_artifacts = os.path.exists(manifest_path) or os.path.exists(best_model_path)
            if not has_artifacts and os.path.exists(self.config.checkpoint_dir):
                for fname in os.listdir(self.config.checkpoint_dir):
                    if fname.endswith(".pt") or fname.endswith(".pkl") or fname.endswith(".json"):
                        has_artifacts = True
                        break
            if has_artifacts:
                raise FileExistsError(
                    f"Checkpoint directory '{self.config.checkpoint_dir}' already contains existing artifacts. "
                    f"To prevent data corruption, specify a clean checkpoint directory, pass --resume, or use --force-new."
                )

        # Initialize Best and Candidate models
        self.best_model = AlphaZeroNet(
            board_size=cfg.board_size,
            num_res_blocks=cfg.num_res_blocks,
            num_filters=cfg.num_filters,
        ).to(self.device)

        self.candidate_model = AlphaZeroNet(
            board_size=cfg.board_size,
            num_res_blocks=cfg.num_res_blocks,
            num_filters=cfg.num_filters,
        ).to(self.device)

        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=cfg.buffer_size, board_size=cfg.board_size)

        # Trainer for candidate model
        self.trainer = Trainer(
            model=self.candidate_model,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            optimizer_type=cfg.optimizer,
            milestones=default_cfg.LR_MILESTONES,
            lr_gamma=default_cfg.LR_GAMMA,
            device=self.device,
            use_amp=cfg.use_amp and self.device.type == "cuda",
        )

        # Elo Tracker
        self.elo_tracker = EloTracker(initial_rating=1000.0)

        # Determine starting iteration and resume state if requested
        self.start_iteration = 1
        self.current_best_id = "iter_0"
        self.best_optimizer_state = copy.deepcopy(self.trainer.optimizer.state_dict())
        self.best_scheduler_state = copy.deepcopy(self.trainer.scheduler.state_dict())

        if self.config.resume:
            # 1. First check atomic manifest (single source of truth)
            manifest_loaded = False
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                validate_checkpoint_compatibility(manifest, self.config)
                self.start_iteration = manifest["iteration"] + 1
                self.current_best_id = manifest.get("best_model_id", "iter_0")
                if "scheduler_state_dict" in manifest:
                    self.trainer.scheduler.load_state_dict(manifest["scheduler_state_dict"])
                    self.best_scheduler_state = copy.deepcopy(manifest["scheduler_state_dict"])
                manifest_loaded = True
                print(f"[+] Loaded atomic manifest from {manifest_path} (Iteration {manifest['iteration']}, Best: {self.current_best_id})")

            # 2. Resume Elo ratings
            if os.path.exists(ratings_path):
                try:
                    self.elo_tracker = EloTracker.load(ratings_path)
                    print(f"[+] Restored Elo tracker from {ratings_path}")
                except Exception as e:
                    print(f"[!] Warning reading ratings.json: {e}")

            # 3. Resume Replay Buffer
            if os.path.exists(buffer_path):
                try:
                    self.buffer.load(buffer_path)
                    print(f"[+] Restored Replay Buffer with {len(self.buffer)} samples from {buffer_path}")
                except Exception as e:
                    print(f"[!] Warning reading buffer.pkl: {e}")

            # 4. Fallback iteration detection if manifest was absent
            if not manifest_loaded:
                iter_numbers = []
                if os.path.exists(self.config.checkpoint_dir):
                    for fname in os.listdir(self.config.checkpoint_dir):
                        match = re.match(r"checkpoint_iter_(\d+)\.pt", fname)
                        if match:
                            iter_numbers.append(int(match.group(1)))

                if iter_numbers:
                    max_iter = max(iter_numbers)
                    self.start_iteration = max_iter + 1

            # 5. Load best model weights and optimizer state
            if os.path.exists(best_model_path):
                ckpt = torch.load(best_model_path, map_location=self.device)
                if not manifest_loaded:
                    best_iter = ckpt.get("iteration", 0) if isinstance(ckpt, dict) else 0
                    self.current_best_id = f"iter_{best_iter}"
                state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                self.best_model.load_state_dict(state)
                if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
                    self.best_optimizer_state = copy.deepcopy(ckpt["optimizer_state_dict"])
                if not manifest_loaded and isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
                    self.best_scheduler_state = copy.deepcopy(ckpt["scheduler_state_dict"])
                    self.trainer.scheduler.load_state_dict(self.best_scheduler_state)
                self._sync_optimizer_lr_with_scheduler()
                print(f"[+] Loaded baseline weights from {best_model_path} (Model ID: {self.current_best_id}, Elo: {self.elo_tracker.get_rating(self.current_best_id):.1f})")

        # Always ensure best_model.pt exists on disk as baseline (reset if force_new with no-resume)
        if not self.config.resume and self.config.force_new:
            _atomic_torch_save(
                {
                    "iteration": 0,
                    "model_state_dict": self.best_model.state_dict(),
                    "optimizer_state_dict": copy.deepcopy(self.best_optimizer_state),
                    "scheduler_state_dict": copy.deepcopy(self.best_scheduler_state),
                    "scaler_state_dict": self.trainer.scaler.state_dict(),
                    "global_step": 0,
                    "elo": 1000.0,
                    "board_size": self.config.board_size,
                    "num_res_blocks": self.config.num_res_blocks,
                    "num_filters": self.config.num_filters,
                },
                best_model_path,
            )
        elif not os.path.exists(best_model_path):
            _atomic_torch_save(
                {
                    "iteration": 0,
                    "model_state_dict": self.best_model.state_dict(),
                    "optimizer_state_dict": copy.deepcopy(self.best_optimizer_state),
                    "scheduler_state_dict": copy.deepcopy(self.best_scheduler_state),
                    "scaler_state_dict": self.trainer.scaler.state_dict(),
                    "global_step": 0,
                    "elo": 1000.0,
                    "board_size": self.config.board_size,
                    "num_res_blocks": self.config.num_res_blocks,
                    "num_filters": self.config.num_filters,
                },
                best_model_path,
            )

        # Candidate starts with identical weights and deepcopied optimizer/scheduler state
        self.candidate_model.load_state_dict(self.best_model.state_dict())
        self.trainer.optimizer.load_state_dict(copy.deepcopy(self.best_optimizer_state))
        self._sync_optimizer_lr_with_scheduler()

        # Arena
        self.arena = Arena(
            board_size=self.config.board_size,
            komi=self.config.komi,
            num_sims=self.config.arena_sims,
            c_puct=self.config.c_puct,
            max_moves=self.config.max_moves,
            win_threshold=self.config.win_threshold,
            resign_threshold=self.config.resign_threshold,
            opening_plies=self.config.opening_plies,
            device=self.device,
        )

    def _sync_optimizer_lr_with_scheduler(self) -> None:
        """Synchronize the current learning rate from global scheduler into optimizer param groups."""
        current_lrs = self.trainer.scheduler.get_last_lr()
        for param_group, lr in zip(self.trainer.optimizer.param_groups, current_lrs):
            param_group["lr"] = lr

    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Execute one complete AlphaZero iteration with dynamic training steps and optimizer synchronization.

        1. Self-Play -> buffer (saved to buffer.pkl once per iteration)
        2. Dynamic Train candidate model (sample-to-train ratio controlled to prevent overfitting)
        3. Arena: candidate vs best model (with gating check & opening plies)
        4. Promote candidate or Rollback model + optimizer/scheduler states with deepcopy
        """
        print(f"\n{'='*30} ITERATION {iteration} {'='*30}")

        # 1. Self-play
        print(f"[*] Step 1/3: Self-play ({self.config.games_per_iter} games, {self.config.num_sims} sims, komi {self.config.komi})...")
        self.best_model.eval()

        buffer_path = os.path.join(self.config.checkpoint_dir, "buffer.pkl")

        def on_game_finish(game_idx, samples):
            # Push game to buffer in memory
            self.buffer.push_game(samples)

        new_samples = generate_selfplay_games(
            model=self.best_model,
            num_games=self.config.games_per_iter,
            board_size=self.config.board_size,
            num_simulations=self.config.num_sims,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_eps=self.config.dirichlet_eps,
            temp_moves=self.config.temp_moves,
            resign_threshold=self.config.resign_threshold,
            resign_disable_frac=self.config.resign_disable_frac,
            max_moves=self.config.max_moves,
            komi=self.config.komi,
            device=self.device,
            progress_bar=self.config.progress_bar,
            game_callback=on_game_finish,
            adjudicate_truncated=self.config.adjudicate_truncated,
            num_workers=self.config.num_workers,
            max_batch_size=self.config.infer_batch_size,
            max_wait_ms=self.config.infer_max_wait_ms,
            use_amp=self.config.use_amp and self.device.type == "cuda",
            playout_cap_randomization=self.config.playout_cap_randomization,
            fast_sims=self.config.fast_sims,
            full_search_prob=self.config.full_search_prob,
            train_only_full_search=self.config.train_only_full_search,
        )

        # Save buffer once per iteration to avoid excessive I/O bottleneck
        self.buffer.save(buffer_path)
        print(f"    [+] Added {len(new_samples)} samples. Total Replay Buffer: {len(self.buffer)} samples (Saved to disk).")

        # 2. Dynamic Training: prevent early overfitting when buffer is small
        max_allowed_steps = max(20, (len(self.buffer) * 2) // self.config.batch_size)
        effective_steps = min(self.config.train_steps, max_allowed_steps)
        print(f"[*] Step 2/3: Dynamic Training ({effective_steps} steps [cap: {self.config.train_steps}], batch size {self.config.batch_size})...")

        # Ensure candidate starts from current best model weights and deepcopied optimizer momentum
        self.candidate_model.load_state_dict(self.best_model.state_dict())
        self.trainer.optimizer.load_state_dict(copy.deepcopy(self.best_optimizer_state))
        self._sync_optimizer_lr_with_scheduler()

        train_metrics = self.trainer.train_iteration(
            replay_buffer=self.buffer,
            steps=effective_steps,
            batch_size=self.config.batch_size,
            augment=self.config.augment,
            progress_bar=self.config.progress_bar,
        )
        avg_loss = train_metrics["total_loss"]
        avg_pi_loss = train_metrics["policy_loss"]
        avg_v_loss = train_metrics["value_loss"]
        print(f"    [+] Train Loss: {avg_loss:.4f} (Policy: {avg_pi_loss:.4f}, Value: {avg_v_loss:.4f})")

        # 3. Arena Evaluation
        cand_id = f"iter_{iteration}"
        best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")

        if self.config.gating:
            print(f"[*] Step 3/3: Arena Evaluation ({self.config.arena_games} games vs best model, opening_plies {self.config.opening_plies}, sims {self.config.arena_sims})...")
            arena_result: ArenaResult = self.arena.evaluate(
                candidate_model=self.candidate_model,
                baseline_model=self.best_model,
                num_games=self.config.arena_games,
                progress_bar=self.config.progress_bar,
            )

            self.elo_tracker.record_match(
                model_a_id=cand_id,
                model_b_id=self.current_best_id,
                wins_a=arena_result.candidate_wins,
                wins_b=arena_result.baseline_wins,
                draws=arena_result.draws,
            )
            cand_elo = self.elo_tracker.get_rating(cand_id)
            best_elo = self.elo_tracker.get_rating(self.current_best_id)

            ci_str = ""
            if arena_result.win_rate_ci:
                ci_low, ci_high = arena_result.win_rate_ci
                ci_str = f" [95% CI: {ci_low*100:.1f}% - {ci_high*100:.1f}%]"

            print(
                f"    Result: {arena_result.candidate_wins}W - {arena_result.baseline_wins}L - {arena_result.draws}D "
                f"({arena_result.candidate_win_rate * 100:.1f}%{ci_str})"
            )
            print(
                f"    By Color: Black {arena_result.candidate_as_black_wins}W - White {arena_result.candidate_as_white_wins}W | "
                f"Candidate Elo: {cand_elo:.1f} | Best Elo: {best_elo:.1f}"
            )

            win_rate = arena_result.candidate_win_rate
            accepted = arena_result.accepted
        else:
            # Continuous AlphaZero (2018): promote candidate directly without gating
            print("    [*] Gating disabled (Continuous AlphaZero training): Candidate promoted directly.")
            cand_elo = self.elo_tracker.get_rating(self.current_best_id)
            best_elo = cand_elo
            win_rate = 1.0
            accepted = True

        # 4. Step LR Scheduler after each iteration (before saving checkpoints)
        current_lr = self.trainer.step_scheduler()
        self.best_scheduler_state = copy.deepcopy(self.trainer.scheduler.state_dict())
        print(f"    [*] Learning Rate: {current_lr:.6f}")

        # 5. Save iteration checkpoint atomically (capturing candidate's trained weights & post-step scheduler)
        iter_ckpt_path = os.path.join(
            self.config.checkpoint_dir, f"checkpoint_iter_{iteration}.pt"
        )
        self.trainer.save_checkpoint(
            filepath=iter_ckpt_path,
            iteration=iteration,
            extra={
                "accepted": accepted,
                "win_rate": win_rate,
                "candidate_elo": cand_elo,
                "best_elo": best_elo,
                "lr": current_lr,
                "board_size": self.config.board_size,
                "num_res_blocks": self.config.num_res_blocks,
                "num_filters": self.config.num_filters,
            },
        )

        # 6. Model Promotion / Rollback with Optimizer Synchronization
        if accepted:
            print(f"    [+] Candidate ACCEPTED! (Winrate {win_rate*100:.1f}% >= {self.config.win_threshold*100:.1f}%)")
            self.best_model.load_state_dict(self.candidate_model.state_dict())
            self.best_optimizer_state = copy.deepcopy(self.trainer.optimizer.state_dict())
            self.current_best_id = cand_id
            _atomic_torch_save(
                {
                    "iteration": iteration,
                    "model_state_dict": self.best_model.state_dict(),
                    "optimizer_state_dict": copy.deepcopy(self.best_optimizer_state),
                    "scheduler_state_dict": copy.deepcopy(self.best_scheduler_state),
                    "scaler_state_dict": self.trainer.scaler.state_dict(),
                    "global_step": self.trainer.global_step,
                    "elo": cand_elo,
                    "board_size": self.config.board_size,
                    "num_res_blocks": self.config.num_res_blocks,
                    "num_filters": self.config.num_filters,
                },
                best_model_path,
            )
        else:
            print(f"    [-] Candidate REJECTED. (Winrate {win_rate*100:.1f}% < {self.config.win_threshold*100:.1f}%)")
            # Rollback candidate weights AND optimizer momentum in memory; global scheduler is NOT rolled back
            self.candidate_model.load_state_dict(self.best_model.state_dict())
            self.trainer.optimizer.load_state_dict(copy.deepcopy(self.best_optimizer_state))
            self._sync_optimizer_lr_with_scheduler()

        # Save Elo state atomically
        ratings_path = os.path.join(self.config.checkpoint_dir, "ratings.json")
        self.elo_tracker.save(ratings_path)

        # Save atomic latest_manifest.json as single source of truth
        manifest_path = os.path.join(self.config.checkpoint_dir, "latest_manifest.json")
        manifest_data = {
            "schema_version": 1,
            "run_id": self.run_id,
            "iteration": iteration,
            "global_step": self.trainer.global_step,
            "board_size": self.config.board_size,
            "num_filters": self.config.num_filters,
            "num_res_blocks": self.config.num_res_blocks,
            "input_planes": 17,
            "komi": self.config.komi,
            "best_model_id": self.current_best_id,
            "best_model_path": "best_model.pt",
            "checkpoint_path": f"checkpoint_iter_{iteration}.pt",
            "candidate_accepted": accepted,
            "buffer_path": "buffer.pkl",
            "ratings_path": "ratings.json",
            "scheduler_state_dict": copy.deepcopy(self.best_scheduler_state),
            "timestamp": time.time(),
        }
        _atomic_json_save(manifest_data, manifest_path)

        return {
            "iteration": iteration,
            "selfplay_samples": len(new_samples),
            "buffer_size": len(self.buffer),
            "train_loss": avg_loss,
            "effective_train_steps": effective_steps,
            "policy_loss": avg_pi_loss,
            "value_loss": avg_v_loss,
            "arena_win_rate": win_rate,
            "arena_win_rate_ci": arena_result.win_rate_ci if self.config.gating else None,
            "arena_black_wins": arena_result.candidate_as_black_wins if self.config.gating else 0,
            "arena_white_wins": arena_result.candidate_as_white_wins if self.config.gating else 0,
            "accepted": accepted,
            "candidate_elo": cand_elo,
            "best_elo": best_elo,
            "lr": current_lr,
        }

    def train(self) -> None:
        """Run full multi-iteration training loop starting from start_iteration."""
        print(f"Starting AlphaZero training on {self.device} for {self.config.num_iters} iterations.")
        if self.start_iteration > 1:
            print(f"Resuming at iteration {self.start_iteration} up to {self.config.num_iters}.")

        for it in range(self.start_iteration, self.config.num_iters + 1):
            self.run_iteration(it)


def main() -> None:
    """CLI entry point for running the AlphaZero pipeline."""
    parser = argparse.ArgumentParser(description="CoVayAI AlphaZero Pipeline")
    parser.add_argument("--iters", type=int, default=default_cfg.NUM_ITERS, help="Total number of iterations")
    parser.add_argument("--games", type=int, default=default_cfg.GAMES_PER_ITER, help="Self-play games per iteration")
    parser.add_argument("--sims", type=int, default=default_cfg.NUM_SIMS, help="MCTS simulations per move for self-play")
    parser.add_argument("--arena-sims", type=int, default=None, help="MCTS simulations per move in Arena (defaults to --sims if not set)")
    parser.add_argument("--train-steps", type=int, default=default_cfg.TRAIN_STEPS, help="Training steps per iter")
    parser.add_argument("--batch-size", type=int, default=default_cfg.BATCH_SIZE, help="Batch size")
    parser.add_argument("--arena-games", type=int, default=default_cfg.ARENA_GAMES, help="Arena match games")
    parser.add_argument("--opening-plies", type=int, default=default_cfg.OPENING_PLIES, help="Opening plies with tau=1 in Arena")
    parser.add_argument("--board-size", type=int, default=default_cfg.BOARD_SIZE, help="Board size")
    parser.add_argument("--komi", type=float, default=None, help="Komi compensation for White (default: 0.5 for <=7x7, 7.5 for >=9x9)")
    parser.add_argument("--num-workers", type=int, default=default_cfg.NUM_WORKERS, help="Number of parallel self-play workers")
    parser.add_argument("--infer-batch-size", type=int, default=default_cfg.INFER_BATCH_SIZE, help="Maximum batch size for GPU inference server")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--resign-threshold", type=float, default=default_cfg.RESIGN_THRESHOLD, help="Threshold below which MCTS resigns")
    parser.add_argument("--resign-disable-frac", type=float, default=default_cfg.RESIGN_DISABLE_FRAC, help="Fraction of selfplay games with resignation disabled")
    parser.add_argument("--max-moves", type=int, default=None, help="Maximum moves before adjudication (default: 2*N^2)")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility")
    parser.add_argument("--force-new", action="store_true", help="Allow starting fresh run in existing checkpoint directory")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not resume from existing checkpoints")
    parser.add_argument("--playout-cap-randomization", action="store_true", default=default_cfg.PLAYOUT_CAP_RANDOMIZATION, help="Enable playout cap randomization (fast vs full MCTS searches)")
    parser.add_argument("--fast-sims", type=int, default=default_cfg.FAST_SIMS, help="Fast search simulation budget for playout cap randomization")
    parser.add_argument("--full-search-prob", type=float, default=default_cfg.FULL_SEARCH_PROB, help="Probability of full search under playout cap randomization")
    parser.add_argument("--no-progress", action="store_false", dest="progress_bar", help="Disable live progress bar")
    parser.add_argument("--no-gating", action="store_false", dest="gating", help="Disable Arena gating (Continuous AlphaZero 2018 training)")
    parser.set_defaults(resume=True, progress_bar=True, gating=True)
    args = parser.parse_args()

    arena_sims = args.arena_sims if args.arena_sims is not None else args.sims

    cfg = PipelineConfig(
        num_iters=args.iters,
        games_per_iter=args.games,
        num_sims=args.sims,
        arena_sims=arena_sims,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        arena_games=args.arena_games,
        opening_plies=args.opening_plies,
        board_size=args.board_size,
        komi=args.komi,
        max_moves=args.max_moves,
        resign_threshold=args.resign_threshold,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        progress_bar=args.progress_bar,
        resume=args.resume,
        resign_disable_frac=args.resign_disable_frac,
        gating=args.gating,
        force_new=args.force_new,
        seed=args.seed,
        num_workers=args.num_workers,
        infer_batch_size=args.infer_batch_size,
        playout_cap_randomization=args.playout_cap_randomization,
        fast_sims=args.fast_sims,
        full_search_prob=args.full_search_prob,
    )

    pipeline = AlphaZeroPipeline(cfg)
    pipeline.train()


if __name__ == "__main__":
    main()
