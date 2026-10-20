"""Main AlphaZero Training Pipeline Orchestrator for CoVayAI.

Coordinates the end-to-end AlphaZero loop:
  1. Self-Play: Generate training data using current best model (with Live Progress Bar)
  2. Dynamic Training: Train candidate model with dynamic step scaling (anti-overfitting on early buffer)
  3. Arena Evaluation: Head-to-head match between candidate and best models (with opening randomization)
  4. Model Promotion & Optimizer Sync: Sync optimizer momentum & scheduler on promotion/rollback
  5. Checkpoint & Resume: Auto-saves buffer.pkl, ratings.json, and supports seamless resuming
"""

import os
import re
import copy
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

from nn.model import AlphaZeroNet
from selfplay.replay_buffer import ReplayBuffer
from selfplay.worker import generate_selfplay_games
from train.trainer import Trainer
from arena.match import Arena, ArenaResult
from arena.elo import EloTracker
import config as default_cfg


@dataclass
class PipelineConfig:
    """Hyperparameters and runtime configuration for the AlphaZero pipeline."""

    board_size: int = default_cfg.BOARD_SIZE
    komi: float = default_cfg.KOMI
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
    max_moves: Optional[int] = None


class AlphaZeroPipeline:
    """Orchestrates Self-Play, Training, Arena, and Model Promotion with live tracking and resume."""

    def __init__(self, cfg: PipelineConfig):
        self.config = cfg
        self.device = torch.device(cfg.device)

        # Create checkpoint directory if not exists
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

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
        self.buffer = ReplayBuffer(capacity=cfg.buffer_size)

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

        buffer_path = os.path.join(self.config.checkpoint_dir, "buffer.pkl")
        ratings_path = os.path.join(self.config.checkpoint_dir, "ratings.json")
        best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")

        if self.config.resume:
            # 1. Resume Elo ratings
            if os.path.exists(ratings_path):
                try:
                    self.elo_tracker = EloTracker.load(ratings_path)
                    print(f"[+] Restored Elo tracker from {ratings_path}")
                except Exception as e:
                    print(f"[!] Warning reading ratings.json: {e}")

            # 2. Resume Replay Buffer
            if os.path.exists(buffer_path):
                try:
                    self.buffer.load(buffer_path)
                    print(f"[+] Restored Replay Buffer with {len(self.buffer)} samples from {buffer_path}")
                except Exception as e:
                    print(f"[!] Warning reading buffer.pkl: {e}")

            # 3. Detect latest iteration from checkpoint files
            iter_numbers = []
            if os.path.exists(self.config.checkpoint_dir):
                for fname in os.listdir(self.config.checkpoint_dir):
                    match = re.match(r"checkpoint_iter_(\d+)\.pt", fname)
                    if match:
                        iter_numbers.append(int(match.group(1)))

            if iter_numbers:
                max_iter = max(iter_numbers)
                self.start_iteration = max_iter + 1
                self.current_best_id = f"iter_{max_iter}"

                # Load best model weights and optimizer/scheduler state
                if os.path.exists(best_model_path):
                    ckpt = torch.load(best_model_path, map_location=self.device)
                    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                    self.best_model.load_state_dict(state)
                    if "optimizer_state_dict" in ckpt:
                        self.best_optimizer_state = copy.deepcopy(ckpt["optimizer_state_dict"])
                        self.trainer.optimizer.load_state_dict(self.best_optimizer_state)
                    if "scheduler_state_dict" in ckpt:
                        self.best_scheduler_state = copy.deepcopy(ckpt["scheduler_state_dict"])
                        self.trainer.scheduler.load_state_dict(self.best_scheduler_state)
                    print(f"[+] Resuming from Iteration {self.start_iteration} with best model from {best_model_path} (Elo: {self.elo_tracker.get_rating(self.current_best_id):.1f})")
            elif os.path.exists(best_model_path):
                ckpt = torch.load(best_model_path, map_location=self.device)
                state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.best_model.load_state_dict(state)
                if "optimizer_state_dict" in ckpt:
                    self.best_optimizer_state = copy.deepcopy(ckpt["optimizer_state_dict"])
                    self.trainer.optimizer.load_state_dict(self.best_optimizer_state)
                if "scheduler_state_dict" in ckpt:
                    self.best_scheduler_state = copy.deepcopy(ckpt["scheduler_state_dict"])
                    self.trainer.scheduler.load_state_dict(self.best_scheduler_state)
                print(f"[+] Loaded baseline weights from {best_model_path}")

        # Always ensure best_model.pt exists on disk as baseline
        if not os.path.exists(best_model_path):
            torch.save(
                {
                    "iteration": 0,
                    "model_state_dict": self.best_model.state_dict(),
                    "optimizer_state_dict": self.best_optimizer_state,
                    "scheduler_state_dict": self.best_scheduler_state,
                    "elo": 1000.0,
                },
                best_model_path,
            )

        # Candidate starts with identical weights and optimizer state to current best
        self.candidate_model.load_state_dict(self.best_model.state_dict())
        self.trainer.optimizer.load_state_dict(self.best_optimizer_state)
        self.trainer.scheduler.load_state_dict(self.best_scheduler_state)

        # Arena
        self.arena = Arena(
            board_size=cfg.board_size,
            komi=cfg.komi,
            num_sims=cfg.arena_sims,
            c_puct=cfg.c_puct,
            max_moves=cfg.max_moves if cfg.max_moves is not None else default_cfg.MAX_MOVES,
            win_threshold=cfg.win_threshold,
            resign_threshold=cfg.resign_threshold,
            opening_plies=cfg.opening_plies,
            device=self.device,
        )

    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Execute one complete AlphaZero iteration with dynamic training steps and optimizer synchronization.

        1. Self-Play -> buffer (auto-saved to buffer.pkl)
        2. Dynamic Train candidate model (sample-to-train ratio controlled to prevent overfitting)
        3. Arena: candidate vs best model (with gating check & opening plies)
        4. Promote candidate or Rollback model + optimizer/scheduler states
        """
        print(f"\n{'='*30} ITERATION {iteration} {'='*30}")

        # 1. Self-play
        print(f"[*] Step 1/3: Self-play ({self.config.games_per_iter} games, {self.config.num_sims} sims, komi {self.config.komi})...")
        self.best_model.eval()

        buffer_path = os.path.join(self.config.checkpoint_dir, "buffer.pkl")

        def on_game_finish(game_idx, samples):
            # Push game to buffer in memory
            self.buffer.push_game(samples)
            # Periodically backup buffer every 5 games or on final game
            if (game_idx + 1) % 5 == 0 or (game_idx + 1) == self.config.games_per_iter:
                self.buffer.save(buffer_path)

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
        )

        print(f"    [+] Added {len(new_samples)} samples. Total Replay Buffer: {len(self.buffer)} samples (Saved to disk).")

        # 2. Dynamic Training: prevent early overfitting when buffer is small
        max_allowed_steps = max(20, (len(self.buffer) * 2) // self.config.batch_size)
        effective_steps = min(self.config.train_steps, max_allowed_steps)
        print(f"[*] Step 2/3: Dynamic Training ({effective_steps} steps [cap: {self.config.train_steps}], batch size {self.config.batch_size})...")

        # Ensure candidate starts from current best model weights and clean optimizer/scheduler momentum
        self.candidate_model.load_state_dict(self.best_model.state_dict())
        self.trainer.optimizer.load_state_dict(self.best_optimizer_state)
        self.trainer.scheduler.load_state_dict(self.best_scheduler_state)

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
            print(f"[*] Step 3/3: Arena Evaluation ({self.config.arena_games} games vs best model, opening_plies {self.config.opening_plies})...")
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

            print(
                f"    Result: {arena_result.candidate_wins}W - {arena_result.baseline_wins}L - {arena_result.draws}D "
                f"({arena_result.candidate_win_rate * 100:.1f}%)"
            )
            print(f"    Candidate Elo: {cand_elo:.1f} | Best Elo: {best_elo:.1f}")

            # 4. Model Promotion / Rollback with Optimizer Synchronization
            if arena_result.accepted:
                print(f"    [+] Candidate ACCEPTED! (Winrate {arena_result.candidate_win_rate*100:.1f}% >= {self.config.win_threshold*100:.1f}%)")
                self.best_model.load_state_dict(self.candidate_model.state_dict())
                self.best_optimizer_state = copy.deepcopy(self.trainer.optimizer.state_dict())
                self.best_scheduler_state = copy.deepcopy(self.trainer.scheduler.state_dict())
                self.current_best_id = cand_id
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": self.best_model.state_dict(),
                        "optimizer_state_dict": self.best_optimizer_state,
                        "scheduler_state_dict": self.best_scheduler_state,
                        "elo": cand_elo,
                    },
                    best_model_path,
                )
            else:
                print(f"    [-] Candidate REJECTED. (Winrate {arena_result.candidate_win_rate*100:.1f}% < {self.config.win_threshold*100:.1f}%)")
                # Rollback candidate weights AND optimizer/scheduler state
                self.candidate_model.load_state_dict(self.best_model.state_dict())
                self.trainer.optimizer.load_state_dict(self.best_optimizer_state)
                self.trainer.scheduler.load_state_dict(self.best_scheduler_state)

            win_rate = arena_result.candidate_win_rate
            accepted = arena_result.accepted
        else:
            # Continuous AlphaZero (2018): promote candidate directly without gating
            print("    [*] Gating disabled (Continuous AlphaZero training): Candidate promoted directly.")
            self.best_model.load_state_dict(self.candidate_model.state_dict())
            self.best_optimizer_state = copy.deepcopy(self.trainer.optimizer.state_dict())
            self.best_scheduler_state = copy.deepcopy(self.trainer.scheduler.state_dict())
            self.current_best_id = cand_id
            cand_elo = self.elo_tracker.get_rating(self.current_best_id) + 10.0
            best_elo = cand_elo
            self.elo_tracker.ratings[cand_id] = cand_elo
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": self.best_model.state_dict(),
                    "optimizer_state_dict": self.best_optimizer_state,
                    "scheduler_state_dict": self.best_scheduler_state,
                    "elo": cand_elo,
                },
                best_model_path,
            )
            win_rate = 1.0
            accepted = True

        # Step LR Scheduler after each iteration
        current_lr = self.trainer.step_scheduler()
        print(f"    [*] Learning Rate: {current_lr:.6f}")

        # Save iteration checkpoint
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
            },
        )

        # Save Elo state
        ratings_path = os.path.join(self.config.checkpoint_dir, "ratings.json")
        self.elo_tracker.save(ratings_path)

        return {
            "iteration": iteration,
            "selfplay_samples": len(new_samples),
            "buffer_size": len(self.buffer),
            "train_loss": avg_loss,
            "effective_train_steps": effective_steps,
            "policy_loss": avg_pi_loss,
            "value_loss": avg_v_loss,
            "arena_win_rate": win_rate,
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
    parser.add_argument("--sims", type=int, default=default_cfg.NUM_SIMS, help="MCTS simulations per move")
    parser.add_argument("--train-steps", type=int, default=default_cfg.TRAIN_STEPS, help="Training steps per iter")
    parser.add_argument("--batch-size", type=int, default=default_cfg.BATCH_SIZE, help="Batch size")
    parser.add_argument("--arena-games", type=int, default=default_cfg.ARENA_GAMES, help="Arena match games")
    parser.add_argument("--opening-plies", type=int, default=default_cfg.OPENING_PLIES, help="Opening plies with tau=1 in Arena")
    parser.add_argument("--board-size", type=int, default=default_cfg.BOARD_SIZE, help="Board size")
    parser.add_argument("--komi", type=float, default=default_cfg.KOMI, help="Komi compensation for White")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--resign-disable-frac", type=float, default=default_cfg.RESIGN_DISABLE_FRAC, help="Fraction of selfplay games with resignation disabled")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not resume from existing checkpoints")
    parser.add_argument("--no-progress", action="store_false", dest="progress_bar", help="Disable live progress bar")
    parser.add_argument("--no-gating", action="store_false", dest="gating", help="Disable Arena gating (Continuous AlphaZero 2018 training)")
    parser.set_defaults(resume=True, progress_bar=True, gating=True)
    args = parser.parse_args()

    cfg = PipelineConfig(
        num_iters=args.iters,
        games_per_iter=args.games,
        num_sims=args.sims,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        arena_games=args.arena_games,
        opening_plies=args.opening_plies,
        board_size=args.board_size,
        komi=args.komi,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        progress_bar=args.progress_bar,
        resume=args.resume,
        resign_disable_frac=args.resign_disable_frac,
        gating=args.gating,
    )

    pipeline = AlphaZeroPipeline(cfg)
    pipeline.train()


if __name__ == "__main__":
    main()
