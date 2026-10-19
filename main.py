"""Main AlphaZero Training Pipeline Orchestrator for CoVayAI.

Coordinates the end-to-end AlphaZero loop:
  1. Self-Play: Generate training data using current best model
  2. Training: Train candidate model on augmented replay buffer
  3. Arena Evaluation: Head-to-head match between candidate and best models
  4. Model Promotion: Accept candidate if win rate >= 55%, update Elo ratings
"""

import os
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
    games_per_iter: int = default_cfg.GAMES_PER_ITER

    # Training
    batch_size: int = default_cfg.BATCH_SIZE
    train_steps: int = default_cfg.TRAIN_STEPS
    lr: float = default_cfg.LR
    weight_decay: float = default_cfg.WEIGHT_DECAY
    buffer_size: int = default_cfg.BUFFER_SIZE
    num_iters: int = default_cfg.NUM_ITERS

    # Arena
    arena_games: int = default_cfg.ARENA_GAMES
    arena_sims: int = default_cfg.ARENA_SIMS
    win_threshold: float = default_cfg.WIN_THRESHOLD

    # Runtime & Checkpoints
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AlphaZeroPipeline:
    """Orchestrates Self-Play, Training, Arena, and Model Promotion."""

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

        # Candidate starts with identical weights
        self.candidate_model.load_state_dict(self.best_model.state_dict())

        # Save initial best model
        initial_best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        if not os.path.exists(initial_best_path):
            torch.save(
                {
                    "iteration": 0,
                    "model_state_dict": self.best_model.state_dict(),
                },
                initial_best_path,
            )

        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=cfg.buffer_size)

        # Trainer for candidate model
        self.trainer = Trainer(
            model=self.candidate_model,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            device=self.device,
            use_amp=cfg.use_amp and self.device.type == "cuda",
        )

        # Elo Tracker
        ratings_path = os.path.join(self.config.checkpoint_dir, "ratings.json")
        if os.path.exists(ratings_path):
            self.elo_tracker = EloTracker.load(ratings_path)
        else:
            self.elo_tracker = EloTracker(initial_rating=1000.0)

        # Arena
        self.arena = Arena(
            board_size=cfg.board_size,
            komi=default_cfg.KOMI,
            num_sims=cfg.arena_sims,
            c_puct=cfg.c_puct,
            win_threshold=cfg.win_threshold,
            resign_threshold=cfg.resign_threshold,
            device=self.device,
        )

        self.current_best_id = "iter_0"

    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Execute one complete AlphaZero iteration.

        1. Self-Play -> buffer
        2. Train candidate model on buffer
        3. Arena: candidate vs best model
        4. Promote candidate if win_rate >= threshold
        """
        print(f"\n{'='*30} ITERATION {iteration} {'='*30}")

        # 1. Self-play
        print(f"[*] Step 1/3: Self-play ({self.config.games_per_iter} games)...")
        self.best_model.eval()
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
            device=self.device,
        )
        self.buffer.push_game(new_samples)
        print(f"    Added {len(new_samples)} samples. Total buffer size: {len(self.buffer)}")

        # 2. Train candidate model
        print(f"[*] Step 2/3: Training ({self.config.train_steps} steps, batch size {self.config.batch_size})...")
        # Ensure candidate starts from current best model weights
        self.candidate_model.load_state_dict(self.best_model.state_dict())
        train_metrics = self.trainer.train_iteration(
            replay_buffer=self.buffer,
            steps=self.config.train_steps,
            batch_size=self.config.batch_size,
        )
        avg_loss = train_metrics["total_loss"]
        avg_pi_loss = train_metrics["policy_loss"]
        avg_v_loss = train_metrics["value_loss"]
        print(f"    Train Loss: {avg_loss:.4f} (Policy: {avg_pi_loss:.4f}, Value: {avg_v_loss:.4f})")

        # 3. Arena Evaluation
        print(f"[*] Step 3/3: Arena Evaluation ({self.config.arena_games} games vs best model)...")
        arena_result: ArenaResult = self.arena.evaluate(
            candidate_model=self.candidate_model,
            baseline_model=self.best_model,
            num_games=self.config.arena_games,
        )

        cand_id = f"iter_{iteration}"
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

        # 4. Model Promotion
        best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        if arena_result.accepted:
            print(f"    [+] Candidate ACCEPTED! (Winrate {arena_result.candidate_win_rate*100:.1f}% >= {self.config.win_threshold*100:.1f}%)")
            self.best_model.load_state_dict(self.candidate_model.state_dict())
            self.current_best_id = cand_id
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": self.best_model.state_dict(),
                    "elo": cand_elo,
                },
                best_model_path,
            )
        else:
            print(f"    [-] Candidate REJECTED. (Winrate {arena_result.candidate_win_rate*100:.1f}% < {self.config.win_threshold*100:.1f}%)")
            self.candidate_model.load_state_dict(self.best_model.state_dict())

        # Save iteration checkpoint
        iter_ckpt_path = os.path.join(
            self.config.checkpoint_dir, f"checkpoint_iter_{iteration}.pt"
        )
        self.trainer.save_checkpoint(
            filepath=iter_ckpt_path,
            iteration=iteration,
            extra={
                "accepted": arena_result.accepted,
                "win_rate": arena_result.candidate_win_rate,
                "candidate_elo": cand_elo,
                "best_elo": best_elo,
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
            "policy_loss": avg_pi_loss,
            "value_loss": avg_v_loss,
            "arena_win_rate": arena_result.candidate_win_rate,
            "accepted": arena_result.accepted,
            "candidate_elo": cand_elo,
            "best_elo": best_elo,
        }

    def train(self) -> None:
        """Run full multi-iteration training loop."""
        print(f"Starting AlphaZero training on {self.device} for {self.config.num_iters} iterations.")
        for it in range(1, self.config.num_iters + 1):
            self.run_iteration(it)


def main() -> None:
    """CLI entry point for running the AlphaZero pipeline."""
    parser = argparse.ArgumentParser(description="CoVayAI AlphaZero Pipeline")
    parser.add_argument("--iters", type=int, default=default_cfg.NUM_ITERS, help="Number of iterations")
    parser.add_argument("--games", type=int, default=default_cfg.GAMES_PER_ITER, help="Self-play games per iteration")
    parser.add_argument("--sims", type=int, default=default_cfg.NUM_SIMS, help="MCTS simulations")
    parser.add_argument("--train-steps", type=int, default=default_cfg.TRAIN_STEPS, help="Training steps per iter")
    parser.add_argument("--batch-size", type=int, default=default_cfg.BATCH_SIZE, help="Batch size")
    parser.add_argument("--arena-games", type=int, default=default_cfg.ARENA_GAMES, help="Arena match games")
    parser.add_argument("--board-size", type=int, default=default_cfg.BOARD_SIZE, help="Board size")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    cfg = PipelineConfig(
        num_iters=args.iters,
        games_per_iter=args.games,
        num_sims=args.sims,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        arena_games=args.arena_games,
        board_size=args.board_size,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    pipeline = AlphaZeroPipeline(cfg)
    pipeline.train()


if __name__ == "__main__":
    main()
