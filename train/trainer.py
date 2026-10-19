"""AlphaZero Training Pipeline with Dual-Head Loss, AMP (FP16), and Checkpointing."""

from typing import Dict, Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from selfplay.replay_buffer import ReplayBuffer

class Trainer:
    """Orchestrates network optimization, loss computation, and checkpointing for AlphaZero."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adam",
        use_amp: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")

        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_step(
        self, states: torch.Tensor, target_pis: torch.Tensor, target_zs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Execute one optimization step on a mini-batch.
        
        Loss:
            L = MSE(v, z) - pi^T log(p) + c ||theta||^2
        """
        self.model.train()
        states = states.to(self.device)
        target_pis = target_pis.to(self.device)
        target_zs = target_zs.to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            policy_logits, value = self.model(states)

            # Value Loss: MSE between predicted value and actual game outcome z
            loss_v = F.mse_loss(value, target_zs)

            # Policy Loss: Cross-entropy with soft probability distribution pi
            log_probs = F.log_softmax(policy_logits, dim=-1)
            loss_p = -(target_pis * log_probs).sum(dim=-1).mean()

            total_loss = loss_v + loss_p

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(loss_p.item()),
            "value_loss": float(loss_v.item()),
        }

    def train_iteration(
        self,
        replay_buffer: ReplayBuffer,
        steps: int = 400,
        batch_size: int = 256,
        augment: bool = True,
    ) -> Dict[str, float]:
        """Train over K steps sampled from the replay buffer and return average metrics."""
        if len(replay_buffer) == 0:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0

        for _ in range(steps):
            states, target_pis, target_zs = replay_buffer.sample_batch(
                batch_size=batch_size, augment=augment
            )
            metrics = self.train_step(states, target_pis, target_zs)

            total_loss_accum += metrics["total_loss"]
            policy_loss_accum += metrics["policy_loss"]
            value_loss_accum += metrics["value_loss"]

        return {
            "total_loss": total_loss_accum / steps,
            "policy_loss": policy_loss_accum / steps,
            "value_loss": value_loss_accum / steps,
        }

    def save_checkpoint(self, filepath: str, iteration: int = 0, extra: Optional[dict] = None) -> None:
        """Save model and optimizer state to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """Load model and optimizer state from file and return saved iteration."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("iteration", 0)
