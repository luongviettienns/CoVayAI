"""AlphaZero Training Pipeline with Dual-Head Loss, AMP (FP16), AdamW, and MultiStepLR."""

from typing import Dict, Optional, Tuple, List
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
        weight_decay: float = 1e-2,
        optimizer_type: str = "adamw",
        milestones: Optional[List[int]] = None,
        lr_gamma: float = 0.1,
        use_amp: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")

        # Separate parameters into 2D/4D (weights: Conv, Linear) and 1D (biases, BatchNorm)
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        opt_name = optimizer_type.lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr)
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(param_groups, lr=lr)
        else:
            self.optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=0.9)

        # Learning rate schedule across iterations (MultiStepLR is more stable than plateau on dynamic buffers)
        self.milestones = milestones or [30, 60]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.milestones, gamma=lr_gamma
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def step_scheduler(self) -> float:
        """Step the learning rate scheduler and return the new learning rate."""
        self.scheduler.step()
        return self.get_lr()

    def get_lr(self) -> float:
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

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
        progress_bar: bool = False,
    ) -> Dict[str, float]:
        """Train over K steps sampled from the replay buffer and return average metrics."""
        if len(replay_buffer) == 0:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0

        iterator = range(steps)
        pbar = None
        if progress_bar and tqdm is not None:
            pbar = tqdm(iterator, desc="Training", unit="step", dynamic_ncols=True)
            iterator = pbar

        for step_idx in iterator:
            states, target_pis, target_zs = replay_buffer.sample_batch(
                batch_size=batch_size, augment=augment
            )
            metrics = self.train_step(states, target_pis, target_zs)

            total_loss_accum += metrics["total_loss"]
            policy_loss_accum += metrics["policy_loss"]
            value_loss_accum += metrics["value_loss"]

            if pbar is not None:
                pbar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "p_loss": f"{metrics['policy_loss']:.4f}",
                    "v_loss": f"{metrics['value_loss']:.4f}",
                })

        return {
            "total_loss": total_loss_accum / steps,
            "policy_loss": policy_loss_accum / steps,
            "value_loss": value_loss_accum / steps,
        }

    def save_checkpoint(self, filepath: str, iteration: int = 0, extra: Optional[dict] = None) -> None:
        """Save model, optimizer, and scheduler state to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if extra:
            checkpoint.update(extra)
        tmp_path = filepath + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """Load model, optimizer, and scheduler state from file and return saved iteration."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("iteration", 0)
