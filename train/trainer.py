"""AlphaZero Training Pipeline with Dual-Head Loss, AMP (FP16), AdamW, and MultiStepLR."""

from typing import Dict, Optional, Tuple, List
import os
import random
import subprocess
import numpy as np
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
        self.global_step: int = 0

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
        self.global_step += 1

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

    def save_checkpoint(
        self,
        filepath: str,
        iteration: int = 0,
        extra: Optional[dict] = None,
    ) -> None:
        """Save model, optimizer, scheduler, scaler, RNG states, and git SHA atomically to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            git_sha = "unknown"

        numpy_rng = np.random.get_state()
        clean_numpy_rng = (
            numpy_rng[0],
            numpy_rng[1].tolist() if isinstance(numpy_rng[1], np.ndarray) else numpy_rng[1],
            numpy_rng[2],
            numpy_rng[3],
            numpy_rng[4],
        )

        rng_states = {
            "random": random.getstate(),
            "numpy": clean_numpy_rng,
            "torch": torch.get_rng_state().cpu(),
        }
        if torch.cuda.is_available() and self.device.type == "cuda":
            rng_states["torch_cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]

        checkpoint = {
            "iteration": iteration,
            "architecture_version": getattr(self.model, "architecture_version", 2),
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "rng_states": rng_states,
            "git_sha": git_sha,
        }
        if extra:
            checkpoint.update(extra)
        tmp_path = filepath + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """Load model, optimizer, scheduler, scaler, and RNG state from file and return saved iteration."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        if hasattr(self.model, "load_compatible_weights"):
            self.model.load_compatible_weights(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
        if "rng_states" in checkpoint:
            rng = checkpoint["rng_states"]
            if "random" in rng:
                random.setstate(rng["random"])
            if "numpy" in rng:
                nrng = rng["numpy"]
                if isinstance(nrng, (list, tuple)) and len(nrng) == 5:
                    reconstructed = (
                        nrng[0],
                        np.array(nrng[1], dtype=np.uint32) if isinstance(nrng[1], list) else nrng[1],
                        nrng[2],
                        nrng[3],
                        nrng[4],
                    )
                    np.random.set_state(reconstructed)
                else:
                    np.random.set_state(nrng)
            if "torch" in rng:
                torch_state = rng["torch"]
                if isinstance(torch_state, torch.Tensor):
                    torch_state = torch_state.cpu()
                torch.set_rng_state(torch_state)
            if "torch_cuda" in rng and torch.cuda.is_available() and self.device.type == "cuda":
                cuda_states = rng["torch_cuda"]
                if isinstance(cuda_states, list):
                    cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)

        return checkpoint.get("iteration", 0)
