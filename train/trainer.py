"""AlphaZero Training Pipeline with Dual-Head Loss, AMP (FP16), AdamW, and MultiStepLR."""

from typing import Dict, Optional, Tuple, List, Any
import os
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from selfplay.replay_buffer import ReplayBuffer


def compute_adaptive_train_steps(
    new_samples: int,
    steps_per_sample: float = 0.5,
    min_steps: int = 20,
    max_steps: int = 400,
) -> int:
    """
    Compute training steps dynamically scaled by newly collected samples.
    Prevents wasteful overtraining on small initial buffers while ensuring sufficient
    gradient updates as fresh experience arrives.
    """
    if new_samples <= 0:
        return min_steps
    scaled = int(round(new_samples * steps_per_sample))
    return max(min_steps, min(scaled, max_steps))


def split_games_train_val(
    games: List[Any],
    val_ratio: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[List[Any], List[Any]]:
    """
    Partition games into disjoint training and held-out validation sets by complete games.
    Guarantees zero data leakage across game trajectories.
    """
    if not games or val_ratio <= 0.0:
        return list(games), []
    if val_ratio >= 1.0:
        return [], list(games)

    rng = random.Random(seed) if seed is not None else random.Random()
    indices = list(range(len(games)))
    rng.shuffle(indices)

    num_val = max(1, int(round(len(games) * val_ratio)))
    if num_val >= len(games) and len(games) > 1:
        num_val = len(games) - 1

    val_idx_set = set(indices[:num_val])
    train_games = [g for i, g in enumerate(games) if i not in val_idx_set]
    val_games = [g for i, g in enumerate(games) if i in val_idx_set]
    return train_games, val_games


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
        ownership_coef: float = 0.25,
        score_coef: float = 0.02,
        device: Optional[torch.device] = None,
        channels_last: bool = False,
        compile_model: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.ownership_coef = ownership_coef
        self.score_coef = score_coef
        self.global_step: int = 0
        self.channels_last = channels_last

        if self.channels_last:
            if hasattr(self.model, "to_channels_last"):
                self.model.to_channels_last()
            else:
                self.model.to(memory_format=torch.channels_last)

        self.is_compiled = False
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                self.is_compiled = True
                logger.info("[Trainer] torch.compile enabled successfully.")
            except Exception as e:
                logger.warning(f"[Trainer] torch.compile failed or unsupported on this platform: {e}. Falling back to standard eager mode.")

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
        self,
        states: torch.Tensor,
        target_pis: torch.Tensor,
        target_zs: torch.Tensor,
        target_ownerships: Optional[torch.Tensor] = None,
        target_scores: Optional[torch.Tensor] = None,
        aux_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Execute one optimization step on a mini-batch with primary and optional auxiliary losses.
        
        Loss:
            L = MSE(v, z) - pi^T log(p) + c_own * MSE(own, own_target) + c_score * MSE(score, score_target)
        """
        self.model.train()
        states = states.to(self.device)
        if self.channels_last and states.dim() == 4 and not states.is_contiguous(memory_format=torch.channels_last):
            states = states.to(memory_format=torch.channels_last)
        target_pis = target_pis.to(self.device)
        target_zs = target_zs.to(self.device)

        has_aux = (
            target_ownerships is not None
            and getattr(self.model, "aux_heads", False)
        )

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            if has_aux:
                policy_logits, value, pred_own, pred_score = self.model(states, return_aux=True)
            else:
                policy_logits, value = self.model(states)

            # Value Loss: MSE between predicted value and actual game outcome z
            loss_v = F.mse_loss(value, target_zs)

            # Policy Loss: Cross-entropy with soft probability distribution pi
            log_probs = F.log_softmax(policy_logits, dim=-1)
            loss_p = -(target_pis * log_probs).sum(dim=-1).mean()

            total_loss = loss_v + loss_p
            loss_own_val = 0.0
            loss_score_val = 0.0

            if has_aux:
                target_ownerships = target_ownerships.to(self.device)
                target_scores = target_scores.to(self.device)
                if aux_masks is not None:
                    aux_masks = aux_masks.to(self.device)
                    valid_count = aux_masks.sum()
                else:
                    valid_count = torch.tensor(states.shape[0], device=self.device)

                if valid_count > 0:
                    h, w = states.shape[-2], states.shape[-1]
                    mask_own = aux_masks.unsqueeze(-1) if aux_masks is not None else 1.0
                    loss_own = (((pred_own - target_ownerships) ** 2) * mask_own).sum() / (valid_count * h * w + 1e-8)

                    mask_score = aux_masks if aux_masks is not None else 1.0
                    loss_score = (((pred_score - target_scores) ** 2) * mask_score).sum() / (valid_count + 1e-8)

                    total_loss = total_loss + self.ownership_coef * loss_own + self.score_coef * loss_score
                    loss_own_val = float(loss_own.item())
                    loss_score_val = float(loss_score.item())

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
            "ownership_loss": loss_own_val,
            "score_loss": loss_score_val,
        }

    def train_iteration(
        self,
        replay_buffer: ReplayBuffer,
        steps: Optional[int] = 400,
        batch_size: int = 256,
        augment: bool = True,
        progress_bar: bool = False,
        use_aux: Optional[bool] = None,
        adaptive_steps: bool = False,
        new_samples: Optional[int] = None,
        steps_per_sample: float = 0.5,
        min_steps: int = 20,
        max_steps: int = 400,
    ) -> Dict[str, float]:
        """Train over K steps sampled from the replay buffer and return average metrics."""
        if len(replay_buffer) == 0:
            return {
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "ownership_loss": 0.0,
                "score_loss": 0.0,
                "actual_steps": 0,
                "effective_epochs": 0.0,
            }

        if adaptive_steps or (steps is None and new_samples is not None):
            actual_steps = compute_adaptive_train_steps(
                new_samples=new_samples or 0,
                steps_per_sample=steps_per_sample,
                min_steps=min_steps,
                max_steps=max_steps,
            )
        else:
            actual_steps = steps if steps is not None else 400

        actual_steps = max(1, actual_steps)
        effective_epochs = float(actual_steps * batch_size) / max(1, len(replay_buffer))

        if use_aux is None:
            use_aux = getattr(self.model, "aux_heads", False)

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        own_loss_accum = 0.0
        score_loss_accum = 0.0

        iterator = range(actual_steps)
        pbar = None
        if progress_bar and tqdm is not None:
            pbar = tqdm(iterator, desc="Training", unit="step", dynamic_ncols=True)
            iterator = pbar

        for step_idx in iterator:
            if use_aux:
                states, target_pis, target_zs, target_owns, target_scores, aux_masks = (
                    replay_buffer.sample_batch(batch_size=batch_size, augment=augment, return_aux=True)
                )
                metrics = self.train_step(
                    states,
                    target_pis,
                    target_zs,
                    target_ownerships=target_owns,
                    target_scores=target_scores,
                    aux_masks=aux_masks,
                )
            else:
                states, target_pis, target_zs = replay_buffer.sample_batch(
                    batch_size=batch_size, augment=augment
                )
                metrics = self.train_step(states, target_pis, target_zs)

            total_loss_accum += metrics["total_loss"]
            policy_loss_accum += metrics["policy_loss"]
            value_loss_accum += metrics["value_loss"]
            own_loss_accum += metrics.get("ownership_loss", 0.0)
            score_loss_accum += metrics.get("score_loss", 0.0)

            if pbar is not None:
                pbar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "p_loss": f"{metrics['policy_loss']:.4f}",
                    "v_loss": f"{metrics['value_loss']:.4f}",
                })

        return {
            "total_loss": total_loss_accum / actual_steps,
            "policy_loss": policy_loss_accum / actual_steps,
            "value_loss": value_loss_accum / actual_steps,
            "ownership_loss": own_loss_accum / actual_steps,
            "score_loss": score_loss_accum / actual_steps,
            "actual_steps": actual_steps,
            "effective_epochs": round(effective_epochs, 4),
        }

    @torch.inference_mode()
    def evaluate(
        self,
        val_buffer: ReplayBuffer,
        batch_size: int = 256,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on held-out validation buffer without data augmentation or gradients.
        Computes validation loss, policy KL divergence, policy entropy, and value calibration.
        """
        if len(val_buffer) == 0:
            return {
                "val_loss": 0.0,
                "val_policy_loss": 0.0,
                "val_value_loss": 0.0,
                "policy_kl": 0.0,
                "policy_entropy": 0.0,
                "value_calibration": 0.0,
            }

        self.model.eval()
        total_loss_accum = 0.0
        p_loss_accum = 0.0
        v_loss_accum = 0.0
        kl_accum = 0.0
        entropy_accum = 0.0
        calib_accum = 0.0

        total_samples = len(val_buffer)
        num_batches = max(1, (total_samples + batch_size - 1) // batch_size)
        if max_batches is not None:
            num_batches = min(num_batches, max_batches)

        has_aux = getattr(self.model, "aux_heads", False)

        for _ in range(num_batches):
            if has_aux:
                states, target_pis, target_zs, _, _, _ = val_buffer.sample_batch(
                    batch_size=min(batch_size, total_samples), augment=False, return_aux=True
                )
            else:
                states, target_pis, target_zs = val_buffer.sample_batch(
                    batch_size=min(batch_size, total_samples), augment=False
                )

            states = states.to(self.device)
            if self.channels_last and states.dim() == 4 and not states.is_contiguous(memory_format=torch.channels_last):
                states = states.to(memory_format=torch.channels_last)
            target_pis = target_pis.to(self.device)
            target_zs = target_zs.to(self.device)

            if has_aux:
                policy_logits, value, _, _ = self.model(states, return_aux=True)
            else:
                policy_logits, value = self.model(states)

            loss_v = F.mse_loss(value, target_zs)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            probs = F.softmax(policy_logits, dim=-1)
            loss_p = -(target_pis * log_probs).sum(dim=-1).mean()

            # KL(pi_target || p_pred) = sum(pi * (log(pi + eps) - log(p + eps)))
            eps = 1e-8
            target_log_pis = torch.log(target_pis + eps)
            kl = (target_pis * (target_log_pis - log_probs)).sum(dim=-1).mean()

            # Policy entropy = -sum(p * log(p))
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # Value calibration = |mean(value) - mean(target_zs)|
            calib = torch.abs(value.mean() - target_zs.mean())

            total_loss = loss_v + loss_p

            total_loss_accum += float(total_loss.item())
            p_loss_accum += float(loss_p.item())
            v_loss_accum += float(loss_v.item())
            kl_accum += float(kl.item())
            entropy_accum += float(entropy.item())
            calib_accum += float(calib.item())

        return {
            "val_loss": total_loss_accum / num_batches,
            "val_policy_loss": p_loss_accum / num_batches,
            "val_value_loss": v_loss_accum / num_batches,
            "policy_kl": kl_accum / num_batches,
            "policy_entropy": entropy_accum / num_batches,
            "value_calibration": calib_accum / num_batches,
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
