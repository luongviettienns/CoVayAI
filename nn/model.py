"""AlphaZero Size-Independent Dual-Head Residual Neural Network for Go."""

import logging
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """Standard Residual Block with 3x3 Convolutions and Batch Normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out

class AlphaZeroNet(nn.Module):
    """
    Size-Independent Dual-Head ResNet f_theta(s) = (p, v) for Go.

    Shared architecture for 9x9, 13x13, 19x19, and any board size N.
    - Policy Head:
        * Spatial moves: 1x1 convs preserve 2D topology -> spatial logits [B, N*N]
        * Pass move: global pooled trunk features -> pass logit [B, 1]
        * Combined: policy_logits of shape [B, N*N + 1]
    - Value Head:
        * Conv -> BN -> Global Average Pooling [B, C, N, N] -> [B, C] -> MLP -> tanh [B, 1]

    Inputs:
        Tensor of shape (B, in_channels, H, W), typically in_channels = 17, H = W = board_size.
    Outputs:
        policy_logits: shape (B, H * W + 1), where logits[:, :H*W] are spatial actions
                       and logits[:, -1] is Pass.
        value: shape (B, 1) in [-1, 1]
    """

    def __init__(
        self,
        board_size: Optional[int] = 9,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        in_channels: int = 17,
        value_channels: int = 32,
    ):
        super().__init__()
        self.board_size = board_size
        self.action_size = (board_size * board_size + 1) if board_size is not None else None
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.value_channels = value_channels
        self.architecture_version = 2

        # Initial convolutional trunk
        self.conv_in = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)

        # Residual backbone tower
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Spatial Policy Head: maps trunk [B, num_filters, N, N] -> [B, 32, N, N] -> [B, 1, N, N]
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_spatial = nn.Conv2d(32, 1, kernel_size=1, bias=True)

        # Pass Head: Global average pooling over pass conv branch -> [B, 1]
        self.pass_conv = nn.Conv2d(num_filters, 16, kernel_size=1, bias=False)
        self.pass_bn = nn.BatchNorm2d(16)
        self.pass_fc = nn.Linear(16, 1)

        # Size-Independent Value Head: Conv -> BN -> Global Avg Pooling -> MLP -> tanh
        self.value_conv = nn.Conv2d(num_filters, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input trunk
        out = F.relu(self.bn_in(self.conv_in(x)))

        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # Spatial policy branch: [B, num_filters, H, W] -> [B, 1, H, W] -> [B, H*W]
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        spatial_logits = self.policy_spatial(p).flatten(start_dim=1)

        # Pass branch: permutation-invariant global pooling -> [B, 1]
        p_pass = F.relu(self.pass_bn(self.pass_conv(out)))
        pass_pooled = p_pass.mean(dim=(-2, -1))
        pass_logit = self.pass_fc(pass_pooled)

        policy_logits = torch.cat([spatial_logits, pass_logit], dim=-1)

        # Size-independent value head: global pooling over spatial dimensions -> [B, 1]
        v = F.relu(self.value_bn(self.value_conv(out)))
        v_pooled = v.mean(dim=(-2, -1))
        v = F.relu(self.value_fc1(v_pooled))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def load_compatible_weights(self, state_dict: dict) -> Tuple[bool, bool]:
        """
        Load state dict with architecture compatibility checks.

        If all keys and shapes match, loads strictly.
        If loading an older architecture (e.g. v1 with fixed linear heads),
        copies compatible backbone weights and leaves heads randomly initialized.

        Returns:
            (loaded_backbone: bool, reinit_heads: bool)
        """
        model_dict = self.state_dict()
        matching_dict = {}
        mismatched_keys = []

        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    matching_dict[k] = v
                else:
                    mismatched_keys.append(f"{k} (shape mismatch: expected {model_dict[k].shape}, got {v.shape})")
            else:
                mismatched_keys.append(f"{k} (unknown key)")

        if len(matching_dict) == len(model_dict):
            # Perfect match (Architecture v2)
            self.load_state_dict(matching_dict, strict=True)
            return True, False
        else:
            # Partial match (Migration from v1 or different heads)
            self.load_state_dict(matching_dict, strict=False)
            logger.warning(
                f"[AlphaZeroNet] Architecture compatibility migration: loaded {len(matching_dict)} matching tensors. "
                f"Reinitialized heads due to mismatches: {mismatched_keys}"
            )
            return True, True

    @torch.inference_mode()
    def predict(
        self,
        state_features: np.ndarray,
        legal_moves_mask: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Fast inference evaluation for MCTS across any board size.

        Args:
            state_features: numpy array of shape (C, N, N) or (B, C, N, N)
            legal_moves_mask: optional binary mask of shape (N*N + 1,)
            device: target torch device (default: next(self.parameters()).device)
            use_amp: if True and on CUDA, use Automatic Mixed Precision (FP16)

        Returns:
            (probabilities: np.ndarray, value: float)
        """
        single_input = state_features.ndim == 3
        if single_input:
            batch_features = state_features[np.newaxis, ...]
            batch_masks = legal_moves_mask[np.newaxis, ...] if legal_moves_mask is not None else None
            probs, vals = self.predict_batch(batch_features, legal_moves_masks=batch_masks, device=device, use_amp=use_amp)
            return probs[0], float(vals[0, 0])
        else:
            probs, vals = self.predict_batch(state_features, legal_moves_masks=legal_moves_mask, device=device, use_amp=use_amp)
            return probs, vals

    @torch.inference_mode()
    def predict_batch(
        self,
        state_features: np.ndarray,
        legal_moves_masks: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast batch inference evaluation for multiple states across any board size.

        Args:
            state_features: numpy array of shape (B, C, N, N)
            legal_moves_masks: optional binary masks of shape (B, N*N + 1)
            device: target torch device (default: next(self.parameters()).device)
            use_amp: if True and on CUDA, use Automatic Mixed Precision (FP16)

        Returns:
            (probabilities: np.ndarray of shape (B, action_size), values: np.ndarray of shape (B, 1))
        """
        self.eval()
        dev = device if device is not None else next(self.parameters()).device

        x_tensor = torch.from_numpy(state_features).to(dev)

        if use_amp and dev.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                policy_logits, value = self(x_tensor)
        else:
            policy_logits, value = self(x_tensor)

        if legal_moves_masks is not None:
            mask_tensor = torch.from_numpy(legal_moves_masks).to(dev)
            # Prevent all-zero mask row producing NaNs through softmax
            row_sums = mask_tensor.sum(dim=-1, keepdim=True)
            safe_mask = torch.where(row_sums > 0, mask_tensor, torch.ones_like(mask_tensor))
            policy_logits = torch.where(safe_mask > 0, policy_logits, torch.tensor(-1e9, device=dev, dtype=policy_logits.dtype))

        probs = F.softmax(policy_logits.float(), dim=-1).cpu().numpy()
        val = value.float().cpu().numpy()

        return probs, val

