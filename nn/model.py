"""AlphaZero Dual-Head Residual Neural Network for Go."""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Dual-Head ResNet f_theta(s) = (p, v) for Go.
    
    Inputs:
        Tensor of shape (B, in_channels, board_size, board_size), typically in_channels = 17.
    Outputs:
        policy_logits: shape (B, board_size * board_size + 1)
        value: shape (B, 1) in [-1, 1]
    """

    def __init__(
        self,
        board_size: int = 9,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        in_channels: int = 17,
    ):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size + 1

        # Initial convolutional block
        self.conv_in = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)

        # Residual backbone
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input trunk
        out = F.relu(self.bn_in(self.conv_in(x)))

        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @torch.no_grad()
    def predict(
        self,
        state_features: np.ndarray,
        legal_moves_mask: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Fast inference evaluation for MCTS.
        
        Args:
            state_features: numpy array of shape (C, N, N) or (B, C, N, N)
            legal_moves_mask: optional binary mask of shape (N*N + 1,)
            device: target torch device (default: next(self.parameters()).device)
            
        Returns:
            (probabilities: np.ndarray, value: float)
        """
        self.eval()
        dev = device if device is not None else next(self.parameters()).device

        # Handle single state input vs batch
        single_input = state_features.ndim == 3
        if single_input:
            x_tensor = torch.from_numpy(state_features).unsqueeze(0).to(dev)
        else:
            x_tensor = torch.from_numpy(state_features).to(dev)

        policy_logits, value = self(x_tensor)

        logits = policy_logits[0] if single_input else policy_logits

        # Mask illegal moves if mask provided
        if legal_moves_mask is not None:
            mask_tensor = torch.from_numpy(legal_moves_mask).to(dev)
            logits = torch.where(mask_tensor > 0, logits, torch.tensor(-1e9, device=dev))

        probs = F.softmax(logits, dim=-1).cpu().numpy()
        val = float(value[0, 0].item()) if single_input else value.cpu().numpy()

        return probs, val

    @torch.no_grad()
    def predict_batch(
        self,
        state_features: np.ndarray,
        legal_moves_masks: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast batch inference evaluation for multiple states.

        Args:
            state_features: numpy array of shape (B, C, N, N)
            legal_moves_masks: optional binary masks of shape (B, N*N + 1)
            device: target torch device (default: next(self.parameters()).device)

        Returns:
            (probabilities: np.ndarray of shape (B, action_size), values: np.ndarray of shape (B, 1))
        """
        self.eval()
        dev = device if device is not None else next(self.parameters()).device

        x_tensor = torch.from_numpy(state_features).to(dev)
        policy_logits, value = self(x_tensor)

        if legal_moves_masks is not None:
            mask_tensor = torch.from_numpy(legal_moves_masks).to(dev)
            policy_logits = torch.where(mask_tensor > 0, policy_logits, torch.tensor(-1e9, device=dev))

        probs = F.softmax(policy_logits, dim=-1).cpu().numpy()
        val = value.cpu().numpy()

        return probs, val

