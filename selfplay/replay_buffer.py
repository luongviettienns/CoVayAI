"""Experience Replay Buffer for AlphaZero self-play training."""

from typing import List, Tuple, Optional
from collections import deque
import pickle
import numpy as np
import torch
from nn.features import apply_random_symmetry

class ReplayBuffer:
    """Sliding-window replay buffer storing tuples of (state, pi, z) with D4 augmentation."""

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, pi: np.ndarray, z: float) -> None:
        """Store a single training sample."""
        self.buffer.append((state, pi, float(z)))

    def push_game(self, game_samples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Store all samples from a completed game."""
        for state, pi, z in game_samples:
            self.push(state, pi, z)

    def sample_batch(
        self, batch_size: int, augment: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch of training data.
        
        If augment is True, applies random D4 symmetry transformation (rotation + reflection)
        directly to each sampled state and policy target.
        
        Returns:
            (states, pis, zs) as PyTorch tensors:
                states: shape (B, C, N, N), float32
                pis: shape (B, N*N + 1), float32
                zs: shape (B, 1), float32
        """
        buf_len = len(self.buffer)
        if buf_len == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")

        replace = buf_len < batch_size
        indices = np.random.choice(buf_len, size=batch_size, replace=replace)

        batch_states = []
        batch_pis = []
        batch_zs = []

        for idx in indices:
            s, pi, z = self.buffer[idx]

            if augment:
                s_aug, pi_aug = apply_random_symmetry(s, pi)
                batch_states.append(s_aug)
                batch_pis.append(pi_aug)
            else:
                batch_states.append(s)
                batch_pis.append(pi)

            batch_zs.append(z)

        states_tensor = torch.from_numpy(np.array(batch_states, dtype=np.float32))
        pis_tensor = torch.from_numpy(np.array(batch_pis, dtype=np.float32))
        zs_tensor = torch.from_numpy(np.array(batch_zs, dtype=np.float32)).unsqueeze(1)

        return states_tensor, pis_tensor, zs_tensor

    def save(self, filepath: str) -> None:
        """Serialize buffer to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, filepath: str) -> None:
        """Deserialize buffer from disk."""
        with open(filepath, "rb") as f:
            items = pickle.load(f)
            self.buffer = deque(items, maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)
