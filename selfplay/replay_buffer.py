"""Experience Replay Buffer for AlphaZero self-play training with O(1) Ring Buffer."""

from typing import List, Tuple, Optional, Any
import os
import pickle
import numpy as np
import torch
from nn.features import apply_random_symmetry

BUFFER_SCHEMA_VERSION = 2


class _BufferView:
    """Compatibility sequence view exposing chronological access (oldest to newest)."""

    def __init__(self, replay_buffer: "ReplayBuffer"):
        self._rb = replay_buffer

    def __len__(self) -> int:
        return self._rb.size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        size = self._rb.size
        if idx < 0:
            idx += size
        if idx < 0 or idx >= size:
            raise IndexError(f"Buffer index {idx} out of range for size {size}")

        if self._rb.size < self._rb.capacity:
            slot = idx
        else:
            slot = (self._rb.write_index + idx) % self._rb.capacity

        s = self._rb.states[slot].astype(np.float32)
        pi = self._rb.pis[slot].copy()
        z = float(self._rb.zs[slot, 0])
        return (s, pi, z)


class ReplayBuffer:
    """
    High-performance ring buffer storing (state, pi, z) experience tuples.

    Key technical design:
      - Memory efficiency: Preallocated contiguous numpy arrays. States are stored
        as uint8 (17 binary planes), reducing RAM by 75% compared to float32.
      - O(1) Random Access: Directly indexes preallocated arrays without linked list/deque overhead.
      - D4 Augmentation: Vectorized and fused with batch sampling.
      - Persistence: Schema v2 with atomic writes (.tmp -> os.replace) and migration for Schema v1.
    """

    def __init__(self, capacity: int = 200_000, board_size: Optional[int] = None):
        self.capacity = int(capacity)
        self.board_size = board_size
        self.size = 0
        self.write_index = 0

        self.states: Optional[np.ndarray] = None
        self.pis: Optional[np.ndarray] = None
        self.zs: Optional[np.ndarray] = None

        if board_size is not None:
            self._allocate(board_size)

    def _allocate(self, board_size: int) -> None:
        """Preallocate contiguous numpy arrays."""
        self.board_size = board_size
        action_size = board_size * board_size + 1
        self.states = np.zeros((self.capacity, 17, board_size, board_size), dtype=np.uint8)
        self.pis = np.zeros((self.capacity, action_size), dtype=np.float32)
        self.zs = np.zeros((self.capacity, 1), dtype=np.float32)

    def push(self, state: np.ndarray, pi: np.ndarray, z: float) -> None:
        """Store a single training sample."""
        if self.states is None:
            self._allocate(board_size=state.shape[-1])

        self.states[self.write_index] = (state > 0.5).astype(np.uint8) if state.dtype != np.uint8 else state
        self.pis[self.write_index] = pi.astype(np.float32)
        self.zs[self.write_index, 0] = float(z)

        self.write_index = (self.write_index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def push_game(self, game_samples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Store all samples from a completed game."""
        for state, pi, z in game_samples:
            self.push(state, pi, z)

    def sample_batch(
        self, batch_size: int, augment: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch of training data in O(1) time per item.

        If augment is True, applies random D4 symmetry transformation (rotation + reflection)
        directly to each sampled state and policy target.

        Returns:
            (states, pis, zs) as PyTorch tensors:
                states: shape (B, 17, N, N), float32
                pis: shape (B, N*N + 1), float32
                zs: shape (B, 1), float32
        """
        if self.size == 0 or self.states is None:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")

        replace = self.size < batch_size
        indices = np.random.choice(self.size, size=batch_size, replace=replace)

        if not augment:
            batch_states = self.states[indices].astype(np.float32)
            batch_pis = self.pis[indices].copy()
        else:
            batch_states = np.empty(
                (batch_size, 17, self.board_size, self.board_size), dtype=np.float32
            )
            batch_pis = np.empty(
                (batch_size, self.board_size * self.board_size + 1), dtype=np.float32
            )
            for i, idx in enumerate(indices):
                s = self.states[idx].astype(np.float32)
                pi = self.pis[idx]
                s_aug, pi_aug = apply_random_symmetry(s, pi)
                batch_states[i] = s_aug
                batch_pis[i] = pi_aug

        batch_zs = self.zs[indices].copy()

        states_tensor = torch.from_numpy(batch_states)
        pis_tensor = torch.from_numpy(batch_pis)
        zs_tensor = torch.from_numpy(batch_zs)

        return states_tensor, pis_tensor, zs_tensor

    @property
    def buffer(self) -> _BufferView:
        """Chronological sequence view compatible with legacy deque interface."""
        return _BufferView(self)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        return self.buffer[idx]

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        """Reset the buffer to empty state."""
        self.size = 0
        self.write_index = 0
        if self.states is not None:
            self.states.fill(0)
            self.pis.fill(0)
            self.zs.fill(0)

    def save(self, filepath: str) -> None:
        """Serialize buffer to disk atomically with Schema v2 metadata."""
        tmp_path = filepath + ".tmp"
        payload = {
            "schema_version": BUFFER_SCHEMA_VERSION,
            "capacity": self.capacity,
            "size": self.size,
            "write_index": self.write_index,
            "board_size": self.board_size,
        }

        if self.states is not None and self.size > 0:
            if self.size < self.capacity:
                payload["states"] = self.states[: self.size].copy()
                payload["pis"] = self.pis[: self.size].copy()
                payload["zs"] = self.zs[: self.size].copy()
            else:
                payload["states"] = self.states.copy()
                payload["pis"] = self.pis.copy()
                payload["zs"] = self.zs.copy()
        else:
            payload["states"] = None
            payload["pis"] = None
            payload["zs"] = None

        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, filepath)

    def load(self, filepath: str) -> None:
        """
        Deserialize buffer from disk.
        Supports Schema v2 and automatically migrates legacy Schema v1 files.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list):
            # Migration from Schema v1 (plain list of tuples)
            self.clear()
            for item in data:
                if isinstance(item, (tuple, list)) and len(item) == 3:
                    s, pi, z = item
                    self.push(s, pi, z)
                else:
                    raise ValueError("Corrupted Schema v1 buffer: invalid item format")
            return

        if isinstance(data, dict):
            version = data.get("schema_version")
            if version == BUFFER_SCHEMA_VERSION:
                loaded_board_size = data.get("board_size")
                if self.board_size is not None and loaded_board_size is not None:
                    if self.board_size != loaded_board_size:
                        raise ValueError(
                            f"Buffer board_size mismatch: file has {loaded_board_size}, but buffer configured with {self.board_size}"
                        )

                board_size = loaded_board_size if loaded_board_size is not None else self.board_size
                if board_size is not None and self.states is None:
                    self._allocate(board_size)

                loaded_size = data.get("size", 0)
                states = data.get("states")
                pis = data.get("pis")
                zs = data.get("zs")

                if states is None or loaded_size == 0:
                    self.clear()
                    return

                if self.states is None:
                    self._allocate(states.shape[-1])

                if data.get("capacity") == self.capacity and loaded_size == len(states):
                    if loaded_size < self.capacity:
                        self.states[:loaded_size] = states
                        self.pis[:loaded_size] = pis
                        self.zs[:loaded_size] = zs
                        self.write_index = data.get("write_index", loaded_size)
                        self.size = loaded_size
                    else:
                        self.states[:] = states
                        self.pis[:] = pis
                        self.zs[:] = zs
                        self.write_index = data.get("write_index", 0)
                        self.size = loaded_size
                else:
                    # Capacity changed: migrate chronologically
                    self.clear()
                    old_cap = data.get("capacity", len(states))
                    old_w_idx = data.get("write_index", 0)
                    if loaded_size < old_cap:
                        for i in range(loaded_size):
                            self.push(states[i], pis[i], float(zs[i, 0]))
                    else:
                        for i in range(loaded_size):
                            slot = (old_w_idx + i) % old_cap
                            self.push(states[slot], pis[slot], float(zs[slot, 0]))
                return

            raise ValueError(f"Unsupported buffer schema version: {version}")

        raise ValueError(f"Unrecognized buffer file format: {type(data)}")
