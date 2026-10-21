"""Experience Replay Buffer for AlphaZero self-play training with compact O(1) Ring Buffer."""

from typing import List, Tuple, Optional, Any, Dict
import os
import pickle
import numpy as np
import torch
from nn.features import apply_random_symmetry

BUFFER_SCHEMA_VERSION = 3


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
        pi = self._rb.pis[slot].astype(np.float32)
        z = float(self._rb.zs[slot, 0])
        return (s, pi, z)


class ReplayBuffer:
    """
    High-performance, compact ring buffer storing (state, pi, z) experience tuples.

    Key technical design:
      - Extreme memory efficiency: States stored as uint8 (17 binary planes),
        policy targets stored as float16, outcomes stored as int8 (-1, 0, 1).
        Reduces raw RAM usage by ~3.79x compared to float32 representations.
      - O(1) Random Access: Directly indexes preallocated contiguous NumPy arrays.
      - Dynamic casting on sample_batch: Returns FP32 PyTorch tensors on the fly.
      - Rich metadata per sample/game: Tracks board_size, komi, termination_reason, seed, etc.
      - Persistence: Schema v3 with atomic writes, backward-compatible with Schema v1 and v2.
    """

    def __init__(self, capacity: int = 200_000, board_size: Optional[int] = None):
        self.capacity = int(capacity)
        self.board_size = board_size
        self.size = 0
        self.write_index = 0

        self.states: Optional[np.ndarray] = None
        self.pis: Optional[np.ndarray] = None
        self.zs: Optional[np.ndarray] = None
        self.ownerships: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None
        self.aux_masks: Optional[np.ndarray] = None
        self.metadata: Optional[List[Optional[Dict[str, Any]]]] = None

        if board_size is not None:
            self._allocate(board_size)

    def _allocate(self, board_size: int) -> None:
        """Preallocate contiguous compact NumPy arrays."""
        self.board_size = board_size
        action_size = board_size * board_size + 1
        self.states = np.zeros((self.capacity, 17, board_size, board_size), dtype=np.uint8)
        self.pis = np.zeros((self.capacity, action_size), dtype=np.float16)
        self.zs = np.zeros((self.capacity, 1), dtype=np.int8)
        self.ownerships = np.zeros((self.capacity, board_size, board_size), dtype=np.int8)
        self.scores = np.zeros((self.capacity, 1), dtype=np.float16)
        self.aux_masks = np.zeros((self.capacity, 1), dtype=np.uint8)
        self.metadata = [None] * self.capacity

    def push(
        self,
        state: np.ndarray,
        pi: np.ndarray,
        z: float,
        metadata: Optional[Dict[str, Any]] = None,
        ownership: Optional[np.ndarray] = None,
        score: Optional[float] = None,
        has_aux: bool = False,
    ) -> None:
        """Store a single training sample with optional metadata and auxiliary targets."""
        if self.states is None:
            self._allocate(board_size=state.shape[-1])

        self.states[self.write_index] = (state > 0.5).astype(np.uint8) if state.dtype != np.uint8 else state
        self.pis[self.write_index] = pi.astype(np.float16)
        self.zs[self.write_index, 0] = int(round(z)) if z != 0 else 0

        if ownership is not None and has_aux:
            self.ownerships[self.write_index] = np.round(ownership).astype(np.int8)
            self.scores[self.write_index, 0] = np.float16(score if score is not None else 0.0)
            self.aux_masks[self.write_index, 0] = 1
        elif self.ownerships is not None:
            self.ownerships[self.write_index].fill(0)
            self.scores[self.write_index, 0] = 0.0
            self.aux_masks[self.write_index, 0] = 0

        self.metadata[self.write_index] = metadata

        self.write_index = (self.write_index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def push_game(self, game_samples: Any) -> None:
        """Store all samples from a completed game or GameResult with metadata and auxiliary targets."""
        if hasattr(game_samples, "samples"):
            # GameResult instance
            meta = {
                "board_size": getattr(game_samples, "board_size", self.board_size),
                "komi": getattr(game_samples, "komi", 7.5),
                "termination_reason": getattr(game_samples, "termination_reason", "terminal"),
                "seed": getattr(game_samples, "seed", None),
                "moves": getattr(game_samples, "moves", 0),
                "winner": getattr(game_samples, "winner", 0),
            }
            has_aux = getattr(game_samples, "has_aux", False)
            term_own = getattr(game_samples, "ownership", None)
            term_score = getattr(game_samples, "score_diff", None)

            for item in game_samples.samples:
                if len(item) >= 5:
                    s, p, z, own, sc = item[:5]
                    self.push(s, p, z, metadata=meta, ownership=own, score=sc, has_aux=True)
                else:
                    s, p, z = item[:3]
                    if has_aux and term_own is not None and term_score is not None:
                        is_black = (np.mean(s[16]) > 0.5) if s.shape[0] > 16 else True
                        own_p = (term_own * (1.0 if is_black else -1.0)).astype(np.int8)
                        sc_p = float(np.tanh((term_score if is_black else -term_score) / 20.0))
                        self.push(s, p, z, metadata=meta, ownership=own_p, score=sc_p, has_aux=True)
                    else:
                        self.push(s, p, z, metadata=meta)
        else:
            for item in game_samples:
                if len(item) >= 5:
                    s, p, z, own, sc = item[:5]
                    self.push(s, p, z, ownership=own, score=sc, has_aux=True)
                elif len(item) == 4:
                    s, p, z, m = item
                    self.push(s, p, z, metadata=m)
                else:
                    s, p, z = item[:3]
                    self.push(s, p, z)

    def get_metadata(self, idx: int) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for sample at chronological index idx."""
        size = self.size
        if idx < 0:
            idx += size
        if idx < 0 or idx >= size:
            raise IndexError(f"Buffer index {idx} out of range for size {size}")

        if self.size < self.capacity:
            slot = idx
        else:
            slot = (self.write_index + idx) % self.capacity

        return self.metadata[slot] if self.metadata is not None else None

    def sample_batch(
        self, batch_size: int, augment: bool = True, return_aux: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a mini-batch of training data in O(1) time per item.
        Converts compact uint8, float16, int8 into float32 PyTorch tensors.
        If return_aux=True, also returns augmented ownership maps, score targets, and aux_masks.
        """
        if self.size == 0 or self.states is None:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")

        replace = self.size < batch_size
        indices = np.random.choice(self.size, size=batch_size, replace=replace)

        if not augment:
            batch_states = self.states[indices].astype(np.float32)
            batch_pis = self.pis[indices].astype(np.float32)
            if return_aux:
                batch_owns = (
                    self.ownerships[indices].astype(np.float32)
                    if self.ownerships is not None
                    else np.zeros((batch_size, self.board_size, self.board_size), dtype=np.float32)
                )
        else:
            batch_states = np.empty(
                (batch_size, 17, self.board_size, self.board_size), dtype=np.float32
            )
            batch_pis = np.empty(
                (batch_size, self.board_size * self.board_size + 1), dtype=np.float32
            )
            batch_owns = (
                np.empty((batch_size, self.board_size, self.board_size), dtype=np.float32)
                if return_aux
                else None
            )

            for i, idx in enumerate(indices):
                s = self.states[idx].astype(np.float32)
                pi = self.pis[idx].astype(np.float32)
                flip = bool(np.random.choice([False, True]))
                rot = int(np.random.randint(4))
                s_aug, pi_aug = apply_random_symmetry(s, pi, flip=flip, rot=rot)
                batch_states[i] = s_aug
                batch_pis[i] = pi_aug
                if return_aux and batch_owns is not None:
                    own = (
                        self.ownerships[idx].astype(np.float32)
                        if self.ownerships is not None
                        else np.zeros((self.board_size, self.board_size), dtype=np.float32)
                    )
                    own_flip = np.fliplr(own) if flip else own
                    own_rot = np.rot90(own_flip, k=rot)
                    batch_owns[i] = np.ascontiguousarray(own_rot)

        batch_zs = self.zs[indices].astype(np.float32)

        states_tensor = torch.from_numpy(batch_states)
        pis_tensor = torch.from_numpy(batch_pis)
        zs_tensor = torch.from_numpy(batch_zs)

        if not return_aux:
            return states_tensor, pis_tensor, zs_tensor

        batch_scores = (
            self.scores[indices].astype(np.float32)
            if self.scores is not None
            else np.zeros((batch_size, 1), dtype=np.float32)
        )
        batch_masks = (
            self.aux_masks[indices].astype(np.float32)
            if self.aux_masks is not None
            else np.zeros((batch_size, 1), dtype=np.float32)
        )

        owns_tensor = torch.from_numpy(batch_owns)
        scores_tensor = torch.from_numpy(batch_scores)
        masks_tensor = torch.from_numpy(batch_masks)

        return states_tensor, pis_tensor, zs_tensor, owns_tensor, scores_tensor, masks_tensor

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
        if self.ownerships is not None:
            self.ownerships.fill(0)
            self.scores.fill(0)
            self.aux_masks.fill(0)
        if self.metadata is not None:
            self.metadata = [None] * self.capacity

    def save(self, filepath: str) -> None:
        """Serialize buffer to disk atomically with Schema v3 metadata."""
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
                payload["ownerships"] = self.ownerships[: self.size].copy() if self.ownerships is not None else None
                payload["scores"] = self.scores[: self.size].copy() if self.scores is not None else None
                payload["aux_masks"] = self.aux_masks[: self.size].copy() if self.aux_masks is not None else None
                payload["metadata"] = self.metadata[: self.size] if self.metadata else None
            else:
                payload["states"] = self.states.copy()
                payload["pis"] = self.pis.copy()
                payload["zs"] = self.zs.copy()
                payload["ownerships"] = self.ownerships.copy() if self.ownerships is not None else None
                payload["scores"] = self.scores.copy() if self.scores is not None else None
                payload["aux_masks"] = self.aux_masks.copy() if self.aux_masks is not None else None
                payload["metadata"] = self.metadata.copy() if self.metadata else None
        else:
            payload["states"] = None
            payload["pis"] = None
            payload["zs"] = None
            payload["ownerships"] = None
            payload["scores"] = None
            payload["aux_masks"] = None
            payload["metadata"] = None

        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, filepath)

    def load(self, filepath: str) -> None:
        """
        Deserialize buffer from disk.
        Supports Schema v3, Schema v2 (float32 auto-downcast), and Schema v1.
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
            if version in (2, 3):
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
                meta = data.get("metadata")
                owns = data.get("ownerships")
                scores = data.get("scores")
                masks = data.get("aux_masks")

                if states is None or loaded_size == 0:
                    self.clear()
                    return

                if self.states is None:
                    self._allocate(states.shape[-1])

                # Cast pis to float16 and zs to int8 if loaded from v2 (float32)
                conv_pis = pis.astype(np.float16) if pis.dtype != np.float16 else pis
                conv_zs = np.round(zs).astype(np.int8) if zs.dtype != np.int8 else zs

                if data.get("capacity") == self.capacity and loaded_size == len(states):
                    if loaded_size < self.capacity:
                        self.states[:loaded_size] = states
                        self.pis[:loaded_size] = conv_pis
                        self.zs[:loaded_size] = conv_zs
                        if owns is not None and self.ownerships is not None:
                            self.ownerships[:loaded_size] = owns
                        if scores is not None and self.scores is not None:
                            self.scores[:loaded_size] = scores
                        if masks is not None and self.aux_masks is not None:
                            self.aux_masks[:loaded_size] = masks
                        if meta is not None:
                            self.metadata[:loaded_size] = meta
                        self.write_index = data.get("write_index", loaded_size)
                        self.size = loaded_size
                    else:
                        self.states[:] = states
                        self.pis[:] = conv_pis
                        self.zs[:] = conv_zs
                        if owns is not None and self.ownerships is not None:
                            self.ownerships[:] = owns
                        if scores is not None and self.scores is not None:
                            self.scores[:] = scores
                        if masks is not None and self.aux_masks is not None:
                            self.aux_masks[:] = masks
                        if meta is not None:
                            self.metadata[:] = meta
                        self.write_index = data.get("write_index", 0)
                        self.size = loaded_size
                else:
                    # Capacity changed: migrate chronologically
                    self.clear()
                    old_cap = data.get("capacity", len(states))
                    old_w_idx = data.get("write_index", 0)
                    if loaded_size < old_cap:
                        for i in range(loaded_size):
                            m = meta[i] if meta else None
                            self.push(states[i], conv_pis[i], float(conv_zs[i, 0]), metadata=m)
                    else:
                        for i in range(loaded_size):
                            slot = (old_w_idx + i) % old_cap
                            m = meta[slot] if meta else None
                            self.push(states[slot], conv_pis[slot], float(conv_zs[slot, 0]), metadata=m)
                return

            raise ValueError(f"Unsupported buffer schema version: {version}")

        raise ValueError(f"Unrecognized buffer file format: {type(data)}")


class MultiSizeReplayBuffer:
    """
    Experience buffer partitioning samples by board size (e.g. 9x9 and 19x19).
    Ensures homogenous batches per forward pass without padding overhead.
    """

    def __init__(self, capacity_per_size: int = 100_000):
        self.capacity_per_size = capacity_per_size
        self.buffers: Dict[int, ReplayBuffer] = {}

    def _get_or_create(self, board_size: int) -> ReplayBuffer:
        if board_size not in self.buffers:
            self.buffers[board_size] = ReplayBuffer(
                capacity=self.capacity_per_size, board_size=board_size
            )
        return self.buffers[board_size]

    def push(
        self,
        state: np.ndarray,
        pi: np.ndarray,
        z: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        board_size = state.shape[-1]
        buf = self._get_or_create(board_size)
        buf.push(state, pi, z, metadata=metadata)

    def push_game(self, game_samples_or_result: Any) -> None:
        board_size = getattr(game_samples_or_result, "board_size", None)
        if board_size is None and hasattr(game_samples_or_result, "__iter__"):
            first = next(iter(game_samples_or_result), None)
            if first is not None:
                board_size = first[0].shape[-1]
        if board_size is None:
            raise ValueError("Cannot determine board_size for multi-size buffer insertion.")
        buf = self._get_or_create(board_size)
        buf.push_game(game_samples_or_result)

    def sample_batch(
        self,
        batch_size: int,
        board_size: Optional[int] = None,
        augment: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.buffers:
            raise ValueError("Cannot sample from an empty MultiSizeReplayBuffer.")
        if board_size is not None:
            if board_size not in self.buffers or len(self.buffers[board_size]) == 0:
                raise ValueError(f"No samples for board size {board_size}")
            return self.buffers[board_size].sample_batch(batch_size=batch_size, augment=augment)
        else:
            valid_sizes = [bs for bs, b in self.buffers.items() if len(b) > 0]
            if not valid_sizes:
                raise ValueError("All multi-size buffers are empty.")
            chosen_size = int(np.random.choice(valid_sizes))
            return self.buffers[chosen_size].sample_batch(batch_size=batch_size, augment=augment)

    def __len__(self) -> int:
        return sum(len(b) for b in self.buffers.values())
