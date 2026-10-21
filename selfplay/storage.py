"""Experience Storage Sharding, SHA-256 Checksums, and Atomic Persistence."""

import os
import hashlib
import time
from typing import Dict, Any, Optional, List
import numpy as np


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hexadecimal digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def save_shard(
    filepath: str,
    buffer,
    start_idx: int = 0,
    num_samples: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save buffer samples into a compact, compressed NumPy shard atomically with SHA-256 validation.

    Args:
        filepath: Final target shard filepath (e.g. 'shard_0001.npz')
        buffer: ReplayBuffer instance
        start_idx: Offset index in buffer
        num_samples: Number of samples to write (defaults to all buffer samples)
        extra_metadata: Optional dictionary with custom run metadata

    Returns:
        Hexadecimal SHA-256 checksum string of the written shard file.
    """
    if buffer.size == 0 or buffer.states is None:
        raise ValueError("Cannot save shard from an empty ReplayBuffer.")

    total = buffer.size if num_samples is None else min(num_samples, buffer.size)
    end_idx = min(start_idx + total, buffer.size)

    states = buffer.states[start_idx:end_idx].copy()
    pis = buffer.pis[start_idx:end_idx].copy()
    zs = buffer.zs[start_idx:end_idx].copy()

    meta_list = []
    if hasattr(buffer, "metadata") and buffer.metadata is not None:
        meta_list = buffer.metadata[start_idx:end_idx]

    header = {
        "version": 3,
        "board_size": buffer.board_size,
        "num_samples": len(states),
        "timestamp": time.time(),
    }
    if extra_metadata:
        header.update(extra_metadata)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if not filepath.endswith(".npz"):
        target_path = filepath + ".npz"
    else:
        target_path = filepath

    tmp_path = target_path[:-4] + ".tmp.npz"

    np.savez_compressed(
        tmp_path,
        states=states,
        pis=pis,
        zs=zs,
        header=np.array(header, dtype=object),
        metadata=np.array(meta_list, dtype=object),
    )

    checksum = compute_file_sha256(tmp_path)
    os.replace(tmp_path, target_path)

    return checksum


def load_shard(
    filepath: str,
    buffer,
    expected_checksum: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load buffer samples from a compact NumPy shard with optional SHA-256 verification.

    Args:
        filepath: Path to shard file (.npz)
        buffer: Target ReplayBuffer to populate
        expected_checksum: Optional SHA-256 string to verify against

    Returns:
        Header dictionary contained within the shard.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Shard file not found: {filepath}")

    if expected_checksum is not None:
        actual_checksum = compute_file_sha256(filepath)
        if actual_checksum.lower() != expected_checksum.lower():
            raise ValueError(
                f"Checksum mismatch for shard {filepath}: "
                f"expected {expected_checksum}, got {actual_checksum} (file may be corrupted)"
            )

    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load corrupted shard file {filepath}: {e}") from e

    states = data["states"]
    pis = data["pis"]
    zs = data["zs"]
    header = data["header"].item() if "header" in data else {}
    meta_array = data["metadata"] if "metadata" in data else None

    # Check board size compatibility
    shard_board_size = header.get("board_size", states.shape[-1])
    if buffer.board_size is not None and buffer.board_size != shard_board_size:
        raise ValueError(
            f"Buffer board_size mismatch: shard has {shard_board_size}, "
            f"buffer configured with {buffer.board_size}"
        )

    for i in range(len(states)):
        m = meta_array[i] if meta_array is not None and i < len(meta_array) else None
        buffer.push(states[i], pis[i], float(zs[i, 0]), metadata=m)

    return header


class BufferShardStorage:
    """Manages versioned, checksummed experience shards on disk."""

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.storage_dir, "shards_manifest.json")

    def save_buffer_as_shard(self, buffer, shard_id: str, extra_metadata: Optional[Dict[str, Any]] = None) -> str:
        shard_filename = f"{shard_id}.npz"
        shard_path = os.path.join(self.storage_dir, shard_filename)
        return save_shard(shard_path, buffer, extra_metadata=extra_metadata)

    def load_shard_into_buffer(self, shard_id: str, buffer, expected_checksum: Optional[str] = None) -> Dict[str, Any]:
        shard_filename = f"{shard_id}.npz"
        shard_path = os.path.join(self.storage_dir, shard_filename)
        return load_shard(shard_path, buffer, expected_checksum=expected_checksum)
