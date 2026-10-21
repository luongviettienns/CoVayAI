"""Tests for compact experience storage, float16 policy precision, and checksum-validated sharding."""

import os
import tempfile
import numpy as np
import pytest
import torch

from selfplay.replay_buffer import ReplayBuffer
from selfplay.storage import BufferShardStorage, save_shard, load_shard
from selfplay.worker import GameResult


def test_compact_buffer_dtypes_and_memory():
    """Verify that states are uint8, pis are float16, and zs are int8."""
    buf = ReplayBuffer(capacity=100, board_size=9)
    assert buf.states.dtype == np.uint8
    assert buf.pis.dtype == np.float16
    assert buf.zs.dtype == np.int8

    s = np.ones((17, 9, 9), dtype=np.float32)
    pi = np.full(82, 1.0 / 82.0, dtype=np.float32)
    z = 1.0

    buf.push(s, pi, z)
    assert len(buf) == 1
    assert buf.pis[0].dtype == np.float16
    assert buf.zs[0, 0] == 1
    assert buf.zs[0].dtype == np.int8

    # Sequence access should cast back to float32 and float
    sample = buf[0]
    assert sample[0].dtype == np.float32
    assert sample[1].dtype == np.float32
    assert isinstance(sample[2], float)
    assert sample[2] == 1.0


def test_float16_policy_precision_and_entropy():
    """Verify float16 policy target preserves probability distribution and entropy."""
    buf = ReplayBuffer(capacity=10, board_size=9)
    s = np.zeros((17, 9, 9), dtype=np.float32)

    # Generate a realistic peaked policy distribution (e.g. Dirichlet-like or Softmax)
    raw_logits = np.random.randn(82).astype(np.float32) * 2.0
    orig_pi = np.exp(raw_logits) / np.sum(np.exp(raw_logits))

    buf.push(s, orig_pi, z=-1.0)
    recovered_pi = buf[0][1]

    # Max difference between float32 and float16 representation
    max_diff = np.max(np.abs(orig_pi - recovered_pi))
    assert max_diff < 5e-4, f"Max policy diff {max_diff} exceeded tolerance"

    # Entropy comparison: - sum(p * log(p))
    orig_entropy = -np.sum(orig_pi * np.log(orig_pi + 1e-12))
    recov_entropy = -np.sum(recovered_pi * np.log(recovered_pi + 1e-12))
    assert np.isclose(orig_entropy, recov_entropy, atol=1e-3)


def test_sample_batch_returns_float32_tensors():
    """Verify sample_batch converts uint8, float16, int8 to float32 tensors."""
    buf = ReplayBuffer(capacity=50, board_size=9)
    s = np.zeros((17, 9, 9), dtype=np.float32)
    s[0, 0, 0] = 1.0
    pi = np.full(82, 1.0 / 82.0, dtype=np.float32)

    for i in range(20):
        z_val = 1.0 if i % 2 == 0 else -1.0
        buf.push(s, pi, z=z_val)

    states, pis, zs = buf.sample_batch(batch_size=8, augment=True)

    assert isinstance(states, torch.Tensor)
    assert states.dtype == torch.float32
    assert states.shape == (8, 17, 9, 9)

    assert isinstance(pis, torch.Tensor)
    assert pis.dtype == torch.float32
    assert pis.shape == (8, 82)
    # Probabilities should sum close to 1.0
    sums = pis.sum(dim=-1).numpy()
    np.testing.assert_allclose(sums, np.ones(8), atol=1e-3)

    assert isinstance(zs, torch.Tensor)
    assert zs.dtype == torch.float32
    assert zs.shape == (8, 1)
    for val in zs.flatten().tolist():
        assert val in (-1.0, 1.0)


def test_metadata_extraction_from_game_result():
    """Verify push_game captures and stores sample-level metadata from GameResult."""
    buf = ReplayBuffer(capacity=50, board_size=9)
    s = np.zeros((17, 9, 9), dtype=np.float32)
    pi = np.full(82, 1.0 / 82.0, dtype=np.float32)

    game = GameResult(
        samples=[(s, pi, 1.0), (s, pi, -1.0)],
        winner=1,
        black_score=10.0,
        white_score=0.0,
        moves=2,
        end_reason="two_passes",
        termination_reason="two_passes",
        is_truncated=False,
        board_size=9,
        komi=7.5,
        seed=12345,
    )

    buf.push_game(game)
    assert len(buf) == 2
    meta = buf.get_metadata(0)
    assert meta["board_size"] == 9
    assert meta["komi"] == 7.5
    assert meta["termination_reason"] == "two_passes"
    assert meta["seed"] == 12345


def test_storage_shard_checksum_and_atomic_write():
    """Verify BufferShardStorage writes atomically with SHA-256 and detects tampering."""
    buf = ReplayBuffer(capacity=50, board_size=9)
    s = np.ones((17, 9, 9), dtype=np.float32)
    pi = np.full(82, 1.0 / 82.0, dtype=np.float32)
    for i in range(10):
        buf.push(s * (i + 1), pi, z=1.0 if i % 2 == 0 else -1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = os.path.join(tmpdir, "shard_0001.npz")
        checksum = save_shard(shard_path, buf)
        assert os.path.exists(shard_path)
        assert len(checksum) == 64  # SHA-256 hex string

        # Load shard into new buffer
        loaded_buf = ReplayBuffer(capacity=50, board_size=9)
        load_shard(shard_path, loaded_buf, expected_checksum=checksum)
        assert len(loaded_buf) == 10
        assert np.array_equal(loaded_buf.states[:10], buf.states[:10])
        np.testing.assert_allclose(loaded_buf.pis[:10], buf.pis[:10], atol=1e-3)
        assert np.array_equal(loaded_buf.zs[:10], buf.zs[:10])

        # Tamper with file
        with open(shard_path, "r+b") as f:
            f.seek(100)
            f.write(b"\xFF\xFF")

        corrupt_buf = ReplayBuffer(capacity=50, board_size=9)
        with pytest.raises(ValueError, match="Checksum mismatch|corrupted"):
            load_shard(shard_path, corrupt_buf, expected_checksum=checksum)


def test_multi_size_replay_buffer():
    """Verify MultiSizeReplayBuffer groups samples by board size cleanly."""
    from selfplay.replay_buffer import MultiSizeReplayBuffer
    multi_buf = MultiSizeReplayBuffer(capacity_per_size=50)

    s9 = np.ones((17, 9, 9), dtype=np.float32)
    pi9 = np.full(82, 1.0 / 82.0, dtype=np.float32)
    s19 = np.ones((17, 19, 19), dtype=np.float32)
    pi19 = np.full(362, 1.0 / 362.0, dtype=np.float32)

    multi_buf.push(s9, pi9, 1.0)
    multi_buf.push(s19, pi19, -1.0)
    assert len(multi_buf) == 2

    # Sample specifically from board size 9
    st9, p9, z9 = multi_buf.sample_batch(batch_size=4, board_size=9, augment=False)
    assert st9.shape == (4, 17, 9, 9)
    assert p9.shape == (4, 82)

    # Sample specifically from board size 19
    st19, p19, z19 = multi_buf.sample_batch(batch_size=4, board_size=19, augment=False)
    assert st19.shape == (4, 17, 19, 19)
    assert p19.shape == (4, 362)


def test_buffer_shard_storage_directory_manager():
    """Verify BufferShardStorage class persists shards in a designated directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = BufferShardStorage(storage_dir=tmpdir)
        buf = ReplayBuffer(capacity=50, board_size=9)
        s = np.zeros((17, 9, 9), dtype=np.float32)
        pi = np.full(82, 1.0 / 82.0, dtype=np.float32)
        for i in range(5):
            buf.push(s, pi, 1.0)

        checksum = storage.save_buffer_as_shard(buf, shard_id="shard_01")
        assert len(checksum) == 64

        new_buf = ReplayBuffer(capacity=50, board_size=9)
        storage.load_shard_into_buffer("shard_01", new_buf, expected_checksum=checksum)
        assert len(new_buf) == 5


def test_loss_tolerance_between_float32_and_float16_targets():
    """Verify cross-entropy policy loss difference with float16 targets is < 1e-4."""
    logits = torch.randn(16, 362, requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Simulated true target policy (float32)
    raw_target = torch.softmax(torch.randn(16, 362), dim=-1)
    target_fp32 = raw_target.clone()
    target_fp16 = raw_target.half().float()

    loss_fp32 = -(target_fp32 * log_probs).sum(dim=-1).mean()
    loss_fp16 = -(target_fp16 * log_probs).sum(dim=-1).mean()

    abs_loss_diff = torch.abs(loss_fp32 - loss_fp16).item()
    assert abs_loss_diff < 1e-4, f"Policy loss diff {abs_loss_diff} exceeded tolerance 1e-4"

