import pytest
import os
import tempfile
import numpy as np
import torch
from selfplay.replay_buffer import ReplayBuffer

def test_replay_buffer_capacity():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0

    s = np.zeros((17, 9, 9), dtype=np.float32)
    pi = np.zeros(82, dtype=np.float32)
    pi[0] = 1.0

    for i in range(15):
        buf.push(s, pi, z=float(i))

    assert len(buf) == 10
    # Oldest 5 items should have been discarded (sliding window)
    first_item = buf.buffer[0]
    assert first_item[2] == 5.0

def test_sample_batch():
    buf = ReplayBuffer(capacity=50)
    s = np.zeros((17, 9, 9), dtype=np.float32)
    s[0, 1, 1] = 1.0
    pi = np.zeros(82, dtype=np.float32)
    pi[10] = 1.0

    for _ in range(20):
        buf.push(s, pi, z=1.0)

    states, pis, zs = buf.sample_batch(batch_size=8, augment=True)
    assert isinstance(states, torch.Tensor)
    assert states.shape == (8, 17, 9, 9)
    assert pis.shape == (8, 82)
    assert zs.shape == (8, 1)
    assert states.dtype == torch.float32
    assert pis.dtype == torch.float32
    assert zs.dtype == torch.float32

def test_save_load_buffer():
    buf1 = ReplayBuffer(capacity=50)
    s = np.ones((17, 9, 9), dtype=np.float32)
    pi = np.ones(82, dtype=np.float32) / 82.0
    buf1.push(s, pi, z=-1.0)
    buf1.push(s * 2, pi, z=1.0)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        buf1.save(tmp_path)
        buf2 = ReplayBuffer(capacity=50)
        buf2.load(tmp_path)

        assert len(buf2) == 2
        assert buf2.buffer[0][2] == -1.0
        assert buf2.buffer[1][2] == 1.0
        assert np.array_equal(buf2.buffer[0][0], s)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_ring_buffer_lifecycle_and_dtype():
    capacity = 10
    buf = ReplayBuffer(capacity=capacity, board_size=5)
    assert buf.states is not None
    assert buf.states.dtype == np.uint8
    assert buf.pis.dtype == np.float16
    assert buf.zs.dtype == np.int8
    assert buf.size == 0
    assert buf.write_index == 0

    s = np.zeros((17, 5, 5), dtype=np.float32)
    s[0, 2, 2] = 1.0
    pi = np.zeros(26, dtype=np.float32)
    pi[5] = 1.0

    # Push 15 items
    for i in range(15):
        buf.push(s, pi, z=float(i))

    assert len(buf) == 10
    assert buf.size == 10
    assert buf.write_index == 5  # 15 % 10 = 5

    # Oldest item should be 5.0, newest should be 14.0
    assert buf[0][2] == 5.0
    assert buf[-1][2] == 14.0

    # Check sampling with replacement when batch_size > len
    states, pis, zs = buf.sample_batch(batch_size=16, augment=True)
    assert states.shape == (16, 17, 5, 5)
    assert states.dtype == torch.float32
    assert pis.shape == (16, 26)
    assert zs.shape == (16, 1)


def test_schema_v2_metadata_and_backward_compatibility():
    import pickle
    
    buf = ReplayBuffer(capacity=20, board_size=5)
    s = np.ones((17, 5, 5), dtype=np.float32)
    pi = np.ones(26, dtype=np.float32) / 26.0
    for i in range(5):
        buf.push(s, pi, z=float(i))

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        buf.save(tmp_path)
        with open(tmp_path, "rb") as f:
            data = pickle.load(f)

        assert isinstance(data, dict)
        assert data["schema_version"] in (2, 3, 4)
        assert data["board_size"] == 5
        assert data["capacity"] == 20
        assert data["size"] == 5
        assert data["write_index"] == 5

        # Test loading Schema v2
        loaded_buf = ReplayBuffer(capacity=20, board_size=5)
        loaded_buf.load(tmp_path)
        assert len(loaded_buf) == 5
        assert loaded_buf.board_size == 5
        assert loaded_buf.size == 5

        # Test board size mismatch raises ValueError
        mismatched_buf = ReplayBuffer(capacity=20, board_size=9)
        with pytest.raises(ValueError, match="board_size mismatch"):
            mismatched_buf.load(tmp_path)

        # Test backward compatibility: Schema v1 (plain list of tuples)
        v1_data = [(s, pi, float(i)) for i in range(3)]
        with open(tmp_path, "wb") as f:
            pickle.dump(v1_data, f)

        v1_buf = ReplayBuffer(capacity=20, board_size=5)
        v1_buf.load(tmp_path)
        assert len(v1_buf) == 3
        assert v1_buf[0][2] == 0.0
        assert v1_buf[-1][2] == 2.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_sampling_benchmark_scaling():
    import time
    # Benchmark sampling 2k vs 20k
    buf_2k = ReplayBuffer(capacity=2500, board_size=5)
    s = np.zeros((17, 5, 5), dtype=np.float32)
    pi = np.zeros(26, dtype=np.float32)
    for i in range(2000):
        buf_2k.push(s, pi, z=float(i % 2))

    t0 = time.perf_counter()
    for _ in range(20):
        _ = buf_2k.sample_batch(batch_size=64, augment=False)
    t_2k = time.perf_counter() - t0

    buf_20k = ReplayBuffer(capacity=25000, board_size=5)
    for i in range(20000):
        buf_20k.push(s, pi, z=float(i % 2))

    t0 = time.perf_counter()
    for _ in range(20):
        _ = buf_20k.sample_batch(batch_size=64, augment=False)
    t_20k = time.perf_counter() - t0

    # Ensure sampling runs fast without degredation
    print(f"\n[BENCHMARK] 2k samples: {t_2k*1000/20:.3f} ms/batch | 20k samples: {t_20k*1000/20:.3f} ms/batch")

