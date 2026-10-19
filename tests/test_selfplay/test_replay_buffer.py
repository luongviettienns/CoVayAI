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
