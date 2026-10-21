"""Performance profiling and throughput benchmarks for CoVayAI components."""

import time
import pytest
import numpy as np
import torch
from go.board import Board
from nn.features import extract_features
from nn.model import AlphaZeroNet
from mcts.search import MCTS


def test_board_and_feature_microbenchmarks():
    """Benchmark Board clone, legal move generation, and feature extraction."""
    b = Board(size=9)
    # Play 10 moves to populate history and stones
    for act in [20, 21, 22, 30, 31, 32, 40, 41, 42, 50]:
        b.play(act)

    # 1. Benchmark Board.copy()
    num_iters = 1000
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = b.copy()
    t_copy = (time.perf_counter() - t0) * 1e6 / num_iters

    # 2. Benchmark get_legal_moves()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = b.get_legal_moves()
    t_legal = (time.perf_counter() - t0) * 1e6 / num_iters

    # 3. Benchmark extract_features()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = extract_features(b)
    t_feat = (time.perf_counter() - t0) * 1e6 / num_iters

    print(
        f"\n[MICROBENCHMARK 9x9]\n"
        f"  - Board.copy(): {t_copy:.2f} µs/call\n"
        f"  - Board.get_legal_moves(): {t_legal:.2f} µs/call\n"
        f"  - extract_features(b): {t_feat:.2f} µs/call"
    )
    assert t_copy < 500.0  # Must be fast (< 500 µs)
    assert t_legal < 1000.0


def test_nn_inference_scaling_and_throughput():
    """Benchmark NN inference throughput on single vs batched inputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNet(board_size=9, num_res_blocks=6, num_filters=96).to(device)
    net.eval()

    state = np.zeros((17, 9, 9), dtype=np.float32)
    # Warmup
    for _ in range(10):
        _ = net.predict(state, device=device)

    # Benchmark single predict
    num_evals = 100
    t0 = time.perf_counter()
    for _ in range(num_evals):
        _ = net.predict(state, device=device)
    t_single = (time.perf_counter() - t0) * 1000.0 / num_evals

    # Benchmark batched predict with B=16
    batch_states = np.zeros((16, 17, 9, 9), dtype=np.float32)
    for _ in range(10):
        _ = net.predict_batch(batch_states, device=device)

    num_batch_evals = 50
    t0 = time.perf_counter()
    for _ in range(num_batch_evals):
        _ = net.predict_batch(batch_states, device=device)
    t_batch = (time.perf_counter() - t0) * 1000.0 / (num_batch_evals * 16)

    print(
        f"\n[NN INFERENCE BENCHMARK ({device})]\n"
        f"  - Single predict: {t_single:.3f} ms/state ({1000.0/t_single:.1f} states/s)\n"
        f"  - Batch-16 predict: {t_batch:.3f} ms/state ({1000.0/t_batch:.1f} states/s)\n"
        f"  - Speedup factor: {t_single/t_batch:.2f}x"
    )
    assert t_single < 50.0  # Must be reasonably responsive


def test_mcts_throughput():
    """Benchmark MCTS simulations per second on 5x5 and 9x9."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNet(board_size=5, num_res_blocks=2, num_filters=32).to(device)
    net.eval()

    b = Board(size=5)
    mcts = MCTS(model=net, num_simulations=50, device=device, add_noise=False)

    # Measure 50 simulations
    t0 = time.perf_counter()
    _ = mcts.search(b)
    elapsed = time.perf_counter() - t0
    sims_per_sec = 50.0 / elapsed

    print(
        f"\n[MCTS 5x5 BENCHMARK ({device})]\n"
        f"  - 50 simulations time: {elapsed*1000.0:.2f} ms\n"
        f"  - MCTS Throughput: {sims_per_sec:.1f} sims/sec"
    )
    assert sims_per_sec > 50.0
