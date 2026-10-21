"""Benchmark MCTS search performance: nodes/s (simulations/s) and search latency.

Profiles MCTS search across board sizes [9, 13, 19] and game phases [opening, middle_game]
with varying simulation counts [32, 64, 128].
"""

from typing import Dict, Any, List
import time
import torch
from go.board import Board
from mcts.search import MCTS
from nn.model import AlphaZeroNet
from benchmarks.fixtures import create_deterministic_board


def benchmark_single_mcts_search(
    mcts: MCTS,
    board: Board,
    device: torch.device,
    warmup: bool = True,
) -> Dict[str, float]:
    """Measure duration and throughput for a single MCTS search call."""
    if warmup:
        _ = mcts.search(board.copy(), temperature=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    start = time.perf_counter()
    _ = mcts.search(board.copy(), temperature=0.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    sims = mcts.num_simulations
    nodes_per_sec = sims / elapsed if elapsed > 0 else 0.0
    latency_ms = elapsed * 1e3

    return {
        "simulations": sims,
        "elapsed_sec": elapsed,
        "latency_ms": latency_ms,
        "nodes_per_sec": nodes_per_sec,
    }


def run_mcts_benchmarks(
    device: torch.device = None,
    sizes: List[int] = None,
    phases: List[str] = None,
    sim_counts: List[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Execute MCTS search benchmarks."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sizes is None:
        sizes = [9, 13, 19]
    if phases is None:
        phases = ["opening", "middle_game"]
    if sim_counts is None:
        sim_counts = [32, 64, 128]

    results = []
    print(f"Running MCTS Search Benchmarks on device: {device.type.upper()}")

    for size in sizes:
        model = AlphaZeroNet(board_size=size, num_res_blocks=6, num_filters=96).to(device)
        model.eval()

        for phase in phases:
            board = create_deterministic_board(size=size, phase=phase, seed=seed)
            for sims in sim_counts:
                mcts = MCTS(
                    model=model,
                    num_simulations=sims,
                    c_puct=1.5,
                    add_noise=False,
                    device=device,
                )
                res = benchmark_single_mcts_search(mcts, board, device=device, warmup=True)
                record = {
                    "benchmark": "mcts",
                    "size": size,
                    "phase": phase,
                    "simulations": sims,
                    "latency_ms": res["latency_ms"],
                    "nodes_per_sec": res["nodes_per_sec"],
                }
                results.append(record)
                print(
                    f"[MCTS {size}x{size} | {phase:11s} | {sims:3d} sims] "
                    f"Throughput: {res['nodes_per_sec']:6.1f} nodes/s | "
                    f"Latency: {res['latency_ms']:7.1f} ms"
                )

    return results


if __name__ == "__main__":
    print("=== CoVayAI MCTS Search Benchmarks (P0) ===")
    run_mcts_benchmarks()
