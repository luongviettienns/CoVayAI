"""Benchmark board operations: board.copy/s, get_legal_moves/s, play/s.

Profiles performance on board sizes 9, 13, and 19 across opening,
middle-game, and late-game states.
"""

from typing import Dict, Any, List
import time
import numpy as np
import random
from benchmarks.fixtures import create_deterministic_board, PHASE_MOVE_TARGETS


def benchmark_board_copy(board, iterations: int = 5000) -> Dict[str, float]:
    """Measure board.copy() execution time."""
    start = time.perf_counter()
    for _ in range(iterations):
        _ = board.copy()
    elapsed = time.perf_counter() - start
    ops_per_sec = iterations / elapsed if elapsed > 0 else 0.0
    mean_us = (elapsed / iterations) * 1e6
    return {
        "iterations": iterations,
        "elapsed_sec": elapsed,
        "ops_per_sec": ops_per_sec,
        "mean_us": mean_us,
    }


def benchmark_get_legal_moves(board, iterations: int = 500) -> Dict[str, float]:
    """Measure board.get_legal_moves() execution time."""
    start = time.perf_counter()
    for _ in range(iterations):
        _ = board.get_legal_moves()
    elapsed = time.perf_counter() - start
    ops_per_sec = iterations / elapsed if elapsed > 0 else 0.0
    mean_us = (elapsed / iterations) * 1e6
    return {
        "iterations": iterations,
        "elapsed_sec": elapsed,
        "ops_per_sec": ops_per_sec,
        "mean_us": mean_us,
    }


def benchmark_play_moves(board, iterations: int = 500) -> Dict[str, float]:
    """Measure board.play() execution time on a fresh copy per cycle."""
    legal_mask = board.get_legal_moves()
    legal_actions = np.flatnonzero(legal_mask)
    action = int(legal_actions[0]) if len(legal_actions) > 0 else board.pass_action
    
    start = time.perf_counter()
    for _ in range(iterations):
        b = board.copy()
        b.play(action)
    elapsed = time.perf_counter() - start
    ops_per_sec = iterations / elapsed if elapsed > 0 else 0.0
    mean_us = (elapsed / iterations) * 1e6
    return {
        "iterations": iterations,
        "elapsed_sec": elapsed,
        "ops_per_sec": ops_per_sec,
        "mean_us": mean_us,
    }


def run_board_benchmarks(seed: int = 42) -> List[Dict[str, Any]]:
    """Execute all board benchmarks across sizes and phases."""
    results = []
    sizes = [9, 13, 19]
    phases = ["opening", "middle_game", "late_game"]

    iter_config = {
        9: {"copy": 10000, "legal": 1000, "play": 1000},
        13: {"copy": 5000, "legal": 500, "play": 500},
        19: {"copy": 2000, "legal": 200, "play": 200},
    }

    for size in sizes:
        for phase in phases:
            board = create_deterministic_board(size=size, phase=phase, seed=seed)
            cfg = iter_config[size]

            copy_res = benchmark_board_copy(board, iterations=cfg["copy"])
            legal_res = benchmark_get_legal_moves(board, iterations=cfg["legal"])
            play_res = benchmark_play_moves(board, iterations=cfg["play"])

            record = {
                "benchmark": "board",
                "size": size,
                "phase": phase,
                "move_count": board.move_count,
                "copy_ops_per_sec": copy_res["ops_per_sec"],
                "copy_mean_us": copy_res["mean_us"],
                "legal_ops_per_sec": legal_res["ops_per_sec"],
                "legal_mean_us": legal_res["mean_us"],
                "play_ops_per_sec": play_res["ops_per_sec"],
                "play_mean_us": play_res["mean_us"],
            }
            results.append(record)
            print(
                f"[Board {size}x{size} | {phase:11s}] "
                f"Copy: {copy_res['ops_per_sec']:8.0f} ops/s ({copy_res['mean_us']:6.2f} µs) | "
                f"Legal: {legal_res['ops_per_sec']:6.0f} ops/s ({legal_res['mean_us']:7.2f} µs) | "
                f"Play: {play_res['ops_per_sec']:6.0f} ops/s ({play_res['mean_us']:7.2f} µs)"
            )

    return results


if __name__ == "__main__":
    print("=== CoVayAI Board Operation Benchmarks (P0) ===")
    run_board_benchmarks()
