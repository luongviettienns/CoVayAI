"""Benchmark end-to-end self-play pipeline: positions/s, games/hour, and memory footprint.

Measures throughput of self-play generation and tracks peak RAM and VRAM.
"""

from typing import Dict, Any, List
import time
import ctypes
import torch
from nn.model import AlphaZeroNet
from selfplay.worker import play_selfplay_game


def get_system_ram_usage_mb() -> float:
    """Query current system RAM usage in MB using Windows API."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        used_phys_bytes = stat.ullTotalPhys - stat.ullAvailPhys
        return used_phys_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def benchmark_selfplay_throughput(
    board_size: int,
    num_simulations: int = 32,
    max_moves: int = 60,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Execute a self-play game and measure throughput and memory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlphaZeroNet(board_size=board_size, num_res_blocks=6, num_filters=96).to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    ram_before_mb = get_system_ram_usage_mb()
    start_time = time.perf_counter()

    game_result = play_selfplay_game(
        model=model,
        board_size=board_size,
        num_simulations=num_simulations,
        max_moves=max_moves,
        c_puct=1.5,
        temp_moves=10,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.perf_counter() - start_time

    ram_after_mb = get_system_ram_usage_mb()
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    num_moves = len(game_result.samples)
    positions_per_sec = num_moves / elapsed_sec if elapsed_sec > 0 else 0.0
    games_per_hour = (3600.0 / elapsed_sec) if elapsed_sec > 0 else 0.0

    return {
        "board_size": board_size,
        "num_simulations": num_simulations,
        "num_moves": num_moves,
        "winner": game_result.winner,
        "termination_reason": game_result.end_reason,
        "elapsed_sec": elapsed_sec,
        "positions_per_sec": positions_per_sec,
        "games_per_hour": games_per_hour,
        "peak_vram_mb": peak_vram_mb,
        "ram_used_mb": ram_after_mb,
    }


def run_pipeline_benchmarks(
    device: torch.device = None,
    configs: List[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Execute selfplay pipeline benchmarks."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configs is None:
        configs = [
            {"board_size": 9, "num_simulations": 32, "max_moves": 60},
            {"board_size": 9, "num_simulations": 64, "max_moves": 60},
            {"board_size": 19, "num_simulations": 16, "max_moves": 30},
            {"board_size": 19, "num_simulations": 32, "max_moves": 30},
        ]

    results = []
    print(f"Running Self-Play Pipeline Benchmarks on device: {device.type.upper()}")

    for cfg in configs:
        size = cfg["board_size"]
        sims = cfg["num_simulations"]
        res = benchmark_selfplay_throughput(
            board_size=size,
            num_simulations=sims,
            max_moves=cfg["max_moves"],
            device=device,
        )
        res["benchmark"] = "pipeline"
        results.append(res)
        print(
            f"[Pipeline {size}x{size} | {sims:2d} sims | {res['num_moves']:2d} moves | {res['termination_reason']}] "
            f"Positions: {res['positions_per_sec']:5.2f} pos/s | "
            f"Games/hr: {res['games_per_hour']:5.1f} | "
            f"Peak VRAM: {res['peak_vram_mb']:5.1f} MB"
        )

    return results


if __name__ == "__main__":
    print("=== CoVayAI Self-Play Pipeline Benchmarks (P0) ===")
    run_pipeline_benchmarks()
