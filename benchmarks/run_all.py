"""Master benchmark runner for CoVayAI P0 Baseline.

Gathers hardware environment metadata, executes board, inference, MCTS,
and pipeline benchmarks, and outputs reproducible JSON and CSV results.
"""

from typing import Dict, Any
import os
import sys
import json
import csv
import time
import platform
import subprocess
import ctypes
import torch

from benchmarks.bench_board import run_board_benchmarks
from benchmarks.bench_inference import run_inference_benchmarks
from benchmarks.bench_mcts import run_mcts_benchmarks
from benchmarks.bench_pipeline import run_pipeline_benchmarks


def get_git_commit_hash() -> str:
    """Return the current HEAD git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def get_environment_metadata() -> Dict[str, Any]:
    """Capture precise execution environment specifications."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    ram_gb = 0.0
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
        ram_gb = stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass

    return {
        "git_commit": get_git_commit_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "vram_total_gb": round(vram_gb, 2),
        "cpu_processor": platform.processor(),
        "ram_total_gb": round(ram_gb, 2),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
    }


def save_csv(filepath: str, data: list):
    """Save a list of dictionaries as a CSV file."""
    if not data:
        return
    keys = list(data[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def main():
    print("=" * 70)
    print(" CoVayAI P0: Comprehensive Baseline Performance Profiling")
    print("=" * 70)

    env = get_environment_metadata()
    print("Hardware & Software Environment:")
    for k, v in env.items():
        print(f"  {k:20s}: {v}")
    print("-" * 70)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Board Benchmarks
    print("\n[1/4] Running Board Engine Benchmarks (copy, legal moves, play)...")
    board_results = run_board_benchmarks(seed=42)
    save_csv(os.path.join(results_dir, "baseline_board.csv"), board_results)

    # 2. Inference Benchmarks
    print("\n[2/4] Running Neural Network Inference Benchmarks...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_results = run_inference_benchmarks(device=device)
    save_csv(os.path.join(results_dir, "baseline_inference.csv"), infer_results)

    # 3. MCTS Benchmarks
    print("\n[3/4] Running MCTS Search Benchmarks...")
    mcts_results = run_mcts_benchmarks(device=device, seed=42)
    save_csv(os.path.join(results_dir, "baseline_mcts.csv"), mcts_results)

    # 4. Pipeline Benchmarks
    print("\n[4/4] Running End-to-End Self-Play Pipeline Benchmarks...")
    pipeline_results = run_pipeline_benchmarks(device=device)
    save_csv(os.path.join(results_dir, "baseline_pipeline.csv"), pipeline_results)

    # Combine into Master JSON
    master_report = {
        "environment": env,
        "benchmarks": {
            "board": board_results,
            "inference": infer_results,
            "mcts": mcts_results,
            "pipeline": pipeline_results,
        },
    }

    json_path = os.path.join(results_dir, "baseline_p0.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    print("\n" + "=" * 70)
    print(f" Baseline benchmarks completed successfully!")
    print(f" Master JSON report saved to: {json_path}")
    print(f" CSV reports saved in: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
