"""Benchmark neural network inference throughput (NN evaluations/s) and latency.

Tests batch sizes [1, 8, 16, 32, 64, 128] on board sizes [9, 13, 19]
with CUDA warmup and torch.cuda.synchronize() for accurate timing.
"""

from typing import Dict, Any, List
import time
import numpy as np
import torch
import torch.nn as nn
from nn.model import AlphaZeroNet


def benchmark_model_inference(
    model: nn.Module,
    batch_size: int,
    board_size: int,
    device: torch.device,
    use_fp16: bool = False,
    warmup_runs: int = 15,
    test_runs: int = 50,
) -> Dict[str, float]:
    """Benchmark inference latency and throughput for a fixed batch size."""
    model.eval()
    in_channels = 17
    # Dummy input representing batch of canonical feature planes
    x = torch.randn(batch_size, in_channels, board_size, board_size, device=device)
    if use_fp16 and device.type == "cuda":
        x = x.half()
        model = model.half()
    else:
        x = x.float()
        model = model.float()

    # Reset peak memory stats if CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Measured runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(test_runs):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_evals = batch_size * test_runs
    evals_per_sec = total_evals / elapsed if elapsed > 0 else 0.0
    latency_per_batch_ms = (elapsed / test_runs) * 1e3
    latency_per_item_ms = latency_per_batch_ms / batch_size

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "batch_size": batch_size,
        "board_size": board_size,
        "use_fp16": use_fp16,
        "evals_per_sec": evals_per_sec,
        "latency_per_batch_ms": latency_per_batch_ms,
        "latency_per_item_ms": latency_per_item_ms,
        "peak_vram_mb": peak_vram_mb,
    }


def run_inference_benchmarks(
    device: torch.device = None,
    sizes: List[int] = None,
    batch_sizes: List[int] = None,
) -> List[Dict[str, Any]]:
    """Execute all inference benchmarks across sizes and batch sizes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sizes is None:
        sizes = [9, 13, 19]
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32, 64, 128]

    results = []
    print(f"Running Inference Benchmarks on device: {device.type.upper()} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")

    for size in sizes:
        model = AlphaZeroNet(board_size=size, num_res_blocks=6, num_filters=96).to(device)
        for bsz in batch_sizes:
            # Benchmark in FP32
            res_fp32 = benchmark_model_inference(
                model=model,
                batch_size=bsz,
                board_size=size,
                device=device,
                use_fp16=False,
            )
            res_fp32["precision"] = "fp32"
            res_fp32["benchmark"] = "inference"
            results.append(res_fp32)

            print(
                f"[Infer {size}x{size} | Batch {bsz:3d} | FP32] "
                f"Throughput: {res_fp32['evals_per_sec']:7.1f} evals/s | "
                f"Latency: {res_fp32['latency_per_batch_ms']:6.2f} ms/batch ({res_fp32['latency_per_item_ms']:6.3f} ms/item) | "
                f"VRAM: {res_fp32['peak_vram_mb']:5.1f} MB"
            )

    return results


if __name__ == "__main__":
    print("=== CoVayAI Neural Network Inference Benchmarks (P0) ===")
    run_inference_benchmarks()
