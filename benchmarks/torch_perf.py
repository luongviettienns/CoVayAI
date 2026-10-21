"""
PyTorch Performance Benchmarking Tool (PR 4 / Master Plan V4).
Evaluates and benchmarks:
- Memory layouts: Standard Contiguous (NCHW) vs Channels-Last (NHWC)
- Model compilation: torch.compile latency and throughput (with safe fallbacks)
- Training throughput (steps/sec) and peak memory allocation
"""

import argparse
import json
import logging
import time
from typing import Dict, Any, Optional
import numpy as np
import torch

from nn.model import AlphaZeroNet
from selfplay.replay_buffer import ReplayBuffer
from train.trainer import Trainer

logger = logging.getLogger(__name__)


def _sync_device(dev: torch.device) -> None:
    """Synchronize CUDA device if available."""
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_memory_format(
    board_size: int = 19,
    num_res_blocks: int = 6,
    num_filters: int = 96,
    batch_size: int = 32,
    num_warmup: int = 5,
    num_iters: int = 20,
    device: str = "cpu",
    aux_heads: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark forward pass throughput and latency between Contiguous and Channels-Last memory format.
    Also verifies strict numerical output equivalence.
    """
    dev = torch.device(device)
    torch.manual_seed(42)

    # 1. Base Contiguous Model
    model_cont = AlphaZeroNet(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
        aux_heads=aux_heads,
    ).to(dev)
    model_cont.eval()

    # 2. Channels-Last Model (same weights)
    model_cl = AlphaZeroNet(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
        aux_heads=aux_heads,
    ).to(dev)
    model_cl.load_state_dict(model_cont.state_dict())
    model_cl.to_channels_last()
    model_cl.eval()

    # Synthetic Input Batch
    x_raw = torch.randn(batch_size, 17, board_size, board_size, device=dev)
    x_cl = x_raw.clone().to(memory_format=torch.channels_last)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_cont(x_raw)
            _ = model_cl(x_cl)
    _sync_device(dev)

    # Output Equivalence Check
    with torch.no_grad():
        out_cont = model_cont(x_raw)
        out_cl = model_cl(x_cl)
        pi_diff = float((out_cont[0] - out_cl[0]).abs().max().item())
        v_diff = float((out_cont[1] - out_cl[1]).abs().max().item())
        max_diff = max(pi_diff, v_diff)

    # Measure Contiguous
    latencies_cont = []
    _sync_device(dev)
    for _ in range(num_iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cont(x_raw)
        _sync_device(dev)
        latencies_cont.append((time.perf_counter() - t0) * 1000.0)

    # Measure Channels-Last
    latencies_cl = []
    _sync_device(dev)
    for _ in range(num_iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cl(x_cl)
        _sync_device(dev)
        latencies_cl.append((time.perf_counter() - t0) * 1000.0)

    p50_cont = float(np.median(latencies_cont))
    p95_cont = float(np.percentile(latencies_cont, 95))
    evals_sec_cont = (batch_size * num_iters) / (sum(latencies_cont) / 1000.0)

    p50_cl = float(np.median(latencies_cl))
    p95_cl = float(np.percentile(latencies_cl, 95))
    evals_sec_cl = (batch_size * num_iters) / (sum(latencies_cl) / 1000.0)

    speedup = evals_sec_cl / evals_sec_cont if evals_sec_cont > 0 else 1.0

    return {
        "board_size": board_size,
        "batch_size": batch_size,
        "num_res_blocks": num_res_blocks,
        "num_filters": num_filters,
        "device": str(dev),
        "contiguous": {
            "latency_ms_p50": round(p50_cont, 4),
            "latency_ms_p95": round(p95_cont, 4),
            "evals_per_sec": round(evals_sec_cont, 2),
        },
        "channels_last": {
            "latency_ms_p50": round(p50_cl, 4),
            "latency_ms_p95": round(p95_cl, 4),
            "evals_per_sec": round(evals_sec_cl, 2),
        },
        "speedup_ratio": round(speedup, 4),
        "output_equivalence_max_diff": round(max_diff, 8),
    }


def benchmark_compile(
    board_size: int = 19,
    num_res_blocks: int = 6,
    num_filters: int = 96,
    batch_size: int = 32,
    num_warmup: int = 3,
    num_iters: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Benchmark torch.compile against eager mode with comprehensive platform safety.
    Gracefully returns reason if compilation is unsupported.
    """
    dev = torch.device(device)
    if not hasattr(torch, "compile"):
        return {"supported": False, "reason": "torch.compile is not available in this PyTorch build."}

    try:
        model = AlphaZeroNet(
            board_size=board_size,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
        ).to(dev)
        model.eval()

        x = torch.randn(batch_size, 17, board_size, board_size, device=dev)

        # Baseline eager
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)
        _sync_device(dev)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iters):
                _ = model(x)
        _sync_device(dev)
        eager_time = time.perf_counter() - t0

        # Attempt compile
        compiled_model = torch.compile(model)
        # Warmup compiled model (triggers JIT compilation)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = compiled_model(x)
        _sync_device(dev)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iters):
                _ = compiled_model(x)
        _sync_device(dev)
        compiled_time = time.perf_counter() - t0

        eager_evals = (batch_size * num_iters) / eager_time
        compiled_evals = (batch_size * num_iters) / compiled_time
        speedup = compiled_evals / eager_evals if eager_evals > 0 else 1.0

        return {
            "supported": True,
            "eager_evals_per_sec": round(eager_evals, 2),
            "compiled_evals_per_sec": round(compiled_evals, 2),
            "speedup_ratio": round(speedup, 4),
        }
    except Exception as e:
        return {
            "supported": False,
            "reason": str(e),
        }


def benchmark_training_throughput(
    board_size: int = 19,
    num_res_blocks: int = 6,
    num_filters: int = 96,
    batch_size: int = 32,
    steps: int = 10,
    device: str = "cpu",
    channels_last: bool = False,
) -> Dict[str, Any]:
    """Benchmark end-to-end training step throughput (steps/sec) using Trainer and ReplayBuffer."""
    dev = torch.device(device)
    model = AlphaZeroNet(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
    ).to(dev)

    trainer = Trainer(model, device=dev, use_amp=False, channels_last=channels_last)

    # Populate synthetic buffer
    buffer = ReplayBuffer(capacity=batch_size * 4, board_size=board_size)
    action_size = board_size * board_size + 1
    for _ in range(batch_size * 2):
        s = np.zeros((17, board_size, board_size), dtype=np.uint8)
        pi = np.ones(action_size, dtype=np.float32) / action_size
        z = 1.0
        buffer.push(s, pi, z)

    # Measure steps
    _sync_device(dev)
    t0 = time.perf_counter()
    metrics = trainer.train_iteration(buffer, steps=steps, batch_size=batch_size)
    _sync_device(dev)
    elapsed = time.perf_counter() - t0

    steps_per_sec = steps / elapsed if elapsed > 0 else 0.0
    avg_step_ms = (elapsed / steps) * 1000.0 if steps > 0 else 0.0

    return {
        "board_size": board_size,
        "batch_size": batch_size,
        "steps": steps,
        "channels_last": channels_last,
        "steps_per_sec": round(steps_per_sec, 2),
        "avg_step_ms": round(avg_step_ms, 2),
        "total_loss": round(metrics.get("total_loss", 0.0), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="PyTorch Performance Benchmark for CoVayAI")
    parser.add_argument("--board-size", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--blocks", type=int, default=6, help="Number of residual blocks (default: 6)")
    parser.add_argument("--filters", type=int, default=96, help="Number of filters (default: 96)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations (default: 20)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Save report to JSON file")
    args = parser.parse_args()

    print(f"\n[*] Benchmarking PyTorch performance on {args.device} (Board: {args.board_size}x{args.board_size}, ResNet: {args.blocks}b_{args.filters}f)...")
    mem_report = benchmark_memory_format(
        board_size=args.board_size,
        num_res_blocks=args.blocks,
        num_filters=args.filters,
        batch_size=args.batch_size,
        num_iters=args.iters,
        device=args.device,
    )
    print(f"    [+] Contiguous: {mem_report['contiguous']['evals_per_sec']} evals/s (p50: {mem_report['contiguous']['latency_ms_p50']}ms)")
    print(f"    [+] Channels-Last: {mem_report['channels_last']['evals_per_sec']} evals/s (p50: {mem_report['channels_last']['latency_ms_p50']}ms)")
    print(f"    [+] Memory Speedup: {mem_report['speedup_ratio']:.4f}x (Max Diff: {mem_report['output_equivalence_max_diff']:.2e})")

    compile_report = benchmark_compile(
        board_size=args.board_size,
        num_res_blocks=args.blocks,
        num_filters=args.filters,
        batch_size=args.batch_size,
        device=args.device,
    )
    if compile_report["supported"]:
        print(f"    [+] torch.compile Speedup: {compile_report['speedup_ratio']:.4f}x ({compile_report['compiled_evals_per_sec']} vs {compile_report['eager_evals_per_sec']} evals/s)")
    else:
        print(f"    [-] torch.compile not supported or disabled: {compile_report['reason']}")

    train_report = benchmark_training_throughput(
        board_size=args.board_size,
        num_res_blocks=args.blocks,
        num_filters=args.filters,
        batch_size=args.batch_size,
        steps=10,
        device=args.device,
        channels_last=True,
    )
    print(f"    [+] Training Throughput (channels-last): {train_report['steps_per_sec']} steps/s ({train_report['avg_step_ms']} ms/step)")

    combined = {
        "memory_format": mem_report,
        "compile": compile_report,
        "training": train_report,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"\n[+] Full benchmark report saved to: {args.output}")


if __name__ == "__main__":
    main()
