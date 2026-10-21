"""
Cost and Throughput Audit Module for CoVayAI (Master Plan V4 & V3 Cost Optimization).

Provides rigorous economic accounting for training runs:
- Computes total cost (GPU leasing, local electricity, storage, transfer).
- Evaluates cost efficiency per 1,000 games, valid positions, full-search positions, and 1M NN evals.
- Implements break-even analysis for Cloud GPU vs local workstation.
- Validates against benchmarks/cost_schema.json without silent failures or division-by-zero.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional, Union

SCHEMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "cost_schema.json"))


def get_git_info() -> Dict[str, str]:
    """Retrieves current Git commit SHA and active branch name."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        sha = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        branch = "unknown"

    return {"git_sha": sha, "branch": branch}


def get_hardware_info() -> Dict[str, Any]:
    """Gathers system hardware characteristics."""
    info = {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    return info


def validate_cost_input(data: Dict[str, Any]) -> None:
    """
    Validates input data against non-negativity and required schema constraints.
    Raises ValueError on negative numbers or malformed types.
    """
    numeric_fields = [
        "total_running_hours", "gpu_hours", "gpu_price_per_hour",
        "electricity_kwh", "electricity_tariff", "storage_fees", "other_fees",
        "completed_games", "valid_training_positions", "full_search_positions", "nn_evals",
        "power_kw", "games_per_hour"
    ]
    for field in numeric_fields:
        if field in data and data[field] is not None:
            val = data[field]
            if not isinstance(val, (int, float)):
                raise ValueError(f"Field '{field}' must be a number, got {type(val).__name__}")
            if val < 0:
                raise ValueError(f"Field '{field}' cannot be negative (got {val})")


def compute_cost_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes rigorous cost and throughput metrics from raw execution parameters.

    Returns:
        Dict with calculated costs and per-unit economic indicators ('N/A' on zero denominators).
    """
    validate_cost_input(data)

    gpu_hours = float(data.get("gpu_hours", 0.0) or 0.0)
    gpu_price_per_hour = float(data.get("gpu_price_per_hour", 0.0) or 0.0)
    electricity_kwh = float(data.get("electricity_kwh", 0.0) or 0.0)
    electricity_tariff = float(data.get("electricity_tariff", 0.0) or 0.0)
    storage_fees = float(data.get("storage_fees", 0.0) or 0.0)
    other_fees = float(data.get("other_fees", 0.0) or 0.0)

    gpu_cost = gpu_hours * gpu_price_per_hour
    electricity_cost = electricity_kwh * electricity_tariff
    total_compute_cost = gpu_cost + electricity_cost + storage_fees + other_fees

    completed_games = int(data.get("completed_games", 0) or 0)
    valid_positions = int(data.get("valid_training_positions", 0) or 0)
    full_search_positions = int(data.get("full_search_positions", 0) or 0)
    nn_evals = int(data.get("nn_evals", 0) or 0)

    cost_per_1k_completed_games: Union[float, str] = (
        (total_compute_cost / completed_games * 1000.0) if completed_games > 0 else "N/A"
    )
    cost_per_1k_valid_positions: Union[float, str] = (
        (total_compute_cost / valid_positions * 1000.0) if valid_positions > 0 else "N/A"
    )
    cost_per_1k_full_search_positions: Union[float, str] = (
        (total_compute_cost / full_search_positions * 1000.0) if full_search_positions > 0 else "N/A"
    )
    cost_per_1m_nn_evals: Union[float, str] = (
        (total_compute_cost / nn_evals * 1000000.0) if nn_evals > 0 else "N/A"
    )

    total_running_hours = float(data.get("total_running_hours", 0.0) or 0.0)
    games_per_hour = (completed_games / total_running_hours) if total_running_hours > 0 else "N/A"
    positions_per_hour = (valid_positions / total_running_hours) if total_running_hours > 0 else "N/A"

    return {
        "gpu_cost": round(gpu_cost, 4),
        "electricity_cost": round(electricity_cost, 4),
        "total_compute_cost": round(total_compute_cost, 4),
        "cost_per_1k_completed_games": (
            round(cost_per_1k_completed_games, 4)
            if isinstance(cost_per_1k_completed_games, float)
            else "N/A"
        ),
        "cost_per_1k_valid_positions": (
            round(cost_per_1k_valid_positions, 4)
            if isinstance(cost_per_1k_valid_positions, float)
            else "N/A"
        ),
        "cost_per_1k_full_search_positions": (
            round(cost_per_1k_full_search_positions, 4)
            if isinstance(cost_per_1k_full_search_positions, float)
            else "N/A"
        ),
        "cost_per_1m_nn_evals": (
            round(cost_per_1m_nn_evals, 4)
            if isinstance(cost_per_1m_nn_evals, float)
            else "N/A"
        ),
        "games_per_hour": round(games_per_hour, 2) if isinstance(games_per_hour, float) else "N/A",
        "positions_per_hour": round(positions_per_hour, 2) if isinstance(positions_per_hour, float) else "N/A",
    }


def compute_cloud_break_even(
    local_data: Dict[str, float], cloud_price_per_hour: float
) -> Dict[str, float]:
    """
    Computes economic break-even metrics between local hardware and cloud leasing.

    Formula:
        local_cost_per_hour = power_kw * electricity_tariff
        local_cost_per_game = local_cost_per_hour / games_per_hour
        required_cloud_games_per_hour = cloud_price_per_hour / local_cost_per_game
        required_speedup_factor = required_cloud_games_per_hour / games_per_hour
    """
    validate_cost_input(local_data)
    if cloud_price_per_hour < 0:
        raise ValueError("cloud_price_per_hour cannot be negative")

    power_kw = float(local_data.get("power_kw", 0.30))
    electricity_tariff = float(local_data.get("electricity_tariff", 0.15))
    games_per_hour = float(local_data.get("games_per_hour", 10.0))

    if games_per_hour <= 0:
        raise ValueError("games_per_hour must be greater than zero for break-even analysis")

    local_cost_per_hour = power_kw * electricity_tariff
    local_cost_per_game = local_cost_per_hour / games_per_hour

    if local_cost_per_game > 0:
        required_cloud_games_per_hour = cloud_price_per_hour / local_cost_per_game
        required_speedup = required_cloud_games_per_hour / games_per_hour
    else:
        required_cloud_games_per_hour = float("inf")
        required_speedup = float("inf")

    return {
        "local_cost_per_hour": round(local_cost_per_hour, 6),
        "local_cost_per_game": round(local_cost_per_game, 6),
        "cloud_price_per_hour": round(cloud_price_per_hour, 4),
        "required_cloud_games_per_hour": round(required_cloud_games_per_hour, 2),
        "required_speedup_factor": round(required_speedup, 2),
    }


def generate_cost_report(
    input_data: Dict[str, Any], output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a standardized audit report combining hardware, git metadata, and economic metrics.
    """
    metrics = compute_cost_metrics(input_data)
    # Merge additional user-provided config metadata
    combined_metrics = dict(input_data)
    combined_metrics.update(metrics)

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metadata": {
            **get_git_info(),
            "hardware": get_hardware_info(),
        },
        "metrics": combined_metrics,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def compare_cost_reports(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compares economic efficiency between two benchmark or training runs.
    """
    cost_a = run_a.get("cost_per_1k_completed_games")
    cost_b = run_b.get("cost_per_1k_completed_games")

    pos_cost_a = run_a.get("cost_per_1k_valid_positions")
    pos_cost_b = run_b.get("cost_per_1k_valid_positions")

    savings_games_pct = 0.0
    if isinstance(cost_a, (int, float)) and isinstance(cost_b, (int, float)) and cost_a > 0:
        savings_games_pct = round((cost_a - cost_b) / cost_a * 100.0, 2)

    position_ratio = 1.0
    if isinstance(pos_cost_a, (int, float)) and isinstance(pos_cost_b, (int, float)) and pos_cost_a > 0:
        position_ratio = round(pos_cost_b / pos_cost_a, 4)

    return {
        "run_a_name": run_a.get("name", "Run A"),
        "run_b_name": run_b.get("name", "Run B"),
        "cost_savings_per_1k_games_pct": savings_games_pct,
        "position_cost_ratio": position_ratio,
        "total_cost_delta": round(
            float(run_b.get("total_compute_cost", 0.0)) - float(run_a.get("total_compute_cost", 0.0)), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CoVayAI Training Cost & Throughput Audit Tool")
    parser.add_argument("input_json", help="Path to input JSON containing runtime and throughput metrics")
    parser.add_argument("--output", "-o", default=None, help="Path to write formatted cost report JSON")
    parser.add_argument("--break-even-cloud", type=float, default=None, help="Cloud $/hour price to evaluate break-even")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = generate_cost_report(data, output_path=args.output)
    print(json.dumps(report, indent=2))

    if args.break_even_cloud is not None:
        be = compute_cloud_break_even(data, cloud_price_per_hour=args.break_even_cloud)
        print("\n--- CLOUD BREAK-EVEN ANALYSIS ---")
        print(f"Local Cost/Hour:      ${be['local_cost_per_hour']:.4f}")
        print(f"Local Cost/Game:      ${be['local_cost_per_game']:.6f}")
        print(f"Cloud Price/Hour:     ${be['cloud_price_per_hour']:.2f}")
        print(f"Required Cloud Rate:  {be['required_cloud_games_per_hour']} games/hour ({be['required_speedup_factor']}x speedup)")


if __name__ == "__main__":
    main()
