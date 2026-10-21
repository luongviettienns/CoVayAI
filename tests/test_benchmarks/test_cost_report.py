"""
Tests for Cost and Throughput Audit Module (benchmarks/cost_report.py & benchmarks/cost_schema.json).
Verifies economic metrics calculation, zero division handling, negative value rejection,
schema validation, break-even analysis, and comparative run audits.
"""

import json
import os
import tempfile
import pytest

from benchmarks.cost_report import (
    compute_cost_metrics,
    compute_cloud_break_even,
    generate_cost_report,
    compare_cost_reports,
    validate_cost_input,
    SCHEMA_PATH,
)


class TestCostReportAndSchema:
    def test_cost_schema_exists_and_is_valid_json(self):
        """Verifies cost_schema.json exists and is valid JSON Schema."""
        assert os.path.exists(SCHEMA_PATH), f"Schema not found at {SCHEMA_PATH}"
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        assert "properties" in schema
        assert "required" in schema
        assert "total_running_hours" in schema["properties"]
        assert "completed_games" in schema["properties"]

    def test_compute_cost_metrics_nominal(self):
        """Verifies standard cost metrics calculation with complete data."""
        data = {
            "gpu_hours": 10.0,
            "gpu_price_per_hour": 0.50,  # e.g. RTX 3090 on Cloud
            "electricity_kwh": 3.0,
            "electricity_tariff": 0.15,
            "storage_fees": 1.0,
            "other_fees": 0.5,
            "completed_games": 200,
            "valid_training_positions": 50000,
            "full_search_positions": 12500,
            "nn_evals": 1000000,
        }
        res = compute_cost_metrics(data)

        # Expected:
        # gpu_cost = 10 * 0.50 = 5.0
        # electricity_cost = 3.0 * 0.15 = 0.45
        # total_compute_cost = 5.0 + 0.45 + 1.0 + 0.5 = 6.95
        assert pytest.approx(res["gpu_cost"], 1e-4) == 5.0
        assert pytest.approx(res["electricity_cost"], 1e-4) == 0.45
        assert pytest.approx(res["total_compute_cost"], 1e-4) == 6.95

        # cost_per_1k_completed_games = 6.95 / 200 * 1000 = 34.75
        assert pytest.approx(res["cost_per_1k_completed_games"], 1e-4) == 34.75

        # cost_per_1k_valid_positions = 6.95 / 50000 * 1000 = 0.139
        assert pytest.approx(res["cost_per_1k_valid_positions"], 1e-4) == 0.139

        # cost_per_1k_full_search_positions = 6.95 / 12500 * 1000 = 0.556
        assert pytest.approx(res["cost_per_1k_full_search_positions"], 1e-4) == 0.556

        # cost_per_1m_nn_evals = 6.95 / 1000000 * 1000000 = 6.95
        assert pytest.approx(res["cost_per_1m_nn_evals"], 1e-4) == 6.95

    def test_zero_denominators_return_na_no_zerodivisionerror(self):
        """Verifies that missing or zero throughput metrics return 'N/A' without ZeroDivisionError."""
        data = {
            "gpu_hours": 2.0,
            "gpu_price_per_hour": 1.0,
            "completed_games": 0,
            "valid_training_positions": 0,
            "full_search_positions": 0,
            "nn_evals": 0,
        }
        res = compute_cost_metrics(data)
        assert res["total_compute_cost"] == 2.0
        assert res["cost_per_1k_completed_games"] == "N/A"
        assert res["cost_per_1k_valid_positions"] == "N/A"
        assert res["cost_per_1k_full_search_positions"] == "N/A"
        assert res["cost_per_1m_nn_evals"] == "N/A"

    def test_negative_values_raise_valueerror(self):
        """Verifies negative cost or count values are strictly rejected."""
        with pytest.raises(ValueError, match="negative"):
            compute_cost_metrics({"gpu_hours": -1.0})

        with pytest.raises(ValueError, match="negative"):
            compute_cost_metrics({"gpu_hours": 1.0, "gpu_price_per_hour": -0.5})

        with pytest.raises(ValueError, match="negative"):
            compute_cost_metrics({"gpu_hours": 1.0, "completed_games": -10})

    def test_missing_optional_fields_defaults_and_validation(self):
        """Verifies omitted fields default gracefully without errors."""
        data = {
            "gpu_hours": 5.0,
            "completed_games": 50,
        }
        res = compute_cost_metrics(data)
        assert res["gpu_cost"] == 0.0
        assert res["electricity_cost"] == 0.0
        assert res["total_compute_cost"] == 0.0
        assert res["cost_per_1k_completed_games"] == 0.0
        assert res["cost_per_1k_valid_positions"] == "N/A"

    def test_break_even_analysis_local_vs_cloud(self):
        """Verifies break-even analysis comparing local electricity cost vs cloud renting."""
        # Local system: 0.3 kW, 3000 VND (~$0.12) / kWh, 10 games/hour
        local_data = {
            "power_kw": 0.30,
            "electricity_tariff": 0.12,  # $/kWh
            "games_per_hour": 10.0,
        }
        # Cloud: $0.50/hour
        break_even = compute_cloud_break_even(local_data, cloud_price_per_hour=0.50)

        # Local cost per hour = 0.30 * 0.12 = $0.036/h
        # Local cost per game = 0.036 / 10 = $0.0036/game
        # To match $0.0036/game on a $0.50/h cloud instance, cloud must produce:
        # 0.50 / 0.0036 = 138.89 games/hour (i.e. 13.89x speedup)
        assert pytest.approx(break_even["local_cost_per_hour"], 1e-4) == 0.036
        assert pytest.approx(break_even["local_cost_per_game"], 1e-4) == 0.0036
        assert pytest.approx(break_even["required_cloud_games_per_hour"], 1e-2) == 138.89
        assert pytest.approx(break_even["required_speedup_factor"], 1e-2) == 13.89

    def test_generate_and_save_cost_report(self):
        """Verifies report generation with metadata and schema validation."""
        data = {
            "total_running_hours": 4.0,
            "gpu_hours": 4.0,
            "gpu_price_per_hour": 0.27,
            "completed_games": 80,
            "valid_training_positions": 20000,
            "full_search_positions": 5000,
            "nn_evals": 320000,
            "board_size": 19,
            "network_arch": "resnet_spatial_v2",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "cost_report.json")
            report = generate_cost_report(data, output_path=out_json)

            assert os.path.exists(out_json)
            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)

            assert "metadata" in saved
            assert "git_sha" in saved["metadata"]
            assert "metrics" in saved
            assert saved["metrics"]["board_size"] == 19
            assert saved["metrics"]["total_compute_cost"] == 4.0 * 0.27

    def test_compare_two_runs_cost_efficiency(self):
        """Verifies comparison between two distinct runs."""
        run_a = {
            "name": "Run A (Baseline 19x19)",
            "total_compute_cost": 20.0,
            "completed_games": 100,
            "valid_training_positions": 25000,
            "cost_per_1k_completed_games": 200.0,
            "cost_per_1k_valid_positions": 0.80,
        }
        run_b = {
            "name": "Run B (Cost-Optimized V4)",
            "total_compute_cost": 15.0,
            "completed_games": 150,
            "valid_training_positions": 45000,
            "cost_per_1k_completed_games": 100.0,
            "cost_per_1k_valid_positions": 0.3333,
        }
        comp = compare_cost_reports(run_a, run_b)
        assert comp["cost_savings_per_1k_games_pct"] == 50.0  # (200 - 100) / 200 * 100
        assert comp["position_cost_ratio"] < 1.0
