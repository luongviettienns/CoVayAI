"""Unit tests for Elo rating calculation and tracking."""

import os
import pytest
import tempfile
from arena.elo import (
    expected_score,
    update_ratings,
    estimate_rating_diff,
    EloTracker,
)


def test_expected_score():
    """Verify standard Elo expected score formulas and properties."""
    # Equal ratings -> 50% expectation
    assert pytest.approx(expected_score(1000.0, 1000.0), abs=1e-5) == 0.5

    # Symmetry: E(A, B) + E(B, A) == 1.0
    ea = expected_score(1200.0, 1000.0)
    eb = expected_score(1000.0, 1200.0)
    assert pytest.approx(ea + eb, abs=1e-5) == 1.0

    # 400 point difference: 1 / (1 + 10^(-1)) = 1 / 1.1 = 10/11 ~ 0.90909
    assert pytest.approx(expected_score(1400.0, 1000.0), abs=1e-3) == 0.909


def test_update_ratings():
    """Verify rating adjustment and conservation."""
    r_a, r_b = 1000.0, 1000.0
    k = 32.0

    # A wins (score_a = 1.0, score_b = 0.0)
    new_a, new_b = update_ratings(r_a, r_b, score_a=1.0, score_b=0.0, k_factor=k)
    assert new_a > r_a
    assert new_b < r_b
    # Conservation of total rating points
    assert pytest.approx(new_a + new_b, abs=1e-5) == r_a + r_b
    assert pytest.approx(new_a - r_a, abs=1e-5) == k * 0.5  # E_A was 0.5, score was 1.0 -> +16


def test_estimate_rating_diff():
    """Verify rating difference estimation from win rate."""
    assert pytest.approx(estimate_rating_diff(0.5), abs=1e-5) == 0.0
    assert estimate_rating_diff(0.75) > 0.0
    assert estimate_rating_diff(0.25) < 0.0
    assert pytest.approx(estimate_rating_diff(0.75) + estimate_rating_diff(0.25), abs=1e-5) == 0.0

    # Edge cases: 0.0 and 1.0 clipped safely without inf/nan
    diff_0 = estimate_rating_diff(0.0)
    diff_1 = estimate_rating_diff(1.0)
    assert not (pytest.approx(diff_0) == float("nan"))
    assert not (pytest.approx(diff_1) == float("nan"))
    assert diff_1 > 500.0
    assert diff_0 < -500.0


def test_elo_tracker():
    """Verify EloTracker state, match recording, and leaderboard."""
    tracker = EloTracker(initial_rating=1000.0, k_factor=32.0)
    assert tracker.get_rating("model_0") == 1000.0
    assert tracker.get_rating("model_1") == 1000.0

    # Model 1 beats Model 0 (3 wins vs 1 win, 0 draws)
    tracker.record_match("model_1", "model_0", wins_a=3, wins_b=1, draws=0)
    r1 = tracker.get_rating("model_1")
    r0 = tracker.get_rating("model_0")

    assert r1 > 1000.0
    assert r0 < 1000.0
    assert pytest.approx(r1 + r0, abs=1e-4) == 2000.0

    leaderboard = tracker.get_leaderboard()
    assert leaderboard[0][0] == "model_1"
    assert leaderboard[1][0] == "model_0"


def test_elo_tracker_persistence():
    """Verify saving and loading Elo ratings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "ratings.json")
        tracker = EloTracker(initial_rating=1200.0)
        tracker.record_match("v1", "v0", wins_a=5, wins_b=0)
        tracker.save(filepath)

        loaded = EloTracker.load(filepath)
        assert pytest.approx(loaded.get_rating("v1"), abs=1e-4) == tracker.get_rating("v1")
        assert pytest.approx(loaded.get_rating("v0"), abs=1e-4) == tracker.get_rating("v0")
        assert loaded.initial_rating == 1200.0
