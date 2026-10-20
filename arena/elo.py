"""Elo rating system and historical tracking for AlphaZero models."""

import json
import math
import os
from typing import Dict, List, Tuple, Optional


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate the expected score of player A against player B using standard Elo formula.

    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    """
    exponent = (rating_b - rating_a) / 400.0
    return 1.0 / (1.0 + 10.0 ** exponent)


def update_ratings(
    rating_a: float,
    rating_b: float,
    score_a: float,
    score_b: float,
    k_factor: float = 32.0,
) -> Tuple[float, float]:
    """Calculate updated Elo ratings for player A and player B.

    score_a, score_b should be in [0.0, 1.0] representing normalized scores (e.g. 1.0 for win, 0.5 for draw, 0.0 for loss).
    Guarantees conservation of rating points: (new_a + new_b) == (rating_a + rating_b) when scores sum to 1.
    """
    e_a = expected_score(rating_a, rating_b)
    e_b = 1.0 - e_a

    new_rating_a = rating_a + k_factor * (score_a - e_a)
    new_rating_b = rating_b + k_factor * (score_b - e_b)

    return new_rating_a, new_rating_b


def estimate_rating_diff(win_rate: float, eps: float = 1e-4) -> float:
    """Estimate the Elo rating difference (R_A - R_B) based on head-to-head win rate.

    Delta_R = 400 * log10(win_rate / (1 - win_rate))
    Clamped safely to avoid div by zero or log(0).
    """
    clamped_wr = min(max(win_rate, eps), 1.0 - eps)
    return 400.0 * math.log10(clamped_wr / (1.0 - clamped_wr))


class EloTracker:
    """Tracks model Elo ratings across tournament and evaluation matches."""

    def __init__(self, initial_rating: float = 1000.0, k_factor: float = 32.0):
        self.initial_rating = float(initial_rating)
        self.k_factor = float(k_factor)
        self.ratings: Dict[str, float] = {}
        self.history: List[dict] = []

    def get_rating(self, model_id: str) -> float:
        """Get the current rating of a model, defaulting to initial_rating if unrecorded."""
        return self.ratings.get(model_id, self.initial_rating)

    def record_match(
        self,
        model_a_id: str,
        model_b_id: str,
        wins_a: int,
        wins_b: int,
        draws: int = 0,
        k_factor: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Record the results of a match between model A and model B and update their ratings.

        Args:
            model_a_id: Unique identifier for model A.
            model_b_id: Unique identifier for model B.
            wins_a: Number of wins for model A.
            wins_b: Number of wins for model B.
            draws: Number of draws.
            k_factor: Optional custom K-factor for this match.

        Returns:
            Tuple of (new_rating_a, new_rating_b).
        """
        total = wins_a + wins_b + draws
        if total == 0:
            return self.get_rating(model_a_id), self.get_rating(model_b_id)

        k = k_factor if k_factor is not None else self.k_factor
        score_a = (wins_a + 0.5 * draws) / total
        score_b = (wins_b + 0.5 * draws) / total

        curr_a = self.get_rating(model_a_id)
        curr_b = self.get_rating(model_b_id)

        new_a, new_b = update_ratings(curr_a, curr_b, score_a, score_b, k_factor=k)

        self.ratings[model_a_id] = new_a
        self.ratings[model_b_id] = new_b

        match_record = {
            "model_a": model_a_id,
            "model_b": model_b_id,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "rating_a_before": curr_a,
            "rating_b_before": curr_b,
            "rating_a_after": new_a,
            "rating_b_after": new_b,
        }
        self.history.append(match_record)

        return new_a, new_b

    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Return leaderboard as list of (model_id, rating) sorted from highest to lowest."""
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)

    def save(self, file_path: str) -> None:
        """Save the tracker state to a JSON file."""
        data = {
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "ratings": self.ratings,
            "history": self.history,
        }
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, file_path)

    @classmethod
    def load(cls, file_path: str) -> "EloTracker":
        """Load tracker state from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tracker = cls(
            initial_rating=data.get("initial_rating", 1000.0),
            k_factor=data.get("k_factor", 32.0),
        )
        tracker.ratings = data.get("ratings", {})
        tracker.history = data.get("history", [])
        return tracker
