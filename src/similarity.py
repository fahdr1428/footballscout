"""
Player similarity.

Two distance metrics over the *standardised, position-specific* feature space:

* **cosine** (default) - the cosine of the two z-score vectors. Because the
  features are centred on the positional average, this measures whether two
  players deviate from their peers in the *same direction*: the same style,
  whether or not at the same intensity. That is what a scout usually wants,
  and it is why a lower-level player can still read as a close match. The
  percentage is the cosine itself, floored at zero.

* **euclidean** - the root-mean-square difference in z-scores across the
  position's features, which does care about intensity. It is converted to a
  percentage against a published reference: the median RMS distance between two
  randomly chosen players in the same position pool. So

      similarity % = 100 * (1 - rms_distance / typical_pair_distance)

  100% means an identical statistical profile; 0% means "as different as two
  randomly picked players in this position". Nothing here is a fudge factor -
  the reference distance is printed in the app.

Every result also carries a feature-level explanation: which metrics match,
which diverge, and how much each metric contributed to the distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import METRIC_LABELS

# Features must differ by at least this many z-units to be called a difference.
DIFFERENCE_Z_THRESHOLD = 0.6
# A "match" is more interesting when both players are away from average.
DISTINCTIVENESS_Z = 0.4


@dataclass
class SimilarityResult:
    """One similar-player row plus the evidence behind it."""

    index: int
    similarity: float
    distance: float
    matches: list[dict]
    differences: list[dict]
    contributions: pd.DataFrame


class SimilarityEngine:
    """Nearest-neighbour search inside one position group."""

    def __init__(
        self,
        z: pd.DataFrame,
        pool: pd.DataFrame,
        weights: dict[str, float] | None = None,
        random_state: int = 0,
    ) -> None:
        self.features = list(z.columns)
        self.pool = pool
        self.z = z
        self.weights = self._normalise_weights(weights)
        # Scaling columns by sqrt(w_i) makes a plain Euclidean distance equal
        # to the weighted RMS z-difference.
        self._zw = z.to_numpy() * np.sqrt(self.weights)
        self._euclidean = NearestNeighbors(metric="euclidean").fit(self._zw)
        self._cosine = NearestNeighbors(metric="cosine").fit(self._zw)
        self.reference_distance = self._reference_distance(random_state)

    # -- setup ---------------------------------------------------------
    def _normalise_weights(self, weights: dict[str, float] | None) -> np.ndarray:
        if not weights:
            raw = np.ones(len(self.features))
        else:
            raw = np.array([max(float(weights.get(f, 0.0)), 0.0) for f in self.features])
            if raw.sum() <= 0:
                raw = np.ones(len(self.features))
        return raw / raw.sum()

    def _reference_distance(self, random_state: int, sample: int = 4000) -> float:
        """Median RMS distance between two random players in this pool."""
        n = len(self.z)
        if n < 3:
            return 1.0
        rng = np.random.default_rng(random_state)
        size = min(sample, n * 4)
        a = rng.integers(0, n, size)
        b = rng.integers(0, n, size)
        keep = a != b
        distances = np.linalg.norm(self._zw[a[keep]] - self._zw[b[keep]], axis=1)
        return float(np.median(distances)) or 1.0

    # -- queries -------------------------------------------------------
    def _row(self, index) -> np.ndarray:
        return self._zw[self.z.index.get_loc(index)]

    def distance_to_all(self, index, metric: str = "cosine") -> np.ndarray:
        row = self._row(index)
        if metric == "cosine":
            norms = np.linalg.norm(self._zw, axis=1) * np.linalg.norm(row)
            with np.errstate(invalid="ignore", divide="ignore"):
                cosine = np.where(norms > 0, self._zw @ row / np.where(norms > 0, norms, 1), 0.0)
            return 1 - cosine
        return np.linalg.norm(self._zw - row, axis=1)

    def similarity_percent(self, distances: np.ndarray, metric: str = "cosine") -> np.ndarray:
        if metric == "cosine":
            return np.clip((1 - distances) * 100, 0, 100)
        return np.clip((1 - distances / self.reference_distance) * 100, 0, 100)

    def neighbours(
        self,
        index,
        n: int = 10,
        metric: str = "cosine",
        candidate_mask: pd.Series | None = None,
        exclude_same_player: bool = True,
    ) -> pd.DataFrame:
        """Ranked similar players, with similarity % and the pool metadata."""
        distances = self.distance_to_all(index, metric=metric)
        result = pd.DataFrame(
            {
                "distance": distances,
                "similarity": self.similarity_percent(distances, metric=metric),
            },
            index=self.z.index,
        )
        result = result.join(self.pool)
        result = result.drop(index=index, errors="ignore")
        if exclude_same_player and "player_id" in result.columns:
            player_id = self.pool.loc[index, "player_id"]
            result = result[result["player_id"] != player_id]
        if candidate_mask is not None:
            result = result[candidate_mask.reindex(result.index).fillna(False)]
        result = result.sort_values("distance").head(n)
        result["similarity"] = result["similarity"].round(1)
        result["rank"] = range(1, len(result) + 1)
        return result

    # -- explanation ---------------------------------------------------
    def explain(self, index_a, index_b, metric: str = "cosine") -> SimilarityResult:
        """Feature-level account of why two players are (or are not) alike."""
        za = self.z.loc[index_a]
        zb = self.z.loc[index_b]
        delta = (za - zb).abs()
        squared = (za - zb) ** 2 * self.weights
        total = squared.sum()

        table = pd.DataFrame(
            {
                "feature": self.features,
                "metric": [METRIC_LABELS.get(f, f) for f in self.features],
                "value_a": [self.pool.loc[index_a, f] for f in self.features],
                "value_b": [self.pool.loc[index_b, f] for f in self.features],
                "z_a": za.to_numpy(),
                "z_b": zb.to_numpy(),
                "abs_z_gap": delta.to_numpy(),
                "distance_share": (squared / total if total > 0 else squared).to_numpy(),
            }
        ).sort_values("abs_z_gap")

        # Matches: close together, and ideally both away from positional average.
        distinctive = table[
            (table["abs_z_gap"] < DIFFERENCE_Z_THRESHOLD)
            & (table[["z_a", "z_b"]].abs().min(axis=1) >= DISTINCTIVENESS_Z)
        ]
        if len(distinctive) < 3:
            distinctive = table[table["abs_z_gap"] < DIFFERENCE_Z_THRESHOLD]
        matches = distinctive.head(5).to_dict("records")

        differences = (
            table[table["abs_z_gap"] >= DIFFERENCE_Z_THRESHOLD]
            .sort_values("abs_z_gap", ascending=False)
            .head(5)
            .to_dict("records")
        )

        distance = float(np.sqrt(total))
        pair_distance = float(self.distance_to_all(index_a, metric=metric)[self.z.index.get_loc(index_b)])
        return SimilarityResult(
            index=index_b,
            similarity=float(self.similarity_percent(np.array([pair_distance]), metric=metric)[0]),
            distance=distance,
            matches=matches,
            differences=differences,
            contributions=table.sort_values("distance_share", ascending=False),
        )


def _phrase(label: str) -> str:
    """Lower-case a metric label for mid-sentence use, but keep acronyms (xG, xA)."""
    if len(label) > 1 and label[1].isupper():
        return label
    return label[0].lower() + label[1:]


def explanation_sentences(
    result: SimilarityResult, name_a: str, name_b: str, pool: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Plain-English bullets for the similarity explanation."""
    similar = [
        f"Similar {_phrase(row['metric'])} "
        f"({_fmt(row['value_a'])} vs {_fmt(row['value_b'])})"
        for row in result.matches
    ]
    different = []
    for row in result.differences:
        leader, trailer = (name_a, name_b) if row["z_a"] > row["z_b"] else (name_b, name_a)
        high, low = (
            (row["value_a"], row["value_b"]) if row["z_a"] > row["z_b"] else (row["value_b"], row["value_a"])
        )
        different.append(
            f"{leader} records more {_phrase(row['metric'])} than {trailer} "
            f"({_fmt(high)} vs {_fmt(low)})"
        )
    return similar, different


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def rank_correlation(rank_a: list, rank_b: list) -> float:
    """Spearman correlation between two ranked lists of the same players.

    Used by the sensitivity tests: if dropping a metric reshuffles the top ten,
    the similarity model was leaning on that one metric too heavily.
    """
    common = [item for item in rank_a if item in set(rank_b)]
    if len(common) < 3:
        return float("nan")
    pos_a = {item: i for i, item in enumerate(rank_a)}
    pos_b = {item: i for i, item in enumerate(rank_b)}
    a = pd.Series([pos_a[item] for item in common])
    b = pd.Series([pos_b[item] for item in common])
    return float(a.corr(b, method="spearman"))


def jaccard(a: list, b: list) -> float:
    """Overlap between two shortlists (used for sensitivity testing)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
