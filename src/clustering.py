"""
Statistical archetypes via K-Means, with automatically generated labels.

The number of clusters is chosen from data (silhouette score, cross-checked
against the inertia elbow), and each cluster's *name* is derived from its own
centroid: the concepts the cluster is strongest in become the adjectives, the
position supplies the noun. Nothing is hand-assigned, so re-running on a
different dataset produces different - but still meaningful - archetype names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import LOWER_IS_BETTER, METRIC_LABELS, POSITION_GROUP_NAMES, categories_for

# Extra naming concepts on top of the radar categories.
EXTRA_CONCEPTS = {
    "Pressing": ["pressures_per90", "pressure_success_pct", "ball_recoveries_per90"],
    "Passing Volume": ["passes_attempted_per90", "touches_per90"],
    "Long Passing": ["long_passes_attempted_per90", "long_pass_pct", "switches_per90"],
    "Crossing": ["crosses_per90"],
}

CONCEPT_ADJECTIVES = {
    "Finishing": "goalscoring",
    "Box Threat": "penalty-box",
    "Chance Creation": "creative",
    "Passing": "ball-playing",
    "Ball Progression": "progressive",
    "Dribbling": "dribbling",
    "Defending": "ball-winning",
    "Aerial": "aerially dominant",
    "Pressing": "high-pressing",
    "Passing Volume": "high-volume",
    "Long Passing": "long-passing",
    "Crossing": "crossing",
    "Shot Stopping": "shot-stopping",
    "Goal Prevention": "goal-preventing",
    "Claiming Crosses": "commanding",
    "Sweeping": "sweeper",
    "Distribution": "ball-playing",
    "Long Distribution": "direct",
}

# Used when a cluster has no strength worth naming: scouts still describe those
# players by what they *do not* do, and that is more useful than "cluster 3".
DEFICIT_ADJECTIVES = {
    "Finishing": "low-shooting",
    "Box Threat": "deep-lying",
    "Chance Creation": "low-creativity",
    "Passing": "low-involvement",
    "Ball Progression": "conservative",
    "Dribbling": "non-dribbling",
    "Defending": "low-intervention",
    "Aerial": "ground-based",
    "Pressing": "passive-pressing",
    "Passing Volume": "low-volume",
    "Long Passing": "short-passing",
    "Crossing": "narrow",
    "Shot Stopping": "error-prone",
    "Goal Prevention": "heavily-worked",
    "Claiming Crosses": "line-bound",
    "Sweeping": "box-bound",
    "Distribution": "low-involvement",
    "Long Distribution": "short-distributing",
}

POSITION_NOUNS = {
    "GK": "goalkeeper",
    "CB": "centre-back",
    "FB": "full-back",
    "DM": "holding midfielder",
    "CM": "midfielder",
    "AM": "attacking midfielder",
    "W": "winger",
    "FW": "forward",
}

MIN_CLUSTER_FLOOR = 20        # never accept an archetype with fewer players than this
MIN_CLUSTER_SHARE = 0.04      # ...or fewer than this share of the position group

STRONG_CONCEPT_Z = 0.30       # concept must be this far above average to name
SECOND_CONCEPT_Z = 0.25       # second adjective threshold
NOTABLE_FEATURE_Z = 0.45      # feature included in the written description


@dataclass
class ClusterModel:
    """Everything the app needs to talk about one position's archetypes."""

    position_group: str
    k: int
    kmeans: KMeans
    labels: pd.Series
    centroids: pd.DataFrame           # cluster x feature, in z units
    names: dict[int, str]
    descriptions: dict[int, str]
    evaluation: pd.DataFrame          # k, inertia, silhouette
    silhouette: float
    inertia: float
    elbow_k: int
    concept_scores: pd.DataFrame = field(default_factory=pd.DataFrame)


def evaluate_k(
    z: pd.DataFrame, k_min: int = 3, k_max: int = 9, random_state: int = 42
) -> pd.DataFrame:
    """Inertia and silhouette score for each candidate number of clusters."""
    rows = []
    k_max = int(min(k_max, max(k_min, len(z) // 25)))
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(z)
        sizes = pd.Series(labels).value_counts()
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(z, labels)) if len(set(labels)) > 1 else np.nan,
                "smallest_cluster": int(sizes.min()),
            }
        )
    return pd.DataFrame(rows)


def elbow_k(evaluation: pd.DataFrame) -> int:
    """Elbow point: the k furthest from the line joining the first and last k.

    This is the standard geometric reading of an elbow plot, done numerically
    so the choice is reproducible rather than eyeballed.
    """
    if len(evaluation) < 3:
        return int(evaluation["k"].iloc[0])
    x = evaluation["k"].to_numpy(dtype=float)
    y = evaluation["inertia"].to_numpy(dtype=float)
    x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-9)
    start, end = np.array([x_norm[0], y_norm[0]]), np.array([x_norm[-1], y_norm[-1]])
    line = end - start
    line = line / np.linalg.norm(line)
    points = np.column_stack([x_norm, y_norm]) - start
    projection = np.outer(points @ line, line)
    distances = np.linalg.norm(points - projection, axis=1)
    return int(x[int(np.argmax(distances))])


def choose_k(
    evaluation: pd.DataFrame, tolerance: float = 0.90, min_cluster_size: int = 0
) -> int:
    """Pick k from the silhouette curve, preferring granularity where it is free.

    Silhouette almost always favours k=2 on football data, because playing
    styles form a continuum rather than well-separated blobs. Taking the raw
    argmax therefore returns "two kinds of centre-back", which is true but
    useless to a scout. Instead we take the **largest k whose silhouette is
    still within `tolerance` of the best score** - the most granular set of
    archetypes that costs essentially nothing in cluster quality.

    A second constraint keeps the result usable: any k that produces a cluster
    smaller than `min_cluster_size` is rejected. An archetype supported by nine
    players is a curiosity, not a role, and its centroid - and therefore its
    generated name - is dominated by noise. The inertia elbow is reported
    alongside as a cross-check.
    """
    valid = evaluation.dropna(subset=["silhouette"])
    if valid.empty:
        return int(evaluation["k"].iloc[0])
    if min_cluster_size and "smallest_cluster" in valid.columns:
        supported = valid[valid["smallest_cluster"] >= min_cluster_size]
        if not supported.empty:
            valid = supported
    best = valid["silhouette"].max()
    if best <= 0:
        return int(valid.loc[valid["silhouette"].idxmax(), "k"])
    acceptable = valid[valid["silhouette"] >= tolerance * best]
    return int(acceptable["k"].max())


def concept_map(position_group: str, features: list[str]) -> dict[str, list[str]]:
    """Concepts (with their features) available for this position's model."""
    concepts: dict[str, list[str]] = {}
    available = set(features)
    for name, metrics in {**categories_for(position_group), **EXTRA_CONCEPTS}.items():
        present = [m for m in metrics if m in available]
        if present:
            concepts[name] = present
    return concepts


def _concept_scores(centroids: pd.DataFrame, concepts: dict[str, list[str]]) -> pd.DataFrame:
    """Mean centroid z-score per concept: cluster x concept.

    Metrics where less is better (goals conceded, miscontrols, errors) have
    their sign flipped first, so a positive concept score always means "good at
    this", never "does a lot of this".
    """
    oriented = centroids.copy()
    for metric in oriented.columns:
        if metric in LOWER_IS_BETTER:
            oriented[metric] = -oriented[metric]
    data = {name: oriented[metrics].mean(axis=1) for name, metrics in concepts.items()}
    return pd.DataFrame(data, index=centroids.index).round(3)


def name_clusters(
    position_group: str, concept_scores: pd.DataFrame
) -> dict[int, str]:
    """Compose an archetype name for each cluster from its own centroid."""
    noun = POSITION_NOUNS.get(position_group, POSITION_GROUP_NAMES.get(position_group, "player"))
    names: dict[int, str] = {}
    used: set[str] = set()

    for cluster in concept_scores.index:
        ranked = concept_scores.loc[cluster].sort_values(ascending=False)
        parts: list[str] = []
        for concept, value in ranked.items():
            threshold = STRONG_CONCEPT_Z if not parts else SECOND_CONCEPT_Z
            if value < threshold or len(parts) == 2:
                break
            adjective = CONCEPT_ADJECTIVES.get(concept)
            if adjective and adjective not in parts:
                parts.append(adjective)

        if not parts:
            weakest, weakest_z = ranked.index[-1], ranked.iloc[-1]
            if weakest_z <= -STRONG_CONCEPT_Z:
                name = f"{DEFICIT_ADJECTIVES.get(weakest, 'low-output')} {noun}".capitalize()
            else:
                name = f"All-round {noun}"
        else:
            name = f"{' '.join(parts)} {noun}".capitalize()

        # Guarantee uniqueness within the position group.
        if name in used:
            for concept, _value in ranked.items():
                adjective = CONCEPT_ADJECTIVES.get(concept)
                if not adjective or adjective in name.lower():
                    continue
                candidate = f"{adjective} {name[0].lower() + name[1:]}".capitalize()
                if candidate not in used:
                    name = candidate
                    break
            else:
                name = f"{name} (variant {cluster + 1})"
        used.add(name)
        names[int(cluster)] = name
    return names


def describe_clusters(centroids: pd.DataFrame, sizes: pd.Series) -> dict[int, str]:
    """One sentence per cluster, listing what actually defines it."""
    descriptions: dict[int, str] = {}
    for cluster in centroids.index:
        row = centroids.loc[cluster].sort_values(ascending=False)
        highs = [(m, v) for m, v in row.items() if v >= NOTABLE_FEATURE_Z][:4]
        lows = [(m, v) for m, v in row.items() if v <= -NOTABLE_FEATURE_Z][-3:]
        pieces = [f"high {METRIC_LABELS.get(m, m).lower()} ({v:+.2f} SD)" for m, v in highs]
        pieces += [f"low {METRIC_LABELS.get(m, m).lower()} ({v:+.2f} SD)" for m, v in reversed(lows)]
        if not pieces:
            pieces = ["no metric more than 0.45 SD from the positional average"]
        descriptions[int(cluster)] = (
            f"{int(sizes.get(cluster, 0))} players. Characterised by " + ", ".join(pieces) + "."
        )
    return descriptions


def fit_clusters(
    z: pd.DataFrame,
    position_group: str,
    k: int | None = None,
    k_min: int = 3,
    k_max: int = 9,
    random_state: int = 42,
) -> ClusterModel:
    """Fit K-Means for one position group and label the resulting archetypes."""
    evaluation = evaluate_k(z, k_min=k_min, k_max=k_max, random_state=random_state)
    # An archetype needs enough players behind it for its centroid - and so its
    # generated name - to mean anything.
    floor = max(MIN_CLUSTER_FLOOR, int(MIN_CLUSTER_SHARE * len(z)))
    chosen = int(k or choose_k(evaluation, min_cluster_size=floor))
    chosen = max(2, min(chosen, len(z) - 1))

    kmeans = KMeans(n_clusters=chosen, n_init=10, random_state=random_state)
    labels = pd.Series(kmeans.fit_predict(z), index=z.index, name="cluster")
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=z.columns).round(3)
    centroids.index.name = "cluster"

    concepts = concept_map(position_group, list(z.columns))
    scores = _concept_scores(centroids, concepts)
    sizes = labels.value_counts()

    silhouette = float(silhouette_score(z, labels)) if labels.nunique() > 1 else float("nan")
    return ClusterModel(
        position_group=position_group,
        k=chosen,
        kmeans=kmeans,
        labels=labels,
        centroids=centroids,
        names=name_clusters(position_group, scores),
        descriptions=describe_clusters(centroids, sizes),
        evaluation=evaluation,
        silhouette=silhouette,
        inertia=float(kmeans.inertia_),
        elbow_k=elbow_k(evaluation),
        concept_scores=scores,
    )


def pca_projection(
    z: pd.DataFrame, n_components: int = 2, random_state: int = 42
) -> tuple[pd.DataFrame, PCA]:
    """2-D projection used for the archetype map."""
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(z)
    columns = [f"pc{i + 1}" for i in range(n_components)]
    return pd.DataFrame(coords, index=z.index, columns=columns).round(3), pca


def component_loadings(pca: PCA, features: list[str], top: int = 6) -> dict[str, list[tuple[str, float]]]:
    """Which metrics drive each principal component (so the axes mean something)."""
    out: dict[str, list[tuple[str, float]]] = {}
    for i, component in enumerate(pca.components_):
        order = np.argsort(-np.abs(component))[:top]
        out[f"pc{i + 1}"] = [
            (METRIC_LABELS.get(features[j], features[j]), round(float(component[j]), 3)) for j in order
        ]
    return out
