"""
Scouting report generator.

Produces a written report from the fitted models - percentiles, archetype,
nearest neighbours and fit score - with no canned prose. Every sentence is
assembled from a number that appears elsewhere in the app, and every caveat
(sample size, league level, one-season finishing noise) is raised by a rule
that inspects the player's own data.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .config import LEAGUE_STRENGTH, LEAGUE_TIER, METRIC_LABELS, POSITION_GROUP_NAMES
from .pipeline import ScoutingPlatform, format_metric
from .recruitment import RecruitmentBrief, fit_scores
from .similarity import explanation_sentences

HEADLINE_METRICS = [
    "np_goals_per90", "npxg_per90", "assists_per90", "xa_per90", "shots_per90",
    "key_passes_per90", "sca_per90", "progressive_passes_per90",
    "progressive_carries_per90", "dribbles_completed_per90", "pass_pct",
    "pass_pct_under_pressure", "defensive_actions_per90", "aerial_win_pct",
]

GK_HEADLINE_METRICS = [
    "gk_save_pct", "gk_psxg_minus_ga_per90", "gk_saves_per90", "gk_goals_against_per90",
    "gk_cross_stop_pct", "gk_def_actions_outside_box_per90", "pass_pct",
    "passes_attempted_per90", "gk_launch_pct",
]

RELIABLE_MINUTES = 1500
YOUNG_AGE = 23.0
VETERAN_AGE = 31.0


def headline_metrics(position_group: str) -> list[str]:
    return GK_HEADLINE_METRICS if position_group == "GK" else HEADLINE_METRICS


def ordinal(value: float) -> str:
    """1st, 2nd, 3rd, 11th ... for percentile prose."""
    n = int(round(value))
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([" --- "] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def key_takeaways(platform: ScoutingPlatform, index) -> list[str]:
    """Rule-based conclusions, each triggered by a number in the player's row."""
    row = platform.row(index)
    categories = platform.category_scores(index)
    takeaways: list[str] = []

    if categories:
        best = max(categories, key=categories.get)
        worst = min(categories, key=categories.get)
        takeaways.append(
            f"Profile is led by **{best.lower()}** ({ordinal(categories[best])} percentile) and "
            f"limited by **{worst.lower()}** ({ordinal(categories[worst])} percentile) among "
            f"{POSITION_GROUP_NAMES[row['position_group']].lower()}s in the filtered pool."
        )

    if platform.has_age and row["age"] <= YOUNG_AGE:
        strong = [c for c, v in categories.items() if v >= 75]
        if strong:
            takeaways.append(
                f"At {row['age']:.1f} he already sits in the top quartile for "
                f"{', '.join(c.lower() for c in strong)} - a development-profile target rather "
                "than a finished product."
            )
        else:
            takeaways.append(
                f"At {row['age']:.1f} there is room for development, but no category currently "
                "reaches the top quartile of his positional peers."
            )
    elif platform.has_age and row["age"] >= VETERAN_AGE:
        takeaways.append(
            f"At {row['age']:.1f} this is a short-horizon signing; resale value is unlikely to be "
            "part of the case."
        )

    if row["minutes"] < RELIABLE_MINUTES:
        takeaways.append(
            f"Sample caution: {row['minutes']:,.0f} minutes. Rate statistics from under "
            f"{RELIABLE_MINUTES:,} minutes carry meaningful noise, especially finishing metrics."
        )

    tier = int(row.get("league_tier", 1))
    if tier > 1:
        takeaways.append(
            f"Plays in {row['league']} (level {tier} in this app's league table, coefficient "
            f"{float(row.get('league_strength', 0.8)):.2f}). Output should be discounted against "
            "stronger leagues; that coefficient is an assumption, not a measurement."
        )

    if row["position_group"] != "GK":
        finishing = row.get("np_goals_minus_npxg_per90", np.nan)
        if pd.notna(finishing) and abs(finishing) >= 0.10:
            direction = "over" if finishing > 0 else "under"
            takeaways.append(
                f"Finishing {direction}-performance of {finishing:+.2f} non-penalty goals vs xG per 90. "
                "One season of finishing variance is not a repeatable skill signal - treat the xG "
                "figure as the better estimate of chance quality."
            )
    return takeaways


def generate_report(
    platform: ScoutingPlatform,
    index,
    n_similar: int = 5,
    brief: RecruitmentBrief | None = None,
    similarity_metric: str = "cosine",
) -> str:
    """Full markdown scouting report for one player-season."""
    row = platform.row(index)
    group = row["position_group"]
    archetype, archetype_description = platform.archetype(index)
    categories = platform.category_scores(index)

    parts: list[str] = []
    parts.append(f"# Scouting report - {row['player']}")
    parts.append(
        f"*Generated {date.today().isoformat()} from {platform.spec.label}. "
        f"Pool: {len(platform.pool):,} player-seasons, minimum {platform.min_minutes:,} minutes. "
        f"Percentiles are measured against {platform.peer_group_label} "
        f"({len(platform.peers(index)):,} of them).*"
    )

    parts.append("\n## Player")
    parts.append(
        _table(
            [
                ["Name", str(row["player"])],
                *([["Age", f"{row['age']:.1f}"]] if platform.has_age else []),
                *([["Nationality", str(row["nationality"])]]
                  if "nationality" in row.index and pd.notna(row.get("nationality")) else []),
                ["Position", f"{row['position']} ({POSITION_GROUP_NAMES[group]})"],
                ["Club", str(row["team"])],
                ["League", f"{row['league']} (level {int(row.get('league_tier', 1))})"],
                ["Season", str(row["season"])],
                ["Minutes", f"{row['minutes']:,.0f} across {row['matches']:,.0f} appearances"],
                *([["Height", f"{row['height_cm']:.0f} cm"]] if platform.has("height_cm") else []),
            ],
            ["Field", "Value"],
        )
    )

    parts.append("\n## Statistical profile")
    metrics = [m for m in headline_metrics(group) if m in platform.pool.columns]
    frame = platform.percentile_frame(index, metrics)
    parts.append(
        _table(
            [
                [r["metric"], format_metric(r["key"], r["value"]), f"{r['percentile']:.0f}"]
                for _, r in frame.iterrows()
            ],
            ["Metric", "Per 90 / rate", "Positional percentile"],
        )
    )

    if categories:
        parts.append("\n### Attribute categories")
        parts.append(
            _table(
                [[c, f"{v:.0f}"] for c, v in sorted(categories.items(), key=lambda kv: -kv[1])],
                ["Category", "Percentile (mean of its metrics)"],
            )
        )

    parts.append("\n## Strengths")
    strengths = platform.strengths(index, n=5)
    if strengths.empty:
        parts.append("No metric reaches the 70th percentile against positional peers.")
    else:
        parts += [
            f"- **{r['metric']}** - {format_metric(r['key'], r['value'])} "
            f"({ordinal(r['percentile'])} percentile)"
            for _, r in strengths.iterrows()
        ]

    parts.append("\n## Weaknesses")
    weaknesses = platform.weaknesses(index, n=5)
    if weaknesses.empty:
        parts.append("No metric falls below the 30th percentile against positional peers.")
    else:
        parts += [
            f"- **{r['metric']}** - {format_metric(r['key'], r['value'])} "
            f"({ordinal(r['percentile'])} percentile)"
            for _, r in weaknesses.iterrows()
        ]

    parts.append("\n## Archetype")
    parts.append(f"**{archetype}** - {archetype_description}")
    parts.append(
        "Archetypes come from K-Means run on this position group only; the label is generated "
        "from the cluster centroid, not assigned by hand."
    )

    parts.append("\n## Similar players")
    similar = platform.similar(index, n=n_similar, metric=similarity_metric)
    if similar.empty:
        parts.append("No comparable players in the current pool.")
    else:
        parts.append(
            _table(
                [
                    [
                        f"{r['player']}",
                        f"{r['similarity']:.0f}%",
                        f"{r['age']:.1f}",
                        str(r["team"]),
                        str(r["league"]),
                        f"{r['minutes']:,.0f}",
                    ]
                    for _, r in similar.iterrows()
                ],
                ["Player", "Similarity", "Age", "Club", "League", "Minutes"],
            )
        )
        top = similar.index[0]
        result = platform.explain_similarity(index, top, metric=similarity_metric)
        matches, differences = explanation_sentences(
            result, row["player"], platform.pool.loc[top, "player"], platform.pool
        )
        parts.append(f"\n**Why {platform.pool.loc[top, 'player']} is the closest match**")
        parts += [f"- {m}" for m in matches]
        if differences:
            parts.append("\nBut:")
            parts += [f"- {d}" for d in differences]

    if brief is not None:
        parts.append("\n## Recruitment fit")
        weights = brief.resolved_weights()
        scores = fit_scores(platform.categories.loc[[index]], weights)
        fit = float(scores["fit_score"].iloc[0])
        parts.append(f"**Fit score {fit:.0f} / 100** against the brief:")
        parts.append(
            _table(
                [
                    [category, f"{weights[category]:.0f}%", f"{categories.get(category, float('nan')):.0f}",
                     f"{scores[f'contrib_{category}'].iloc[0]:.1f}"]
                    for category in weights
                    if f"contrib_{category}" in scores.columns
                ],
                ["Category", "Weight", "Percentile", "Points contributed"],
            )
        )

    parts.append("\n## Key takeaways")
    parts += [f"- {t}" for t in key_takeaways(platform, index)]

    parts.append("\n---")
    parts.append(f"*{platform.spec.attribution}*")
    parts.append(
        "*Descriptive statistics (per-90 rates, percentages) are measurements of what happened. "
        "Archetypes, similarity scores and fit scores are model outputs built on those "
        "measurements and carry the assumptions described in the app's Methodology page.*"
    )
    return "\n\n".join(parts)
