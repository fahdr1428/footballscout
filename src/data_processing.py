"""
Ingestion and cleaning.

Two entry points matter:

* `load_raw_players()`  - reads data/raw/players_raw.csv.gz, generating the
  simulated dataset on first use. It can also read a real FBref-style export
  (see `load_external_csv`).
* `clean_players()`     - deduplicates, repairs or drops impossible values,
  imputes the optional advanced metrics, and returns a `CleaningReport`
  describing exactly what it did. The report is displayed in the app so no
  cleaning decision is hidden from the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    COUNTING_STATS, DATA_SOURCES, DEFAULT_SOURCE, POSITION_TO_GROUP, RAW_DIR, RAW_PLAYERS_CSV,
)

MAX_MINUTES = 38 * 90  # a full domestic league season
MIN_HEIGHT, MAX_HEIGHT = 150, 215
MIN_AGE, MAX_AGE = 15, 45

IDENTITY_COLUMNS = [
    "player_id", "player", "position", "position_group", "team", "league",
    "league_tier", "league_strength", "season", "age", "height_cm",
    "minutes", "matches", "starts", "team_possession",
]

# Season totals that must never exceed their attempt column.
CONSISTENCY_PAIRS = [
    ("passes_completed", "passes_attempted"),
    ("long_passes_completed", "long_passes_attempted"),
    ("tackles_won", "tackles"),
    ("pressures_successful", "pressures"),
    ("dribbles_completed", "dribbles_attempted"),
    ("shots_on_target", "shots"),
    ("np_goals", "shots"),
    ("pens_scored", "pens_taken"),
    ("gk_saves", "gk_shots_on_target_against"),
    ("gk_crosses_stopped", "gk_crosses_faced"),
    ("gk_launches_completed", "gk_launches_attempted"),
    ("carries_into_final_third", "progressive_carries"),
]

# Advanced metrics that are genuinely optional in real feeds.
OPTIONAL_METRICS = ["xa", "xg", "npxg", "gk_psxg", "sca", "gca", "pressures", "pressures_successful"]


@dataclass
class CleaningReport:
    """Audit trail for everything the cleaner changed."""

    rows_in: int = 0
    rows_out: int = 0
    exact_duplicates: int = 0
    duplicate_player_seasons: int = 0
    invalid_minutes: int = 0
    negative_counts: int = 0
    implausible_heights: int = 0
    implausible_ages: int = 0
    consistency_fixes: int = 0
    imputed: dict[str, int] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = ""

    def as_rows(self) -> list[tuple[str, str]]:
        rows = [
            ("Rows read", f"{self.rows_in:,}"),
            ("Exact duplicate rows removed", f"{self.exact_duplicates:,}"),
            ("Duplicate player-seasons collapsed", f"{self.duplicate_player_seasons:,}"),
            ("Rows dropped - impossible minutes", f"{self.invalid_minutes:,}"),
            ("Rows dropped - negative counting stats", f"{self.negative_counts:,}"),
            ("Heights repaired (out of 150-215cm)", f"{self.implausible_heights:,}"),
            ("Ages repaired (out of 15-45)", f"{self.implausible_ages:,}"),
            ("Attempt/success inconsistencies clipped", f"{self.consistency_fixes:,}"),
            ("Rows after cleaning", f"{self.rows_out:,}"),
        ]
        for column, count in sorted(self.imputed.items()):
            rows.append((f"Imputed missing `{column}`", f"{count:,}"))
        for column in sorted(set(self.unavailable)):
            rows.append((f"Not supplied by this source: `{column}`", "left missing"))
        return rows


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_raw_players(path=None, regenerate: bool = False, seed: int = 7) -> pd.DataFrame:
    """Load the simulated raw player-season table, generating it on first use."""
    path = path or RAW_PLAYERS_CSV
    if regenerate or not path.exists():
        from .data_generation import generate_dataset

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        data = generate_dataset(seed=seed)
        data.to_csv(path, index=False)
        return data
    return pd.read_csv(path)


def load_source(source: str = DEFAULT_SOURCE) -> tuple[pd.DataFrame, str]:
    """Load one of the configured datasets, falling back if it is not built yet.

    Returns the frame and the source key actually used, so the app can tell the
    user when it fell back (the real dataset has to be downloaded once with
    `python scripts/fetch_statsbomb.py`).
    """
    spec = DATA_SOURCES.get(source) or DATA_SOURCES[DEFAULT_SOURCE]
    if spec.kind == "simulated":
        return load_raw_players(), spec.key
    if spec.path.exists():
        frame = pd.read_csv(spec.path)
        for stat in COUNTING_STATS:
            if stat not in frame.columns:
                frame[stat] = np.nan
        return frame, spec.key
    return load_raw_players(), "simulated"


def load_external_csv(path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a real player-season export (e.g. an FBref scrape) into our schema.

    `column_map` maps *source* column names to the names used here. Any of the
    counting stats in `config.COUNTING_STATS` that are absent are created as
    NaN, so the platform degrades gracefully when a feed is less detailed than
    the reference schema.
    """
    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns=column_map)
    required = {"player", "position", "team", "league", "season", "minutes", "age"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"External file is missing required columns: {sorted(missing)}")
    for stat in COUNTING_STATS:
        if stat not in df.columns:
            df[stat] = np.nan
    if "position_group" not in df.columns:
        df["position_group"] = df["position"].map(_map_position_group)
    if "player_id" not in df.columns:
        df["player_id"] = (
            df["player"].astype(str) + "|" + df["team"].astype(str)
        ).factorize()[0]
        df["player_id"] = "X" + df["player_id"].astype(str).str.zfill(5)
    for column, default in [("height_cm", np.nan), ("team_possession", 50.0),
                            ("league_tier", 1), ("league_strength", 1.0),
                            ("matches", np.nan), ("starts", np.nan)]:
        if column not in df.columns:
            df[column] = default
    return df


def _map_position_group(position: str) -> str:
    """Best-effort mapping of a free-text position string to a model group."""
    if not isinstance(position, str):
        return "CM"
    token = position.split(",")[0].strip().upper()
    if token in POSITION_TO_GROUP:
        return POSITION_TO_GROUP[token]
    lookup = {
        "GK": "GK", "DF": "CB", "CB": "CB", "LB": "FB", "RB": "FB", "WB": "FB",
        "DM": "DM", "MF": "CM", "CM": "CM", "AM": "AM", "LW": "W", "RW": "W",
        "W": "W", "FW": "FW", "ST": "FW", "CF": "FW",
    }
    return lookup.get(token, "CM")


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean_players(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Deduplicate, validate and impute. Returns the clean frame and a report."""
    report = CleaningReport(rows_in=len(df))
    df = df.copy()

    # 1. Exact duplicates -------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    report.exact_duplicates = before - len(df)

    # 2. Duplicate player-seasons: keep the row with the most minutes.
    if {"player_id", "season"}.issubset(df.columns):
        before = len(df)
        df = (
            df.sort_values("minutes", ascending=False)
            .drop_duplicates(subset=["player_id", "season"], keep="first")
            .reset_index(drop=True)
        )
        report.duplicate_player_seasons = before - len(df)

    # 3. Impossible minutes ----------------------------------------------
    bad_minutes = (df["minutes"] <= 0) | (df["minutes"] > MAX_MINUTES) | df["minutes"].isna()
    report.invalid_minutes = int(bad_minutes.sum())
    if report.invalid_minutes:
        report.notes.append(
            f"{report.invalid_minutes} rows had minutes outside 1-{MAX_MINUTES} and were dropped "
            "rather than clipped: a bad minutes value corrupts every per-90 rate."
        )
    df = df[~bad_minutes].reset_index(drop=True)

    # 4. Negative counting stats -----------------------------------------
    count_cols = [c for c in COUNTING_STATS if c in df.columns]
    negatives = (df[count_cols] < 0).any(axis=1)
    report.negative_counts = int(negatives.sum())
    df = df[~negatives].reset_index(drop=True)

    # 5. Implausible physical attributes ---------------------------------
    if "height_cm" in df.columns:
        bad_height = df["height_cm"].notna() & (
            (df["height_cm"] < MIN_HEIGHT) | (df["height_cm"] > MAX_HEIGHT)
        )
        report.implausible_heights = int(bad_height.sum())
        df.loc[bad_height, "height_cm"] = np.nan
    if "age" in df.columns:
        bad_age = df["age"].notna() & ((df["age"] < MIN_AGE) | (df["age"] > MAX_AGE))
        report.implausible_ages = int(bad_age.sum())
        df.loc[bad_age, "age"] = np.nan

    # 6. Attempt/success consistency --------------------------------------
    fixes = 0
    for made, attempted in CONSISTENCY_PAIRS:
        if made in df.columns and attempted in df.columns:
            broken = df[made] > df[attempted]
            fixes += int(broken.fillna(False).sum())
            df.loc[broken.fillna(False), made] = df.loc[broken.fillna(False), attempted]
    report.consistency_fixes = fixes

    # 7. Imputation --------------------------------------------------------
    # Physical attributes: positional median. Rate-like season totals: the
    # positional median *per 90*, rescaled to the player's own minutes.
    for column in ["height_cm", "age"]:
        if column not in df.columns:
            continue
        if df[column].isna().all():
            # The source cannot supply this at all (StatsBomb publishes no birth
            # dates). Leave it missing rather than inventing a value; the app
            # switches off the features that depend on it.
            report.unavailable.append(column)
            continue
        if df[column].isna().any():
            missing = int(df[column].isna().sum())
            df[column] = df.groupby("position_group")[column].transform(
                lambda s: s.fillna(s.median())
            )
            df[column] = df[column].fillna(df[column].median())
            report.imputed[column] = missing

    # Goalkeeping columns only apply to goalkeepers; they stay NaN for
    # outfielders so that "no value" is never confused with "zero".
    gk_mask = df["position_group"].eq("GK")
    for column in OPTIONAL_METRICS:
        if column not in df.columns:
            continue
        applicable = gk_mask if column.startswith("gk_") else pd.Series(True, index=df.index)
        gaps = applicable & df[column].isna()
        if not gaps.any():
            continue
        if df.loc[applicable, column].isna().all():
            report.unavailable.append(column)
            continue
        per90 = df[column] / df["minutes"] * 90
        median_per90 = per90.groupby(df["position_group"]).transform("median").fillna(0.0)
        df.loc[gaps, column] = (median_per90 * df["minutes"] / 90)[gaps]
        report.imputed[column] = int(gaps.sum())

    # Remaining counting stats are genuinely zero-or-absent events - except any
    # column the source does not publish at all, which stays missing so the
    # models drop it rather than reading a real zero into it.
    unavailable = set(report.unavailable)
    outfield_cols = [c for c in count_cols if not c.startswith("gk_") and c not in unavailable]
    gk_cols = [c for c in count_cols if c.startswith("gk_") and c not in unavailable]
    df[outfield_cols] = df[outfield_cols].fillna(0)
    df.loc[gk_mask, gk_cols] = df.loc[gk_mask, gk_cols].fillna(0)
    df.loc[~gk_mask, [c for c in count_cols if c.startswith("gk_")]] = np.nan

    if "age" in df.columns and df["age"].notna().any():
        df["age"] = df["age"].round(1)
    if "height_cm" in df.columns and df["height_cm"].notna().any():
        df["height_cm"] = df["height_cm"].round()
    report.rows_out = len(df)
    return df, report


# --------------------------------------------------------------------------
# Outliers and pool filtering
# --------------------------------------------------------------------------

def flag_outliers(
    df: pd.DataFrame, columns: list[str], threshold: float = 5.0, group: str = "position_group"
) -> pd.DataFrame:
    """Robust (median / MAD) outlier flags, computed within position group.

    Outliers are *flagged, not deleted*: an extreme per-90 rate is usually a
    small-minutes artefact, which the minimum-minutes filter already handles,
    and occasionally a genuinely exceptional player we do not want to discard.
    """
    columns = [c for c in columns if c in df.columns]
    scores = pd.DataFrame(index=df.index)
    for column in columns:
        def modified_z(series: pd.Series) -> pd.Series:
            median = series.median()
            mad = (series - median).abs().median()
            scale = 1.4826 * mad
            if not np.isfinite(scale) or scale == 0:
                return pd.Series(0.0, index=series.index)
            return (series - median) / scale

        scores[column] = df.groupby(group)[column].transform(modified_z).abs()
    out = pd.DataFrame(index=df.index)
    out["outlier_metrics"] = (scores > threshold).sum(axis=1)
    out["max_robust_z"] = scores.max(axis=1)
    out["is_outlier"] = out["outlier_metrics"] > 0
    return out


def filter_pool(
    df: pd.DataFrame,
    min_minutes: int = 900,
    seasons: list[str] | None = None,
    leagues: list[str] | None = None,
    position_groups: list[str] | None = None,
    age_range: tuple[float, float] | None = None,
    max_minutes: int | None = None,
) -> pd.DataFrame:
    """Apply the sample-size and scope filters that define a comparison pool."""
    pool = df[df["minutes"] >= min_minutes]
    if max_minutes is not None:
        pool = pool[pool["minutes"] <= max_minutes]
    if seasons:
        pool = pool[pool["season"].isin(seasons)]
    if leagues:
        pool = pool[pool["league"].isin(leagues)]
    if position_groups:
        pool = pool[pool["position_group"].isin(position_groups)]
    if age_range and "age" in pool.columns and pool["age"].notna().any():
        low, high = age_range
        pool = pool[pool["age"].between(low, high)]
    return pool.reset_index(drop=True)
