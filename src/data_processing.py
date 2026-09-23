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
    COUNTING_STATS, DATA_SOURCES, DEFAULT_FLANK, DEFAULT_SOURCE, FLANKS, POSITION_TO_GROUP,
    RAW_DIR, RAW_PLAYERS_CSV, TRANSFERMARKT_MARKET_CSV, WIDE_GROUPS,
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
    ("passes_under_pressure", "passes_attempted"),
    ("passes_completed_under_pressure", "passes_under_pressure"),
    ("npxg_open_play", "npxg"),
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

# A count column has to be measured for at least this share of a season's
# players before its gaps are read as "this did not happen" and zero-filled.
# Below it, the gaps are more likely a scrape or join hole than genuine zeros,
# and are left missing instead - see the fillna(0) loop below for the case
# that motivated this floor.
SEASON_COVERAGE_FLOOR = 0.90

# Advanced metrics that are genuinely optional in real feeds.
OPTIONAL_METRICS = [
    "xa", "xg", "npxg", "gk_psxg", "sca", "gca", "pressures", "pressures_successful",
    "passes_under_pressure", "passes_completed_under_pressure", "npxg_open_play", "npxg_set_piece",
]


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
    # Columns that ARE supplied, just not reliably enough in specific seasons
    # to zero-fill their gaps - column -> the season labels below the floor.
    # Kept separate from `unavailable` (never supplied, any season) because
    # "not supplied" and "not supplied for 2024-25" are different claims, and
    # a source with eight good seasons and one thin one should not be reported
    # as though it lacked the metric altogether.
    partial: dict[str, list[str]] = field(default_factory=dict)
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
        unavailable = sorted(set(self.unavailable))
        if unavailable:
            rows.append(
                ("Columns this source does not supply (left missing, never zero-filled)",
                 f"{len(unavailable):,}")
            )
        if self.partial:
            rows.append(
                ("Columns unreliable in specific seasons (left missing there, not zero-filled)",
                 f"{len(self.partial):,}")
            )
        return rows

    @property
    def unavailable_columns(self) -> list[str]:
        return sorted(set(self.unavailable))

    @property
    def partial_columns(self) -> dict[str, list[str]]:
        return {column: sorted(set(seasons)) for column, seasons in self.partial.items()}


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
        # Add every unsupplied counting stat in one concat: inserting ~90 columns
        # one at a time fragments the frame badly on a wide source.
        absent = [stat for stat in COUNTING_STATS if stat not in frame.columns]
        if absent:
            frame = pd.concat(
                [frame, pd.DataFrame(np.nan, index=frame.index, columns=absent)], axis=1
            )
        frame = _apply_market_enrichment(frame, spec.key)
        return frame, spec.key
    return load_raw_players(), "simulated"


ENRICHMENT_NOTE = "market_enrichment"


def _apply_market_enrichment(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Join Transfermarkt market values and true positions, when they are built.

    A performance feed with four positional buckets and no valuation becomes far
    more useful with a real position and a real price attached. The enrichment
    is opt-in by existence: build the spine with
    `scripts/fetch_transfermarkt.py --market-only` and every source that can be
    matched by name picks it up. Nothing is guessed - unmatched players keep
    exactly what they had.
    """
    if source == "transfermarkt" or not TRANSFERMARKT_MARKET_CSV.exists():
        return frame
    try:
        from .transfermarkt import enrich

        market = pd.read_csv(TRANSFERMARKT_MARKET_CSV)
        enriched, report = enrich(frame, market)
        enriched.attrs[ENRICHMENT_NOTE] = report
        return enriched
    except Exception:  # enrichment is a bonus; never let it break ingestion
        return frame


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
    enrichment = df.attrs.get(ENRICHMENT_NOTE)
    if enrichment:
        report.notes.append(
            f"Transfermarkt market data matched {enrichment['matched']:,} of "
            f"{enrichment['of']:,} player-seasons, adding market value and a true position. "
            "Unmatched players keep the source's own position."
        )
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
    # Age and height are never imputed. They are facts about a person, not
    # measurements of a season, and no model reads them as a feature - they
    # only feed filters, the hidden-gem age term and the "younger version of"
    # search, which is exactly where a positional median does damage: it
    # would tell an age filter that an unknown player is 26.3, or put
    # "183 cm" on a profile as if someone had measured him. Unknown stays
    # unknown, and every consumer treats it as such.
    for column in ["height_cm", "age"]:
        if column not in df.columns:
            continue
        if df[column].isna().all():
            # The source cannot supply this at all (StatsBomb publishes no birth
            # dates); the app switches off the features that depend on it.
            report.unavailable.append(column)
            continue
        missing = int(df[column].isna().sum())
        if missing:
            report.notes.append(
                f"{missing:,} player-seasons have no recorded {column.replace('_cm', '')} "
                "and are left without one rather than given a positional average; an "
                "age or height filter excludes them instead of guessing."
            )

    # Rate-like season totals: the positional median *per 90*, rescaled to the
    # player's own minutes, within seasons that measured them (below).

    # Goalkeeping columns only apply to goalkeepers; they stay NaN for
    # outfielders so that "no value" is never confused with "zero".
    gk_mask = df["position_group"].eq("GK")
    # Same per-season coverage floor as the counting stats below, applied to
    # imputation rather than zero-fill. A *global* median would paper over a
    # provider switch exactly like FBref's Opta cutover, which took
    # "pressures" from ~99% covered to a flat 0% for three straight seasons:
    # imputing those seasons from the pre-cutover median would hand every
    # 2022/23-on player a fabricated pressures count that reads as real data.
    # Only gaps in a season that clears the floor get the per-90
    # positional-median treatment, computed from that reliable data alone; a
    # season with no real signal is left missing and reported as partial (or
    # unavailable if no season ever clears the floor), never guessed at from
    # a different era.
    seasons = df["season"] if "season" in df.columns else pd.Series("all", index=df.index)
    season_groups = df.groupby(seasons, observed=True).groups
    for column in OPTIONAL_METRICS:
        if column not in df.columns:
            continue
        applicable = gk_mask if column.startswith("gk_") else pd.Series(True, index=df.index)
        reliable_scope: list[pd.Index] = []
        thin_seasons = []
        for season, block in season_groups.items():
            rows = df.index.intersection(block)
            scope = rows[applicable.loc[rows]]
            if len(scope) == 0:
                continue
            if df.loc[scope, column].notna().mean() >= SEASON_COVERAGE_FLOOR:
                reliable_scope.append(scope)
            else:
                thin_seasons.append(str(season))
        if not reliable_scope:
            # Not reliably measured in any season - genuinely unsupplied.
            report.unavailable.append(column)
            continue
        reliable_index = reliable_scope[0].append(reliable_scope[1:])
        if thin_seasons:
            report.partial.setdefault(column, []).extend(thin_seasons)
        gaps = reliable_index[df.loc[reliable_index, column].isna()]
        if len(gaps) == 0:
            continue
        per90 = df.loc[reliable_index, column] / df.loc[reliable_index, "minutes"] * 90
        median_per90 = per90.groupby(
            df.loc[reliable_index, "position_group"]
        ).transform("median").fillna(0.0)
        df.loc[gaps, column] = (median_per90 * df.loc[reliable_index, "minutes"] / 90).loc[gaps]
        report.imputed[column] = int(len(gaps))

    # Remaining counting stats are genuinely zero-or-absent events - except any
    # column the source does not publish at all, which stays missing so the
    # models drop it rather than reading a real zero into it. "Did not attempt a
    # single long pass all season" and "this feed does not count long passes"
    # are completely different claims, and only one of them is true.
    # A missing count means "this did not happen" only when the season measured
    # it at all. Summary feeds add metrics over time - the Premier League feed
    # began counting tackles in 2025/26 - so the test is applied season by
    # season: a column absent for a whole season stays missing there, and is
    # zero-filled in the seasons that do measure it.
    # A season/column combination can be mostly measured and still not fully -
    # FBref's defense, possession and misc blocks cover only ~75% of players in
    # 2024/25 (they stop being scraped part-way through, well before the whole
    # column goes empty in 2025/26). The old rule only caught a column once it
    # was *entirely* null for a season, so that 75% would have zero-filled the
    # other quarter: a real defender reading as zero tackles for a whole season
    # because his row happened to miss one block's join, not because he made
    # none. A coverage floor catches that case while still zero-filling the
    # ordinary ~95% baseline noise every source carries.
    #
    # Two passes: first measure every (column, season) pair, then decide what
    # to report. A column below the floor in every season is genuinely
    # unavailable; one below the floor in only some seasons is available, just
    # not everywhere - reported separately so a source with eight good
    # seasons and one thin one is never described as lacking the metric.
    # (`seasons`/`season_groups` were already computed above, for the same
    # floor applied to OPTIONAL_METRICS imputation.)
    for column in count_cols:
        applicable = gk_mask if column.startswith("gk_") else pd.Series(True, index=df.index)
        coverage_by_season: dict[str, tuple[pd.Index, float]] = {}
        for season, block in season_groups.items():
            rows = df.index.intersection(block)
            scope = rows[applicable.loc[rows]]
            if len(scope) == 0:
                continue
            coverage_by_season[str(season)] = (
                scope, df.loc[scope, column].notna().mean()
            )
        if not coverage_by_season:
            continue

        reliable = {s: sc for s, (sc, cov) in coverage_by_season.items()
                   if cov >= SEASON_COVERAGE_FLOOR}
        if not reliable:
            # Not reliably measured in any season - genuinely unsupplied,
            # whether that means a hard 0% or scattered noise below the floor
            # everywhere. Real values, if any, are left in place; nothing is
            # zero-filled.
            report.unavailable.append(column)
            continue

        thin_seasons = sorted(s for s in coverage_by_season if s not in reliable)
        if thin_seasons:
            report.partial.setdefault(column, []).extend(thin_seasons)
        for scope in reliable.values():
            df.loc[scope, column] = df.loc[scope, column].fillna(0)
    # Goalkeeping columns never apply to outfielders.
    df.loc[~gk_mask, [c for c in count_cols if c.startswith("gk_")]] = np.nan

    if "age" in df.columns and df["age"].notna().any():
        df["age"] = df["age"].round(1)
    if "height_cm" in df.columns and df["height_cm"].notna().any():
        df["height_cm"] = df["height_cm"].round()

    df = add_flank(df)
    report.rows_out = len(df)
    return df, report


# --------------------------------------------------------------------------
# Side of the pitch
# --------------------------------------------------------------------------

def add_flank(df: pd.DataFrame) -> pd.DataFrame:
    """Which side a player is listed on, and whether that side is inverted.

    The separability test says a left-back and a right-back do the same job, so
    they share a model. But a club recruiting a left-back does not want
    right-backs on the shortlist, which makes side a filter rather than a group.

    `inverted` is the scouting idea the pair unlocks: a right-footed left winger
    cuts inside onto his stronger foot; a left-footed one goes outside and
    crosses. Same position, different player, and until now nothing here could
    tell them apart. It is only set for the wide groups, where the distinction
    means something - a right-footed centre-back is not "inverted".
    """
    df = df.copy()
    if "position" not in df.columns:
        return df

    df["flank"] = df["position"].map(FLANKS).fillna(DEFAULT_FLANK)

    if "foot" not in df.columns or df["foot"].isna().all():
        return df
    foot = df["foot"].astype(str).str.strip().str.lower()
    wide = df["position_group"].isin(WIDE_GROUPS) if "position_group" in df.columns else False
    opposite = ((df["flank"].eq("Left") & foot.eq("right"))
                | (df["flank"].eq("Right") & foot.eq("left")))
    same = ((df["flank"].eq("Left") & foot.eq("left"))
            | (df["flank"].eq("Right") & foot.eq("right")))
    df["footed_side"] = np.where(
        ~wide | ~(opposite | same), None,
        np.where(opposite, "Inverted", "Natural"),
    )
    return df


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
        pool = pool[age_mask(pool["age"], age_range)]
    return pool.reset_index(drop=True)


class AgeRange(tuple):
    """An (low, high) age range that knows whether it actually narrows anything.

    Still a plain tuple to every caller that unpacks it; the flag only matters
    to `age_mask`.
    """

    active: bool

    def __new__(cls, low: float, high: float, active: bool = True):
        obj = super().__new__(cls, (low, high))
        obj.active = active
        return obj


def age_mask(ages: pd.Series, age_range) -> pd.Series:
    """Rows whose age is inside the range.

    A player with no recorded age can never be *confirmed* to be inside a
    narrowed range, so he is left out of "under 23" - but while the range is
    untouched it filters nothing, and nobody should vanish from a list just
    because a source did not publish his birth date. A plain tuple counts as
    narrowed; an `AgeRange` says for itself.
    """
    if age_range is None:
        return pd.Series(True, index=ages.index)
    inside = ages.between(*age_range)
    if getattr(age_range, "active", True):
        return inside
    return inside | ages.isna()
