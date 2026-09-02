"""
Plotly chart builders.

All charts share one dark surface, one type scale and one colour system:

* up to three players are distinguished by the three validated categorical
  hues; anything with more categories (archetype maps, league splits) uses
  **highlight-against-grey** or small multiples instead of inventing more hues;
* percentiles are a *polarity* encoding - blue above the positional average,
  red below, grey at the 50th - because 50 is a real midpoint, not a minimum;
* inertia and silhouette are drawn as two separate charts. They have different
  units, and a twin-axis chart would invite a comparison that means nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import FONT_FAMILY, SERIES_COLORS, THEME

HOVER = dict(
    bgcolor=THEME["panel"],
    bordercolor=THEME["baseline"],
    font=dict(color=THEME["ink"], family=FONT_FAMILY, size=12),
)


def style(fig: go.Figure, height: int | None = None, title: str | None = None,
          showlegend: bool | None = None, margin: dict | None = None) -> go.Figure:
    """Apply the shared chart chrome."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["surface"],
        plot_bgcolor=THEME["surface"],
        font=dict(family=FONT_FAMILY, color=THEME["ink_secondary"], size=13),
        title=dict(text=title, font=dict(color=THEME["ink"], size=16)) if title else None,
        margin=margin or dict(l=8, r=8, t=48 if title else 16, b=8),
        hoverlabel=HOVER,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=THEME["ink_secondary"]),
        ),
    )
    if height:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    fig.update_xaxes(
        gridcolor=THEME["grid"], zerolinecolor=THEME["baseline"],
        linecolor=THEME["baseline"], tickfont=dict(color=THEME["ink_muted"]),
    )
    fig.update_yaxes(
        gridcolor=THEME["grid"], zerolinecolor=THEME["baseline"],
        linecolor=THEME["baseline"], tickfont=dict(color=THEME["ink_muted"]),
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def percentile_color(value: float) -> str:
    """Diverging colour for a percentile: blue above 50, red below, grey at 50."""
    if value is None or not np.isfinite(value):
        return THEME["context"]
    if value >= 50:
        weight = (value - 50) / 50
        return _rgba(THEME["diverging_high"], 0.35 + 0.65 * weight)
    weight = (50 - value) / 50
    return _rgba(THEME["diverging_low"], 0.35 + 0.65 * weight)


# --------------------------------------------------------------------------
# Radar
# --------------------------------------------------------------------------

def radar_chart(
    series: list[tuple[str, dict[str, float]]],
    title: str | None = None,
    height: int = 460,
) -> go.Figure:
    """Percentile radar for one to three players on shared category axes."""
    fig = go.Figure()
    categories = list(series[0][1].keys()) if series else []
    for i, (name, values) in enumerate(series[:3]):
        colour = SERIES_COLORS[i % len(SERIES_COLORS)]
        ordered = [values.get(c, np.nan) for c in categories]
        fig.add_trace(
            go.Scatterpolar(
                r=ordered + ordered[:1],
                theta=categories + categories[:1],
                name=name,
                mode="lines+markers",
                line=dict(color=colour, width=2),
                marker=dict(size=8, color=colour, line=dict(color=THEME["surface"], width=2)),
                fill="toself",
                fillcolor=_rgba(colour, 0.10),
                hovertemplate="<b>%{theta}</b><br>" + name + ": %{r:.0f}th percentile<extra></extra>",
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor=THEME["surface"],
            radialaxis=dict(
                range=[0, 100], tickvals=[25, 50, 75, 100], tickfont=dict(color=THEME["ink_muted"], size=10),
                gridcolor=THEME["grid"], linecolor=THEME["grid"], angle=90,
            ),
            angularaxis=dict(
                tickfont=dict(color=THEME["ink_secondary"], size=12),
                gridcolor=THEME["grid"], linecolor=THEME["baseline"],
            ),
        )
    )
    return style(fig, height=height, title=title, showlegend=len(series) > 1)


# --------------------------------------------------------------------------
# Percentile bars
# --------------------------------------------------------------------------

def percentile_bars(
    frame: pd.DataFrame,
    label_col: str = "metric",
    value_col: str = "percentile",
    raw_col: str | None = "value",
    title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Horizontal percentile bars, ordered best to worst, labelled at the tip."""
    data = frame.dropna(subset=[value_col]).sort_values(value_col)
    colours = [percentile_color(v) for v in data[value_col]]
    raw = data[raw_col] if raw_col and raw_col in data.columns else data[value_col]
    fig = go.Figure(
        go.Bar(
            x=data[value_col],
            y=data[label_col],
            orientation="h",
            marker=dict(color=colours, line=dict(width=0)),
            text=[f"{v:.0f}" for v in data[value_col]],
            textposition="outside",
            textfont=dict(color=THEME["ink_secondary"], size=11),
            customdata=np.stack([raw], axis=-1),
            hovertemplate="<b>%{y}</b><br>%{x:.0f}th percentile<br>value: %{customdata[0]:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=50, line=dict(color=THEME["baseline"], width=1))
    fig.update_xaxes(range=[0, 112], tickvals=[0, 25, 50, 75, 100], title=None)
    fig.update_yaxes(title=None, automargin=True)
    fig.update_layout(bargap=0.35)
    return style(fig, height=height or max(260, 26 * len(data) + 70), title=title, showlegend=False)


def category_bars(
    scores: dict[str, dict[str, float]], title: str | None = None, height: int = 380
) -> go.Figure:
    """Grouped category percentiles for two or three players (the precise read)."""
    names = list(scores)
    categories = list(next(iter(scores.values())).keys())
    fig = go.Figure()
    for i, name in enumerate(names[:3]):
        colour = SERIES_COLORS[i % len(SERIES_COLORS)]
        fig.add_trace(
            go.Bar(
                name=name,
                x=categories,
                y=[scores[name].get(c, np.nan) for c in categories],
                marker=dict(color=colour, line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>" + name + ": %{y:.0f}th percentile<extra></extra>",
            )
        )
    fig.add_hline(y=50, line=dict(color=THEME["baseline"], width=1))
    fig.update_yaxes(range=[0, 100], title="Percentile vs positional peers")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    return style(fig, height=height, title=title, showlegend=len(names) > 1)


def metric_comparison_bars(
    frame: pd.DataFrame, metric_col: str, players: list[str], title: str | None = None
) -> go.Figure:
    """Raw per-90 comparison, one grouped bar per metric."""
    fig = go.Figure()
    for i, player in enumerate(players[:3]):
        colour = SERIES_COLORS[i % len(SERIES_COLORS)]
        fig.add_trace(
            go.Bar(
                name=player,
                y=frame[metric_col],
                x=frame[player],
                orientation="h",
                marker=dict(color=colour, line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>" + player + ": %{x:.2f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title=None, automargin=True)
    return style(fig, height=max(300, 34 * len(frame) + 90), title=title, showlegend=len(players) > 1)


# --------------------------------------------------------------------------
# Scatter / exploration
# --------------------------------------------------------------------------

def scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    hover_name: str,
    hover_cols: list[str] | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    title: str | None = None,
    highlight: pd.Series | None = None,
    highlight_label: str = "Shortlist",
    trend: bool = False,
    height: int = 480,
) -> go.Figure:
    """Scatter with optional highlight set drawn over grey context points."""
    hover_cols = hover_cols or []
    fig = go.Figure()

    def add(subset: pd.DataFrame, colour: str, name: str, size: int, opacity: float):
        if subset.empty:
            return
        custom = subset[hover_cols].to_numpy() if hover_cols else None
        lines = "<br>".join(
            f"{col.replace('_', ' ')}: %{{customdata[{i}]}}" for i, col in enumerate(hover_cols)
        )
        fig.add_trace(
            go.Scatter(
                x=subset[x], y=subset[y], mode="markers", name=name,
                marker=dict(
                    size=size, color=colour, opacity=opacity,
                    line=dict(color=THEME["surface"], width=2 if size >= 9 else 0),
                ),
                text=subset[hover_name], customdata=custom,
                hovertemplate="<b>%{text}</b><br>" + lines
                + f"<br>{x_title or x}: %{{x:.2f}}<br>{y_title or y}: %{{y:.2f}}<extra></extra>",
            )
        )

    if highlight is not None:
        mask = highlight.reindex(data.index).fillna(False).astype(bool)
        add(data[~mask], THEME["context"], "All players", 7, 0.55)
        add(data[mask], THEME["series_1"], highlight_label, 10, 0.95)
    else:
        add(data, THEME["series_1"], "Players", 8, 0.8)

    if trend and len(data) > 3:
        valid = data[[x, y]].dropna()
        if len(valid) > 3:
            slope, intercept = np.polyfit(valid[x], valid[y], 1)
            grid = np.linspace(valid[x].min(), valid[x].max(), 50)
            fig.add_trace(
                go.Scatter(
                    x=grid, y=slope * grid + intercept, mode="lines", name="Linear trend",
                    line=dict(color=THEME["ink_muted"], width=2, dash="dot"),
                    hovertemplate="trend<extra></extra>",
                )
            )
    fig.update_xaxes(title=x_title or x)
    fig.update_yaxes(title=y_title or y)
    return style(fig, height=height, title=title,
                 showlegend=highlight is not None or trend)


def cluster_map(
    coords: pd.DataFrame,
    meta: pd.DataFrame,
    archetype_col: str = "archetype",
    highlight: str | None = None,
    focus_index=None,
    explained: tuple[float, float] | None = None,
    height: int = 560,
) -> go.Figure:
    """PCA map of a position group, highlighting one archetype against context.

    Six archetypes cannot be told apart by hue alone at scatter density, so the
    map highlights one archetype at a time rather than colouring them all.
    """
    frame = coords.join(meta)
    fig = go.Figure()
    hover_cols = [c for c in ["player", "team", "league", "age", "minutes", archetype_col] if c in frame.columns]
    lines = "<br>".join(
        f"{col.replace('_', ' ')}: %{{customdata[{i}]}}" for i, col in enumerate(hover_cols)
    )

    def add(subset, colour, name, size, opacity):
        if subset.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=subset["pc1"], y=subset["pc2"], mode="markers", name=name,
                marker=dict(size=size, color=colour, opacity=opacity,
                            line=dict(color=THEME["surface"], width=2 if size >= 9 else 0)),
                customdata=subset[hover_cols].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><br>" + lines + "<extra></extra>",
            )
        )

    if highlight and archetype_col in frame.columns:
        mask = frame[archetype_col].eq(highlight)
        add(frame[~mask], THEME["context"], "Other archetypes", 7, 0.5)
        add(frame[mask], THEME["series_1"], highlight, 10, 0.95)
    else:
        add(frame, THEME["series_1"], "Players", 7, 0.7)

    if focus_index is not None and focus_index in frame.index:
        point = frame.loc[[focus_index]]
        add(point, THEME["series_2"], str(point["player"].iloc[0]), 15, 1.0)

    x_title = "Principal component 1" + (f" ({explained[0]:.0%} of variance)" if explained else "")
    y_title = "Principal component 2" + (f" ({explained[1]:.0%} of variance)" if explained else "")
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    return style(fig, height=height, showlegend=True)


def cluster_facets(
    coords: pd.DataFrame, meta: pd.DataFrame, archetype_col: str = "archetype", height: int = 620
) -> go.Figure:
    """Small multiples: one panel per archetype, each against grey context."""
    frame = coords.join(meta)
    archetypes = sorted(frame[archetype_col].dropna().unique())
    cols = min(3, max(1, len(archetypes)))
    rows = int(np.ceil(len(archetypes) / cols))
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[a[:38] for a in archetypes],
        horizontal_spacing=0.06, vertical_spacing=0.12,
    )
    for i, archetype in enumerate(archetypes):
        row, col = i // cols + 1, i % cols + 1
        mask = frame[archetype_col].eq(archetype)
        fig.add_trace(
            go.Scatter(
                x=frame.loc[~mask, "pc1"], y=frame.loc[~mask, "pc2"], mode="markers",
                marker=dict(size=4, color=THEME["context"], opacity=0.4), showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.loc[mask, "pc1"], y=frame.loc[mask, "pc2"], mode="markers",
                marker=dict(size=7, color=THEME["series_1"], opacity=0.9), showlegend=False,
                text=frame.loc[mask, "player"],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ),
            row=row, col=col,
        )
    fig.update_annotations(font=dict(size=12, color=THEME["ink_secondary"]))
    fig.update_xaxes(showticklabels=False, title=None)
    fig.update_yaxes(showticklabels=False, title=None)
    return style(fig, height=height, showlegend=False)


def distribution_with_marker(
    values: pd.Series, marker_value: float, marker_label: str,
    metric_label: str, height: int = 260,
) -> go.Figure:
    """Where a player sits inside the positional distribution of one metric."""
    fig = go.Figure(
        go.Histogram(
            x=values.dropna(), nbinsx=34,
            marker=dict(color=_rgba(THEME["series_1"], 0.45), line=dict(width=0)),
            hovertemplate=metric_label + ": %{x:.2f}<br>%{y} players<extra></extra>",
            name="Positional peers",
        )
    )
    fig.add_vline(
        x=marker_value, line=dict(color=THEME["series_2"], width=2),
        annotation_text=marker_label, annotation_position="top",
        annotation_font=dict(color=THEME["ink"], size=11),
    )
    fig.update_xaxes(title=metric_label)
    fig.update_yaxes(title="Players")
    fig.update_layout(bargap=0.04)
    return style(fig, height=height, showlegend=False)


# --------------------------------------------------------------------------
# Model diagnostics
# --------------------------------------------------------------------------

def elbow_chart(evaluation: pd.DataFrame, chosen_k: int, height: int = 300) -> go.Figure:
    """Inertia against k. Drawn separately from silhouette - different units."""
    fig = go.Figure(
        go.Scatter(
            x=evaluation["k"], y=evaluation["inertia"], mode="lines+markers",
            line=dict(color=THEME["series_1"], width=2),
            marker=dict(size=8, color=THEME["series_1"], line=dict(color=THEME["surface"], width=2)),
            hovertemplate="k=%{x}<br>inertia %{y:,.0f}<extra></extra>", name="Inertia",
        )
    )
    fig.add_vline(x=chosen_k, line=dict(color=THEME["ink_muted"], width=1, dash="dot"))
    fig.update_xaxes(title="Number of clusters (k)", dtick=1)
    fig.update_yaxes(title="Inertia (within-cluster sum of squares)")
    return style(fig, height=height, showlegend=False)


def silhouette_chart(evaluation: pd.DataFrame, chosen_k: int, height: int = 300) -> go.Figure:
    """Silhouette score against k."""
    fig = go.Figure(
        go.Scatter(
            x=evaluation["k"], y=evaluation["silhouette"], mode="lines+markers",
            line=dict(color=THEME["series_3"], width=2),
            marker=dict(size=8, color=THEME["series_3"], line=dict(color=THEME["surface"], width=2)),
            hovertemplate="k=%{x}<br>silhouette %{y:.3f}<extra></extra>", name="Silhouette",
        )
    )
    fig.add_vline(x=chosen_k, line=dict(color=THEME["ink_muted"], width=1, dash="dot"))
    fig.update_xaxes(title="Number of clusters (k)", dtick=1)
    fig.update_yaxes(title="Mean silhouette score")
    return style(fig, height=height, showlegend=False)


def contribution_chart(
    contributions: pd.DataFrame, top: int = 10, title: str | None = None
) -> go.Figure:
    """Share of the distance between two players carried by each metric."""
    data = contributions.head(top).sort_values("distance_share")
    fig = go.Figure(
        go.Bar(
            x=data["distance_share"] * 100, y=data["metric"], orientation="h",
            marker=dict(color=_rgba(THEME["series_2"], 0.85), line=dict(width=0)),
            text=[f"{v * 100:.0f}%" for v in data["distance_share"]],
            textposition="outside", textfont=dict(color=THEME["ink_secondary"], size=11),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% of the distance<extra></extra>",
        )
    )
    fig.update_xaxes(title="Share of the total distance between the two players (%)")
    fig.update_yaxes(title=None, automargin=True)
    fig.update_layout(bargap=0.35)
    return style(fig, height=max(280, 28 * len(data) + 90), title=title, showlegend=False)


def correlation_heatmap(corr: pd.DataFrame, labels: list[str], height: int = 620) -> go.Figure:
    """Feature correlation matrix on the diverging blue-red scale."""
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1,
            colorscale=[
                [0.0, THEME["diverging_low"]],
                [0.5, THEME["diverging_mid"]],
                [1.0, THEME["diverging_high"]],
            ],
            colorbar=dict(title="r", tickfont=dict(color=THEME["ink_muted"])),
            hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9), automargin=True)
    return style(fig, height=height, showlegend=False)


def component_bar(values: dict[str, float], title: str | None = None, height: int = 300) -> go.Figure:
    """Component breakdown of a composite score (e.g. hidden-gem components)."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
            marker=dict(color=[percentile_color(v) for _, v in items], line=dict(width=0)),
            text=[f"{v:.0f}" for _, v in items], textposition="outside",
            textfont=dict(color=THEME["ink_secondary"], size=11),
            hovertemplate="<b>%{y}</b><br>%{x:.0f} / 100<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 112], title="Component score (0-100)")
    fig.update_yaxes(title=None, automargin=True)
    fig.update_layout(bargap=0.35)
    return style(fig, height=height, title=title, showlegend=False)
