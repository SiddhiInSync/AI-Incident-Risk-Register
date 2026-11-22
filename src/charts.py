"""Plotly figures shared by the dashboard. Dark surface, thin marks, recessive grid."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.taxonomy import CATEGORIES, CATEGORY_COLORS, SEVERITY_COLORS, SEVERITY_LEVELS

SURFACE = "#171a21"
TEXT = "#f0efec"
TEXT_2 = "#c3c2b7"
MUTED = "#8a8980"
GRID = "#2a2e37"
# Sequential (one hue, dark -> light on a dark surface) for count heatmaps.
SEQ_BLUE = [[0.0, "#1b2230"], [0.25, "#123a6e"], [0.6, "#1c5cab"], [1.0, "#3f8ae6"]]


def _layout(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_2, size=12),
        hoverlabel=dict(bgcolor="#0f1115", bordercolor=GRID, font=dict(color=TEXT)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=TEXT_2), bgcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def category_frequency_bar(freq: pd.DataFrame) -> go.Figure:
    """Horizontal bars, most frequent on top, split by severity share."""
    d = freq.sort_values("incidents")
    fig = go.Figure(go.Bar(
        x=d["incidents"], y=d["risk_category"], orientation="h",
        marker=dict(color=[CATEGORY_COLORS[c] for c in d["risk_category"]],
                    line=dict(color=SURFACE, width=2), cornerradius=4),
        text=[f"{n} ({s:.0f}%)" for n, s in zip(d["incidents"], d["share_pct"])],
        textposition="outside", textfont=dict(color=TEXT_2),
        customdata=d[["share_pct", "high_crit_pct", "avg_severity"]],
        hovertemplate="<b>%{y}</b><br>%{x} incidents (%{customdata[0]}%)"
                      "<br>High/Critical: %{customdata[1]}%<br>Avg severity: %{customdata[2]} / 4<extra></extra>",
    ))
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes(range=[0, d["incidents"].max() * 1.5], showgrid=False, showticklabels=False)
    fig.update_yaxes(tickfont=dict(color=TEXT_2, size=13))
    return _layout(fig, height=300, legend=False)


def severity_heatmap(m: pd.DataFrame, title_y: str = "") -> go.Figure:
    """Category (rows) x severity (cols) count matrix — the register's risk matrix."""
    fig = go.Figure(go.Heatmap(
        z=m.values, x=list(m.columns), y=list(m.index),
        colorscale=SEQ_BLUE, showscale=False, xgap=2, ygap=2,
        text=m.values, texttemplate="%{text}", textfont=dict(color=TEXT, size=13),
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z} incidents<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_2, size=12), title=title_y, showgrid=False)
    fig.update_xaxes(side="top", tickfont=dict(color=TEXT_2, size=12), showgrid=False)
    return _layout(fig, height=40 + 44 * len(m), legend=False)


def trend_lines(t: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c in CATEGORIES:
        if c not in t or t[c].sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=t.index, y=t[c], name=c, mode="lines+markers",
            line=dict(color=CATEGORY_COLORS[c], width=2),
            marker=dict(size=8, color=CATEGORY_COLORS[c], line=dict(color=SURFACE, width=2)),
            hovertemplate=f"{c}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="Incidents", rangemode="tozero")
    return _layout(fig, height=340)


def stacked_trend_bars(t: pd.DataFrame) -> go.Figure:
    """Quarterly incident volume, stacked by category: shows total growth and mix together."""
    fig = go.Figure()
    for c in CATEGORIES:
        if c not in t or t[c].sum() == 0:
            continue
        fig.add_trace(go.Bar(
            x=t.index, y=t[c], name=c,
            marker=dict(color=CATEGORY_COLORS[c], line=dict(color=SURFACE, width=1)),
            hovertemplate=f"{c}: %{{y}}<extra></extra>",
        ))
    totals = t.sum(axis=1)
    fig.add_trace(go.Scatter(x=t.index, y=totals, mode="text", text=totals, textposition="top center",
                             textfont=dict(color=MUTED, size=10), showlegend=False, hoverinfo="skip"))
    fig.update_layout(barmode="stack", hovermode="x unified", bargap=0.2, legend_traceorder="normal")
    fig.update_yaxes(title="Incidents", range=[0, totals.max() * 1.15])
    return _layout(fig, height=340)


def stacked_severity_bar(m: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Horizontal 100%-stacked severity mix per row (sector, population...)."""
    d = m.loc[m.sum(axis=1).sort_values().index]
    pct = d.div(d.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for s in SEVERITY_LEVELS:
        fig.add_trace(go.Bar(
            y=d.index, x=pct[s], name=s, orientation="h",
            marker=dict(color=SEVERITY_COLORS[s], line=dict(color=SURFACE, width=2)),
            customdata=d[s], hovertemplate=f"<b>%{{y}}</b><br>{s}: %{{customdata}} (%{{x:.0f}}%)<extra></extra>",
        ))
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(ticksuffix="%", range=[0, 110], tickvals=[0, 25, 50, 75, 100])
    fig.update_yaxes(tickfont=dict(color=TEXT_2))
    # Annotate total incidents at bar end
    for y, total in d.sum(axis=1).items():
        fig.add_annotation(x=101, y=y, text=f"{total}", showarrow=False, xanchor="left",
                           font=dict(color=MUTED, size=11))
    return _layout(fig, height=height or 60 + 30 * len(d))


def sector_category_heatmap(m: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=m.values, x=list(m.columns), y=list(m.index),
        colorscale=SEQ_BLUE, showscale=False, xgap=2, ygap=2,
        text=m.values, texttemplate="%{text}", textfont=dict(color=TEXT, size=12),
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z} incidents<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TEXT_2), showgrid=False)
    fig.update_xaxes(side="top", tickfont=dict(color=TEXT_2), showgrid=False)
    return _layout(fig, height=60 + 34 * len(m), legend=False)


def simple_bar(df: pd.DataFrame, x: str, y: str, color: str = "#3987e5", fmt: str = "",
               hover_extra: str = "") -> go.Figure:
    d = df.sort_values(x)
    fig = go.Figure(go.Bar(
        x=d[x], y=d[y], orientation="h",
        marker=dict(color=color, line=dict(color=SURFACE, width=2), cornerradius=4),
        text=[f"{v:{fmt}}" for v in d[x]], textposition="outside", textfont=dict(color=TEXT_2),
        hovertemplate=f"<b>%{{y}}</b><br>%{{x}}{hover_extra}<extra></extra>",
    ))
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes(range=[0, d[x].max() * 1.25], showgrid=False, showticklabels=False)
    fig.update_yaxes(tickfont=dict(color=TEXT_2))
    return _layout(fig, height=60 + 32 * len(d), legend=False)


def growth_bar(g: pd.DataFrame) -> go.Figure:
    d = g.sort_values("change_pct")
    colors = ["#d03b3b" if v > 0 else "#0ca30c" for v in d["change_pct"]]
    fig = go.Figure(go.Bar(
        x=d["change_pct"], y=d["risk_category"], orientation="h",
        marker=dict(color=colors, line=dict(color=SURFACE, width=2), cornerradius=4),
        text=[f"{v:+.0f}%" for v in d["change_pct"]], textposition="outside", textfont=dict(color=TEXT_2),
        customdata=d[["recent", "prior"]],
        hovertemplate="<b>%{y}</b><br>Last 12 mo: %{customdata[0]}<br>Prior 12 mo: %{customdata[1]}"
                      "<br>Change: %{x:+.1f}%<extra></extra>",
    ))
    lo, hi = min(0, d["change_pct"].min()), max(0, d["change_pct"].max())
    pad = (hi - lo) * 0.25 + 5
    fig.update_xaxes(range=[lo - (pad if lo < 0 else 0), hi + pad], ticksuffix="%", zerolinecolor=MUTED)
    fig.update_yaxes(tickfont=dict(color=TEXT_2))
    return _layout(fig, height=260, legend=False)
