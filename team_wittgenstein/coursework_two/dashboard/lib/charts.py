"""Reusable Plotly chart factories.

Each function takes a DataFrame and returns a Plotly figure styled with
the dashboard theme. Pages just call these - no Plotly code in pages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import COLORS, FACTORS, SECTORS


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex string + alpha (0-1) to an rgba() string.

    Plotly rejects 8-digit hex (#rrggbbaa); use rgba() instead.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


# Standard Plotly toolbar config used across the dashboard.
# Friendly UX: +/- zoom, pan, reset (zoom out to full view), download.
# Removes box-zoom (confusing for new users) and other rarely used tools.
def chart_config(filename: str = "chart") -> dict:
    """Return a Plotly config dict with a friendly toolbar."""
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d",
            "lasso2d",
            "select2d",
            "autoScale2d",
        ],
        "scrollZoom": True,
        "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 2},
    }


# ---------------------------------------------------------------------------
# Performance charts
# ---------------------------------------------------------------------------


def equity_curve(returns: pd.DataFrame, show_benchmark: bool = True) -> go.Figure:
    """Cumulative net return vs benchmark."""
    fig = go.Figure()
    if returns.empty:
        return fig

    portfolio_cum = ((1 + returns["net_return"]).cumprod() - 1) * 100
    fig.add_trace(
        go.Scatter(
            x=returns["rebalance_date"],
            y=portfolio_cum,
            name="Strategy",
            line=dict(color=COLORS["primary"], width=2.5),
            hovertemplate="<b>Strategy</b><br>%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
        )
    )

    if show_benchmark:
        bench_cum = ((1 + returns["benchmark_return"]).cumprod() - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=returns["rebalance_date"],
                y=bench_cum,
                name="MSCI USA",
                line=dict(color=COLORS["secondary"], width=2, dash="dash"),
                hovertemplate=(
                    "<b>Benchmark</b><br>%{x|%b %Y}<br>%{y:.2f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=420,
        yaxis_title="Cumulative Return (%)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=50, b=80),
    )
    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    return fig


def drawdown(returns: pd.DataFrame) -> go.Figure:
    """Drawdown from running peak, filled red."""
    fig = go.Figure()
    if returns.empty:
        return fig

    cum = (1 + returns["net_return"]).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()) * 100

    fig.add_trace(
        go.Scatter(
            x=returns["rebalance_date"],
            y=dd,
            fill="tozeroy",
            fillcolor=_rgba(COLORS["danger"], 0.2),
            line=dict(color=COLORS["danger"], width=1.5),
            name="Drawdown",
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        height=280,
        yaxis_title="Drawdown (%)",
        showlegend=False,
    )
    return fig


def monthly_excess(returns: pd.DataFrame) -> go.Figure:
    """Bar chart of monthly excess returns, green/red coloured."""
    fig = go.Figure()
    if returns.empty:
        return fig

    excess = returns["excess_return"] * 100
    colors = [COLORS["long"] if v >= 0 else COLORS["short"] for v in excess]

    fig.add_trace(
        go.Bar(
            x=returns["rebalance_date"],
            y=excess,
            marker_color=colors,
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
            name="Excess Return",
        )
    )

    fig.update_layout(
        height=320,
        yaxis_title="Excess Return vs Benchmark (%)",
        showlegend=False,
        bargap=0.15,
    )
    return fig


def long_short_contribution(returns: pd.DataFrame) -> go.Figure:
    """Cumulative long and short return contributions."""
    fig = go.Figure()
    if returns.empty:
        return fig

    long_cum = returns["long_return"].cumsum() * 100
    short_cum = returns["short_return"].cumsum() * 100

    fig.add_trace(
        go.Scatter(
            x=returns["rebalance_date"],
            y=long_cum,
            name="Long contribution",
            line=dict(color=COLORS["long"], width=2),
            fill="tozeroy",
            fillcolor=_rgba(COLORS["long"], 0.13),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=returns["rebalance_date"],
            y=short_cum,
            name="Short contribution",
            line=dict(color=COLORS["short"], width=2),
            fill="tozeroy",
            fillcolor=_rgba(COLORS["short"], 0.13),
        )
    )

    fig.update_layout(
        height=350,
        yaxis_title="Cumulative Contribution (%)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=50, b=80),
    )
    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    return fig


def rolling_sharpe(returns: pd.DataFrame, window: int = 12) -> go.Figure:
    """Rolling N-month annualised Sharpe."""
    fig = go.Figure()
    if returns.empty or len(returns) < window:
        return fig

    net = returns["net_return"]
    rolling_mean = net.rolling(window).mean() * 12
    rolling_std = net.rolling(window).std() * np.sqrt(12)
    rolling = rolling_mean / rolling_std

    fig.add_trace(
        go.Scatter(
            x=returns["rebalance_date"],
            y=rolling,
            line=dict(color=COLORS["primary"], width=2),
            name="Rolling Sharpe",
            hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        yaxis_title="Sharpe Ratio",
        showlegend=False,
    )
    fig.add_hline(y=0, line_color=COLORS["border"], line_dash="dash")
    fig.add_hline(y=1, line_color=COLORS["success"], line_dash="dash", opacity=0.3)
    return fig


def monthly_turnover(returns: pd.DataFrame) -> go.Figure:
    """Bar chart of monthly turnover with mean line."""
    fig = go.Figure()
    if returns.empty:
        return fig

    turnover_pct = returns["turnover"] * 100
    fig.add_trace(
        go.Bar(
            x=returns["rebalance_date"],
            y=turnover_pct,
            marker=dict(
                color=COLORS["primary"],
                line=dict(color=COLORS["surface"], width=1),
            ),
            hovertemplate="%{x|%b %Y}<br>Turnover: %{y:.1f}%<extra></extra>",
            name="Turnover",
        )
    )
    avg = float(turnover_pct.mean())
    fig.add_hline(
        y=avg,
        line_color=COLORS["secondary"],
        line_dash="dash",
        line_width=2,
        annotation_text=f"<b>Mean {avg:.1f}%</b>",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["secondary"], size=14),
        annotation_bgcolor=_rgba(COLORS["secondary"], 0.12),
    )
    fig.update_layout(
        height=320,
        yaxis_title="Turnover (%)",
        xaxis_title="",
        showlegend=False,
        bargap=0.1,
        margin=dict(t=40, b=50),
    )
    return fig


def returns_histogram(returns: pd.DataFrame) -> go.Figure:
    """Histogram showing the distribution of monthly returns.

    Buckets the 60 monthly returns by value range. Each bar = how many
    months fell into that return range. Cannot identify specific months
    by design - use the Monthly excess chart for that.
    """
    fig = go.Figure()
    if returns.empty:
        return fig

    net = (returns["net_return"] * 100).to_numpy()
    counts, edges = np.histogram(net, bins=15)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_widths = edges[1:] - edges[:-1]
    bin_ranges = [f"{lo:+.1f}% to {hi:+.1f}%" for lo, hi in zip(edges[:-1], edges[1:])]

    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=bin_widths * 0.95,
            customdata=bin_ranges,
            marker=dict(
                color=COLORS["primary"],
                line=dict(color=COLORS["surface"], width=1),
            ),
            opacity=0.92,
            hovertemplate=(
                "<b>Returns from %{customdata}</b>"
                "<br>%{y} months in this range<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=net.mean(),
        line_color=COLORS["secondary"],
        line_dash="dash",
        annotation_text=f"Mean {net.mean():.2f}%",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["secondary"]),
    )
    fig.update_layout(
        height=320,
        xaxis_title="Monthly return (%)",
        yaxis_title="Number of months",
        showlegend=False,
        bargap=0,
        margin=dict(t=20, l=50, r=20, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Factor charts
# ---------------------------------------------------------------------------


def ic_weights_evolution(ic_weights: pd.DataFrame) -> go.Figure:
    """Stacked area: IC weight per factor over time."""
    fig = go.Figure()
    if ic_weights.empty:
        return fig

    pivot = ic_weights.pivot(
        index="rebalance_date", columns="factor_name", values="ic_weight"
    )
    factor_order = ["value", "quality", "momentum", "low_vol"]

    for factor in factor_order:
        if factor in pivot.columns:
            fig.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[factor],
                    name=factor.replace("_", " ").title(),
                    line=dict(color=FACTORS[factor], width=2),
                    stackgroup="one",
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "%{x|%b %Y}<br>%{y:.3f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=400,
        yaxis_title="IC Weight",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=50, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Sector charts
# ---------------------------------------------------------------------------


def sector_stock_count_bars(holdings: pd.DataFrame, direction: str) -> go.Figure:
    """Horizontal bar chart of how many stocks fall into each sector.

    Strategy is sector-neutral by design (each sector gets ~11.82% long /
    ~2.73% short), so weight-based bars look identical across sectors. This
    chart shows STOCK COUNT instead - reveals concentration (sector with
    1 stock vs 8 stocks both get the same total weight).
    """
    fig = go.Figure()
    if holdings.empty:
        return fig

    side = holdings[holdings["direction"] == direction]
    counts = side.groupby("sector").size().sort_values()

    color = COLORS["long"] if direction == "long" else COLORS["short"]

    fig.add_trace(
        go.Bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            marker_color=color,
            text=counts.values,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} stocks<extra></extra>",
            name=direction.title(),
        )
    )

    fig.update_layout(
        height=380,
        xaxis_title="Number of stocks",
        showlegend=False,
        margin=dict(t=60, l=160, r=50, b=50),
    )
    return fig


def sector_allocation_bars(
    holdings: pd.DataFrame, direction: str, target: float
) -> go.Figure:
    """Horizontal bar chart of sector allocation with target reference line."""
    fig = go.Figure()
    if holdings.empty:
        return fig

    side = holdings[holdings["direction"] == direction]
    by_sector = side.groupby("sector")["final_weight"].sum().abs() * 100

    color = COLORS["long"] if direction == "long" else COLORS["short"]

    fig.add_trace(
        go.Bar(
            x=by_sector.values,
            y=by_sector.index,
            orientation="h",
            marker_color=color,
            hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
            name=direction.title(),
        )
    )
    fig.add_vline(
        x=target * 100,
        line_color=COLORS["text_muted"],
        line_dash="dash",
        annotation_text=f"Target: {target * 100:.1f}%",
        annotation_position="top right",
    )

    fig.update_layout(
        height=380,
        xaxis_title="Allocation (%)",
        showlegend=False,
        margin=dict(t=60, l=160, r=30, b=50),
    )
    return fig


def net_sector_exposure(holdings: pd.DataFrame) -> go.Figure:
    """Long minus short exposure per sector. Should hover near zero."""
    fig = go.Figure()
    if holdings.empty:
        return fig

    longs = (
        holdings[holdings["direction"] == "long"]
        .groupby("sector")["final_weight"]
        .sum()
    )
    shorts = (
        holdings[holdings["direction"] == "short"]
        .groupby("sector")["final_weight"]
        .sum()
    )
    net = (longs.add(shorts, fill_value=0)) * 100

    colors = [COLORS["long"] if v >= 0 else COLORS["short"] for v in net.values]

    fig.add_trace(
        go.Bar(
            x=net.values,
            y=net.index,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Net: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=COLORS["border"], line_width=2)

    fig.update_layout(
        height=380,
        xaxis_title="Net Exposure (Long - Short, %)",
        showlegend=False,
        margin=dict(t=60, l=160, r=30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Robustness charts
# ---------------------------------------------------------------------------


def equity_curve_compare(
    returns_a: pd.DataFrame, label_a: str, returns_b: pd.DataFrame, label_b: str
) -> go.Figure:
    """Two scenarios overlaid on the same equity curve."""
    fig = go.Figure()
    if returns_a.empty and returns_b.empty:
        return fig

    if not returns_a.empty:
        cum_a = ((1 + returns_a["net_return"]).cumprod() - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=returns_a["rebalance_date"],
                y=cum_a,
                name=label_a,
                line=dict(color=COLORS["primary"], width=2.5),
                hovertemplate=(
                    f"<b>{label_a}</b><br>" "%{x|%b %Y}<br>%{y:.2f}%<extra></extra>"
                ),
            )
        )

    if not returns_b.empty:
        cum_b = ((1 + returns_b["net_return"]).cumprod() - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=returns_b["rebalance_date"],
                y=cum_b,
                name=label_b,
                line=dict(color=COLORS["secondary"], width=2.5, dash="dash"),
                hovertemplate=(
                    f"<b>{label_b}</b><br>" "%{x|%b %Y}<br>%{y:.2f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=420,
        yaxis_title="Cumulative Return (%)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=50, b=80),
    )
    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    return fig


def factor_correlation_heatmap(zscores_df: pd.DataFrame) -> go.Figure:
    """4x4 heatmap of pairwise correlations between the 4 factor z-scores."""
    fig = go.Figure()
    if zscores_df.empty:
        return fig

    # Compute correlation matrix
    rename = {
        "z_value": "Value",
        "z_quality": "Quality",
        "z_momentum": "Momentum",
        "z_low_vol": "Low Vol",
    }
    df = zscores_df[list(rename.keys())].rename(columns=rename)
    corr = df.corr()

    # Build text labels for each cell showing the correlation value
    text_labels = corr.round(3).astype(str).values

    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[
                [0, COLORS["short"]],
                [0.5, COLORS["surface_alt"]],
                [1, COLORS["long"]],
            ],
            zmid=0,
            zmin=-1,
            zmax=1,
            text=text_labels,
            texttemplate="<b>%{text}</b>",
            textfont=dict(color=COLORS["text"], size=14),
            colorbar=dict(
                title=dict(text="Corr", font=dict(color=COLORS["text"])),
                tickfont=dict(color=COLORS["text_secondary"]),
                outlinewidth=0,
            ),
            hovertemplate="<b>%{y} vs %{x}</b><br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        xaxis=dict(side="bottom", showspikes=False),
        yaxis=dict(autorange="reversed", showspikes=False),  # Value at top-left
        hovermode="closest",  # override global "x unified" - no spike lines
        margin=dict(t=20, l=80, r=20, b=50),
    )
    return fig


def composite_histogram(scores: pd.DataFrame) -> go.Figure:
    """Histogram of composite scores with long/short cutoff lines.

    The strategy picks the top 10% per sector for long and bottom 10% per
    sector for short. The chart adds two vertical lines at the 90th and
    10th percentile of the universe-wide composite, so the user can see
    roughly where those baskets begin (per-sector cutoffs vary slightly,
    but this gives a sense of the boundary).
    """
    fig = go.Figure()
    if scores.empty:
        return fig

    s = scores["composite_score"].to_numpy()
    counts, edges = np.histogram(s, bins=30)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_widths = edges[1:] - edges[:-1]
    bin_ranges = [f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(edges[:-1], edges[1:])]

    # Long basket = top 10% (right tail). Short basket = bottom 10% (left tail).
    p10 = float(np.percentile(s, 10))
    p90 = float(np.percentile(s, 90))

    # Colour bars by which basket region they fall into.
    # Middle 80% uses a neutral grey - using `primary` would look identical
    # to `long` because the brand colour IS green.
    bar_colors = []
    for c in centers:
        if c >= p90:
            bar_colors.append(COLORS["long"])  # green - long basket
        elif c <= p10:
            bar_colors.append(COLORS["short"])  # red - short basket
        else:
            bar_colors.append(COLORS["text_muted"])  # grey - middle 80%

    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=bin_widths * 0.95,
            customdata=bin_ranges,
            marker=dict(
                color=bar_colors,
                line=dict(color=COLORS["surface"], width=1),
            ),
            opacity=0.92,
            hovertemplate=("<b>Score %{customdata}</b><br>%{y} stocks<extra></extra>"),
        )
    )

    # Cutoff lines (without inline annotations - those go above plot area)
    fig.add_vline(x=p90, line_color=COLORS["long"], line_dash="dash", line_width=2)
    fig.add_vline(x=p10, line_color=COLORS["short"], line_dash="dash", line_width=2)

    # Annotations sit ABOVE the plot area (yref="paper", y > 1) so they
    # never overlap the bars themselves.
    fig.add_annotation(
        x=p90,
        y=1.12,
        yref="paper",
        text=f"<b>~Long basket cutoff: {p90:+.2f}</b>",
        showarrow=False,
        xanchor="center",
        font=dict(color=COLORS["long"], size=13),
        bgcolor=_rgba(COLORS["long"], 0.15),
        bordercolor=COLORS["long"],
        borderwidth=1,
        borderpad=4,
    )
    fig.add_annotation(
        x=p10,
        y=1.12,
        yref="paper",
        text=f"<b>~Short basket cutoff: {p10:+.2f}</b>",
        showarrow=False,
        xanchor="center",
        font=dict(color=COLORS["short"], size=13),
        bgcolor=_rgba(COLORS["short"], 0.15),
        bordercolor=COLORS["short"],
        borderwidth=1,
        borderpad=4,
    )

    fig.update_layout(
        height=440,
        xaxis_title="Composite score (universe-wide)",
        yaxis_title="Number of stocks",
        showlegend=False,
        bargap=0,
        margin=dict(t=80, l=50, r=20, b=50),
    )
    return fig


def factor_zscore_boxplot(zscores: pd.DataFrame, factor_label: str) -> go.Figure:
    """Boxplot of z-scores by sector for one factor on one date."""
    fig = go.Figure()
    if zscores.empty:
        return fig
    sectors = sorted(zscores["sector"].dropna().unique())
    for i, sector in enumerate(sectors):
        sector_z = zscores[zscores["sector"] == sector]["z"]
        fig.add_trace(
            go.Box(
                y=sector_z,
                name=sector,
                marker_color=SECTORS[i % len(SECTORS)],
                boxmean=True,
            )
        )
    fig.add_hline(y=0, line_color=COLORS["border"])
    fig.update_layout(
        height=400,
        yaxis_title=f"{factor_label} z-score",
        showlegend=False,
        margin=dict(t=20, l=50, r=20, b=120),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def stock_price_with_markers(
    prices: pd.DataFrame, position_history: pd.DataFrame
) -> go.Figure:
    """Daily price line with markers for long/short entry rebalances."""
    fig = go.Figure()
    if prices.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=prices["trade_date"],
            y=prices["adjusted_close"],
            mode="lines",
            name="Adjusted close",
            line=dict(color=COLORS["text_secondary"], width=1.5),
            hovertemplate="%{x|%b %d %Y}<br>$%{y:.2f}<extra></extra>",
        )
    )

    if not position_history.empty:
        longs = position_history[position_history["direction"] == "long"]
        shorts = position_history[position_history["direction"] == "short"]
        # Get price at each rebalance date
        price_lookup = prices.set_index("trade_date")["adjusted_close"]

        def _price_at(d):
            try:
                return float(price_lookup.asof(pd.Timestamp(d)))
            except Exception:
                return None

        if not longs.empty:
            xs = longs["rebalance_date"]
            ys = [_price_at(d) for d in xs]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name="Long held",
                    marker=dict(color=COLORS["long"], size=9, symbol="circle"),
                    hovertemplate="<b>LONG</b><br>%{x|%b %Y}<br>"
                    "Weight: %{customdata:.2%}<extra></extra>",
                    customdata=longs["final_weight"],
                )
            )
        if not shorts.empty:
            xs = shorts["rebalance_date"]
            ys = [_price_at(d) for d in xs]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name="Short held",
                    marker=dict(color=COLORS["short"], size=9, symbol="circle"),
                    hovertemplate="<b>SHORT</b><br>%{x|%b %Y}<br>"
                    "Weight: %{customdata:.2%}<extra></extra>",
                    customdata=shorts["final_weight"],
                )
            )

    fig.update_layout(
        height=420,
        yaxis_title="Adjusted close ($)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=30, b=80),
    )
    return fig


def stock_factor_zscores(scores: pd.DataFrame) -> go.Figure:
    """4 factor z-scores over time for a stock."""
    fig = go.Figure()
    if scores.empty:
        return fig

    factor_cols = {
        "z_value": ("Value", FACTORS["value"]),
        "z_quality": ("Quality", FACTORS["quality"]),
        "z_momentum": ("Momentum", FACTORS["momentum"]),
        "z_low_vol": ("Low Vol", FACTORS["low_vol"]),
    }
    for col, (label, color) in factor_cols.items():
        if col in scores.columns:
            fig.add_trace(
                go.Scatter(
                    x=scores["score_date"],
                    y=scores[col],
                    name=label,
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{label}</b><br>"
                    "%{x|%b %Y}<br>z = %{y:.2f}<extra></extra>",
                )
            )
    fig.add_hline(y=0, line_color=COLORS["border"])
    fig.add_hline(y=1, line_color=COLORS["border"], line_dash="dash", opacity=0.4)
    fig.add_hline(y=-1, line_color=COLORS["border"], line_dash="dash", opacity=0.4)

    fig.update_layout(
        height=380,
        yaxis_title="Sector-relative z-score",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=30, b=80),
    )
    return fig


def stock_fundamental_line(metrics: pd.DataFrame, col: str, title: str) -> go.Figure:
    """Single-metric line chart (used 4x for ROE, P/B, leverage, vol)."""
    fig = go.Figure()
    if metrics.empty or col not in metrics.columns:
        return fig
    fig.add_trace(
        go.Scatter(
            x=metrics["calc_date"],
            y=metrics[col],
            mode="lines",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor=_rgba(COLORS["primary"], 0.1),
            hovertemplate=f"<b>{title}</b><br>" "%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=240,
        yaxis_title=title,
        margin=dict(t=20, l=50, r=20, b=40),
        showlegend=False,
    )
    return fig


def selection_status_over_time(status_history: pd.DataFrame) -> go.Figure:
    """Stacked area showing how many stocks were in each status per month.

    Status values: top_10, long_buffer, bottom_10, short_buffer.
    """
    fig = go.Figure()
    if status_history.empty:
        return fig

    counts = (
        status_history.groupby(["rebalance_date", "status"])
        .size()
        .unstack(fill_value=0)
    )
    counts.index = pd.to_datetime(counts.index)

    status_colors = {
        "long_core": COLORS["long"],
        "long_buffer": _rgba(COLORS["long"], 0.5),
        "short_core": COLORS["short"],
        "short_buffer": _rgba(COLORS["short"], 0.5),
    }
    status_labels = {
        "long_core": "Long top 10%",
        "long_buffer": "Long buffer (10-20%)",
        "short_core": "Short bottom 10%",
        "short_buffer": "Short buffer (80-90%)",
    }

    for status_key in ["long_core", "long_buffer", "short_core", "short_buffer"]:
        if status_key in counts.columns:
            fig.add_trace(
                go.Scatter(
                    x=counts.index,
                    y=counts[status_key],
                    name=status_labels[status_key],
                    mode="lines",
                    line=dict(width=0.5, color=status_colors[status_key]),
                    fillcolor=status_colors[status_key],
                    stackgroup="one",
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "%{x|%b %Y}<br>%{y} stocks<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=380,
        yaxis_title="Number of stocks",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=30, b=80),
    )
    return fig


def sector_exposure_heatmap(positions_history: pd.DataFrame) -> go.Figure:
    """Sector x month heatmap of net exposure (long - short)."""
    fig = go.Figure()
    if positions_history.empty:
        return fig

    df = positions_history.copy()
    df["signed_weight"] = df.apply(
        lambda r: (
            r["final_weight"] if r["direction"] == "long" else -abs(r["final_weight"])
        ),
        axis=1,
    )
    pivot = df.pivot_table(
        index="sector",
        columns="rebalance_date",
        values="signed_weight",
        aggfunc="sum",
        fill_value=0,
    )
    pivot.columns = pd.to_datetime(pivot.columns)
    pivot = pivot * 100  # to %

    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0, COLORS["short"]],
                [0.5, COLORS["surface"]],
                [1, COLORS["long"]],
            ],
            zmid=0,
            colorbar=dict(
                title=dict(text="Net %", font=dict(color=COLORS["text"])),
                tickfont=dict(color=COLORS["text_secondary"]),
                outlinewidth=0,
            ),
            hovertemplate="<b>%{y}</b><br>%{x|%b %Y}<br>Net: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=400,
        yaxis_title="GICS sector",
        xaxis_title="",
        hovermode="closest",
        xaxis=dict(showspikes=False),
        yaxis=dict(showspikes=False),
        margin=dict(t=20, l=140, r=20, b=50),
    )
    return fig


def scenario_comparison_bars(
    summaries: pd.DataFrame, metric: str = "sharpe_ratio", title: str = ""
) -> go.Figure:
    """Bar chart comparing one metric across scenarios."""
    fig = go.Figure()
    if summaries.empty:
        return fig

    summaries = summaries.sort_values(metric, ascending=True)
    colors = [
        COLORS["primary"] if sid == "baseline" else COLORS["surface_alt"]
        for sid in summaries["scenario_id"]
    ]

    fig.add_trace(
        go.Bar(
            x=summaries[metric],
            y=summaries["scenario_id"],
            orientation="h",
            marker_color=colors,
            marker_line_color=COLORS["border"],
            marker_line_width=1,
            hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(400, 24 * len(summaries)),
        title=title,
        xaxis_title=metric.replace("_", " ").title(),
        showlegend=False,
    )
    return fig
