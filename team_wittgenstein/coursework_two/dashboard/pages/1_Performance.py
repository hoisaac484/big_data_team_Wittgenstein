"""Page 2 - Performance.

Detailed performance analysis for a single chosen scenario.

Layout:
  - Sidebar: scenario picker + date range filter
  - Headline KPIs (Sharpe, Sortino, Calmar, IR, Alpha, Vol, Max DD, TE)
  - Equity curve + drawdown stacked
  - Monthly returns histogram + monthly excess + long/short contribution
  - Rolling 12-month Sharpe
"""

import streamlit as st
from lib import charts as ch
from lib import queries as q
from lib.components import kpi_card, page_setup, section_header
from lib.format import num, pct, pct_signed, safe_get, scenario_label
from lib.theme import install_template

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

page_setup("Performance", icon="📈")
install_template()

st.title("Performance")
st.caption(
    "Deep-dive into the metrics and returns for any of the 23 backtested scenarios. "
    "Use the sidebar to switch scenario or filter to a date range."
)


# ---------------------------------------------------------------------------
# Sidebar - scenario picker + date range
# ---------------------------------------------------------------------------

scenarios = q.get_scenario_list()

st.sidebar.markdown("### Scenario")
selected = st.sidebar.selectbox(
    "Select scenario",
    options=scenarios,
    index=scenarios.index("baseline") if "baseline" in scenarios else 0,
    format_func=scenario_label,
    label_visibility="collapsed",
)

returns = q.get_returns(selected)
summary = q.get_summary(selected)

if returns.empty or summary.empty:
    st.warning(f"No data found for scenario '{selected}'.")
    st.stop()

min_date = returns["rebalance_date"].min().date()
max_date = returns["rebalance_date"].max().date()

st.sidebar.markdown("### Date range")
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="MMM YYYY",
    label_visibility="collapsed",
)

returns_filtered = returns[
    (returns["rebalance_date"].dt.date >= date_range[0])
    & (returns["rebalance_date"].dt.date <= date_range[1])
].copy()


# ---------------------------------------------------------------------------
# Headline KPIs (8 cards in 2 rows)
# ---------------------------------------------------------------------------

section_header(
    f"Headline metrics — {scenario_label(selected)}",
    f"{date_range[0].strftime('%b %Y')} - {date_range[1].strftime('%b %Y')} "
    f"({len(returns_filtered)} months)",
)

row1 = st.columns(4)

with row1[0]:
    s = safe_get(summary, "sharpe_ratio")
    b = safe_get(summary, "benchmark_sharpe")
    kpi_card(
        "Sharpe Ratio",
        num(s, 2),
        delta=f"{(s - b):+.2f} vs benchmark",
        delta_positive=s > b,
        sub=f"Benchmark {num(b, 2)}",
    )

with row1[1]:
    s = safe_get(summary, "sortino_ratio")
    b = safe_get(summary, "benchmark_sortino")
    kpi_card(
        "Sortino Ratio",
        num(s, 2),
        delta=f"{(s - b):+.2f} vs benchmark",
        delta_positive=s > b,
        sub=f"Benchmark {num(b, 2)}",
    )

with row1[2]:
    s = safe_get(summary, "calmar_ratio")
    b = safe_get(summary, "benchmark_calmar")
    kpi_card(
        "Calmar Ratio",
        num(s, 2),
        delta=f"{(s - b):+.2f} vs benchmark",
        delta_positive=s > b,
        sub=f"Benchmark {num(b, 2)}",
    )

with row1[3]:
    ir = safe_get(summary, "information_ratio")
    te = safe_get(summary, "tracking_error")
    kpi_card(
        "Information Ratio",
        num(ir, 2),
        delta_positive=ir > 0,
        delta=f"Tracking error {pct(te)}",
    )

row2 = st.columns(4)

with row2[0]:
    a = safe_get(summary, "alpha")
    sret = safe_get(summary, "annualised_return")
    bret = safe_get(summary, "benchmark_return_ann")
    kpi_card(
        "Alpha (annualised)",
        pct_signed(a),
        delta=f"Strategy {pct(sret)}",
        delta_positive=a > 0,
        sub=f"Benchmark {pct(bret)}",
    )

with row2[1]:
    v = safe_get(summary, "annualised_volatility")
    bv = safe_get(summary, "benchmark_volatility")
    kpi_card(
        "Volatility (ann.)",
        pct(v),
        delta=f"{pct_signed(v - bv)} vs benchmark",
        delta_positive=v < bv,
        sub=f"Benchmark {pct(bv)}",
    )

with row2[2]:
    dd = safe_get(summary, "max_drawdown")
    bdd = safe_get(summary, "benchmark_max_drawdown")
    kpi_card(
        "Max Drawdown",
        pct(dd),
        delta=f"{pct_signed(dd - bdd)} vs benchmark",
        delta_positive=dd > bdd,
        sub=f"Benchmark {pct(bdd)}",
    )

with row2[3]:
    cum = safe_get(summary, "cumulative_return")
    bcum = safe_get(summary, "benchmark_return_cum")
    kpi_card(
        "Total Return",
        pct(cum),
        delta=f"vs benchmark {pct(bcum)}",
        delta_positive=cum > bcum,
        sub=f"Over {len(returns)} months",
    )


# ---------------------------------------------------------------------------
# Equity curve + Drawdown
# ---------------------------------------------------------------------------

section_header("Equity curve and drawdown")

eq_col, dd_col = st.columns([1, 1])
with eq_col:
    st.plotly_chart(
        ch.equity_curve(returns_filtered),
        config=ch.chart_config(f"{selected}_equity"),
        width="stretch",
    )
with dd_col:
    st.plotly_chart(
        ch.drawdown(returns_filtered),
        config=ch.chart_config(f"{selected}_drawdown"),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Monthly returns analysis - 3 charts in a row
# ---------------------------------------------------------------------------

section_header("Monthly return analysis")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Distribution of monthly returns**")
    st.caption(
        "Buckets monthly returns by value range. Tall bars = many "
        "months in that range. Cannot identify individual months."
    )
    st.plotly_chart(
        ch.returns_histogram(returns_filtered),
        config=ch.chart_config(f"{selected}_histogram"),
        width="stretch",
    )
with c2:
    st.markdown("**Monthly excess return vs benchmark**")
    st.caption(
        "Each bar is one month. Green = beat benchmark, red = "
        "underperformed. Hover to see the date."
    )
    st.plotly_chart(
        ch.monthly_excess(returns_filtered),
        config=ch.chart_config(f"{selected}_excess"),
        width="stretch",
    )
with c3:
    st.markdown("**Long vs Short cumulative contribution**")
    st.caption(
        "Cumulative return from long positions and short positions, "
        "tracked separately."
    )
    st.plotly_chart(
        ch.long_short_contribution(returns_filtered),
        config=ch.chart_config(f"{selected}_contribution"),
        width="stretch",
    )

st.caption(
    "Tip: any chart can be expanded to fullscreen via the diagonal-arrow "
    "icon (top-right). Press **Esc** or click the **X** to exit fullscreen "
    "- do not use the browser back button."
)


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------

section_header(
    "Rolling 12-month Sharpe Ratio",
    "Annualised Sharpe over a 12-month rolling window. Shows how performance "
    "stability evolved over time.",
)

st.plotly_chart(
    ch.rolling_sharpe(returns_filtered, window=12),
    config=ch.chart_config(f"{selected}_rolling_sharpe"),
    width="stretch",
)


# ---------------------------------------------------------------------------
# Monthly turnover
# ---------------------------------------------------------------------------

section_header(
    "Monthly turnover",
    "Sum of absolute weight changes each rebalance. Lower turnover = "
    "lower transaction costs. The dashed line shows the period average.",
)

st.plotly_chart(
    ch.monthly_turnover(returns_filtered),
    config=ch.chart_config(f"{selected}_turnover"),
    width="stretch",
)
