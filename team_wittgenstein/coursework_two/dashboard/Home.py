"""Team Wittgenstein - 130/30 Multi-Factor Strategy Dashboard.

Home / overview page. Describes the strategy at a high level and shows
system health. Performance metrics and equity curves live on the
Performance page.

Run with:
    poetry run streamlit run dashboard/Home.py
"""

import pandas as pd
import streamlit as st
from lib import queries as q
from lib.components import (
    db_status_badge,
    hero_header,
    kpi_card,
    page_setup,
    section_header,
)
from lib.db import health_check
from lib.format import big_num, fmt_date_range, pct
from lib.theme import COLORS, install_template

# ---------------------------------------------------------------------------
# Page setup (must be first Streamlit call)
# ---------------------------------------------------------------------------

page_setup("Home", icon="🏠")
install_template()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

connected = health_check()
st.markdown(db_status_badge(connected), unsafe_allow_html=True)

if not connected:
    st.error(
        "Cannot reach the database. Make sure Postgres is running:\n\n"
        "```bash\ndocker compose up -d postgres_db\n```"
    )
    st.stop()

stats = q.get_database_stats()
date_range = fmt_date_range(stats.get("start_date"), stats.get("end_date"))

hero_header(
    "Team Wittgenstein - 130/30 Multi-Factor Strategy",
    f"{date_range} · {stats.get('stocks_used', 0)} stocks in pre-filter universe · "
    f"{stats.get('months', 0)} monthly rebalances · "
    f"{stats.get('scenarios', 0)} backtested scenarios",
)


# ---------------------------------------------------------------------------
# Strategy pipeline diagram
# ---------------------------------------------------------------------------

section_header(
    "Strategy pipeline",
    "Each rebalance month, the strategy runs every stock through the same "
    "pipeline. Performance metrics and the equity curve are on the **Performance** page.",
)


# Render an HTML/CSS box-and-arrow flow diagram. Cleaner than graphviz here.
def _stage(label: str, sub: str, accent: str) -> str:
    return (
        f'<div style="background:{COLORS["surface"]};'
        f"border:1px solid {accent};border-left:3px solid {accent};"
        f"border-radius:8px;padding:0.85rem 1rem;flex:1;min-width:140px;"
        f'text-align:center;">'
        f'<div style="font-weight:600;font-size:0.95rem;'
        f'color:{COLORS["text"]};margin-bottom:0.25rem;">{label}</div>'
        f'<div style="font-size:0.78rem;color:{COLORS["text_muted"]};'
        f'line-height:1.4;">{sub}</div></div>'
    )


arrow = (
    f'<div style="display:flex;align-items:center;color:{COLORS["text_muted"]};'
    f'font-size:1.5rem;padding:0 0.25rem;">→</div>'
)

stages = [
    ("Universe", "~430 US stocks<br>11 GICS sectors", COLORS["primary"]),
    ("Liquidity Filter", "ADV ≥ $1M<br>Drop top 10% Amihud ILLIQ", COLORS["secondary"]),
    ("Factor Scoring", "Value, Quality<br>Momentum, Low Vol", COLORS["long"]),
    ("IC Composite", "Rolling 36m IC<br>z-floored weights", "#a855f7"),
    ("Selection", "Top/Bottom 10%<br>per sector + buffer", COLORS["warning"]),
    ("Risk-adj Weights", "EWMA σ (λ=0.94)<br>composite ÷ vol", COLORS["secondary"]),
    ("Liquidity Cap", "Cap at 5%<br>of 20-day ADTV", COLORS["primary"]),
    ("No-Trade Zone", "1% threshold<br>holds frozen", COLORS["long"]),
    ("130/30 Sizing", "130% long<br>30% short", COLORS["short"]),
]

flow_html = (
    '<div style="display:flex;align-items:stretch;flex-wrap:nowrap;'
    'overflow-x:auto;gap:0;padding:0.5rem 0 1rem 0;">'
)
for i, (label, sub, color) in enumerate(stages):
    flow_html += _stage(label, sub, color)
    if i < len(stages) - 1:
        flow_html += arrow
flow_html += "</div>"
st.markdown(flow_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# System health indicators
# ---------------------------------------------------------------------------

section_header(
    "System health",
    "Live checks against the latest rebalance.",
)

active_factors = q.get_active_factor_count()
universe_size = q.get_universe_size_latest()
exposure = q.get_latest_net_exposure()
last_rebalance = stats.get("end_date")

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    long_w = exposure["long"]
    short_w = abs(exposure["short"])
    long_ok = abs(long_w - 1.30) <= 0.02
    short_ok = abs(short_w - 0.30) <= 0.02
    both_ok = long_ok and short_ok
    kpi_card(
        "Net exposure",
        pct(exposure["net"]),
        delta=("Constraints OK" if both_ok else "Outside ±2%"),
        delta_positive=both_ok,
        sub=f"Long {pct(long_w)} | Short {pct(short_w)}",
    )

with c2:
    kpi_card(
        "Active factors",
        f"{active_factors} / 4",
        delta=(
            "All factors active"
            if active_factors == 4
            else f"{4 - active_factors} muted"
        ),
        delta_positive=active_factors == 4,
        sub="Avg IC weight > 0.001",
    )

with c3:
    kpi_card(
        "Last rebalance",
        last_rebalance.strftime("%b %Y") if last_rebalance else "-",
        sub="Most recent month in DB",
    )

with c4:
    kpi_card(
        "Universe size",
        str(universe_size),
        sub="Stocks held at latest rebalance",
    )


# ---------------------------------------------------------------------------
# Strategy parameters table
# ---------------------------------------------------------------------------

section_header(
    "Strategy parameters",
    "Fixed inputs for the baseline scenario. The **Strategy Tuner** page "
    "lets you vary these one at a time.",
)

params = pd.DataFrame(
    [
        ("Selection threshold", "10%", "Top/bottom 10% per sector enter long/short"),
        ("Buffer exit threshold", "20%", "Hold zone before forcing exit"),
        ("Buffer max months", "3 months", "How long a stock can sit in the buffer"),
        ("IC lookback window", "36 months", "Rolling window for IC weights"),
        ("EWMA lambda", "0.94", "RiskMetrics-standard volatility decay"),
        ("Liquidity cap", "5% ADTV", "Max position vs stock's daily volume"),
        ("No-trade threshold", "1%", "Min weight change to trigger a trade"),
        ("Transaction cost", "25 bps", "One-way per trade (moderate scenario)"),
        ("Short borrow cost", "0.75% / year", "Annual fee on short notional"),
        ("Long target", "130%", "Sum of long position weights"),
        ("Short target", "30%", "Sum of |short position weights|"),
        ("Benchmark", "MSCI USA Index", "Tracked via EUSA ETF returns"),
        ("AUM", "$50M", "Used for liquidity cap dollar sizing"),
    ],
    columns=["Parameter", "Value", "Description"],
)
st.dataframe(params, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Strategy at a glance + scenarios summary
# ---------------------------------------------------------------------------

left, right = st.columns([1.4, 1], gap="medium")

with left:
    section_header("Strategy at a glance")
    st.markdown(f"""
        - **130% long / 30% short** with 100% net market exposure
        - **Sector neutral** across all 11 GICS sectors
        - **Monthly rebalance** with no-trade and buffer zones for stability
        - **4 factors** combined via IC-weighting (negative ICs floored to zero)
        - **Risk-adjusted** position sizing (composite ÷ EWMA volatility)
        - **Liquidity aware** at both universe and position level
        - **Walk-forward backtest** with point-in-time data, no look-ahead
        - **Pre-filter universe:** {big_num(stats.get('stocks_used', 0))} US-listed stocks (before liquidity filter)
        - **Backtested:** {date_range}
        """)

with right:
    section_header("Scenarios run")
    st.markdown(f"""
        | Group | Count | Examples |
        |-------|-------|----------|
        | Baseline | 1 | `baseline` |
        | Cost sensitivity | 3 | `cost_low`, `cost_high` |
        | Factor exclusion | 4 | `excl_quality` |
        | Parameter sensitivity | 15 | `sens_ic_24`, `sens_ewma_0.97` |
        | **Total** | **{stats.get('scenarios', 0)}** | |
        """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Big Data in Quantitative Finance - CW2 - "
    "Built by Team Wittgenstein. Use the sidebar to navigate to detailed pages: "
    "Performance, Compare Scenarios, Strategy Tuner, Portfolio Composition, "
    "Stock Deep-Dive, Factor Analysis."
)
