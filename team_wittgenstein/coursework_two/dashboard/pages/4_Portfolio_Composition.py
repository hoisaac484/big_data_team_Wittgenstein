"""Portfolio Composition.

The strategy mechanics in motion. For any rebalance date you can see:
  - How many stocks are in each selection status (top 10%, buffer, etc.)
  - The actual long and short holdings with their weights
  - Sector breakdown including a sector-neutrality check
  - Constraint health (long/short sums, liquidity caps, no-trade zone)
  - How composition has evolved over time

Reads from baseline tables only. Variant scenarios use in-memory
pipelines and don't write to portfolio_positions / selection_status.
"""

import pandas as pd
import streamlit as st
from lib import charts as ch
from lib import queries as q
from lib.components import kpi_card, page_setup, section_header
from lib.format import pct
from lib.theme import COLORS, install_template

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

page_setup("Portfolio Composition", icon="💼")
install_template()

st.title("Portfolio Composition")
st.caption(
    "See the actual strategy decisions for any rebalance date - which "
    "stocks were held long, which were shorted, sector exposures, and "
    "constraint compliance. **Baseline scenario only**: variant scenarios "
    "(cost / factor exclusion / parameter sensitivity) compute positions in "
    "memory and don't persist them."
)


# ---------------------------------------------------------------------------
# Date picker
# ---------------------------------------------------------------------------

dates = q.get_rebalance_dates()
if not dates:
    st.warning("No rebalance dates found. Has the pipeline been run?")
    st.stop()

# Default to the most recent date
default_idx = len(dates) - 1
selected_date = st.select_slider(
    "Rebalance date",
    options=dates,
    value=dates[default_idx],
    format_func=lambda d: d.strftime("%b %Y"),
    help="Drag to step through any of the monthly rebalances.",
)


# ---------------------------------------------------------------------------
# Fetch data for the selected date
# ---------------------------------------------------------------------------

holdings = q.get_holdings(selected_date)
selection = q.get_selection_status(selected_date)


# ---------------------------------------------------------------------------
# Status breakdown - long & short side by side
# ---------------------------------------------------------------------------

section_header(
    f"Selection breakdown - {selected_date.strftime('%B %Y')}",
    "How many stocks were in each selection state, with new entries and "
    "exits this rebalance.",
)


def _count_status(value: str) -> int:
    return int((selection["status"] == value).sum()) if not selection.empty else 0


def _count_new_entries(direction: str) -> int:
    """Stocks newly in core this rebalance."""
    if selection.empty:
        return 0
    core_label = f"{direction}_core"
    in_core = selection[selection["status"] == core_label]
    if in_core.empty:
        return 0
    # 'New' = entry_date == this rebalance_date
    return int(
        (pd.to_datetime(in_core["entry_date"]).dt.date == selected_date.date()).sum()
    )


def _count_exits(direction: str) -> int:
    """Stocks that exited this rebalance (have an exit_reason populated)."""
    if selection.empty:
        return 0
    return int(
        (
            (selection["status"].str.startswith(direction))
            & selection["exit_reason"].notna()
        ).sum()
    )


long_col, short_col = st.columns(2, gap="medium")

with long_col:
    st.markdown(
        f'<div style="text-align:center;font-weight:600;font-size:1.05rem;'
        f'color:{COLORS["long"]};margin-bottom:0.75rem;">LONG SIDE (130%)</div>',
        unsafe_allow_html=True,
    )
    a, b, c, d = st.columns(4, gap="medium")
    with a:
        kpi_card("Top 10%", str(_count_status("long_core")), sub="Active longs")
    with b:
        kpi_card("Buffer", str(_count_status("long_buffer")), sub="11-20% rank")
    with c:
        kpi_card("New entries", str(_count_new_entries("long")), sub="This month")
    with d:
        kpi_card("Exits", str(_count_exits("long")), sub="This month")

with short_col:
    st.markdown(
        f'<div style="text-align:center;font-weight:600;font-size:1.05rem;'
        f'color:{COLORS["short"]};margin-bottom:0.75rem;">SHORT SIDE (30%)</div>',
        unsafe_allow_html=True,
    )
    a, b, c, d = st.columns(4, gap="medium")
    with a:
        kpi_card("Bottom 10%", str(_count_status("short_core")), sub="Active shorts")
    with b:
        kpi_card("Buffer", str(_count_status("short_buffer")), sub="80-90% rank")
    with c:
        kpi_card("New entries", str(_count_new_entries("short")), sub="This month")
    with d:
        kpi_card("Exits", str(_count_exits("short")), sub="This month")


# ---------------------------------------------------------------------------
# Holdings tables - long and short side by side
# ---------------------------------------------------------------------------

section_header(
    "Holdings",
    "Sortable by any column. Click a row, then click the button "
    "below to deep-dive on that stock.",
)
st.caption(
    "**Capped column** = TRUE if the strategy wanted a bigger weight on this "
    "stock but had to cap it at 5% of the stock's 20-day average daily volume "
    "to avoid market impact. With $50M AUM trading large-cap US stocks the "
    "cap almost never binds (only 2 caps across the entire 5-year backtest), "
    "so expect this column to be all FALSE in most months."
)

if holdings.empty:
    st.info("No holdings for this rebalance date.")
else:
    # Build a display DataFrame
    holdings_display = holdings.copy()
    holdings_display["weight_%"] = (holdings_display["final_weight"] * 100).round(2)
    holdings_display["risk_adj_score"] = holdings_display["risk_adj_score"].round(3)
    holdings_display["ewma_vol"] = (holdings_display["ewma_vol"] * 100).round(2)

    longs = holdings_display[holdings_display["direction"] == "long"][
        [
            "symbol",
            "sector",
            "weight_%",
            "risk_adj_score",
            "ewma_vol",
            "liquidity_capped",
        ]
    ].rename(
        columns={
            "symbol": "Symbol",
            "sector": "Sector",
            "weight_%": "Weight (%)",
            "risk_adj_score": "Risk-adj score",
            "ewma_vol": "EWMA vol (%)",
            "liquidity_capped": "Capped",
        }
    )
    shorts = holdings_display[holdings_display["direction"] == "short"][
        [
            "symbol",
            "sector",
            "weight_%",
            "risk_adj_score",
            "ewma_vol",
            "liquidity_capped",
        ]
    ].rename(
        columns={
            "symbol": "Symbol",
            "sector": "Sector",
            "weight_%": "Weight (%)",
            "risk_adj_score": "Risk-adj score",
            "ewma_vol": "EWMA vol (%)",
            "liquidity_capped": "Capped",
        }
    )

    long_t, short_t = st.columns(2, gap="medium")
    with long_t:
        st.markdown(
            f'<div style="color:{COLORS["long"]};font-weight:600;'
            f'margin-bottom:0.5rem;">Long holdings ({len(longs)} stocks)</div>',
            unsafe_allow_html=True,
        )
        long_event = st.dataframe(
            longs,
            width="stretch",
            hide_index=True,
            height=420,
            on_select="rerun",
            selection_mode="single-row",
            key="longs_table",
        )

    with short_t:
        st.markdown(
            f'<div style="color:{COLORS["short"]};font-weight:600;'
            f'margin-bottom:0.5rem;">Short holdings ({len(shorts)} stocks)</div>',
            unsafe_allow_html=True,
        )
        short_event = st.dataframe(
            shorts,
            width="stretch",
            hide_index=True,
            height=420,
            on_select="rerun",
            selection_mode="single-row",
            key="shorts_table",
        )

    # If a row was selected in either table, deep-dive on that stock.
    selected_symbol = None
    if long_event and long_event.selection.rows:
        idx = long_event.selection.rows[0]
        selected_symbol = longs.iloc[idx]["Symbol"]
    elif short_event and short_event.selection.rows:
        idx = short_event.selection.rows[0]
        selected_symbol = shorts.iloc[idx]["Symbol"]

    if selected_symbol:
        st.success(
            f"Selected **{selected_symbol}**. Click the button to open its "
            "Stock Deep-Dive page."
        )
        if st.button(
            f"Open Stock Deep-Dive for {selected_symbol}",
            type="primary",
            width="stretch",
        ):
            # Use a separate "pending" key for the handoff. The Deep-Dive
            # page reads this BEFORE the selectbox renders and uses it as
            # the initial index. Avoids conflict with the widget's own
            # state key.
            st.session_state["_pending_deep_dive_symbol"] = selected_symbol
            st.switch_page("pages/5_Stock_Deep_Dive.py")


# ---------------------------------------------------------------------------
# Sector breakdown - 3 charts
# ---------------------------------------------------------------------------

section_header(
    "Sector breakdown",
    "By design every sector gets ~11.82% long and ~2.73% short, so weight "
    "bars would all be identical. Instead we show **stock count per sector** "
    "(reveals concentration) plus a **net exposure check** (proves neutrality "
    "at this rebalance).",
)

if not holdings.empty:
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        st.markdown("**Long stocks per sector**")
        st.caption(
            "How many stocks make up each sector's ~11.82% allocation. "
            "Sectors with fewer stocks are more concentrated."
        )
        st.plotly_chart(
            ch.sector_stock_count_bars(holdings, "long"),
            config=ch.chart_config("long_sector_count"),
            width="stretch",
        )
    with s2:
        st.markdown("**Short stocks per sector**")
        st.caption("How many stocks make up each sector's ~2.73% short allocation.")
        st.plotly_chart(
            ch.sector_stock_count_bars(holdings, "short"),
            config=ch.chart_config("short_sector_count"),
            width="stretch",
        )
    with s3:
        st.markdown("**Net sector exposure (long − short)**")
        st.caption("Should hover near zero - that's the sector-neutrality check.")
        st.plotly_chart(
            ch.net_sector_exposure(holdings),
            config=ch.chart_config("net_sector"),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Constraint health
# ---------------------------------------------------------------------------

section_header("Constraint health check")

if not holdings.empty:
    long_sum = float(holdings[holdings["direction"] == "long"]["final_weight"].sum())
    short_sum = float(holdings[holdings["direction"] == "short"]["final_weight"].sum())
    capped = int(holdings["liquidity_capped"].sum())
    actions = holdings["trade_action"].value_counts()

    cc1, cc2, cc3, cc4 = st.columns(4, gap="medium")
    with cc1:
        long_ok = abs(long_sum - 1.30) <= 0.02
        kpi_card(
            "Long sum",
            pct(long_sum),
            delta=("Within tolerance" if long_ok else "Outside ±2%"),
            delta_positive=long_ok,
            sub="Target 130%",
        )
    with cc2:
        short_ok = abs(short_sum - 0.30) <= 0.02
        kpi_card(
            "Short sum",
            pct(short_sum),
            delta=("Within tolerance" if short_ok else "Outside ±2%"),
            delta_positive=short_ok,
            sub="Target 30%",
        )
    with cc3:
        kpi_card(
            "Liquidity-capped",
            str(capped),
            sub="Position size > 5% ADTV",
        )
    with cc4:
        traded = int(actions.get("trade", 0))
        held = int(actions.get("hold", 0))
        kpi_card(
            "Trade activity",
            f"{traded} traded",
            sub=f"{held} held (no-trade zone)",
        )


# ---------------------------------------------------------------------------
# Composition over time
# ---------------------------------------------------------------------------

section_header(
    "Composition over time",
    "How the count of stocks in each status has evolved across all rebalances.",
)

status_history = q.get_selection_status_history()
if not status_history.empty:
    status_history = status_history.copy()
    status_history["rebalance_date"] = pd.to_datetime(status_history["rebalance_date"])
    st.plotly_chart(
        ch.selection_status_over_time(status_history),
        config=ch.chart_config("selection_over_time"),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Sector exposure heatmap
# ---------------------------------------------------------------------------

section_header(
    "Sector exposure heatmap",
    "Each cell = net (long - short) exposure for that sector in that month. "
    "Bright cells = drift away from sector neutrality. Most cells should be "
    "near zero (light).",
)

all_positions = q.get_all_positions()
if not all_positions.empty:
    st.plotly_chart(
        ch.sector_exposure_heatmap(all_positions),
        config=ch.chart_config("sector_heatmap"),
        width="stretch",
    )

st.caption(
    "Tip: any chart can be expanded to fullscreen via the diagonal-arrow "
    "icon (top-right). Press **Esc** or click the **X** to exit fullscreen."
)
