"""Stock Deep-Dive.

Pick any stock in the universe and explore its history:
  - Daily price chart with rebalance markers (long / short held)
  - 4 factor z-scores over time
  - 4 raw fundamental metrics over time
  - Composite score history
  - Selection history table
"""

import pandas as pd
import streamlit as st
from lib import charts as ch
from lib import queries as q
from lib.components import kpi_card, page_setup, section_header
from lib.format import num
from lib.theme import install_template

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

page_setup("Stock Deep-Dive", icon="🔍")
install_template()

st.title("Stock Deep-Dive")
st.caption(
    "Search any stock in the universe to inspect its factor history, "
    "fundamentals, and the times the strategy held it long or short."
)


# ---------------------------------------------------------------------------
# Ticker search
#
# Honours an inbound `?symbol=AAPL` query parameter (set by the
# Portfolio Composition page when a row is clicked) so the search box
# is pre-filled.
# ---------------------------------------------------------------------------

universe = q.get_symbols()
if universe.empty:
    st.warning("No symbols found in the universe.")
    st.stop()

symbols = universe["symbol"].tolist()

# Determine the symbol to display.
# Priority:
#   1. _pending_deep_dive_symbol (one-shot handoff from another page's button)
#   2. _last_user_choice (the user's previous selection on this page)
#   3. ?symbol=XYZ URL param (for shared links / fresh tabs)
#   4. First symbol alphabetically
if "_pending_deep_dive_symbol" in st.session_state:
    target = str(st.session_state.pop("_pending_deep_dive_symbol"))
elif "_dd_last_choice" in st.session_state:
    target = st.session_state["_dd_last_choice"]
else:
    target = st.query_params.get("symbol") or symbols[0]

if target not in symbols:
    target = symbols[0]

# IMPORTANT: NO `key=` on the selectbox. Using key= makes Streamlit auto-
# manage widget state, which can ignore programmatic updates from another
# page's button click. We track the user's choice manually via _dd_last_choice
# below, which is reliable across page navigations.
selected = st.selectbox(
    "Stock symbol",
    options=symbols,
    index=symbols.index(target),
    format_func=lambda s: (
        f"{s} - {universe[universe['symbol'] == s].iloc[0]['security']}"
        if len(universe[universe["symbol"] == s]) > 0
        else s
    ),
)

# Persist user's selection for next render and keep URL in sync
st.session_state["_dd_last_choice"] = selected
if st.query_params.get("symbol") != selected:
    st.query_params["symbol"] = selected

company = universe[universe["symbol"] == selected].iloc[0]


# ---------------------------------------------------------------------------
# Company card
# ---------------------------------------------------------------------------

position_history = q.get_position_history(selected)
factor_scores = q.get_factor_scores(selected)
factor_metrics = q.get_factor_metrics(selected)
prices = q.get_prices(selected)


# Has the strategy ever held this stock?
held_count = len(position_history)
held_long = int((position_history["direction"] == "long").sum()) if held_count else 0
held_short = int((position_history["direction"] == "short").sum()) if held_count else 0

# Latest scores - skip rows where composite hasn't been computed yet
# (the most recent month is expected to be NULL until the pipeline runs)
if not factor_scores.empty:
    valid = factor_scores.dropna(subset=["composite_score"])
    latest_score = valid.iloc[-1] if not valid.empty else None
else:
    latest_score = None

c1, c2, c3, c4 = st.columns(4, gap="medium")
with c1:
    kpi_card("Symbol", selected, sub=str(company["security"]))
with c2:
    kpi_card(
        "Sector",
        str(company["gics_sector"]) or "-",
        sub=str(company["gics_industry"]) or "",
    )
with c3:
    if latest_score is not None:
        kpi_card(
            "Latest composite",
            num(float(latest_score["composite_score"]), 2),
            sub=f"as of {pd.to_datetime(latest_score['score_date']).strftime('%b %Y')}",
        )
    else:
        kpi_card("Latest composite", "-", sub="No scores")
with c4:
    if held_count:
        kpi_card(
            "Times held",
            str(held_count),
            sub=f"{held_long} long / {held_short} short",
        )
    else:
        kpi_card("Times held", "0", sub="Never selected")


# ---------------------------------------------------------------------------
# Price chart with markers
# ---------------------------------------------------------------------------

section_header(
    "Price history with rebalance markers",
    "Green dots: months held long. Red dots: months held short. "
    "Hover for the held weight.",
)

if prices.empty:
    st.info("No price data for this symbol.")
else:
    st.plotly_chart(
        ch.stock_price_with_markers(prices, position_history),
        config=ch.chart_config(f"{selected}_price"),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Factor z-scores
# ---------------------------------------------------------------------------

section_header(
    "Factor z-scores over time",
    "Sector-relative z-scores per factor. Dashed lines mark ±1 σ. "
    "Above +1 = strong on that factor, below -1 = weak.",
)

if factor_scores.empty:
    st.info("No factor scores for this symbol.")
else:
    st.plotly_chart(
        ch.stock_factor_zscores(factor_scores),
        config=ch.chart_config(f"{selected}_zscores"),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Raw fundamentals - 4 charts in a row
# ---------------------------------------------------------------------------

section_header("Raw fundamental metrics")

if factor_metrics.empty:
    st.info("No fundamental data for this symbol.")
else:
    fc = st.columns(4, gap="medium")
    with fc[0]:
        st.markdown("**ROE**")
        st.plotly_chart(
            ch.stock_fundamental_line(factor_metrics, "roe", "ROE"),
            config=ch.chart_config(f"{selected}_roe"),
            width="stretch",
        )
    with fc[1]:
        st.markdown("**P/B ratio**")
        st.plotly_chart(
            ch.stock_fundamental_line(factor_metrics, "pb_ratio", "P/B"),
            config=ch.chart_config(f"{selected}_pb"),
            width="stretch",
        )
    with fc[2]:
        st.markdown("**Leverage (D/E)**")
        st.plotly_chart(
            ch.stock_fundamental_line(factor_metrics, "leverage", "D/E"),
            config=ch.chart_config(f"{selected}_lev"),
            width="stretch",
        )
    with fc[3]:
        st.markdown("**12-month volatility**")
        st.plotly_chart(
            ch.stock_fundamental_line(
                factor_metrics, "volatility_12m", "Vol (annualised)"
            ),
            config=ch.chart_config(f"{selected}_vol"),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Selection / position history table
# ---------------------------------------------------------------------------

section_header("Position history", "All months when the strategy held this stock.")

if position_history.empty:
    st.info("This stock was never selected by the strategy.")
else:
    history = position_history.copy()
    history["weight_%"] = (history["final_weight"] * 100).round(2)
    history["target_%"] = (history["target_weight"] * 100).round(2)
    history["risk_adj_score"] = history["risk_adj_score"].round(3)
    history["ewma_vol_%"] = (history["ewma_vol"] * 100).round(2)
    history["rebalance_date"] = history["rebalance_date"].dt.strftime("%b %Y")

    display = history[
        [
            "rebalance_date",
            "direction",
            "weight_%",
            "target_%",
            "risk_adj_score",
            "ewma_vol_%",
            "trade_action",
            "liquidity_capped",
        ]
    ].rename(
        columns={
            "rebalance_date": "Date",
            "direction": "Direction",
            "weight_%": "Weight (%)",
            "target_%": "Target (%)",
            "risk_adj_score": "Risk-adj",
            "ewma_vol_%": "EWMA vol (%)",
            "trade_action": "Action",
            "liquidity_capped": "Capped",
        }
    )

    st.dataframe(display, width="stretch", hide_index=True)
