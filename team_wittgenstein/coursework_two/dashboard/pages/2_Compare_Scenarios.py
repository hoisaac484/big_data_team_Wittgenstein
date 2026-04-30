"""Compare Scenarios (Option B).

Side-by-side comparison of any two of the 23 backtested scenarios.
Two columns: Scenario A on the left, Scenario B on the right.
Each metric appears in both columns at the same vertical position so
the eye can compare row by row.
"""

import pandas as pd
import streamlit as st
from lib import charts as ch
from lib import queries as q
from lib.components import kpi_card, page_setup, section_header
from lib.format import num, pct, pct_signed, safe_get, scenario_label
from lib.theme import COLORS, install_template

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

page_setup("Compare Scenarios", icon="⚖️")
install_template()

st.title("Compare Scenarios")
st.caption(
    "**Use this page to compare any two scenarios head-to-head** "
    "(e.g. baseline vs Quality excluded, or cost_high vs cost_frictionless). "
    "Both dropdowns let you pick any of the 23 backtested scenarios. "
    "If you only want to vary parameters one at a time relative to baseline, "
    "use the **Strategy Tuner** page instead."
)


# ---------------------------------------------------------------------------
# Scenario pickers (top, side by side)
# ---------------------------------------------------------------------------

scenarios = q.get_scenario_list()

picker_a, picker_b = st.columns(2)

with picker_a:
    sid_a = st.selectbox(
        "Scenario A",
        options=scenarios,
        index=scenarios.index("baseline") if "baseline" in scenarios else 0,
        format_func=scenario_label,
        key="compare_a",
    )

with picker_b:
    default_b = "excl_quality" if "excl_quality" in scenarios else scenarios[1]
    sid_b = st.selectbox(
        "Scenario B",
        options=scenarios,
        index=scenarios.index(default_b),
        format_func=scenario_label,
        key="compare_b",
    )

if sid_a == sid_b:
    st.info("Pick two different scenarios to see a meaningful comparison.")
    st.stop()


# ---------------------------------------------------------------------------
# Fetch data once
# ---------------------------------------------------------------------------

ret_a = q.get_returns(sid_a)
ret_b = q.get_returns(sid_b)
sum_a = q.get_summary(sid_a)
sum_b = q.get_summary(sid_b)


# ---------------------------------------------------------------------------
# Side-by-side KPI grid
#
# Each row = one metric.
# Left column = Scenario A's value. Right column = Scenario B's value.
# Whichever is better gets a green delta showing the gap.
# ---------------------------------------------------------------------------

section_header(
    "Side-by-side metrics",
    "Each row is one metric. Green delta = that scenario beats the other.",
)

# Column headers above all the metric rows so the user can clearly see
# which side belongs to which scenario.
headerA, headerB = st.columns(2)


def _column_header(text: str, accent: str) -> None:
    st.markdown(
        f'<div style="background:{accent}22;'
        f"border:1px solid {accent}66;"
        f"border-radius:6px;padding:0.6rem 1rem;"
        f"text-align:center;font-weight:600;font-size:0.95rem;"
        f'color:{COLORS["text"]};margin-bottom:0.75rem;">{text}</div>',
        unsafe_allow_html=True,
    )


with headerA:
    _column_header(f"SCENARIO A: {scenario_label(sid_a)}", COLORS["primary"])
with headerB:
    _column_header(f"SCENARIO B: {scenario_label(sid_b)}", COLORS["secondary"])


def render_metric_row(
    label: str,
    key: str,
    fmt,
    higher_is_better: bool = True,
    places: int | None = None,
) -> None:
    """Render one KPI row as 2 columns: A on the left, B on the right."""
    a_val = safe_get(sum_a, key)
    b_val = safe_get(sum_b, key)
    diff = a_val - b_val
    a_better = (diff > 0) if higher_is_better else (diff < 0)

    a_str = fmt(a_val, places) if places is not None else fmt(a_val)
    b_str = fmt(b_val, places) if places is not None else fmt(b_val)

    if places is not None:
        a_delta = f"{diff:+.2f}"
        b_delta = f"{-diff:+.2f}"
    else:
        a_delta = pct_signed(diff)
        b_delta = pct_signed(-diff)

    cA, cB = st.columns(2)
    with cA:
        kpi_card(
            label,
            a_str,
            delta=f"{a_delta} vs B",
            delta_positive=a_better,
        )
    with cB:
        kpi_card(
            label,
            b_str,
            delta=f"{b_delta} vs A",
            delta_positive=not a_better,
        )


metrics = [
    ("Sharpe Ratio", "sharpe_ratio", num, True, 2),
    ("Sortino Ratio", "sortino_ratio", num, True, 2),
    ("Calmar Ratio", "calmar_ratio", num, True, 2),
    ("Information Ratio", "information_ratio", num, True, 2),
    ("Annualised Return", "annualised_return", pct, True, None),
    ("Annualised Volatility", "annualised_volatility", pct, False, None),
    ("Max Drawdown", "max_drawdown", pct, True, None),
    ("Alpha", "alpha", pct_signed, True, None),
]

for label, key, fmt, higher, places in metrics:
    render_metric_row(label, key, fmt, higher, places)
    st.write("")  # vertical spacer


# ---------------------------------------------------------------------------
# Overlaid equity curves
# ---------------------------------------------------------------------------

section_header(
    "Cumulative return - both scenarios overlaid",
    "Net returns after transaction cost, compounded from 0%.",
)

st.plotly_chart(
    ch.equity_curve_compare(ret_a, scenario_label(sid_a), ret_b, scenario_label(sid_b)),
    config=ch.chart_config(f"{sid_a}_vs_{sid_b}"),
    width="stretch",
)


# ---------------------------------------------------------------------------
# Detailed numerical comparison table
# ---------------------------------------------------------------------------

section_header("Full metric breakdown")

rows = []
all_metrics = [
    ("Annualised Return", "annualised_return", pct, True),
    ("Cumulative Return", "cumulative_return", pct, True),
    ("Annualised Volatility", "annualised_volatility", pct, False),
    ("Sharpe Ratio", "sharpe_ratio", lambda v: num(v, 3), True),
    ("Sortino Ratio", "sortino_ratio", lambda v: num(v, 3), True),
    ("Calmar Ratio", "calmar_ratio", lambda v: num(v, 3), True),
    ("Information Ratio", "information_ratio", lambda v: num(v, 3), True),
    ("Max Drawdown", "max_drawdown", pct, True),
    ("Tracking Error", "tracking_error", pct, False),
    ("Alpha", "alpha", pct, True),
    ("Avg Monthly Turnover", "avg_monthly_turnover", pct, False),
]
for label, key, fmt, higher_better in all_metrics:
    a_val = safe_get(sum_a, key)
    b_val = safe_get(sum_b, key)
    diff = a_val - b_val
    a_wins = (diff > 0) == higher_better
    rows.append(
        {
            "Metric": label,
            scenario_label(sid_a): fmt(a_val),
            scenario_label(sid_b): fmt(b_val),
            "Winner": "A" if a_wins else "B",
        }
    )

st.dataframe(
    pd.DataFrame(rows),
    width="stretch",
    hide_index=True,
)
