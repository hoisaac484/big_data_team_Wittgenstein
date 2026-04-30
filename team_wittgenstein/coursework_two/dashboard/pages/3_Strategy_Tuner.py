"""Strategy Tuner (Option A).

Adjust strategy parameters via sliders. Each slider snaps to a value
that has a pre-computed scenario in the DB. Moving the slider switches
which scenario is displayed - feels live, is actually a lookup.

Layout:
  - Sidebar: 6 parameter sliders + reset button
  - Main: 2 columns side by side - Active scenario on the left,
    Baseline on the right - so the user can compare the chosen
    parameter set against the default at a glance.
"""

import streamlit as st
from lib import charts as ch
from lib import queries as q
from lib.components import kpi_card, page_setup, section_header
from lib.format import num, pct, pct_signed, safe_get, scenario_label
from lib.theme import COLORS, install_template

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

page_setup("Strategy Tuner", icon="🎛️")
install_template()

st.title("Strategy Tuner")
st.caption(
    "**Use this page to test how changing one parameter affects the strategy.** "
    "Move a slider in the sidebar to see the strategy's performance under that "
    "parameter change, always compared against the baseline. Only one parameter "
    "can be changed at a time - moving a different slider auto-resets the "
    "previous one. To compare two arbitrary scenarios, use the **Compare "
    "Scenarios** page instead."
)


# ---------------------------------------------------------------------------
# Reset handler runs BEFORE sliders are instantiated to avoid the
# 'cannot modify session_state after widget instantiation' bug.
# ---------------------------------------------------------------------------

if st.session_state.get("_reset_tuner"):
    for k in list(st.session_state.keys()):
        if k.startswith("tuner_slider_"):
            del st.session_state[k]
    st.session_state["_reset_tuner"] = False
    st.rerun()


# ---------------------------------------------------------------------------
# Parameter groups - sliders snap to pre-computed values only
# ---------------------------------------------------------------------------

PARAM_GROUPS = [
    {
        "key": "selection",
        "label": "Selection threshold",
        "help": "Top/bottom % of stocks selected for long/short.",
        "options": [0.05, 0.10, 0.15, 0.20],
        "baseline": 0.10,
        "scenarios": {
            0.05: "sens_sel_0.05",
            0.10: "baseline",
            0.15: "sens_sel_0.15",
            0.20: "sens_sel_0.20",
        },
        "format": lambda v: f"{int(v * 100)}%",
    },
    {
        "key": "ic",
        "label": "IC lookback",
        "help": "Trailing window used to compute IC weights.",
        "options": [24, 36, 48, 60],
        "baseline": 36,
        "scenarios": {
            24: "sens_ic_24",
            36: "baseline",
            48: "sens_ic_48",
            60: "sens_ic_60",
        },
        "format": lambda v: f"{v} months",
    },
    {
        "key": "ewma",
        "label": "EWMA lambda",
        "help": "Decay factor for volatility estimation.",
        "options": [0.90, 0.92, 0.94, 0.97],
        "baseline": 0.94,
        "scenarios": {
            0.90: "sens_ewma_0.90",
            0.92: "sens_ewma_0.92",
            0.94: "baseline",
            0.97: "sens_ewma_0.97",
        },
        "format": lambda v: f"{v:.2f}",
    },
    {
        "key": "notrade",
        "label": "No-trade threshold",
        "help": "Minimum weight change required to trigger a trade.",
        "options": [0.005, 0.010, 0.015, 0.020],
        "baseline": 0.010,
        "scenarios": {
            0.005: "sens_notrade_0.005",
            0.010: "baseline",
            0.015: "sens_notrade_0.015",
            0.020: "sens_notrade_0.020",
        },
        "format": lambda v: f"{v * 100:.1f}%",
    },
    {
        "key": "buffer",
        "label": "Buffer exit threshold",
        "help": "Buffer zone width before forcing a stock out of the basket.",
        "options": [0.10, 0.15, 0.20, 0.25],
        "baseline": 0.20,
        "scenarios": {
            0.10: "sens_buffer_none",
            0.15: "sens_buffer_0.15",
            0.20: "baseline",
            0.25: "sens_buffer_0.25",
        },
        "format": lambda v: ("No buffer" if v == 0.10 else f"{int(v * 100)}%"),
    },
    {
        "key": "cost",
        "label": "Transaction cost",
        "help": "One-way transaction cost on turnover.",
        "options": [0, 10, 25, 50],
        "baseline": 25,
        "scenarios": {
            0: "cost_frictionless",
            10: "cost_low",
            25: "baseline",
            50: "cost_high",
        },
        "format": lambda v: f"{int(v)} bps",
    },
]


# ---------------------------------------------------------------------------
# on_change callback: when ONE slider moves, reset all OTHER sliders to
# baseline. This enforces "only one parameter change at a time" without
# showing a warning - the UI just snaps the others back automatically.
# ---------------------------------------------------------------------------


def _enforce_single_change(my_key: str) -> None:
    for group in PARAM_GROUPS:
        other_key = f"tuner_slider_{group['key']}"
        if other_key != my_key:
            st.session_state[other_key] = group["baseline"]


# ---------------------------------------------------------------------------
# Sidebar - all sliders live here for cleanliness
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Adjust parameters")
st.sidebar.caption(
    "Only one parameter can change at a time - moving a different slider "
    "auto-resets the previous one."
)

selected_scenarios = []

for group in PARAM_GROUPS:
    slider_key = f"tuner_slider_{group['key']}"
    chosen_value = st.sidebar.select_slider(
        group["label"],
        options=group["options"],
        value=group["baseline"],
        format_func=group["format"],
        help=group["help"],
        key=slider_key,
        on_change=_enforce_single_change,
        kwargs={"my_key": slider_key},
    )
    sid = group["scenarios"][chosen_value]
    selected_scenarios.append((group["label"], sid, chosen_value, group["baseline"]))

st.sidebar.divider()

if st.sidebar.button("Reset all to baseline", width="stretch"):
    st.session_state["_reset_tuner"] = True
    st.rerun()


# ---------------------------------------------------------------------------
# Pick which scenario is "active" - exactly one or zero non-baseline now
# ---------------------------------------------------------------------------

non_baseline = [
    (label, sid, val, base)
    for (label, sid, val, base) in selected_scenarios
    if sid != "baseline"
]

if not non_baseline:
    active_scenario = "baseline"
    changed_label = None
else:
    # By construction (on_change callback) there's only ever 1
    label, sid, _, _ = non_baseline[0]
    active_scenario = sid
    changed_label = label


# ---------------------------------------------------------------------------
# Header banner showing what's being shown
# ---------------------------------------------------------------------------

if active_scenario == "baseline":
    st.markdown(
        f'<div style="background:{COLORS["surface_alt"]};'
        f'border-left:3px solid {COLORS["primary"]};'
        f'padding:1rem 1.25rem;border-radius:6px;margin-bottom:1.5rem;">'
        f"<b>Showing: Baseline</b>"
        f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;'
        f'margin-top:0.25rem;">All sliders are at their default values.</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div style="background:{COLORS["surface_alt"]};'
        f'border-left:3px solid {COLORS["secondary"]};'
        f'padding:1rem 1.25rem;border-radius:6px;margin-bottom:1.5rem;">'
        f"<b>Showing: {scenario_label(active_scenario)}</b>"
        f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;'
        f'margin-top:0.25rem;">Variant from baseline: '
        f"{changed_label}.</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Two display modes:
#   - All sliders at default -> show baseline performance only (no compare)
#   - Any slider moved -> show side-by-side: Your scenario | Baseline
# ---------------------------------------------------------------------------

baseline_summary = q.get_summary("baseline")
active_summary = q.get_summary(active_scenario)
active_returns = q.get_returns(active_scenario)
baseline_returns = q.get_returns("baseline")


metrics = [
    ("Sharpe Ratio", "sharpe_ratio", num, True, 2),
    ("Sortino Ratio", "sortino_ratio", num, True, 2),
    ("Annualised Return", "annualised_return", pct, True, None),
    ("Annualised Volatility", "annualised_volatility", pct, False, None),
    ("Max Drawdown", "max_drawdown", pct, True, None),
    ("Alpha", "alpha", pct_signed, True, None),
]


def _column_header(text: str, accent: str) -> None:
    """Sticky-style column header that labels which side belongs to which scenario."""
    st.markdown(
        f'<div style="background:{accent}22;'
        f"border:1px solid {accent}66;"
        f"border-radius:6px;padding:0.6rem 1rem;"
        f"text-align:center;font-weight:600;font-size:0.95rem;"
        f'color:{COLORS["text"]};margin-bottom:0.75rem;">{text}</div>',
        unsafe_allow_html=True,
    )


if active_scenario == "baseline":
    # MODE 1: nothing changed - just show baseline performance
    section_header(
        "Baseline performance",
        "All sliders are at their default values. Move a slider to see how a "
        "single parameter change would affect performance.",
    )

    cols = st.columns(3)
    for col, (label, key, fmt, _, places) in zip(
        cols * 2, metrics  # 6 metrics, cycle through 3 columns
    ):
        with col:
            v = safe_get(baseline_summary, key)
            v_str = fmt(v, places) if places is not None else fmt(v)
            kpi_card(label, v_str, sub="Baseline value")

    section_header("Cumulative return - baseline")
    st.plotly_chart(
        ch.equity_curve(baseline_returns),
        config=ch.chart_config("baseline_equity"),
        width="stretch",
    )

else:
    # MODE 2: a slider was moved - show side-by-side with column headers
    section_header(
        "Your scenario vs Baseline",
        f"You changed: **{changed_label}** -> "
        f"{scenario_label(active_scenario)}. Green delta = your scenario beats baseline.",
    )

    # Column headers above all the metric rows
    headerL, headerR = st.columns(2)
    with headerL:
        _column_header(
            f"YOUR SCENARIO: {scenario_label(active_scenario)}",
            COLORS["secondary"],
        )
    with headerR:
        _column_header("BASELINE (reference)", COLORS["primary"])

    def render_metric_row(label, key, fmt, higher_is_better, places):
        a_val = safe_get(active_summary, key)
        b_val = safe_get(baseline_summary, key)
        diff = a_val - b_val
        active_better = (diff > 0) if higher_is_better else (diff < 0)

        a_str = fmt(a_val, places) if places is not None else fmt(a_val)
        b_str = fmt(b_val, places) if places is not None else fmt(b_val)
        a_delta = f"{diff:+.2f}" if places is not None else pct_signed(diff)
        b_delta = f"{-diff:+.2f}" if places is not None else pct_signed(-diff)

        cA, cB = st.columns(2)
        with cA:
            kpi_card(
                label,
                a_str,
                delta=f"{a_delta} vs baseline",
                delta_positive=active_better,
            )
        with cB:
            kpi_card(
                label,
                b_str,
                delta=f"{b_delta} vs scenario",
                delta_positive=not active_better,
            )

    for label, key, fmt, higher, places in metrics:
        render_metric_row(label, key, fmt, higher, places)
        st.write("")

    section_header(
        "Cumulative return - your scenario overlaid on baseline",
        "Net returns after transaction cost, compounded from 0%.",
    )
    st.plotly_chart(
        ch.equity_curve_compare(
            active_returns,
            scenario_label(active_scenario),
            baseline_returns,
            "Baseline",
        ),
        config=ch.chart_config(f"{active_scenario}_vs_baseline"),
        width="stretch",
    )
