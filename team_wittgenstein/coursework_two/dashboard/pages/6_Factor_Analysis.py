"""Factor Analysis.

Inspect the IC weighting and the cross-sectional shape of factor scores.
"""

import streamlit as st
from lib import charts as ch
from lib import queries as q
from lib.components import page_setup, section_header
from lib.theme import COLORS, install_template

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

page_setup("Factor Analysis", icon="📊")
install_template()

st.title("Factor Analysis")
st.caption(
    "How the four factors have been weighted over time, and how their "
    "cross-sectional z-scores were distributed across sectors."
)


# ---------------------------------------------------------------------------
# IC weight evolution
# ---------------------------------------------------------------------------

section_header(
    "IC weight evolution",
    "Factor weights derived from rolling 36-month IC. Stacked area shows "
    "how reliance on each factor shifted over time. Sum to 100% each month.",
)

ic_weights = q.get_ic_weights()
if ic_weights.empty:
    st.info("No IC weight data available.")
else:
    st.plotly_chart(
        ch.ic_weights_evolution(ic_weights),
        config=ch.chart_config("ic_weights"),
        width="stretch",
    )

    # Average IC and weight per factor
    summary = (
        ic_weights.groupby("factor_name")
        .agg(avg_ic=("ic_mean_36m", "mean"), avg_weight=("ic_weight", "mean"))
        .reset_index()
        .sort_values("avg_weight", ascending=False)
    )
    summary["avg_ic"] = summary["avg_ic"].round(4)
    summary["avg_weight"] = summary["avg_weight"].round(3)
    summary = summary.rename(
        columns={
            "factor_name": "Factor",
            "avg_ic": "Avg 36m IC",
            "avg_weight": "Avg weight",
        }
    )
    st.markdown("**Time-averaged IC and weight per factor**")
    st.dataframe(summary, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Composite score distribution on a chosen date
# ---------------------------------------------------------------------------

section_header(
    "Composite score distribution",
    "Each stock gets one composite score per month - a single number "
    "summarising how strong it looks across all four factors (Value, Quality, "
    "Momentum, Low Vol). The histogram shows how those ~390 scores are "
    "spread across the universe on the chosen date.",
)

# Colour key for the bars
st.markdown(
    f"""
    <div style="display:flex;gap:1.25rem;flex-wrap:wrap;
        font-size:0.9rem;margin:0.5rem 0 1rem 0;">
      <div style="display:flex;align-items:center;gap:0.4rem;">
        <span style="display:inline-block;width:14px;height:14px;
            background:{COLORS['long']};border-radius:3px;"></span>
        <span><b>Top 10% universe-wide</b> - candidates for the long basket</span>
      </div>
      <div style="display:flex;align-items:center;gap:0.4rem;">
        <span style="display:inline-block;width:14px;height:14px;
            background:{COLORS['short']};border-radius:3px;"></span>
        <span><b>Bottom 10% universe-wide</b> - candidates for the short basket</span>
      </div>
      <div style="display:flex;align-items:center;gap:0.4rem;">
        <span style="display:inline-block;width:14px;height:14px;
            background:{COLORS['text_muted']};border-radius:3px;"></span>
        <span><b>Middle 80%</b> - not selected</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "**Colour follows percentile, not absolute value.** A bar can be green "
    "even if its score is below zero - it's green because it's in the top "
    "10% of the universe that month. The strategy ranks stocks, it doesn't "
    "use a fixed score threshold.\n\n"
    "**Cutoff lines are approximate.** The strategy actually picks the top "
    "and bottom 10% **per sector**, not universe-wide. So the dashed lines "
    "show the universe-wide 10th and 90th percentiles as a rough indicator - "
    "real per-sector cutoffs vary slightly above and below these lines.\n\n"
    "**Wide spread** = factors are clearly differentiating stocks (good for "
    "the strategy). **Narrow spread** = factors aren't discriminating well "
    "that month."
)

dates = q.get_rebalance_dates()
if dates:
    # The most recent date often has not yet had its composite computed
    # (because IC needs the next month's returns). Default one back.
    default_idx = max(0, len(dates) - 2)
    selected_date = st.select_slider(
        "Rebalance date",
        options=dates,
        value=dates[default_idx],
        format_func=lambda d: d.strftime("%b %Y"),
        key="factor_analysis_date",
    )
    composite_dist = q.get_composite_distribution(selected_date)
    if composite_dist.empty:
        st.info(
            f"**No composite scores for {selected_date.strftime('%B %Y')}.** "
            "This is expected for the most recent month - composite scores "
            "are produced by the rebalance pipeline once that month closes. "
            "Slide to an earlier date to see the distribution."
        )
    else:
        st.plotly_chart(
            ch.composite_histogram(composite_dist),
            config=ch.chart_config(f"composite_{selected_date.date()}"),
            width="stretch",
        )
        st.caption(
            f"{len(composite_dist)} stocks scored. "
            f"Range {composite_dist['composite_score'].min():.2f} to "
            f"{composite_dist['composite_score'].max():.2f}, "
            f"median {composite_dist['composite_score'].median():.2f}."
        )


# ---------------------------------------------------------------------------
# Sector z-score boxplots per factor (uses the same selected_date above)
# ---------------------------------------------------------------------------

section_header(
    "Z-score distribution by sector",
    "How each factor's z-scores are spread across the 11 GICS sectors on the "
    "chosen rebalance date. Boxes near zero = balanced; long tails = sector skew.",
)

if dates:
    factor_choice = st.selectbox(
        "Factor",
        options=["z_value", "z_quality", "z_momentum", "z_low_vol"],
        format_func=lambda c: {
            "z_value": "Value",
            "z_quality": "Quality",
            "z_momentum": "Momentum",
            "z_low_vol": "Low Volatility",
        }[c],
        key="factor_analysis_factor",
    )
    zscores = q.get_zscore_by_sector(selected_date, factor_choice)
    if zscores.empty:
        st.info("No z-scores for this date/factor.")
    else:
        st.plotly_chart(
            ch.factor_zscore_boxplot(zscores, factor_choice.replace("z_", "").title()),
            config=ch.chart_config(f"{factor_choice}_boxplot"),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Factor correlation matrix
# ---------------------------------------------------------------------------

section_header(
    "Factor correlation matrix",
    "Pairwise Pearson correlations between the four factors over the entire "
    "backtest period. Independent factors (correlation near 0) are good - they "
    "carry distinct information. High correlation (e.g. > 0.5) means two "
    "factors are giving similar signals.",
)

corr_df = q.get_factor_correlations()
if corr_df.empty:
    st.info("No factor scores available for correlation analysis.")
else:
    st.plotly_chart(
        ch.factor_correlation_heatmap(corr_df),
        config=ch.chart_config("factor_correlations"),
        width="stretch",
    )
    st.caption(
        f"Computed across **{len(corr_df):,}** stock-month observations. "
        "Note: Low Volatility is orthogonalised against Momentum during "
        "factor construction, so its correlation with Momentum should be "
        "close to zero by design."
    )
