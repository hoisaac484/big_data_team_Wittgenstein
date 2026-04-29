"""Tests for portfolio/position_builder.py — Steps 4-7."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.portfolio.position_builder import (
    PositionConfig,
    apply_liquidity_cap,
    apply_no_trade_zone,
    build_portfolio_positions,
    compute_sector_weights,
    fetch_adv,
    fetch_previous_weights,
    verify_constraints,
)

REBALANCE = date(2024, 3, 29)
LONG_BUDGET = 1.3
SHORT_BUDGET = 0.3


# ── Helpers ───────────────────────────────────────────────────────────────────


def _scored(n_long=5, n_short=5, sectors=None, score=1.0):
    """Build a minimal scored DataFrame for testing."""
    if sectors is None:
        sectors = ["SectorA"] * (n_long + n_short)
    symbols_l = [f"L{i:02d}" for i in range(n_long)]
    symbols_s = [f"S{i:02d}" for i in range(n_short)]
    rows = []
    for sym, sec in zip(symbols_l, sectors[:n_long]):
        rows.append(
            {
                "symbol": sym,
                "sector": sec,
                "direction": "long",
                "composite_score": score,
                "ewma_vol": 0.20,
                "risk_adj_score": score / 0.20,
            }
        )
    for sym, sec in zip(symbols_s, sectors[n_long:]):
        rows.append(
            {
                "symbol": sym,
                "sector": sec,
                "direction": "short",
                "composite_score": score,
                "ewma_vol": 0.20,
                "risk_adj_score": score / 0.20,
            }
        )
    return pd.DataFrame(rows)


def _adv(symbols, adv_val=1e9):
    return pd.DataFrame({"symbol": symbols, "adv_20d": [adv_val] * len(symbols)})


# ── compute_sector_weights (Step 4) ───────────────────────────────────────────


class TestComputeSectorWeights:

    def test_long_weights_sum_to_long_budget(self):
        df = _scored(n_long=5, n_short=5)
        result = compute_sector_weights(df)
        long_sum = result.loc[result["direction"] == "long", "target_weight"].sum()
        assert long_sum == pytest.approx(LONG_BUDGET, abs=1e-6)

    def test_short_weights_sum_to_short_budget(self):
        df = _scored(n_long=5, n_short=5)
        result = compute_sector_weights(df)
        short_sum = result.loc[result["direction"] == "short", "target_weight"].sum()
        assert short_sum == pytest.approx(SHORT_BUDGET, abs=1e-6)

    def test_weights_proportional_to_risk_adj_score(self):
        """Higher risk_adj_score → higher weight within same sector."""
        df = _scored(n_long=2, n_short=0)
        df.loc[0, "risk_adj_score"] = 3.0
        df.loc[1, "risk_adj_score"] = 1.0
        result = compute_sector_weights(df)
        w0 = result.loc[0, "target_weight"]
        w1 = result.loc[1, "target_weight"]
        assert w0 == pytest.approx(w1 * 3, abs=1e-6)

    def test_multi_sector_budget_split(self):
        """With 2 sectors, each gets half the long budget."""
        sectors = ["SectorA", "SectorA", "SectorB", "SectorB"]
        df = _scored(n_long=4, n_short=0, sectors=sectors * 2)
        # Keep only longs
        df = df[df["direction"] == "long"].copy()
        result = compute_sector_weights(df)
        a_sum = result.loc[result["sector"] == "SectorA", "target_weight"].sum()
        b_sum = result.loc[result["sector"] == "SectorB", "target_weight"].sum()
        assert a_sum == pytest.approx(LONG_BUDGET / 2, abs=1e-6)
        assert b_sum == pytest.approx(LONG_BUDGET / 2, abs=1e-6)

    def test_equal_fallback_when_zero_scores(self):
        """When all risk_adj_scores are zero, equal weights are assigned."""
        df = _scored(n_long=4, n_short=0, score=0.0)
        df["risk_adj_score"] = 0.0
        result = compute_sector_weights(df)
        w = result.loc[result["direction"] == "long", "target_weight"]
        assert w.nunique() == 1  # all equal

    def test_target_weight_column_added(self):
        df = _scored()
        result = compute_sector_weights(df)
        assert "target_weight" in result.columns


# ── apply_liquidity_cap (Step 5) ──────────────────────────────────────────────


class TestApplyLiquidityCap:

    def test_uncapped_when_adv_large(self):
        df = compute_sector_weights(_scored(n_long=5, n_short=5))
        adv = _adv(df["symbol"].tolist(), adv_val=1e10)
        result = apply_liquidity_cap(df, adv, aum=1e9, cap_pct=0.05)
        assert not result["liquidity_capped"].any()

    def test_caps_oversized_positions(self):
        """A very small ADV should cap the position weight."""
        df = compute_sector_weights(_scored(n_long=2, n_short=0))
        # Tiny ADV → cap = 0.05 * 1000 / 1e9 ≈ 5e-8 (much smaller than weight)
        adv = _adv(df["symbol"].tolist(), adv_val=1000.0)
        result = apply_liquidity_cap(df, adv, aum=1e9, cap_pct=0.05)
        assert result["liquidity_capped"].any()

    def test_total_weight_preserved_after_capping(self):
        """Total weight in a direction is preserved after redistribution."""
        df = compute_sector_weights(_scored(n_long=5, n_short=0))
        pre_total = df.loc[df["direction"] == "long", "target_weight"].sum()
        # Use small AUM so only the illiquid stock (adv=100) is capped;
        # first 4 stocks have cap = 0.05 * 1e9 / 1e6 = 50 >> 0.26 (uncapped).
        adv_vals = [1e9, 1e9, 1e9, 1e9, 100.0]  # last stock very illiquid
        symbols = df["symbol"].tolist()
        adv = pd.DataFrame({"symbol": symbols, "adv_20d": adv_vals})
        result = apply_liquidity_cap(df, adv, aum=1e6, cap_pct=0.05)
        post_total = result.loc[result["direction"] == "long", "target_weight"].sum()
        assert post_total == pytest.approx(pre_total, abs=1e-6)

    def test_missing_adv_no_cap(self):
        """Stocks without ADV data should not be capped (cap_weight = inf)."""
        df = compute_sector_weights(_scored(n_long=3, n_short=0))
        empty_adv = pd.DataFrame(columns=["symbol", "adv_20d"])
        result = apply_liquidity_cap(df, empty_adv, aum=1e9, cap_pct=0.05)
        assert not result["liquidity_capped"].any()

    def test_liquidity_capped_flag_set(self):
        df = compute_sector_weights(_scored(n_long=2, n_short=0))
        adv = _adv(df["symbol"].tolist(), adv_val=100.0)
        result = apply_liquidity_cap(df, adv, aum=1e9, cap_pct=0.05)
        assert "liquidity_capped" in result.columns

    def test_group_not_over_cap_is_skipped(self):
        """Groups where no stock exceeds the cap are skipped in the inner loop."""
        sectors = ["SectorA", "SectorA", "SectorB", "SectorB"]
        df_in = _scored(n_long=4, n_short=0, sectors=sectors)
        df_in = df_in[df_in["direction"] == "long"].reset_index(drop=True)
        df = compute_sector_weights(df_in)
        # SectorA: tiny ADV → capped; SectorB: huge ADV → never capped
        adv_vals = df["sector"].map({"SectorA": 100.0, "SectorB": 1e12})
        adv = pd.DataFrame({"symbol": df["symbol"], "adv_20d": adv_vals})
        result = apply_liquidity_cap(df, adv, aum=1e9, cap_pct=0.05)
        assert result.loc[result["sector"] == "SectorA", "liquidity_capped"].all()
        assert not result.loc[result["sector"] == "SectorB", "liquidity_capped"].any()


# ── apply_no_trade_zone (Step 6) ──────────────────────────────────────────────


class TestApplyNoTradeZone:

    def _with_weights(self, target_w=0.10):
        df = _scored(n_long=2, n_short=0)
        df["target_weight"] = target_w
        return df

    def test_new_positions_always_trade(self):
        df = self._with_weights(0.10)
        previous = pd.DataFrame(columns=["symbol", "direction", "final_weight"])
        result = apply_no_trade_zone(df, previous, threshold=0.005)
        assert (result["trade_action"] == "trade").all()

    def test_hold_when_deviation_below_threshold(self):
        df = self._with_weights(0.101)
        prev_w = 0.100
        previous = pd.DataFrame(
            {
                "symbol": df["symbol"].tolist(),
                "direction": ["long"] * len(df),
                "final_weight": [prev_w] * len(df),
            }
        )
        result = apply_no_trade_zone(df, previous, threshold=0.005)
        assert (result["trade_action"] == "hold").all()
        assert result["final_weight"].tolist() == pytest.approx([prev_w] * len(result))

    def test_trade_when_deviation_above_threshold(self):
        df = self._with_weights(0.15)
        previous = pd.DataFrame(
            {
                "symbol": df["symbol"].tolist(),
                "direction": ["long"] * len(df),
                "final_weight": [0.10] * len(df),
            }
        )
        result = apply_no_trade_zone(df, previous, threshold=0.005)
        assert (result["trade_action"] == "trade").all()
        assert result["final_weight"].tolist() == pytest.approx([0.15] * len(result))

    def test_direction_change_always_trades(self):
        """A stock that flips long→short should always trade."""
        df = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "sector": "IT",
                    "direction": "short",
                    "composite_score": 1.0,
                    "ewma_vol": 0.2,
                    "risk_adj_score": 5.0,
                    "target_weight": 0.10,
                }
            ]
        )
        previous = pd.DataFrame(
            [{"symbol": "AAPL", "direction": "long", "final_weight": 0.10}]
        )
        result = apply_no_trade_zone(df, previous, threshold=0.005)
        assert result.iloc[0]["trade_action"] == "trade"

    def test_final_weight_and_trade_action_columns_present(self):
        df = self._with_weights()
        result = apply_no_trade_zone(
            df, pd.DataFrame(columns=["symbol", "direction", "final_weight"]), 0.005
        )
        assert "final_weight" in result.columns
        assert "trade_action" in result.columns

    def test_zero_traded_target_sum_equal_split(self):
        """Traded stocks with zero target_weight trigger the equal-split fallback."""
        df = pd.DataFrame(
            [
                {
                    "symbol": "NEW",
                    "sector": "SectorA",
                    "direction": "long",
                    "target_weight": 0.0,
                }
            ]
        )
        previous = pd.DataFrame(columns=["symbol", "direction", "final_weight"])
        result = apply_no_trade_zone(df, previous, threshold=0.005)
        assert result.iloc[0]["trade_action"] == "trade"
        assert result.iloc[0]["final_weight"] == pytest.approx(0.0)

    def test_hold_within_one_percent_threshold(self):
        """Deviation of 0.8% is below the spec 1% threshold → hold."""
        df = self._with_weights(0.108)  # deviation = 0.008 < 0.01
        previous = pd.DataFrame(
            {
                "symbol": df["symbol"].tolist(),
                "direction": ["long"] * len(df),
                "final_weight": [0.100] * len(df),
            }
        )
        result = apply_no_trade_zone(df, previous, threshold=0.01)
        assert (result["trade_action"] == "hold").all()

    def test_trade_above_one_percent_threshold(self):
        """Deviation of 1.5% is above the spec 1% threshold → trade."""
        df = self._with_weights(0.115)  # deviation = 0.015 > 0.01
        previous = pd.DataFrame(
            {
                "symbol": df["symbol"].tolist(),
                "direction": ["long"] * len(df),
                "final_weight": [0.100] * len(df),
            }
        )
        result = apply_no_trade_zone(df, previous, threshold=0.01)
        assert (result["trade_action"] == "trade").all()

    def test_default_config_threshold_is_one_percent(self):
        assert PositionConfig().no_trade_threshold == pytest.approx(0.01)


# ── verify_constraints (Step 7) ───────────────────────────────────────────────


class TestVerifyConstraints:

    def _positions(self, long_w=1.3 / 10, short_w=0.3 / 10, n=10):
        longs = pd.DataFrame({"direction": ["long"] * n, "final_weight": [long_w] * n})
        shorts = pd.DataFrame(
            {"direction": ["short"] * n, "final_weight": [short_w] * n}
        )
        return pd.concat([longs, shorts], ignore_index=True)

    def test_passes_when_constraints_met(self):
        df = self._positions()
        assert verify_constraints(df, tolerance=0.01) is True

    def test_fails_when_long_sum_wrong(self):
        df = self._positions(long_w=0.05)  # Σlong = 0.5, not 1.3
        assert verify_constraints(df, tolerance=0.01) is False

    def test_fails_when_short_sum_wrong(self):
        df = self._positions(short_w=0.10)  # Σshort = 1.0, not 0.3
        assert verify_constraints(df, tolerance=0.01) is False

    def test_returns_false_for_empty_df(self):
        assert verify_constraints(pd.DataFrame(), tolerance=0.01) is False

    def test_net_exposure_check(self):
        """Net exposure = Σlong - Σshort should be ≈ 1.0."""
        df = self._positions()
        # 1.3 - 0.3 = 1.0 → should pass
        assert verify_constraints(df, tolerance=0.01) is True


# ── build_portfolio_positions (orchestrator) ──────────────────────────────────


_PATCH_ADV = "modules.portfolio.position_builder.fetch_adv"
_PATCH_PREV = "modules.portfolio.position_builder.fetch_previous_weights"


class TestBuildPortfolioPositions:

    def _run(self, scored, adv_val=1e9, previous=None):
        """Run build_portfolio_positions with DB helpers patched."""
        adv_df = _adv(scored["symbol"].tolist(), adv_val=adv_val)
        prev_df = (
            previous
            if previous is not None
            else pd.DataFrame(columns=["symbol", "direction", "final_weight"])
        )
        db = MagicMock()
        with patch(_PATCH_ADV, return_value=adv_df), patch(
            _PATCH_PREV, return_value=prev_df
        ):
            return build_portfolio_positions(db, scored, REBALANCE, PositionConfig())

    def test_returns_dataframe(self):
        result = self._run(_scored(n_long=5, n_short=5))
        assert isinstance(result, pd.DataFrame)

    def test_rebalance_date_column_present(self):
        result = self._run(_scored(n_long=5, n_short=5))
        assert "rebalance_date" in result.columns
        assert (result["rebalance_date"] == REBALANCE).all()

    def test_empty_scored_returns_empty(self):
        db = MagicMock()
        result = build_portfolio_positions(
            db, pd.DataFrame(), REBALANCE, PositionConfig()
        )
        assert result.empty

    def test_expected_output_columns(self):
        result = self._run(_scored(n_long=3, n_short=3))
        for col in [
            "rebalance_date",
            "symbol",
            "direction",
            "target_weight",
            "final_weight",
            "liquidity_capped",
            "trade_action",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_all_new_positions_trade(self):
        """On first run (no previous positions), all trade_action should be 'trade'."""
        result = self._run(_scored(n_long=3, n_short=3))
        assert (result["trade_action"] == "trade").all()


# ── fetch_adv ─────────────────────────────────────────────────────────────────


class TestFetchAdv:

    def test_returns_empty_when_db_returns_none(self):
        db = MagicMock()
        db.read_query.return_value = None
        result = fetch_adv(db, ["AAPL"], REBALANCE)
        assert list(result.columns) == ["symbol", "adv_20d"]
        assert result.empty

    def test_returns_empty_when_db_returns_empty(self):
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()
        result = fetch_adv(db, ["AAPL"], REBALANCE)
        assert list(result.columns) == ["symbol", "adv_20d"]
        assert result.empty

    def test_computes_rolling_mean(self):
        """adv_20d equals the rolling mean over the lookback window."""
        db = MagicMock()
        lookback = 3
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        db.read_query.return_value = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 5,
                "trade_date": dates,
                "dollar_vol": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )
        result = fetch_adv(db, ["AAPL"], REBALANCE, lookback_days=lookback)
        # Rolling mean of last 3 values: (300 + 400 + 500) / 3 = 400.0
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["adv_20d"] == pytest.approx(400.0)


# ── fetch_previous_weights ────────────────────────────────────────────────────


class TestFetchPreviousWeights:

    def test_returns_empty_when_db_returns_none(self):
        db = MagicMock()
        db.read_query.return_value = None
        result = fetch_previous_weights(db, REBALANCE)
        assert result.empty
        assert list(result.columns) == ["symbol", "direction", "final_weight"]

    def test_returns_empty_when_db_returns_empty(self):
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame(
            columns=["symbol", "direction", "final_weight"]
        )
        result = fetch_previous_weights(db, REBALANCE)
        assert result.empty

    def test_passes_through_previous_weights(self):
        db = MagicMock()
        prev = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "direction": ["long", "short"],
                "final_weight": [0.10, 0.05],
            }
        )
        db.read_query.return_value = prev
        result = fetch_previous_weights(db, REBALANCE)
        assert len(result) == 2
        assert list(result["symbol"]) == ["AAPL", "MSFT"]
