"""Tests for the two-stage liquidity filter."""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from modules.liquidity.liquidity_filter import (
    LiquidityConfig,
    apply_adtv_floor,
    apply_illiq_filter,
    compute_adtv,
    compute_amihud_illiq,
    run_liquidity_filter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(symbols: list[str], n_days: int, base_date: date) -> pd.DataFrame:
    """Generate synthetic price data for testing."""
    rows = []
    for sym in symbols:
        for i in range(n_days):
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": base_date - timedelta(days=n_days - 1 - i),
                    "adjusted_close": 100.0 + i * 0.5,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_adtv
# ---------------------------------------------------------------------------


class TestComputeAdtv:

    def test_basic(self):
        """ADTV equals mean dollar volume over lookback window."""
        prices = _make_prices(["AAPL"], n_days=25, base_date=date(2024, 1, 31))
        result = compute_adtv(prices, lookback=20)

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"

        # Manually compute expected ADTV: mean of last 20 dollar volumes
        df = prices.sort_values("trade_date")
        dv = (df["adjusted_close"] * df["volume"]).tail(20)
        expected = dv.mean()
        assert abs(result.iloc[0]["adtv"] - expected) < 0.01

    def test_insufficient_history(self):
        """Symbols with fewer than lookback days are excluded."""
        prices = _make_prices(["SHORT"], n_days=15, base_date=date(2024, 1, 31))
        result = compute_adtv(prices, lookback=20)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# apply_adtv_floor
# ---------------------------------------------------------------------------


class TestApplyAdtvFloor:

    def test_filters_correctly(self):
        """Stocks below the $1M threshold are removed."""
        adtv_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "adtv": [5_000_000, 500_000, 10_000_000],
            }
        )
        result = apply_adtv_floor(adtv_df, adtv_min_dollar=1_000_000)
        assert set(result["symbol"]) == {"A", "C"}
        assert "B" not in result["symbol"].values

    def test_empty_input(self):
        """Empty DataFrame returns empty."""
        empty = pd.DataFrame(columns=["symbol", "adtv"])
        result = apply_adtv_floor(empty, adtv_min_dollar=1_000_000)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# compute_amihud_illiq
# ---------------------------------------------------------------------------


class TestComputeAmihudIlliq:

    def test_formula(self):
        """ILLIQ matches manual calculation for known prices."""
        # 25 days, known prices for manual verification
        n = 25
        base = date(2024, 1, 31)
        prices_list = [100.0 + i for i in range(n)]
        rows = [
            {
                "symbol": "X",
                "trade_date": base - timedelta(days=n - 1 - i),
                "adjusted_close": prices_list[i],
                "volume": 500_000,
            }
            for i in range(n)
        ]
        prices = pd.DataFrame(rows)

        result = compute_amihud_illiq(prices, ["X"], lookback=21)
        assert len(result) == 1

        # Manual calculation: last 21 returns (indices 4..24 have rolling coverage)
        adj = np.array(prices_list)
        log_ret = np.log(adj[1:] / adj[:-1])
        dv = adj[1:] * 500_000
        illiq_ratios = np.abs(log_ret) / dv
        # The rolling(21) mean of illiq_ratios uses the last 21 values
        expected = np.mean(illiq_ratios[-21:])
        assert abs(result.iloc[0]["amihud_illiq"] - expected) < 1e-12

    def test_only_survivors(self):
        """Non-survivor symbols are excluded from computation."""
        prices = _make_prices(["A", "B", "C"], n_days=25, base_date=date(2024, 1, 31))
        result = compute_amihud_illiq(prices, ["A", "C"], lookback=21)
        assert set(result["symbol"]) == {"A", "C"}

    def test_zero_return(self):
        """Flat prices produce zero ILLIQ, not errors."""
        n = 25
        base = date(2024, 1, 31)
        rows = [
            {
                "symbol": "FLAT",
                "trade_date": base - timedelta(days=n - 1 - i),
                "adjusted_close": 50.0,  # constant price
                "volume": 1_000_000,
            }
            for i in range(n)
        ]
        prices = pd.DataFrame(rows)
        result = compute_amihud_illiq(prices, ["FLAT"], lookback=21)
        assert len(result) == 1
        assert result.iloc[0]["amihud_illiq"] == 0.0


# ---------------------------------------------------------------------------
# apply_illiq_filter
# ---------------------------------------------------------------------------


class TestApplyIlliqFilter:

    def test_removes_top_decile(self):
        """Top 10% most illiquid stocks are removed."""
        illiq_df = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(10)],
                "amihud_illiq": list(range(1, 11)),
            }
        )
        result = apply_illiq_filter(illiq_df, removal_pct=0.10)
        # S9 (illiq=10) is rank 1.0, should be removed
        assert "S9" not in result["symbol"].values
        assert len(result) == 9

    def test_cross_sectional(self):
        """Ranking is across all stocks, not per sector (no sector column used)."""
        illiq_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "amihud_illiq": [1.0, 2.0, 3.0, 100.0],
            }
        )
        result = apply_illiq_filter(illiq_df, removal_pct=0.25)
        # D is the most illiquid (rank=1.0), should be removed
        assert "D" not in result["symbol"].values
        assert len(result) == 3


# ---------------------------------------------------------------------------
# run_liquidity_filter (orchestrator)
# ---------------------------------------------------------------------------


class TestRunLiquidityFilter:

    def test_full_flow(self):
        """End-to-end: mocked DB returns prices, verify survivors and persist."""
        n_symbols = 20
        n_days = 30
        base = date(2024, 1, 31)
        rows = []
        for idx in range(n_symbols):
            for i in range(n_days):
                # Vary volume per symbol so ILLIQ values differ
                rows.append(
                    {
                        "symbol": f"S{idx:02d}",
                        "trade_date": base - timedelta(days=n_days - 1 - i),
                        "adjusted_close": 100.0 + i * 0.5,
                        "volume": (idx + 1) * 100_000,
                    }
                )
        prices = pd.DataFrame(rows)

        db = MagicMock()
        db.read_query.return_value = prices

        config = LiquidityConfig(
            adtv_lookback_days=20,
            illiq_lookback_days=21,
            illiq_removal_pct=0.10,
            adtv_min_dollar=1,  # tiny threshold so no stocks fail ADTV floor
        )

        survivors = run_liquidity_filter(db, date(2024, 2, 1), config)

        # 20 stocks, remove top 10% ILLIQ → 18 survive
        assert len(survivors) == 18

        # Persist is called with ALL 20 stocks, not just survivors
        db.write_dataframe_on_conflict_do_nothing.assert_called_once()
        persisted_df = db.write_dataframe_on_conflict_do_nothing.call_args[0][0]
        assert len(persisted_df) == 20
        assert "passes_adv" in persisted_df.columns
        assert "passes_illiq" in persisted_df.columns
        assert "passes_filter" in persisted_df.columns
        assert persisted_df["passes_filter"].sum() == 18

    def test_empty_prices(self):
        """No price data returns empty list without persisting."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()

        config = LiquidityConfig()
        survivors = run_liquidity_filter(db, date(2024, 2, 1), config)

        assert survivors == []
        db.write_dataframe_on_conflict_do_nothing.assert_not_called()

    def test_point_in_time(self):
        """SQL query uses strict less-than on rebalance_date."""
        db = MagicMock()
        db.read_query.return_value = pd.DataFrame()

        config = LiquidityConfig()
        run_liquidity_filter(db, date(2024, 2, 1), config)

        sql_arg = db.read_query.call_args[0][0]
        assert "trade_date < :rebalance_date" in sql_arg
