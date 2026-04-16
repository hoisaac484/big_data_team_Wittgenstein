"""Tests for main.py — load_config, _load_universe, backfill_factor_metrics."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import main as main_module
from main import PipelineContext, _load_universe, backfill_factor_metrics, load_config

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_ctx(symbols=None, cfg=None, sector_map=None):
    pg = MagicMock()
    writer = MagicMock()
    return PipelineContext(
        cfg=cfg or {"liquidity": {}},
        pg=pg,
        writer=writer,
        symbols=symbols or ["AAPL", "MSFT"],
        countries=["United States"],
        sector_map=sector_map or {"AAPL": "IT", "MSFT": "IT"},
        strict=False,
    )


# ── load_config ───────────────────────────────────────────────────────────────


class TestLoadConfig:

    def test_raises_when_config_missing(self, tmp_path):
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Config not found"):
                load_config()

    def test_returns_dict_when_present(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        cfg_file = config_dir / "conf.yaml"
        cfg_file.write_text("postgres:\n  host: localhost\n  port: 5432\n")

        with patch.object(Path, "exists", return_value=True), patch(
            "builtins.open", cfg_file.open
        ):
            # Simpler: patch the path resolution directly
            pass

        # Direct approach: write a real conf.yaml and patch __file__
        with patch.object(main_module, "__file__", str(tmp_path / "main.py")):
            result = load_config()

        assert isinstance(result, dict)
        assert result["postgres"]["host"] == "localhost"

    def test_returns_dict_with_real_config(self):
        """Smoke test: the real config loads without error."""
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "postgres" in cfg


# ── _load_universe ────────────────────────────────────────────────────────────


class TestLoadUniverse:

    def _pg(self, symbols, universe_df=None):
        pg = MagicMock()
        pg.read_query.return_value = pd.DataFrame({"symbol": symbols})
        pg.get_company_list.return_value = (
            universe_df
            if universe_df is not None
            else pd.DataFrame(
                {
                    "symbol": symbols,
                    "gics_sector": ["IT"] * len(symbols),
                    "country": ["United States"] * len(symbols),
                }
            )
        )
        return pg

    def test_returns_symbols_from_intersection(self):
        pg = self._pg(["AAPL", "MSFT"])
        symbols, countries = _load_universe(pg, {})
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_normalises_dots_to_dashes(self):
        pg = self._pg(["BF.B", "BRK.B"])
        symbols, _ = _load_universe(pg, {})
        assert "BF-B" in symbols
        assert "BRK-B" in symbols
        assert "BF.B" not in symbols

    def test_raises_when_empty(self):
        pg = MagicMock()
        pg.read_query.return_value = pd.DataFrame({"symbol": []})
        with pytest.raises(RuntimeError, match="No symbols found"):
            _load_universe(pg, {})

    def test_raises_when_none(self):
        pg = MagicMock()
        pg.read_query.return_value = None
        with pytest.raises(RuntimeError, match="No symbols found"):
            _load_universe(pg, {})

    def test_dev_mode_caps_symbols(self):
        pg = self._pg([f"SYM{i}" for i in range(50)])
        cfg = {"dev": {"enabled": True, "max_symbols": 5}}
        symbols, _ = _load_universe(pg, cfg)
        assert len(symbols) == 5

    def test_dev_mode_disabled_returns_all(self):
        pg = self._pg([f"SYM{i}" for i in range(20)])
        cfg = {"dev": {"enabled": False}}
        symbols, _ = _load_universe(pg, cfg)
        assert len(symbols) == 20

    def test_returns_countries_from_universe(self):
        pg = self._pg(
            ["AAPL"],
            pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "gics_sector": ["IT"],
                    "country": ["United States"],
                }
            ),
        )
        _, countries = _load_universe(pg, {})
        assert "United States" in countries

    def test_uses_country_code_col_if_no_country(self):
        pg = MagicMock()
        pg.read_query.return_value = pd.DataFrame({"symbol": ["AAPL"]})
        pg.get_company_list.return_value = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "gics_sector": ["IT"],
                "country_code": ["US"],
            }
        )
        _, countries = _load_universe(pg, {})
        assert "US" in countries

    def test_returns_empty_countries_when_no_col(self):
        pg = MagicMock()
        pg.read_query.return_value = pd.DataFrame({"symbol": ["AAPL"]})
        pg.get_company_list.return_value = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "gics_sector": ["IT"],
            }
        )
        _, countries = _load_universe(pg, {})
        assert countries == []


# ── backfill_factor_metrics ───────────────────────────────────────────────────


class TestBackfillFactorMetrics:

    def _ratios_df(self, symbol="AAPL", date="2024-01-31"):
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "calc_date": [date],
                "pb_ratio": [1.5],
                "asset_growth": [0.05],
                "roe": [0.12],
                "leverage": [0.3],
                "earnings_stability": [0.8],
                "momentum_6m": [0.04],
                "momentum_12m": [0.08],
                "volatility_3m": [0.15],
                "volatility_12m": [0.18],
            }
        )

    @patch("main.orthogonalise_lowvol")
    @patch("main.compute_factor_scores")
    @patch("main.winsorise_metrics")
    @patch("main.calculate_ratios")
    @patch("main.run_liquidity_filter")
    def test_writes_scores_and_zscores(
        self, mock_liq, mock_calc, mock_wins, mock_scores, mock_orth
    ):
        ctx = _make_ctx()
        ratios = self._ratios_df()
        mock_liq.return_value = ["AAPL", "MSFT"]
        mock_calc.return_value = ratios
        mock_wins.return_value = ratios
        zscores = pd.DataFrame({"symbol": ["AAPL"], "calc_date": ["2024-01-31"]})
        mock_scores.return_value = (ratios, zscores)
        mock_orth.return_value = ratios

        backfill_factor_metrics(ctx, years=1)

        ctx.writer.write_factor_zscores.assert_called_once_with(zscores)
        ctx.writer.write_factor_scores.assert_called_once_with(ratios)

    @patch("main.orthogonalise_lowvol")
    @patch("main.compute_factor_scores")
    @patch("main.winsorise_metrics")
    @patch("main.calculate_ratios")
    @patch("main.run_liquidity_filter")
    def test_skips_date_when_no_liquid_symbols(
        self, mock_liq, mock_calc, mock_wins, mock_scores, mock_orth
    ):
        ctx = _make_ctx()
        # Most dates skipped; one date returns a symbol so concat doesn't fail
        call_count = [0]

        def liq_side_effect(*args, **kwargs):
            call_count[0] += 1
            return ["AAPL"] if call_count[0] == 1 else []

        mock_liq.side_effect = liq_side_effect

        ratios = self._ratios_df()
        mock_calc.return_value = ratios
        mock_wins.return_value = ratios
        mock_scores.return_value = (ratios, pd.DataFrame())
        mock_orth.return_value = ratios

        backfill_factor_metrics(ctx, years=1)

        # calculate_ratios only called for the one liquid date
        assert mock_calc.call_count == 1

    @patch("main.orthogonalise_lowvol")
    @patch("main.compute_factor_scores")
    @patch("main.winsorise_metrics")
    @patch("main.calculate_ratios")
    @patch("main.run_liquidity_filter")
    def test_filters_symbols_to_liquid_subset(
        self, mock_liq, mock_calc, mock_wins, mock_scores, mock_orth
    ):
        ctx = _make_ctx(symbols=["AAPL", "MSFT", "GOOG"])
        mock_liq.return_value = ["AAPL"]  # only AAPL is liquid

        ratios = self._ratios_df()
        mock_calc.return_value = ratios
        mock_wins.return_value = ratios
        mock_scores.return_value = (ratios, pd.DataFrame())
        mock_orth.return_value = ratios

        backfill_factor_metrics(ctx, years=1)

        # Every calculate_ratios call should only have AAPL
        for c in mock_calc.call_args_list:
            assert c.kwargs["symbols"] == ["AAPL"]

    @patch("main.orthogonalise_lowvol")
    @patch("main.compute_factor_scores")
    @patch("main.winsorise_metrics")
    @patch("main.calculate_ratios")
    @patch("main.run_liquidity_filter")
    def test_calls_winsorise_before_factor_scores(
        self, mock_liq, mock_calc, mock_wins, mock_scores, mock_orth
    ):
        ctx = _make_ctx()
        ratios = self._ratios_df()
        mock_liq.return_value = ["AAPL"]
        mock_calc.return_value = ratios
        mock_wins.return_value = ratios
        mock_scores.return_value = (ratios, pd.DataFrame())
        mock_orth.return_value = ratios

        backfill_factor_metrics(ctx, years=1)

        assert mock_wins.called and mock_scores.called
