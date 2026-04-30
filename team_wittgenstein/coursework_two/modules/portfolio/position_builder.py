"""Steps 4-7: Portfolio weight construction for the 130/30 strategy.

Step 4 normalises risk-adjusted scores within each sector and direction.
Target weights follow ``risk_adj_score / sector_group_sum * sector_budget``.
The long budget per sector is ``1.3 / K_long`` and the short budget per sector
is ``0.3 / K_short``, where ``K`` is the number of active sectors in that
direction.

Step 5 applies a liquidity cap. If ``weight * AUM`` exceeds
``liquidity_cap_pct * 20-day ADV``, the position is capped and excess weight is
redistributed pro-rata to uncapped names in the same sector and direction.

Step 6 applies the no-trade zone. If
``abs(target_weight - prev_final_weight) < threshold``, the prior weight is
held. Otherwise the new target weight is traded. New entries always trade.

Step 7 verifies the portfolio constraints. Long gross should be about 1.3,
short gross about 0.3, and net exposure about 1.0. Violations are logged as
warnings rather than raised as exceptions.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from modules.db.db_connection import PostgresConnection

logger = logging.getLogger(__name__)
SCHEMA = "team_wittgenstein"

LONG_BUDGET = 1.3
SHORT_BUDGET = 0.3
MAX_REDISTRIBUTE_ITERS = 10


@dataclass(frozen=True)
class PositionConfig:
    """Parameters for portfolio construction steps 4-7."""

    aum: float = 1_000_000_000.0  # $1 B AUM
    liquidity_cap_pct: float = 0.05  # max 5 % of 20-day ADV per position
    no_trade_threshold: float = 0.01  # 1.0 % weight deviation triggers a trade
    adv_lookback_days: int = 20
    constraint_tolerance: float = 0.02  # 2 % slack in constraint checks


# ── DB helpers ────────────────────────────────────────────────────────────────


def fetch_adv(
    db: PostgresConnection,
    symbols: list,
    rebalance_date: date,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Fetch 20-day average dollar volume (adjusted_close * volume) per symbol.

    Returns a DataFrame with columns ['symbol', 'adv_20d'].
    Symbols with insufficient history are omitted.
    """
    start_date = rebalance_date - timedelta(days=lookback_days * 3)
    query = """
        SELECT symbol, trade_date, adjusted_close * volume AS dollar_vol
        FROM team_wittgenstein.price_data
        WHERE symbol    IN :symbols
          AND trade_date <  :rebalance_date
          AND trade_date >= :start_date
          AND adjusted_close IS NOT NULL
          AND volume IS NOT NULL
          AND volume > 0
        ORDER BY symbol, trade_date
    """
    prices = db.read_query(
        query,
        {
            "symbols": tuple(symbols),
            "rebalance_date": rebalance_date,
            "start_date": start_date,
        },
    )
    if prices is None or prices.empty:
        return pd.DataFrame(columns=["symbol", "adv_20d"])

    # Rolling mean, keep last row per symbol
    prices = prices.sort_values(["symbol", "trade_date"])
    prices["adtv"] = prices.groupby("symbol")["dollar_vol"].transform(
        lambda s: s.rolling(lookback_days, min_periods=lookback_days).mean()
    )
    result = (
        prices.groupby("symbol")
        .tail(1)[["symbol", "adtv"]]
        .rename(columns={"adtv": "adv_20d"})
        .dropna(subset=["adv_20d"])
        .reset_index(drop=True)
    )
    return result


def fetch_previous_weights(
    db: PostgresConnection,
    rebalance_date: date,
) -> pd.DataFrame:
    """Fetch final weights from the most recent portfolio_positions before this date.

    Returns a DataFrame with columns ['symbol', 'direction', 'final_weight'].
    Returns an empty DataFrame when no prior positions exist.
    """
    query = """
        SELECT symbol, direction, final_weight
        FROM team_wittgenstein.portfolio_positions
        WHERE rebalance_date = (
            SELECT MAX(rebalance_date)
            FROM  team_wittgenstein.portfolio_positions
            WHERE rebalance_date < :rebalance_date
        )
    """
    result = db.read_query(query, {"rebalance_date": rebalance_date})
    if result is None or result.empty:
        return pd.DataFrame(columns=["symbol", "direction", "final_weight"])
    return result


# ── Step 4 ────────────────────────────────────────────────────────────────────


def compute_sector_weights(scored: pd.DataFrame) -> pd.DataFrame:
    """Normalise risk-adjusted scores into target portfolio weights (Step 4).

    Within each ``(sector, direction)`` group the weight is proportional to
    ``risk_adj_score`` and scaled to the per-sector budget. Long groups use
    ``LONG_BUDGET / K_long`` and short groups use ``SHORT_BUDGET / K_short``,
    where ``K`` is the number of active sectors on that side.

    Args:
        scored: DataFrame with columns symbol, sector, direction,
                risk_adj_score (and any others to pass through).

    Returns:
        Input DataFrame with an added 'target_weight' column.
    """
    df = scored.copy()

    k_long = df.loc[df["direction"] == "long", "sector"].nunique()
    k_short = df.loc[df["direction"] == "short", "sector"].nunique()

    long_budget_per_sector = LONG_BUDGET / k_long if k_long > 0 else 0.0
    short_budget_per_sector = SHORT_BUDGET / k_short if k_short > 0 else 0.0

    weights = {}
    for (sector, direction), group in df.groupby(["sector", "direction"]):
        budget = (
            long_budget_per_sector if direction == "long" else short_budget_per_sector
        )
        total_score = group["risk_adj_score"].sum()
        if total_score <= 0:
            # Fallback: equal weight within group
            w = pd.Series(budget / len(group), index=group.index)
        else:
            w = (group["risk_adj_score"] / total_score) * budget
        weights.update(w.to_dict())

    df["target_weight"] = df.index.map(weights)

    logger.info(
        "Step 4: %d long positions (Σ=%.4f), %d short positions (Σ=%.4f)",
        (df["direction"] == "long").sum(),
        df.loc[df["direction"] == "long", "target_weight"].sum(),
        (df["direction"] == "short").sum(),
        df.loc[df["direction"] == "short", "target_weight"].sum(),
    )
    return df


# ── Step 5 ────────────────────────────────────────────────────────────────────


def apply_liquidity_cap(
    df: pd.DataFrame,
    adv: pd.DataFrame,
    aum: float,
    cap_pct: float,
) -> pd.DataFrame:
    """Cap positions exceeding cap_pct of 20-day ADV and redistribute (Step 5).

    The cap is applied iteratively (up to MAX_REDISTRIBUTE_ITERS passes) so
    that redistributed weight does not push another stock over its cap.

    Args:
        df: DataFrame with symbol, sector, direction, target_weight.
        adv: DataFrame with symbol, adv_20d.
        aum: Portfolio AUM in dollars.
        cap_pct: Maximum fraction of ADV allowed per position.

    Returns:
        DataFrame with updated target_weight and a boolean liquidity_capped flag.
    """
    df = df.merge(adv[["symbol", "adv_20d"]], on="symbol", how="left")
    df["liquidity_capped"] = False

    # Cap weight = cap_pct * ADV / AUM; infinity where ADV is unknown
    df["_cap_w"] = df["adv_20d"].apply(
        lambda v: (
            (cap_pct * float(v) / aum)
            if (v is not None and pd.notna(v) and float(v) > 0)
            else float("inf")
        )
    )

    for _ in range(MAX_REDISTRIBUTE_ITERS):
        over = df["target_weight"] > df["_cap_w"]
        if not over.any():
            break

        df.loc[over, "liquidity_capped"] = True

        for (sector, direction), grp_idx in df.groupby(
            ["sector", "direction"]
        ).groups.items():
            grp = df.loc[grp_idx]
            over_grp = grp[grp["target_weight"] > grp["_cap_w"]]
            if over_grp.empty:
                continue

            excess = 0.0
            for idx in over_grp.index:
                excess += df.loc[idx, "target_weight"] - df.loc[idx, "_cap_w"]
                df.loc[idx, "target_weight"] = df.loc[idx, "_cap_w"]

            uncapped = grp[grp["target_weight"] < grp["_cap_w"]]
            if uncapped.empty:
                logger.warning(
                    "All stocks capped in sector=%s dir=%s; " "%.4f excess weight lost",
                    sector,
                    direction,
                    excess,
                )
                continue

            uncapped_sum = uncapped["target_weight"].sum()
            if uncapped_sum > 0:
                for idx in uncapped.index:
                    df.loc[idx, "target_weight"] += (
                        excess * df.loc[idx, "target_weight"] / uncapped_sum
                    )

    n_capped = df["liquidity_capped"].sum()
    if n_capped:
        logger.info("Step 5: liquidity-capped %d positions", n_capped)
    else:
        logger.info("Step 5: no positions required liquidity capping")

    return df.drop(columns=["adv_20d", "_cap_w"])


# ── Step 6 ────────────────────────────────────────────────────────────────────


def apply_no_trade_zone(
    df: pd.DataFrame,
    previous: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Suppress small rebalancing trades to reduce unnecessary turnover (Step 6).

    Rules:
    - New position not in ``previous``: trade to ``target_weight``
    - Direction change from previous month: trade to ``target_weight``
    - ``abs(target - prev) < threshold``: hold ``prev_weight``
    - ``abs(target - prev) >= threshold``: trade to ``target_weight``

    Args:
        df: DataFrame with symbol, direction, target_weight.
        previous: Prior-month DataFrame with symbol, direction, final_weight.
        threshold: Minimum absolute weight deviation that triggers a trade.

    Returns:
        Input DataFrame with 'final_weight' and 'trade_action' columns added.
    """
    # Build lookup: symbol → (direction, final_weight)
    prev_map: dict[str, tuple[str, float]] = {}
    if not previous.empty:
        for _, row in previous.iterrows():
            prev_map[row["symbol"]] = (
                str(row["direction"]),
                float(row["final_weight"]),
            )

    final_weights = []
    trade_actions = []

    for _, row in df.iterrows():
        sym = row["symbol"]
        target_w = float(row["target_weight"])
        cur_dir = str(row["direction"])

        if sym not in prev_map:
            # Brand-new position
            final_weights.append(target_w)
            trade_actions.append("trade")
        else:
            prev_dir, prev_w = prev_map[sym]
            if prev_dir != cur_dir:
                # Direction flip — always trade
                final_weights.append(target_w)
                trade_actions.append("trade")
            elif abs(target_w - prev_w) < threshold:
                final_weights.append(prev_w)
                trade_actions.append("hold")
            else:
                final_weights.append(target_w)
                trade_actions.append("trade")

    df = df.copy()
    df["final_weight"] = final_weights
    df["trade_action"] = trade_actions

    # Budget redistribution: for each (sector, direction) group, the held
    # stocks lock in their old weights; traded stocks share the remaining
    # sector budget proportionally to their target weights.
    for (sector, direction), grp_idx in df.groupby(
        ["sector", "direction"]
    ).groups.items():
        grp = df.loc[grp_idx]
        sector_budget = grp["target_weight"].sum()  # pre-hold total = correct budget

        held = grp[grp["trade_action"] == "hold"]
        traded = grp[grp["trade_action"] == "trade"]

        locked = held["final_weight"].sum()
        remaining = sector_budget - locked

        if traded.empty:
            continue

        traded_target_sum = traded["target_weight"].sum()
        if traded_target_sum <= 0:
            # Fallback: equal share of remaining budget
            for idx in traded.index:
                df.loc[idx, "final_weight"] = remaining / len(traded)
        else:
            for idx in traded.index:
                df.loc[idx, "final_weight"] = (
                    remaining * df.loc[idx, "target_weight"] / traded_target_sum
                )

    n_hold = trade_actions.count("hold")
    n_trade = trade_actions.count("trade")
    logger.info(
        "Step 6: %d trades, %d holds (threshold=%.3f)", n_trade, n_hold, threshold
    )
    return df


# ── Step 7 ────────────────────────────────────────────────────────────────────


def verify_constraints(
    df: pd.DataFrame,
    tolerance: float = 0.02,
) -> bool:
    """Verify 130/30 portfolio constraints (Step 7).

    Checks:
        Σ long  final_weights  ≈ 1.3
        Σ short final_weights  ≈ 0.3  (weights stored positive, direction='short')
        net exposure           ≈ 1.0

    Args:
        df: Portfolio positions with direction and final_weight.
        tolerance: Allowed absolute deviation from each target.

    Returns:
        True when all three constraints pass; False otherwise.
    """
    if df.empty:
        logger.warning("Constraint check: no positions — skipping")
        return False

    long_sum = df.loc[df["direction"] == "long", "final_weight"].sum()
    short_sum = df.loc[df["direction"] == "short", "final_weight"].sum()
    net = long_sum - short_sum

    passed = True

    def _check(label: str, actual: float, target: float) -> bool:
        if abs(actual - target) > tolerance:
            logger.warning(
                "Constraint FAIL  %-20s actual=%.4f  target=%.1f  tol=±%.3f",
                label,
                actual,
                target,
                tolerance,
            )
            return False
        logger.info(
            "Constraint OK    %-20s actual=%.4f  target=%.1f",
            label,
            actual,
            target,
        )
        return True

    passed &= _check("Σ long weights", long_sum, LONG_BUDGET)
    passed &= _check("Σ short weights", short_sum, SHORT_BUDGET)
    passed &= _check("net exposure", net, 1.0)

    return passed


# ── Orchestrator ──────────────────────────────────────────────────────────────


def build_portfolio_positions(
    db: PostgresConnection,
    scored: pd.DataFrame,
    rebalance_date: date,
    config: PositionConfig,
    prior_positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run Steps 4-7 and return the final portfolio positions DataFrame.

    Args:
        db:              Active database connection (used for ADV lookup and,
                         when prior_positions is None, previous-weights fetch).
        scored:          Output of compute_risk_adjusted_scores - must contain
                         symbol, sector, direction, composite_score, ewma_vol,
                         risk_adj_score.
        rebalance_date:  Month-end rebalancing date.
        config:          PositionConfig instance.
        prior_positions: If provided, use this DataFrame as the previous month's
                         positions for the no-trade zone (columns: symbol,
                         direction, final_weight). Required for variant scenarios
                         (e.g. factor exclusion) so no-trade zone uses the
                         variant's own history rather than baseline.

    Returns:
        DataFrame with rebalance_date, symbol, sector, direction, ewma_vol,
        risk_adj_score, target_weight, final_weight, liquidity_capped,
        trade_action.  Ready for insertion into portfolio_positions.
        Returns an empty DataFrame when scored is empty.
    """
    if scored is None or scored.empty:
        logger.warning("No risk-adjusted scores for %s — skipping", rebalance_date)
        return pd.DataFrame()

    # Step 4
    df = compute_sector_weights(scored)

    # Step 5
    symbols = df["symbol"].tolist()
    adv = fetch_adv(db, symbols, rebalance_date, config.adv_lookback_days)
    df = apply_liquidity_cap(df, adv, config.aum, config.liquidity_cap_pct)

    # Step 6 - use prior_positions override if given, else fetch from DB
    previous = (
        prior_positions
        if prior_positions is not None
        else fetch_previous_weights(db, rebalance_date)
    )
    df = apply_no_trade_zone(df, previous, config.no_trade_threshold)

    # Step 7
    verify_constraints(df, config.constraint_tolerance)

    out_cols = [
        "symbol",
        "sector",
        "direction",
        "ewma_vol",
        "risk_adj_score",
        "target_weight",
        "final_weight",
        "liquidity_capped",
        "trade_action",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df = df.copy()
    df["rebalance_date"] = rebalance_date
    return df[["rebalance_date"] + out_cols].reset_index(drop=True)
