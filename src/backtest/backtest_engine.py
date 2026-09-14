from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import pandas as pd

from src.backtest.trade_simulator import (
    Trade,
    build_trade_from_position_size,
    simulate_trade,
)

from src.backtest.metrics import calculate_metrics, print_metrics


# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest(
    position_sizes: Iterable[Any],
    df_5m: pd.DataFrame,
    max_bars: int = 2000,
) -> list[Trade]:
    """
    Convert PositionSize objects into Trades and simulate them.
    """

    trades: list[Trade] = []

    for ps in position_sizes:

        trade = build_trade_from_position_size(ps)
        trade = simulate_trade(
            trade=trade,
            df_5m=df_5m,
            max_bars=max_bars,
        )

        trades.append(trade)

    return trades


# ============================================================
# EXPORT
# ============================================================

def trades_to_dataframe(trades: Iterable[Trade]) -> pd.DataFrame:
    rows = [asdict(t) for t in trades]

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


__all__ = [
    "run_backtest",
    "trades_to_dataframe",
    "calculate_metrics",
    "print_metrics",
]