from __future__ import annotations

from statistics import mean, median
from typing import Any


# ============================================================
# HELPERS
# ============================================================

def _hold_hours(trade: Any) -> float | None:
    entry = getattr(trade, "entry_timestamp", None)
    exit_ = getattr(trade, "exit_timestamp", None)

    if entry is None or exit_ is None:
        return None

    try:
        import pandas as pd

        e = pd.Timestamp(entry)
        x = pd.Timestamp(exit_)

        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        if x.tzinfo is None:
            x = x.tz_localize("UTC")

        delta = x - e
        return delta.total_seconds() / 3600.0

    except Exception:
        return None


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades: list) -> dict:
    """
    Compute aggregate metrics from a list of Trade objects.

    Closed trades: outcome in {"win", "loss"}.
    Open trades:   outcome == "open" (excluded from PnL/R stats).
    NoData:        excluded entirely.
    """

    total = len(trades)

    no_data = [
        t for t in trades
        if getattr(t, "outcome", None) is None
        or getattr(t, "exit_reason", None) == "NoData"
    ]

    closed = [
        t for t in trades
        if getattr(t, "outcome", None) in {"win", "loss"}
    ]

    open_trades = [
        t for t in trades
        if getattr(t, "outcome", None) == "open"
    ]

    if not closed:
        return {
            "total_trades": total,
            "closed_trades": 0,
            "open_trades": len(open_trades),
            "no_data_trades": len(no_data),
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "median_r": 0.0,
            "total_r": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown_r": 0.0,
            "avg_hold_hours": 0.0,
            "median_hold_hours": 0.0,
        }

    wins = [t for t in closed if t.outcome == "win"]
    losses = [t for t in closed if t.outcome == "loss"]

    r_multiples = [
        float(t.r_multiple)
        for t in closed
        if t.r_multiple is not None
    ]

    pnls = [
        float(t.pnl)
        for t in closed
        if t.pnl is not None
    ]

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    # Expectancy per trade in R
    expectancy = mean(r_multiples) if r_multiples else 0.0

    # Max drawdown in cumulative R
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    for r in r_multiples:
        cumulative += r
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    # Hold time
    hold_hours = [
        h for h in (_hold_hours(t) for t in closed)
        if h is not None
    ]

    return {
        "total_trades": total,
        "closed_trades": len(closed),
        "open_trades": len(open_trades),
        "no_data_trades": len(no_data),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(closed),
        "avg_r": mean(r_multiples) if r_multiples else 0.0,
        "median_r": median(r_multiples) if r_multiples else 0.0,
        "total_r": sum(r_multiples) if r_multiples else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown_r": max_dd,
        "avg_hold_hours": mean(hold_hours) if hold_hours else 0.0,
        "median_hold_hours": median(hold_hours) if hold_hours else 0.0,
    }


def print_metrics(metrics: dict) -> None:
    print()
    print("=" * 72)
    print("BACKTEST METRICS")
    print("=" * 72)

    print(f"Total trades:      {metrics['total_trades']}")
    print(f"Closed trades:     {metrics['closed_trades']}")
    print(f"Open trades:       {metrics['open_trades']}")
    print(f"NoData trades:     {metrics['no_data_trades']}")
    print()

    print(f"Wins:              {metrics['wins']}")
    print(f"Losses:            {metrics['losses']}")
    print(f"Win rate:          {metrics['win_rate']:.2%}")
    print()

    print(f"Avg R:             {metrics['avg_r']:+.4f}")
    print(f"Median R:          {metrics['median_r']:+.4f}")
    print(f"Total R:           {metrics['total_r']:+.4f}")
    print()

    print(f"Gross profit:      ${metrics['gross_profit']:,.2f}")
    print(f"Gross loss:        ${metrics['gross_loss']:,.2f}")
    print(f"Profit factor:     {metrics['profit_factor']:.4f}")
    print(f"Expectancy (R):    {metrics['expectancy']:+.4f}")
    print()

    print(f"Max drawdown (R):  {metrics['max_drawdown_r']:.4f}")
    print()

    print(f"Avg hold (hours):  {metrics['avg_hold_hours']:.2f}")
    print(f"Median hold:       {metrics['median_hold_hours']:.2f}")