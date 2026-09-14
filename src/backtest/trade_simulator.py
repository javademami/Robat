from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


# ============================================================
# TRADE DATA MODEL
# ============================================================

@dataclass
class Trade:
    # ----- Entry info (from PositionSize) -----
    entry_timestamp: Any
    direction: str

    entry_price: float
    stop_loss: float
    take_profit: float

    position_size: float
    risk_amount: float

    risk_per_unit: float = 0.0
    reward_per_unit: float = 0.0
    risk_reward: float = 0.0

    notional_value: float = 0.0
    margin_required: float = 0.0
    leverage: float = 1.0

    # ----- Metadata -----
    zone_type: Optional[str] = None
    score: Optional[float] = None
    grade: Optional[str] = None

    # ----- Exit info (filled by simulator) -----
    exit_timestamp: Any = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None       # "SL" | "TP" | "EOD" | "NoData"
    bars_held: Optional[int] = None

    # ----- Result -----
    r_multiple: Optional[float] = None
    pnl: Optional[float] = None
    outcome: Optional[str] = None           # "win" | "loss" | "open"


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_direction(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw).strip().lower()

    if "." in text:
        text = text.split(".")[-1]

    if text in {"bullish", "long", "buy"}:
        return "bullish"

    if text in {"bearish", "short", "sell"}:
        return "bearish"

    return text


def _to_utc_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None

    try:
        ts = pd.Timestamp(value)

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        return ts

    except (TypeError, ValueError):
        return None


# ============================================================
# BUILD TRADE FROM POSITION SIZE
# ============================================================

def build_trade_from_position_size(position_size: Any) -> Trade:
    """
    Convert a PositionSize dataclass into a Trade (entry-only).
    """
    return Trade(
        entry_timestamp=getattr(position_size, "timestamp", None),
        direction=_normalize_direction(
            getattr(position_size, "direction", "")
        ),
        entry_price=float(getattr(position_size, "entry_price", 0.0)),
        stop_loss=float(getattr(position_size, "stop_loss", 0.0)),
        take_profit=float(getattr(position_size, "take_profit", 0.0)),
        position_size=float(getattr(position_size, "position_size", 0.0)),
        risk_amount=float(getattr(position_size, "risk_amount", 0.0)),
        risk_per_unit=float(getattr(position_size, "risk_per_unit", 0.0)),
        reward_per_unit=float(getattr(position_size, "reward_per_unit", 0.0)),
        risk_reward=float(getattr(position_size, "risk_reward", 0.0)),
        notional_value=float(getattr(position_size, "notional_value", 0.0)),
        margin_required=float(getattr(position_size, "margin_required", 0.0)),
        leverage=float(getattr(position_size, "leverage", 1.0)),
        zone_type=getattr(position_size, "zone_type", None),
        score=getattr(position_size, "score", None),
        grade=getattr(position_size, "grade", None),
    )


# ============================================================
# SIMULATE ONE TRADE
# ============================================================

def simulate_trade(
    trade: Trade,
    df_5m: pd.DataFrame,
    max_bars: int = 2000,
    timestamp_column: str = "timestamp",
) -> Trade:
    """
    Simulate a single trade against future 5M candles.

    Rules:
    - Only candles with timestamp strictly > entry_timestamp are used.
    - If a candle's range touches both SL and TP, SL wins (conservative).
    - If neither is touched within `max_bars`, trade is marked "open"
      at the last available close.
    """

    entry_ts = _to_utc_timestamp(trade.entry_timestamp)

    if entry_ts is None:
        trade.exit_reason = "NoData"
        trade.outcome = "open"
        return trade

    # --------------------------------------------------------
    # Prepare future candles
    # --------------------------------------------------------

    df = df_5m

    if timestamp_column in df.columns:
        ts_series = pd.to_datetime(
            df[timestamp_column],
            utc=True,
        )
    else:
        ts_series = pd.to_datetime(
            df.index,
            utc=True,
        )

    future_mask = ts_series > entry_ts

    if not future_mask.any():
        trade.exit_reason = "NoData"
        trade.outcome = "open"
        return trade

    future = df.loc[future_mask].head(max_bars).copy()
    future_ts = ts_series[future_mask].head(max_bars).reset_index(drop=True)
    future = future.reset_index(drop=True)

    if future.empty:
        trade.exit_reason = "NoData"
        trade.outcome = "open"
        return trade

    direction = _normalize_direction(trade.direction)
    entry = trade.entry_price
    sl = trade.stop_loss
    tp = trade.take_profit

    risk_per_unit = abs(entry - sl)

    if risk_per_unit <= 0:
        trade.exit_reason = "NoData"
        trade.outcome = "open"
        return trade

    # --------------------------------------------------------
    # Walk forward candle by candle
    # --------------------------------------------------------

    for i in range(len(future)):

        row = future.iloc[i]
        candle_ts = future_ts.iloc[i]

        high = float(row["high"])
        low = float(row["low"])

        # ----- BULLISH -----
        if direction == "bullish":

            # Conservative: SL first if both are touched
            if low <= sl:
                trade.exit_timestamp = candle_ts
                trade.exit_price = sl
                trade.exit_reason = "SL"
                trade.bars_held = i + 1
                trade.r_multiple = -1.0
                trade.pnl = -trade.risk_amount
                trade.outcome = "loss"
                return trade

            if high >= tp:
                trade.exit_timestamp = candle_ts
                trade.exit_price = tp
                trade.exit_reason = "TP"
                trade.bars_held = i + 1
                trade.r_multiple = trade.risk_reward
                trade.pnl = trade.risk_amount * trade.risk_reward
                trade.outcome = "win"
                return trade

        # ----- BEARISH -----
        elif direction == "bearish":

            # Conservative: SL first if both are touched
            if high >= sl:
                trade.exit_timestamp = candle_ts
                trade.exit_price = sl
                trade.exit_reason = "SL"
                trade.bars_held = i + 1
                trade.r_multiple = -1.0
                trade.pnl = -trade.risk_amount
                trade.outcome = "loss"
                return trade

            if low <= tp:
                trade.exit_timestamp = candle_ts
                trade.exit_price = tp
                trade.exit_reason = "TP"
                trade.bars_held = i + 1
                trade.r_multiple = trade.risk_reward
                trade.pnl = trade.risk_amount * trade.risk_reward
                trade.outcome = "win"
                return trade

        else:
            # Unknown direction
            trade.exit_reason = "NoData"
            trade.outcome = "open"
            return trade

    # --------------------------------------------------------
    # End of future data — still open
    # --------------------------------------------------------

    last = future.iloc[-1]
    last_ts = future_ts.iloc[-1]
    last_close = float(last["close"])

    trade.exit_timestamp = last_ts
    trade.exit_price = last_close
    trade.exit_reason = "EOD"
    trade.bars_held = len(future)
    trade.outcome = "open"

    # Compute R-multiple at current price
    if direction == "bullish":
        r = (last_close - entry) / risk_per_unit
    else:
        r = (entry - last_close) / risk_per_unit

    trade.r_multiple = r
    trade.pnl = r * trade.risk_amount

    return trade


# ============================================================
# SIMULATE MANY
# ============================================================

def simulate_trades(
    trades: list[Trade],
    df_5m: pd.DataFrame,
    max_bars: int = 2000,
) -> list[Trade]:
    """
    Simulate a list of trades. Returns the same list, mutated.
    """

    results: list[Trade] = []

    for trade in trades:
        result = simulate_trade(
            trade=trade,
            df_5m=df_5m,
            max_bars=max_bars,
        )
        results.append(result)

    return results