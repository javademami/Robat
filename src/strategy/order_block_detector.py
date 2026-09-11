from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .displacement_detector import DisplacementEvent


@dataclass(frozen=True)
class OrderBlockEvent:
    timestamp: pd.Timestamp
    direction: str

    displacement_timestamp: pd.Timestamp
    mss_timestamp: pd.Timestamp

    candle_index: int

    open: float
    high: float
    low: float
    close: float

    zone_high: float
    zone_low: float

    bars_before_displacement: int


def detect_order_blocks(
    df: pd.DataFrame,
    displacements: List[DisplacementEvent],
    lookback: int = 8,
) -> List[OrderBlockEvent]:
    """
    Detect the last opposite-direction candle before displacement.

    Bullish displacement:
        last bearish candle before displacement.

    Bearish displacement:
        last bullish candle before displacement.

    The OB zone is the full candle range [low, high].

    No lookahead:
        only candles strictly before the displacement candle are inspected.
    """

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values("timestamp").reset_index(drop=True)

    results: List[OrderBlockEvent] = []

    for displacement in displacements:
        displacement_ts = pd.Timestamp(displacement.timestamp)

        matches = work.index[work["timestamp"] == displacement_ts]
        if len(matches) == 0:
            continue

        displacement_index = int(matches[0])

        start = max(0, displacement_index - lookback)

        candidates = work.iloc[start:displacement_index]

        if displacement.direction == "bullish":
            # Last bearish candle.
            candidates = candidates[
                candidates["close"] < candidates["open"]
            ]

        elif displacement.direction == "bearish":
            # Last bullish candle.
            candidates = candidates[
                candidates["close"] > candidates["open"]
            ]

        else:
            continue

        if candidates.empty:
            continue

        ob = candidates.iloc[-1]

        results.append(
            OrderBlockEvent(
                timestamp=pd.Timestamp(ob["timestamp"]),
                direction=displacement.direction,
                displacement_timestamp=displacement_ts,
                mss_timestamp=pd.Timestamp(displacement.mss_timestamp),
                candle_index=int(ob.name),
                open=float(ob["open"]),
                high=float(ob["high"]),
                low=float(ob["low"]),
                close=float(ob["close"]),
                zone_high=float(ob["high"]),
                zone_low=float(ob["low"]),
                bars_before_displacement=(
                    displacement_index - int(ob.name)
                ),
            )
        )

    return deduplicate_order_blocks(results)


def deduplicate_order_blocks(
    events: List[OrderBlockEvent],
) -> List[OrderBlockEvent]:
    seen = set()
    result = []

    for event in events:
        key = (
            event.timestamp,
            event.direction,
            event.displacement_timestamp,
            round(event.zone_high, 8),
            round(event.zone_low, 8),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(event)

    return sorted(
        result,
        key=lambda x: x.displacement_timestamp,
    )


def order_blocks_to_dataframe(
    events: List[OrderBlockEvent],
) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "timestamp": e.timestamp,
                "direction": e.direction,
                "displacement_timestamp": e.displacement_timestamp,
                "mss_timestamp": e.mss_timestamp,
                "candle_index": e.candle_index,
                "open": e.open,
                "high": e.high,
                "low": e.low,
                "close": e.close,
                "zone_high": e.zone_high,
                "zone_low": e.zone_low,
                "bars_before_displacement": e.bars_before_displacement,
            }
            for e in events
        ]
    )