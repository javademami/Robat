from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.sweep_detector import SweepDirection
from src.strategy.mss_detector import MSSEvent


@dataclass(frozen=True)
class DisplacementEvent:
    timestamp: pd.Timestamp
    direction: SweepDirection

    mss_timestamp: pd.Timestamp
    bars_after_mss: int

    open: float
    high: float
    low: float
    close: float

    body: float
    atr: float
    body_atr: float
    clv: float

    candle_index: int


def _normalize_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder-style ATR using causal rolling data.

    TR:
        max(
            high-low,
            abs(high-prev_close),
            abs(low-prev_close)
        )
    """

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(
        window=period,
        min_periods=period,
    ).mean()


def _calculate_clv(
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> float:
    """
    Close Location Value normalized to [0, 1].

    1.0 = close at high
    0.0 = close at low
    """

    candle_range = high - low

    if candle_range <= 0:
        return 0.5

    return (close - low) / candle_range


def detect_displacements(
    df: pd.DataFrame,
    mss_events: list[MSSEvent],
    atr_period: int = 14,
    min_body_atr: float = 1.0,
    bullish_clv: float = 0.70,
    bearish_clv: float = 0.30,
    max_bars_after_mss: int = 8,
) -> list[DisplacementEvent]:
    """
    Detect the first valid displacement after each MSS.

    Main conditions:

    Bullish:
        close > open
        body / ATR >= min_body_atr
        CLV >= bullish_clv

    Bearish:
        close < open
        body / ATR >= min_body_atr
        CLV <= bearish_clv

    Important:
        The MSS candle itself is NOT tested.
        Search starts from the next candle.

    This prevents the MSS candle from automatically becoming
    a displacement signal.
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
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty or not mss_events:
        return []

    if atr_period <= 0:
        raise ValueError("atr_period must be > 0")

    if min_body_atr <= 0:
        raise ValueError("min_body_atr must be > 0")

    if not 0 < bearish_clv < bullish_clv < 1:
        raise ValueError(
            "CLV thresholds must satisfy "
            "0 < bearish_clv < bullish_clv < 1"
        )

    if max_bars_after_mss <= 0:
        raise ValueError(
            "max_bars_after_mss must be > 0"
        )

    work = df.copy()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        utc=True,
    )

    work = (
        work
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in numeric_columns:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work["atr"] = _calculate_atr(
        work,
        period=atr_period,
    )

    timestamps = work["timestamp"].tolist()

    time_to_index = {
        ts: index
        for index, ts in enumerate(timestamps)
    }

    results: list[DisplacementEvent] = []

    ordered_mss = sorted(
        mss_events,
        key=lambda event: _normalize_timestamp(
            event.timestamp
        ),
    )

    for mss in ordered_mss:

        mss_time = _normalize_timestamp(
            mss.timestamp
        )

        if mss_time not in time_to_index:
            continue

        mss_index = time_to_index[mss_time]

        # IMPORTANT:
        # Start from the candle AFTER MSS.
        start_index = mss_index + 1

        end_index = min(
            mss_index + max_bars_after_mss + 1,
            len(work),
        )

        for candle_index in range(
            start_index,
            end_index,
        ):

            row = work.iloc[candle_index]

            open_price = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            atr = row["atr"]

            if pd.isna(atr):
                continue

            atr = float(atr)

            if atr <= 0:
                continue

            body = abs(close - open_price)

            body_atr = body / atr

            # First filter: body strength
            if body_atr < min_body_atr:
                continue

            clv = _calculate_clv(
                open_price=open_price,
                high=high,
                low=low,
                close=close,
            )

            bars_after_mss = (
                candle_index - mss_index
            )

            # -------------------------------------------------
            # BULLISH DISPLACEMENT
            # -------------------------------------------------

            if mss.direction == SweepDirection.BULLISH:

                # Candle must be bullish
                if close <= open_price:
                    continue

                # Strong close near high
                if clv < bullish_clv:
                    continue

                results.append(
                    DisplacementEvent(
                        timestamp=row["timestamp"],
                        direction=SweepDirection.BULLISH,

                        mss_timestamp=mss_time,
                        bars_after_mss=bars_after_mss,

                        open=open_price,
                        high=high,
                        low=low,
                        close=close,

                        body=body,
                        atr=atr,
                        body_atr=body_atr,
                        clv=clv,

                        candle_index=candle_index,
                    )
                )

                # Only FIRST valid displacement
                break

            # -------------------------------------------------
            # BEARISH DISPLACEMENT
            # -------------------------------------------------

            elif mss.direction == SweepDirection.BEARISH:

                # Candle must be bearish
                if close >= open_price:
                    continue

                # Strong close near low
                if clv > bearish_clv:
                    continue

                results.append(
                    DisplacementEvent(
                        timestamp=row["timestamp"],
                        direction=SweepDirection.BEARISH,

                        mss_timestamp=mss_time,
                        bars_after_mss=bars_after_mss,

                        open=open_price,
                        high=high,
                        low=low,
                        close=close,

                        body=body,
                        atr=atr,
                        body_atr=body_atr,
                        clv=clv,

                        candle_index=candle_index,
                    )
                )

                # Only FIRST valid displacement
                break

    return deduplicate_displacements(results)


def deduplicate_displacements(
    events: list[DisplacementEvent],
) -> list[DisplacementEvent]:

    if not events:
        return []

    ordered = sorted(
        events,
        key=lambda event: (
            _normalize_timestamp(event.timestamp),
            event.direction.value,
        ),
    )

    result: list[DisplacementEvent] = []

    seen = set()

    for event in ordered:

        key = (
            _normalize_timestamp(event.timestamp),
            event.direction,
            _normalize_timestamp(
                event.mss_timestamp
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(event)

    return result


def get_latest_displacement(
    events: list[DisplacementEvent],
    current_time: Optional[pd.Timestamp] = None,
) -> Optional[DisplacementEvent]:

    if not events:
        return None

    if current_time is None:
        return events[-1]

    current_time = _normalize_timestamp(
        current_time
    )

    available = [
        event
        for event in events
        if _normalize_timestamp(
            event.timestamp
        ) <= current_time
    ]

    if not available:
        return None

    return max(
        available,
        key=lambda event: _normalize_timestamp(
            event.timestamp
        ),
    )


def displacements_to_dataframe(
    events: list[DisplacementEvent],
) -> pd.DataFrame:

    columns = [
        "timestamp",
        "direction",
        "mss_timestamp",
        "bars_after_mss",
        "open",
        "high",
        "low",
        "close",
        "body",
        "atr",
        "body_atr",
        "clv",
        "candle_index",
    ]

    if not events:
        return pd.DataFrame(
            columns=columns
        )

    return pd.DataFrame(
        [
            {
                "timestamp": event.timestamp,
                "direction": (
                    event.direction.value
                    if hasattr(
                        event.direction,
                        "value",
                    )
                    else str(event.direction)
                ),
                "mss_timestamp": (
                    event.mss_timestamp
                ),
                "bars_after_mss": (
                    event.bars_after_mss
                ),
                "open": event.open,
                "high": event.high,
                "low": event.low,
                "close": event.close,
                "body": event.body,
                "atr": event.atr,
                "body_atr": event.body_atr,
                "clv": event.clv,
                "candle_index": (
                    event.candle_index
                ),
            }
            for event in events
        ]
    )