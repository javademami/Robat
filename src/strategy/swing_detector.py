from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SwingPoint:
    """
    A confirmed swing point.

    index:
        Index of the actual pivot candle.

    timestamp:
        Timestamp of the pivot candle.

    confirmed_at:
        Time at which the pivot becomes known to the strategy.

    price:
        Swing price.

    kind:
        "high" or "low"
    """

    index: int
    timestamp: pd.Timestamp
    confirmed_at: pd.Timestamp
    price: float
    kind: str


def detect_pivots(
    df: pd.DataFrame,
    left_bars: int = 2,
    right_bars: int = 2,
    timeframe: str = "4h",
) -> list[SwingPoint]:
    """
    Detect confirmed Pivot High / Pivot Low.

    IMPORTANT:
    A pivot is NOT available at the pivot candle itself.

    Example with right_bars=2:

        candle[i-2]
        candle[i-1]
        candle[i]       <- pivot
        candle[i+1]
        candle[i+2]     <- confirmation

    The pivot becomes usable only after candle[i+2]
    has completed.

    This prevents look-ahead bias.
    """

    if left_bars < 1:
        raise ValueError(
            "left_bars must be >= 1"
        )

    if right_bars < 1:
        raise ValueError(
            "right_bars must be >= 1"
        )

    required_columns = {
        "timestamp",
        "high",
        "low",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if len(df) < (
        left_bars + right_bars + 1
    ):
        return []

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    )

    data = (
        data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    timeframe_delta = pd.Timedelta(
        timeframe
    )

    swings: list[SwingPoint] = []

    start = left_bars
    end = len(data) - right_bars

    for i in range(start, end):

        current_high = float(
            data.at[i, "high"]
        )

        current_low = float(
            data.at[i, "low"]
        )

        left_highs = data.loc[
            i - left_bars:i - 1,
            "high",
        ]

        right_highs = data.loc[
            i + 1:i + right_bars,
            "high",
        ]

        left_lows = data.loc[
            i - left_bars:i - 1,
            "low",
        ]

        right_lows = data.loc[
            i + 1:i + right_bars,
            "low",
        ]

        is_swing_high = (
            current_high > left_highs.max()
            and current_high >= right_highs.max()
        )

        is_swing_low = (
            current_low < left_lows.min()
            and current_low <= right_lows.min()
        )

        pivot_timestamp = pd.Timestamp(
            data.at[i, "timestamp"]
        )

        confirmation_index = i + right_bars

        confirmation_timestamp = (
            pd.Timestamp(
                data.at[
                    confirmation_index,
                    "timestamp",
                ]
            )
            + timeframe_delta
        )

        if is_swing_high:

            swings.append(
                SwingPoint(
                    index=i,
                    timestamp=pivot_timestamp,
                    confirmed_at=confirmation_timestamp,
                    price=current_high,
                    kind="high",
                )
            )

        if is_swing_low:

            swings.append(
                SwingPoint(
                    index=i,
                    timestamp=pivot_timestamp,
                    confirmed_at=confirmation_timestamp,
                    price=current_low,
                    kind="low",
                )
            )

    swings.sort(
        key=lambda swing: (
            swing.confirmed_at,
            swing.timestamp,
            swing.kind,
        )
    )

    return swings


def get_confirmed_swings(
    swings: list[SwingPoint],
    current_time: pd.Timestamp,
) -> list[SwingPoint]:
    """
    Return only swings that are known by current_time.
    """

    current_time = pd.Timestamp(
        current_time
    )

    if current_time.tzinfo is None:
        current_time = current_time.tz_localize(
            "UTC"
        )
    else:
        current_time = current_time.tz_convert(
            "UTC"
        )

    return [
        swing
        for swing in swings
        if swing.confirmed_at <= current_time
    ]


def get_last_confirmed_swing(
    swings: list[SwingPoint],
    kind: str,
    current_time: pd.Timestamp,
) -> SwingPoint | None:
    """
    Return the latest confirmed swing of a given type.
    """

    if kind not in {"high", "low"}:
        raise ValueError(
            "kind must be 'high' or 'low'"
        )

    confirmed = get_confirmed_swings(
        swings,
        current_time,
    )

    matching = [
        swing
        for swing in confirmed
        if swing.kind == kind
    ]

    if not matching:
        return None

    return matching[-1]
