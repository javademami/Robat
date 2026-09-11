from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .order_block_detector import OrderBlockEvent
from .fvg_detector import FVGEvent


@dataclass(frozen=True)
class RetestEvent:
    zone_type: str
    direction: str

    zone_timestamp: pd.Timestamp
    displacement_timestamp: pd.Timestamp
    mss_timestamp: pd.Timestamp

    retest_timestamp: pd.Timestamp
    retest_candle_index: int

    zone_high: float
    zone_low: float
    zone_size: float

    penetration: float
    penetration_ratio: float

    close_inside: bool
    close_through: bool

    bars_after_zone: int


def _touches_zone(
    candle_low: float,
    candle_high: float,
    zone_low: float,
    zone_high: float,
) -> bool:
    return (
        candle_high >= zone_low
        and candle_low <= zone_high
    )


def _calculate_penetration(
    candle_low: float,
    candle_high: float,
    zone_low: float,
    zone_high: float,
    direction: str,
) -> float:

    zone_size = zone_high - zone_low

    if zone_size <= 0:
        return 0.0

    if direction == "bullish":
        penetration = zone_high - candle_low
    else:
        penetration = candle_high - zone_low

    return max(
        0.0,
        min(penetration, zone_size),
    )


def _build_retest(
    *,
    zone_type: str,
    direction: str,
    zone_timestamp: pd.Timestamp,
    displacement_timestamp: pd.Timestamp,
    mss_timestamp: pd.Timestamp,
    candle: pd.Series,
    candle_index: int,
    zone_index: int,
    zone_low: float,
    zone_high: float,
) -> RetestEvent:

    candle_low = float(candle["low"])
    candle_high = float(candle["high"])
    candle_close = float(candle["close"])

    zone_size = zone_high - zone_low

    penetration = _calculate_penetration(
        candle_low,
        candle_high,
        zone_low,
        zone_high,
        direction,
    )

    penetration_ratio = (
        penetration / zone_size
        if zone_size > 0
        else 0.0
    )

    close_inside = (
        zone_low
        <= candle_close
        <= zone_high
    )

    if direction == "bullish":
        close_through = candle_close < zone_low
    else:
        close_through = candle_close > zone_high

    return RetestEvent(
        zone_type=zone_type,
        direction=direction,

        zone_timestamp=zone_timestamp,
        displacement_timestamp=displacement_timestamp,
        mss_timestamp=mss_timestamp,

        retest_timestamp=pd.Timestamp(
            candle["timestamp"]
        ),
        retest_candle_index=candle_index,

        zone_high=zone_high,
        zone_low=zone_low,
        zone_size=zone_size,

        penetration=penetration,
        penetration_ratio=penetration_ratio,

        close_inside=close_inside,
        close_through=close_through,

        bars_after_zone=candle_index - zone_index,
    )


def detect_ob_retests(
    df: pd.DataFrame,
    order_blocks: List[OrderBlockEvent],
    max_bars_after_zone: int = 24,
) -> List[RetestEvent]:

    required = {
        "timestamp",
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
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

    results: List[RetestEvent] = []

    for zone in order_blocks:

        displacement_timestamp = pd.Timestamp(
            zone.displacement_timestamp
        )

        displacement_matches = work.index[
            work["timestamp"] == displacement_timestamp
        ]

        if len(displacement_matches) == 0:
            continue

        displacement_index = int(
            displacement_matches[0]
        )

        zone_matches = work.index[
            work["timestamp"] == pd.Timestamp(zone.timestamp)
        ]

        if len(zone_matches) == 0:
            continue

        zone_index = int(zone_matches[0])

        # IMPORTANT:
        # Retest starts AFTER displacement.
        #
        # The OB itself is not allowed to count as
        # a retest.
        start = displacement_index + 1

        end = min(
            len(work),
            start + max_bars_after_zone,
        )

        zone_low = float(zone.zone_low)
        zone_high = float(zone.zone_high)

        if zone_high <= zone_low:
            continue

        for i in range(start, end):

            candle = work.iloc[i]

            if not _touches_zone(
                float(candle["low"]),
                float(candle["high"]),
                zone_low,
                zone_high,
            ):
                continue

            results.append(
                _build_retest(
                    zone_type="OB",
                    direction=zone.direction,

                    zone_timestamp=pd.Timestamp(
                        zone.timestamp
                    ),

                    displacement_timestamp=(
                        displacement_timestamp
                    ),

                    mss_timestamp=pd.Timestamp(
                        zone.mss_timestamp
                    ),

                    candle=candle,
                    candle_index=i,
                    zone_index=zone_index,

                    zone_low=zone_low,
                    zone_high=zone_high,
                )
            )

            # First touch only.
            break

    return results


def detect_fvg_retests(
    df: pd.DataFrame,
    fvgs: List[FVGEvent],
    max_bars_after_zone: int = 24,
) -> List[RetestEvent]:

    required = {
        "timestamp",
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
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

    results: List[RetestEvent] = []

    for zone in fvgs:

        # FVG is only confirmed after C3.
        confirmation_timestamp = pd.Timestamp(
            zone.c3_timestamp
        )

        matches = work.index[
            work["timestamp"] == confirmation_timestamp
        ]

        if len(matches) == 0:
            continue

        zone_index = int(matches[0])

        start = zone_index + 1

        end = min(
            len(work),
            start + max_bars_after_zone,
        )

        zone_low = float(zone.zone_low)
        zone_high = float(zone.zone_high)

        if zone_high <= zone_low:
            continue

        for i in range(start, end):

            candle = work.iloc[i]

            if not _touches_zone(
                float(candle["low"]),
                float(candle["high"]),
                zone_low,
                zone_high,
            ):
                continue

            results.append(
                _build_retest(
                    zone_type="FVG",
                    direction=zone.direction,

                    zone_timestamp=pd.Timestamp(
                        zone.c2_timestamp
                    ),

                    displacement_timestamp=pd.Timestamp(
                        zone.displacement_timestamp
                    ),

                    mss_timestamp=pd.Timestamp(
                        zone.mss_timestamp
                    ),

                    candle=candle,
                    candle_index=i,
                    zone_index=zone_index,

                    zone_low=zone_low,
                    zone_high=zone_high,
                )
            )

            # First touch only.
            break

    return results


def retests_to_dataframe(
    events: List[RetestEvent],
) -> pd.DataFrame:

    if not events:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "zone_type": e.zone_type,
                "direction": e.direction,

                "zone_timestamp": e.zone_timestamp,
                "displacement_timestamp":
                    e.displacement_timestamp,
                "mss_timestamp":
                    e.mss_timestamp,

                "retest_timestamp":
                    e.retest_timestamp,
                "retest_candle_index":
                    e.retest_candle_index,

                "zone_high": e.zone_high,
                "zone_low": e.zone_low,
                "zone_size": e.zone_size,

                "penetration": e.penetration,
                "penetration_ratio":
                    e.penetration_ratio,

                "close_inside":
                    e.close_inside,

                "close_through":
                    e.close_through,

                "bars_after_zone":
                    e.bars_after_zone,
            }
            for e in events
        ]
    )