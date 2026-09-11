from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .displacement_detector import DisplacementEvent


@dataclass(frozen=True)
class FVGEvent:
    timestamp: pd.Timestamp
    direction: str

    displacement_timestamp: pd.Timestamp
    mss_timestamp: pd.Timestamp

    candle_index: int

    c1_timestamp: pd.Timestamp
    c2_timestamp: pd.Timestamp
    c3_timestamp: pd.Timestamp

    zone_high: float
    zone_low: float

    gap_size: float
    gap_atr: float


def detect_fvgs(
    df: pd.DataFrame,
    displacements: List[DisplacementEvent],
    atr_period: int = 14,
    search_bars: int = 3,
) -> List[FVGEvent]:
    """
    Detect 3-candle Fair Value Gaps associated with displacement.

    Bullish FVG:
        High(C1) < Low(C3)

        zone = [High(C1), Low(C3)]

    Bearish FVG:
        Low(C1) > High(C3)

        zone = [High(C3), Low(C1)]

    Only FVGs occurring at/after the displacement candle are considered.
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

    # ATR
    prev_close = work["close"].shift(1)

    tr = pd.concat(
        [
            work["high"] - work["low"],
            (work["high"] - prev_close).abs(),
            (work["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    work["_atr"] = tr.rolling(atr_period).mean()

    results: List[FVGEvent] = []

    for displacement in displacements:
        displacement_ts = pd.Timestamp(displacement.timestamp)

        matches = work.index[work["timestamp"] == displacement_ts]
        if len(matches) == 0:
            continue

        displacement_index = int(matches[0])

        start = displacement_index
        end = min(
            len(work) - 3,
            displacement_index + search_bars - 1,
        )

        for i in range(start, end + 1):
            c1 = work.iloc[i]
            c2 = work.iloc[i + 1]
            c3 = work.iloc[i + 2]

            atr = float(c2["_atr"])

            if pd.isna(atr) or atr <= 0:
                continue

            if displacement.direction == "bullish":
                if float(c1["high"]) < float(c3["low"]):
                    zone_low = float(c1["high"])
                    zone_high = float(c3["low"])
                    gap_size = zone_high - zone_low

                    results.append(
                        FVGEvent(
                            timestamp=pd.Timestamp(c2["timestamp"]),
                            direction="bullish",
                            displacement_timestamp=displacement_ts,
                            mss_timestamp=pd.Timestamp(
                                displacement.mss_timestamp
                            ),
                            candle_index=i + 1,
                            c1_timestamp=pd.Timestamp(c1["timestamp"]),
                            c2_timestamp=pd.Timestamp(c2["timestamp"]),
                            c3_timestamp=pd.Timestamp(c3["timestamp"]),
                            zone_high=zone_high,
                            zone_low=zone_low,
                            gap_size=gap_size,
                            gap_atr=gap_size / atr,
                        )
                    )

            elif displacement.direction == "bearish":
                if float(c1["low"]) > float(c3["high"]):
                    zone_low = float(c3["high"])
                    zone_high = float(c1["low"])
                    gap_size = zone_high - zone_low

                    results.append(
                        FVGEvent(
                            timestamp=pd.Timestamp(c2["timestamp"]),
                            direction="bearish",
                            displacement_timestamp=displacement_ts,
                            mss_timestamp=pd.Timestamp(
                                displacement.mss_timestamp
                            ),
                            candle_index=i + 1,
                            c1_timestamp=pd.Timestamp(c1["timestamp"]),
                            c2_timestamp=pd.Timestamp(c2["timestamp"]),
                            c3_timestamp=pd.Timestamp(c3["timestamp"]),
                            zone_high=zone_high,
                            zone_low=zone_low,
                            gap_size=gap_size,
                            gap_atr=gap_size / atr,
                        )
                    )

    return deduplicate_fvgs(results)


def deduplicate_fvgs(
    events: List[FVGEvent],
) -> List[FVGEvent]:
    seen = set()
    result = []

    for event in events:
        key = (
            event.c1_timestamp,
            event.c2_timestamp,
            event.c3_timestamp,
            event.direction,
            round(event.zone_high, 8),
            round(event.zone_low, 8),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(event)

    return sorted(
        result,
        key=lambda x: x.timestamp,
    )


def fvg_to_dataframe(
    events: List[FVGEvent],
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
                "c1_timestamp": e.c1_timestamp,
                "c2_timestamp": e.c2_timestamp,
                "c3_timestamp": e.c3_timestamp,
                "zone_high": e.zone_high,
                "zone_low": e.zone_low,
                "gap_size": e.gap_size,
                "gap_atr": e.gap_atr,
            }
            for e in events
        ]
    )