from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.strategy.sweep_detector import (
    SweepDirection,
    SweepEvent,
)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class MSSEvent:

    timestamp: pd.Timestamp

    direction: SweepDirection

    sweep_timestamp: pd.Timestamp
    sweep_liquidity_price: float
    sweep_extreme: float

    broken_swing_price: float
    broken_swing_timestamp: pd.Timestamp

    bars_after_sweep: int

    candle_index: int
    swing_index: int


# ============================================================
# TIME HELPERS
# ============================================================

def _normalize_timestamp(
    value,
) -> Optional[pd.Timestamp]:

    if value is None:
        return None

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.as_unit("ns")


def _normalize_datetime_series(
    series: pd.Series,
) -> pd.Series:

    return (
        pd.to_datetime(
            series,
            utc=True,
        )
        .astype("datetime64[ns, UTC]")
    )


# ============================================================
# MSS DETECTOR
# ============================================================

def detect_mss(
    df: pd.DataFrame,
    sweeps: list[SweepEvent],
    internal_swings: list,
    max_bars_after_sweep: int = 12,
) -> list[MSSEvent]:
    """
    Market Structure Shift detector.

    Bullish MSS:

        Bullish Sweep
             ↓
        latest confirmed
        internal Swing High
             ↓
        Close > Swing High

    Bearish MSS:

        Bearish Sweep
             ↓
        latest confirmed
        internal Swing Low
             ↓
        Close < Swing Low

    MSS must occur within max_bars_after_sweep.

    No future swing information is used.
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

    if df.empty or not sweeps:
        return []

    # --------------------------------------------------------
    # Candle dataframe
    # --------------------------------------------------------

    work = df.copy()

    work["timestamp"] = (
        _normalize_datetime_series(
            work["timestamp"]
        )
    )

    work = (
        work
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ========================================================
    # PREPARE INTERNAL SWINGS
    # ========================================================

    high_swings = []
    low_swings = []

    for swing in internal_swings:

        confirmed_at = _normalize_timestamp(
            swing.confirmed_at
        )

        if confirmed_at is None:
            continue

        item = {
            "confirmed_at": confirmed_at,
            "swing_time": _normalize_timestamp(
                swing.timestamp
            ),
            "price": float(
                swing.price
            ),
            "index": int(
                swing.index
            ),
        }

        if swing.kind == "high":
            high_swings.append(item)

        elif swing.kind == "low":
            low_swings.append(item)

    # --------------------------------------------------------
    # Sort once.
    # This avoids repeatedly sorting during detection.
    # --------------------------------------------------------

    high_swings.sort(
        key=lambda x: x["confirmed_at"]
    )

    low_swings.sort(
        key=lambda x: x["confirmed_at"]
    )

    # ========================================================
    # FAST POINTER LOOKUP
    # ========================================================

    def latest_swing_before(
        swings,
        timestamp,
    ):

        if not swings:
            return None

        timestamp = _normalize_timestamp(
            timestamp
        )

        latest = None

        for swing in swings:

            if (
                swing["confirmed_at"]
                <= timestamp
            ):
                latest = swing
            else:
                break

        return latest

    # ========================================================
    # CANDLE INDEX MAP
    # ========================================================

    time_to_idx = {
        timestamp: index
        for index, timestamp
        in enumerate(
            work["timestamp"]
        )
    }

    # ========================================================
    # PROCESS SWEEPS
    # ========================================================

    ordered_sweeps = sorted(
        sweeps,
        key=lambda x: _normalize_timestamp(
            x.timestamp
        ),
    )

    results: list[MSSEvent] = []

    for sweep in ordered_sweeps:

        sweep_time = _normalize_timestamp(
            sweep.timestamp
        )

        if sweep_time not in time_to_idx:
            continue

        sweep_index = (
            time_to_idx[sweep_time]
        )

        start_index = (
            sweep_index + 1
        )

        end_index = min(
            sweep_index
            + max_bars_after_sweep
            + 1,
            len(work),
        )

        # ====================================================
        # FIX: freeze the reference swing ONCE, at sweep_time.
        #
        # The old code called latest_swing_before(..., timestamp)
        # inside the per-candle loop below, using each candidate
        # candle's own timestamp. That let the reference swing
        # "move" if a new internal swing got confirmed sometime
        # between the sweep and the eventual break candle — i.e.
        # a swing that didn't exist yet at the moment of the sweep
        # could still end up being the one MSS is measured against.
        #
        # Not a look-ahead bug (confirmed_at <= timestamp always
        # held), but it violates the intended methodology: MSS
        # should be judged against "the last internal swing known
        # at the moment of the sweep", full stop — not a reference
        # that can silently update while we wait for the break.
        # ====================================================

        if (
            sweep.direction
            == SweepDirection.BULLISH
        ):
            reference_swing = latest_swing_before(
                high_swings,
                sweep_time,
            )
        elif (
            sweep.direction
            == SweepDirection.BEARISH
        ):
            reference_swing = latest_swing_before(
                low_swings,
                sweep_time,
            )
        else:
            reference_swing = None

        if reference_swing is None:
            continue

        reference_price = float(
            reference_swing["price"]
        )

        # ====================================================
        # SEARCH FOR MSS
        # ====================================================

        for i in range(
            start_index,
            end_index,
        ):

            row = work.iloc[i]

            timestamp = row["timestamp"]

            close = float(
                row["close"]
            )

            bars_after_sweep = (
                i - sweep_index
            )

            # =================================================
            # BULLISH MSS
            # =================================================

            if (
                sweep.direction
                == SweepDirection.BULLISH
            ):

                # Structure break must be by CLOSE.
                if close > reference_price:

                    results.append(
                        MSSEvent(
                            timestamp=timestamp,
                            direction=(
                                SweepDirection.BULLISH
                            ),
                            sweep_timestamp=(
                                sweep_time
                            ),
                            sweep_liquidity_price=(
                                float(
                                    sweep.liquidity_price
                                )
                            ),
                            sweep_extreme=(
                                float(
                                    sweep.sweep_extreme
                                )
                            ),
                            broken_swing_price=(
                                reference_price
                            ),
                            broken_swing_timestamp=(
                                reference_swing["swing_time"]
                            ),
                            bars_after_sweep=(
                                bars_after_sweep
                            ),
                            candle_index=i,
                            swing_index=(
                                reference_swing["index"]
                            ),
                        )
                    )

                    # First MSS after this sweep.
                    break

            # =================================================
            # BEARISH MSS
            # =================================================

            elif (
                sweep.direction
                == SweepDirection.BEARISH
            ):

                # Structure break must be by CLOSE.
                if close < reference_price:

                    results.append(
                        MSSEvent(
                            timestamp=timestamp,
                            direction=(
                                SweepDirection.BEARISH
                            ),
                            sweep_timestamp=(
                                sweep_time
                            ),
                            sweep_liquidity_price=(
                                float(
                                    sweep.liquidity_price
                                )
                            ),
                            sweep_extreme=(
                                float(
                                    sweep.sweep_extreme
                                )
                            ),
                            broken_swing_price=(
                                reference_price
                            ),
                            broken_swing_timestamp=(
                                reference_swing["swing_time"]
                            ),
                            bars_after_sweep=(
                                bars_after_sweep
                            ),
                            candle_index=i,
                            swing_index=(
                                reference_swing["index"]
                            ),
                        )
                    )

                    break

    return deduplicate_mss(
        results
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_mss(
    events: list[MSSEvent],
) -> list[MSSEvent]:

    if not events:
        return []

    ordered = sorted(
        events,
        key=lambda x: (
            x.timestamp,
            x.bars_after_sweep,
        ),
    )

    result = []
    seen = set()

    for event in ordered:

        key = (
            event.timestamp,
            event.direction,
            round(
                event.broken_swing_price,
                8,
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(event)

    return result


# ============================================================
# LATEST MSS
# ============================================================

def get_latest_mss(
    events: list[MSSEvent],
    current_time: Optional[pd.Timestamp] = None,
) -> Optional[MSSEvent]:

    if not events:
        return None

    if current_time is None:
        return events[-1]

    current_time = _normalize_timestamp(
        current_time
    )

    for event in reversed(events):

        event_time = _normalize_timestamp(
            event.timestamp
        )

        if event_time <= current_time:
            return event

    return None


# ============================================================
# DATAFRAME EXPORT
# ============================================================

def mss_to_dataframe(
    events: list[MSSEvent],
) -> pd.DataFrame:

    columns = [
        "timestamp",
        "direction",
        "sweep_timestamp",
        "sweep_liquidity_price",
        "sweep_extreme",
        "broken_swing_price",
        "broken_swing_timestamp",
        "bars_after_sweep",
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
                    else str(
                        event.direction
                    )
                ),
                "sweep_timestamp": (
                    event.sweep_timestamp
                ),
                "sweep_liquidity_price": (
                    event.sweep_liquidity_price
                ),
                "sweep_extreme": (
                    event.sweep_extreme
                ),
                "broken_swing_price": (
                    event.broken_swing_price
                ),
                "broken_swing_timestamp": (
                    event.broken_swing_timestamp
                ),
                "bars_after_sweep": (
                    event.bars_after_sweep
                ),
            }
            for event in events
        ]
    )