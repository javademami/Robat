from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.liquidity import (
    LiquidityType,
    LiquidityZone,
    atr,
)


# ============================================================
# ENUMS
# ============================================================

class SweepDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class SweepDepth(str, Enum):
    WEAK = "weak"
    NORMAL = "normal"
    DEEP = "deep"


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class SweepEvent:
    timestamp: pd.Timestamp
    direction: SweepDirection

    liquidity_price: float
    sweep_extreme: float
    reclaim_close: float

    liquidity_type: LiquidityType
    liquidity_strength: int

    depth: float
    depth_atr: float
    depth_class: SweepDepth

    zone_created_at: pd.Timestamp
    source_time: pd.Timestamp

    candle_index: int


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

    result = pd.to_datetime(
        series,
        utc=True,
    )

    # Explicitly force nanosecond precision.
    return result.astype("datetime64[ns, UTC]")


def _timestamp_ns(
    value,
) -> Optional[int]:

    ts = _normalize_timestamp(value)

    if ts is None:
        return None

    return int(ts.value)


# ============================================================
# SWEEP DEPTH
# ============================================================

def classify_sweep_depth(
    depth_atr: float,
) -> SweepDepth:

    if depth_atr < 0.10:
        return SweepDepth.WEAK

    if depth_atr <= 0.50:
        return SweepDepth.NORMAL

    return SweepDepth.DEEP


# ============================================================
# LIQUIDITY TYPE
# ============================================================

def liquidity_type_is_low(
    liquidity_type: LiquidityType,
) -> bool:

    return liquidity_type in {
        LiquidityType.PDL,
        LiquidityType.EQUAL_LOW,
        LiquidityType.SWING_LOW,
    }


def liquidity_type_is_high(
    liquidity_type: LiquidityType,
) -> bool:

    return liquidity_type in {
        LiquidityType.PDH,
        LiquidityType.EQUAL_HIGH,
        LiquidityType.SWING_HIGH,
    }


# ============================================================
# ZONE VISIBILITY
# ============================================================

def _zone_was_known(
    zone: LiquidityZone,
    candle_time: pd.Timestamp,
) -> bool:

    created_at = _normalize_timestamp(
        zone.created_at
    )

    if created_at is None:
        return False

    candle_time = _normalize_timestamp(
        candle_time
    )

    if candle_time is None:
        return False

    return created_at <= candle_time


# ============================================================
# PREPARE ZONES
# ============================================================

def _prepare_zone_arrays(
    zones: list[LiquidityZone],
):
    """
    Convert LiquidityZone objects into compact numpy-friendly
    arrays.

    This function deliberately avoids creating temporary
    LiquidityZone objects during sweep detection.
    """

    low_zones = []
    high_zones = []

    for zone in zones:

        created_ns = _timestamp_ns(
            zone.created_at
        )

        if created_ns is None:
            continue

        source_time = _normalize_timestamp(
            zone.source_time
        )

        item = {
            "price": float(zone.midpoint),
            "type": zone.liquidity_type,
            "strength": int(zone.strength),
            "created_ns": created_ns,
            "created_at": _normalize_timestamp(
                zone.created_at
            ),
            "source_time": source_time,
        }

        if liquidity_type_is_low(
            zone.liquidity_type
        ):
            low_zones.append(item)

        elif liquidity_type_is_high(
            zone.liquidity_type
        ):
            high_zones.append(item)

    return low_zones, high_zones


# ============================================================
# FIND FIRST VALID SWEEP FOR ZONE
# ============================================================

def _find_first_bullish_sweep(
    lows: np.ndarray,
    closes: np.ndarray,
    timestamps_ns: np.ndarray,
    atr_values: np.ndarray,
    zone: dict,
    min_depth_atr: float,
) -> Optional[tuple]:

    price = float(zone["price"])
    created_ns = int(zone["created_ns"])

    # --------------------------------------------------------
    # Only candles after zone creation.
    # --------------------------------------------------------

    start_idx = int(
        np.searchsorted(
            timestamps_ns,
            created_ns,
            side="left",
        )
    )

    if start_idx >= len(lows):
        return None

    local_lows = lows[start_idx:]
    local_closes = closes[start_idx:]
    local_atr = atr_values[start_idx:]

    # --------------------------------------------------------
    # Bullish sweep:
    #
    # Low < liquidity
    # Close > liquidity
    # --------------------------------------------------------

    mask = (
        (local_lows < price)
        & (local_closes > price)
        & np.isfinite(local_atr)
        & (local_atr > 0)
    )

    candidate_indices = np.flatnonzero(mask)

    if len(candidate_indices) == 0:
        return None

    # --------------------------------------------------------
    # Check depth.
    # --------------------------------------------------------

    for local_idx in candidate_indices:

        i = start_idx + int(local_idx)

        current_atr = float(
            atr_values[i]
        )

        if not np.isfinite(current_atr):
            continue

        depth = (
            price - float(lows[i])
        )

        depth_atr = (
            depth / current_atr
        )

        if depth_atr < min_depth_atr:
            continue

        return (
            i,
            depth,
            depth_atr,
        )

    return None


def _find_first_bearish_sweep(
    highs: np.ndarray,
    closes: np.ndarray,
    timestamps_ns: np.ndarray,
    atr_values: np.ndarray,
    zone: dict,
    min_depth_atr: float,
) -> Optional[tuple]:

    price = float(zone["price"])
    created_ns = int(zone["created_ns"])

    # --------------------------------------------------------
    # Only candles after zone creation.
    # --------------------------------------------------------

    start_idx = int(
        np.searchsorted(
            timestamps_ns,
            created_ns,
            side="left",
        )
    )

    if start_idx >= len(highs):
        return None

    local_highs = highs[start_idx:]
    local_closes = closes[start_idx:]
    local_atr = atr_values[start_idx:]

    # --------------------------------------------------------
    # Bearish sweep:
    #
    # High > liquidity
    # Close < liquidity
    # --------------------------------------------------------

    mask = (
        (local_highs > price)
        & (local_closes < price)
        & np.isfinite(local_atr)
        & (local_atr > 0)
    )

    candidate_indices = np.flatnonzero(mask)

    if len(candidate_indices) == 0:
        return None

    # --------------------------------------------------------
    # Check depth.
    # --------------------------------------------------------

    for local_idx in candidate_indices:

        i = start_idx + int(local_idx)

        current_atr = float(
            atr_values[i]
        )

        if not np.isfinite(current_atr):
            continue

        depth = (
            float(highs[i]) - price
        )

        depth_atr = (
            depth / current_atr
        )

        if depth_atr < min_depth_atr:
            continue

        return (
            i,
            depth,
            depth_atr,
        )

    return None


# ============================================================
# SWEEP DETECTOR
# ============================================================

def detect_sweeps(
    df: pd.DataFrame,
    zones: list[LiquidityZone],
    atr_period: int = 14,
    min_depth_atr: float = 0.0,
) -> list[SweepEvent]:
    """
    Optimized Smart Money Concepts liquidity sweep detector.

    Bullish sweep:

        Low < liquidity
        Close > liquidity

    Bearish sweep:

        High > liquidity
        Close < liquidity

    Rules:

    - A zone can only be swept after it was created.
    - Multiple liquidity zones are evaluated.
    - No future liquidity information is used.
    - A liquidity zone is consumed after its first sweep.
    - Sweep alone does NOT create a trade.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

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

    if df.empty:
        return []

    if not zones:
        return []

    # ========================================================
    # PREPARE DATA
    # ========================================================

    work = df.copy()

    work["timestamp"] = _normalize_datetime_series(
        work["timestamp"]
    )

    work = (
        work
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Numeric arrays
    # --------------------------------------------------------

    highs = (
        pd.to_numeric(
            work["high"],
            errors="coerce",
        )
        .to_numpy(dtype=np.float64)
    )

    lows = (
        pd.to_numeric(
            work["low"],
            errors="coerce",
        )
        .to_numpy(dtype=np.float64)
    )

    closes = (
        pd.to_numeric(
            work["close"],
            errors="coerce",
        )
        .to_numpy(dtype=np.float64)
    )

    timestamps = work["timestamp"]

    timestamps_ns = (
        timestamps.astype("int64")
        .to_numpy()
    )

    # ========================================================
    # ATR
    # ========================================================

    atr_values = (
        atr(
            work,
            period=atr_period,
        )
        .to_numpy(dtype=np.float64)
    )

    # ========================================================
    # PREPARE ZONES
    # ========================================================

    low_zones, high_zones = (
        _prepare_zone_arrays(zones)
    )

    # ========================================================
    # EVENTS
    # ========================================================

    events: list[SweepEvent] = []

    # ========================================================
    # BULLISH SWEEPS
    # ========================================================

    for zone in low_zones:

        result = _find_first_bullish_sweep(
            lows=lows,
            closes=closes,
            timestamps_ns=timestamps_ns,
            atr_values=atr_values,
            zone=zone,
            min_depth_atr=min_depth_atr,
        )

        if result is None:
            continue

        candle_index, depth, depth_atr = result

        timestamp = timestamps.iloc[
            candle_index
        ]

        events.append(
            SweepEvent(
                timestamp=timestamp,
                direction=SweepDirection.BULLISH,

                liquidity_price=float(
                    zone["price"]
                ),

                sweep_extreme=float(
                    lows[candle_index]
                ),

                reclaim_close=float(
                    closes[candle_index]
                ),

                liquidity_type=zone["type"],

                liquidity_strength=int(
                    zone["strength"]
                ),

                depth=float(depth),

                depth_atr=float(
                    depth_atr
                ),

                depth_class=(
                    classify_sweep_depth(
                        depth_atr
                    )
                ),

                zone_created_at=(
                    zone["created_at"]
                ),

                source_time=(
                    zone["source_time"]
                ),

                candle_index=int(
                    candle_index
                ),
            )
        )

    # ========================================================
    # BEARISH SWEEPS
    # ========================================================

    for zone in high_zones:

        result = _find_first_bearish_sweep(
            highs=highs,
            closes=closes,
            timestamps_ns=timestamps_ns,
            atr_values=atr_values,
            zone=zone,
            min_depth_atr=min_depth_atr,
        )

        if result is None:
            continue

        candle_index, depth, depth_atr = result

        timestamp = timestamps.iloc[
            candle_index
        ]

        events.append(
            SweepEvent(
                timestamp=timestamp,
                direction=SweepDirection.BEARISH,

                liquidity_price=float(
                    zone["price"]
                ),

                sweep_extreme=float(
                    highs[candle_index]
                ),

                reclaim_close=float(
                    closes[candle_index]
                ),

                liquidity_type=zone["type"],

                liquidity_strength=int(
                    zone["strength"]
                ),

                depth=float(depth),

                depth_atr=float(
                    depth_atr
                ),

                depth_class=(
                    classify_sweep_depth(
                        depth_atr
                    )
                ),

                zone_created_at=(
                    zone["created_at"]
                ),

                source_time=(
                    zone["source_time"]
                ),

                candle_index=int(
                    candle_index
                ),
            )
        )

    # ========================================================
    # DEDUPLICATE
    # ========================================================

    return deduplicate_sweeps(
        events
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_sweeps(
    sweeps: list[SweepEvent],
) -> list[SweepEvent]:

    if not sweeps:
        return []

    ordered = sorted(
        sweeps,
        key=lambda x: (
            x.timestamp,
            -x.liquidity_strength,
            x.depth_atr,
        ),
    )

    result: list[SweepEvent] = []

    seen = set()

    for sweep in ordered:

        key = (
            sweep.timestamp,
            sweep.direction,
            round(
                sweep.liquidity_price,
                6,
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(
            sweep
        )

    return sorted(
        result,
        key=lambda x: x.timestamp,
    )


# ============================================================
# LATEST SWEEP
# ============================================================

def get_latest_sweep(
    sweeps: list[SweepEvent],
    current_time: Optional[pd.Timestamp] = None,
) -> Optional[SweepEvent]:

    if not sweeps:
        return None

    if current_time is None:
        return sweeps[-1]

    current_time = _normalize_timestamp(
        current_time
    )

    if current_time is None:
        return None

    for sweep in reversed(sweeps):

        sweep_time = _normalize_timestamp(
            sweep.timestamp
        )

        if (
            sweep_time is not None
            and sweep_time <= current_time
        ):
            return sweep

    return None


# ============================================================
# DATAFRAME EXPORT
# ============================================================

def sweeps_to_dataframe(
    sweeps: list[SweepEvent],
) -> pd.DataFrame:

    columns = [
        "timestamp",
        "direction",
        "liquidity_price",
        "sweep_extreme",
        "reclaim_close",
        "liquidity_type",
        "liquidity_strength",
        "depth",
        "depth_atr",
        "depth_class",
    ]

    if not sweeps:

        return pd.DataFrame(
            columns=columns
        )

    return pd.DataFrame(
        [
            {
                "timestamp": s.timestamp,

                "direction": (
                    s.direction.value
                ),

                "liquidity_price": (
                    s.liquidity_price
                ),

                "sweep_extreme": (
                    s.sweep_extreme
                ),

                "reclaim_close": (
                    s.reclaim_close
                ),

                "liquidity_type": (
                    s.liquidity_type.name
                    if hasattr(
                        s.liquidity_type,
                        "name",
                    )
                    else str(
                        s.liquidity_type
                    )
                ),

                "liquidity_strength": (
                    s.liquidity_strength
                ),

                "depth": s.depth,

                "depth_atr": (
                    s.depth_atr
                ),

                "depth_class": (
                    s.depth_class.value
                ),
            }

            for s in sweeps
        ]
    )