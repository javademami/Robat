from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.strategy.swing_detector import SwingPoint


class LiquidityType(str, Enum):
    PDH = "PDH"
    PDL = "PDL"
    EQUAL_HIGH = "EQUAL_HIGH"
    EQUAL_LOW = "EQUAL_LOW"
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"


@dataclass(frozen=True)
class LiquidityZone:
    """
    A price area where liquidity is expected.

    price:
        Main liquidity price.

    lower:
        Lower boundary of the zone.

    upper:
        Upper boundary of the zone.

    liquidity_type:
        Type of liquidity.

    strength:
        Relative importance.

    created_at:
        When the zone became known.

    source_time:
        Time of the underlying price/swing.
    """

    price: float
    lower: float
    upper: float

    liquidity_type: LiquidityType

    strength: int

    created_at: pd.Timestamp
    source_time: pd.Timestamp

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0


def normalize_timestamp(
    value: pd.Timestamp,
) -> pd.Timestamp:

    value = pd.Timestamp(value)

    if value.tzinfo is None:
        return value.tz_localize("UTC")

    return value.tz_convert("UTC")


def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Wilder-style ATR using EMA.

    Used for adaptive liquidity tolerance.
    """

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def _local_pivot_mask(
    series: pd.Series,
    kind: str,
    pivot_window: int = 1,
) -> "pd.Series[bool]":
    """
    Mark indices that are a local high/low within
    +/- pivot_window candles.

    This is intentionally simple (no confirmation-time
    handling here — the caller is responsible for that).
    Its only job is to stop every single candle from being
    treated as an "equal level" candidate, which is what
    was causing thousands of near-duplicate EQUAL_HIGH /
    EQUAL_LOW zones on 5m data.
    """

    if kind not in {"high", "low"}:
        raise ValueError("kind must be 'high' or 'low'")

    is_pivot = pd.Series(False, index=series.index)

    n = len(series)

    for i in range(pivot_window, n - pivot_window):

        window = series.iloc[
            i - pivot_window: i + pivot_window + 1
        ]

        value = series.iloc[i]

        if kind == "high":
            is_pivot.iloc[i] = value == window.max()
        else:
            is_pivot.iloc[i] = value == window.min()

    return is_pivot


def calculate_equal_tolerance(
    df: pd.DataFrame,
    atr_period: int = 14,
    tolerance_atr: float = 0.15,
) -> pd.Series:

    if tolerance_atr <= 0:
        raise ValueError(
            "tolerance_atr must be > 0"
        )

    return atr(
        df,
        atr_period,
    ) * tolerance_atr


def detect_equal_highs(
    df: pd.DataFrame,
    tolerance_atr: float = 0.15,
    atr_period: int = 14,
    lookback: int = 100,
    pivot_window: int = 3,
) -> list[LiquidityZone]:
    """
    Detect pairs of nearby highs.

    Two highs are considered equal when their distance
    is <= tolerance_atr * ATR.

    Initial parameter:
        0.15 ATR

    This is deliberately configurable for later
    parameter sensitivity testing.

    IMPORTANT (fix):
    Candidates are restricted to local pivot highs
    (+/- pivot_window candles), not every raw candle.
    Comparing every candle's high to every other nearby
    candle's high produced tens of thousands of near-
    duplicate EQUAL_HIGH zones on 5m data, which in turn
    inflated downstream Sweep counts.
    """

    required = {
        "timestamp",
        "high",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

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

    tolerance = calculate_equal_tolerance(
        data,
        atr_period=atr_period,
        tolerance_atr=tolerance_atr,
    )

    pivot_mask = _local_pivot_mask(
        data["high"],
        kind="high",
        pivot_window=pivot_window,
    )

    pivot_indices = [
        idx
        for idx in data.index
        if pivot_mask.iloc[idx]
    ]

    zones: list[LiquidityZone] = []

    start = max(
        atr_period,
        1,
    )

    for pos, i in enumerate(pivot_indices):

        if i < start:
            continue

        window_start = max(
            start,
            i - lookback,
        )

        current_price = float(
            data.at[i, "high"]
        )

        current_tolerance = float(
            tolerance.iloc[i]
        )

        if current_tolerance <= 0:
            continue

        previous_indices = [
            j
            for j in pivot_indices[:pos]
            if j >= window_start
        ]

        for j in previous_indices:

            previous_price = float(
                data.at[j, "high"]
            )

            if (
                abs(
                    current_price
                    - previous_price
                )
                <= current_tolerance
            ):

                lower = min(
                    current_price,
                    previous_price,
                )

                upper = max(
                    current_price,
                    previous_price,
                )

                zones.append(
                    LiquidityZone(
                        price=(
                            current_price
                            + previous_price
                        ) / 2.0,
                        lower=lower,
                        upper=upper,
                        liquidity_type=(
                            LiquidityType.EQUAL_HIGH
                        ),
                        strength=4,
                        created_at=(
                            normalize_timestamp(
                                data.at[
                                    i,
                                    "timestamp",
                                ]
                            )
                        ),
                        source_time=(
                            normalize_timestamp(
                                data.at[
                                    j,
                                    "timestamp",
                                ]
                            )
                        ),
                    )
                )

                break

    return deduplicate_zones(
        zones
    )


def detect_equal_lows(
    df: pd.DataFrame,
    tolerance_atr: float = 0.15,
    atr_period: int = 14,
    lookback: int = 100,
    pivot_window: int = 3,
) -> list[LiquidityZone]:
    """
    Detect pairs of nearby lows.

    IMPORTANT (fix):
    Candidates are restricted to local pivot lows
    (+/- pivot_window candles), not every raw candle.
    See detect_equal_highs for the rationale.
    """

    required = {
        "timestamp",
        "low",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

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

    tolerance = calculate_equal_tolerance(
        data,
        atr_period=atr_period,
        tolerance_atr=tolerance_atr,
    )

    pivot_mask = _local_pivot_mask(
        data["low"],
        kind="low",
        pivot_window=pivot_window,
    )

    pivot_indices = [
        idx
        for idx in data.index
        if pivot_mask.iloc[idx]
    ]

    zones: list[LiquidityZone] = []

    start = max(
        atr_period,
        1,
    )

    for pos, i in enumerate(pivot_indices):

        if i < start:
            continue

        window_start = max(
            start,
            i - lookback,
        )

        current_price = float(
            data.at[i, "low"]
        )

        current_tolerance = float(
            tolerance.iloc[i]
        )

        if current_tolerance <= 0:
            continue

        previous_indices = [
            j
            for j in pivot_indices[:pos]
            if j >= window_start
        ]

        for j in previous_indices:

            previous_price = float(
                data.at[j, "low"]
            )

            if (
                abs(
                    current_price
                    - previous_price
                )
                <= current_tolerance
            ):

                lower = min(
                    current_price,
                    previous_price,
                )

                upper = max(
                    current_price,
                    previous_price,
                )

                zones.append(
                    LiquidityZone(
                        price=(
                            current_price
                            + previous_price
                        ) / 2.0,
                        lower=lower,
                        upper=upper,
                        liquidity_type=(
                            LiquidityType.EQUAL_LOW
                        ),
                        strength=4,
                        created_at=(
                            normalize_timestamp(
                                data.at[
                                    i,
                                    "timestamp",
                                ]
                            )
                        ),
                        source_time=(
                            normalize_timestamp(
                                data.at[
                                    j,
                                    "timestamp",
                                ]
                            )
                        ),
                    )
                )

                break

    return deduplicate_zones(
        zones
    )


def detect_swing_liquidity(
    swings: list[SwingPoint],
    major_strength: int = 4,
) -> list[LiquidityZone]:
    """
    Convert confirmed swing points into liquidity zones.

    IMPORTANT:
    The zone becomes available at swing.confirmed_at,
    NOT at the pivot timestamp.
    """

    zones: list[LiquidityZone] = []

    for swing in swings:

        if swing.kind == "high":

            liquidity_type = (
                LiquidityType.SWING_HIGH
            )

        elif swing.kind == "low":

            liquidity_type = (
                LiquidityType.SWING_LOW
            )

        else:

            continue

        confirmed_at = normalize_timestamp(
            swing.confirmed_at
        )

        source_time = normalize_timestamp(
            swing.timestamp
        )

        zones.append(
            LiquidityZone(
                price=float(swing.price),
                lower=float(swing.price),
                upper=float(swing.price),
                liquidity_type=liquidity_type,
                strength=major_strength,
                created_at=confirmed_at,
                source_time=source_time,
            )
        )

    return zones


def detect_previous_day_liquidity(
    df: pd.DataFrame,
) -> list[LiquidityZone]:
    """
    Create Previous Day High / Previous Day Low.

    For each trading day, the liquidity level is taken
    from the PREVIOUS completed UTC day.

    Therefore today's PDH/PDL never uses today's future
    candles.
    """

    required = {
        "timestamp",
        "high",
        "low",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

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

    data["date"] = (
        data["timestamp"]
        .dt.floor("D")
    )

    daily = (
        data
        .groupby("date", sort=True)
        .agg(
            high=("high", "max"),
            low=("low", "min"),
        )
    )

    zones: list[LiquidityZone] = []

    dates = list(daily.index)

    for i in range(1, len(dates)):

        previous_date = dates[i - 1]
        current_date = dates[i]

        previous_high = float(
            daily.loc[
                previous_date,
                "high",
            ]
        )

        previous_low = float(
            daily.loc[
                previous_date,
                "low",
            ]
        )

        current_start = normalize_timestamp(
            pd.Timestamp(current_date)
        )

        zones.append(
            LiquidityZone(
                price=previous_high,
                lower=previous_high,
                upper=previous_high,
                liquidity_type=LiquidityType.PDH,
                strength=5,
                created_at=current_start,
                source_time=normalize_timestamp(
                    pd.Timestamp(previous_date)
                ),
            )
        )

        zones.append(
            LiquidityZone(
                price=previous_low,
                lower=previous_low,
                upper=previous_low,
                liquidity_type=LiquidityType.PDL,
                strength=5,
                created_at=current_start,
                source_time=normalize_timestamp(
                    pd.Timestamp(previous_date)
                ),
            )
        )

    return zones


def deduplicate_zones(
    zones: list[LiquidityZone],
    price_tolerance: float = 1e-8,
) -> list[LiquidityZone]:
    """
    Remove exact/near duplicate zones while keeping
    the strongest occurrence.

    FIX:
    The old check only merged zones whose midpoint price
    was equal to within 1e-8 — i.e. essentially only exact
    duplicates. Two EQUAL_HIGH zones a few dollars apart
    (both valid within the ATR tolerance used to build them)
    were treated as separate zones, which is why thousands
    of overlapping zones survived deduplication and each
    produced its own Sweep downstream.

    Now two zones of the same type are considered duplicates
    if their [lower, upper] ranges overlap OR their midpoints
    are within price_tolerance, whichever is looser. When
    duplicates are merged, the surviving zone's bounds are
    widened to cover both, so information isn't silently lost.
    """

    result: list[LiquidityZone] = []

    sorted_zones = sorted(
        zones,
        key=lambda z: (
            z.created_at,
            -z.strength,
        ),
    )

    for zone in sorted_zones:

        duplicate_idx = None

        for idx, existing in enumerate(result):

            same_type = (
                existing.liquidity_type
                == zone.liquidity_type
            )

            if not same_type:
                continue

            ranges_overlap = (
                existing.lower <= zone.upper
                and zone.lower <= existing.upper
            )

            close_price = (
                abs(
                    existing.price
                    - zone.price
                )
                <= price_tolerance
            )

            if ranges_overlap or close_price:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            result.append(zone)
            continue

        existing = result[duplicate_idx]

        merged_lower = min(existing.lower, zone.lower)
        merged_upper = max(existing.upper, zone.upper)

        # Keep the earlier created_at (it was known first)
        # and the stronger of the two strengths.
        kept_created_at = min(
            existing.created_at,
            zone.created_at,
        )

        kept_source_time = min(
            existing.source_time,
            zone.source_time,
        )

        kept_strength = max(
            existing.strength,
            zone.strength,
        )

        result[duplicate_idx] = LiquidityZone(
            price=(merged_lower + merged_upper) / 2.0,
            lower=merged_lower,
            upper=merged_upper,
            liquidity_type=existing.liquidity_type,
            strength=kept_strength,
            created_at=kept_created_at,
            source_time=kept_source_time,
        )

    result = _merge_transitive_overlaps(
        result
    )

    result.sort(
        key=lambda z: z.created_at
    )

    return result


def _merge_transitive_overlaps(
    zones: list[LiquidityZone],
) -> list[LiquidityZone]:
    """
    Second pass, run once per liquidity_type.

    The first pass in deduplicate_zones processes zones in
    chronological (created_at) order, so a chain like
    A-overlaps-B-overlaps-C can leave A and C as two separate
    zones if B (the bridge) happened to be compared after both
    were already independently kept. This pass does a standard
    "merge overlapping intervals" sweep, sorted by price, which
    is order-independent and closes that gap.
    """

    merged: list[LiquidityZone] = []

    by_type: dict[LiquidityType, list[LiquidityZone]] = {}

    for zone in zones:
        by_type.setdefault(
            zone.liquidity_type, []
        ).append(zone)

    for liquidity_type, group in by_type.items():

        group.sort(key=lambda z: z.lower)

        current = None

        for zone in group:

            if current is None:
                current = zone
                continue

            if zone.lower <= current.upper:

                merged_lower = min(
                    current.lower, zone.lower
                )
                merged_upper = max(
                    current.upper, zone.upper
                )

                current = LiquidityZone(
                    price=(
                        merged_lower + merged_upper
                    ) / 2.0,
                    lower=merged_lower,
                    upper=merged_upper,
                    liquidity_type=liquidity_type,
                    strength=max(
                        current.strength,
                        zone.strength,
                    ),
                    created_at=min(
                        current.created_at,
                        zone.created_at,
                    ),
                    source_time=min(
                        current.source_time,
                        zone.source_time,
                    ),
                )

            else:

                merged.append(current)
                current = zone

        if current is not None:
            merged.append(current)

    return merged


def get_active_liquidity(
    zones: list[LiquidityZone],
    current_time: pd.Timestamp,
) -> list[LiquidityZone]:
    """
    Return zones that are already known at current_time.
    """

    current_time = normalize_timestamp(
        current_time
    )

    return [
        zone
        for zone in zones
        if zone.created_at <= current_time
    ]


def liquidity_strength_name(
    strength: int,
) -> str:

    if strength >= 5:
        return "VERY_HIGH"

    if strength >= 4:
        return "HIGH"

    if strength >= 3:
        return "MEDIUM"

    return "LOW"