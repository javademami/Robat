from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MicroMSSEvent:
    timestamp: pd.Timestamp
    direction: str

    zone_type: str
    retest_timestamp: pd.Timestamp
    displacement_timestamp: pd.Timestamp
    mss_timestamp: pd.Timestamp

    broken_swing_price: float
    broken_swing_timestamp: pd.Timestamp

    bars_after_retest: int

    candle_index: int
    swing_index: int

    reference_distance_atr: float


def _get_swing_attr(swing, *names, default=None):
    """
    Supports small differences between SwingPoint implementations.
    """
    for name in names:
        if hasattr(swing, name):
            return getattr(swing, name)
    return default


def _swing_kind(swing) -> str | None:
    value = _get_swing_attr(
        swing,
        "kind",
        "type",
        "direction",
        default=None,
    )

    if value is None:
        return None

    value = str(value).lower()

    if value in {"high", "swing_high", "sh"}:
        return "high"

    if value in {"low", "swing_low", "sl"}:
        return "low"

    return value


def _swing_timestamp(swing):
    return _get_swing_attr(
        swing,
        "timestamp",
        "time",
        "datetime",
        default=None,
    )


def _swing_price(swing):
    return _get_swing_attr(
        swing,
        "price",
        "level",
        "value",
        default=None,
    )


def _swing_confirmed_at(swing):
    return _get_swing_attr(
        swing,
        "confirmed_at",
        "confirmation_time",
        "confirmed_timestamp",
        default=None,
    )


def _swing_index(swing):
    return _get_swing_attr(
        swing,
        "index",
        "candle_index",
        "bar_index",
        default=-1,
    )


def _normalize_timestamp(value) -> pd.Timestamp:
    return pd.Timestamp(value)


def _prepare_swings(
    internal_swings: Iterable,
) -> list:
    """
    Sort swings by confirmation time.

    A pivot is usable only after its right-side candles have closed.
    """
    swings = []

    for swing in internal_swings:
        ts = _swing_timestamp(swing)
        price = _swing_price(swing)
        confirmed_at = _swing_confirmed_at(swing)
        kind = _swing_kind(swing)

        if ts is None or price is None or confirmed_at is None:
            continue

        if kind not in {"high", "low"}:
            continue

        swings.append(swing)

    swings.sort(
        key=lambda s: _normalize_timestamp(_swing_confirmed_at(s))
    )

    return swings


def _latest_confirmed_swing(
    swings: list,
    *,
    kind: str,
    confirmed_before: pd.Timestamp,
):
    """
    Returns the latest confirmed swing of the requested kind
    available at the moment the retest candle has closed.
    """
    candidate = None

    for swing in swings:
        confirmed_at = _normalize_timestamp(
            _swing_confirmed_at(swing)
        )

        if confirmed_at > confirmed_before:
            break

        if _swing_kind(swing) == kind:
            candidate = swing

    return candidate


def _find_reference_swings(
    swings: list,
    *,
    direction: str,
    retest_end: pd.Timestamp,
):
    """
    Bullish setup:
        break the latest confirmed swing HIGH.

    Bearish setup:
        break the latest confirmed swing LOW.
    """
    if direction == "bullish":
        return _latest_confirmed_swing(
            swings,
            kind="high",
            confirmed_before=retest_end,
        )

    if direction == "bearish":
        return _latest_confirmed_swing(
            swings,
            kind="low",
            confirmed_before=retest_end,
        )

    return None


def _validate_5m_dataframe(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"5M dataframe missing columns: {sorted(missing)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "5M dataframe index must be a pandas DatetimeIndex."
        )


def _compute_atr(
    df: pd.DataFrame,
    period: int,
) -> pd.Series:
    """
    Wilder-style simple-rolling ATR (matches the rolling-mean
    convention already used elsewhere in this project's
    ATR helpers).
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

    return tr.rolling(period).mean()


def detect_micro_mss(
    df_5m: pd.DataFrame,
    retests: Iterable,
    internal_swings_5m: Iterable,
    *,
    max_bars_after_retest: int = 24,
    atr_period: int = 14,
    min_reference_distance_atr: float = 0.25,
) -> list[MicroMSSEvent]:
    """
    Detect the first valid 5M Micro-MSS after each 1H retest.

    Rules:

    Bullish:
        Retest -> close above latest confirmed 5M swing high.

    Bearish:
        Retest -> close below latest confirmed 5M swing low.

    Important:
        The reference swing is fixed at retest confirmation time.
        New pivots created after the retest cannot replace it.

    Also enforces:

        MSS_1H < Displacement_1H < Retest_1H < Micro-MSS_5M

    Assumption:
        1H timestamps represent candle OPEN timestamps.

        Therefore a retest at 10:00 is only fully known at 11:00.
        Micro-MSS scanning starts at 11:00.

    V1.1 (quality filter):
        The reference swing must be at least
        `min_reference_distance_atr` * ATR(5m) away from the
        retest close. This drops structurally trivial "breaks"
        of a swing that is essentially at the current price
        already, which were previously accepted as valid
        Micro-MSS despite carrying little structural meaning.
        ATR is computed on the 5m series itself (not 1h), since
        the structure being measured is the 5m structure.
    """

    _validate_5m_dataframe(df_5m)

    if max_bars_after_retest <= 0:
        raise ValueError(
            "max_bars_after_retest must be greater than zero."
        )

    if min_reference_distance_atr < 0:
        raise ValueError(
            "min_reference_distance_atr must be >= 0."
        )

    swings = _prepare_swings(internal_swings_5m)

    events: list[MicroMSSEvent] = []

    if not swings:
        return events

    # ---------------------------------------------------------
    # ATR computed ONCE over the whole 5m series, not per-retest.
    # ---------------------------------------------------------
    atr_5m = _compute_atr(df_5m, atr_period)

    for retest in retests:
        raw_direction = getattr(retest, "direction", "")

        # ---------------------------------------------------------
        # ROBUSTNESS FIX:
        #
        # Some upstream events (e.g. OrderBlockEvent.direction,
        # when it's copied straight from a SweepDirection Enum
        # member instead of its plain string .value) are NOT
        # plain strings. str(SweepDirection.BULLISH) returns
        # 'SweepDirection.BULLISH', not 'bullish', which silently
        # failed every direction match below. Unwrap .value first
        # if it exists; otherwise treat the value as already a
        # plain string.
        # ---------------------------------------------------------
        direction = str(
            getattr(raw_direction, "value", raw_direction)
        ).lower()

        zone_type = str(
            getattr(retest, "zone_type", "")
        )

        retest_timestamp = _normalize_timestamp(
            getattr(retest, "retest_timestamp")
        )

        displacement_timestamp = _normalize_timestamp(
            getattr(retest, "displacement_timestamp")
        )

        mss_timestamp = _normalize_timestamp(
            getattr(retest, "mss_timestamp")
        )

        # ---------------------------------------------------------
        # Strict structural sequence
        # ---------------------------------------------------------
        if not (
            mss_timestamp
            < displacement_timestamp
            < retest_timestamp
        ):
            continue

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # retest_timestamp is the OPEN time of the 1H candle.
        #
        # The retest is only confirmed after that candle closes.
        # ---------------------------------------------------------
        retest_end = retest_timestamp + pd.Timedelta(hours=1)

        reference_swing = _find_reference_swings(
            swings,
            direction=direction,
            retest_end=retest_end,
        )

        if reference_swing is None:
            continue

        broken_price = float(
            _swing_price(reference_swing)
        )

        broken_timestamp = _normalize_timestamp(
            _swing_timestamp(reference_swing)
        )

        swing_index = int(
            _swing_index(reference_swing)
        )

        # ---------------------------------------------------------
        # V1.1 FILTER: minimum structure distance.
        #
        # Use the last 5m candle known strictly before retest_end
        # as "the retest close" for this measurement.
        # ---------------------------------------------------------
        retest_5m_rows = df_5m.loc[
            df_5m.index < retest_end
        ]

        if retest_5m_rows.empty:
            continue

        retest_5m_timestamp = retest_5m_rows.index[-1]

        atr_value = atr_5m.loc[retest_5m_timestamp]

        if pd.isna(atr_value) or float(atr_value) <= 0:
            continue

        retest_close = float(
            retest_5m_rows.iloc[-1]["close"]
        )

        reference_distance = abs(
            broken_price - retest_close
        )

        reference_distance_atr = (
            reference_distance / float(atr_value)
        )

        if reference_distance_atr < min_reference_distance_atr:
            continue

        # ---------------------------------------------------------
        # Scan only FUTURE 5M candles
        # ---------------------------------------------------------
        future_df = df_5m.loc[
            df_5m.index >= retest_end
        ]

        if future_df.empty:
            continue

        future_df = future_df.iloc[
            :max_bars_after_retest
        ]

        for bars_after_retest, (timestamp, candle) in enumerate(
            future_df.iterrows(),
            start=1,
        ):
            close = float(candle["close"])

            valid_break = False

            if direction == "bullish":
                valid_break = close > broken_price

            elif direction == "bearish":
                valid_break = close < broken_price

            if not valid_break:
                continue

            candle_index = df_5m.index.get_loc(timestamp)

            events.append(
                MicroMSSEvent(
                    timestamp=_normalize_timestamp(timestamp),
                    direction=direction,
                    zone_type=zone_type,
                    retest_timestamp=retest_timestamp,
                    displacement_timestamp=displacement_timestamp,
                    mss_timestamp=mss_timestamp,
                    broken_swing_price=broken_price,
                    broken_swing_timestamp=broken_timestamp,
                    bars_after_retest=bars_after_retest,
                    candle_index=int(candle_index),
                    swing_index=swing_index,
                    reference_distance_atr=reference_distance_atr,
                )
            )

            # First valid Micro-MSS only.
            break

    return events


def deduplicate_micro_mss(
    events: Iterable[MicroMSSEvent],
) -> list[MicroMSSEvent]:
    """
    Remove exact duplicate Micro-MSS events.

    Different setups are intentionally NOT merged here.
    """
    seen = set()
    result = []

    for event in events:
        key = (
            event.timestamp,
            event.direction,
            event.zone_type,
            event.retest_timestamp,
            event.displacement_timestamp,
            event.mss_timestamp,
            event.broken_swing_price,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(event)

    return result


def micro_mss_to_dataframe(
    events: Iterable[MicroMSSEvent],
) -> pd.DataFrame:
    rows = []

    for event in events:
        rows.append(
            {
                "timestamp": event.timestamp,
                "direction": event.direction,
                "zone_type": event.zone_type,
                "retest_timestamp": event.retest_timestamp,
                "displacement_timestamp": event.displacement_timestamp,
                "mss_timestamp": event.mss_timestamp,
                "broken_swing_price": event.broken_swing_price,
                "broken_swing_timestamp": event.broken_swing_timestamp,
                "bars_after_retest": event.bars_after_retest,
                "candle_index": event.candle_index,
                "swing_index": event.swing_index,
                "reference_distance_atr": event.reference_distance_atr,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "direction",
                "zone_type",
                "retest_timestamp",
                "displacement_timestamp",
                "mss_timestamp",
                "broken_swing_price",
                "broken_swing_timestamp",
                "bars_after_retest",
                "candle_index",
                "swing_index",
                "reference_distance_atr",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        "timestamp"
    ).reset_index(drop=True)