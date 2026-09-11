from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, SYMBOL, YEAR
from src.data.storage import load_year
from src.data.aggregator import resample_ohlcv
from src.strategy.swing_detector import detect_pivots
from src.strategy.liquidity import (
    detect_equal_highs,
    detect_equal_lows,
    detect_previous_day_liquidity,
    detect_swing_liquidity,
    deduplicate_zones,
)
from src.strategy.sweep_detector import detect_sweeps
from src.strategy.mss_detector import detect_mss
from src.strategy.displacement_detector import detect_displacements
from src.strategy.order_block_detector import detect_order_blocks
from src.strategy.fvg_detector import detect_fvgs
from src.strategy.retest_detector import detect_ob_retests, detect_fvg_retests
from src.strategy.micro_mss_detector import (
    detect_micro_mss,
    deduplicate_micro_mss,
    micro_mss_to_dataframe,
    _prepare_swings,
    _find_reference_swings,
    _normalize_timestamp,
)


OUTPUT_ROOT = Path("data") / "validation"
WINDOW_BEFORE_5M = 48
WINDOW_AFTER_5M = 24


def normalize_value(value):
    """Make Enum/string/timestamp values CSV-friendly."""
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def event_attr(event, name, default=None):
    value = getattr(event, name, default)
    return normalize_value(value)


def build_pipeline():
    print("=" * 72)
    print("BUILDING MICRO-MSS VALIDATION DATASET")
    print("=" * 72)

    print("\n[1] Loading 5M...")
    df_5m = load_year(DATA_DIR, SYMBOL, "5m", YEAR)
    print(f"5M candles: {len(df_5m):,}")

    print("\n[2] Resampling...")
    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")
    print(f"1H candles: {len(df_1h):,}")
    print(f"4H candles: {len(df_4h):,}")

    print("\n[3] 4H swings...")
    swings_4h = detect_pivots(
        df_4h, left_bars=2, right_bars=2, timeframe="4h"
    )
    print(f"4H swings: {len(swings_4h):,}")

    print("\n[4] Liquidity...")
    swing_zones = detect_swing_liquidity(swings_4h)
    equal_highs = detect_equal_highs(
        df_1h,
        tolerance_atr=0.15,
        atr_period=14,
        lookback=100,
        pivot_window=3,
    )
    equal_lows = detect_equal_lows(
        df_1h,
        tolerance_atr=0.15,
        atr_period=14,
        lookback=100,
        pivot_window=3,
    )
    previous_day = detect_previous_day_liquidity(df_1h)
    zones = deduplicate_zones(
        swing_zones + equal_highs + equal_lows + previous_day
    )
    print(f"Liquidity zones: {len(zones):,}")

    print("\n[5] Sweeps...")
    sweeps = detect_sweeps(df_1h, zones, atr_period=14)
    print(f"Sweeps: {len(sweeps):,}")

    print("\n[6] 1H internal swings...")
    internal_swings_1h = detect_pivots(
        df_1h, left_bars=2, right_bars=2, timeframe="1h"
    )
    print(f"1H internal swings: {len(internal_swings_1h):,}")

    print("\n[7] MSS...")
    mss_events = detect_mss(
        df_1h,
        sweeps,
        internal_swings_1h,
        max_bars_after_sweep=12,
    )
    print(f"MSS: {len(mss_events):,}")

    print("\n[8] Displacement...")
    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=14,
        min_body_atr=1.0,
        bullish_clv=0.70,
        bearish_clv=0.30,
        max_bars_after_mss=8,
    )
    print(f"Displacements: {len(displacement_events):,}")

    print("\n[9] Order Blocks...")
    ob_events = detect_order_blocks(
        df_1h,
        displacement_events,
        lookback=8,
    )
    print(f"Order Blocks: {len(ob_events):,}")

    print("\n[10] FVG...")
    fvg_events = detect_fvgs(
        df_1h,
        displacement_events,
        search_bars=3,
        atr_period=14,
    )
    print(f"FVG: {len(fvg_events):,}")

    print("\n[11] Retests...")
    ob_retests = detect_ob_retests(
        df_1h, ob_events, max_bars_after_zone=24
    )
    fvg_retests = detect_fvg_retests(
        df_1h, fvg_events, max_bars_after_zone=24
    )
    all_retests = list(ob_retests) + list(fvg_retests)
    print(f"OB retests: {len(ob_retests):,}")
    print(f"FVG retests: {len(fvg_retests):,}")
    print(f"Total retests: {len(all_retests):,}")

    print("\n[12] 5M swings...")
    internal_swings_5m = detect_pivots(
        df_5m,
        left_bars=2,
        right_bars=2,
        timeframe="5m",
    )
    print(f"5M internal swings: {len(internal_swings_5m):,}")

    df_5m_indexed = df_5m.copy()
    df_5m_indexed["timestamp"] = pd.to_datetime(
        df_5m_indexed["timestamp"], utc=True
    )
    df_5m_indexed = (
        df_5m_indexed
        .set_index("timestamp")
        .sort_index()
    )

    print("\n[13] Micro-MSS...")
    micro_events = detect_micro_mss(
        df_5m_indexed,
        all_retests,
        internal_swings_5m,
        max_bars_after_retest=24,
        min_reference_distance_atr=0.25,
    )
    micro_events = deduplicate_micro_mss(micro_events)
    print(f"Micro-MSS: {len(micro_events):,}")

    return (
        df_5m_indexed,
        all_retests,
        internal_swings_5m,
        micro_events,
    )


def validate_events(
    df_5m_indexed,
    all_retests,
    internal_swings_5m,
    micro_events,
):
    """Validate chronology and reference-swing constraints."""
    rows = []

    for event_id, event in enumerate(micro_events, start=1):
        mss_ts = _normalize_timestamp(event.mss_timestamp)
        displacement_ts = _normalize_timestamp(event.displacement_timestamp)
        retest_ts = _normalize_timestamp(event.retest_timestamp)
        micro_ts = _normalize_timestamp(event.timestamp)

        sequence_ok = bool(
            mss_ts < displacement_ts < retest_ts < micro_ts
        )
        micro_after_retest_close = bool(
            micro_ts >= retest_ts + pd.Timedelta(hours=1)
        )

        reference_swing = None
        reference_error = None

        try:
            prepared = _prepare_swings(internal_swings_5m)

            direction = getattr(event, "direction", "")
            direction = str(
                getattr(direction, "value", direction)
            ).lower()

            reference_swing = _find_reference_swings(
                prepared,
                direction=direction,
                retest_end=retest_ts + pd.Timedelta(hours=1),
            )
        except Exception as exc:
            reference_error = str(exc)

        reference_ts = None
        reference_price = None

        if reference_swing is not None:
            if isinstance(reference_swing, dict):
                reference_ts = (
                    reference_swing.get("timestamp")
                    or reference_swing.get("price_timestamp")
                )
                reference_price = (
                    reference_swing.get("price")
                    or reference_swing.get("high")
                    or reference_swing.get("low")
                )
            else:
                reference_ts = getattr(
                    reference_swing, "timestamp", None
                )
                reference_price = getattr(
                    reference_swing, "price", None
                )

        if reference_ts is not None:
            reference_ts = _normalize_timestamp(reference_ts)

        reference_before_retest = (
            reference_ts is not None
            and reference_ts < retest_ts
        )

        # The detector already exposes this value. We only validate that
        # the final event respects the V1.1 minimum.
        distance_atr = getattr(
            event, "reference_distance_atr", None
        )
        distance_ok = (
            distance_atr is not None
            and float(distance_atr) >= 0.25
        )

        row = {
            "event_id": event_id,
            "timestamp": normalize_value(event.timestamp),
            "direction": normalize_value(event.direction),
            "zone_type": normalize_value(event.zone_type),
            "mss_timestamp": normalize_value(event.mss_timestamp),
            "displacement_timestamp": normalize_value(
                event.displacement_timestamp
            ),
            "retest_timestamp": normalize_value(event.retest_timestamp),
            "broken_swing_timestamp": normalize_value(
                getattr(event, "broken_swing_timestamp", None)
            ),
            "broken_swing_price": getattr(
                event, "broken_swing_price", None
            ),
            "reference_distance_atr": distance_atr,
            "bars_after_retest": getattr(
                event, "bars_after_retest", None
            ),
            "event_candle_index": getattr(
                event, "candle_index", None
            ),
            "sequence_ok": sequence_ok,
            "micro_after_retest_close": micro_after_retest_close,
            "reference_before_retest": reference_before_retest,
            "reference_found_during_validation": (
                reference_swing is not None
            ),
            "reference_validation_error": reference_error,
            "distance_atr_ok": distance_ok,
            "validation_status": (
                "PASS"
                if (
                    sequence_ok
                    and micro_after_retest_close
                    and reference_before_retest
                    and distance_ok
                    and reference_error is None
                )
                else "CHECK"
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def export_5m_windows(df_5m_indexed, events, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0

    for event_id, event in enumerate(events, start=1):
        event_ts = _normalize_timestamp(event.timestamp)

        start = event_ts - pd.Timedelta(
            minutes=5 * WINDOW_BEFORE_5M
        )
        end = event_ts + pd.Timedelta(
            minutes=5 * WINDOW_AFTER_5M
        )

        window = df_5m_indexed.loc[
            (df_5m_indexed.index >= start)
            & (df_5m_indexed.index <= end)
        ].copy()

        if window.empty:
            continue

        window = window.reset_index()
        window.insert(0, "event_id", event_id)
        window["micro_mss_timestamp"] = event_ts.isoformat()
        window["direction"] = normalize_value(event.direction)
        window["zone_type"] = normalize_value(event.zone_type)
        window["retest_timestamp"] = normalize_value(
            event.retest_timestamp
        )

        filename = output_dir / f"micro_mss_{event_id:03d}.csv"
        window.to_csv(filename, index=False)
        exported += 1

    return exported


def main():
    (
        df_5m_indexed,
        all_retests,
        internal_swings_5m,
        micro_events,
    ) = build_pipeline()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n[14] Validating Micro-MSS events...")
    validation_df = validate_events(
        df_5m_indexed,
        all_retests,
        internal_swings_5m,
        micro_events,
    )

    events_csv = OUTPUT_ROOT / "micro_mss_events.csv"
    validation_df.to_csv(events_csv, index=False)

    windows_dir = OUTPUT_ROOT / "micro_mss_5m_windows"
    exported_windows = export_5m_windows(
        df_5m_indexed,
        micro_events,
        windows_dir,
    )

    # Also preserve the detector's native dataframe for comparison.
    native_df = micro_mss_to_dataframe(micro_events)
    native_csv = OUTPUT_ROOT / "micro_mss_native.csv"
    native_df.to_csv(native_csv, index=False)

    total = len(micro_events)
    pass_count = int(
        (validation_df["validation_status"] == "PASS").sum()
    ) if not validation_df.empty else 0

    invalid_sequence = int(
        (~validation_df["sequence_ok"]).sum()
    ) if not validation_df.empty else 0

    before_retest = int(
        (~validation_df["micro_after_retest_close"]).sum()
    ) if not validation_df.empty else 0

    bad_reference = int(
        (~validation_df["reference_before_retest"]).sum()
    ) if not validation_df.empty else 0

    bad_distance = int(
        (~validation_df["distance_atr_ok"]).sum()
    ) if not validation_df.empty else 0

    direction_counts = Counter(
        normalize_value(event.direction)
        for event in micro_events
    )
    zone_counts = Counter(
        normalize_value(event.zone_type)
        for event in micro_events
    )

    print("\n" + "=" * 72)
    print("MICRO-MSS VISUAL VALIDATION")
    print("=" * 72)
    print(f"Total retests        : {len(all_retests):,}")
    print(f"Micro-MSS            : {total:,}")
    print(
        f"Retest -> Micro-MSS  : "
        f"{(total / len(all_retests) * 100):.2f}%"
        if all_retests else
        "Retest -> Micro-MSS  : 0.00%"
    )

    print("\nDirection:")
    for key, value in sorted(direction_counts.items()):
        print(f"  {key:10s}: {value:,}")

    print("\nZone type:")
    for key, value in sorted(zone_counts.items()):
        print(f"  {key:10s}: {value:,}")

    print("\nValidation:")
    print(f"  PASS events                 : {pass_count:,}/{total:,}")
    print(f"  Invalid chronology          : {invalid_sequence:,}")
    print(f"  Micro-MSS before 1H close  : {before_retest:,}")
    print(f"  Reference after retest     : {bad_reference:,}")
    print(f"  Distance < 0.25 ATR        : {bad_distance:,}")

    print("\nOutput:")
    print(f"  Events CSV : {events_csv}")
    print(f"  Native CSV : {native_csv}")
    print(f"  5M windows : {windows_dir}")
    print(f"  Windows exported: {exported_windows:,}")

    status = (
        "PASS"
        if total > 0
        and pass_count == total
        and invalid_sequence == 0
        and before_retest == 0
        and bad_reference == 0
        and bad_distance == 0
        else "CHECK"
    )

    print("\n" + "=" * 72)
    print(f"STATUS: {status}")
    print("=" * 72)


if __name__ == "__main__":
    main()
