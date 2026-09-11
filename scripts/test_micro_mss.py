from collections import Counter

import pandas as pd

from src.config import DATA_DIR, SYMBOL, YEAR

# FIX (high confidence): confirmed in this project's own earlier
# traceback that load_year lives in src.data.storage, not
# src.data.loader (which doesn't exist in this project).
from src.data.storage import load_year

# FIX (medium confidence): src/data/resampler.py was never part of
# this project's documented architecture. The original AUDIT_REPORT
# describes src/data/aggregator.py as the module responsible for
# "تبدیل تایم‌فریم‌ها" (timeframe conversion), so resample_ohlcv is
# assumed to live there. If this import still fails, tell me the
# actual filename/function and I'll correct it immediately.
from src.data.aggregator import resample_ohlcv

# FIX (high confidence): detect_pivots is defined in
# src/strategy/swing_detector.py (verified by reading that file
# directly earlier in this session). src/strategy/structure.py
# does not exist in this project; market_structure.py exists but
# does not define detect_pivots.
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

from src.strategy.displacement_detector import (
    detect_displacements,
)

from src.strategy.order_block_detector import (
    detect_order_blocks,
)

from src.strategy.fvg_detector import (
    detect_fvgs,
)

from src.strategy.retest_detector import (
    detect_ob_retests,
    detect_fvg_retests,
)

from src.strategy.micro_mss_detector import (
    detect_micro_mss,
    deduplicate_micro_mss,
    micro_mss_to_dataframe,
)


def main():

    print("=" * 70)
    print("MICRO-MSS V1 TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load canonical 5M data
    # ---------------------------------------------------------
    print("\n[1] Loading 5M data...")

    df_5m = load_year(
        DATA_DIR,
        SYMBOL,
        "5m",
        YEAR,
    )

    print(f"5M candles: {len(df_5m):,}")

    # ---------------------------------------------------------
    # 2. Build 1H / 4H
    # ---------------------------------------------------------
    print("\n[2] Resampling...")

    df_1h = resample_ohlcv(
        df_5m,
        "1h",
    )

    df_4h = resample_ohlcv(
        df_5m,
        "4h",
    )

    print(f"1H candles: {len(df_1h):,}")
    print(f"4H candles: {len(df_4h):,}")

    # ---------------------------------------------------------
    # 3. 4H external structure
    # ---------------------------------------------------------
    print("\n[3] Detecting 4H swings...")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=2,
        right_bars=2,
        timeframe="4h",
    )

    print(f"4H swings: {len(swings_4h):,}")

    # ---------------------------------------------------------
    # 4. Liquidity
    # ---------------------------------------------------------
    print("\n[4] Detecting liquidity...")

    swing_zones = detect_swing_liquidity(
        swings_4h
    )

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

    previous_day = detect_previous_day_liquidity(
        df_1h
    )

    zones = deduplicate_zones(
        swing_zones
        + equal_highs
        + equal_lows
        + previous_day
    )

    print(f"Liquidity zones: {len(zones):,}")

    # ---------------------------------------------------------
    # 5. Sweeps
    # ---------------------------------------------------------
    print("\n[5] Detecting sweeps...")

    sweeps = detect_sweeps(
        df_1h,
        zones,
        atr_period=14,
    )

    print(f"Sweeps: {len(sweeps):,}")

    # ---------------------------------------------------------
    # 6. 1H internal structure
    # ---------------------------------------------------------
    print("\n[6] Detecting 1H internal swings...")

    internal_swings_1h = detect_pivots(
        df_1h,
        left_bars=2,
        right_bars=2,
        timeframe="1h",
    )

    print(
        f"1H internal swings: "
        f"{len(internal_swings_1h):,}"
    )

    # ---------------------------------------------------------
    # 7. MSS
    # ---------------------------------------------------------
    print("\n[7] Detecting MSS...")

    mss_events = detect_mss(
        df_1h,
        sweeps,
        internal_swings_1h,
        max_bars_after_sweep=12,
    )

    print(f"MSS: {len(mss_events):,}")

    # ---------------------------------------------------------
    # 8. Displacement
    # ---------------------------------------------------------
    print("\n[8] Detecting displacement...")

    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=14,
        min_body_atr=1.0,
        bullish_clv=0.70,
        bearish_clv=0.30,
        max_bars_after_mss=8,
    )

    print(
        f"Displacements: "
        f"{len(displacement_events):,}"
    )

    # ---------------------------------------------------------
    # 9. OB
    # ---------------------------------------------------------
    print("\n[9] Detecting Order Blocks...")

    ob_events = detect_order_blocks(
        df_1h,
        displacement_events,
        lookback=8,
    )

    print(
        f"Order Blocks: "
        f"{len(ob_events):,}"
    )

    # ---------------------------------------------------------
    # 10. FVG
    # ---------------------------------------------------------
    print("\n[10] Detecting FVG...")

    fvg_events = detect_fvgs(
        df_1h,
        displacement_events,
        search_bars=3,
        atr_period=14,
    )

    print(
        f"FVG: "
        f"{len(fvg_events):,}"
    )

    # ---------------------------------------------------------
    # 11. OB Retests
    # ---------------------------------------------------------
    print("\n[11] Detecting OB retests...")

    ob_retests = detect_ob_retests(
        df_1h,
        ob_events,
        max_bars_after_zone=24,
    )

    print(
        f"OB retests: "
        f"{len(ob_retests):,}"
    )

    # ---------------------------------------------------------
    # 12. FVG Retests
    # ---------------------------------------------------------
    print("\n[12] Detecting FVG retests...")

    fvg_retests = detect_fvg_retests(
        df_1h,
        fvg_events,
        max_bars_after_zone=24,
    )

    print(
        f"FVG retests: "
        f"{len(fvg_retests):,}"
    )

    # ---------------------------------------------------------
    # 13. Combine setups
    # ---------------------------------------------------------
    all_retests = (
        list(ob_retests)
        + list(fvg_retests)
    )

    print(
        f"\nTotal retests: "
        f"{len(all_retests):,}"
    )

    # ---------------------------------------------------------
    # 14. 5M internal swings
    # ---------------------------------------------------------
    print("\n[13] Detecting 5M internal swings...")

    internal_swings_5m = detect_pivots(
        df_5m,
        left_bars=2,
        right_bars=2,
        timeframe="5m",
    )

    print(
        f"5M internal swings: "
        f"{len(internal_swings_5m):,}"
    )

    # ---------------------------------------------------------
    # 14.5 micro_mss_detector expects a DatetimeIndex, unlike
    # the rest of the pipeline (which uses a "timestamp" column).
    # detect_pivots above still needs the column-based df_5m, so
    # we build an indexed copy only for this next call.
    # ---------------------------------------------------------

    df_5m_indexed = df_5m.copy()

    df_5m_indexed["timestamp"] = pd.to_datetime(
        df_5m_indexed["timestamp"],
        utc=True,
    )

    df_5m_indexed = (
        df_5m_indexed
        .set_index("timestamp")
        .sort_index()
    )

    # ---------------------------------------------------------
    # 14.6 DIAGNOSTIC: why does OB vs FVG differ so much?
    #
    # Replays the same filtering logic detect_micro_mss uses
    # internally, but tags *why* each retest was dropped, broken
    # down by zone_type. Remove this block once the asymmetry
    # is explained/resolved.
    # ---------------------------------------------------------

    from src.strategy.micro_mss_detector import (
        _prepare_swings,
        _find_reference_swings,
        _normalize_timestamp,
    )

    diag_swings = _prepare_swings(internal_swings_5m)

    diag_counts: dict[str, Counter] = {}

    for retest in all_retests:

        zone_type = str(
            getattr(retest, "zone_type", "UNKNOWN")
        )

        diag_counts.setdefault(
            zone_type, Counter()
        )["total"] += 1

        direction = str(
            getattr(retest, "direction", "")
        ).lower()

        try:
            retest_timestamp = _normalize_timestamp(
                getattr(retest, "retest_timestamp")
            )
            displacement_timestamp = _normalize_timestamp(
                getattr(retest, "displacement_timestamp")
            )
            mss_timestamp = _normalize_timestamp(
                getattr(retest, "mss_timestamp")
            )
        except Exception:
            diag_counts[zone_type]["missing_attr"] += 1
            continue

        if not (
            mss_timestamp
            < displacement_timestamp
            < retest_timestamp
        ):
            diag_counts[zone_type]["invalid_sequence"] += 1
            continue

        retest_end = retest_timestamp + pd.Timedelta(hours=1)

        reference_swing = _find_reference_swings(
            diag_swings,
            direction=direction,
            retest_end=retest_end,
        )

        if reference_swing is None:
            diag_counts[zone_type]["no_reference_swing"] += 1
            continue

        future_df = df_5m_indexed.loc[
            df_5m_indexed.index >= retest_end
        ]

        if future_df.empty:
            diag_counts[zone_type]["no_future_candles"] += 1
            continue

        diag_counts[zone_type]["passed_filters"] += 1

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: retest drop-off reasons by zone_type")
    print("=" * 70)

    for zone_type, counts in diag_counts.items():
        print(f"\n  {zone_type}:")
        for reason, n in counts.items():
            print(f"    {reason:20s}: {n}")

    # ---------------------------------------------------------
    # 15. Micro-MSS
    # ---------------------------------------------------------
    print("\n[14] Detecting Micro-MSS...")

    micro_events = detect_micro_mss(
        df_5m_indexed,
        all_retests,
        internal_swings_5m,
        max_bars_after_retest=24,
        min_reference_distance_atr=0.25,
    )

    micro_events = deduplicate_micro_mss(
        micro_events
    )

    print(
        f"Micro-MSS: "
        f"{len(micro_events):,}"
    )

    # ---------------------------------------------------------
    # 16. Results
    # ---------------------------------------------------------
    if not all_retests:
        print("\nNo retests found.")
        return

    conversion = (
        len(micro_events)
        / len(all_retests)
        * 100
    )

    print("\n" + "=" * 70)
    print("MICRO-MSS RESULTS")
    print("=" * 70)

    print(
        f"Total retests:      {len(all_retests):,}"
    )

    print(
        f"Micro-MSS:          {len(micro_events):,}"
    )

    print(
        f"Retest -> Micro-MSS: {conversion:.2f}%"
    )

    # ---------------------------------------------------------
    # Direction
    # ---------------------------------------------------------
    direction_counts = Counter(
        event.direction
        for event in micro_events
    )

    print("\nDirection:")

    for direction, count in sorted(
        direction_counts.items()
    ):
        print(
            f"  {direction:8s}: {count:,}"
        )

    # ---------------------------------------------------------
    # Zone type
    # ---------------------------------------------------------
    zone_counts = Counter(
        event.zone_type
        for event in micro_events
    )

    print("\nZone type:")

    for zone_type, count in sorted(
        zone_counts.items()
    ):
        print(
            f"  {zone_type:8s}: {count:,}"
        )

    # ---------------------------------------------------------
    # Bars after retest
    # ---------------------------------------------------------
    bars_counts = Counter(
        event.bars_after_retest
        for event in micro_events
    )

    print("\nBars after retest:")

    for bars, count in sorted(
        bars_counts.items()
    ):
        print(
            f"  {bars:2d}: {count:,}"
        )

    # ---------------------------------------------------------
    # Timing sanity
    # ---------------------------------------------------------
    invalid_sequence = 0
    invalid_before_retest = 0

    for event in micro_events:

        if not (
            event.mss_timestamp
            < event.displacement_timestamp
            < event.retest_timestamp
            < event.timestamp
        ):
            invalid_sequence += 1

        if event.timestamp <= event.retest_timestamp:
            invalid_before_retest += 1

    print("\nSanity checks:")

    print(
        f"  Invalid sequence: "
        f"{invalid_sequence}"
    )

    print(
        f"  Micro-MSS before/equal retest: "
        f"{invalid_before_retest}"
    )

    # ---------------------------------------------------------
    # DataFrame
    # ---------------------------------------------------------
    result_df = micro_mss_to_dataframe(
        micro_events
    )

    if not result_df.empty:

        print("\nFirst 20 Micro-MSS:")

        print(
            result_df[
                [
                    "timestamp",
                    "direction",
                    "zone_type",
                    "retest_timestamp",
                    "bars_after_retest",
                    "broken_swing_price",
                    "reference_distance_atr",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        print("\nLast 20 Micro-MSS:")

        print(
            result_df[
                [
                    "timestamp",
                    "direction",
                    "zone_type",
                    "retest_timestamp",
                    "bars_after_retest",
                    "broken_swing_price",
                    "reference_distance_atr",
                ]
            ]
            .tail(20)
            .to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()