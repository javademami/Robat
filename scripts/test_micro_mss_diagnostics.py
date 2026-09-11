from collections import Counter

import pandas as pd

from src.config import DATA_DIR, SYMBOL, YEAR

# FIX (high confidence, verified in this project's own traceback):
from src.data.storage import load_year

# FIX (medium confidence, based on AUDIT_REPORT's description of
# aggregator.py's role — tell me if this still fails):
from src.data.aggregator import resample_ohlcv

# FIX (high confidence, verified by reading swing_detector.py):
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

from src.strategy.retest_detector import (
    detect_ob_retests,
    detect_fvg_retests,
)

from src.strategy.micro_mss_detector import (
    detect_micro_mss,
    deduplicate_micro_mss,
)


def get_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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


def main():

    print("=" * 70)
    print("MICRO-MSS DIAGNOSTICS")
    print("=" * 70)

    # =========================================================
    # DATA
    # =========================================================

    print("\n[1] Loading data...")

    df_5m = load_year(
        DATA_DIR,
        SYMBOL,
        "5m",
        YEAR,
    )

    df_1h = resample_ohlcv(
        df_5m,
        "1h",
    )

    df_4h = resample_ohlcv(
        df_5m,
        "4h",
    )

    print(f"5M: {len(df_5m):,}")
    print(f"1H: {len(df_1h):,}")
    print(f"4H: {len(df_4h):,}")

    # =========================================================
    # 4H STRUCTURE
    # =========================================================

    print("\n[2] 4H structure...")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=2,
        right_bars=2,
        timeframe="4h",
    )

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

    sweeps = detect_sweeps(
        df_1h,
        zones,
        atr_period=14,
    )

    # =========================================================
    # 1H MSS
    # =========================================================

    print("\n[3] 1H MSS...")

    internal_swings_1h = detect_pivots(
        df_1h,
        left_bars=2,
        right_bars=2,
        timeframe="1h",
    )

    mss_events = detect_mss(
        df_1h,
        sweeps,
        internal_swings_1h,
        max_bars_after_sweep=12,
    )

    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=14,
        min_body_atr=1.0,
        bullish_clv=0.70,
        bearish_clv=0.30,
        max_bars_after_mss=8,
    )

    # =========================================================
    # OB / FVG
    # =========================================================

    print("\n[4] OB / FVG...")

    ob_events = detect_order_blocks(
        df_1h,
        displacement_events,
        lookback=8,
    )

    fvg_events = detect_fvgs(
        df_1h,
        displacement_events,
        search_bars=3,
        atr_period=14,
    )

    ob_retests = detect_ob_retests(
        df_1h,
        ob_events,
        max_bars_after_zone=24,
    )

    fvg_retests = detect_fvg_retests(
        df_1h,
        fvg_events,
        max_bars_after_zone=24,
    )

    ob_retests = list(ob_retests)
    fvg_retests = list(fvg_retests)

    all_retests = (
        ob_retests
        + fvg_retests
    )

    print(f"OB retests:  {len(ob_retests)}")
    print(f"FVG retests: {len(fvg_retests)}")
    print(f"Total:       {len(all_retests)}")

    # =========================================================
    # 5M STRUCTURE
    # =========================================================

    print("\n[5] 5M structure...")

    internal_swings_5m = detect_pivots(
        df_5m,
        left_bars=2,
        right_bars=2,
        timeframe="5m",
    )

    # ---------------------------------------------------------
    # FIX: micro_mss_detector requires df_5m to carry a
    # DatetimeIndex, unlike the rest of the pipeline which uses
    # a plain "timestamp" column. detect_pivots above still needs
    # the column-based df_5m, so build an indexed copy just for
    # this call.
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

    micro_events = detect_micro_mss(
        df_5m_indexed,
        all_retests,
        internal_swings_5m,
        max_bars_after_retest=24,
    )

    micro_events = deduplicate_micro_mss(
        micro_events
    )

    print(f"Micro-MSS: {len(micro_events)}")

    # =========================================================
    # 1. OB VS FVG CONVERSION
    # =========================================================

    print("\n" + "=" * 70)
    print("1. ZONE TYPE CONVERSION")
    print("=" * 70)

    micro_by_retest = Counter(
        event.zone_type
        for event in micro_events
    )

    ob_micro = micro_by_retest.get("OB", 0)
    fvg_micro = micro_by_retest.get("FVG", 0)

    print(
        f"OB:  {len(ob_retests):3d} retests "
        f"-> {ob_micro:3d} Micro-MSS "
        f"= {ob_micro / len(ob_retests) * 100:.2f}%"
        if ob_retests
        else "OB: no retests"
    )

    print(
        f"FVG: {len(fvg_retests):3d} retests "
        f"-> {fvg_micro:3d} Micro-MSS "
        f"= {fvg_micro / len(fvg_retests) * 100:.2f}%"
        if fvg_retests
        else "FVG: no retests"
    )

    # =========================================================
    # 2. MICRO-MSS TIMING
    # =========================================================

    print("\n" + "=" * 70)
    print("2. MICRO-MSS TIMING")
    print("=" * 70)

    timing_buckets = {
        "1": 0,
        "2-4": 0,
        "5-8": 0,
        "9-16": 0,
        "17-24": 0,
    }

    for event in micro_events:

        n = event.bars_after_retest

        if n == 1:
            timing_buckets["1"] += 1
        elif n <= 4:
            timing_buckets["2-4"] += 1
        elif n <= 8:
            timing_buckets["5-8"] += 1
        elif n <= 16:
            timing_buckets["9-16"] += 1
        else:
            timing_buckets["17-24"] += 1

    for bucket, count in timing_buckets.items():

        pct = (
            count / len(micro_events) * 100
            if micro_events
            else 0
        )

        print(
            f"{bucket:>5s} bars: "
            f"{count:3d} "
            f"({pct:6.2f}%)"
        )

    # =========================================================
    # 3. DISTANCE FROM RETEST CLOSE TO BROKEN SWING
    # =========================================================

    print("\n" + "=" * 70)
    print("3. RETEST -> BROKEN SWING DISTANCE")
    print("=" * 70)

    # ---------------------------------------------------------
    # FIX: this section looks up candles by timestamp via
    # df_1h.loc[...] / "in df_1h.index", which requires a
    # DatetimeIndex. df_1h itself (used above for detect_pivots,
    # detect_sweeps, detect_mss, ...) keeps its normal "timestamp"
    # column, so build a separate indexed copy here, and compute
    # ATR on that copy so its index lines up for the .loc lookups.
    # ---------------------------------------------------------

    df_1h_indexed = df_1h.copy()

    df_1h_indexed["timestamp"] = pd.to_datetime(
        df_1h_indexed["timestamp"],
        utc=True,
    )

    df_1h_indexed = (
        df_1h_indexed
        .set_index("timestamp")
        .sort_index()
    )

    atr_1h = get_atr(
        df_1h_indexed,
        period=14,
    )

    rows = []

    for event in micro_events:

        retest_timestamp = pd.Timestamp(
            event.retest_timestamp
        )

        if retest_timestamp not in df_1h_indexed.index:
            continue

        retest_candle = df_1h_indexed.loc[
            retest_timestamp
        ]

        retest_close = float(
            retest_candle["close"]
        )

        atr = float(
            atr_1h.loc[retest_timestamp]
        )

        if atr <= 0 or pd.isna(atr):
            continue

        broken_price = float(
            event.broken_swing_price
        )

        distance = abs(
            broken_price - retest_close
        )

        distance_atr = distance / atr

        distance_pct = (
            distance
            / retest_close
            * 100
        )

        rows.append(
            {
                "timestamp": event.timestamp,
                "zone_type": event.zone_type,
                "direction": event.direction,
                "retest_timestamp": retest_timestamp,
                "retest_close": retest_close,
                "broken_swing": broken_price,
                "distance": distance,
                "distance_atr": distance_atr,
                "distance_pct": distance_pct,
                "bars_after_retest": event.bars_after_retest,
            }
        )

    distance_df = pd.DataFrame(rows)

    if distance_df.empty:

        print("No distance data.")

    else:

        print(
            f"Mean distance:   "
            f"{distance_df['distance_atr'].mean():.3f} ATR"
        )

        print(
            f"Median distance: "
            f"{distance_df['distance_atr'].median():.3f} ATR"
        )

        print(
            f"Min distance:    "
            f"{distance_df['distance_atr'].min():.3f} ATR"
        )

        print(
            f"Max distance:    "
            f"{distance_df['distance_atr'].max():.3f} ATR"
        )

        print(
            f"Mean %:           "
            f"{distance_df['distance_pct'].mean():.3f}%"
        )

        print("\nBy zone type:")

        grouped = (
            distance_df
            .groupby("zone_type")["distance_atr"]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max",
                ]
            )
        )

        print(
            grouped.to_string(
                float_format=lambda x: f"{x:.3f}"
            )
        )

    # =========================================================
    # 4. DISTANCE BUCKETS
    # =========================================================

    print("\n" + "=" * 70)
    print("4. DISTANCE BUCKETS")
    print("=" * 70)

    if not distance_df.empty:

        buckets = {
            "< 0.25 ATR": (
                distance_df["distance_atr"] < 0.25
            ),
            "0.25 - 0.50 ATR": (
                (distance_df["distance_atr"] >= 0.25)
                & (distance_df["distance_atr"] < 0.50)
            ),
            "0.50 - 1.00 ATR": (
                (distance_df["distance_atr"] >= 0.50)
                & (distance_df["distance_atr"] < 1.00)
            ),
            "1.00 - 2.00 ATR": (
                (distance_df["distance_atr"] >= 1.00)
                & (distance_df["distance_atr"] < 2.00)
            ),
            ">= 2.00 ATR": (
                distance_df["distance_atr"] >= 2.00
            ),
        }

        for name, mask in buckets.items():

            count = int(mask.sum())

            pct = (
                count
                / len(distance_df)
                * 100
            )

            print(
                f"{name:>18s}: "
                f"{count:3d} "
                f"({pct:6.2f}%)"
            )

    # =========================================================
    # 5. MICRO-MSS DETAIL
    # =========================================================

    print("\n" + "=" * 70)
    print("5. MICRO-MSS DETAIL")
    print("=" * 70)

    if not distance_df.empty:

        display_columns = [
            "timestamp",
            "zone_type",
            "direction",
            "retest_timestamp",
            "bars_after_retest",
            "distance_atr",
            "distance_pct",
        ]

        print(
            distance_df[
                display_columns
            ]
            .sort_values("timestamp")
            .to_string(index=False)
        )

    # =========================================================
    # 6. FINAL SUMMARY
    # =========================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"OB Retests:       {len(ob_retests)}"
    )

    print(
        f"FVG Retests:      {len(fvg_retests)}"
    )

    print(
        f"Total Retests:    {len(all_retests)}"
    )

    print(
        f"Micro-MSS:        {len(micro_events)}"
    )

    if ob_retests:
        print(
            f"OB conversion:    "
            f"{ob_micro / len(ob_retests) * 100:.2f}%"
        )

    if fvg_retests:
        print(
            f"FVG conversion:   "
            f"{fvg_micro / len(fvg_retests) * 100:.2f}%"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()