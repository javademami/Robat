from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR
from src.data.storage import load_year
from src.data.aggregator import resample_ohlcv

# 🌟 اصلاح آدرس فایل: تغییر از structure به swing_detector
from src.strategy.swing_detector import detect_pivots

from src.strategy.liquidity import (
    detect_swing_liquidity,
    detect_equal_highs,
    detect_equal_lows,
    detect_previous_day_liquidity,
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


SYMBOL = "BTCUSDT"
YEAR = 2025


def main():

    print("=" * 70)
    print("RETEST TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

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

    print(f"5M candles : {len(df_5m):,}")
    print(f"1H candles : {len(df_1h):,}")
    print(f"4H candles : {len(df_4h):,}")

    # ---------------------------------------------------------
    # STRUCTURE
    # ---------------------------------------------------------

    swings_4h = detect_pivots(
        df_4h,
        left_bars=2,
        right_bars=2,
        timeframe="4h",
    )

    # ---------------------------------------------------------
    # LIQUIDITY
    # ---------------------------------------------------------

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

    print(f"Liquidity  : {len(zones):,}")

    # ---------------------------------------------------------
    # SWEEPS
    # ---------------------------------------------------------

    sweeps = detect_sweeps(
        df_1h,
        zones,
        atr_period=14,
    )

    print(f"Sweeps     : {len(sweeps):,}")

    # ---------------------------------------------------------
    # INTERNAL STRUCTURE
    # ---------------------------------------------------------

    internal_swings = detect_pivots(
        df_1h,
        left_bars=2,
        right_bars=2,
        timeframe="1h",
    )

    print(
        f"Internal swings : "
        f"{len(internal_swings):,}"
    )

    # ---------------------------------------------------------
    # MSS
    # ---------------------------------------------------------

    mss_events = detect_mss(
        df_1h,
        sweeps,
        internal_swings,
        max_bars_after_sweep=12,
    )

    print(
        f"MSS        : "
        f"{len(mss_events):,}"
    )

    # ---------------------------------------------------------
    # DISPLACEMENT
    # ---------------------------------------------------------

    displacements = detect_displacements(
        df_1h,
        mss_events,
        atr_period=14,
        min_body_atr=1.0,
        bullish_clv=0.70,
        bearish_clv=0.30,
        max_bars_after_mss=8,
    )

    print(
        f"Displacement : "
        f"{len(displacements):,}"
    )

    # ---------------------------------------------------------
    # OB
    # ---------------------------------------------------------

    order_blocks = detect_order_blocks(
        df_1h,
        displacements,
        lookback=8,
    )

    print(
        f"Order Blocks  : "
        f"{len(order_blocks):,}"
    )

    # ---------------------------------------------------------
    # FVG
    # ---------------------------------------------------------

    fvgs = detect_fvgs(
        df_1h,
        displacements,
        atr_period=14,
        search_bars=3,
    )

    print(
        f"FVGs          : "
        f"{len(fvgs):,}"
    )

    # ---------------------------------------------------------
    # RETEST OB
    # ---------------------------------------------------------

    ob_retests = detect_ob_retests(
        df_1h,
        order_blocks,
        max_bars_after_zone=24,
    )

    # ---------------------------------------------------------
    # RETEST FVG
    # ---------------------------------------------------------

    fvg_retests = detect_fvg_retests(
        df_1h,
        fvgs,
        max_bars_after_zone=24,
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETEST RESULTS")
    print("=" * 70)

    print(
        f"OB Retests  : "
        f"{len(ob_retests):,} / "
        f"{len(order_blocks):,}"
    )

    print(
        f"FVG Retests : "
        f"{len(fvg_retests):,} / "
        f"{len(fvgs):,}"
    )

    # ---------------------------------------------------------
    # OB STATS
    # ---------------------------------------------------------

    if ob_retests:

        df_ob = pd.DataFrame(
            [
                {
                    "direction": x.direction,
                    "bars_after_zone": x.bars_after_zone,
                    "penetration_ratio": x.penetration_ratio,
                    "close_inside": x.close_inside,
                    "close_through": x.close_through,
                }
                for x in ob_retests
            ]
        )

        print("\n" + "-" * 70)
        print("OB RETEST STATS")
        print("-" * 70)

        print("\nDirection:")
        print(df_ob["direction"].value_counts().to_string())

        print("\nBars after zone:")
        print(df_ob["bars_after_zone"].value_counts().sort_index().to_string())

        print("\nPenetration ratio:")
        print(df_ob["penetration_ratio"].describe().to_string())

        print("\nClose inside:")
        print(df_ob["close_inside"].value_counts().to_string())

        print("\nClose through:")
        print(df_ob["close_through"].value_counts().to_string())

    # ---------------------------------------------------------
    # FVG STATS (بخش ناقص کات شده کاملاً بازسازی شد)
    # ---------------------------------------------------------

    if fvg_retests:

        df_fvg = pd.DataFrame(
            [
                {
                    "direction": x.direction,
                    "bars_after_zone": x.bars_after_zone,
                    "penetration_ratio": x.penetration_ratio,
                    "close_inside": x.close_inside,
                    "close_through": x.close_through,
                }
                for x in fvg_retests
            ]
        )

        print("\n" + "-" * 70)
        print("FVG RETEST STATS")
        print("-" * 70)

        print("\nDirection:")
        print(df_fvg["direction"].value_counts().to_string())

        print("\nBars after zone:")
        print(df_fvg["bars_after_zone"].value_counts().sort_index().to_string())

        print("\nPenetration ratio:")
        print(df_fvg["penetration_ratio"].describe().to_string())

        print("\nClose inside:")
        print(df_fvg["close_inside"].value_counts().to_string())

        print("\nClose through:")
        print(df_fvg["close_through"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("FINAL PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Liquidity       : {len(zones):,}")
    print(f"Sweeps          : {len(sweeps):,}")
    print(f"MSS             : {len(mss_events):,}")
    print(f"Displacement    : {len(displacements):,}")
    print(f"Order Blocks    : {len(order_blocks):,}")
    print(f"FVGs            : {len(fvgs):,}")
    print(f"OB Retests      : {len(ob_retests):,}")
    print(f"FVG Retests     : {len(fvg_retests):,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
