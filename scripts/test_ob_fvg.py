from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR
from src.data.storage import load_year
from src.data.aggregator import resample_ohlcv

# 🌟 اصلاح خط ۹: تغییر از structure به swing_detector
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
from src.strategy.displacement_detector import detect_displacements

from src.strategy.order_block_detector import (
    detect_order_blocks,
)
from src.strategy.fvg_detector import (
    detect_fvgs,
)


SYMBOL = "BTCUSDT"
YEAR = 2025


def main():

    print("=" * 70)
    print("OB + FVG TEST")
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

    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")

    print(f"5M candles : {len(df_5m):,}")
    print(f"1H candles : {len(df_1h):,}")
    print(f"4H candles : {len(df_4h):,}")

    # ---------------------------------------------------------
    # 4H STRUCTURE
    # ---------------------------------------------------------

    swings_4h = detect_pivots(
        df_4h,
        left_bars=2,
        right_bars=2,
        timeframe="4h",
    )

    print(f"4H swings  : {len(swings_4h):,}")

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
    # SWEEP
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
        f"Internal swings : {len(internal_swings):,}"
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

    print(f"MSS        : {len(mss_events):,}")

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
        f"Displacement : {len(displacements):,}"
    )

    # ---------------------------------------------------------
    # ORDER BLOCK
    # ---------------------------------------------------------

    order_blocks = detect_order_blocks(
        df_1h,
        displacements,
        lookback=8,
    )

    print(
        f"Order Blocks  : {len(order_blocks):,}"
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
        f"FVGs          : {len(fvgs):,}"
    )

    # ---------------------------------------------------------
    # STATS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ORDER BLOCK STATS")
    print("=" * 70)

    if order_blocks:

        ob_df = pd.DataFrame(
            [
                {
                    "direction": x.direction,
                    "bars_before_displacement":
                        x.bars_before_displacement,
                    "zone_size":
                        x.zone_high - x.zone_low,
                }
                for x in order_blocks
            ]
        )

        print(
            ob_df["direction"]
            .value_counts()
            .to_string()
        )

        print(
            "\nBars before displacement:"
        )

        print(
            ob_df[
                "bars_before_displacement"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "\nOB zone size:"
        )

        print(
            ob_df["zone_size"]
            .describe()
            .to_string()
        )

    print("\n" + "=" * 70)
    print("FVG STATS")
    print("=" * 70)

    if fvgs:

        fvg_df = pd.DataFrame(
            [
                {
                    "direction": x.direction,
                    "gap_size": x.gap_size,
                    "gap_atr": x.gap_atr,
                }
                for x in fvgs
            ]
        )

        print(
            fvg_df["direction"]
            .value_counts()
            .to_string()
        )

        print("\nGap / ATR:")

        print(
            fvg_df["gap_atr"]
            .describe()
            .to_string()
        )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL PIPELINE")
    print("=" * 70)

    print(
        f"Liquidity       : {len(zones):,}"
    )
    print(
        f"Sweeps          : {len(sweeps):,}"
    )
    print(
        f"MSS             : {len(mss_events):,}"
    )
    print(
        f"Displacement    : {len(displacements):,}"
    )
    print(
        f"Order Blocks    : {len(order_blocks):,}"
    )
    print(
        f"FVG             : {len(fvgs):,}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
