from pathlib import Path

import pandas as pd

from src.data.storage import load_year
# تغییر خط زیر: استفاده از resample_ohlcv به جای تابع منسوخ شده
from src.data.aggregator import resample_ohlcv
from src.strategy.swing_detector import detect_pivots
from src.strategy.liquidity import (
    detect_equal_highs,
    detect_equal_lows,
    detect_previous_day_liquidity,
    detect_swing_liquidity,
    deduplicate_zones,
)
from src.strategy.sweep_detector import (
    detect_sweeps,
    sweeps_to_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

SYMBOL = "BTCUSDT"
YEAR = 2025


def main():

    print("=" * 70)
    print("SWEEP ENGINE TEST")
    print("=" * 70)

    print("\n[1] Loading 5M data...")

    df_5m = load_year(
        DATA_DIR,
        SYMBOL,
        "5m",
        YEAR,
    )

    print(f"5M candles: {len(df_5m):,}")

    print("\n[2] Building 1H and 4H...")

    # تغییر خط زیر
    df_1h = resample_ohlcv(
        df_5m,
        "1h",
    )

    # تغییر خط زیر
    df_4h = resample_ohlcv(
        df_5m,
        "4h",
    )

    print(f"1H candles: {len(df_1h):,}")
    print(f"4H candles: {len(df_4h):,}")

    print("\n[3] Detecting 4H swings...")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=2,
        right_bars=2,
        timeframe="4h",
    )

    print(f"4H swings: {len(swings_4h):,}")

    print("\n[4] Detecting liquidity...")

    swing_zones = detect_swing_liquidity(
        swings_4h,
    )

    equal_highs = detect_equal_highs(
        df_1h,
        tolerance_atr=0.15,
        atr_period=14,
        lookback=100,
    )

    equal_lows = detect_equal_lows(
        df_1h,
        tolerance_atr=0.15,
        atr_period=14,
        lookback=100,
    )

    previous_day = detect_previous_day_liquidity(
        df_1h,
    )

    zones = deduplicate_zones(
        swing_zones
        + equal_highs
        + equal_lows
        + previous_day
    )

    print(f"Liquidity zones: {len(zones):,}")

    print("\n[5] Detecting sweeps...")

    sweeps = detect_sweeps(
        df_1h,
        zones,
        atr_period=14,
    )

    print(f"Total sweeps: {len(sweeps):,}")

    if not sweeps:
        print("\nNo sweeps detected.")
        return

    result = sweeps_to_dataframe(
        sweeps
    )

    print("\n" + "=" * 70)
    print("LATEST SWEEPS")
    print("=" * 70)

    print(
        result.tail(30).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)

    print(
        "\nDirection:"
    )

    print(
        result["direction"]
        .value_counts()
    )

    print(
        "\nLiquidity type:"
    )

    print(
        result["liquidity_type"]
        .value_counts()
    )

    print(
        "\nDepth class:"
    )

    print(
        result["depth_class"]
        .value_counts()
    )

    print(
        "\nAverage depth ATR:"
    )

    print(
        round(
            result["depth_atr"].mean(),
            3,
        )
    )


if __name__ == "__main__":
    main()
