from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import DATA_DIR

from src.data.aggregator import (
    load_5m_file,
    build_timeframes,
)

from src.strategy.swing_detector import (
    detect_pivots,
)

from src.strategy.liquidity import (
    detect_equal_highs,
    detect_equal_lows,
    detect_swing_liquidity,
    detect_previous_day_liquidity,
)


def main():

    symbol = "BTCUSDT"
    year = 2025

    path = (
        DATA_DIR
        / symbol
        / "5m"
        / f"{year}.parquet"
    )

    print("=" * 70)
    print("ROBAt LIQUIDITY ENGINE TEST")
    print("=" * 70)

    df_5m = load_5m_file(path)

    timeframes = build_timeframes(
        df_5m
    )

    df_1h = timeframes["1h"]
    df_4h = timeframes["4h"]

    print()
    print(
        f"5M candles: {len(df_5m):,}"
    )

    print(
        f"1H candles: {len(df_1h):,}"
    )

    print(
        f"4H candles: {len(df_4h):,}"
    )

    print()
    print("-" * 70)
    print("4H SWING LIQUIDITY")
    print("-" * 70)

    swings = detect_pivots(
        df_4h,
        left_bars=2,
        right_bars=2,
        timeframe="4h",
    )

    swing_liquidity = detect_swing_liquidity(
        swings
    )

    print(
        f"Swing liquidity zones: "
        f"{len(swing_liquidity):,}"
    )

    for zone in swing_liquidity[-10:]:

        print(
            f"{zone.created_at} | "
            f"{zone.liquidity_type.value:12} | "
            f"{zone.price:,.2f} | "
            f"strength={zone.strength}"
        )

    print()
    print("-" * 70)
    print("1H EQUAL HIGH / LOW")
    print("-" * 70)

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

    print(
        f"Equal High zones: "
        f"{len(equal_highs):,}"
    )

    print(
        f"Equal Low zones:  "
        f"{len(equal_lows):,}"
    )

    print()
    print("Latest Equal Highs:")

    for zone in equal_highs[-10:]:

        print(
            f"{zone.created_at} | "
            f"{zone.price:,.2f}"
        )

    print()
    print("Latest Equal Lows:")

    for zone in equal_lows[-10:]:

        print(
            f"{zone.created_at} | "
            f"{zone.price:,.2f}"
        )

    print()
    print("-" * 70)
    print("PREVIOUS DAY LIQUIDITY")
    print("-" * 70)

    pd_zones = detect_previous_day_liquidity(
        df_1h
    )

    print(
        f"PDH/PDL zones: "
        f"{len(pd_zones):,}"
    )

    for zone in pd_zones[-10:]:

        print(
            f"{zone.created_at} | "
            f"{zone.liquidity_type.value:4} | "
            f"{zone.price:,.2f} | "
            f"strength={zone.strength}"
        )

    print()
    print("=" * 70)
    print("LIQUIDITY ENGINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
