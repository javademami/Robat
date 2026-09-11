from pathlib import Path

import pandas as pd

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

from src.strategy.sweep_detector import (
    detect_sweeps,
)

from src.strategy.mss_detector import (
    detect_mss,
    mss_to_dataframe,
)

from src.strategy.displacement_detector import (
    detect_displacements,
    displacements_to_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

SYMBOL = "BTCUSDT"
YEAR = 2025

PIVOT_WINDOW = 3

ATR_PERIOD = 14

MIN_BODY_ATR = 1.0

BULLISH_CLV = 0.70
BEARISH_CLV = 0.30

MAX_BARS_AFTER_MSS = 8


def main():

    print("=" * 80)
    print("DISPLACEMENT ENGINE TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    print("\n[1] Loading 5M data...")

    df_5m = load_year(
        DATA_DIR,
        SYMBOL,
        "5m",
        YEAR,
    )

    print(
        f"5M candles: {len(df_5m):,}"
    )

    # ---------------------------------------------------------
    # 2. BUILD TIMEFRAMES
    # ---------------------------------------------------------

    print("\n[2] Building 1H and 4H...")

    df_1h = resample_ohlcv(
        df_5m,
        "1h",
    )

    df_4h = resample_ohlcv(
        df_5m,
        "4h",
    )

    print(
        f"1H candles: {len(df_1h):,}"
    )

    print(
        f"4H candles: {len(df_4h):,}"
    )

    # ---------------------------------------------------------
    # 3. 4H SWINGS
    # ---------------------------------------------------------

    print("\n[3] Detecting 4H swings...")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=2,
        right_bars=2,
        timeframe="4h",
    )

    print(
        f"4H swings: {len(swings_4h):,}"
    )

    # ---------------------------------------------------------
    # 4. LIQUIDITY
    # ---------------------------------------------------------

    print("\n[4] Detecting liquidity...")

    swing_zones = detect_swing_liquidity(
        swings_4h,
    )

    equal_highs = detect_equal_highs(
        df_1h,
        tolerance_atr=0.15,
        atr_period=14,
        lookback=100,
        pivot_window=PIVOT_WINDOW,
    )

    equal_lows = detect_equal_lows(
        df_1h,
        tolerance_atr=0.15,
        atr_period=14,
        lookback=100,
        pivot_window=PIVOT_WINDOW,
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

    print(
        f"Equal highs: {len(equal_highs):,}"
    )

    print(
        f"Equal lows:  {len(equal_lows):,}"
    )

    print(
        f"Total zones: {len(zones):,}"
    )

    # ---------------------------------------------------------
    # 5. SWEEPS
    # ---------------------------------------------------------

    print("\n[5] Detecting sweeps...")

    sweeps = detect_sweeps(
        df_1h,
        zones,
        atr_period=ATR_PERIOD,
    )

    print(
        f"Sweeps: {len(sweeps):,}"
    )

    if not sweeps:
        print("\nNo sweeps detected.")
        return

    # ---------------------------------------------------------
    # 6. INTERNAL 1H SWINGS
    # ---------------------------------------------------------

    print("\n[6] Detecting 1H internal swings...")

    internal_swings = detect_pivots(
        df_1h,
        left_bars=2,
        right_bars=2,
        timeframe="1h",
    )

    print(
        f"1H internal swings: "
        f"{len(internal_swings):,}"
    )

    # ---------------------------------------------------------
    # 7. MSS
    # ---------------------------------------------------------

    print("\n[7] Detecting MSS...")

    mss_events = detect_mss(
        df_1h,
        sweeps,
        internal_swings,
        max_bars_after_sweep=12,
    )

    print(
        f"MSS: {len(mss_events):,}"
    )

    if not mss_events:
        print("\nNo MSS detected.")
        return

    # ---------------------------------------------------------
    # 8. DISPLACEMENT
    # ---------------------------------------------------------

    print("\n[8] Detecting displacement...")

    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=MIN_BODY_ATR,
        bullish_clv=BULLISH_CLV,
        bearish_clv=BEARISH_CLV,
        max_bars_after_mss=MAX_BARS_AFTER_MSS,
    )

    print(
        f"Displacements: "
        f"{len(displacement_events):,}"
    )

    # ---------------------------------------------------------
    # 9. CONVERSION
    # ---------------------------------------------------------

    conversion = (
        len(displacement_events)
        / len(mss_events)
        * 100
    )

    print(
        f"\nMSS -> Displacement: "
        f"{conversion:.2f}%"
    )

    # ---------------------------------------------------------
    # 10. DATAFRAME
    # ---------------------------------------------------------

    result = displacements_to_dataframe(
        displacement_events
    )

    if result.empty:
        print("\nNo displacement events.")
        return

    # ---------------------------------------------------------
    # 11. SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("DISPLACEMENT SUMMARY")
    print("=" * 80)

    print("\nDirection:")

    print(
        result["direction"].value_counts()
    )

    print("\nBars after MSS:")

    print(
        result["bars_after_mss"]
        .value_counts()
        .sort_index()
    )

    print("\nBody / ATR:")

    print(
        result["body_atr"].describe()[
            [
                "count",
                "mean",
                "50%",
                "min",
                "max",
            ]
        ]
    )

    print("\nCLV:")

    print(
        result["clv"].describe()[
            [
                "count",
                "mean",
                "50%",
                "min",
                "max",
            ]
        ]
    )

    print("\nBullish displacement:")

    bullish = result[
        result["direction"] == "bullish"
    ]

    if not bullish.empty:
        print(
            f"Count: {len(bullish):,}"
        )

        print(
            f"Average Body/ATR: "
            f"{bullish['body_atr'].mean():.3f}"
        )

        print(
            f"Average CLV: "
            f"{bullish['clv'].mean():.3f}"
        )

    print("\nBearish displacement:")

    bearish = result[
        result["direction"] == "bearish"
    ]

    if not bearish.empty:
        print(
            f"Count: {len(bearish):,}"
        )

        print(
            f"Average Body/ATR: "
            f"{bearish['body_atr'].mean():.3f}"
        )

        print(
            f"Average CLV: "
            f"{bearish['clv'].mean():.3f}"
        )

    # ---------------------------------------------------------
    # 12. LATEST EVENTS
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("LATEST DISPLACEMENTS")
    print("=" * 80)

    display_columns = [
        "timestamp",
        "direction",
        "mss_timestamp",
        "bars_after_mss",
        "close",
        "body_atr",
        "clv",
    ]

    print(
        result[
            display_columns
        ]
        .tail(30)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 13. BASIC SANITY CHECKS
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    invalid_body = (
        result["body_atr"]
        < MIN_BODY_ATR
    ).sum()

    print(
        f"Invalid Body/ATR: "
        f"{invalid_body}"
    )

    bullish_invalid_clv = (
        (
            result["direction"]
            == "bullish"
        )
        &
        (
            result["clv"]
            < BULLISH_CLV
        )
    ).sum()

    bearish_invalid_clv = (
        (
            result["direction"]
            == "bearish"
        )
        &
        (
            result["clv"]
            > BEARISH_CLV
        )
    ).sum()

    print(
        f"Invalid bullish CLV: "
        f"{bullish_invalid_clv}"
    )

    print(
        f"Invalid bearish CLV: "
        f"{bearish_invalid_clv}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()