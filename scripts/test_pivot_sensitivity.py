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

from src.strategy.sweep_detector import detect_sweeps

from src.strategy.mss_detector import detect_mss


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SYMBOL = "BTCUSDT"
YEAR = 2025

PIVOT_WINDOWS = [2, 3, 5]

EQUAL_TOLERANCE_ATR = 0.15
ATR_PERIOD = 14
LOOKBACK = 100
MAX_BARS_AFTER_SWEEP = 12


def pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (part / total) * 100.0


def main():
    print("=" * 80)
    print("PIVOT WINDOW SENSITIVITY TEST")
    print("=" * 80)

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
    # 2. Aggregate once
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

    print(f"1H candles: {len(df_1h):,}")
    print(f"4H candles: {len(df_4h):,}")

    # ---------------------------------------------------------
    # 3. Fixed 4H structural liquidity
    # ---------------------------------------------------------
    print("\n[3] Building 4H swing liquidity...")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=2,
        right_bars=2,
        timeframe="4h",
    )

    swing_zones = detect_swing_liquidity(
        swings_4h,
    )

    previous_day = detect_previous_day_liquidity(
        df_1h,
    )

    print(f"4H swings: {len(swings_4h):,}")
    print(f"4H swing zones: {len(swing_zones):,}")
    print(f"Previous-day zones: {len(previous_day):,}")

    # ---------------------------------------------------------
    # 4. Internal swings for MSS
    #
    # Keep this FIXED.
    # We are testing liquidity pivot_window only.
    # ---------------------------------------------------------
    print("\n[4] Building fixed internal swings for MSS...")

    internal_swings = detect_pivots(
        df_1h,
        left_bars=2,
        right_bars=2,
        timeframe="1h",
    )

    print(f"1H internal swings: {len(internal_swings):,}")

    # ---------------------------------------------------------
    # 5. Sensitivity loop
    # ---------------------------------------------------------
    results = []

    for pivot_window in PIVOT_WINDOWS:

        print("\n" + "-" * 80)
        print(f"PIVOT WINDOW = {pivot_window}")
        print("-" * 80)

        # -----------------------------------------------------
        # Equal highs
        # -----------------------------------------------------
        equal_highs = detect_equal_highs(
            df_1h,
            tolerance_atr=EQUAL_TOLERANCE_ATR,
            atr_period=ATR_PERIOD,
            lookback=LOOKBACK,
            pivot_window=pivot_window,
        )

        # -----------------------------------------------------
        # Equal lows
        # -----------------------------------------------------
        equal_lows = detect_equal_lows(
            df_1h,
            tolerance_atr=EQUAL_TOLERANCE_ATR,
            atr_period=ATR_PERIOD,
            lookback=LOOKBACK,
            pivot_window=pivot_window,
        )

        # -----------------------------------------------------
        # Combine + deduplicate
        # -----------------------------------------------------
        zones = deduplicate_zones(
            swing_zones
            + equal_highs
            + equal_lows
            + previous_day
        )

        print(f"Equal highs: {len(equal_highs):,}")
        print(f"Equal lows:  {len(equal_lows):,}")
        print(f"Total zones: {len(zones):,}")

        # -----------------------------------------------------
        # Sweeps
        # -----------------------------------------------------
        print("Detecting sweeps...")

        sweeps = detect_sweeps(
            df_1h,
            zones,
            atr_period=ATR_PERIOD,
        )

        print(f"Sweeps: {len(sweeps):,}")

        # -----------------------------------------------------
        # MSS
        # -----------------------------------------------------
        print("Detecting MSS...")

        mss_events = detect_mss(
            df_1h,
            sweeps,
            internal_swings,
            max_bars_after_sweep=MAX_BARS_AFTER_SWEEP,
        )

        print(f"MSS: {len(mss_events):,}")

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------
        days = max(
            (
                df_1h["timestamp"].max()
                - df_1h["timestamp"].min()
            ).total_seconds() / 86400.0,
            1.0,
        )

        sweeps_per_day = len(sweeps) / days
        mss_per_day = len(mss_events) / days

        conversion = pct(
            len(mss_events),
            len(sweeps),
        )

        results.append(
            {
                "pivot_window": pivot_window,
                "equal_highs": len(equal_highs),
                "equal_lows": len(equal_lows),
                "zones": len(zones),
                "sweeps": len(sweeps),
                "MSS": len(mss_events),
                "sweep_to_mss_%": round(conversion, 2),
                "sweeps_per_day": round(sweeps_per_day, 2),
                "MSS_per_day": round(mss_per_day, 2),
            }
        )

    # ---------------------------------------------------------
    # 6. Final comparison
    # ---------------------------------------------------------
    result_df = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("FINAL SENSITIVITY RESULTS")
    print("=" * 80)

    print(
        result_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 7. Compare against current baseline pivot_window=3
    # ---------------------------------------------------------
    baseline = result_df[
        result_df["pivot_window"] == 3
    ]

    if not baseline.empty:
        base = baseline.iloc[0]

        print("\n")
        print("=" * 80)
        print("BASELINE = pivot_window 3")
        print("=" * 80)

        for _, row in result_df.iterrows():

            if row["pivot_window"] == 3:
                continue

            zone_change = pct(
                row["zones"] - base["zones"],
                base["zones"],
            )

            sweep_change = pct(
                row["sweeps"] - base["sweeps"],
                base["sweeps"],
            )

            mss_change = pct(
                row["MSS"] - base["MSS"],
                base["MSS"],
            )

            print(
                f"\npivot_window={int(row['pivot_window'])}"
            )

            print(
                f"  zones:  "
                f"{int(row['zones']):,} "
                f"({zone_change:+.1f}%)"
            )

            print(
                f"  sweeps: "
                f"{int(row['sweeps']):,} "
                f"({sweep_change:+.1f}%)"
            )

            print(
                f"  MSS:    "
                f"{int(row['MSS']):,} "
                f"({mss_change:+.1f}%)"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()