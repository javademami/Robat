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

from src.strategy.market_structure import (
    classify_swings,
    determine_bias,
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
    print("ROBAt MARKET STRUCTURE TEST")
    print("=" * 70)

    df_5m = load_5m_file(path)

    timeframes = build_timeframes(
        df_5m
    )

    df_4h = timeframes["4h"]

    print()
    print(
        f"4H candles: {len(df_4h):,}"
    )

    swings = detect_pivots(
        df_4h,
        left_bars=2,
        right_bars=2,
        timeframe="4h",
    )

    print(
        f"Confirmed pivots available "
        f"in dataset: {len(swings):,}"
    )

    events = classify_swings(
        swings
    )

    print(
        f"Structure events: {len(events):,}"
    )

    print()
    print("-" * 70)
    print("LAST 20 STRUCTURE EVENTS")
    print("-" * 70)

    for event in events[-20:]:

        print(
            f"{event.confirmed_at} | "
            f"{event.kind:2} | "
            f"{event.price:,.2f}"
        )

    bias = determine_bias(
        events
    )

    print()
    print("-" * 70)
    print(
        f"CURRENT STRUCTURE BIAS: {bias}"
    )
    print("-" * 70)


if __name__ == "__main__":
    main()
