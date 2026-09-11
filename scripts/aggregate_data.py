from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.data.aggregator import (
    load_5m_file,
    build_timeframes,
    save_timeframes,
)


def main():

    symbol = "BTCUSDT"
    year = 2025

    source = (
        DATA_DIR
        / symbol
        / "5m"
        / f"{year}.parquet"
    )

    output_dir = (
        DATA_DIR
        / symbol
    )

    print("=" * 70)
    print("ROBAt TIMEFRAME AGGREGATOR")
    print("=" * 70)

    print(f"Source: {source}")

    df_5m = load_5m_file(source)

    print()
    print(f"5M candles: {len(df_5m):,}")
    print(f"Start: {df_5m['timestamp'].min()}")
    print(f"End:   {df_5m['timestamp'].max()}")

    timeframes = build_timeframes(
        df_5m
    )

    print()
    print(f"1H candles: {len(timeframes['1h']):,}")
    print(f"4H candles: {len(timeframes['4h']):,}")

    saved = save_timeframes(
        timeframes,
        output_dir,
        year,
    )

    print()
    print("Saved files:")

    for timeframe, path in saved.items():
        print(
            f"{timeframe}: {path}"
        )

    print()
    print("=" * 70)
    print("AGGREGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
