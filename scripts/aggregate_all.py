"""
Aggregate 5m data into 1h and 4h for all symbols.
"""

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


def aggregate_one(symbol: str, year: int):

    print()
    print("=" * 70)
    print(f"AGGREGATING {symbol} {year}")
    print("=" * 70)

    source = DATA_DIR / symbol / "5m" / f"{year}.parquet"

    if not source.exists():
        print(f"  ✗ file not found: {source}")
        return

    df_5m = load_5m_file(source)

    print(f"5M candles: {len(df_5m):,}")

    timeframes = build_timeframes(df_5m)

    print(f"1H candles: {len(timeframes['1h']):,}")
    print(f"4H candles: {len(timeframes['4h']):,}")

    saved = save_timeframes(
        timeframes,
        DATA_DIR / symbol,
        year,
    )

    print("Saved:")
    for timeframe, path in saved.items():
        print(f"  {timeframe}: {path}")


def main():

    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]

    year = 2025

    for symbol in symbols:
        try:
            aggregate_one(symbol, year)
        except Exception as exc:
            print(f"  ✗ {symbol}: {exc}")

    print()
    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()