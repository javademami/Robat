"""
Download 5m klines for the new symbols from Binance Futures.

Uses BINANCE_FUTURES_URL from src.config.

Saves as:
    data/raw/<SYMBOL>/<SYMBOL>_5m_<YEAR>.parquet
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    BINANCE_FUTURES_URL,
    DEFAULT_INTERVAL,
    MAX_LIMIT,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    REQUEST_DELAY_SECONDS,
    YEAR,
)

# Symbols to download (all except BTC/ETH/SOL which you already have)
NEW_SYMBOLS = [
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "NEARUSDT",
]


def _ms(year: int, month: int, day: int) -> int:
    import datetime as _dt
    return int(_dt.datetime(year, month, day, tzinfo=_dt.timezone.utc).timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }

        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(
                    BINANCE_FUTURES_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(1 + attempt)

        if not batch:
            break

        rows.extend(batch)
        cursor = batch[-1][0] + 1
        time.sleep(REQUEST_DELAY_SECONDS)

    return rows


def download_symbol(symbol: str, year: int, interval: str) -> pd.DataFrame:
    start_ms = _ms(year, 1, 1)
    end_ms = _ms(year + 1, 1, 1) - 1

    print(f"  Fetching {symbol} {interval} {year} ...")
    raw = fetch_klines(symbol, interval, start_ms, end_ms)
    print(f"  Got {len(raw):,} candles")

    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def main() -> None:
    year = YEAR
    interval = DEFAULT_INTERVAL

    print(f"Downloading {len(NEW_SYMBOLS)} symbols for {year} ({interval})")
    print(f"Data dir: {DATA_DIR}")
    print()

    for symbol in NEW_SYMBOLS:
        out_dir = DATA_DIR / "raw" / symbol
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{symbol}_5m_{year}.parquet"

        if out_path.exists():
            print(f"  SKIP {symbol}: {out_path.name} already exists")
            continue

        try:
            df = download_symbol(symbol, year, interval)
            df.to_parquet(out_path, index=False)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  ERROR {symbol}: {e}")
        print()


if __name__ == "__main__":
    main()