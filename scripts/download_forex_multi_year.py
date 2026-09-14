"""
Download Forex / Gold / Indices data for multiple years.

Saves as:
    data/raw/<SYMBOL>/<SYMBOL>_1h_<YEAR>.parquet

Usage:
    python scripts/download_forex_multi_year.py
"""

from pathlib import Path
import sys
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR


SYMBOLS = {
    "EURUSD":  "EURUSD=X",
    "GBPUSD":  "GBPUSD=X",
    "USDJPY":  "USDJPY=X",
    "XAUUSD":  "GC=F",
    "SP500":   "^GSPC",
    "NAS100":  "^NDX",
}

YEARS = [2023, 2024, 2025]


def download_symbol_year(label, ticker, year):
    out_dir = DATA_DIR / "raw" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{label}_1h_{year}.parquet"

    if out_path.exists():
        print(f"  SKIP {label} {year}: already exists")
        return

    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    print(f"  Downloading {label} ({ticker}) {year} ...")

    try:
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval="1h",
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    if df is None or df.empty:
        print(f"    ERROR: no data")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    for c in ["Datetime", "Date", "index"]:
        if c in df.columns:
            df = df.rename(columns={c: "timestamp"})
            break

    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })

    for col in ["timestamp", "open", "high", "low", "close"]:
        if col not in df.columns:
            print(f"    ERROR: missing {col}")
            return

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    print(f"    Saved: {out_path} ({len(df):,} rows)")


def main():
    print("=" * 72)
    print("DOWNLOAD MULTI-YEAR FOREX DATA")
    print("=" * 72)
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Years:   {YEARS}")
    print(f"Total:   {len(SYMBOLS) * len(YEARS)} files")
    print()

    for year in YEARS:
        print(f"\n--- {year} ---")
        for label, ticker in SYMBOLS.items():
            download_symbol_year(label, ticker, year)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()