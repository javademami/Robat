"""
Download 10 new Forex pairs + commodities (2025 only).

New symbols:
    AUDUSD, USDCAD, USDCHF, NZDUSD
    EURGBP, EURJPY, GBPJPY, AUDJPY
    XAGUSD (Silver), WTI (Crude Oil)

Saves as:
    data/raw/<SYMBOL>/<SYMBOL>_1h_<YEAR>.parquet

Usage:
    python scripts/download_forex_new_symbols.py
"""

from pathlib import Path
import sys
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR


NEW_SYMBOLS = {
    # Major pairs
    "AUDUSD":  "AUDUSD=X",
    "USDCAD":  "USDCAD=X",
    "USDCHF":  "USDCHF=X",
    "NZDUSD":  "NZDUSD=X",

    # Cross pairs
    "EURGBP":  "EURGBP=X",
    "EURJPY":  "EURJPY=X",
    "GBPJPY":  "GBPJPY=X",
    "AUDJPY":  "AUDJPY=X",

    # Commodities
    "XAGUSD":  "SI=F",       # Silver futures
    "WTI":     "CL=F",       # Crude Oil futures
}

YEARS = [2025]


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
    print("DOWNLOAD NEW FOREX PAIRS (2025 only)")
    print("=" * 72)
    print(f"New symbols: {len(NEW_SYMBOLS)}")
    print(f"Years:       {YEARS}")
    print()

    for year in YEARS:
        print(f"\n--- {year} ---")
        for label, ticker in NEW_SYMBOLS.items():
            download_symbol_year(label, ticker, year)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()