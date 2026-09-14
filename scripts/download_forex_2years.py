"""
Download 2 years of 1H Forex data from Yahoo Finance.

Yahoo limit: 730 days for 1H.
Today: ~2026-09-13
Oldest allowed: ~2024-09-14

Saves as:
    data/raw/<SYMBOL>/<SYMBOL>_1h_<YEAR>.parquet
"""

from pathlib import Path
import sys
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR


# ============================================================
# SYMBOLS
# ============================================================

SYMBOLS = {
    # 6 original
    "EURUSD":  "EURUSD=X",
    "GBPUSD":  "GBPUSD=X",
    "USDJPY":  "USDJPY=X",
    "XAUUSD":  "GC=F",
    "SP500":   "^GSPC",
    "NAS100":  "^NDX",

    # 10 new
    "AUDUSD":  "AUDUSD=X",
    "USDCAD":  "USDCAD=X",
    "USDCHF":  "USDCHF=X",
    "NZDUSD":  "NZDUSD=X",
    "EURGBP":  "EURGBP=X",
    "EURJPY":  "EURJPY=X",
    "GBPJPY":  "GBPJPY=X",
    "AUDJPY":  "AUDJPY=X",
    "XAGUSD":  "SI=F",
    "WTI":     "CL=F",
}


# ============================================================
# DATE RANGE (2 years max for 1H)
# ============================================================

START_DATE = "2024-09-14"
END_DATE = "2025-12-31"


# ============================================================
# DOWNLOAD
# ============================================================

def download_symbol(label, ticker):
    """
    Download 1H data for a 2-year window.

    Saves one file per year:
        data/raw/<SYMBOL>/<SYMBOL>_1h_2024.parquet
        data/raw/<SYMBOL>/<SYMBOL>_1h_2025.parquet
    """

    out_dir = DATA_DIR / "raw" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading {label} ({ticker}) ...")
    print(f"    Range: {START_DATE} → {END_DATE}")

    try:
        df = yf.download(
            tickers=ticker,
            start=START_DATE,
            end=END_DATE,
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

    # Flatten MultiIndex columns
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

    required = ["timestamp", "open", "high", "low", "close"]
    for col in required:
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

    # Split by year and save
    df["year"] = df["timestamp"].dt.year

    for year, group in df.groupby("year"):
        group = group.drop(columns=["year"]).reset_index(drop=True)
        out_path = out_dir / f"{label}_1h_{year}.parquet"
        group.to_parquet(out_path, index=False)
        print(f"    Saved: {out_path.name} ({len(group):,} rows)")

    # Also save combined for convenience
    combined_path = out_dir / f"{label}_1h_combined.parquet"
    df.drop(columns=["year"]).to_parquet(combined_path, index=False)
    print(f"    Combined: {combined_path.name} ({len(df):,} rows)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("DOWNLOAD 2 YEARS OF 1H FOREX DATA")
    print("=" * 72)
    print(f"Symbols:    {len(SYMBOLS)}")
    print(f"Start:      {START_DATE}")
    print(f"End:        {END_DATE}")
    print(f"Total days: ~470 (Yahoo limit is 730)")
    print()

    for label, ticker in SYMBOLS.items():
        download_symbol(label, ticker)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()