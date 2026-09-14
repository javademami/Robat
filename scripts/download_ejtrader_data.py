"""
Download 10 years of H1 data from ejtraderLabs/historical-data GitHub repo.

Available symbols:
    AUDJPY, AUDUSD, EURCHF, EURGBP, EURJPY, EURUSD,
    GBPJPY, GBPUSD, USDCAD, USDCHF, USDJPY, XAUUSD

Data range: last 10 years (rolling)
Timeframe:  H1 (and M15, M30, H4, D1)

Saves as:
    data/raw/<SYMBOL>/<SYMBOL>_1h_10y.parquet

Usage:
    python scripts/download_ejtrader_data.py
"""

from pathlib import Path
import sys
import io
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR


# ============================================================
# CONFIG
# ============================================================

# GitHub repo base URL
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "ejtraderLabs/historical-data/main"
)

# Symbols available in the repo
SYMBOLS = [
    "AUDJPY",
    "AUDUSD",
    "EURCHF",
    "EURGBP",
    "EURJPY",
    "EURUSD",
    "GBPJPY",
    "GBPUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
    "XAUUSD",
]

# Timeframe: use H1
TIMEFRAME = "H1"

# Possible file paths to try
FILE_PATTERNS = [
    "{symbol}/{symbol}_{tf}.csv",
    "{symbol}/{tf}/{symbol}_{tf}.csv",
    "{symbol}_{tf}.csv",
    "{symbol}/{symbol}.csv",
]


# ============================================================
# DOWNLOAD
# ============================================================

def try_download(symbol, tf):
    """
    Try different possible file paths in the repo.
    Returns (url, content) if successful, else (None, None).
    """
    for pattern in FILE_PATTERNS:
        rel_path = pattern.format(symbol=symbol, tf=tf)
        url = f"{BASE_URL}/{rel_path}"

        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 100:
                return url, r.content
        except Exception:
            continue

    return None, None


def parse_csv(content):
    """
    Parse the downloaded CSV into our standard format.
    Handles various possible column layouts.
    """
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        print(f"    CSV parse error: {e}")
        return None

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in {"date", "datetime", "time", "timestamp"}:
            col_map[c] = "timestamp"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl in {"volume", "vol"}:
            col_map[c] = "volume"

    df = df.rename(columns=col_map)

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"    Missing columns: {missing}")
        print(f"    Available: {list(df.columns)}")
        return None

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    return df


def download_symbol(symbol, tf):
    out_dir = DATA_DIR / "raw" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{symbol}_1h_10y.parquet"

    if out_path.exists():
        print(f"  SKIP {symbol}: already exists")
        return True

    print(f"  Downloading {symbol} ({tf}) ...")

    url, content = try_download(symbol, tf)
    if url is None:
        print(f"    NOT FOUND in repo")
        return False

    df = parse_csv(content)
    if df is None or len(df) == 0:
        print(f"    Empty or invalid data")
        return False

    df.to_parquet(out_path, index=False)

    print(
        f"    ✓ Saved: {out_path.name} "
        f"({len(df):,} rows, "
        f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()})"
    )
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("DOWNLOAD 10Y DATA FROM ejtraderLabs/historical-data")
    print("=" * 72)
    print(f"Symbols:   {len(SYMBOLS)}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Base URL:  {BASE_URL}")
    print()

    success = 0
    failed = []

    for sym in SYMBOLS:
        if download_symbol(sym, TIMEFRAME):
            success += 1
        else:
            failed.append(sym)

    print()
    print("=" * 72)
    print(f"SUCCESS: {success}/{len(SYMBOLS)}")
    if failed:
        print(f"FAILED:  {', '.join(failed)}")
    print("=" * 72)


if __name__ == "__main__":
    main()