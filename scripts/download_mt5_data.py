"""
Extract historical data from MetaTrader 5.

Requires:
    - MetaTrader 5 terminal running and logged in.
    - pip install MetaTrader5

For each symbol, opens the chart in MT5 first (to load history),
then extracts H1 bars from START_DATE to END_DATE.

Usage:
    python scripts/download_mt5_data.py
"""

from pathlib import Path
import sys
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR


# ============================================================
# CONFIG
# ============================================================

# Symbol names as they appear in MT5 (no .pro suffix for MetaQuotes)
SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "XAUUSD",
    "XAGUSD",
    "WTI",
]

# Date range (10 years)
START_DATE = datetime(2015, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 31, tzinfo=timezone.utc)

# Timeframe
TIMEFRAME_NAME = "H1"


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 not installed.")
        print("Run: pip install MetaTrader5")
        return

    if not mt5.initialize():
        print("ERROR: mt5.initialize() failed.")
        print("Make sure MetaTrader 5 is running and logged in.")
        print(f"  last_error: {mt5.last_error()}")
        return

    print("=" * 72)
    print("MT5 HISTORICAL DATA EXTRACTION")
    print("=" * 72)

    info = mt5.terminal_info()
    if info:
        print(f"Terminal: {info.name}")
        print(f"Connected: {info.connected}")

    print(f"Symbols:   {len(SYMBOLS)}")
    print(f"Timeframe: {TIMEFRAME_NAME}")
    print(f"Range:     {START_DATE.date()} -> {END_DATE.date()}")
    print()

    # Map timeframe name to MT5 constant
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    tf = tf_map[TIMEFRAME_NAME]

    # Get all available symbols
    all_symbols = mt5.symbols_get()
    available_names = {s.name for s in all_symbols} if all_symbols else set()
    print(f"Symbols available in MT5: {len(available_names)}")
    print()

    success = 0
    failed = []

    for symbol in SYMBOLS:
        print(f"--- {symbol} ---")

        # Try symbol as-is, then without .pro
        sym = None
        if symbol in available_names:
            sym = symbol
        else:
            alt = symbol.replace(".pro", "")
            if alt in available_names:
                sym = alt
                print(f"  Using alternative: {sym}")

        if sym is None:
            print(f"  SKIP: not found in MT5")
            failed.append(symbol)
            continue

        symbol_info = mt5.symbol_info(sym)
        if symbol_info is None:
            print(f"  SKIP: symbol_info is None")
            failed.append(symbol)
            continue

        # Make visible in Market Watch
        if not symbol_info.visible:
            mt5.symbol_select(sym, True)

        # Extract rates
        rates = mt5.copy_rates_range(sym, tf, START_DATE, END_DATE)

        if rates is None or len(rates) == 0:
            print(f"  ERROR: no data returned")
            print(f"  Tip: open the chart in MT5 first (Ctrl+M -> right-click -> Chart Window)")
            failed.append(symbol)
            continue

        # Build DataFrame
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)

        df = df[["timestamp", "open", "high", "low", "close", "tick_volume"]]
        df = df.rename(columns={"tick_volume": "volume"})
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Save
        clean_name = sym.replace(".pro", "")
        out_dir = DATA_DIR / "raw" / clean_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{clean_name}_1h_mt5.parquet"
        df.to_parquet(out_path, index=False)

        print(
            f"  Saved: {out_path.name} "
            f"({len(df):,} rows, "
            f"{df['timestamp'].min().date()} -> "
            f"{df['timestamp'].max().date()})"
        )
        success += 1
        print()

    mt5.shutdown()

    print("=" * 72)
    print(f"SUCCESS: {success}/{len(SYMBOLS)}")
    if failed:
        print(f"FAILED:  {', '.join(failed)}")
    print("=" * 72)


if __name__ == "__main__":
    main()