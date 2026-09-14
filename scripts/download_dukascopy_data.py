"""
Download multi-year tick data from Dukascopy via tick-vault.

Saves raw compressed .bi5 files under ./my_tick_data/
and provides a helper to read them as pandas DataFrames.

Usage:
    python scripts/download_dukascopy_data.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, UTC

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import tick-vault
from tick_vault import download_range, reload_config


# ============================================================
# CONFIGURATION
# ============================================================

# Where to store raw tick data (compressed .bi5 files)
TICK_DATA_DIR = PROJECT_ROOT / "data" / "dukascopy_ticks"

# Symbols to download (must be supported by tick-vault)
SYMBOLS = [
    # Forex Majors
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    # Metals
    "XAUUSD", "XAGUSD",
    # Indices (if supported)
    "SP500", "NAS100",
]

# Date range to download
START_DATE = datetime(2023, 1, 1, tzinfo=UTC)
END_DATE = datetime(2025, 12, 31, tzinfo=UTC)


# ============================================================
# SETUP
# ============================================================

def setup():
    """Configure tick-vault to use our custom directory."""
    reload_config(
        base_directory=str(TICK_DATA_DIR),
        # You can increase worker count if you have proxies configured
        # worker_per_proxy=20,
    )
    print(f"Tick data directory: {TICK_DATA_DIR}")
    TICK_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOWNLOAD
# ============================================================

async def download_symbol(symbol: str):
    """Download all tick data for one symbol over the configured date range."""
    print(f"\n--- Downloading {symbol} ---")
    print(f"  From: {START_DATE.date()}")
    print(f"  To:   {END_DATE.date()}")

    try:
        await download_range(
            symbol=symbol,
            start=START_DATE,
            end=END_DATE,
        )
        print(f"  ✓ {symbol} download complete.")
    except Exception as e:
        print(f"  ✗ {symbol} failed: {e}")


async def download_all():
    """Download all configured symbols sequentially."""
    print("=" * 72)
    print("DUKASCOPY MULTI-YEAR DOWNLOAD")
    print("=" * 72)
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Range:   {START_DATE.date()} → {END_DATE.date()}")
    print()

    for symbol in SYMBOLS:
        await download_symbol(symbol)

    print()
    print("=" * 72)
    print("DOWNLOAD COMPLETE")
    print("=" * 72)
    print()
    print("To read the data as a pandas DataFrame, use:")
    print("  from tick_vault import read_tick_data")
    print("  df = read_tick_data(symbol='EURUSD')")
    print()
    print("Or with a specific date range:")
    print("  df = read_tick_data('EURUSD', start=datetime(2024,1,1), end=datetime(2024,2,1))")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    setup()
    asyncio.run(download_all())