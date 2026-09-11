from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"

DEFAULT_INTERVAL = "5m"
MAX_LIMIT = 1500

REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 0.15

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

# --------------------------------------------------------------
# Default single-symbol / single-year constants.
#
# Several ad-hoc test scripts (test_mss.py, test_micro_mss.py, ...)
# import a single SYMBOL / YEAR instead of iterating over SYMBOLS.
# These were missing from config.py, causing:
#   ImportError: cannot import name 'SYMBOL' from 'src.config'
#   ImportError: cannot import name 'YEAR' from 'src.config'
#
# SYMBOLS itself is left untouched since other scripts
# (e.g. download_data.py) depend on the full list.
# --------------------------------------------------------------

SYMBOL = SYMBOLS[0]   # "BTCUSDT" — change here if a test script
                      # should default to a different symbol

YEAR = 2025           # matches the cached data used throughout
                      # this project's test scripts so far
                      # (BTCUSDT_2025_mss_cache.parquet)