from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"

DEFAULT_INTERVAL = "5m"
MAX_LIMIT = 1500

REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 0.15

# --------------------------------------------------------------
# Active symbols list.
#
# The first three (BTC, ETH, SOL) already have cached 2025 data
# in data/raw/. The remaining seven are downloaded on demand by
# scripts/download_new_symbols.py.
#
# NOTE: MATIC was renamed to POL in 2024. If Binance returns an
# error for MATICUSDT, replace it with "POLUSDT".
# --------------------------------------------------------------

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "NEARUSDT",
]

# --------------------------------------------------------------
# Default single-symbol / single-year constants.
#
# Several ad-hoc test scripts (test_mss.py, test_micro_mss.py,
# test_backtest.py, ...) import a single SYMBOL / YEAR instead
# of iterating over SYMBOLS.
#
# To switch the active symbol for these test scripts, change the
# index below:
#   SYMBOLS[0]  -> "BTCUSDT"
#   SYMBOLS[1]  -> "ETHUSDT"
#   SYMBOLS[2]  -> "SOLUSDT"
#   SYMBOLS[3]  -> "ADAUSDT"
#   ...
# --------------------------------------------------------------

SYMBOL = SYMBOLS[0]   # "BTCUSDT" — currently active

YEAR = 2025           # matches the cached data used throughout
                      # this project's test scripts so far
                      # (e.g. BTCUSDT_2025_mss_cache.parquet)