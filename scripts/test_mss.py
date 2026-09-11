from pathlib import Path
import pandas as pd

from src.data.storage import load_year
from src.data.aggregator import resample_ohlcv
from src.strategy.swing_detector import detect_pivots
from src.strategy.liquidity import (
    detect_equal_highs,
    detect_equal_lows,
    detect_previous_day_liquidity,
    detect_swing_liquidity,
    deduplicate_zones,
)
from src.strategy.sweep_detector import detect_sweeps
from src.strategy.mss_detector import detect_mss, mss_to_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SYMBOL = "BTCUSDT"
YEAR = 2025
# 🌟 تعریف مسیر فایل ذخیره نتایج (Cache)
CACHE_FILE = DATA_DIR / f"{SYMBOL}_{YEAR}_mss_cache.parquet"


def main():
    print("=" * 70)
    print("SWEEP & MSS ENGINE TEST WITH CACHE")
    print("=" * 70)

    # 🟢 گام اول: بررسی وجود فایل کَش روی هارد برای صرفه‌جویی در زمان
    if CACHE_FILE.exists():
        print(f"\n[!] Cache file found at {CACHE_FILE}")
        print("[!] Loading pre-calculated MSS events from hard drive...")
        
        result_df = pd.read_parquet(CACHE_FILE)
        
        print(f"Total Cached MSS Events Loaded: {len(result_df):,}")
        print("=" * 70)
        print("LATEST MSS EVENTS (FROM CACHE)")
        print("=" * 70)
        print(result_df.tail(30).to_string(index=False))
        return

    # 🔴 گام دوم: اگر فایل وجود نداشته باشد، محاسبات از اول اجرا می‌شود
    print("\n[1] Cache not found. Loading 5M data...")
    df_5m = load_year(DATA_DIR, SYMBOL, "5m", YEAR)
    print(f"5M candles: {len(df_5m):,}")

    print("\n[2] Building 1H and 4H timeframes...")
    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")
    print(f"1H candles: {len(df_1h):,}")
    print(f"4H candles: {len(df_4h):,}")

    print("\n[3] Detecting 4H & 1H swings...")
    swings_4h = detect_pivots(df_4h, left_bars=2, right_bars=2, timeframe="4h")
    swings_1h = detect_pivots(df_1h, left_bars=2, right_bars=2, timeframe="1h") # سویینگ‌های داخلی
    print(f"4H swings: {len(swings_4h):,}")
    print(f"1H Internal swings: {len(swings_1h):,}")

    print("\n[4] Building liquidity zones...")
    swing_zones = detect_swing_liquidity(swings_4h)
    equal_highs = detect_equal_highs(df_1h, tolerance_atr=0.15, atr_period=14, lookback=100)
    equal_lows = detect_equal_lows(df_1h, tolerance_atr=0.15, atr_period=14, lookback=100)
    previous_day = detect_previous_day_liquidity(df_1h)

    zones = deduplicate_zones(swing_zones + equal_highs + equal_lows + previous_day)
    print(f"Liquidity zones: {len(zones):,}")

    print("\n[5] Detecting raw sweeps...")
    sweeps = detect_sweeps(df_1h, zones, atr_period=14)
    print(f"Sweeps: {len(sweeps):,}")

    print("\n[6] Detecting MSS (Running 14 Billion Calculations)...")
    mss_events = detect_mss(df_1h, sweeps, swings_1h, max_bars_after_sweep=12)
    print(f"Total verified MSS events found: {len(mss_events):,}")

    # تبدیل خروجی به دیتابیس پانداس
    result_df = mss_to_dataframe(mss_events)

    # 🌟 گام سوم: ذخیره آنی نتایج روی هارد دیسک برای دفعات بعدی
    print(f"\n[+] Saving results to cache: {CACHE_FILE}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(CACHE_FILE, index=False, engine="pyarrow")
    print("[+] Cache saved successfully!")

    print("\n" + "=" * 70)
    print("LATEST MSS EVENTS")
    print("=" * 70)
    print(result_df.tail(30).to_string(index=False))


if __name__ == "__main__":
    main()
