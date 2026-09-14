"""
Paper Trading Monitor

Runs the best-combination pipeline on live/recent MT5 data
and generates entry signals for manual execution.

What it does:
    1. Connects to MT5 (running terminal)
    2. Loads latest 1H data for the 4 best symbols
    3. Runs the full best-combination pipeline
    4. Finds setups that are currently actionable
    5. Prints signals with entry / SL / TP / size
    6. Optionally saves signals to a CSV

Run this once per hour (e.g., with Task Scheduler) to get fresh signals.

Usage:
    python scripts/paper_trading_monitor.py

Optional:
    --save   Save signals to data/paper_trading/signals_YYYYMMDD_HHMM.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
import pandas as pd

from src.strategy.swing_detector import detect_pivots
from src.strategy.liquidity import (
    detect_equal_highs, detect_equal_lows,
    detect_previous_day_liquidity, detect_swing_liquidity,
    deduplicate_zones,
)
from src.strategy.sweep_detector import detect_sweeps
from src.strategy.mss_detector import detect_mss
from src.strategy.displacement_detector import detect_displacements
from src.strategy.order_block_detector import detect_order_blocks
from src.strategy.fvg_detector import detect_fvgs
from src.strategy.retest_detector import detect_ob_retests, detect_fvg_retests
from src.strategy.micro_mss_detector import detect_micro_mss, deduplicate_micro_mss
from src.strategy.entry_detector import detect_entry_candidates
from src.strategy.risk_manager import (
    build_risk_plans, calculate_atr, get_valid_risk_plans,
)
from src.strategy.position_sizer import (
    build_position_sizes, get_valid_position_sizes,
)


# ============================================================
# CONFIG
# ============================================================

SYMBOLS = ["XAUUSD", "GBPJPY", "USDJPY", "EURJPY"]

ACCOUNT_EQUITY = 10_000.0
RISK_PERCENT = 1.0
LEVERAGE = 1.0

ATR_PERIOD = 14
STRUCTURE_PIVOT_WINDOW_4H = 2
LIQUIDITY_PIVOT_WINDOW = 3
INTERNAL_PIVOT_WINDOW_1H = 2
EQUAL_TOLERANCE_ATR = 0.15
SWEEP_ATR_PERIOD = 14
MSS_MAX_BARS_AFTER_SWEEP = 12
DISPLACEMENT_MIN_BODY_ATR = 1.0
DISPLACEMENT_BULLISH_CLV = 0.70
DISPLACEMENT_BEARISH_CLV = 0.30
DISPLACEMENT_MAX_BARS_AFTER_MSS = 8
OB_LOOKBACK = 8
FVG_SEARCH_BARS = 3
MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR = 0.25

DIRECTION_FILTER = "bullish"

BEST_RR = 3.0
BEST_BUFFER = 0.10
BEST_RETEST = 72
BEST_MICRO = 60

# How many recent candles to look for fresh setups
LOOKBACK_HOURS = 24

# MT5 path (adjust if needed)
MT5_PATH = r"C:\Users\EnRival\AppData\Roaming\MetaTrader 5\terminal64.exe"

# Historical bars to load (in 1H candles)
HISTORY_BARS = 3000

# Output
OUTPUT_DIR = DATA_DIR / "paper_trading"


# ============================================================
# HELPERS
# ============================================================

def to_utc(value):
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def safe_float(value):
    try:
        if value is None:
            return None
        v = float(value)
        if v != v:
            return None
        return v
    except (TypeError, ValueError):
        return None


def normalize_direction(value):
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    text = str(raw).strip().lower()
    if "." in text:
        text = text.split(".")[-1]
    if text in {"bullish", "long", "buy"}:
        return "bullish"
    if text in {"bearish", "short", "sell"}:
        return "bearish"
    return text


def get_attr(obj, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


def timestamp_key(value):
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return str(ts)
    except Exception:
        return str(value)


def resample_from_1h(df_1h, rule):
    df = df_1h.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    out = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return out.reset_index()


# ============================================================
# MT5 DATA LOADER
# ============================================================

def load_mt5_data(symbol, bars=HISTORY_BARS):
    """Load recent 1H data from MT5."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: pip install MetaTrader5")
        return None

    if not mt5.initialize(MT5_PATH):
        print(f"ERROR: mt5.initialize failed: {mt5.last_error()}")
        return None

    info = mt5.terminal_info()
    if info is None or not info.connected:
        print("ERROR: MT5 not connected. Is it running?")
        mt5.shutdown()
        return None

    # Make sure symbol is visible
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        alt = symbol.replace(".pro", "")
        symbol_info = mt5.symbol_info(alt)
        if symbol_info is None:
            print(f"  SKIP {symbol}: not found in MT5")
            mt5.shutdown()
            return None
        symbol = alt

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    # Fetch recent bars
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"  SKIP {symbol}: no data returned")
        return None

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "tick_volume"]]
    df = df.rename(columns={"tick_volume": "volume"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# PIPELINE
# ============================================================

def find_setups(symbol, df_1h):
    """
    Run the pipeline on the loaded data and return actionable setups.
    """

    if df_1h is None or len(df_1h) < 200:
        return []

    # ---- Lower layers ----
    df_4h = resample_from_1h(df_1h, "4h")

    swings_4h = detect_pivots(
        df_4h, left_bars=STRUCTURE_PIVOT_WINDOW_4H,
        right_bars=STRUCTURE_PIVOT_WINDOW_4H, timeframe="4h",
    )

    eq_h = detect_equal_highs(df_4h, atr_period=ATR_PERIOD,
                              tolerance_atr=EQUAL_TOLERANCE_ATR,
                              pivot_window=LIQUIDITY_PIVOT_WINDOW)
    eq_l = detect_equal_lows(df_4h, atr_period=ATR_PERIOD,
                             tolerance_atr=EQUAL_TOLERANCE_ATR,
                             pivot_window=LIQUIDITY_PIVOT_WINDOW)
    pd_zones = detect_previous_day_liquidity(df_4h)
    sw_zones = detect_swing_liquidity(swings_4h)

    liq_zones = deduplicate_zones(
        eq_h + eq_l + pd_zones + sw_zones, price_tolerance=1e-8,
    )

    sweeps = detect_sweeps(df_1h, liq_zones, atr_period=SWEEP_ATR_PERIOD)

    swings_1h = detect_pivots(
        df_1h, left_bars=INTERNAL_PIVOT_WINDOW_1H,
        right_bars=INTERNAL_PIVOT_WINDOW_1H, timeframe="1h",
    )

    mss_events = detect_mss(df_1h, sweeps, swings_1h,
                            max_bars_after_sweep=MSS_MAX_BARS_AFTER_SWEEP)

    disps = detect_displacements(
        df_1h, mss_events, atr_period=ATR_PERIOD,
        min_body_atr=DISPLACEMENT_MIN_BODY_ATR,
        bullish_clv=DISPLACEMENT_BULLISH_CLV,
        bearish_clv=DISPLACEMENT_BEARISH_CLV,
        max_bars_after_mss=DISPLACEMENT_MAX_BARS_AFTER_MSS,
    )

    obs = detect_order_blocks(df_1h, disps, lookback=OB_LOOKBACK)
    fvgs = detect_fvgs(df_1h, disps, search_bars=FVG_SEARCH_BARS,
                       atr_period=ATR_PERIOD)

    ob_r = detect_ob_retests(df_1h, obs, max_bars_after_zone=BEST_RETEST)
    fvg_r = detect_fvg_retests(df_1h, fvgs, max_bars_after_zone=BEST_RETEST)
    retests = list(ob_r) + list(fvg_r)

    if not retests:
        return []

    df_1h_idx = df_1h.copy()
    df_1h_idx["timestamp"] = pd.to_datetime(df_1h_idx["timestamp"], utc=True)
    df_1h_idx = df_1h_idx.set_index("timestamp").sort_index()

    micro = detect_micro_mss(
        df_1h_idx, retests, swings_1h,
        max_bars_after_retest=BEST_MICRO,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )
    micro = deduplicate_micro_mss(micro)

    if not micro:
        return []

    # ---- Entry Detector ----
    bias_map = {}
    sw4_sorted = sorted(
        swings_4h, key=lambda s: str(getattr(s, "timestamp", "")),
    )
    for mi in micro:
        mts = getattr(mi, "timestamp", None)
        if mts is None:
            continue
        latest = None
        for s in sw4_sorted:
            sts = getattr(s, "timestamp", None)
            if sts and sts < mts:
                latest = s
            else:
                break
        if latest:
            d = normalize_direction(getattr(latest, "direction", ""))
            if d:
                bias_map[str(mts)] = d

    entry_prices = {}
    for _, row in df_1h.iterrows():
        ts = timestamp_key(row.get("timestamp"))
        c = safe_float(row.get("close"))
        if ts and c is not None:
            entry_prices[ts] = c

    sweeps_by_mss = {}
    for sw in sweeps:
        sw_ts = get_attr(sw, "timestamp", "sweep_timestamp")
        if sw_ts is None:
            continue
        k = timestamp_key(sw_ts)
        if k:
            sweeps_by_mss[k] = sw

    context = {
        "entry_prices": entry_prices,
        "sweeps_by_mss_timestamp": sweeps_by_mss,
        "risk_reward": BEST_RR,
        "bias_by_timestamp": bias_map,
    }

    candidates = detect_entry_candidates(
        micro, retests, mss_events, disps, obs, fvgs,
        context=context, debug=False,
    )

    tradable = [
        c for c in candidates
        if getattr(c, "tradable", False)
        and normalize_direction(getattr(c, "direction", "")) == DIRECTION_FILTER
    ]

    if not tradable:
        return []

    # Filter: recent only
    cutoff_ts = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    recent = []
    for c in tradable:
        c_ts = to_utc(getattr(c, "timestamp", None))
        if c_ts is None:
            continue
        if c_ts >= cutoff_ts:
            recent.append(c)

    if not recent:
        return []

    # ---- Risk + Sizing ----
    atr_series = calculate_atr(df_1h_idx, period=ATR_PERIOD).sort_index()

    atr_map = {}
    for c in recent:
        c_ts = getattr(c, "timestamp", None)
        if c_ts is None:
            continue
        try:
            ts = pd.Timestamp(c_ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            r = atr_series.asof(ts)
            if r is not None and not pd.isna(r):
                atr_map[c_ts] = float(r)
        except Exception:
            pass

    zones = list(obs) + list(fvgs)
    plans = build_risk_plans(
        recent, zones, atr_map,
        target_rr=BEST_RR,
        buffer_atr=BEST_BUFFER,
    )
    valid_plans = get_valid_risk_plans(plans)
    if not valid_plans:
        return []

    sizes = build_position_sizes(
        valid_plans,
        account_equity=ACCOUNT_EQUITY,
        risk_percent=RISK_PERCENT,
        leverage=LEVERAGE,
    )
    valid_sizes = get_valid_position_sizes(sizes)

    return valid_sizes


# ============================================================
# SIGNAL REPORTING
# ============================================================

def print_signal(symbol, ps):
    direction = normalize_direction(getattr(ps, "direction", ""))
    entry = safe_float(getattr(ps, "entry_price", None))
    sl = safe_float(getattr(ps, "stop_loss", None))
    tp = safe_float(getattr(ps, "take_profit", None))
    size = safe_float(getattr(ps, "position_size", None))
    risk_amt = safe_float(getattr(ps, "risk_amount", None))
    risk_unit = safe_float(getattr(ps, "risk_per_unit", None))
    rr = safe_float(getattr(ps, "risk_reward", None))
    ts = to_utc(getattr(ps, "timestamp", None))
    zone_type = getattr(ps, "zone_type", None)

    print()
    print("=" * 60)
    print(f"  SIGNAL: {symbol}  ({direction.upper()})")
    print("=" * 60)
    print(f"  Time:       {ts}")
    print(f"  Zone type:  {zone_type}")
    print(f"  Entry:      {entry:.5f}")
    print(f"  Stop Loss:  {sl:.5f}  (risk {risk_unit:.5f})")
    print(f"  Take Profit:{tp:.5f}  (R:R {rr:.2f})")
    print(f"  Size:       {size:.4f} units")
    print(f"  Risk ($):   {risk_amt:.2f}")
    print("=" * 60)


def save_signals(signals, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"signals_{ts}.csv"

    rows = []
    for symbol, ps in signals:
        rows.append({
            "symbol": symbol,
            "timestamp": str(to_utc(getattr(ps, "timestamp", None))),
            "direction": normalize_direction(getattr(ps, "direction", "")),
            "entry": safe_float(getattr(ps, "entry_price", None)),
            "stop_loss": safe_float(getattr(ps, "stop_loss", None)),
            "take_profit": safe_float(getattr(ps, "take_profit", None)),
            "position_size": safe_float(getattr(ps, "position_size", None)),
            "risk_amount": safe_float(getattr(ps, "risk_amount", None)),
            "risk_per_unit": safe_float(getattr(ps, "risk_per_unit", None)),
            "risk_reward": safe_float(getattr(ps, "risk_reward", None)),
            "zone_type": getattr(ps, "zone_type", None),
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("PAPER TRADING MONITOR")
    print("=" * 72)
    print(f"Time:       {datetime.now(timezone.utc).isoformat()}")
    print(f"Symbols:    {SYMBOLS}")
    print(f"Direction:  {DIRECTION_FILTER}")
    print(f"Lookback:   {LOOKBACK_HOURS} hours")
    print(f"Account:    ${ACCOUNT_EQUITY:,.2f}")
    print(f"Risk:       {RISK_PERCENT:.2f}% per trade")
    print()

    # Check if save flag is set
    save = "--save" in sys.argv

    # ---- Load & process each symbol ----
    all_signals = []

    for sym in SYMBOLS:
        print(f"Processing {sym} ...")

        df = load_mt5_data(sym)
        if df is None:
            continue

        print(f"  Loaded {len(df)} candles "
              f"({df['timestamp'].min()} → {df['timestamp'].max()})")

        setups = find_setups(sym, df)
        print(f"  Found {len(setups)} recent setups")

        for ps in setups:
            print_signal(sym, ps)
            all_signals.append((sym, ps))

    # ---- Summary ----
    print()
    print("=" * 72)
    print(f"TOTAL SIGNALS: {len(all_signals)}")
    print("=" * 72)

    if not all_signals:
        print()
        print("No actionable signals right now.")
        print("Check again in 1 hour, or run during London/NY sessions.")
        return

    print()
    print("Actionable signals:")
    for sym, ps in all_signals:
        direction = normalize_direction(getattr(ps, "direction", ""))
        entry = safe_float(getattr(ps, "entry_price", None))
        sl = safe_float(getattr(ps, "stop_loss", None))
        tp = safe_float(getattr(ps, "take_profit", None))
        print(
            f"  {sym:<10} {direction:<8} "
            f"Entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}"
        )

    # ---- Save ----
    if save:
        save_signals(all_signals, OUTPUT_DIR)


if __name__ == "__main__":
    main()