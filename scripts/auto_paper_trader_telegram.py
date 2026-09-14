"""
Auto Paper Trader + Telegram Notifications

Extends auto_paper_trader.py to send signals to Telegram.

Setup complete:
    TOKEN:    8499078795:AAHZlwdnQ1pmURPFM1Cegm2TC3lnna_7a9c
    CHAT_ID:  6744828203

Usage:
    python scripts/auto_paper_trader_telegram.py
"""

from __future__ import annotations

import sys
import time
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

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
# TELEGRAM CONFIG
# ============================================================

TELEGRAM_TOKEN = "8499078795:AAHZlwdnQ1pmURPFM1Cegm2TC3lnna_7a9c"
TELEGRAM_CHAT_ID = "6744828203"

# Send heartbeat even when no signals found?
SEND_HEARTBEAT = False


# ============================================================
# STRATEGY CONFIG
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

LOOKBACK_HOURS = 24

MT5_PATH = r"C:\Users\EnRival\AppData\Roaming\MetaTrader 5\terminal64.exe"
HISTORY_BARS = 3000


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_message(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        else:
            print(f"  TELEGRAM ERROR: {r.status_code} {r.text}")
            return False
    except Exception as e:
        print(f"  TELEGRAM ERROR: {e}")
        return False


def format_signal_message(symbol, ps):
    direction = get_attr(ps, "direction", "")
    entry = safe_float(get_attr(ps, "entry_price", None))
    sl = safe_float(get_attr(ps, "stop_loss", None))
    tp = safe_float(get_attr(ps, "take_profit", None))
    size = safe_float(get_attr(ps, "position_size", None))
    risk = safe_float(get_attr(ps, "risk_amount", None))
    rr = safe_float(get_attr(ps, "risk_reward", None))
    zone = get_attr(ps, "zone_type", "?")
    ts = get_attr(ps, "timestamp", None)

    text = (
        f"🚨 *ICT SIGNAL* 🚨\n\n"
        f"📊 *Symbol:* `{symbol}`\n"
        f"📈 *Direction:* {str(direction).upper()}\n"
        f"🎯 *Zone:* {zone}\n"
        f"⏰ *Time:* {ts}\n\n"
        f"💰 *Entry:* `{entry:.5f}`\n"
        f"🛑 *SL:* `{sl:.5f}`\n"
        f"✅ *TP:* `{tp:.5f}`\n"
        f"📐 *R:R:* {rr:.2f}\n\n"
        f"📦 *Size:* {size:.4f}\n"
        f"💵 *Risk:* ${risk:.2f}\n\n"
        f"⚡ _Place order in MT5 manually_"
    )
    return text


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
# PIPELINE
# ============================================================

def find_setups(symbol, df_1h):
    if df_1h is None or len(df_1h) < 200:
        return []

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
# MT5
# ============================================================

def load_mt5_data(mt5, symbol, bars=HISTORY_BARS):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        alt = symbol.replace(".pro", "")
        symbol_info = mt5.symbol_info(alt)
        if symbol_info is None:
            return None, None
        symbol = alt

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
    if rates is None or len(rates) == 0:
        return None, symbol

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "tick_volume"]]
    df = df.rename(columns={"tick_volume": "volume"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df, symbol


# ============================================================
# MAIN
# ============================================================

def main():
    import MetaTrader5 as mt5

    print("=" * 72)
    print("AUTO PAPER TRADER + TELEGRAM")
    print("=" * 72)
    print(f"Time:      {datetime.now(timezone.utc).isoformat()}")
    print(f"Symbols:   {SYMBOLS}")
    print(f"Direction: {DIRECTION_FILTER}")
    print()

    # Init MT5
    if not mt5.initialize(MT5_PATH):
        print(f"ERROR: mt5.initialize failed: {mt5.last_error()}")
        return

    acc = mt5.account_info()
    if acc is None:
        print("ERROR: no account info")
        mt5.shutdown()
        return

    print(f"Account:   {acc.login}")
    print(f"Server:    {acc.server}")
    print(f"Balance:   ${acc.balance:,.2f}")
    print()

    # Process each symbol
    all_signals = []

    for sym in SYMBOLS:
        print(f"Processing {sym} ...")

        df, actual_symbol = load_mt5_data(mt5, sym)
        if df is None:
            print(f"  SKIP: no data")
            continue

        print(f"  Loaded {len(df)} candles")
        setups = find_setups(actual_symbol, df)
        print(f"  Found {len(setups)} setups")

        for ps in setups:
            all_signals.append((actual_symbol, ps))

    # Report
    print()
    print("=" * 72)
    print(f"TOTAL SIGNALS: {len(all_signals)}")
    print("=" * 72)

    if all_signals:
        for sym, ps in all_signals:
            print()
            print(f"  {sym}")

            # Send to Telegram
            msg = format_signal_message(sym, ps)
            sent = send_telegram_message(msg)

            if sent:
                print(f"  ✅ Sent to Telegram")
            else:
                print(f"  ❌ Telegram failed")
    else:
        print("No signals.")
        if SEND_HEARTBEAT:
            send_telegram_message(
                f"ℹ️ Paper Trader ran at "
                f"{datetime.now(timezone.utc).strftime('%H:%M UTC')} — no signals."
            )

    mt5.shutdown()


if __name__ == "__main__":
    main()