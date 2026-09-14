"""
Auto Paper Trader — Fully Automated

Connects to MT5, detects signals using the best combination,
and places orders automatically (with SL/TP).

Safety:
    - Only works on DEMO account (checks account type)
    - Only bullish signals
    - 1% risk per trade
    - Max 1 open position per symbol
    - Max 3 total open positions

Usage:
    python scripts/auto_paper_trader.py

Optional:
    --dry-run    Show signals but don't place orders
"""

from __future__ import annotations

import sys
import time
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

LOOKBACK_HOURS = 24

MT5_PATH = r"C:\Users\EnRival\AppData\Roaming\MetaTrader 5\terminal64.exe"
HISTORY_BARS = 3000

# Safety limits
MAX_OPEN_POSITIONS = 3
MAGIC_NUMBER = 20260914   # for tracking our orders

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
# MT5 HELPERS
# ============================================================

def init_mt5(mt5):
    if not mt5.initialize(MT5_PATH):
        print(f"ERROR: mt5.initialize failed: {mt5.last_error()}")
        return False
    return True


def is_demo_account(mt5):
    acc = mt5.account_info()
    if acc is None:
        return False
    # MetaQuotes-Demo is demo
    return "demo" in (acc.server or "").lower() or acc.trade_mode == 0


def count_open_positions(mt5):
    positions = mt5.positions_get()
    if positions is None:
        return 0
    return len(positions)


def has_open_position_for_symbol(mt5, symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    return len(positions) > 0


def load_mt5_data(mt5, symbol, bars=HISTORY_BARS):
    """Load recent 1H data from MT5 (assumes initialized)."""
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


def get_symbol_info_price(mt5, symbol):
    """Return current bid/ask/point/digits for a symbol."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return {
        "bid": info.bid,
        "ask": info.ask,
        "point": info.point,
        "digits": info.digits,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "trade_tick_value": info.trade_tick_value,
        "trade_tick_size": info.trade_tick_size,
        "trade_contract_size": info.trade_contract_size,
    }


def calc_volume_from_risk(mt5, symbol, entry, sl, risk_amount):
    """
    Calculate MT5 volume (in lots) such that the SL distance
    equals risk_amount in account currency.

    For most symbols:
        risk_per_lot = (entry - sl) / point * tick_value
        volume = risk_amount / risk_per_lot

    Adjust to volume_step and clamp to [volume_min, volume_max].
    """
    info = get_symbol_info_price(mt5, symbol)
    if info is None:
        return None

    point = info["point"]
    tick_value = info["trade_tick_value"]
    tick_size = info["trade_tick_size"]
    volume_step = info["volume_step"]
    volume_min = info["volume_min"]
    volume_max = info["volume_max"]

    if tick_value is None or tick_value <= 0 or tick_size <= 0:
        return None

    # Distance in price
    sl_distance_price = abs(entry - sl)
    if sl_distance_price <= 0:
        return None

    # Ticks
    sl_distance_ticks = sl_distance_price / tick_size

    # Loss per 1 lot if SL hit
    loss_per_lot = sl_distance_ticks * tick_value
    if loss_per_lot <= 0:
        return None

    volume = risk_amount / loss_per_lot

    # Round to step
    if volume_step > 0:
        volume = round(volume / volume_step) * volume_step

    volume = max(volume_min, min(volume_max, volume))

    # Round to step digits
    step_str = str(volume_step)
    if "." in step_str:
        digits = len(step_str.split(".")[1])
    else:
        digits = 0
    volume = round(volume, digits)

    return volume


def place_buy_order(mt5, symbol, volume, sl, tp, comment="auto_paper"):
    """Place a market BUY order with SL/TP."""
    info = get_symbol_info_price(mt5, symbol)
    if info is None:
        return None

    price = info["ask"]
    digits = info["digits"]

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return result


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
# MAIN
# ============================================================

def main():
    import MetaTrader5 as mt5

    dry_run = "--dry-run" in sys.argv

    print("=" * 72)
    print("AUTO PAPER TRADER")
    print("=" * 72)
    print(f"Time:      {datetime.now(timezone.utc).isoformat()}")
    print(f"Symbols:   {SYMBOLS}")
    print(f"Direction: {DIRECTION_FILTER}")
    print(f"Dry-run:   {dry_run}")
    print()

    # ---- Init MT5 ----
    if not init_mt5(mt5):
        return

    # ---- Check Demo ----
    acc = mt5.account_info()
    if acc is None:
        print("ERROR: no account info")
        mt5.shutdown()
        return

    print(f"Account:   {acc.login}")
    print(f"Server:    {acc.server}")
    print(f"Balance:   ${acc.balance:,.2f}")
    print(f"Equity:    ${acc.equity:,.2f}")
    print(f"Demo:      {is_demo_account(mt5)}")
    print()

    if not is_demo_account(mt5):
        print("ERROR: This is NOT a demo account. Aborting for safety.")
        mt5.shutdown()
        return

    # ---- Check open positions ----
    open_count = count_open_positions(mt5)
    print(f"Open positions: {open_count} / {MAX_OPEN_POSITIONS}")
    print()

    if open_count >= MAX_OPEN_POSITIONS:
        print("Max open positions reached. No new orders.")
        mt5.shutdown()
        return

    # ---- Process each symbol ----
    signals = []

    for sym in SYMBOLS:
        print(f"Processing {sym} ...")

        if has_open_position_for_symbol(mt5, sym):
            print(f"  SKIP: already have open position for {sym}")
            continue

        df, actual_symbol = load_mt5_data(mt5, sym)
        if df is None:
            print(f"  SKIP: no data")
            continue

        print(f"  Loaded {len(df)} candles "
              f"({df['timestamp'].min()} → {df['timestamp'].max()})")

        setups = find_setups(actual_symbol, df)
        print(f"  Found {len(setups)} recent setups")

        for ps in setups:
            signals.append((actual_symbol, ps))

    # ---- Place orders ----
    print()
    print("=" * 72)
    print(f"TOTAL SIGNALS: {len(signals)}")
    print("=" * 72)

    if not signals:
        print("No signals. Nothing to do.")
        mt5.shutdown()
        return

    placed = 0

    for sym, ps in signals:
        if placed >= MAX_OPEN_POSITIONS - open_count:
            print("Max positions reached. Stopping.")
            break

        entry = safe_float(getattr(ps, "entry_price", None))
        sl = safe_float(getattr(ps, "stop_loss", None))
        tp = safe_float(getattr(ps, "take_profit", None))
        risk = safe_float(getattr(ps, "risk_amount", None))

        if None in (entry, sl, tp, risk):
            print(f"  SKIP {sym}: missing entry/sl/tp/risk")
            continue

        # Calculate volume based on current price
        info = get_symbol_info_price(mt5, sym)
        if info is None:
            print(f"  SKIP {sym}: no symbol info")
            continue

        current_price = info["ask"]

        # Recalculate SL/TP based on current price if entry is far away
        # (safety: skip if entry is more than 3% away from current price)
        distance_pct = abs(current_price - entry) / entry
        if distance_pct > 0.03:
            print(f"  SKIP {sym}: entry {entry:.5f} too far "
                  f"from current {current_price:.5f} "
                  f"({distance_pct*100:.2f}%)")
            continue

        # Use the SL distance from the plan
        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            print(f"  SKIP {sym}: zero SL distance")
            continue

        # Recalculate SL/TP based on current price
        new_sl = current_price - sl_distance
        new_tp = current_price + (sl_distance * BEST_RR)

        # Calculate volume
        volume = calc_volume_from_risk(mt5, sym, current_price, new_sl, risk)
        if volume is None or volume <= 0:
            print(f"  SKIP {sym}: could not calculate volume")
            continue

        print()
        print("=" * 72)
        print(f"PLACING ORDER: {sym}")
        print("=" * 72)
        print(f"  Direction:   BUY")
        print(f"  Entry:       {current_price:.5f} (market)")
        print(f"  SL:          {new_sl:.5f} (distance {sl_distance:.5f})")
        print(f"  TP:          {new_tp:.5f} (R:R {BEST_RR:.2f})")
        print(f"  Volume:      {volume}")
        print(f"  Risk:        ${risk:.2f}")
        print("=" * 72)

        if dry_run:
            print("  DRY RUN: order not sent")
            placed += 1
            continue

        result = place_buy_order(
            mt5, sym, volume, new_sl, new_tp,
        )

        if result is None:
            print("  ERROR: order_send returned None")
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ ORDER PLACED")
            print(f"     Deal:   {result.deal}")
            print(f"     Order:  {result.order}")
            print(f"     Volume: {result.volume}")
            print(f"     Price:  {result.price}")
            placed += 1
        else:
            print(f"  ❌ ORDER FAILED: retcode={result.retcode}")
            print(f"     comment={result.comment}")

    print()
    print("=" * 72)
    print(f"ORDERS PLACED: {placed}")
    print("=" * 72)

    mt5.shutdown()


if __name__ == "__main__":
    main()