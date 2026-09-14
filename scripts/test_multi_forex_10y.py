from __future__ import annotations

"""
10-Year Multi-Symbol Forex Backtest — Comprehensive Report

Extracts and saves:
    1. Trade-level data (per trade)
    2. Per-symbol metrics
    3. Per-year metrics
    4. Per-quarter metrics
    5. Per-direction metrics
    6. Per-zone-type metrics
    7. Per-session metrics
    8. Portfolio metrics (full, train, test)
    9. Equity curve
    10. Monthly returns
    11. R-multiple distribution
    12. MAE/MFE analysis

Outputs:
    data/validation/10y/
        trades.csv
        per_symbol.csv
        per_year.csv
        per_quarter.csv
        per_direction.csv
        per_zone.csv
        per_session.csv
        portfolio.json
        equity_curve.csv
        monthly_returns.csv
        r_distribution.csv
        mae_mfe.csv

Usage:
    python scripts/test_multi_forex_10y.py
"""

from pathlib import Path
import sys
import json
from statistics import mean, median
from collections import defaultdict, Counter

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
from src.backtest.backtest_engine import run_backtest


# ============================================================
# SYMBOLS
# ============================================================

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY",
    "XAUUSD", "XAGUSD",
    # WTI excluded (only ~19k candles, missing recent data)
]


# ============================================================
# CONFIG
# ============================================================

ACCOUNT_EQUITY = 10_000.0
RISK_PERCENT = 1.0
LEVERAGE = 1.0

ATR_PERIOD = 14
STRUCTURE_PIVOT_WINDOW_4H = 2
LIQUIDITY_PIVOT_WINDOW = 3
INTERNAL_PIVOT_WINDOW_1H = 2
MICRO_PIVOT_WINDOW_1H = 2
EQUAL_TOLERANCE_ATR = 0.15
SWEEP_ATR_PERIOD = 14
MSS_MAX_BARS_AFTER_SWEEP = 12
DISPLACEMENT_MIN_BODY_ATR = 1.0
DISPLACEMENT_BULLISH_CLV = 0.70
DISPLACEMENT_BEARISH_CLV = 0.30
DISPLACEMENT_MAX_BARS_AFTER_MSS = 8
OB_LOOKBACK = 8
FVG_SEARCH_BARS = 3
RETEST_MAX_BARS_AFTER_ZONE = 48
MICRO_MSS_MAX_BARS_AFTER_RETEST = 72
MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR = 0.25
RISK_REWARD = 2.0
BACKTEST_MAX_BARS = 5_000

# Train/Test split
TRAIN_END = "2022-01-01T00:00:00+00:00"

# Output dir
OUTPUT_DIR = DATA_DIR / "validation" / "10y"


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


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


def resample_from_1h(df_1h, rule):
    df = df_1h.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    out = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return out.reset_index()


def session_of(ts):
    """Return session name for a timestamp."""
    if ts is None:
        return "unknown"
    h = ts.hour
    if 0 <= h < 8:
        return "asia"
    if 8 <= h < 13:
        return "london"
    if 13 <= h < 21:
        return "ny"
    return "late"


# ============================================================
# PIPELINE
# ============================================================

def process_symbol(symbol):
    data_file = DATA_DIR / "raw" / symbol / f"{symbol}_1h_mt5.parquet"

    if not data_file.exists():
        print(f"  SKIP {symbol}: no data file")
        return []

    try:
        df_1h = pd.read_parquet(data_file)
        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], utc=True)
        df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"  SKIP {symbol}: {e}")
        return []

    if len(df_1h) == 0:
        return []

    df_4h = resample_from_1h(df_1h, "4h")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=STRUCTURE_PIVOT_WINDOW_4H,
        right_bars=STRUCTURE_PIVOT_WINDOW_4H,
        timeframe="4h",
    )

    equal_highs = detect_equal_highs(
        df_4h, atr_period=ATR_PERIOD,
        tolerance_atr=EQUAL_TOLERANCE_ATR,
        pivot_window=LIQUIDITY_PIVOT_WINDOW,
    )
    equal_lows = detect_equal_lows(
        df_4h, atr_period=ATR_PERIOD,
        tolerance_atr=EQUAL_TOLERANCE_ATR,
        pivot_window=LIQUIDITY_PIVOT_WINDOW,
    )
    previous_day_zones = detect_previous_day_liquidity(df_4h)
    swing_zones = detect_swing_liquidity(swings_4h)

    liquidity_zones = (
        equal_highs + equal_lows + previous_day_zones + swing_zones
    )
    liquidity_zones = deduplicate_zones(liquidity_zones, price_tolerance=1e-8)

    sweeps = detect_sweeps(df_1h, liquidity_zones, atr_period=SWEEP_ATR_PERIOD)

    swings_1h = detect_pivots(
        df_1h,
        left_bars=INTERNAL_PIVOT_WINDOW_1H,
        right_bars=INTERNAL_PIVOT_WINDOW_1H,
        timeframe="1h",
    )

    mss_events = detect_mss(
        df_1h, sweeps, swings_1h,
        max_bars_after_sweep=MSS_MAX_BARS_AFTER_SWEEP,
    )

    displacement_events = detect_displacements(
        df_1h, mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=DISPLACEMENT_MIN_BODY_ATR,
        bullish_clv=DISPLACEMENT_BULLISH_CLV,
        bearish_clv=DISPLACEMENT_BEARISH_CLV,
        max_bars_after_mss=DISPLACEMENT_MAX_BARS_AFTER_MSS,
    )

    order_blocks = detect_order_blocks(df_1h, displacement_events, lookback=OB_LOOKBACK)
    fvgs = detect_fvgs(
        df_1h, displacement_events,
        search_bars=FVG_SEARCH_BARS, atr_period=ATR_PERIOD,
    )

    ob_retests = detect_ob_retests(
        df_1h, order_blocks,
        max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE,
    )
    fvg_retests = detect_fvg_retests(
        df_1h, fvgs,
        max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE,
    )
    all_retests = list(ob_retests) + list(fvg_retests)

    swings_micro = detect_pivots(
        df_1h,
        left_bars=MICRO_PIVOT_WINDOW_1H,
        right_bars=MICRO_PIVOT_WINDOW_1H,
        timeframe="1h",
    )

    df_1h_indexed = df_1h.copy()
    df_1h_indexed["timestamp"] = pd.to_datetime(df_1h_indexed["timestamp"], utc=True)
    df_1h_indexed = df_1h_indexed.set_index("timestamp").sort_index()

    micro_mss_events = detect_micro_mss(
        df_1h_indexed, all_retests, swings_micro,
        max_bars_after_retest=MICRO_MSS_MAX_BARS_AFTER_RETEST,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )
    micro_mss_events = deduplicate_micro_mss(micro_mss_events)

    context = {}

    entry_prices = {}
    for _, row in df_1h.iterrows():
        ts = timestamp_key(row.get("timestamp"))
        close = safe_float(row.get("close"))
        if ts is not None and close is not None:
            entry_prices[ts] = close
    context["entry_prices"] = entry_prices

    sweeps_by_mss_timestamp = {}
    for sweep in sweeps:
        sweep_ts = get_attr(sweep, "timestamp", "sweep_timestamp")
        if sweep_ts is None:
            continue
        key = timestamp_key(sweep_ts)
        if key is not None:
            sweeps_by_mss_timestamp[key] = sweep
    context["sweeps_by_mss_timestamp"] = sweeps_by_mss_timestamp

    bias_by_timestamp = {}
    swings_4h_sorted = sorted(swings_4h, key=lambda s: str(getattr(s, "timestamp", "")))
    for micro in micro_mss_events:
        micro_ts = getattr(micro, "timestamp", None)
        if micro_ts is None:
            continue
        latest_swing = None
        for s in swings_4h_sorted:
            s_ts = getattr(s, "timestamp", None)
            if s_ts is not None and s_ts < micro_ts:
                latest_swing = s
            else:
                break
        if latest_swing is not None:
            d = normalize_direction(getattr(latest_swing, "direction", ""))
            if d:
                bias_by_timestamp[str(micro_ts)] = d
    context["bias_by_timestamp"] = bias_by_timestamp
    context["risk_reward"] = RISK_REWARD

    entry_candidates = detect_entry_candidates(
        micro_mss_events,
        all_retests,
        mss_events,
        displacement_events,
        order_blocks,
        fvgs,
        context=context,
        debug=False,
    )

    tradable_entries = [c for c in entry_candidates if getattr(c, "tradable", False)]
    if not tradable_entries:
        return []

    atr_1h_series = calculate_atr(df_1h_indexed, period=ATR_PERIOD).sort_index()

    def _atr_as_of(value):
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            r = atr_1h_series.asof(ts)
        except Exception:
            return None
        if r is None or pd.isna(r):
            return None
        return float(r)

    atr_by_timestamp = {}
    for candidate in tradable_entries:
        c_ts = getattr(candidate, "timestamp", None)
        if c_ts is None:
            continue
        v = _atr_as_of(c_ts)
        if v is not None:
            atr_by_timestamp[c_ts] = v

    risk_zones = list(order_blocks) + list(fvgs)

    risk_plans = build_risk_plans(
        tradable_entries, risk_zones, atr_by_timestamp,
        target_rr=RISK_REWARD,
    )
    valid_risk_plans = get_valid_risk_plans(risk_plans)

    if not valid_risk_plans:
        return []

    position_sizes = build_position_sizes(
        valid_risk_plans,
        account_equity=ACCOUNT_EQUITY,
        risk_percent=RISK_PERCENT,
        leverage=LEVERAGE,
    )
    valid_position_sizes = get_valid_position_sizes(position_sizes)

    if not valid_position_sizes:
        return []

    trades = run_backtest(
        valid_position_sizes,
        df_1h,
        max_bars=BACKTEST_MAX_BARS,
    )

    # Enrich with symbol
    for t in trades:
        try:
            t.symbol = symbol
        except Exception:
            pass

    print(f"  {symbol:<10} trades={len(trades)}")
    return trades


# ============================================================
# ANALYSIS
# ============================================================

def trade_to_dict(t):
    entry_ts = to_utc(get_attr(t, "entry_timestamp"))
    exit_ts = to_utc(get_attr(t, "exit_timestamp"))
    hold_hours = None
    if entry_ts is not None and exit_ts is not None:
        hold_hours = (exit_ts - entry_ts).total_seconds() / 3600.0

    return {
        "symbol": get_attr(t, "symbol", default="unknown"),
        "entry_timestamp": str(entry_ts) if entry_ts else None,
        "exit_timestamp": str(exit_ts) if exit_ts else None,
        "direction": normalize_direction(get_attr(t, "direction")),
        "entry_price": safe_float(get_attr(t, "entry_price")),
        "stop_loss": safe_float(get_attr(t, "stop_loss")),
        "take_profit": safe_float(get_attr(t, "take_profit")),
        "position_size": safe_float(get_attr(t, "position_size")),
        "risk_amount": safe_float(get_attr(t, "risk_amount")),
        "risk_per_unit": safe_float(get_attr(t, "risk_per_unit")),
        "reward_per_unit": safe_float(get_attr(t, "reward_per_unit")),
        "risk_reward": safe_float(get_attr(t, "risk_reward")),
        "exit_reason": get_attr(t, "exit_reason"),
        "bars_held": get_attr(t, "bars_held"),
        "hold_hours": hold_hours,
        "r_multiple": safe_float(get_attr(t, "r_multiple")),
        "pnl": safe_float(get_attr(t, "pnl")),
        "outcome": get_attr(t, "outcome"),
        "zone_type": get_attr(t, "zone_type"),
        "score": safe_float(get_attr(t, "score")),
        "grade": get_attr(t, "grade"),
    }


def summarize(trades, label="", verbose=True):
    """Basic summary dict."""
    closed = [t for t in trades if t["outcome"] in {"win", "loss"}]
    if not closed:
        return {
            "n": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "total_r": 0.0, "pf": 0.0, "avg_r": 0.0,
            "median_r": 0.0, "max_dd_r": 0.0,
            "avg_hold_h": 0.0, "median_hold_h": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "expectancy": 0.0,
        }

    r_values = [t["r_multiple"] for t in closed if t["r_multiple"] is not None]
    pnls = [t["pnl"] for t in closed if t["pnl"] is not None]
    holds = [t["hold_hours"] for t in closed if t["hold_hours"] is not None]

    wins = sum(1 for t in closed if t["outcome"] == "win")
    losses = sum(1 for t in closed if t["outcome"] == "loss")

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown in R
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in r_values:
        cumulative += r
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    result = {
        "n": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(closed) if closed else 0.0,
        "total_r": sum(r_values) if r_values else 0.0,
        "pf": pf,
        "avg_r": mean(r_values) if r_values else 0.0,
        "median_r": median(r_values) if r_values else 0.0,
        "max_dd_r": max_dd,
        "avg_hold_h": mean(holds) if holds else 0.0,
        "median_hold_h": median(holds) if holds else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "expectancy": mean(r_values) if r_values else 0.0,
    }

    if verbose and label:
        print(f"\n{label}:")
        print(f"  Trades:      {result['n']}")
        print(f"  Wins:        {result['wins']}")
        print(f"  Losses:      {result['losses']}")
        print(f"  Win rate:    {result['win_rate']*100:.2f}%")
        print(f"  Total R:     {result['total_r']:+.2f}")
        print(f"  Profit Factor: {result['pf']:.2f}")
        print(f"  Avg R:       {result['avg_r']:+.4f}")
        print(f"  Median R:    {result['median_r']:+.4f}")
        print(f"  Max DD (R):  {result['max_dd_r']:.2f}")
        print(f"  Avg Hold:    {result['avg_hold_h']:.1f}h")

    return result


def save_csv(rows, path, name):
    if not rows:
        print(f"  {name}: no rows")
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved: {path.name} ({len(df)} rows)")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("10-YEAR MULTI-SYMBOL FOREX BACKTEST")
    print(f"Symbols:    {len(SYMBOLS)}")
    print(f"Split date: {TRAIN_END}")
    print(f"Output:     {OUTPUT_DIR}")

    # ---- Run pipeline ----
    print_section("RUNNING PIPELINE")

    all_trades_raw = []
    for sym in SYMBOLS:
        trades = process_symbol(sym)
        all_trades_raw.extend(trades)

    # ---- Convert to dicts ----
    all_trades = [trade_to_dict(t) for t in all_trades_raw]

    print(f"\nTotal trades: {len(all_trades)}")

    # ========================================================
    # 1. TRADES CSV
    # ========================================================

    print_section("1. TRADE-LEVEL DATA")
    save_csv(all_trades, OUTPUT_DIR / "trades.csv", "Trades")

    # ========================================================
    # 2. PER-SYMBOL
    # ========================================================

    print_section("2. PER-SYMBOL METRICS")

    per_symbol = []
    for sym in SYMBOLS:
        sym_trades = [t for t in all_trades if t["symbol"] == sym]
        if not sym_trades:
            continue
        s = summarize(sym_trades, verbose=False)
        s["symbol"] = sym
        per_symbol.append(s)

    # Sort by Total R
    per_symbol.sort(key=lambda x: -x["total_r"])

    print(f"\n{'Symbol':<10} {'Trades':>7} {'Wins':>6} {'Losses':>7} "
          f"{'WinRate':>9} {'TotalR':>9} {'PF':>7}")
    print("-" * 72)
    for s in per_symbol:
        pf = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "inf"
        print(f"{s['symbol']:<10} {s['n']:>7} {s['wins']:>6} {s['losses']:>7} "
              f"{s['win_rate']*100:>8.2f}% {s['total_r']:>+9.2f} {pf:>7}")

    save_csv(per_symbol, OUTPUT_DIR / "per_symbol.csv", "Per-symbol")

    # ========================================================
    # 3. PER-YEAR
    # ========================================================

    print_section("3. PER-YEAR METRICS")

    per_year_dict = defaultdict(list)
    for t in all_trades:
        ts = to_utc(t["entry_timestamp"])
        if ts is None:
            continue
        per_year_dict[ts.year].append(t)

    per_year = []
    for year in sorted(per_year_dict.keys()):
        s = summarize(per_year_dict[year], verbose=False)
        s["year"] = year
        per_year.append(s)

    print(f"\n{'Year':<6} {'Trades':>7} {'Wins':>6} {'Losses':>7} "
          f"{'WinRate':>9} {'TotalR':>9} {'PF':>7}")
    print("-" * 72)
    for s in per_year:
        pf = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "inf"
        print(f"{s['year']:<6} {s['n']:>7} {s['wins']:>6} {s['losses']:>7} "
              f"{s['win_rate']*100:>8.2f}% {s['total_r']:>+9.2f} {pf:>7}")

    save_csv(per_year, OUTPUT_DIR / "per_year.csv", "Per-year")

    # ========================================================
    # 4. PER-QUARTER
    # ========================================================

    print_section("4. PER-QUARTER METRICS")

    per_quarter_dict = defaultdict(list)
    for t in all_trades:
        ts = to_utc(t["entry_timestamp"])
        if ts is None:
            continue
        q = (ts.month - 1) // 3 + 1
        key = f"{ts.year}Q{q}"
        per_quarter_dict[key].append(t)

    per_quarter = []
    for key in sorted(per_quarter_dict.keys()):
        s = summarize(per_quarter_dict[key], verbose=False)
        s["quarter"] = key
        per_quarter.append(s)

    print(f"\n{'Quarter':<10} {'Trades':>7} {'WinRate':>9} {'TotalR':>9}")
    print("-" * 72)
    for s in per_quarter:
        print(f"{s['quarter']:<10} {s['n']:>7} "
              f"{s['win_rate']*100:>8.2f}% {s['total_r']:>+9.2f}")

    save_csv(per_quarter, OUTPUT_DIR / "per_quarter.csv", "Per-quarter")

    # ========================================================
    # 5. PER-DIRECTION
    # ========================================================

    print_section("5. PER-DIRECTION METRICS")

    per_dir = []
    for direction in ["bullish", "bearish"]:
        dir_trades = [t for t in all_trades if t["direction"] == direction]
        if not dir_trades:
            continue
        s = summarize(dir_trades, verbose=False)
        s["direction"] = direction
        per_dir.append(s)

    print(f"\n{'Direction':<10} {'Trades':>7} {'WinRate':>9} {'TotalR':>9}")
    print("-" * 72)
    for s in per_dir:
        print(f"{s['direction']:<10} {s['n']:>7} "
              f"{s['win_rate']*100:>8.2f}% {s['total_r']:>+9.2f}")

    save_csv(per_dir, OUTPUT_DIR / "per_direction.csv", "Per-direction")

    # ========================================================
    # 6. PER-ZONE-TYPE
    # ========================================================

    print_section("6. PER-ZONE-TYPE METRICS")

    per_zone = []
    for zone in ["ob", "fvg"]:
        zone_trades = [t for t in all_trades if t["zone_type"] == zone]
        if not zone_trades:
            continue
        s = summarize(zone_trades, verbose=False)
        s["zone_type"] = zone
        per_zone.append(s)

    print(f"\n{'Zone':<8} {'Trades':>7} {'WinRate':>9} {'TotalR':>9}")
    print("-" * 72)
    for s in per_zone:
        print(f"{s['zone_type']:<8} {s['n']:>7} "
              f"{s['win_rate']*100:>8.2f}% {s['total_r']:>+9.2f}")

    save_csv(per_zone, OUTPUT_DIR / "per_zone.csv", "Per-zone")

    # ========================================================
    # 7. PER-SESSION
    # ========================================================

    print_section("7. PER-SESSION METRICS")

    per_session_dict = defaultdict(list)
    for t in all_trades:
        ts = to_utc(t["entry_timestamp"])
        if ts is None:
            continue
        sess = session_of(ts)
        per_session_dict[sess].append(t)

    per_session = []
    for sess in ["asia", "london", "ny", "late", "unknown"]:
        if sess not in per_session_dict:
            continue
        s = summarize(per_session_dict[sess], verbose=False)
        s["session"] = sess
        per_session.append(s)

    print(f"\n{'Session':<10} {'Trades':>7} {'WinRate':>9} {'TotalR':>9}")
    print("-" * 72)
    for s in per_session:
        print(f"{s['session']:<10} {s['n']:>7} "
              f"{s['win_rate']*100:>8.2f}% {s['total_r']:>+9.2f}")

    save_csv(per_session, OUTPUT_DIR / "per_session.csv", "Per-session")

    # ========================================================
    # 8. PORTFOLIO (FULL / TRAIN / TEST)
    # ========================================================

    print_section("8. PORTFOLIO METRICS")

    split_ts = to_utc(TRAIN_END)

    train_trades = []
    test_trades = []
    for t in all_trades:
        ts = to_utc(t["entry_timestamp"])
        if ts is None:
            continue
        if ts < split_ts:
            train_trades.append(t)
        else:
            test_trades.append(t)

    full_s = summarize(all_trades, "FULL 10-YEAR")
    train_s = summarize(train_trades, f"TRAIN (before {TRAIN_END[:10]})")
    test_s = summarize(test_trades, f"TEST (from {TRAIN_END[:10]})")

    portfolio = {
        "full": full_s,
        "train": train_s,
        "test": test_s,
    }

    with open(OUTPUT_DIR / "portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=2)
    print(f"\n  Saved: portfolio.json")

    # ========================================================
    # 9. EQUITY CURVE
    # ========================================================

    print_section("9. EQUITY CURVE")

    sorted_trades = sorted(
        [t for t in all_trades if t["outcome"] in {"win", "loss"}],
        key=lambda x: x["entry_timestamp"] or "",
    )

    equity = ACCOUNT_EQUITY
    equity_curve = [{"timestamp": None, "equity": equity}]

    for t in sorted_trades:
        if t["pnl"] is not None:
            equity += t["pnl"]
            equity_curve.append({
                "timestamp": t["exit_timestamp"],
                "equity": equity,
            })

    save_csv(equity_curve, OUTPUT_DIR / "equity_curve.csv", "Equity curve")

    # ========================================================
    # 10. MONTHLY RETURNS
    # ========================================================

    print_section("10. MONTHLY RETURNS")

    monthly_dict = defaultdict(list)
    for t in all_trades:
        ts = to_utc(t["entry_timestamp"])
        if ts is None:
            continue
        key = f"{ts.year}-{ts.month:02d}"
        monthly_dict[key].append(t)

    monthly = []
    for key in sorted(monthly_dict.keys()):
        s = summarize(monthly_dict[key], verbose=False)
        monthly.append({
            "month": key,
            "trades": s["n"],
            "total_r": s["total_r"],
            "win_rate": s["win_rate"],
        })

    save_csv(monthly, OUTPUT_DIR / "monthly_returns.csv", "Monthly returns")

    # ========================================================
    # 11. R-MULTIPLE DISTRIBUTION
    # ========================================================

    print_section("11. R-MULTIPLE DISTRIBUTION")

    r_values = [t["r_multiple"] for t in all_trades if t["r_multiple"] is not None]

    bins = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 5]
    labels = ["<-2", "-2~-1.5", "-1.5~-1", "-1~-0.5", "-0.5~0",
              "0~0.5", "0.5~1", "1~1.5", "1.5~2", "2~2.5", "2.5~3", ">3"]

    dist = []
    for i in range(len(bins) - 1):
        count = sum(1 for r in r_values if bins[i] <= r < bins[i+1])
        dist.append({"range": labels[i], "count": count})

    print(f"\n{'R Range':<12} {'Count':>8}")
    print("-" * 24)
    for d in dist:
        print(f"{d['range']:<12} {d['count']:>8}")

    save_csv(dist, OUTPUT_DIR / "r_distribution.csv", "R distribution")

    # ========================================================
    # 12. FINAL INTERPRETATION
    # ========================================================

    print_section("12. INTERPRETATION")

    print(f"\nFull 10-year:  R = {full_s['total_r']:+.2f}  "
          f"WinRate = {full_s['win_rate']*100:.2f}%  n = {full_s['n']}")
    print(f"TRAIN:         R = {train_s['total_r']:+.2f}  "
          f"WinRate = {train_s['win_rate']*100:.2f}%  n = {train_s['n']}")
    print(f"TEST:          R = {test_s['total_r']:+.2f}  "
          f"WinRate = {test_s['win_rate']*100:.2f}%  n = {test_s['n']}")

    print()
    if test_s["total_r"] > 0:
        print("-> TEST is POSITIVE. Strategy holds out-of-sample.")
    else:
        print("-> TEST is NEGATIVE. Strategy does NOT hold out-of-sample.")

    print()
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()