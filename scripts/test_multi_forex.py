from __future__ import annotations

"""
Multi-Symbol Forex Backtest

Runs the adapted 1H pipeline for all Forex / Gold / Index symbols
and produces a comparative summary.

Symbols (16):
    EURUSD, GBPUSD, USDJPY, XAUUSD, SP500, NAS100
    AUDUSD, USDCAD, USDCHF, NZDUSD
    EURGBP, EURJPY, GBPJPY, AUDJPY
    XAGUSD, WTI

Usage:
    python scripts/test_multi_forex.py
"""

from pathlib import Path
import sys
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import DATA_DIR, YEAR
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
# SYMBOLS (16 total)
# ============================================================

SYMBOLS = [
    # Original 6
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "XAUUSD",
    "SP500",
    "NAS100",

    # New 10
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "XAGUSD",
    "WTI",
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


# ============================================================
# PER-SYMBOL PIPELINE
# ============================================================

def process_symbol(symbol):
    """
    Run the full 1H-adapted pipeline for one symbol.
    Returns a result dict (with trades list).
    """

    data_file = DATA_DIR / "raw" / symbol / f"{symbol}_1h_{YEAR}.parquet"

    result = {
        "symbol": symbol,
        "data_file": str(data_file),
        "error": None,
        "candles_1h": 0,
        "micro_mss": 0,
        "tradable": 0,
        "valid_plans": 0,
        "trades": [],
    }

    if not data_file.exists():
        result["error"] = "data file missing"
        return result

    # ---- Load 1H ----
    try:
        df_1h = pd.read_parquet(data_file)
        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], utc=True)
        df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        result["error"] = f"load failed: {e}"
        return result

    if len(df_1h) == 0:
        result["error"] = "empty data"
        return result

    result["candles_1h"] = len(df_1h)

    # ---- Resample to 4H ----
    df_4h = resample_from_1h(df_1h, "4h")

    # ---- 4H Structure ----
    swings_4h = detect_pivots(
        df_4h,
        left_bars=STRUCTURE_PIVOT_WINDOW_4H,
        right_bars=STRUCTURE_PIVOT_WINDOW_4H,
        timeframe="4h",
    )

    # ---- 4H Liquidity ----
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

    # ---- Sweeps ----
    sweeps = detect_sweeps(df_1h, liquidity_zones, atr_period=SWEEP_ATR_PERIOD)

    # ---- 1H Internal Structure ----
    swings_1h = detect_pivots(
        df_1h,
        left_bars=INTERNAL_PIVOT_WINDOW_1H,
        right_bars=INTERNAL_PIVOT_WINDOW_1H,
        timeframe="1h",
    )

    # ---- MSS ----
    mss_events = detect_mss(
        df_1h, sweeps, swings_1h,
        max_bars_after_sweep=MSS_MAX_BARS_AFTER_SWEEP,
    )

    # ---- Displacement ----
    displacement_events = detect_displacements(
        df_1h, mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=DISPLACEMENT_MIN_BODY_ATR,
        bullish_clv=DISPLACEMENT_BULLISH_CLV,
        bearish_clv=DISPLACEMENT_BEARISH_CLV,
        max_bars_after_mss=DISPLACEMENT_MAX_BARS_AFTER_MSS,
    )

    # ---- OB / FVG ----
    order_blocks = detect_order_blocks(df_1h, displacement_events, lookback=OB_LOOKBACK)
    fvgs = detect_fvgs(
        df_1h, displacement_events,
        search_bars=FVG_SEARCH_BARS, atr_period=ATR_PERIOD,
    )

    # ---- Retests ----
    ob_retests = detect_ob_retests(
        df_1h, order_blocks,
        max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE,
    )
    fvg_retests = detect_fvg_retests(
        df_1h, fvgs,
        max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE,
    )
    all_retests = list(ob_retests) + list(fvg_retests)

    # ---- Micro Structure on 1H ----
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

    result["micro_mss"] = len(micro_mss_events)

    # ---- Entry Detector Context ----
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
    result["tradable"] = len(tradable_entries)

    if not tradable_entries:
        return result

    # ---- ATR as-of ----
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

    # ---- Risk Manager ----
    risk_zones = list(order_blocks) + list(fvgs)

    risk_plans = build_risk_plans(
        tradable_entries, risk_zones, atr_by_timestamp,
        target_rr=RISK_REWARD,
    )
    valid_risk_plans = get_valid_risk_plans(risk_plans)
    result["valid_plans"] = len(valid_risk_plans)

    if not valid_risk_plans:
        return result

    # ---- Position Sizer ----
    position_sizes = build_position_sizes(
        valid_risk_plans,
        account_equity=ACCOUNT_EQUITY,
        risk_percent=RISK_PERCENT,
        leverage=LEVERAGE,
    )
    valid_position_sizes = get_valid_position_sizes(position_sizes)

    if not valid_position_sizes:
        return result

    # ---- Backtest ----
    trades = run_backtest(
        valid_position_sizes,
        df_1h,
        max_bars=BACKTEST_MAX_BARS,
    )
    result["trades"] = trades

    return result


# ============================================================
# ANALYSIS
# ============================================================

def summarize(trades):
    closed = [
        t for t in trades
        if getattr(t, "outcome", None) in {"win", "loss"}
    ]
    if not closed:
        return {
            "n": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "total_r": 0.0, "pf": 0.0, "avg_r": 0.0,
            "avg_hold_h": 0.0,
        }

    wins = sum(1 for t in closed if t.outcome == "win")
    losses = sum(1 for t in closed if t.outcome == "loss")
    total_r = sum(float(t.r_multiple or 0.0) for t in closed)

    gross_profit = sum(
        float(t.pnl or 0.0) for t in closed if t.pnl and t.pnl > 0
    )
    gross_loss = abs(sum(
        float(t.pnl or 0.0) for t in closed if t.pnl and t.pnl < 0
    ))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    hold_hours = []
    for t in closed:
        try:
            e = pd.Timestamp(t.entry_timestamp)
            x = pd.Timestamp(t.exit_timestamp)
            if e.tzinfo is None: e = e.tz_localize("UTC")
            if x.tzinfo is None: x = x.tz_localize("UTC")
            hold_hours.append((x - e).total_seconds() / 3600.0)
        except Exception:
            pass

    return {
        "n": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(closed),
        "total_r": total_r,
        "pf": pf,
        "avg_r": total_r / len(closed),
        "avg_hold_h": mean(hold_hours) if hold_hours else 0.0,
    }


def main():
    print_section("MULTI-FOREX BACKTEST")
    print(f"Year:    {YEAR}")
    print(f"Symbols: {len(SYMBOLS)}")
    for s in SYMBOLS:
        print(f"  - {s}")

    print()
    print("Running pipeline for each symbol ...")

    results = {}
    for sym in SYMBOLS:
        print(f"\n--- {sym} ---")
        r = process_symbol(sym)
        results[sym] = r

        if r["error"]:
            print(f"  ERROR: {r['error']}")
            continue

        print(f"  1H candles:   {r['candles_1h']:,}")
        print(f"  Micro-MSS:    {r['micro_mss']}")
        print(f"  Tradable:     {r['tradable']}")
        print(f"  Valid plans:  {r['valid_plans']}")
        print(f"  Trades:       {len(r['trades'])}")

    # ---- Summary table ----
    print_section("PER-SYMBOL SUMMARY")

    header = (
        f"{'Symbol':<10} "
        f"{'Trades':>7} "
        f"{'Wins':>6} "
        f"{'Losses':>7} "
        f"{'WinRate':>9} "
        f"{'TotalR':>9} "
        f"{'PF':>7} "
        f"{'AvgR':>8} "
        f"{'Hold(h)':>9}"
    )
    print(header)
    print("-" * len(header))

    for sym in SYMBOLS:
        r = results[sym]
        if r["error"]:
            print(f"{sym:<10} ERROR: {r['error']}")
            continue

        s = summarize(r["trades"])
        pf_str = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "inf"

        print(
            f"{sym:<10} "
            f"{s['n']:>7} "
            f"{s['wins']:>6} "
            f"{s['losses']:>7} "
            f"{s['win_rate']*100:>8.2f}% "
            f"{s['total_r']:>+9.2f} "
            f"{pf_str:>7} "
            f"{s['avg_r']:>+8.4f} "
            f"{s['avg_hold_h']:>9.1f}"
        )

    # ---- Portfolio totals ----
    print("-" * len(header))

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_r = 0.0

    for sym in SYMBOLS:
        r = results[sym]
        if r["error"]:
            continue
        s = summarize(r["trades"])
        total_trades += s["n"]
        total_wins += s["wins"]
        total_losses += s["losses"]
        total_r += s["total_r"]

    if total_trades > 0:
        portfolio_wr = total_wins / (total_wins + total_losses)
        print(
            f"{'PORTFOLIO':<10} "
            f"{total_trades:>7} "
            f"{total_wins:>6} "
            f"{total_losses:>7} "
            f"{portfolio_wr*100:>8.2f}% "
            f"{total_r:>+9.2f}"
        )

    # ---- Ranked by Total R ----
    print_section("SYMBOLS RANKED BY TOTAL R")

    ranked = []
    for sym in SYMBOLS:
        r = results[sym]
        if r["error"]:
            continue
        s = summarize(r["trades"])
        ranked.append((sym, s["total_r"], s["win_rate"], s["n"]))

    ranked.sort(key=lambda x: -x[1])

    for i, (sym, tr, wr, n) in enumerate(ranked, 1):
        print(
            f"  {i:>2}. {sym:<10} "
            f"R={tr:+8.2f}  "
            f"WinRate={wr*100:>5.1f}%  "
            f"n={n:>3}"
        )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()