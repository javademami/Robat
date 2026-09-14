from __future__ import annotations

"""
Optimal Combination Test — Top4 + Bullish

Tests different:
    - Risk-Reward ratios: 1.5, 2.0, 2.5, 3.0
    - SL buffer ATR: 0.05, 0.10, 0.15, 0.20
    - Micro-MSS windows: 48, 72, 96
    - Retest windows: 24, 48, 72

For each combination, reports:
    - Full 10-year
    - TRAIN (before 2022)
    - TEST (from 2022)

Uses the combined 10-year MT5 data.

Output:
    data/validation/10y/optimal_combination.csv

Usage:
    python scripts/test_forex_optimal.py
"""

from pathlib import Path
import sys
from statistics import mean, median
from datetime import datetime, timezone

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
# CONFIG
# ============================================================

SYMBOLS = ["XAUUSD", "GBPJPY", "USDJPY", "EURJPY"]

ACCOUNT_EQUITY = 10_000.0
RISK_PERCENT = 1.0
LEVERAGE = 1.0

# Fixed parameters
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
MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR = 0.25
BACKTEST_MAX_BARS = 5_000

# Only bullish trades
DIRECTION_FILTER = "bullish"

# Train/Test
TRAIN_END = "2022-01-01T00:00:00+00:00"


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


def compute_metrics(trades):
    closed = [t for t in trades if getattr(t, "outcome", None) in {"win", "loss"}]
    n = len(closed)
    if n == 0:
        return None

    wins = sum(1 for t in closed if t.outcome == "win")
    losses = sum(1 for t in closed if t.outcome == "loss")

    r_values = [float(t.r_multiple or 0.0) for t in closed]
    pnls = [float(t.pnl or 0.0) for t in closed]

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in r_values:
        cumulative += r
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    hold_hours = []
    for t in closed:
        e = to_utc(getattr(t, "entry_timestamp", None))
        x = to_utc(getattr(t, "exit_timestamp", None))
        if e is not None and x is not None:
            hold_hours.append((x - e).total_seconds() / 3600.0)

    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n,
        "total_r": sum(r_values),
        "pf": pf,
        "avg_r": mean(r_values),
        "median_r": median(r_values),
        "max_dd_r": max_dd,
        "avg_hold_h": mean(hold_hours) if hold_hours else 0.0,
    }


# ============================================================
# PIPELINE
# ============================================================

def process_symbol(symbol, target_rr, buffer_atr, retest_window, micro_mss_window):
    data_file = DATA_DIR / "raw" / symbol / f"{symbol}_1h_mt5.parquet"

    if not data_file.exists():
        return []

    try:
        df_1h = pd.read_parquet(data_file)
        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], utc=True)
        df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)
    except Exception:
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
        max_bars_after_zone=retest_window,
    )
    fvg_retests = detect_fvg_retests(
        df_1h, fvgs,
        max_bars_after_zone=retest_window,
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
        max_bars_after_retest=micro_mss_window,
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
    context["risk_reward"] = target_rr

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

    # Filter: only bullish
    tradable_entries = [
        c for c in entry_candidates
        if getattr(c, "tradable", False)
        and normalize_direction(getattr(c, "direction", "")) == DIRECTION_FILTER
    ]

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
        target_rr=target_rr,
        buffer_atr=buffer_atr,
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

    for t in trades:
        try:
            t.symbol = symbol
        except Exception:
            pass

    return trades


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("OPTIMAL COMBINATION TEST — Top4 + Bullish")
    print("=" * 72)

    # Parameter grids
    rr_options = [1.5, 2.0, 2.5, 3.0]
    buffer_options = [0.05, 0.10, 0.15]
    retest_options = [24, 48]
    micro_mss_options = [48, 72]

    results = []

    total_combos = (
        len(rr_options) * len(buffer_options)
        * len(retest_options) * len(micro_mss_options)
    )

    print(f"Symbols:     {SYMBOLS}")
    print(f"Direction:   {DIRECTION_FILTER}")
    print(f"Combinations: {total_combos}")
    print()

    combo_num = 0
    for rr in rr_options:
        for buf in buffer_options:
            for retest in retest_options:
                for micro in micro_mss_options:
                    combo_num += 1
                    print(
                        f"[{combo_num}/{total_combos}] "
                        f"RR={rr} buffer={buf} retest={retest} micro={micro}"
                    )

                    all_trades = []
                    for sym in SYMBOLS:
                        trades = process_symbol(
                            sym,
                            target_rr=rr,
                            buffer_atr=buf,
                            retest_window=retest,
                            micro_mss_window=micro,
                        )
                        all_trades.extend(trades)

                    if not all_trades:
                        continue

                    # Split Train/Test
                    split_ts = to_utc(TRAIN_END)
                    train = []
                    test = []
                    for t in all_trades:
                        ts = to_utc(getattr(t, "entry_timestamp", None))
                        if ts is None:
                            continue
                        if ts < split_ts:
                            train.append(t)
                        else:
                            test.append(t)

                    full_m = compute_metrics(all_trades)
                    train_m = compute_metrics(train)
                    test_m = compute_metrics(test)

                    if full_m is None:
                        continue

                    row = {
                        "target_rr": rr,
                        "buffer_atr": buf,
                        "retest_window": retest,
                        "micro_mss_window": micro,
                        "full_n": full_m["n"],
                        "full_wr": full_m["win_rate"],
                        "full_total_r": full_m["total_r"],
                        "full_pf": full_m["pf"],
                        "full_max_dd": full_m["max_dd_r"],
                        "full_avg_hold": full_m["avg_hold_h"],
                        "train_n": train_m["n"] if train_m else 0,
                        "train_total_r": train_m["total_r"] if train_m else 0.0,
                        "train_wr": train_m["win_rate"] if train_m else 0.0,
                        "test_n": test_m["n"] if test_m else 0,
                        "test_total_r": test_m["total_r"] if test_m else 0.0,
                        "test_wr": test_m["win_rate"] if test_m else 0.0,
                    }
                    results.append(row)

                    print(
                        f"  → n={full_m['n']}  WR={full_m['win_rate']*100:.1f}%  "
                        f"R={full_m['total_r']:+.1f}  PF={full_m['pf']:.2f}  "
                        f"| train={train_m['total_r'] if train_m else 0:+.1f}  "
                        f"test={test_m['total_r'] if test_m else 0:+.1f}"
                    )

    if not results:
        print("\nNo results. Check data files.")
        return

    # Sort by full Total R
    results.sort(key=lambda x: -x["full_total_r"])

    print()
    print("=" * 72)
    print("TOP 10 COMBINATIONS (by Full Total R)")
    print("=" * 72)
    print()

    header = (
        f"{'#':>2} {'RR':>5} {'Buf':>5} {'Ret':>4} {'Mic':>4} "
        f"{'N':>5} {'WR':>7} {'TotalR':>8} {'PF':>6} "
        f"{'TrainR':>8} {'TestR':>8}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results[:10], 1):
        print(
            f"{i:>2} "
            f"{r['target_rr']:>5.1f} "
            f"{r['buffer_atr']:>5.2f} "
            f"{r['retest_window']:>4} "
            f"{r['micro_mss_window']:>4} "
            f"{r['full_n']:>5} "
            f"{r['full_wr']*100:>6.2f}% "
            f"{r['full_total_r']:>+8.2f} "
            f"{r['full_pf']:>6.2f} "
            f"{r['train_total_r']:>+8.2f} "
            f"{r['test_total_r']:>+8.2f}"
        )

    # Save
    output_csv = DATA_DIR / "validation" / "10y" / "optimal_combination.csv"
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()