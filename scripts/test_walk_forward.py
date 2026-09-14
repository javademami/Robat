"""
Walk-Forward Analysis — Best Combination

Tests the best combination (RR=3.0, buffer=0.10, retest=72, micro=60)
on rolling 2-year windows across the 10-year dataset.

Approach:
    For each window:
        - Compute trades in that window
        - Report metrics (WR, TotalR, PF, DD)

Windows:
    2015-2016, 2016-2017, 2017-2018, 2018-2019, 2019-2020,
    2020-2021, 2021-2022, 2022-2023, 2023-2024, 2024-2025

Also reports a "rolling" walk-forward where each window is
out-of-sample (train on prior years, test on next year).

Usage:
    python scripts/test_walk_forward.py
"""

from pathlib import Path
import sys
import itertools
from statistics import mean
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
BACKTEST_MAX_BARS = 5_000

DIRECTION_FILTER = "bullish"

# BEST PARAMS FROM MULTI-PARAM TEST
BEST_RR = 3.0
BEST_BUFFER = 0.10
BEST_RETEST = 72
BEST_MICRO = 60


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
    r_values = [float(t.r_multiple or 0.0) for t in closed]
    pnls = [float(t.pnl or 0.0) for t in closed]

    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = (gp / gl) if gl > 0 else float("inf")

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in r_values:
        cumulative += r
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    return {
        "n": n, "wins": wins, "losses": n - wins,
        "win_rate": wins / n,
        "total_r": sum(r_values),
        "pf": pf,
        "max_dd_r": max_dd,
    }


# ============================================================
# PIPELINE (best params fixed)
# ============================================================

def precompute_lower(symbol):
    """Lower layers — only depends on symbol, not on params."""
    data_file = DATA_DIR / "raw" / symbol / f"{symbol}_1h_mt5.parquet"
    if not data_file.exists():
        print(f"    ERROR: {data_file} not found")
        return None

    df_1h = pd.read_parquet(data_file)
    df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], utc=True)
    df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)

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

    df_1h_idx = df_1h.copy()
    df_1h_idx["timestamp"] = pd.to_datetime(df_1h_idx["timestamp"], utc=True)
    df_1h_idx = df_1h_idx.set_index("timestamp").sort_index()

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

    atr_series = calculate_atr(df_1h_idx, period=ATR_PERIOD).sort_index()

    return {
        "symbol": symbol,
        "df_1h": df_1h,
        "df_1h_idx": df_1h_idx,
        "swings_4h": swings_4h,
        "swings_1h": swings_1h,
        "obs": obs,
        "fvgs": fvgs,
        "disps": disps,
        "mss_events": mss_events,
        "entry_prices": entry_prices,
        "sweeps_by_mss": sweeps_by_mss,
        "atr_series": atr_series,
    }


def run_best(precomp):
    """Run the full pipeline with best params for one symbol."""

    # Retests
    ob_r = detect_ob_retests(precomp["df_1h"], precomp["obs"],
                              max_bars_after_zone=BEST_RETEST)
    fvg_r = detect_fvg_retests(precomp["df_1h"], precomp["fvgs"],
                                max_bars_after_zone=BEST_RETEST)
    retests = list(ob_r) + list(fvg_r)

    if not retests:
        return []

    # Micro-MSS
    micro = detect_micro_mss(
        precomp["df_1h_idx"], retests, precomp["swings_1h"],
        max_bars_after_retest=BEST_MICRO,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )
    micro = deduplicate_micro_mss(micro)

    if not micro:
        return []

    # Bias map
    bias_map = {}
    sw4_sorted = sorted(
        precomp["swings_4h"],
        key=lambda s: str(getattr(s, "timestamp", "")),
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

    context = {
        "entry_prices": precomp["entry_prices"],
        "sweeps_by_mss_timestamp": precomp["sweeps_by_mss"],
        "risk_reward": BEST_RR,
        "bias_by_timestamp": bias_map,
    }

    candidates = detect_entry_candidates(
        micro, retests, precomp["mss_events"], precomp["disps"],
        precomp["obs"], precomp["fvgs"],
        context=context, debug=False,
    )

    tradable = [
        c for c in candidates
        if getattr(c, "tradable", False)
        and normalize_direction(getattr(c, "direction", "")) == DIRECTION_FILTER
    ]

    if not tradable:
        return []

    # ATR map
    atr_map = {}
    for c in tradable:
        c_ts = getattr(c, "timestamp", None)
        if c_ts is None:
            continue
        try:
            ts = pd.Timestamp(c_ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            r = precomp["atr_series"].asof(ts)
            if r is not None and not pd.isna(r):
                atr_map[c_ts] = float(r)
        except Exception:
            pass

    zones = list(precomp["obs"]) + list(precomp["fvgs"])
    plans = build_risk_plans(
        tradable, zones, atr_map,
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
    if not valid_sizes:
        return []

    trades = run_backtest(
        valid_sizes, precomp["df_1h"], max_bars=BACKTEST_MAX_BARS,
    )

    for t in trades:
        try:
            t.symbol = precomp["symbol"]
        except Exception:
            pass

    return trades


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 72)
    print(f"Symbols:  {SYMBOLS}")
    print(f"Direction: {DIRECTION_FILTER}")
    print(f"Params:  RR={BEST_RR}, buffer={BEST_BUFFER}, "
          f"retest={BEST_RETEST}, micro={BEST_MICRO}")
    print()

    # ---- Precompute and run ----
    print("Running pipeline per symbol ...")
    all_trades = []
    for sym in SYMBOLS:
        print(f"  {sym}:")
        p = precompute_lower(sym)
        if p is None:
            continue
        trades = run_best(p)
        print(f"    trades={len(trades)}")
        all_trades.extend(trades)

    if not all_trades:
        print("No trades. Aborting.")
        return

    print(f"\nTotal trades: {len(all_trades)}")
    print()

    # ---- Overall ----
    full = compute_metrics(all_trades)
    print("=" * 72)
    print("OVERALL (10 years)")
    print("=" * 72)
    print(f"  Trades:      {full['n']}")
    print(f"  Win rate:    {full['win_rate']*100:.2f}%")
    print(f"  Total R:     {full['total_r']:+.2f}")
    print(f"  PF:          {full['pf']:.2f}")
    print(f"  Max DD:      {full['max_dd_r']:.2f}")
    print()

    # ---- Per-year metrics ----
    print("=" * 72)
    print("PER-YEAR METRICS")
    print("=" * 72)
    print()

    by_year = {}
    for t in all_trades:
        ts = to_utc(getattr(t, "entry_timestamp", None))
        if ts is None:
            continue
        year = ts.year
        by_year.setdefault(year, []).append(t)

    header = f"{'Year':<6} {'N':>5} {'WR':>8} {'TotalR':>9} {'PF':>7} {'MaxDD':>7}"
    print(header)
    print("-" * len(header))

    for year in sorted(by_year.keys()):
        m = compute_metrics(by_year[year])
        if m is None:
            continue
        print(
            f"{year:<6} {m['n']:>5} {m['win_rate']*100:>7.2f}% "
            f"{m['total_r']:>+9.2f} {m['pf']:>7.2f} {m['max_dd_r']:>7.2f}"
        )

    # ---- Rolling 2-year windows ----
    print()
    print("=" * 72)
    print("ROLLING 2-YEAR WINDOWS")
    print("=" * 72)
    print()

    years = sorted(by_year.keys())
    header = f"{'Window':<12} {'N':>5} {'WR':>8} {'TotalR':>9} {'PF':>7} {'MaxDD':>7}"
    print(header)
    print("-" * len(header))

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i+1]
        window_trades = by_year.get(y1, []) + by_year.get(y2, [])
        if not window_trades:
            continue
        m = compute_metrics(window_trades)
        if m is None:
            continue
        label = f"{y1}-{y2}"
        print(
            f"{label:<12} {m['n']:>5} {m['win_rate']*100:>7.2f}% "
            f"{m['total_r']:>+9.2f} {m['pf']:>7.2f} {m['max_dd_r']:>7.2f}"
        )

    # ---- Per-quarter ----
    print()
    print("=" * 72)
    print("PER-QUARTER METRICS (last 20)")
    print("=" * 72)
    print()

    by_q = {}
    for t in all_trades:
        ts = to_utc(getattr(t, "entry_timestamp", None))
        if ts is None:
            continue
        q = (ts.month - 1) // 3 + 1
        key = f"{ts.year}Q{q}"
        by_q.setdefault(key, []).append(t)

    header = f"{'Quarter':<10} {'N':>5} {'WR':>8} {'TotalR':>9}"
    print(header)
    print("-" * len(header))

    for key in sorted(by_q.keys())[-20:]:
        m = compute_metrics(by_q[key])
        if m is None:
            continue
        print(
            f"{key:<10} {m['n']:>5} {m['win_rate']*100:>7.2f}% "
            f"{m['total_r']:>+9.2f}"
        )

    # ---- Summary ----
    print()
    print("=" * 72)
    print("WALK-FORWARD SUMMARY")
    print("=" * 72)

    # Number of positive years
    pos_years = 0
    neg_years = 0
    for year in years:
        m = compute_metrics(by_year[year])
        if m and m["total_r"] > 0:
            pos_years += 1
        elif m and m["total_r"] < 0:
            neg_years += 1

    print(f"  Positive years: {pos_years}")
    print(f"  Negative years: {neg_years}")
    print()

    # Number of positive 2-year windows
    pos_windows = 0
    neg_windows = 0
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i+1]
        wt = by_year.get(y1, []) + by_year.get(y2, [])
        if not wt:
            continue
        m = compute_metrics(wt)
        if m and m["total_r"] > 0:
            pos_windows += 1
        elif m and m["total_r"] < 0:
            neg_windows += 1

    print(f"  Positive 2-year windows: {pos_windows}")
    print(f"  Negative 2-year windows: {neg_windows}")
    print()

    # Save
    output_dir = DATA_DIR / "validation" / "10y"
    rows = []
    for t in all_trades:
        rows.append({
            "symbol": get_attr(t, "symbol", default="unknown"),
            "entry_timestamp": str(to_utc(getattr(t, "entry_timestamp", None))),
            "exit_timestamp": str(to_utc(get_attr(t, "exit_timestamp", None))),
            "direction": normalize_direction(get_attr(t, "direction", "")),
            "outcome": get_attr(t, "outcome", None),
            "r_multiple": safe_float(get_attr(t, "r_multiple", None)),
            "pnl": safe_float(get_attr(t, "pnl", None)),
            "exit_reason": get_attr(t, "exit_reason", None),
        })
    df = pd.DataFrame(rows)
    out_csv = output_dir / "walk_forward_trades.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()