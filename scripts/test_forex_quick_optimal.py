"""
Quick Optimal Test — Only RR variations.

Precomputes the pipeline once per symbol, then only varies
the RR-dependent parts (risk manager + position sizer + backtest).

Runs in ~2-3 minutes instead of hours.

Usage:
    python scripts/test_forex_quick_optimal.py
"""

from pathlib import Path
import sys
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
BACKTEST_MAX_BARS = 5_000
BUFFER_ATR = 0.10

DIRECTION_FILTER = "bullish"
TRAIN_END = "2022-01-01T00:00:00+00:00"

RR_OPTIONS = [1.5, 2.0, 2.5, 3.0]


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
        "n": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": wins / n,
        "total_r": sum(r_values),
        "pf": pf,
        "avg_r": mean(r_values),
        "max_dd_r": max_dd,
    }


# ============================================================
# PRECOMPUTE PIPELINE PER SYMBOL
# ============================================================

def precompute(symbol):
    """
    Run all parameter-independent pipeline stages once.
    Returns a dict with the precomputed objects.
    """
    data_file = DATA_DIR / "raw" / symbol / f"{symbol}_1h_mt5.parquet"
    if not data_file.exists():
        print(f"    ERROR: {data_file} not found")
        return None

    df_1h = pd.read_parquet(data_file)
    df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], utc=True)
    df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)

    print(f"    candles: {len(df_1h):,}")

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

    ob_r = detect_ob_retests(df_1h, obs, max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE)
    fvg_r = detect_fvg_retests(df_1h, fvgs, max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE)
    retests = list(ob_r) + list(fvg_r)

    swings_micro = detect_pivots(
        df_1h, left_bars=MICRO_PIVOT_WINDOW_1H,
        right_bars=MICRO_PIVOT_WINDOW_1H, timeframe="1h",
    )

    df_1h_idx = df_1h.copy()
    df_1h_idx["timestamp"] = pd.to_datetime(df_1h_idx["timestamp"], utc=True)
    df_1h_idx = df_1h_idx.set_index("timestamp").sort_index()

    micro = detect_micro_mss(
        df_1h_idx, retests, swings_micro,
        max_bars_after_retest=MICRO_MSS_MAX_BARS_AFTER_RETEST,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )
    micro = deduplicate_micro_mss(micro)

    print(f"    micro-MSS: {len(micro)}")

    # ---- Context ----
    context = {}
    entry_prices = {}
    for _, row in df_1h.iterrows():
        ts = timestamp_key(row.get("timestamp"))
        c = safe_float(row.get("close"))
        if ts and c is not None:
            entry_prices[ts] = c
    context["entry_prices"] = entry_prices

    sweeps_by_mss = {}
    for sw in sweeps:
        sw_ts = get_attr(sw, "timestamp", "sweep_timestamp")
        if sw_ts is None:
            continue
        k = timestamp_key(sw_ts)
        if k:
            sweeps_by_mss[k] = sw
    context["sweeps_by_mss_timestamp"] = sweeps_by_mss

    bias_map = {}
    sw4_sorted = sorted(swings_4h, key=lambda s: str(getattr(s, "timestamp", "")))
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
    context["bias_by_timestamp"] = bias_map

    # ---- ATR series ----
    atr_series = calculate_atr(df_1h_idx, period=ATR_PERIOD).sort_index()

    return {
        "symbol": symbol,
        "df_1h": df_1h,
        "df_1h_idx": df_1h_idx,
        "micro": micro,
        "retests": retests,
        "mss_events": mss_events,
        "disps": disps,
        "obs": obs,
        "fvgs": fvgs,
        "context": context,
        "atr_series": atr_series,
        "swings_4h": swings_4h,
    }


# ============================================================
# RUN RR VARIATION (fast part)
# ============================================================

def run_rr(precomputed, target_rr):
    """Only the RR-dependent part: entry + risk + size + backtest."""

    context = dict(precomputed["context"])
    context["risk_reward"] = target_rr

    candidates = detect_entry_candidates(
        precomputed["micro"],
        precomputed["retests"],
        precomputed["mss_events"],
        precomputed["disps"],
        precomputed["obs"],
        precomputed["fvgs"],
        context=context,
        debug=False,
    )

    tradable = [
        c for c in candidates
        if getattr(c, "tradable", False)
        and normalize_direction(getattr(c, "direction", "")) == DIRECTION_FILTER
    ]

    if not tradable:
        return []

    atr_series = precomputed["atr_series"]

    def _atr_as_of(value):
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            r = atr_series.asof(ts)
        except Exception:
            return None
        if r is None or pd.isna(r):
            return None
        return float(r)

    atr_map = {}
    for c in tradable:
        c_ts = getattr(c, "timestamp", None)
        if c_ts is None:
            continue
        v = _atr_as_of(c_ts)
        if v is not None:
            atr_map[c_ts] = v

    zones = list(precomputed["obs"]) + list(precomputed["fvgs"])

    plans = build_risk_plans(
        tradable, zones, atr_map,
        target_rr=target_rr,
        buffer_atr=BUFFER_ATR,
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
        valid_sizes,
        precomputed["df_1h"],
        max_bars=BACKTEST_MAX_BARS,
    )

    for t in trades:
        try:
            t.symbol = precomputed["symbol"]
        except Exception:
            pass

    return trades


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("QUICK OPTIMAL TEST — RR variations only")
    print("=" * 72)

    print(f"Symbols: {SYMBOLS}")
    print(f"RR options: {RR_OPTIONS}")
    print(f"Direction: {DIRECTION_FILTER}")
    print()

    # ---- Precompute per symbol ----
    print("Precomputing pipeline per symbol ...")
    precomputed_list = []
    for sym in SYMBOLS:
        print(f"  {sym}:")
        p = precompute(sym)
        if p:
            precomputed_list.append(p)
    print(f"\nPrecomputed {len(precomputed_list)} symbols.")
    print()

    # ---- Test each RR ----
    results = []
    for rr in RR_OPTIONS:
        print(f"Testing RR={rr} ...")
        all_trades = []
        for p in precomputed_list:
            trades = run_rr(p, rr)
            all_trades.extend(trades)

        if not all_trades:
            print(f"  → no trades")
            continue

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

        results.append({
            "rr": rr,
            "full_n": full_m["n"],
            "full_wr": full_m["win_rate"],
            "full_total_r": full_m["total_r"],
            "full_pf": full_m["pf"],
            "full_max_dd": full_m["max_dd_r"],
            "train_n": train_m["n"] if train_m else 0,
            "train_total_r": train_m["total_r"] if train_m else 0.0,
            "train_wr": train_m["win_rate"] if train_m else 0.0,
            "test_n": test_m["n"] if test_m else 0,
            "test_total_r": test_m["total_r"] if test_m else 0.0,
            "test_wr": test_m["win_rate"] if test_m else 0.0,
        })

        print(
            f"  → n={full_m['n']}  WR={full_m['win_rate']*100:.1f}%  "
            f"R={full_m['total_r']:+.1f}  PF={full_m['pf']:.2f}  "
            f"| train={train_m['total_r'] if train_m else 0:+.1f}  "
            f"test={test_m['total_r'] if test_m else 0:+.1f}"
        )

    # ---- Report ----
    print()
    print("=" * 72)
    print("RESULTS BY RR")
    print("=" * 72)
    print()
    header = (
        f"{'RR':>5} {'N':>6} {'WR':>8} {'TotalR':>9} "
        f"{'PF':>7} {'MaxDD':>7} {'TrainR':>9} {'TestR':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['rr']:>5.1f} "
            f"{r['full_n']:>6} "
            f"{r['full_wr']*100:>7.2f}% "
            f"{r['full_total_r']:>+9.2f} "
            f"{r['full_pf']:>7.2f} "
            f"{r['full_max_dd']:>7.2f} "
            f"{r['train_total_r']:>+9.2f} "
            f"{r['test_total_r']:>+9.2f}"
        )

    if results:
        best = max(results, key=lambda x: x["full_total_r"])
        print()
        print(f"BEST RR: {best['rr']}")
        print(f"  Total R:  {best['full_total_r']:+.2f}")
        print(f"  PF:       {best['full_pf']:.2f}")
        print(f"  Win rate: {best['full_wr']*100:.2f}%")
        print(f"  Train:    {best['train_total_r']:+.2f}")
        print(f"  Test:     {best['test_total_r']:+.2f}")

    # Save
    output_csv = DATA_DIR / "validation" / "10y" / "quick_rr_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()