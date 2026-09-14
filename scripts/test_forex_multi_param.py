"""
Multi-Parameter Optimization — Buffer, Retest, Micro-MSS

Precomputes the pipeline once per symbol, then varies:
    - Buffer ATR:     0.05, 0.10, 0.15, 0.20
    - Retest window:  24, 36, 48, 72
    - Micro-MSS window: 48, 60, 72, 96
    - RR:             2.5, 3.0

But since Buffer, Retest, and Micro-MSS affect the pipeline itself
(not just the risk manager), we cannot fully precompute them.

Optimization: precompute the shared lower layers (structure, liquidity,
sweeps, MSS, displacement, OB, FVG) once per symbol, then only
re-run retest, micro-MSS, entry, risk, size, backtest for each combo.

Usage:
    python scripts/test_forex_multi_param.py
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

DIRECTION_FILTER = "bullish"
TRAIN_END = "2022-01-01T00:00:00+00:00"

# Parameter grids to test
RR_OPTIONS = [2.5, 3.0]
BUFFER_OPTIONS = [0.05, 0.10, 0.15, 0.20]
RETEST_OPTIONS = [24, 36, 48, 72]
MICRO_MSS_OPTIONS = [48, 60, 72, 96]


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
        "avg_r": mean(r_values),
        "max_dd_r": max_dd,
    }


# ============================================================
# PRECOMPUTE (lower layers only)
# ============================================================

def precompute_lower(symbol):
    """
    Run all parameter-independent pipeline stages once:
    - 4H structure, liquidity
    - sweeps, MSS, displacement
    - OB, FVG
    Returns a dict, or None on error.
    """
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

    # Entry prices map
    entry_prices = {}
    for _, row in df_1h.iterrows():
        ts = timestamp_key(row.get("timestamp"))
        c = safe_float(row.get("close"))
        if ts and c is not None:
            entry_prices[ts] = c

    # Sweeps by MSS timestamp
    sweeps_by_mss = {}
    for sw in sweeps:
        sw_ts = get_attr(sw, "timestamp", "sweep_timestamp")
        if sw_ts is None:
            continue
        k = timestamp_key(sw_ts)
        if k:
            sweeps_by_mss[k] = sw

    # ATR series
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


def atr_as_of(atr_series, value):
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


# ============================================================
# RUN COMBO (upper layers only)
# ============================================================

def run_combo(precomp, target_rr, buffer_atr, retest_window, micro_window):
    """
    Only the parameter-dependent part: retest, micro-MSS, entry,
    risk, size, backtest.
    """
    df_1h = precomp["df_1h"]
    df_1h_idx = precomp["df_1h_idx"]
    swings_micro = precomp["swings_1h"]  # reuse 1H swings for micro

    # ---- Retests ----
    ob_r = detect_ob_retests(
        df_1h, precomp["obs"], max_bars_after_zone=retest_window,
    )
    fvg_r = detect_fvg_retests(
        df_1h, precomp["fvgs"], max_bars_after_zone=retest_window,
    )
    retests = list(ob_r) + list(fvg_r)

    if not retests:
        return []

    # ---- Micro-MSS ----
    micro = detect_micro_mss(
        df_1h_idx, retests, swings_micro,
        max_bars_after_retest=micro_window,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )
    micro = deduplicate_micro_mss(micro)

    if not micro:
        return []

    # ---- Context ----
    context = {
        "entry_prices": precomp["entry_prices"],
        "sweeps_by_mss_timestamp": precomp["sweeps_by_mss"],
        "risk_reward": target_rr,
    }

    # bias_by_timestamp
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
    context["bias_by_timestamp"] = bias_map

    # ---- Entry Detector ----
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

    # ---- ATR map ----
    atr_map = {}
    for c in tradable:
        c_ts = getattr(c, "timestamp", None)
        if c_ts is None:
            continue
        v = atr_as_of(precomp["atr_series"], c_ts)
        if v is not None:
            atr_map[c_ts] = v

    # ---- Risk Manager ----
    zones = list(precomp["obs"]) + list(precomp["fvgs"])
    plans = build_risk_plans(
        tradable, zones, atr_map,
        target_rr=target_rr,
        buffer_atr=buffer_atr,
    )
    valid_plans = get_valid_risk_plans(plans)
    if not valid_plans:
        return []

    # ---- Position Sizer ----
    sizes = build_position_sizes(
        valid_plans,
        account_equity=ACCOUNT_EQUITY,
        risk_percent=RISK_PERCENT,
        leverage=LEVERAGE,
    )
    valid_sizes = get_valid_position_sizes(sizes)
    if not valid_sizes:
        return []

    # ---- Backtest ----
    trades = run_backtest(
        valid_sizes, df_1h, max_bars=BACKTEST_MAX_BARS,
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
    print("MULTI-PARAMETER OPTIMIZATION")
    print("=" * 72)
    print(f"Symbols:      {SYMBOLS}")
    print(f"Direction:    {DIRECTION_FILTER}")
    print(f"RR:           {RR_OPTIONS}")
    print(f"Buffer ATR:   {BUFFER_OPTIONS}")
    print(f"Retest:       {RETEST_OPTIONS}")
    print(f"Micro-MSS:    {MICRO_MSS_OPTIONS}")

    total_combos = (
        len(RR_OPTIONS) * len(BUFFER_OPTIONS)
        * len(RETEST_OPTIONS) * len(MICRO_MSS_OPTIONS)
    )
    print(f"Total combos: {total_combos}")
    print()

    # ---- Precompute lower layers per symbol ----
    print("Precomputing lower layers per symbol ...")
    precomputed = {}
    for sym in SYMBOLS:
        print(f"  {sym}:")
        p = precompute_lower(sym)
        if p:
            precomputed[sym] = p
            print(f"    candles: {len(p['df_1h']):,}  "
                  f"OBs: {len(p['obs'])}  FVGs: {len(p['fvgs'])}  "
                  f"MSS: {len(p['mss_events'])}")
    print()

    if not precomputed:
        print("No symbols precomputed. Aborting.")
        return

    # ---- Test all combos ----
    results = []
    combo_num = 0

    for rr, buf, retest, micro_win in itertools.product(
        RR_OPTIONS, BUFFER_OPTIONS, RETEST_OPTIONS, MICRO_MSS_OPTIONS,
    ):
        combo_num += 1
        print(f"[{combo_num}/{total_combos}] "
              f"RR={rr} buf={buf} retest={retest} micro={micro_win}")

        all_trades = []
        for sym, p in precomputed.items():
            trades = run_combo(p, rr, buf, retest, micro_win)
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
            "buffer_atr": buf,
            "retest_window": retest,
            "micro_mss_window": micro_win,
            "full_n": full_m["n"],
            "full_wr": full_m["win_rate"],
            "full_total_r": full_m["total_r"],
            "full_pf": full_m["pf"],
            "full_max_dd": full_m["max_dd_r"],
            "train_n": train_m["n"] if train_m else 0,
            "train_total_r": train_m["total_r"] if train_m else 0.0,
            "test_n": test_m["n"] if test_m else 0,
            "test_total_r": test_m["total_r"] if test_m else 0.0,
        })

        print(
            f"  → n={full_m['n']}  WR={full_m['win_rate']*100:.1f}%  "
            f"R={full_m['total_r']:+.1f}  PF={full_m['pf']:.2f}  "
            f"| train={train_m['total_r'] if train_m else 0:+.1f}  "
            f"test={test_m['total_r'] if test_m else 0:+.1f}"
        )

    # ---- Sort by Full Total R ----
    results.sort(key=lambda x: -x["full_total_r"])

    print()
    print("=" * 72)
    print("TOP 15 COMBINATIONS (by Full Total R)")
    print("=" * 72)
    print()

    header = (
        f"{'#':>2} {'RR':>4} {'Buf':>5} {'Ret':>4} {'Mic':>4} "
        f"{'N':>5} {'WR':>7} {'TotalR':>8} {'PF':>6} "
        f"{'MaxDD':>6} {'TrainR':>8} {'TestR':>8}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results[:15], 1):
        print(
            f"{i:>2} "
            f"{r['rr']:>4.1f} "
            f"{r['buffer_atr']:>5.2f} "
            f"{r['retest_window']:>4} "
            f"{r['micro_mss_window']:>4} "
            f"{r['full_n']:>5} "
            f"{r['full_wr']*100:>6.2f}% "
            f"{r['full_total_r']:>+8.2f} "
            f"{r['full_pf']:>6.2f} "
            f"{r['full_max_dd']:>6.2f} "
            f"{r['train_total_r']:>+8.2f} "
            f"{r['test_total_r']:>+8.2f}"
        )

    # ---- Save ----
    output_csv = DATA_DIR / "validation" / "10y" / "multi_param_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    if results:
        best = results[0]
        print()
        print("=" * 72)
        print("BEST COMBINATION")
        print("=" * 72)
        print(f"  RR:               {best['rr']}")
        print(f"  Buffer ATR:       {best['buffer_atr']}")
        print(f"  Retest window:    {best['retest_window']}")
        print(f"  Micro-MSS window: {best['micro_mss_window']}")
        print(f"  Trades:           {best['full_n']}")
        print(f"  Win rate:         {best['full_wr']*100:.2f}%")
        print(f"  Total R:          {best['full_total_r']:+.2f}")
        print(f"  PF:               {best['full_pf']:.2f}")
        print(f"  Max DD:           {best['full_max_dd']:.2f}")
        print(f"  TRAIN R:          {best['train_total_r']:+.2f}  (n={best['train_n']})")
        print(f"  TEST R:           {best['test_total_r']:+.2f}  (n={best['test_n']})")


if __name__ == "__main__":
    main()