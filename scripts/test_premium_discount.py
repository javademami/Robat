"""
Premium / Discount Filter Test

Adds a Premium/Discount filter on top of the best combination
(RR=3.0, buffer=0.10, retest=72, micro=60, bullish-only, top4 symbols).

Premium/Discount logic (ICT):
    - Compute the equilibrium of the most recent swing (High, Low)
    - Bullish entries allowed ONLY in Discount (< 50% of swing)
    - Bearish entries allowed ONLY in Premium (> 50% of swing)

We test three strictness levels:
    - "off":       no filter (baseline)
    - "standard":  bullish only in Discount (< 0.50)
    - "strict":    bullish only in deep Discount (< 0.35)
    - "loose":     bullish only in Discount (< 0.60)

For each, reports:
    - Full 10-year
    - Per-year
    - Win rate, Total R, PF

Usage:
    python scripts/test_premium_discount.py
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

# Best params from multi-param test
BEST_RR = 3.0
BEST_BUFFER = 0.10
BEST_RETEST = 72
BEST_MICRO = 60

# Premium/Discount strictness levels
# For bullish: entry_price must be BELOW (level * swing_range) from swing high
# i.e. discount_factor = (swing_high - entry) / (swing_high - swing_low)
# must be >= 1 - level  (e.g. level=0.50 -> discount_factor >= 0.50)
PD_LEVELS = {
    "off":      None,     # no filter
    "loose":    0.40,     # entry must be in bottom 60% (discount >= 0.40)
    "standard": 0.50,     # entry must be in bottom 50% (discount >= 0.50)
    "strict":   0.65,     # entry must be in bottom 35% (discount >= 0.65)
}

# Swing lookback for PD calculation
SWING_LOOKBACK_BARS = 200   # ~8 days on 1H
SWING_PIVOT_WINDOW = 10     # pivot window for swing detection


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
# PREMIUM / DISCOUNT CALCULATION
# ============================================================

def compute_swing_range(df_1h_idx, lookback=SWING_LOOKBACK_BARS,
                        pivot_window=SWING_PIVOT_WINDOW):
    """
    Compute rolling swing high / low / equilibrium for each candle.

    Returns a DataFrame indexed by timestamp with columns:
        swing_high, swing_low, equilibrium
    """
    df = df_1h_idx.copy()

    # Rolling high/low
    roll_high = df["high"].rolling(window=lookback, min_periods=10).max()
    roll_low = df["low"].rolling(window=lookback, min_periods=10).min()

    swing_high = roll_high
    swing_low = roll_low
    equilibrium = (swing_high + swing_low) / 2.0

    out = pd.DataFrame({
        "swing_high": swing_high,
        "swing_low": swing_low,
        "equilibrium": equilibrium,
    })
    return out


def discount_factor(entry_price, swing_high, swing_low):
    """
    For a bullish entry: how deep in discount the entry is.

    discount = (swing_high - entry) / (swing_high - swing_low)

    - 1.0 = at swing low (deepest discount)
    - 0.5 = at equilibrium
    - 0.0 = at swing high (premium)
    """
    if swing_high is None or swing_low is None:
        return None
    rng = swing_high - swing_low
    if rng <= 0:
        return None
    return (swing_high - entry_price) / rng


# ============================================================
# PIPELINE
# ============================================================

def precompute_lower(symbol):
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

    # Entry prices
    entry_prices = {}
    for _, row in df_1h.iterrows():
        ts = timestamp_key(row.get("timestamp"))
        c = safe_float(row.get("close"))
        if ts and c is not None:
            entry_prices[ts] = c

    # Sweeps by MSS
    sweeps_by_mss = {}
    for sw in sweeps:
        sw_ts = get_attr(sw, "timestamp", "sweep_timestamp")
        if sw_ts is None:
            continue
        k = timestamp_key(sw_ts)
        if k:
            sweeps_by_mss[k] = sw

    atr_series = calculate_atr(df_1h_idx, period=ATR_PERIOD).sort_index()

    # --- Premium / Discount data ---
    pd_df = compute_swing_range(df_1h_idx)
    pd_df.index = df_1h_idx.index

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
        "pd_df": pd_df,
    }


def run_best(precomp, pd_level=None):
    """
    Run the best combination pipeline.
    If pd_level is not None, filter bullish entries to be in discount.
    """

    # Retests
    ob_r = detect_ob_retests(precomp["df_1h"], precomp["obs"],
                              max_bars_after_zone=BEST_RETEST)
    fvg_r = detect_fvg_retests(precomp["df_1h"], precomp["fvgs"],
                                max_bars_after_zone=BEST_RETEST)
    retests = list(ob_r) + list(fvg_r)

    if not retests:
        return []

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

    # --- Apply Premium/Discount filter ---
    if pd_level is not None:
        filtered = []
        pd_df = precomp["pd_df"]

        for c in tradable:
            entry_ts = to_utc(getattr(c, "timestamp", None))
            entry_price = safe_float(getattr(c, "entry_price", None))
            if entry_ts is None or entry_price is None:
                continue

            # Find swing high/low as-of this entry
            try:
                row = pd_df.asof(entry_ts)
            except Exception:
                continue

            sh = safe_float(row.get("swing_high"))
            sl = safe_float(row.get("swing_low"))

            df_factor = discount_factor(entry_price, sh, sl)
            if df_factor is None:
                continue

            # For bullish: entry must be in discount
            # df_factor >= pd_level
            if df_factor >= pd_level:
                filtered.append(c)

        tradable = filtered

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
    print("PREMIUM / DISCOUNT FILTER TEST")
    print("=" * 72)
    print(f"Symbols:  {SYMBOLS}")
    print(f"Direction: {DIRECTION_FILTER}")
    print(f"Best params: RR={BEST_RR}, buffer={BEST_BUFFER}, "
          f"retest={BEST_RETEST}, micro={BEST_MICRO}")
    print()
    print(f"PD levels to test: {list(PD_LEVELS.keys())}")
    print()

    # ---- Precompute ----
    print("Precomputing lower layers ...")
    precomputed = {}
    for sym in SYMBOLS:
        print(f"  {sym}:")
        p = precompute_lower(sym)
        if p:
            precomputed[sym] = p
            print(f"    candles: {len(p['df_1h']):,}")
    print()

    if not precomputed:
        print("No symbols precomputed. Aborting.")
        return

    # ---- Test each PD level ----
    results = {}

    for level_name, pd_level in PD_LEVELS.items():
        print(f"Testing PD level: {level_name} (pd_level={pd_level})")

        all_trades = []
        for sym, p in precomputed.items():
            trades = run_best(p, pd_level=pd_level)
            all_trades.extend(trades)

        if not all_trades:
            print(f"  → no trades")
            continue

        full_m = compute_metrics(all_trades)

        # TRAIN/TEST split
        split_ts = to_utc("2022-01-01T00:00:00+00:00")
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

        train_m = compute_metrics(train)
        test_m = compute_metrics(test)

        results[level_name] = {
            "full": full_m,
            "train": train_m,
            "test": test_m,
            "trades": all_trades,
        }

        print(
            f"  → n={full_m['n']}  WR={full_m['win_rate']*100:.2f}%  "
            f"R={full_m['total_r']:+.1f}  PF={full_m['pf']:.2f}  "
            f"| train={train_m['total_r'] if train_m else 0:+.1f}  "
            f"test={test_m['total_r'] if test_m else 0:+.1f}"
        )

    # ---- Report ----
    print()
    print("=" * 72)
    print("RESULTS BY PD LEVEL")
    print("=" * 72)
    print()

    header = (
        f"{'Level':<10} {'N':>6} {'WR':>8} {'TotalR':>9} "
        f"{'PF':>7} {'MaxDD':>7} {'TrainR':>9} {'TestR':>9}"
    )
    print(header)
    print("-" * len(header))

    for level_name in PD_LEVELS.keys():
        if level_name not in results:
            continue
        r = results[level_name]
        m = r["full"]
        tr = r["train"]
        te = r["test"]

        print(
            f"{level_name:<10} "
            f"{m['n']:>6} "
            f"{m['win_rate']*100:>7.2f}% "
            f"{m['total_r']:>+9.2f} "
            f"{m['pf']:>7.2f} "
            f"{m['max_dd_r']:>7.2f} "
            f"{tr['total_r'] if tr else 0:>+9.2f} "
            f"{te['total_r'] if te else 0:>+9.2f}"
        )

    # ---- Save ----
    output_dir = DATA_DIR / "validation" / "10y"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for level_name in PD_LEVELS.keys():
        if level_name not in results:
            continue
        r = results[level_name]
        summary_rows.append({
            "level": level_name,
            "pd_level": PD_LEVELS[level_name],
            "n": r["full"]["n"],
            "win_rate": r["full"]["win_rate"],
            "total_r": r["full"]["total_r"],
            "pf": r["full"]["pf"],
            "max_dd": r["full"]["max_dd_r"],
            "train_r": r["train"]["total_r"] if r["train"] else 0.0,
            "test_r": r["test"]["total_r"] if r["test"] else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = output_dir / "premium_discount_results.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()