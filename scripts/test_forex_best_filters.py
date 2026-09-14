from __future__ import annotations

"""
Combined Filters Test — Find the Best Combination

Tests all combinations of:
    1. Symbol filter      (top 4 vs all 13)
    2. Direction filter   (bullish-only vs all)
    3. Zone filter        (FVG-only vs all)
    4. Session filter     (London+Late vs all)

For each combination, reports:
    - Trades, Win rate, Total R, PF, Max DD

Saves all results to:
    data/validation/10y/filter_combinations.csv

Usage:
    python scripts/test_forex_best_filters.py
"""

from pathlib import Path
import sys
import itertools
import json
from statistics import mean, median
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TRADES_CSV = DATA_DIR / "validation" / "10y" / "trades.csv"
OUTPUT_DIR = DATA_DIR / "validation" / "10y"
OUTPUT_CSV = OUTPUT_DIR / "filter_combinations.csv"

TRAIN_END = "2022-01-01T00:00:00+00:00"

# Top symbols identified from the 10-year backtest
TOP_SYMBOLS = ["XAUUSD", "GBPJPY", "USDJPY", "EURJPY"]

# Sessions: London (08-13 UTC) + Late (21-24 UTC)
BEST_SESSIONS = ["london", "late"]

# Confirmed best zone/direction from 10-year results
BEST_ZONE = "fvg"
BEST_DIRECTION = "bullish"


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


def session_of(ts):
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


def compute_metrics(df):
    """Compute all metrics for a filtered DataFrame."""
    closed = df[df["outcome"].isin(["win", "loss"])]
    n = len(closed)
    if n == 0:
        return None

    wins = (closed["outcome"] == "win").sum()
    losses = (closed["outcome"] == "loss").sum()
    total_r = closed["r_multiple"].sum()
    pnls = closed["pnl"].dropna()

    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    r_series = closed["r_multiple"].dropna()
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in r_series:
        cumulative += r
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    holds = closed["hold_hours"].dropna()

    return {
        "n": n,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": wins / n if n else 0.0,
        "total_r": float(total_r),
        "pf": pf,
        "avg_r": float(r_series.mean()) if len(r_series) else 0.0,
        "median_r": float(r_series.median()) if len(r_series) else 0.0,
        "max_dd_r": max_dd,
        "avg_hold_h": float(holds.mean()) if len(holds) else 0.0,
    }


def apply_filters(df, symbol_filter, direction_filter, zone_filter, session_filter):
    """Apply all four filters to the trades DataFrame."""
    out = df.copy()

    if symbol_filter is not None:
        out = out[out["symbol"].isin(symbol_filter)]

    if direction_filter is not None:
        out = out[out["direction"] == direction_filter]

    if zone_filter is not None:
        out = out[out["zone_type"] == zone_filter]

    if session_filter is not None:
        out["session"] = out["entry_timestamp"].apply(
            lambda x: session_of(to_utc(x))
        )
        out = out[out["session"].isin(session_filter)]

    return out


def split_train_test(df):
    """Split trades into TRAIN and TEST."""
    split_ts = to_utc(TRAIN_END)
    df = df.copy()
    df["_ts"] = df["entry_timestamp"].apply(to_utc)
    train = df[df["_ts"] < split_ts].drop(columns=["_ts"])
    test = df[df["_ts"] >= split_ts].drop(columns=["_ts"])
    return train, test


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("COMBINED FILTERS TEST — 10 YEAR FOREX")
    print("=" * 72)

    if not TRADES_CSV.exists():
        print(f"ERROR: {TRADES_CSV} not found.")
        print("Run: python scripts/test_multi_forex_10y.py first.")
        return

    df = pd.read_csv(TRADES_CSV)
    print(f"Loaded {len(df)} trades from {TRADES_CSV.name}")
    print()

    # ---- Define filter options ----
    filter_options = {
        "symbol": [
            ("all", None),
            ("top4", TOP_SYMBOLS),
        ],
        "direction": [
            ("all", None),
            ("bullish", "bullish"),
        ],
        "zone": [
            ("all", None),
            ("fvg", "fvg"),
        ],
        "session": [
            ("all", None),
            ("london+late", BEST_SESSIONS),
        ],
    }

    # ---- Generate all combinations ----
    keys = list(filter_options.keys())
    value_lists = [filter_options[k] for k in keys]

    combos = list(itertools.product(*value_lists))

    print(f"Testing {len(combos)} filter combinations...")
    print()

    results = []

    for combo in combos:
        # Build filter dict
        filters = {}
        labels = {}
        for k, (label, value) in zip(keys, combo):
            filters[k] = value
            labels[k] = label

        # Apply filters
        filtered = apply_filters(
            df,
            symbol_filter=filters["symbol"],
            direction_filter=filters["direction"],
            zone_filter=filters["zone"],
            session_filter=filters["session"],
        )

        # Split train/test
        train_df, test_df = split_train_test(filtered)

        full_m = compute_metrics(filtered)
        train_m = compute_metrics(train_df)
        test_m = compute_metrics(test_df)

        row = {
            "symbol_filter": labels["symbol"],
            "direction_filter": labels["direction"],
            "zone_filter": labels["zone"],
            "session_filter": labels["session"],
            "full_n": full_m["n"] if full_m else 0,
            "full_win_rate": full_m["win_rate"] if full_m else 0.0,
            "full_total_r": full_m["total_r"] if full_m else 0.0,
            "full_pf": full_m["pf"] if full_m else 0.0,
            "full_max_dd": full_m["max_dd_r"] if full_m else 0.0,
            "train_n": train_m["n"] if train_m else 0,
            "train_total_r": train_m["total_r"] if train_m else 0.0,
            "train_win_rate": train_m["win_rate"] if train_m else 0.0,
            "test_n": test_m["n"] if test_m else 0,
            "test_total_r": test_m["total_r"] if test_m else 0.0,
            "test_win_rate": test_m["win_rate"] if test_m else 0.0,
        }

        results.append(row)

    # ---- Print table ----
    print(f"{'Symbol':<7} {'Dir':<8} {'Zone':<6} {'Sess':<11} "
          f"{'N':>5} {'WR':>7} {'TotalR':>8} {'PF':>6} "
          f"{'TrainR':>8} {'TestR':>8}")
    print("-" * 90)

    for r in results:
        pf = f"{r['full_pf']:.2f}" if r['full_pf'] != float('inf') else "inf"
        print(
            f"{r['symbol_filter']:<7} "
            f"{r['direction_filter']:<8} "
            f"{r['zone_filter']:<6} "
            f"{r['session_filter']:<11} "
            f"{r['full_n']:>5} "
            f"{r['full_win_rate']*100:>6.2f}% "
            f"{r['full_total_r']:>+8.2f} "
            f"{pf:>6} "
            f"{r['train_total_r']:>+8.2f} "
            f"{r['test_total_r']:>+8.2f}"
        )

    # ---- Sort by full Total R ----
    print()
    print("=" * 72)
    print("TOP 10 COMBINATIONS (by Full Total R)")
    print("=" * 72)
    print()

    results_sorted = sorted(results, key=lambda x: -x["full_total_r"])

    print(f"{'#':>2} {'Symbol':<7} {'Dir':<8} {'Zone':<6} {'Sess':<11} "
          f"{'N':>5} {'WR':>7} {'TotalR':>8} {'PF':>6} "
          f"{'TrainR':>8} {'TestR':>8}")
    print("-" * 90)

    for i, r in enumerate(results_sorted[:10], 1):
        pf = f"{r['full_pf']:.2f}" if r['full_pf'] != float('inf') else "inf"
        print(
            f"{i:>2} "
            f"{r['symbol_filter']:<7} "
            f"{r['direction_filter']:<8} "
            f"{r['zone_filter']:<6} "
            f"{r['session_filter']:<11} "
            f"{r['full_n']:>5} "
            f"{r['full_win_rate']*100:>6.2f}% "
            f"{r['full_total_r']:>+8.2f} "
            f"{pf:>6} "
            f"{r['train_total_r']:>+8.2f} "
            f"{r['test_total_r']:>+8.2f}"
        )

    # ---- Sort by TEST Total R (most important) ----
    print()
    print("=" * 72)
    print("TOP 10 COMBINATIONS (by TEST Total R)")
    print("=" * 72)
    print()

    results_by_test = sorted(results, key=lambda x: -x["test_total_r"])

    print(f"{'#':>2} {'Symbol':<7} {'Dir':<8} {'Zone':<6} {'Sess':<11} "
          f"{'N':>5} {'WR':>7} {'TotalR':>8} {'PF':>6} "
          f"{'TrainR':>8} {'TestR':>8}")
    print("-" * 90)

    for i, r in enumerate(results_by_test[:10], 1):
        pf = f"{r['full_pf']:.2f}" if r['full_pf'] != float('inf') else "inf"
        print(
            f"{i:>2} "
            f"{r['symbol_filter']:<7} "
            f"{r['direction_filter']:<8} "
            f"{r['zone_filter']:<6} "
            f"{r['session_filter']:<11} "
            f"{r['full_n']:>5} "
            f"{r['full_win_rate']*100:>6.2f}% "
            f"{r['full_total_r']:>+8.2f} "
            f"{pf:>6} "
            f"{r['train_total_r']:>+8.2f} "
            f"{r['test_total_r']:>+8.2f}"
        )

    # ---- Best combo per criterion ----
    print()
    print("=" * 72)
    print("BEST COMBINATIONS BY CRITERION")
    print("=" * 72)

    best_full_r = max(results, key=lambda x: x["full_total_r"])
    best_test_r = max(results, key=lambda x: x["test_total_r"])
    best_pf = max(
        (r for r in results if r["full_n"] >= 50),
        key=lambda x: x["full_pf"],
    )
    best_wr = max(
        (r for r in results if r["full_n"] >= 50),
        key=lambda x: x["full_win_rate"],
    )

    def print_best(label, r):
        print(f"\n{label}:")
        print(f"  Filters:  {r['symbol_filter']} + {r['direction_filter']} "
              f"+ {r['zone_filter']} + {r['session_filter']}")
        print(f"  Trades:   {r['full_n']}")
        print(f"  Win rate: {r['full_win_rate']*100:.2f}%")
        print(f"  Total R:  {r['full_total_r']:+.2f}")
        print(f"  PF:       {r['full_pf']:.2f}")
        print(f"  Train R:  {r['train_total_r']:+.2f}  (n={r['train_n']})")
        print(f"  Test R:   {r['test_total_r']:+.2f}  (n={r['test_n']})")

    print_best("BEST FULL TOTAL R", best_full_r)
    print_best("BEST TEST TOTAL R", best_test_r)
    print_best("BEST PROFIT FACTOR (n>=50)", best_pf)
    print_best("BEST WIN RATE (n>=50)", best_wr)

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()