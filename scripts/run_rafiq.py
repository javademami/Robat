"""
Rafiq - Multi-Symbol Runner
============================

Usage:
    python scripts/run_rafiq.py                    # BTC + ETH + SOL
    python scripts/run_rafiq.py --symbol ETHUSDT   # one symbol
    python scripts/run_rafiq.py --all              # all symbols
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR, YEAR
from src.data.storage import load_year
from src.data.aggregator import resample_ohlcv
from src.strategy.swing_detector import detect_pivots
from src.strategy.liquidity import (
    detect_equal_highs,
    detect_equal_lows,
    detect_previous_day_liquidity,
    detect_swing_liquidity,
    deduplicate_zones,
)
from src.strategy.sweep_detector import detect_sweeps
from src.strategy.mss_detector import detect_mss
from src.strategy.displacement_detector import detect_displacements
from src.strategy.order_block_detector import detect_order_blocks
from src.strategy.fvg_detector import detect_fvgs
from src.strategy.retest_detector import (
    detect_ob_retests,
    detect_fvg_retests,
)
from src.strategy.micro_mss_detector import (
    detect_micro_mss,
    deduplicate_micro_mss,
)
from src.strategy.entry_detector import (
    detect_entry_candidates,
    get_tradable_entries,
    deduplicate_entries,
    entries_to_dataframe,
    MIN_ENTRY_SCORE,
)


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


# ============================================================
# 4H BIAS
# ============================================================

def detect_4h_bias_at(swings_4h, timestamp) -> str:
    """
    Infer 4H bias from swings confirmed before `timestamp`.
    """

    ts = pd.Timestamp(timestamp)

    highs = []
    lows = []

    for swing in swings_4h:

        confirmed_at = getattr(swing, "confirmed_at", None)
        if confirmed_at is None:
            continue

        try:
            confirmed_ts = pd.Timestamp(confirmed_at)
        except Exception:
            continue

        if confirmed_ts > ts:
            continue

        kind = getattr(swing, "kind", None)
        price = getattr(swing, "price", None)

        if kind is None or price is None:
            continue

        kind = str(kind).lower()

        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        if kind in {"high", "swing_high", "sh"}:
            highs.append(price)
        elif kind in {"low", "swing_low", "sl"}:
            lows.append(price)

    if len(highs) < 2 or len(lows) < 2:
        return "neutral"

    last_high, prev_high = highs[-1], highs[-2]
    last_low, prev_low = lows[-1], lows[-2]

    if last_high > prev_high and last_low > prev_low:
        return "bullish"

    if last_high < prev_high and last_low < prev_low:
        return "bearish"

    return "neutral"


# ============================================================
# CONTEXT BUILDER (full)
# ============================================================

def build_full_context(
    df_5m_indexed,
    micro_events,
    mss_events,
    sweeps,
    swings_4h,
) -> dict[str, Any]:
    """
    Build context with:
        - entry_prices
        - bias_by_timestamp
        - sweeps_by_mss_timestamp
    """

    # 1. Entry prices
    entry_prices: dict[str, float] = {}
    for micro in micro_events:
        ts = getattr(micro, "timestamp", None)
        if ts is None:
            continue
        try:
            entry_prices[str(ts)] = float(
                df_5m_indexed.loc[ts, "close"]
            )
        except KeyError:
            continue

    # 2. Bias by timestamp
    bias_by_ts: dict[str, str] = {}
    for micro in micro_events:
        ts = getattr(micro, "timestamp", None)
        if ts is None:
            continue
        bias_by_ts[str(ts)] = detect_4h_bias_at(swings_4h, ts)

    # 3. Sweeps by MSS timestamp
    sweeps_by_mss_ts: dict[str, Any] = {}

    sweep_by_timestamp = {}
    for sweep in sweeps:
        ts = getattr(sweep, "timestamp", None)
        if ts is None:
            continue
        sweep_by_timestamp[str(ts)] = sweep

    for mss in mss_events:
        mss_ts = getattr(mss, "timestamp", None)
        if mss_ts is None:
            continue

        sweep_ts = getattr(mss, "sweep_timestamp", None)
        if sweep_ts is None:
            continue

        sweep = sweep_by_timestamp.get(str(sweep_ts))
        if sweep is not None:
            sweeps_by_mss_ts[str(mss_ts)] = sweep

    return {
        "entry_prices": entry_prices,
        "bias_by_timestamp": bias_by_ts,
        "sweeps_by_mss_timestamp": sweeps_by_mss_ts,
    }


# ============================================================
# DISCOVER
# ============================================================

def discover_symbols() -> list[str]:
    base = Path(DATA_DIR)
    if not base.exists():
        return []

    symbols = []
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        if item.name == "validation":
            continue
        if item.name.startswith("."):
            continue
        if "USDT" in item.name:
            symbols.append(item.name)
    return symbols


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(symbol: str) -> dict | None:

    print(f"\n{'=' * 72}")
    print(f"RUNNING PIPELINE FOR: {symbol}")
    print(f"{'=' * 72}")

    try:
        df_5m = load_year(DATA_DIR, symbol, "5m", YEAR)
    except Exception as exc:
        print(f"  ✗ could not load {symbol} 5m data: {exc}")
        return None

    if df_5m.empty:
        print(f"  ✗ no data for {symbol}")
        return None

    print(f"5M candles: {len(df_5m):,}")

    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")

    swings_4h = detect_pivots(
        df_4h, left_bars=2, right_bars=2, timeframe="4h"
    )

    swing_zones = detect_swing_liquidity(swings_4h)
    equal_highs = detect_equal_highs(
        df_1h, tolerance_atr=0.15, atr_period=14,
        lookback=100, pivot_window=3,
    )
    equal_lows = detect_equal_lows(
        df_1h, tolerance_atr=0.15, atr_period=14,
        lookback=100, pivot_window=3,
    )
    previous_day = detect_previous_day_liquidity(df_1h)
    zones = deduplicate_zones(
        swing_zones + equal_highs + equal_lows + previous_day
    )

    sweeps = detect_sweeps(df_1h, zones, atr_period=14)

    internal_swings_1h = detect_pivots(
        df_1h, left_bars=2, right_bars=2, timeframe="1h"
    )

    mss_events = detect_mss(
        df_1h, sweeps, internal_swings_1h,
        max_bars_after_sweep=12,
    )

    displacement_events = detect_displacements(
        df_1h, mss_events,
        atr_period=14, min_body_atr=1.0,
        bullish_clv=0.70, bearish_clv=0.30,
        max_bars_after_mss=8,
    )

    ob_events = detect_order_blocks(
        df_1h, displacement_events, lookback=8,
    )
    fvg_events = detect_fvgs(
        df_1h, displacement_events,
        search_bars=3, atr_period=14,
    )

    ob_retests = detect_ob_retests(
        df_1h, ob_events, max_bars_after_zone=24,
    )
    fvg_retests = detect_fvg_retests(
        df_1h, fvg_events, max_bars_after_zone=24,
    )
    all_retests = list(ob_retests) + list(fvg_retests)

    internal_swings_5m = detect_pivots(
        df_5m, left_bars=2, right_bars=2, timeframe="5m",
    )

    df_5m_indexed = df_5m.copy()
    df_5m_indexed["timestamp"] = pd.to_datetime(
        df_5m_indexed["timestamp"], utc=True
    )
    df_5m_indexed = (
        df_5m_indexed.set_index("timestamp").sort_index()
    )

    micro_events = detect_micro_mss(
        df_5m_indexed, all_retests, internal_swings_5m,
        max_bars_after_retest=24,
        min_reference_distance_atr=0.25,
    )
    micro_events = deduplicate_micro_mss(micro_events)

    print(f"Micro-MSS: {len(micro_events):,}")

    # --------------------------------------------------------
    # FULL CONTEXT (bias + sweeps + entry_prices)
    # --------------------------------------------------------
    context = build_full_context(
        df_5m_indexed=df_5m_indexed,
        micro_events=micro_events,
        mss_events=mss_events,
        sweeps=sweeps,
        swings_4h=swings_4h,
    )

    bias_dist = Counter(context["bias_by_timestamp"].values())
    print(f"Bias distribution: {dict(bias_dist)}")
    print(f"Sweeps linked    : {len(context['sweeps_by_mss_timestamp'])}")

    candidates = detect_entry_candidates(
        micro_mss_events=micro_events,
        retests=all_retests,
        mss_events=mss_events,
        displacement_events=displacement_events,
        ob_events=ob_events,
        fvg_events=fvg_events,
        context=context,
        min_score=MIN_ENTRY_SCORE,
    )

    candidates = deduplicate_entries(candidates)
    tradable = get_tradable_entries(
        candidates, min_score=MIN_ENTRY_SCORE,
    )

    out_dir = Path(DATA_DIR) / "validation" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    if candidates:
        entries_to_dataframe(candidates).to_csv(
            out_dir / "entry_candidates.csv", index=False,
        )

    if tradable:
        entries_to_dataframe(tradable).to_csv(
            out_dir / "entry_candidates_tradable.csv",
            index=False,
        )

    scores = [c.score for c in candidates] if candidates else []

    direction_counts = Counter(
        c.direction for c in candidates
    ) if candidates else Counter()

    zone_counts = Counter(
        c.zone_type for c in candidates
    ) if candidates else Counter()

    summary = {
        "symbol": symbol,
        "micro_mss": len(micro_events),
        "candidates": len(candidates),
        "tradable": len(tradable),
        "min_score": round(min(scores), 2) if scores else 0,
        "max_score": round(max(scores), 2) if scores else 0,
        "mean_score": (
            round(sum(scores) / len(scores), 2) if scores else 0
        ),
        "bullish": direction_counts.get("bullish", 0),
        "bearish": direction_counts.get("bearish", 0),
        "ob": zone_counts.get("ob", 0),
        "fvg": zone_counts.get("fvg", 0),
    }

    print(f"\nSummary [{symbol}]:")
    for key, value in summary.items():
        print(f"  {key:<12}: {value}")

    return summary


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rafiq multi-symbol runner",
    )
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def resolve_symbols(args) -> list[str]:
    if args.symbol:
        return [args.symbol]
    if args.all:
        return discover_symbols()
    return DEFAULT_SYMBOLS


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    symbols = resolve_symbols(args)

    print("=" * 72)
    print(f"RAFiQ — running pipeline for {len(symbols)} symbol(s)")
    print("=" * 72)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Year   : {YEAR}")

    summaries = []

    for symbol in symbols:
        try:
            summary = run_pipeline(symbol)
        except Exception as exc:
            print(f"\n✗ pipeline failed for {symbol}: {exc}")
            continue

        if summary is not None:
            summaries.append(summary)

    if summaries:
        comparison = pd.DataFrame(summaries)
        comparison_path = (
            Path(DATA_DIR) / "validation" / "comparison.csv"
        )
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)

        print("\n" + "=" * 72)
        print("COMPARISON")
        print("=" * 72)
        print(comparison.to_string(index=False))
        print(f"\nSaved: {comparison_path}")
    else:
        print("\nNo symbols processed successfully.")


if __name__ == "__main__":
    main()