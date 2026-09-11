from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR, SYMBOL, YEAR
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
from src.strategy.retest_detector import detect_ob_retests, detect_fvg_retests
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


OUTPUT_ROOT = Path("data") / "validation"
OUTPUT_ENTRIES_CSV = OUTPUT_ROOT / "entry_candidates.csv"
OUTPUT_TRADABLE_CSV = OUTPUT_ROOT / "entry_candidates_tradable.csv"


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline():
    """
    Run the full pipeline and return every stage needed by
    the Entry Detector.

    Returns:
        df_5m_indexed
        all_retests
        micro_events
        mss_events
        displacement_events
        ob_events
        fvg_events
        sweeps
        swings_4h
    """

    print("=" * 72)
    print("BUILDING ENTRY DETECTOR TEST DATASET")
    print("=" * 72)

    print("\n[1] Loading 5M...")
    df_5m = load_year(DATA_DIR, SYMBOL, "5m", YEAR)
    print(f"5M candles: {len(df_5m):,}")

    print("\n[2] Resampling...")
    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")
    print(f"1H candles: {len(df_1h):,}")
    print(f"4H candles: {len(df_4h):,}")

    print("\n[3] 4H swings...")
    swings_4h = detect_pivots(
        df_4h, left_bars=2, right_bars=2, timeframe="4h"
    )
    print(f"4H swings: {len(swings_4h):,}")

    print("\n[4] Liquidity...")
    swing_zones = detect_swing_liquidity(swings_4h)
    equal_highs = detect_equal_highs(
        df_1h,
        tolerance_atr=0.15,
        atr_period=14,
        lookback=100,
        pivot_window=3,
    )
    equal_lows = detect_equal_lows(
        df_1h,
        tolerance_atr=0.15,
        atr_period=14,
        lookback=100,
        pivot_window=3,
    )
    previous_day = detect_previous_day_liquidity(df_1h)
    zones = deduplicate_zones(
        swing_zones + equal_highs + equal_lows + previous_day
    )
    print(f"Liquidity zones: {len(zones):,}")

    print("\n[5] Sweeps...")
    sweeps = detect_sweeps(df_1h, zones, atr_period=14)
    print(f"Sweeps: {len(sweeps):,}")

    print("\n[6] 1H internal swings...")
    internal_swings_1h = detect_pivots(
        df_1h, left_bars=2, right_bars=2, timeframe="1h"
    )
    print(f"1H internal swings: {len(internal_swings_1h):,}")

    print("\n[7] MSS...")
    mss_events = detect_mss(
        df_1h,
        sweeps,
        internal_swings_1h,
        max_bars_after_sweep=12,
    )
    print(f"MSS: {len(mss_events):,}")

    print("\n[8] Displacement...")
    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=14,
        min_body_atr=1.0,
        bullish_clv=0.70,
        bearish_clv=0.30,
        max_bars_after_mss=8,
    )
    print(f"Displacements: {len(displacement_events):,}")

    print("\n[9] Order Blocks...")
    ob_events = detect_order_blocks(
        df_1h,
        displacement_events,
        lookback=8,
    )
    print(f"Order Blocks: {len(ob_events):,}")

    print("\n[10] FVG...")
    fvg_events = detect_fvgs(
        df_1h,
        displacement_events,
        search_bars=3,
        atr_period=14,
    )
    print(f"FVG: {len(fvg_events):,}")

    print("\n[11] Retests...")
    ob_retests = detect_ob_retests(
        df_1h, ob_events, max_bars_after_zone=24
    )
    fvg_retests = detect_fvg_retests(
        df_1h, fvg_events, max_bars_after_zone=24
    )
    all_retests = list(ob_retests) + list(fvg_retests)
    print(f"OB retests: {len(ob_retests):,}")
    print(f"FVG retests: {len(fvg_retests):,}")
    print(f"Total retests: {len(all_retests):,}")

    print("\n[12] 5M swings...")
    internal_swings_5m = detect_pivots(
        df_5m,
        left_bars=2,
        right_bars=2,
        timeframe="5m",
    )
    print(f"5M internal swings: {len(internal_swings_5m):,}")

    df_5m_indexed = df_5m.copy()
    df_5m_indexed["timestamp"] = pd.to_datetime(
        df_5m_indexed["timestamp"], utc=True
    )
    df_5m_indexed = (
        df_5m_indexed
        .set_index("timestamp")
        .sort_index()
    )

    print("\n[13] Micro-MSS...")
    micro_events = detect_micro_mss(
        df_5m_indexed,
        all_retests,
        internal_swings_5m,
        max_bars_after_retest=24,
        min_reference_distance_atr=0.25,
    )
    micro_events = deduplicate_micro_mss(micro_events)
    print(f"Micro-MSS: {len(micro_events):,}")

    return (
        df_5m_indexed,
        all_retests,
        micro_events,
        mss_events,
        displacement_events,
        ob_events,
        fvg_events,
        sweeps,
        swings_4h,
    )


# ============================================================
# 4H BIAS DETECTION
# ============================================================

def detect_4h_bias_at(swings_4h, timestamp) -> str:
    """
    Infer 4H bias using only swings confirmed before `timestamp`.

    Rules:
        Bullish: last high > prev high AND last low > prev low
        Bearish: last high < prev high AND last low < prev low
        Otherwise: neutral
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
# CONTEXT BUILDER
# ============================================================

def build_entry_context(
    df_5m_indexed,
    micro_events,
    mss_events,
    sweeps,
    swings_4h,
) -> dict[str, Any]:
    """
    Build a rich context dict for Entry Detector.

    Supplies:
        - entry_prices             : per-micro close price (5M)
        - bias_by_timestamp        : per-micro 4H bias
        - sweeps_by_mss_timestamp  : MSS timestamp -> sweep event

    risk_reward is intentionally omitted (V1 design).
    """

    # --------------------------------------------------------
    # 1. Entry prices
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 2. Bias by micro timestamp
    # --------------------------------------------------------
    bias_by_ts: dict[str, str] = {}

    for micro in micro_events:
        ts = getattr(micro, "timestamp", None)
        if ts is None:
            continue
        bias_by_ts[str(ts)] = detect_4h_bias_at(swings_4h, ts)

    # --------------------------------------------------------
    # 3. Sweeps by MSS timestamp
    # --------------------------------------------------------
    sweeps_by_mss_ts: dict[str, Any] = {}

    # Build sweep map: timestamp -> sweep
    sweep_by_timestamp: dict[str, Any] = {}

    for sweep in sweeps:
        ts = getattr(sweep, "timestamp", None)
        if ts is None:
            continue
        sweep_by_timestamp[str(ts)] = sweep

    # For each MSS, find its sweep via sweep_timestamp
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
# REPORT HELPERS
# ============================================================

def print_score_buckets(df: pd.DataFrame) -> None:
    print("\nScore buckets:")

    if df.empty or "score" not in df.columns:
        print("  (no data)")
        return

    buckets = [
        (0, 50, "< 50"),
        (50, 60, "50-59"),
        (60, 70, "60-69"),
        (70, 80, "70-79"),
        (80, 90, "80-89"),
        (90, 101, "90-100"),
    ]

    for low, high, label in buckets:
        count = int(
            ((df["score"] >= low) & (df["score"] < high)).sum()
        )
        print(f"  {label:>8} : {count:,}")


def print_score_breakdown(df: pd.DataFrame) -> None:
    print("\nScore breakdown (mean per component):")

    if df.empty:
        print("  (no data)")
        return

    component_columns = [
        "score_structure",
        "score_liquidity_sweep",
        "score_mss",
        "score_displacement",
        "score_zone_quality",
        "score_retest_quality",
        "score_micro_mss",
        "score_risk_reward",
    ]

    for column in component_columns:
        if column not in df.columns:
            continue

        mean_value = float(df[column].mean())
        label = column.replace("score_", "")
        print(f"  {label:<20} : {mean_value:.2f}")


# ============================================================
# MAIN
# ============================================================

def main():

    (
        df_5m_indexed,
        all_retests,
        micro_events,
        mss_events,
        displacement_events,
        ob_events,
        fvg_events,
        sweeps,
        swings_4h,
    ) = build_pipeline()

    print("\n[14] Building entry context...")
    context = build_entry_context(
        df_5m_indexed=df_5m_indexed,
        micro_events=micro_events,
        mss_events=mss_events,
        sweeps=sweeps,
        swings_4h=swings_4h,
    )

    print(f"  Entry prices       : {len(context['entry_prices'])}")
    print(f"  Bias samples       : {len(context['bias_by_timestamp'])}")
    print(f"  Sweeps linked      : {len(context['sweeps_by_mss_timestamp'])}")

    bias_dist = Counter(context["bias_by_timestamp"].values())
    print(f"  Bias distribution  : {dict(bias_dist)}")

    print("\n[15] Running Entry Detector...")
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
    print(f"Entry candidates: {len(candidates):,}")

    tradable = get_tradable_entries(
        candidates,
        min_score=MIN_ENTRY_SCORE,
    )
    print(f"Tradable (score >= {MIN_ENTRY_SCORE:.0f}): {len(tradable):,}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    df_all = entries_to_dataframe(candidates)
    df_all.to_csv(OUTPUT_ENTRIES_CSV, index=False)
    print(f"\nSaved: {OUTPUT_ENTRIES_CSV}")

    df_tradable = entries_to_dataframe(tradable)
    df_tradable.to_csv(OUTPUT_TRADABLE_CSV, index=False)
    print(f"Saved: {OUTPUT_TRADABLE_CSV}")

    print("\n" + "=" * 72)
    print("ENTRY DETECTOR SUMMARY")
    print("=" * 72)
    print(f"Micro-MSS events       : {len(micro_events):,}")
    print(f"Entry candidates       : {len(candidates):,}")
    print(f"Tradable               : {len(tradable):,}")

    if candidates:

        scores = [c.score for c in candidates]
        print(f"Min score              : {min(scores):.2f}")
        print(f"Max score              : {max(scores):.2f}")
        print(f"Mean score             : {sum(scores) / len(scores):.2f}")

        grade_counts = Counter(c.grade for c in candidates)
        print("\nGrade distribution:")
        for grade in ["A+", "A", "B", "C", "D"]:
            print(f"  {grade:>3} : {grade_counts.get(grade, 0):,}")

        direction_counts = Counter(c.direction for c in candidates)
        print("\nDirection distribution:")
        for key, value in sorted(direction_counts.items()):
            print(f"  {key:<10} : {value:,}")

        zone_counts = Counter(c.zone_type for c in candidates)
        print("\nZone type distribution:")
        for key, value in sorted(zone_counts.items()):
            print(f"  {key:<10} : {value:,}")

    print_score_buckets(df_all)
    print_score_breakdown(df_all)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()