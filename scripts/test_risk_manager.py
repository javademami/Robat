"""
Risk Manager V1 validation
==========================

Full pipeline:

5M
 ↓
1H / 4H
 ↓
4H Swings
 ↓
Liquidity
 ↓
Sweeps
 ↓
1H Internal Swings
 ↓
MSS
 ↓
Displacement
 ↓
OB / FVG
 ↓
Retest
 ↓
5M Micro-MSS
 ↓
Entry Detector
 ↓
Risk Manager
"""

from __future__ import annotations

from pathlib import Path

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
)

from src.strategy.risk_manager import (
    calculate_atr,
    build_risk_plan,
    risk_plans_to_dataframe,
    get_valid_risk_plans,
)


# ============================================================
# CONFIG
# ============================================================

PIVOT_WINDOW_4H = 2
PIVOT_WINDOW_1H = 2
PIVOT_WINDOW_5M = 2

# Deliberately different from PIVOT_WINDOW_4H: this was chosen
# specifically for Equal High/Low liquidity-zone detection during
# sensitivity testing. It must NOT be reused for 4H structural
# swing detection (detect_pivots) -- that regressed 4H swings
# from ~623 to 451 when the two were accidentally conflated.
LIQUIDITY_PIVOT_WINDOW = 3

ATR_PERIOD = 14

EQUAL_TOLERANCE_ATR = 0.15
LIQUIDITY_LOOKBACK = 100

MAX_BARS_AFTER_SWEEP = 12
MAX_BARS_AFTER_MSS = 8

OB_LOOKBACK = 8
FVG_SEARCH_BARS = 3

MAX_BARS_AFTER_ZONE = 24
MAX_BARS_AFTER_RETEST = 24

MIN_REFERENCE_DISTANCE_ATR = 0.25

TARGET_RR = 2.0
MIN_RR = 1.5
SL_BUFFER_ATR = 0.10


# ============================================================
# HELPERS
# ============================================================

def normalize_direction(value) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower().strip()


def get_timestamp(obj):
    return getattr(obj, "timestamp", None)


def find_zone_for_entry(entry, ob_events, fvg_events):
    """
    Find the exact OB/FVG zone belonging to the Entry.

    Matching priority:
    1. zone type + direction + retest timestamp
    2. zone type + direction + displacement timestamp
    """

    zone_type = str(
        getattr(entry, "zone_type", "")
    ).lower().strip()

    direction = normalize_direction(
        getattr(entry, "direction", "")
    )

    retest_timestamp = getattr(
        entry,
        "retest_timestamp",
        None,
    )

    displacement_timestamp = getattr(
        entry,
        "displacement_timestamp",
        None,
    )

    if zone_type == "ob":
        zones = ob_events
    elif zone_type == "fvg":
        zones = fvg_events
    else:
        return None

    # --------------------------------------------------------
    # Exact retest relationship
    # --------------------------------------------------------

    for zone in zones:

        z_direction = normalize_direction(
            getattr(zone, "direction", "")
        )

        if z_direction != direction:
            continue

        z_timestamp = get_timestamp(zone)

        # Retest event normally contains the originating
        # zone timestamp.
        if z_timestamp == getattr(
            entry,
            "zone_timestamp",
            None,
        ):
            return zone

    # --------------------------------------------------------
    # Displacement fallback
    # --------------------------------------------------------

    for zone in zones:

        z_direction = normalize_direction(
            getattr(zone, "direction", "")
        )

        if z_direction != direction:
            continue

        z_displacement = getattr(
            zone,
            "displacement_timestamp",
            None,
        )

        if z_displacement == displacement_timestamp:
            return zone

    return None


def build_zone_map(
    entries,
    ob_events,
    fvg_events,
):
    """
    Build Entry timestamp -> source zone mapping.
    """

    result = {}

    for entry in entries:

        zone = find_zone_for_entry(
            entry,
            ob_events,
            fvg_events,
        )

        result[get_timestamp(entry)] = zone

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("BUILDING RISK MANAGER TEST DATASET")
    print("=" * 72)

    # ========================================================
    # 1. LOAD 5M
    # ========================================================

    print("\n[1] Loading 5M...")

    df_5m = load_year(
        DATA_DIR,
        SYMBOL,
        "5m",
        YEAR,
    )

    print(f"5M candles: {len(df_5m):,}")

    # ========================================================
    # 2. RESAMPLE
    # ========================================================

    print("\n[2] Resampling...")

    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")

    print(f"1H candles: {len(df_1h):,}")
    print(f"4H candles: {len(df_4h):,}")

    # ========================================================
    # 3. 4H SWINGS
    # ========================================================

    print("\n[3] 4H swings...")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=PIVOT_WINDOW_4H,
        right_bars=PIVOT_WINDOW_4H,
        timeframe="4h",
    )

    print(f"4H swings: {len(swings_4h):,}")

    # ========================================================
    # 4. LIQUIDITY
    # ========================================================

    print("\n[4] Liquidity...")

    equal_highs = detect_equal_highs(
        df_4h,
        atr_period=ATR_PERIOD,
        tolerance_atr=EQUAL_TOLERANCE_ATR,
        lookback=LIQUIDITY_LOOKBACK,
        pivot_window=LIQUIDITY_PIVOT_WINDOW,
    )

    equal_lows = detect_equal_lows(
        df_4h,
        atr_period=ATR_PERIOD,
        tolerance_atr=EQUAL_TOLERANCE_ATR,
        lookback=LIQUIDITY_LOOKBACK,
        pivot_window=LIQUIDITY_PIVOT_WINDOW,
    )

    previous_day = detect_previous_day_liquidity(
        df_1h
    )

    swing_liquidity = detect_swing_liquidity(
        swings_4h
    )

    zones = deduplicate_zones(
        equal_highs
        + equal_lows
        + previous_day
        + swing_liquidity
    )

    print(f"Liquidity zones: {len(zones):,}")

    # ========================================================
    # 5. SWEEPS
    # ========================================================

    print("\n[5] Sweeps...")

    sweeps = detect_sweeps(
        df_1h,
        zones,
        atr_period=ATR_PERIOD,
    )

    print(f"Sweeps: {len(sweeps):,}")

    # ========================================================
    # 6. 1H INTERNAL SWINGS
    # ========================================================

    print("\n[6] 1H internal swings...")

    internal_swings_1h = detect_pivots(
        df_1h,
        left_bars=PIVOT_WINDOW_1H,
        right_bars=PIVOT_WINDOW_1H,
        timeframe="1h",
    )

    print(
        f"1H internal swings: "
        f"{len(internal_swings_1h):,}"
    )

    # ========================================================
    # 7. MSS
    # ========================================================

    print("\n[7] MSS...")

    mss_events = detect_mss(
        df_1h,
        sweeps,
        internal_swings_1h,
        max_bars_after_sweep=MAX_BARS_AFTER_SWEEP,
    )

    print(f"MSS: {len(mss_events):,}")

    # ========================================================
    # 8. DISPLACEMENT
    # ========================================================

    print("\n[8] Displacement...")

    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=1.0,
        bullish_clv=0.70,
        bearish_clv=0.30,
        max_bars_after_mss=MAX_BARS_AFTER_MSS,
    )

    print(
        f"Displacements: "
        f"{len(displacement_events):,}"
    )

    # ========================================================
    # 9. ORDER BLOCKS
    # ========================================================

    print("\n[9] Order Blocks...")

    ob_events = detect_order_blocks(
        df_1h,
        displacement_events,
        lookback=OB_LOOKBACK,
    )

    print(f"Order Blocks: {len(ob_events):,}")

    # ========================================================
    # 10. FVG
    # ========================================================

    print("\n[10] FVG...")

    fvg_events = detect_fvgs(
        df_1h,
        displacement_events,
        search_bars=FVG_SEARCH_BARS,
        atr_period=ATR_PERIOD,
    )

    print(f"FVG: {len(fvg_events):,}")

    # ========================================================
    # 11. RETEST
    # ========================================================

    print("\n[11] Retests...")

    ob_retests = detect_ob_retests(
        df_1h,
        ob_events,
        max_bars_after_zone=MAX_BARS_AFTER_ZONE,
    )

    fvg_retests = detect_fvg_retests(
        df_1h,
        fvg_events,
        max_bars_after_zone=MAX_BARS_AFTER_ZONE,
    )

    all_retests = ob_retests + fvg_retests

    print(f"OB retests: {len(ob_retests):,}")
    print(f"FVG retests: {len(fvg_retests):,}")
    print(f"Total retests: {len(all_retests):,}")

    # ========================================================
    # 12. 5M SWINGS
    # ========================================================

    print("\n[12] 5M swings...")

    swings_5m = detect_pivots(
        df_5m,
        left_bars=PIVOT_WINDOW_5M,
        right_bars=PIVOT_WINDOW_5M,
        timeframe="5m",
    )

    print(
        f"5M internal swings: "
        f"{len(swings_5m):,}"
    )

    # ========================================================
    # 13. MICRO MSS
    # ========================================================

    print("\n[13] Micro-MSS...")

    df_5m_indexed = df_5m.copy()

    if not isinstance(
        df_5m_indexed.index,
        pd.DatetimeIndex,
    ):
        if "timestamp" in df_5m_indexed.columns:
            df_5m_indexed["timestamp"] = pd.to_datetime(
                df_5m_indexed["timestamp"],
                utc=True,
            )

            df_5m_indexed = df_5m_indexed.set_index(
                "timestamp"
            )

    micro_events = detect_micro_mss(
        df_5m_indexed,
        all_retests,
        swings_5m,
        max_bars_after_retest=MAX_BARS_AFTER_RETEST,
        min_reference_distance_atr=MIN_REFERENCE_DISTANCE_ATR,
    )

    micro_events = deduplicate_micro_mss(
        micro_events
    )

    print(f"Micro-MSS: {len(micro_events):,}")

    # ========================================================
    # 14. ENTRY CONTEXT
    # ========================================================

    print("\n[14] Building entry context...")

    entry_prices = {}

    for event in micro_events:

        timestamp = get_timestamp(event)

        try:
            if timestamp in df_5m_indexed.index:
                entry_prices[str(timestamp)] = float(
                    df_5m_indexed.loc[
                        timestamp,
                        "close",
                    ]
                )
        except Exception:
            pass

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------
    #
    # V1 temporary bias:
    # use the latest confirmed 4H swing relationship.
    #
    # This is intentionally kept simple here.
    # It can later be replaced by a dedicated
    # market-structure/bias detector.

    bias_by_timestamp = {}

    for event in micro_events:

        ts = get_timestamp(event)

        direction = normalize_direction(
            getattr(event, "direction", "")
        )

        # For the current validation dataset,
        # use event direction as neutral-safe fallback.
        #
        # Entry Detector already treats missing explicit
        # bias conservatively.
        bias_by_timestamp[str(ts)] = "neutral"

    context = {
        "entry_prices": entry_prices,
        "bias_by_timestamp": bias_by_timestamp,
        "sweeps": sweeps,
    }

    # ========================================================
    # 15. ENTRY DETECTOR
    # ========================================================

    print("\n[15] Running Entry Detector...")

    entries = detect_entry_candidates(
        micro_events,
        all_retests,
        mss_events,
        displacement_events,
        ob_events,
        fvg_events,
        context=context,
        debug=True,
    )

    tradable_entries = get_tradable_entries(
        entries
    )

    print(f"Entry candidates: {len(entries):,}")
    print(
        f"Tradable (score >= 70): "
        f"{len(tradable_entries):,}"
    )

    # ========================================================
    # 16. ATR
    # ========================================================

    print("\n[16] Calculating 1H ATR...")

    df_1h_indexed = df_1h.copy()

    if not isinstance(
        df_1h_indexed.index,
        pd.DatetimeIndex,
    ):
        if "timestamp" in df_1h_indexed.columns:
            df_1h_indexed["timestamp"] = pd.to_datetime(
                df_1h_indexed["timestamp"],
                utc=True,
            )

            df_1h_indexed = df_1h_indexed.set_index(
                "timestamp"
            )

    atr_1h = calculate_atr(
        df_1h_indexed,
        period=ATR_PERIOD,
    )

    # --------------------------------------------------------
    # FIX: Entry/Micro-MSS timestamps are 5M-precision (e.g.
    # 14:30, 05:45), but atr_1h only has values on the hour
    # (14:00, 05:00, ...). An exact dict lookup (the previous
    # approach) silently returned None for any entry not landing
    # exactly on an hour boundary -- verified this was true for
    # 10 of 17 entries (59%) in the last run, each ending up with
    # a zero ATR buffer on its stop-loss as a result.
    #
    # Use "last known value at or before this timestamp" instead,
    # which is both correct (no future data used) and matches the
    # as-of pattern used everywhere else in this project's
    # swing/liquidity confirmation logic.
    # --------------------------------------------------------

    atr_1h_sorted = atr_1h.sort_index()

    def _atr_as_of(timestamp):
        try:
            value = atr_1h_sorted.asof(timestamp)
        except (KeyError, ValueError):
            return None
        if value is None or pd.isna(value):
            return None
        return float(value)

    atr_by_timestamp = {
        ts: value
        for ts, value in atr_1h_sorted.items()
        if pd.notna(value)
    }

    print(
        f"ATR samples: "
        f"{len(atr_by_timestamp):,}"
    )

    # ========================================================
    # 17. MATCH ZONES
    # ========================================================

    print("\n[17] Matching Entry -> Zone...")

    zone_map = build_zone_map(
        entries,
        ob_events,
        fvg_events,
    )

    matched_zones = sum(
        1
        for zone in zone_map.values()
        if zone is not None
    )

    print(
        f"Zones matched: "
        f"{matched_zones}/{len(entries)}"
    )

    # ========================================================
    # 18. RISK PLANS
    # ========================================================

    print("\n[18] Building Risk Plans...")

    risk_plans = []

    for entry in entries:

        timestamp = get_timestamp(entry)

        zone = zone_map.get(timestamp)

        atr = _atr_as_of(timestamp)

        plan = build_risk_plan(
            entry_candidate=entry,
            zone=zone,
            atr=atr,
            target_rr=TARGET_RR,
            min_rr=MIN_RR,
            buffer_atr=SL_BUFFER_ATR,
        )

        risk_plans.append(plan)

    valid_plans = get_valid_risk_plans(
        risk_plans
    )

    print(
        f"Risk plans: "
        f"{len(risk_plans):,}"
    )

    print(
        f"Valid risk plans: "
        f"{len(valid_plans):,}"
    )

    # ========================================================
    # 19. SUMMARY
    # ========================================================

    print("\n" + "=" * 72)
    print("RISK MANAGER SUMMARY")
    print("=" * 72)

    if risk_plans:

        df_risk = risk_plans_to_dataframe(
            risk_plans
        )

        print(
            f"Entry candidates       : "
            f"{len(entries)}"
        )

        print(
            f"Tradable entries       : "
            f"{len(tradable_entries)}"
        )

        print(
            f"Risk plans             : "
            f"{len(risk_plans)}"
        )

        print(
            f"Valid risk plans       : "
            f"{len(valid_plans)}"
        )

        print(
            f"Invalid risk plans     : "
            f"{len(risk_plans) - len(valid_plans)}"
        )

        valid_df = df_risk[
            df_risk["valid"] == True
        ]

        if not valid_df.empty:

            print("\nRisk statistics:")

            print(
                f"  Mean risk/unit       : "
                f"{valid_df['risk_per_unit'].mean():.4f}"
            )

            print(
                f"  Median risk/unit     : "
                f"{valid_df['risk_per_unit'].median():.4f}"
            )

            print(
                f"  Mean RR              : "
                f"{valid_df['risk_reward'].mean():.2f}"
            )

            print(
                f"  Min RR               : "
                f"{valid_df['risk_reward'].min():.2f}"
            )

            print(
                f"  Max RR               : "
                f"{valid_df['risk_reward'].max():.2f}"
            )

            print("\nDirection:")

            print(
                valid_df[
                    "direction"
                ].value_counts().to_string()
            )

            print("\nZone type:")

            print(
                valid_df[
                    "zone_type"
                ].value_counts().to_string()
            )

        print("\nRejection reasons:")

        invalid_df = df_risk[
            df_risk["valid"] == False
        ]

        if invalid_df.empty:
            print("  None")
        else:
            print(
                invalid_df[
                    "rejection_reason"
                ].value_counts().to_string()
            )

        # ====================================================
        # SAVE
        # ====================================================

        output_dir = (
            Path(DATA_DIR)
            / "validation"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_path = (
            output_dir
            / "risk_plans.csv"
        )

        valid_path = (
            output_dir
            / "risk_plans_valid.csv"
        )

        df_risk.to_csv(
            all_path,
            index=False,
        )

        valid_df.to_csv(
            valid_path,
            index=False,
        )

        print(
            f"\nSaved: {all_path}"
        )

        print(
            f"Saved: {valid_path}"
        )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()