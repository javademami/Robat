from __future__ import annotations

"""
Position Sizer V1 integration test.

Pipeline:

5M
 ↓
1H / 4H
 ↓
4H Structure
 ↓
4H Liquidity
 ↓
1H Sweep
 ↓
1H MSS
 ↓
1H Displacement
 ↓
OB / FVG
 ↓
1H Retest
 ↓
5M Micro-MSS V1.2
 ↓
Entry Detector V2
 ↓
Risk Manager
 ↓
Position Sizer V1
 ↓
CSV

Important:
- This test does NOT change strategy parameters.
- Liquidity remains on 4H.
- 4H pivot window remains separate from liquidity pivot window.
- Micro-MSS uses V1.2 look-ahead-safe logic.
- Risk Manager uses real 1H ATR.
- Position Sizer uses fixed account risk.
- Leverage affects margin only, NOT account-level risk.
"""

from pathlib import Path
import sys
from statistics import mean, median


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIG
# ============================================================

from src.config import DATA_DIR, SYMBOL, YEAR


# ============================================================
# DATA
# ============================================================

from src.data.storage import load_year
from src.data.aggregator import resample_ohlcv


# ============================================================
# STRATEGY
# ============================================================

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

from src.strategy.entry_detector import detect_entry_candidates

from src.strategy.risk_manager import (
    build_risk_plans,
    calculate_atr,
    get_valid_risk_plans,
    risk_plans_to_dataframe,
)

from src.strategy.position_sizer import (
    build_position_sizes,
    get_valid_position_sizes,
    position_sizes_to_dataframe,
)


# ============================================================
# TEST CONFIG
# ============================================================

ACCOUNT_EQUITY = 10_000.0

# 1% risk per trade
RISK_PERCENT = 1.0

# Leverage is used only for margin calculation.
LEVERAGE = 1.0

# ============================================================
# STRATEGY PARAMETERS
# ============================================================

ATR_PERIOD = 14

# Keep 4H structure pivot separate.
STRUCTURE_PIVOT_WINDOW_4H = 2

# Liquidity uses the selected baseline.
LIQUIDITY_PIVOT_WINDOW = 3

INTERNAL_PIVOT_WINDOW_1H = 2
MICRO_PIVOT_WINDOW_5M = 2

EQUAL_TOLERANCE_ATR = 0.15

SWEEP_ATR_PERIOD = 14

MSS_MAX_BARS_AFTER_SWEEP = 12

DISPLACEMENT_MIN_BODY_ATR = 1.0
DISPLACEMENT_BULLISH_CLV = 0.70
DISPLACEMENT_BEARISH_CLV = 0.30
DISPLACEMENT_MAX_BARS_AFTER_MSS = 8

OB_LOOKBACK = 8

FVG_SEARCH_BARS = 3

RETEST_MAX_BARS_AFTER_ZONE = 24

MICRO_MSS_MAX_BARS_AFTER_RETEST = 24

MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR = 0.25

# Risk Manager V1
RISK_REWARD = 2.0


# ============================================================
# HELPERS
# ============================================================

def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def safe_float(value):
    try:
        if value is None:
            return None

        value = float(value)

        if value != value:
            return None

        return value

    except (TypeError, ValueError):
        return None


def normalize_direction(value) -> str:
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
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)

            if value is not None:
                return value

    return default


# ============================================================
# MAIN
# ============================================================

def main():

    print_section("BUILDING POSITION SIZER TEST DATASET")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Symbol:       {SYMBOL}")
    print(f"Year:         {YEAR}")

    print()
    print("Position Sizer configuration:")
    print(f"Account equity: {ACCOUNT_EQUITY:,.2f}")
    print(f"Risk percent:   {RISK_PERCENT:.2f}%")
    print(f"Leverage:       {LEVERAGE:.2f}x")
    print(f"Target RR:      {RISK_REWARD:.2f}")

    # ========================================================
    # LOAD 5M
    # ========================================================

    df_5m = load_year(
        DATA_DIR,
        SYMBOL,
        "5m",
        YEAR,
    )

    print(f"\n5M candles: {len(df_5m):,}")

    # ========================================================
    # RESAMPLE
    # ========================================================

    df_1h = resample_ohlcv(
        df_5m,
        "1h",
    )

    df_4h = resample_ohlcv(
        df_5m,
        "4h",
    )

    print(f"1H candles: {len(df_1h):,}")
    print(f"4H candles: {len(df_4h):,}")

    # ========================================================
    # 4H STRUCTURE
    # ========================================================

    print_section("4H STRUCTURE")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=STRUCTURE_PIVOT_WINDOW_4H,
        right_bars=STRUCTURE_PIVOT_WINDOW_4H,
        timeframe="4h",
    )

    print(f"4H swings: {len(swings_4h):,}")

    # ========================================================
    # 4H LIQUIDITY
    # ========================================================

    print_section("4H LIQUIDITY")

    equal_highs = detect_equal_highs(
        df_4h,
        atr_period=ATR_PERIOD,
        tolerance_atr=EQUAL_TOLERANCE_ATR,
        pivot_window=LIQUIDITY_PIVOT_WINDOW,
    )

    equal_lows = detect_equal_lows(
        df_4h,
        atr_period=ATR_PERIOD,
        tolerance_atr=EQUAL_TOLERANCE_ATR,
        pivot_window=LIQUIDITY_PIVOT_WINDOW,
    )

    previous_day_zones = detect_previous_day_liquidity(
        df_4h,
    )

    swing_zones = detect_swing_liquidity(
        swings_4h,
    )

    liquidity_zones = (
        equal_highs
        + equal_lows
        + previous_day_zones
        + swing_zones
    )

    liquidity_zones = deduplicate_zones(
        liquidity_zones,
        price_tolerance=1e-8,
    )

    print(f"Equal highs:       {len(equal_highs):,}")
    print(f"Equal lows:        {len(equal_lows):,}")
    print(f"Previous-day:      {len(previous_day_zones):,}")
    print(f"Swing zones:       {len(swing_zones):,}")
    print(f"Liquidity zones:   {len(liquidity_zones):,}")

    # ========================================================
    # SWEEPS
    # ========================================================

    print_section("SWEEPS")

    sweeps = detect_sweeps(
        df_1h,
        liquidity_zones,
        atr_period=SWEEP_ATR_PERIOD,
    )

    print(f"Sweeps: {len(sweeps):,}")

    # ========================================================
    # 1H INTERNAL STRUCTURE
    # ========================================================

    print_section("1H INTERNAL STRUCTURE")

    swings_1h = detect_pivots(
        df_1h,
        left_bars=INTERNAL_PIVOT_WINDOW_1H,
        right_bars=INTERNAL_PIVOT_WINDOW_1H,
        timeframe="1h",
    )

    print(f"1H internal swings: {len(swings_1h):,}")

    # ========================================================
    # MSS
    # ========================================================

    print_section("1H MSS")

    mss_events = detect_mss(
        df_1h,
        sweeps,
        swings_1h,
        max_bars_after_sweep=MSS_MAX_BARS_AFTER_SWEEP,
    )

    print(f"MSS: {len(mss_events):,}")

    # ========================================================
    # DISPLACEMENT
    # ========================================================

    print_section("DISPLACEMENT")

    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=DISPLACEMENT_MIN_BODY_ATR,
        bullish_clv=DISPLACEMENT_BULLISH_CLV,
        bearish_clv=DISPLACEMENT_BEARISH_CLV,
        max_bars_after_mss=DISPLACEMENT_MAX_BARS_AFTER_MSS,
    )

    print(f"Displacements: {len(displacement_events):,}")

    # ========================================================
    # ORDER BLOCKS
    # ========================================================

    print_section("ORDER BLOCKS")

    order_blocks = detect_order_blocks(
        df_1h,
        displacement_events,
        lookback=OB_LOOKBACK,
    )

    print(f"Order Blocks: {len(order_blocks):,}")

    # ========================================================
    # FVG
    # ========================================================

    print_section("FVG")

    fvgs = detect_fvgs(
        df_1h,
        displacement_events,
        search_bars=FVG_SEARCH_BARS,
        atr_period=ATR_PERIOD,
    )

    print(f"FVG: {len(fvgs):,}")

    # ========================================================
    # RETESTS
    # ========================================================

    print_section("RETESTS")

    ob_retests = detect_ob_retests(
        df_1h,
        order_blocks,
        max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE,
    )

    fvg_retests = detect_fvg_retests(
        df_1h,
        fvgs,
        max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE,
    )

    all_retests = list(ob_retests) + list(fvg_retests)

    print(f"OB retests:  {len(ob_retests):,}")
    print(f"FVG retests: {len(fvg_retests):,}")
    print(f"Total retests: {len(all_retests):,}")

    # ========================================================
    # 5M INTERNAL SWINGS
    # ========================================================

    print_section("5M MICRO STRUCTURE")

    swings_5m = detect_pivots(
        df_5m,
        left_bars=MICRO_PIVOT_WINDOW_5M,
        right_bars=MICRO_PIVOT_WINDOW_5M,
        timeframe="5m",
    )

    print(f"5M internal swings: {len(swings_5m):,}")

    # ========================================================
    # MICRO MSS
    # ========================================================

    print_section("MICRO-MSS V1.2")

    # Micro-MSS requires DatetimeIndex.
    df_5m_indexed = df_5m.copy()

    if "timestamp" in df_5m_indexed.columns:

        df_5m_indexed["timestamp"] = (
            __import__("pandas")
            .to_datetime(
                df_5m_indexed["timestamp"],
                utc=True,
            )
        )

        df_5m_indexed = (
            df_5m_indexed
            .set_index("timestamp")
            .sort_index()
        )

    elif not isinstance(
        df_5m_indexed.index,
        __import__("pandas").DatetimeIndex,
    ):

        df_5m_indexed.index = (
            __import__("pandas")
            .to_datetime(
                df_5m_indexed.index,
                utc=True,
            )
        )

    micro_mss_events = detect_micro_mss(
        df_5m_indexed,
        all_retests,
        swings_5m,
        max_bars_after_retest=MICRO_MSS_MAX_BARS_AFTER_RETEST,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )

    micro_mss_events = deduplicate_micro_mss(
        micro_mss_events,
    )

    print(f"Micro-MSS: {len(micro_mss_events):,}")

    # ========================================================
    # ENTRY DETECTOR
    # ========================================================

    print_section("ENTRY DETECTOR")

    # --------------------------------------------------------
    # Build timestamp-based context
    # --------------------------------------------------------

    def timestamp_key(value):
        if value is None:
            return None

        try:
            ts = __import__("pandas").Timestamp(value)

            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")

            # IMPORTANT: entry_detector.py looks up context dicts
            # (entry_prices, sweeps_by_mss_timestamp) using its own
            # _timestamp_key(), which is str(value). Returning a
            # raw Timestamp object here means every .get() lookup
            # on the consumer side silently misses (dict keyed by
            # Timestamp, looked up by str) -- same failure mode
            # already found and fixed once in test_risk_manager.py.
            return str(ts)

        except Exception:
            return str(value)

    # --------------------------------------------------------
    # Bias context
    #
    # The current Entry Detector supports bias_by_timestamp.
    # We intentionally keep this simple here and do not invent
    # a new structure detector.
    # --------------------------------------------------------

    context = {}

    # --------------------------------------------------------
    # Entry prices
    #
    # Micro-MSS event timestamp corresponds to a 5M candle.
    # Use that candle close as the execution reference.
    # --------------------------------------------------------

    entry_prices = {}

    if "timestamp" in df_5m.columns:

        for _, row in df_5m.iterrows():

            ts = timestamp_key(row["timestamp"])

            close = safe_float(row.get("close"))

            if ts is not None and close is not None:
                entry_prices[ts] = close

    else:

        for idx, row in df_5m.iterrows():

            ts = timestamp_key(idx)

            close = safe_float(row.get("close"))

            if ts is not None and close is not None:
                entry_prices[ts] = close

    context["entry_prices"] = entry_prices

    # --------------------------------------------------------
    # Sweep lookup by MSS timestamp
    # --------------------------------------------------------

    sweeps_by_mss_timestamp = {}

    for sweep in sweeps:

        sweep_ts = get_attr(
            sweep,
            "timestamp",
            "sweep_timestamp",
        )

        if sweep_ts is None:
            continue

        key = timestamp_key(sweep_ts)

        if key is not None:
            sweeps_by_mss_timestamp[key] = sweep

    context["sweeps_by_mss_timestamp"] = (
        sweeps_by_mss_timestamp
    )

    # --------------------------------------------------------
    # Run Entry Detector
    # --------------------------------------------------------

    entry_candidates = detect_entry_candidates(
        micro_mss_events,
        all_retests,
        mss_events,
        displacement_events,
        order_blocks,
        fvgs,
        context=context,
        debug=True,
    )

    tradable_entries = [
        candidate
        for candidate in entry_candidates
        if getattr(candidate, "tradable", False)
    ]

    print(f"Entry candidates: {len(entry_candidates):,}")
    print(f"Tradable:         {len(tradable_entries):,}")

    # ========================================================
    # RISK MANAGER
    # ========================================================

    print_section("RISK MANAGER")

    # --------------------------------------------------------
    # zones: build_risk_plans expects a flat list of OB/FVG
    # zone objects (it matches by timestamp+direction+zone_type
    # internally), not a raw price dataframe.
    # --------------------------------------------------------

    risk_zones = list(order_blocks) + list(fvgs)

    # --------------------------------------------------------
    # atr_by_timestamp: build_risk_plans does a plain
    # atr_by_timestamp.get(candidate.timestamp) -- an exact
    # dict lookup, not an as-of search. Entry timestamps are
    # 5M-precision (e.g. 14:30, 05:45) while ATR is only defined
    # on the hour, so we resolve the as-of ATR value for each
    # candidate's exact timestamp *here* and key the dict by
    # that same timestamp object, so the exact-match .get()
    # inside build_risk_plans succeeds for every candidate.
    # --------------------------------------------------------

    import pandas as _pd

    df_1h_indexed = df_1h.copy()

    if "timestamp" in df_1h_indexed.columns:

        df_1h_indexed["timestamp"] = _pd.to_datetime(
            df_1h_indexed["timestamp"],
            utc=True,
        )

        df_1h_indexed = (
            df_1h_indexed
            .set_index("timestamp")
            .sort_index()
        )

    atr_1h = calculate_atr(
        df_1h_indexed,
        period=ATR_PERIOD,
    ).sort_index()

    def _atr_as_of(value):

        if value is None:
            return None

        try:
            ts = _pd.Timestamp(value)

            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")

            result = atr_1h.asof(ts)

        except (KeyError, ValueError, TypeError):
            return None

        if result is None or _pd.isna(result):
            return None

        return float(result)

    atr_by_timestamp = {}

    for candidate in entry_candidates:

        candidate_ts = getattr(
            candidate,
            "timestamp",
            None,
        )

        if candidate_ts is None:
            continue

        atr_value = _atr_as_of(candidate_ts)

        if atr_value is not None:
            atr_by_timestamp[candidate_ts] = atr_value

    print(
        f"ATR resolved for "
        f"{len(atr_by_timestamp):,} / "
        f"{len(entry_candidates):,} candidates"
    )

    risk_plans = build_risk_plans(
        entry_candidates,
        risk_zones,
        atr_by_timestamp,
        target_rr=RISK_REWARD,
    )

    valid_risk_plans = get_valid_risk_plans(risk_plans)

    print(f"Risk plans:       {len(risk_plans):,}")
    print(f"Valid risk plans: {len(valid_risk_plans):,}")

    invalid_risk_plans = [
        plan
        for plan in risk_plans
        if not getattr(plan, "valid", False)
    ]

    print(f"Invalid:           {len(invalid_risk_plans):,}")

    # ========================================================
    # SAVE RISK PLANS
    # (new — saves both full and valid risk-plan CSVs)
    # ========================================================

    print_section("SAVING RISK PLANS")

    risk_output_dir = (
        PROJECT_ROOT
        / "data"
        / "validation"
    )

    risk_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_risk_all = risk_plans_to_dataframe(risk_plans)

    df_risk_valid = risk_plans_to_dataframe(valid_risk_plans)

    risk_all_path = (
        risk_output_dir
        / "risk_plans.csv"
    )

    risk_valid_path = (
        risk_output_dir
        / "risk_plans_valid.csv"
    )

    df_risk_all.to_csv(
        risk_all_path,
        index=False,
    )

    df_risk_valid.to_csv(
        risk_valid_path,
        index=False,
    )

    print(f"Saved: {risk_all_path}")
    print(f"Saved: {risk_valid_path}")

    # ========================================================
    # POSITION SIZER
    # ========================================================

    print_section("POSITION SIZER V1")

    position_sizes = build_position_sizes(
        valid_risk_plans,
        account_equity=ACCOUNT_EQUITY,
        risk_percent=RISK_PERCENT,
        leverage=LEVERAGE,
    )

    valid_position_sizes = get_valid_position_sizes(
        position_sizes,
    )

    print(f"Position sizes: {len(position_sizes):,}")
    print(f"Valid:          {len(valid_position_sizes):,}")
    print(
        f"Invalid:        "
        f"{len(position_sizes) - len(valid_position_sizes):,}"
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    print_section("POSITION SIZER VALIDATION")

    expected_risk_amount = (
        ACCOUNT_EQUITY * RISK_PERCENT / 100.0
    )

    print(
        f"Expected risk amount/trade: "
        f"${expected_risk_amount:,.2f}"
    )

    risk_errors = []

    for item in valid_position_sizes:

        expected_position_size = (
            expected_risk_amount
            / item.risk_per_unit
        )

        expected_notional = (
            expected_position_size
            * item.entry_price
        )

        expected_margin = (
            expected_notional
            / LEVERAGE
        )

        actual_risk = (
            item.position_size
            * item.risk_per_unit
        )

        if abs(actual_risk - expected_risk_amount) > 1e-8:
            risk_errors.append(
                (
                    item.timestamp,
                    "risk_amount_mismatch",
                    actual_risk,
                    expected_risk_amount,
                )
            )

        if abs(
            item.position_size
            - expected_position_size
        ) > 1e-8:

            risk_errors.append(
                (
                    item.timestamp,
                    "position_size_mismatch",
                    item.position_size,
                    expected_position_size,
                )
            )

        if abs(
            item.notional_value
            - expected_notional
        ) > 1e-8:

            risk_errors.append(
                (
                    item.timestamp,
                    "notional_mismatch",
                    item.notional_value,
                    expected_notional,
                )
            )

        if abs(
            item.margin_required
            - expected_margin
        ) > 1e-8:

            risk_errors.append(
                (
                    item.timestamp,
                    "margin_mismatch",
                    item.margin_required,
                    expected_margin,
                )
            )

    print(
        f"Risk calculation errors: "
        f"{len(risk_errors):,}"
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    if valid_position_sizes:

        print_section("POSITION SIZE STATISTICS")

        risk_per_units = [
            item.risk_per_unit
            for item in valid_position_sizes
        ]

        position_sizes_values = [
            item.position_size
            for item in valid_position_sizes
        ]

        notionals = [
            item.notional_value
            for item in valid_position_sizes
        ]

        margins = [
            item.margin_required
            for item in valid_position_sizes
        ]

        rrs = [
            item.risk_reward
            for item in valid_position_sizes
        ]

        print(
            f"Risk/unit mean:   "
            f"${mean(risk_per_units):,.4f}"
        )

        print(
            f"Risk/unit median: "
            f"${median(risk_per_units):,.4f}"
        )

        print(
            f"Risk/unit min:    "
            f"${min(risk_per_units):,.4f}"
        )

        print(
            f"Risk/unit max:    "
            f"${max(risk_per_units):,.4f}"
        )

        print()

        print(
            f"Position size mean:   "
            f"{mean(position_sizes_values):.8f}"
        )

        print(
            f"Position size median: "
            f"{median(position_sizes_values):.8f}"
        )

        print(
            f"Position size min:    "
            f"{min(position_sizes_values):.8f}"
        )

        print(
            f"Position size max:    "
            f"{max(position_sizes_values):.8f}"
        )

        print()

        print(
            f"Notional mean:   "
            f"${mean(notionals):,.2f}"
        )

        print(
            f"Notional median: "
            f"${median(notionals):,.2f}"
        )

        print(
            f"Notional min:    "
            f"${min(notionals):,.2f}"
        )

        print(
            f"Notional max:    "
            f"${max(notionals):,.2f}"
        )

        print()

        print(
            f"Margin mean:   "
            f"${mean(margins):,.2f}"
        )

        print(
            f"Margin median: "
            f"${median(margins):,.2f}"
        )

        print(
            f"Margin min:    "
            f"${min(margins):,.2f}"
        )

        print(
            f"Margin max:    "
            f"${max(margins):,.2f}"
        )

        print()

        print(
            f"RR mean:   "
            f"{mean(rrs):.2f}"
        )

        print(
            f"RR median: "
            f"{median(rrs):.2f}"
        )

        print(
            f"RR min:    "
            f"{min(rrs):.2f}"
        )

        print(
            f"RR max:    "
            f"{max(rrs):.2f}"
        )

        # ====================================================
        # DIRECTION
        # ====================================================

        bullish = [
            item
            for item in valid_position_sizes
            if normalize_direction(item.direction)
            == "bullish"
        ]

        bearish = [
            item
            for item in valid_position_sizes
            if normalize_direction(item.direction)
            == "bearish"
        ]

        print()
        print(
            f"Bullish: {len(bullish):,}"
        )

        print(
            f"Bearish: {len(bearish):,}"
        )

        # ====================================================
        # ZONE
        # ====================================================

        ob_items = [
            item
            for item in valid_position_sizes
            if str(
                getattr(item, "zone_type", "")
            ).lower()
            in {"ob", "orderblock", "order_block"}
        ]

        fvg_items = [
            item
            for item in valid_position_sizes
            if str(
                getattr(item, "zone_type", "")
            ).lower()
            == "fvg"
        ]

        print()
        print(
            f"OB:  {len(ob_items):,}"
        )

        print(
            f"FVG: {len(fvg_items):,}"
        )

        # ====================================================
        # SCORE / GRADE
        # ====================================================

        grades = {}

        for item in valid_position_sizes:

            grade = getattr(item, "grade", None)

            if grade is None:
                grade = "UNKNOWN"

            grades[grade] = (
                grades.get(grade, 0) + 1
            )

        print()
        print("Grades:")

        for grade in sorted(grades):
            print(
                f"  {grade}: "
                f"{grades[grade]:,}"
            )

    # ========================================================
    # DETAILED TABLE
    # ========================================================

    if valid_position_sizes:

        print_section("POSITION SIZE DETAILS")

        for index, item in enumerate(
            valid_position_sizes,
            start=1,
        ):

            direction = normalize_direction(
                item.direction
            )

            print(
                f"{index:02d}. "
                f"{item.timestamp} | "
                f"{direction:<8} | "
                f"Entry {item.entry_price:,.2f} | "
                f"SL {item.stop_loss:,.2f} | "
                f"TP {item.take_profit:,.2f} | "
                f"Risk/unit "
                f"{item.risk_per_unit:,.2f} | "
                f"Size "
                f"{item.position_size:.8f} | "
                f"Notional "
                f"${item.notional_value:,.2f}"
            )

    # ========================================================
    # SAVE POSITION SIZES CSV
    # ========================================================

    print_section("SAVING RESULTS")

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "validation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_positions = position_sizes_to_dataframe(
        position_sizes,
    )

    df_positions_valid = position_sizes_to_dataframe(
        valid_position_sizes,
    )

    all_path = (
        output_dir
        / "position_sizes.csv"
    )

    valid_path = (
        output_dir
        / "position_sizes_valid.csv"
    )

    df_positions.to_csv(
        all_path,
        index=False,
    )

    df_positions_valid.to_csv(
        valid_path,
        index=False,
    )

    print(f"Saved: {all_path}")
    print(f"Saved: {valid_path}")

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_section("FINAL STATUS")

    # --------------------------------------------------------
    # FIX: The original check
    #   "Valid risk plans": len(valid_risk_plans) == len(risk_plans)
    # was WRONG.
    #
    # It required 100% of risk plans to be valid, which is
    # never the correct expectation: a healthy Risk Manager
    # pipeline MUST reject some candidates (bad R:R, bad
    # structure, missing zone, etc.). Failing whenever at
    # least one plan is rejected made this check permanent
    # noise.
    #
    # Correct semantics:
    #   - at least one valid risk plan exists
    #   - every plan inside valid_risk_plans really has valid=True
    # --------------------------------------------------------

    checks = {
        "Risk plans available":
            len(risk_plans) > 0,

        "Valid risk plans":
            len(valid_risk_plans) > 0,

        "Valid risk plans are valid":
            all(
                getattr(plan, "valid", False)
                for plan in valid_risk_plans
            ) if valid_risk_plans else False,

        "Position sizes created":
            len(position_sizes) == len(valid_risk_plans),

        "All position sizes valid":
            len(valid_position_sizes)
            == len(position_sizes),

        "Risk calculations":
            len(risk_errors) == 0,

        "Fixed account risk":
            all(
                abs(
                    item.position_size
                    * item.risk_per_unit
                    - expected_risk_amount
                ) < 1e-8
                for item in valid_position_sizes
            )
            if valid_position_sizes
            else False,
    }

    print()

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"{status:<6} | {name}"
        )

    print()

    if all(checks.values()):

        print(
            "POSITION SIZER V1: PASS"
        )

        print(
            "All valid Risk Plans were "
            "converted to correctly sized positions."
        )

    else:

        print(
            "POSITION SIZER V1: CHECK"
        )

        failed = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        print(
            "Failed checks:"
        )

        for item in failed:
            print(
                f"  - {item}"
            )

    print()
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()