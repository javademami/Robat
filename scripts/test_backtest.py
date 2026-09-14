from __future__ import annotations

"""
Backtest Engine V1 integration test.

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
Backtest Engine V1
 ↓
Metrics + Trade Log + Train/Test Split + Hypothesis Check

Important:
- This test does NOT change strategy parameters.
- Invalid risk plans are reported (not silently dropped).
- No look-ahead: exit simulation starts strictly after entry.
- SL first when both SL and TP are touched in the same candle.

V1.4 changes:
- Added Train/Test Split and Hypothesis Check.
- Hypothesis Check enforces MIN_SAMPLE_FOR_RULE = 5.
- Subsets below threshold are reported as UNTESTED.
- Fixed timezone comparison error: SPLIT_DATE is now UTC-aware,
  and parse_timestamp() normalizes all timestamps to UTC.
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
)

from src.strategy.position_sizer import (
    build_position_sizes,
    get_valid_position_sizes,
)

from src.backtest.backtest_engine import (
    run_backtest,
    trades_to_dataframe,
    calculate_metrics,
    print_metrics,
)


# ============================================================
# TEST CONFIG
# ============================================================

ACCOUNT_EQUITY = 10_000.0
RISK_PERCENT = 1.0
LEVERAGE = 1.0

# ============================================================
# STRATEGY PARAMETERS
# ============================================================

ATR_PERIOD = 14

STRUCTURE_PIVOT_WINDOW_4H = 2
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

RETEST_MAX_BARS_AFTER_ZONE = 48
MICRO_MSS_MAX_BARS_AFTER_RETEST = 72
MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR = 0.25

RISK_REWARD = 2.0

# Backtest
BACKTEST_MAX_BARS = 20_160

# Train/Test split — UTC-aware
SPLIT_DATE = f"{YEAR}-07-01T00:00:00+00:00"

# Minimum sample size for a rule to be reported
MIN_SAMPLE_FOR_RULE = 5
MIN_SAMPLE_FOR_STRONG = 30


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


def parse_timestamp(value):
    """
    Parse any timestamp-like value into a UTC-aware pandas Timestamp.

    Handles:
      - tz-naive Timestamp / str / datetime
      - tz-aware Timestamp / str / datetime
    """
    if value is None:
        return None
    try:
        import pandas as pd
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print_section("BUILDING BACKTEST DATASET")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Symbol:       {SYMBOL}")
    print(f"Year:         {YEAR}")

    print()
    print("Backtest configuration:")
    print(f"Account equity: {ACCOUNT_EQUITY:,.2f}")
    print(f"Risk percent:   {RISK_PERCENT:.2f}%")
    print(f"Leverage:       {LEVERAGE:.2f}x")
    print(f"Target RR:      {RISK_REWARD:.2f}")
    print(f"Max bars:       {BACKTEST_MAX_BARS}")
    print(f"Split date:     {SPLIT_DATE}")
    print(f"Min sample:     {MIN_SAMPLE_FOR_RULE}")

    # ========================================================
    # LOAD 5M
    # ========================================================

    df_5m = load_year(DATA_DIR, SYMBOL, "5m", YEAR)

    print(f"\n5M candles: {len(df_5m):,}")

    # ========================================================
    # RESAMPLE
    # ========================================================

    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")

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

    previous_day_zones = detect_previous_day_liquidity(df_4h)
    swing_zones = detect_swing_liquidity(swings_4h)

    liquidity_zones = (
        equal_highs + equal_lows + previous_day_zones + swing_zones
    )

    liquidity_zones = deduplicate_zones(
        liquidity_zones,
        price_tolerance=1e-8,
    )

    print(f"Liquidity zones: {len(liquidity_zones):,}")

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
    # 1H STRUCTURE
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
    # OB / FVG
    # ========================================================

    print_section("ORDER BLOCKS")

    order_blocks = detect_order_blocks(
        df_1h,
        displacement_events,
        lookback=OB_LOOKBACK,
    )

    print(f"Order Blocks: {len(order_blocks):,}")

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

    print(f"Total retests: {len(all_retests):,}")

    # ========================================================
    # 5M MICRO STRUCTURE
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
    # MICRO-MSS
    # ========================================================

    print_section("MICRO-MSS V1.2")

    df_5m_indexed = df_5m.copy()
    if "timestamp" in df_5m_indexed.columns:
        df_5m_indexed["timestamp"] = __import__("pandas").to_datetime(
            df_5m_indexed["timestamp"], utc=True,
        )
        df_5m_indexed = (
            df_5m_indexed.set_index("timestamp").sort_index()
        )
    elif not isinstance(df_5m_indexed.index, __import__("pandas").DatetimeIndex):
        df_5m_indexed.index = __import__("pandas").to_datetime(
            df_5m_indexed.index, utc=True,
        )

    micro_mss_events = detect_micro_mss(
        df_5m_indexed,
        all_retests,
        swings_5m,
        max_bars_after_retest=MICRO_MSS_MAX_BARS_AFTER_RETEST,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )

    micro_mss_events = deduplicate_micro_mss(micro_mss_events)

    print(f"Micro-MSS: {len(micro_mss_events):,}")

    # ========================================================
    # ENTRY DETECTOR
    # ========================================================

    print_section("ENTRY DETECTOR")

    def timestamp_key(value):
        if value is None:
            return None
        try:
            ts = __import__("pandas").Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return str(ts)
        except Exception:
            return str(value)

    context = {}

    # --------------------------------------------------------
    # Entry prices
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
        sweep_ts = get_attr(sweep, "timestamp", "sweep_timestamp")
        if sweep_ts is None:
            continue
        key = timestamp_key(sweep_ts)
        if key is not None:
            sweeps_by_mss_timestamp[key] = sweep

    context["sweeps_by_mss_timestamp"] = sweeps_by_mss_timestamp

    # --------------------------------------------------------
    # Bias by timestamp — simple 4H swing bias
    # --------------------------------------------------------

    bias_by_timestamp = {}

    swings_4h_sorted = sorted(
        swings_4h,
        key=lambda s: str(getattr(s, "timestamp", "")),
    )

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
            swing_dir = normalize_direction(
                getattr(latest_swing, "direction", "")
            )
            if swing_dir:
                bias_by_timestamp[str(micro_ts)] = swing_dir

    context["bias_by_timestamp"] = bias_by_timestamp

    # --------------------------------------------------------
    # Risk/Reward — target RR for scoring
    # --------------------------------------------------------

    context["risk_reward"] = RISK_REWARD

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
        c for c in entry_candidates
        if getattr(c, "tradable", False)
    ]

    print(f"Entry candidates: {len(entry_candidates):,}")
    print(f"Tradable:         {len(tradable_entries):,}")

    # ========================================================
    # RISK MANAGER
    # ========================================================

    print_section("RISK MANAGER")

    risk_zones = list(order_blocks) + list(fvgs)

    import pandas as _pd

    df_1h_indexed = df_1h.copy()
    if "timestamp" in df_1h_indexed.columns:
        df_1h_indexed["timestamp"] = _pd.to_datetime(
            df_1h_indexed["timestamp"], utc=True,
        )
        df_1h_indexed = (
            df_1h_indexed.set_index("timestamp").sort_index()
        )

    atr_1h = calculate_atr(df_1h_indexed, period=ATR_PERIOD).sort_index()

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
    for candidate in tradable_entries:
        candidate_ts = getattr(candidate, "timestamp", None)
        if candidate_ts is None:
            continue
        atr_value = _atr_as_of(candidate_ts)
        if atr_value is not None:
            atr_by_timestamp[candidate_ts] = atr_value

    # --------------------------------------------------------
    # Risk Manager receives ONLY tradable_entries
    # --------------------------------------------------------

    risk_plans = build_risk_plans(
        tradable_entries,
        risk_zones,
        atr_by_timestamp,
        target_rr=RISK_REWARD,
    )

    valid_risk_plans = get_valid_risk_plans(risk_plans)

    invalid_risk_plans = [
        p for p in risk_plans if not getattr(p, "valid", False)
    ]

    print(f"Risk plans:       {len(risk_plans):,}")
    print(f"Valid risk plans: {len(valid_risk_plans):,}")
    print(f"Invalid:          {len(invalid_risk_plans):,}")

    if invalid_risk_plans:
        print()
        print("Invalid risk plans (NOT traded):")
        for p in invalid_risk_plans:
            print(
                f"  - {p.timestamp} | "
                f"{normalize_direction(p.direction):<8} | "
                f"{getattr(p, 'rejection_reason', 'unknown')}"
            )

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

    valid_position_sizes = get_valid_position_sizes(position_sizes)

    print(f"Position sizes: {len(position_sizes):,}")
    print(f"Valid:          {len(valid_position_sizes):,}")

    # ========================================================
    # BACKTEST
    # ========================================================

    print_section("BACKTEST ENGINE V1")

    trades = run_backtest(
        valid_position_sizes,
        df_5m,
        max_bars=BACKTEST_MAX_BARS,
    )

    print(f"Trades simulated: {len(trades):,}")

    print()
    print("Trade log:")
    for i, t in enumerate(trades, start=1):
        r_str = f"{t.r_multiple:+.2f}" if t.r_multiple is not None else "—"
        pnl_str = f"${t.pnl:+.2f}" if t.pnl is not None else "—"
        print(
            f"{i:02d}. {t.entry_timestamp} | "
            f"{normalize_direction(t.direction):<8} | "
            f"Entry {t.entry_price:,.2f} | "
            f"Exit {t.exit_reason or '—':<6} | "
            f"R {r_str:>6} | "
            f"PnL {pnl_str:>10}"
        )

    # ========================================================
    # METRICS (full year)
    # ========================================================

    metrics = calculate_metrics(trades)
    print_metrics(metrics)

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print_section("TRAIN / TEST SPLIT (out-of-sample check)")

    print(
        f"Split date: {SPLIT_DATE} "
        f"(Jan-Jun = TRAIN/discovery, Jul-Dec = TEST/validation)"
    )

    split_ts = parse_timestamp(SPLIT_DATE)

    train_trades = []
    test_trades = []

    for t in trades:
        ts = parse_timestamp(t.entry_timestamp)
        if ts is None or split_ts is None:
            continue
        if ts < split_ts:
            train_trades.append(t)
        else:
            test_trades.append(t)

    print(f"TRAIN trades: {len(train_trades)}")
    print(f"TEST trades:  {len(test_trades)}")

    print()
    print("--- TRAIN (Jan-Jun) ---")
    if train_trades:
        train_metrics = calculate_metrics(train_trades)
        print_metrics(train_metrics)
    else:
        print("No trades in TRAIN period.")

    print()
    print("--- TEST (Jul-Dec) ---")
    if test_trades:
        test_metrics = calculate_metrics(test_trades)
        print_metrics(test_metrics)
    else:
        print("No trades in TEST period.")

    # ========================================================
    # HYPOTHESIS CHECK
    # ========================================================

    print_section(
        "HYPOTHESIS CHECK: direction filter "
        "(rule frozen from TRAIN, checked on TEST)"
    )

    # --------------------------------------------------------
    # Statistical disclaimer
    # --------------------------------------------------------

    print()
    print("Statistical disclaimer:")
    print(f"  Minimum sample for meaningful rule: n >= {MIN_SAMPLE_FOR_RULE}")
    print(f"  Minimum sample for strong evidence: n >= {MIN_SAMPLE_FOR_STRONG}")
    print(f"  With fewer samples, results are UNTESTED (not printed).")
    print()

    # --------------------------------------------------------
    # Helper: subset stats
    # --------------------------------------------------------

    def _subset_stats(trades_subset):
        closed = [
            t for t in trades_subset
            if getattr(t, "outcome", None) in {"win", "loss"}
        ]
        n = len(closed)
        if n == 0:
            return {"n": 0, "win_rate": 0.0, "total_r": 0.0, "pf": 0.0}
        wins = sum(1 for t in closed if t.outcome == "win")
        total_r = sum(
            float(t.r_multiple or 0.0) for t in closed
        )
        gross_profit = sum(
            float(t.pnl or 0.0) for t in closed
            if t.pnl and t.pnl > 0
        )
        gross_loss = abs(sum(
            float(t.pnl or 0.0) for t in closed
            if t.pnl and t.pnl < 0
        ))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        return {
            "n": n,
            "win_rate": wins / n,
            "total_r": total_r,
            "pf": pf,
        }

    # --------------------------------------------------------
    # TRAIN subsets
    # --------------------------------------------------------

    train_bull = [
        t for t in train_trades
        if normalize_direction(t.direction) == "bullish"
    ]
    train_bear = [
        t for t in train_trades
        if normalize_direction(t.direction) == "bearish"
    ]

    bull_stats = _subset_stats(train_bull)
    bear_stats = _subset_stats(train_bear)

    def _print_train_row(label, stats):
        if stats["n"] < MIN_SAMPLE_FOR_RULE:
            print(
                f"TRAIN {label}: UNTESTED  "
                f"(n={stats['n']} < {MIN_SAMPLE_FOR_RULE})"
            )
        else:
            print(
                f"TRAIN {label}: "
                f"n={stats['n']:2d}  "
                f"win_rate={stats['win_rate']:.2%}  "
                f"total_r={stats['total_r']:+.2f}"
            )

    _print_train_row("bullish", bull_stats)
    _print_train_row("bearish", bear_stats)

    # --------------------------------------------------------
    # Choose rule (based on TRAIN only)
    # --------------------------------------------------------

    chosen_direction = None

    if (
        bull_stats["n"] < MIN_SAMPLE_FOR_RULE
        and bear_stats["n"] < MIN_SAMPLE_FOR_RULE
    ):
        print()
        print(
            "Frozen rule: NONE — both subsets below "
            f"minimum sample (n < {MIN_SAMPLE_FOR_RULE})."
        )
        print("Cannot derive a directional rule from this data.")
        chosen_direction = None

    elif bull_stats["n"] < MIN_SAMPLE_FOR_RULE:
        print()
        print(
            "Frozen rule: keep bearish_only "
            f"(bullish subset has n={bull_stats['n']} < "
            f"{MIN_SAMPLE_FOR_RULE})."
        )
        chosen_direction = "bearish"

    elif bear_stats["n"] < MIN_SAMPLE_FOR_RULE:
        print()
        print(
            "Frozen rule: keep bullish_only "
            f"(bearish subset has n={bear_stats['n']} < "
            f"{MIN_SAMPLE_FOR_RULE})."
        )
        chosen_direction = "bullish"

    else:
        if bull_stats["total_r"] > bear_stats["total_r"]:
            chosen_direction = "bullish"
        else:
            chosen_direction = "bearish"
        print()
        print(
            f"Frozen rule: keep {chosen_direction}_only "
            f"(higher TRAIN total_r)."
        )

    # --------------------------------------------------------
    # Apply frozen rule to TEST
    # --------------------------------------------------------

    print()
    print("Applying frozen rule to TEST (out-of-sample, NOT re-tuned):")

    if chosen_direction is None:
        print(
            "  SKIPPED — no frozen rule available from TRAIN "
            f"(both subsets below n={MIN_SAMPLE_FOR_RULE})."
        )
    else:
        test_subset = [
            t for t in test_trades
            if normalize_direction(t.direction) == chosen_direction
        ]
        test_stats = _subset_stats(test_subset)

        if test_stats["n"] < MIN_SAMPLE_FOR_RULE:
            print(
                f"  UNTESTED  "
                f"(n={test_stats['n']} < {MIN_SAMPLE_FOR_RULE})"
            )
            print(
                "  Rule cannot be evaluated: "
                "sample size below threshold."
            )
            print(
                "  DO NOT interpret this as evidence "
                "for or against the rule."
            )
        else:
            print(
                f"  n={test_stats['n']:2d}  "
                f"win_rate={test_stats['win_rate']:.2%}  "
                f"total_r={test_stats['total_r']:+.2f}  "
                f"profit_factor={test_stats['pf']:.2f}"
            )

    # --------------------------------------------------------
    # Interpretation guidelines
    # --------------------------------------------------------

    print()
    print("Interpretation guidelines:")
    print(
        f"  n < {MIN_SAMPLE_FOR_RULE}:"
        f"        UNTESTED — no interpretation possible."
    )
    print(
        f"  {MIN_SAMPLE_FOR_RULE} <= n < 10:"
        f"  WEAK EVIDENCE — hypothesis only, needs more data."
    )
    print(
        f"  10 <= n < {MIN_SAMPLE_FOR_STRONG}:"
        f" MODERATE EVIDENCE — promising but not conclusive."
    )
    print(
        f"  n >= {MIN_SAMPLE_FOR_STRONG}:"
        f"      STRONG EVIDENCE — consider walk-forward validation."
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    print_section("SAVING RESULTS")

    output_dir = PROJECT_ROOT / "data" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_trades = trades_to_dataframe(trades)
    trades_path = output_dir / "backtest_trades.csv"
    df_trades.to_csv(trades_path, index=False)
    print(f"Saved: {trades_path}")

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_section("FINAL STATUS")

    checks = {
        "Risk plans available":
            len(risk_plans) > 0,

        "Valid risk plans":
            len(valid_risk_plans) > 0,

        "Position sizes created":
            len(position_sizes) == len(valid_risk_plans),

        "All position sizes valid":
            len(valid_position_sizes) == len(position_sizes),

        "Trades simulated":
            len(trades) == len(valid_position_sizes),

        "All trades exited":
            all(
                getattr(t, "exit_reason", None) is not None
                for t in trades
            ),

        "No look-ahead (entry strict)":
            all(
                t.exit_timestamp is None
                or t.exit_timestamp > t.entry_timestamp
                for t in trades
            ),
    }

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status:<6} | {name}")

    print()
    if all(checks.values()):
        print("BACKTEST V1: PASS")
    else:
        print("BACKTEST V1: CHECK")
        print("Failed checks:")
        for name, passed in checks.items():
            if not passed:
                print(f"  - {name}")

    print()
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()