from __future__ import annotations

"""
Multi-Symbol Backtest

Runs the full pipeline (5M → ... → Backtest) for every symbol
in src.config.SYMBOLS, then aggregates all trades into one
portfolio and performs:

  1. Per-symbol summary table.
  2. Portfolio-level metrics.
  3. Train/Test split (SPLIT_DATE).
  4. Hypothesis Check (direction filter) with n>=5 threshold.

Symbols without cached data are skipped gracefully.

Usage:
    python scripts/test_multi_symbol.py
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

from src.config import DATA_DIR, SYMBOLS, YEAR


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

BACKTEST_MAX_BARS = 20_160

SPLIT_DATE = f"{YEAR}-07-01T00:00:00+00:00"

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


def timestamp_key(value):
    if value is None:
        return None
    try:
        import pandas as pd
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return str(ts)
    except Exception:
        return str(value)


# ============================================================
# PER-SYMBOL PIPELINE
# ============================================================

def run_pipeline_for_symbol(symbol: str) -> list:
    """
    Runs the full pipeline for a single symbol and returns
    a list of Trade objects (with .symbol injected).

    Returns [] if data is missing or any stage fails.
    """

    print()
    print("-" * 72)
    print(f"SYMBOL: {symbol}")
    print("-" * 72)

    # --------------------------------------------------------
    # Load 5M
    # --------------------------------------------------------

    try:
        df_5m = load_year(DATA_DIR, symbol, "5m", YEAR)
    except Exception as e:
        print(f"  SKIP: data not found ({e})")
        return []

    print(f"  5M candles: {len(df_5m):,}")

    if len(df_5m) == 0:
        print("  SKIP: empty data")
        return []

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")

    # --------------------------------------------------------
    # 4H Structure
    # --------------------------------------------------------

    swings_4h = detect_pivots(
        df_4h,
        left_bars=STRUCTURE_PIVOT_WINDOW_4H,
        right_bars=STRUCTURE_PIVOT_WINDOW_4H,
        timeframe="4h",
    )

    # --------------------------------------------------------
    # 4H Liquidity
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sweeps
    # --------------------------------------------------------

    sweeps = detect_sweeps(
        df_1h,
        liquidity_zones,
        atr_period=SWEEP_ATR_PERIOD,
    )

    # --------------------------------------------------------
    # 1H Structure
    # --------------------------------------------------------

    swings_1h = detect_pivots(
        df_1h,
        left_bars=INTERNAL_PIVOT_WINDOW_1H,
        right_bars=INTERNAL_PIVOT_WINDOW_1H,
        timeframe="1h",
    )

    # --------------------------------------------------------
    # MSS
    # --------------------------------------------------------

    mss_events = detect_mss(
        df_1h,
        sweeps,
        swings_1h,
        max_bars_after_sweep=MSS_MAX_BARS_AFTER_SWEEP,
    )

    # --------------------------------------------------------
    # Displacement
    # --------------------------------------------------------

    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=DISPLACEMENT_MIN_BODY_ATR,
        bullish_clv=DISPLACEMENT_BULLISH_CLV,
        bearish_clv=DISPLACEMENT_BEARISH_CLV,
        max_bars_after_mss=DISPLACEMENT_MAX_BARS_AFTER_MSS,
    )

    # --------------------------------------------------------
    # OB / FVG
    # --------------------------------------------------------

    order_blocks = detect_order_blocks(
        df_1h,
        displacement_events,
        lookback=OB_LOOKBACK,
    )

    fvgs = detect_fvgs(
        df_1h,
        displacement_events,
        search_bars=FVG_SEARCH_BARS,
        atr_period=ATR_PERIOD,
    )

    # --------------------------------------------------------
    # Retests
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5M Micro Structure
    # --------------------------------------------------------

    swings_5m = detect_pivots(
        df_5m,
        left_bars=MICRO_PIVOT_WINDOW_5M,
        right_bars=MICRO_PIVOT_WINDOW_5M,
        timeframe="5m",
    )

    # --------------------------------------------------------
    # Micro-MSS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Entry Detector context
    # --------------------------------------------------------

    context = {}

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

    sweeps_by_mss_timestamp = {}
    for sweep in sweeps:
        sweep_ts = get_attr(sweep, "timestamp", "sweep_timestamp")
        if sweep_ts is None:
            continue
        key = timestamp_key(sweep_ts)
        if key is not None:
            sweeps_by_mss_timestamp[key] = sweep

    context["sweeps_by_mss_timestamp"] = sweeps_by_mss_timestamp

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
    context["risk_reward"] = RISK_REWARD

    # --------------------------------------------------------
    # Entry Detector
    # --------------------------------------------------------

    entry_candidates = detect_entry_candidates(
        micro_mss_events,
        all_retests,
        mss_events,
        displacement_events,
        order_blocks,
        fvgs,
        context=context,
        debug=False,
    )

    tradable_entries = [
        c for c in entry_candidates
        if getattr(c, "tradable", False)
    ]

    print(
        f"  Micro-MSS: {len(micro_mss_events):3d}  "
        f"Tradable: {len(tradable_entries):3d}"
    )

    if not tradable_entries:
        return []

    # --------------------------------------------------------
    # ATR as-of map
    # --------------------------------------------------------

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
    # Risk Manager
    # --------------------------------------------------------

    risk_zones = list(order_blocks) + list(fvgs)

    risk_plans = build_risk_plans(
        tradable_entries,
        risk_zones,
        atr_by_timestamp,
        target_rr=RISK_REWARD,
    )

    valid_risk_plans = get_valid_risk_plans(risk_plans)

    # --------------------------------------------------------
    # Position Sizer
    # --------------------------------------------------------

    position_sizes = build_position_sizes(
        valid_risk_plans,
        account_equity=ACCOUNT_EQUITY,
        risk_percent=RISK_PERCENT,
        leverage=LEVERAGE,
    )

    valid_position_sizes = get_valid_position_sizes(position_sizes)

    if not valid_position_sizes:
        return []

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    trades = run_backtest(
        valid_position_sizes,
        df_5m,
        max_bars=BACKTEST_MAX_BARS,
    )

    # Inject symbol for reporting
    for t in trades:
        try:
            t.symbol = symbol
        except Exception:
            pass

    print(f"  Trades simulated: {len(trades)}")

    return trades


# ============================================================
# PORTFOLIO METRICS (reuses backtest.metrics)
# ============================================================

def _summary_stats(trades):
    closed = [
        t for t in trades
        if getattr(t, "outcome", None) in {"win", "loss"}
    ]
    if not closed:
        return {"n": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_r": 0.0, "pf": 0.0}

    wins = sum(1 for t in closed if t.outcome == "win")
    losses = sum(1 for t in closed if t.outcome == "loss")
    total_r = sum(float(t.r_multiple or 0.0) for t in closed)
    gross_profit = sum(
        float(t.pnl or 0.0) for t in closed if t.pnl and t.pnl > 0
    )
    gross_loss = abs(sum(
        float(t.pnl or 0.0) for t in closed if t.pnl and t.pnl < 0
    ))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "n": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(closed),
        "total_r": total_r,
        "pf": pf,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_section("MULTI-SYMBOL BACKTEST")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Year:         {YEAR}")
    print(f"Symbols:      {len(SYMBOLS)}")
    for s in SYMBOLS:
        print(f"  - {s}")

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
    # RUN PIPELINE FOR EACH SYMBOL
    # ========================================================

    print_section("RUNNING PIPELINE PER SYMBOL")

    all_trades = []
    per_symbol = {}

    for symbol in SYMBOLS:
        trades = run_pipeline_for_symbol(symbol)
        per_symbol[symbol] = trades
        all_trades.extend(trades)

    # ========================================================
    # PER-SYMBOL SUMMARY
    # ========================================================

    print_section("PER-SYMBOL SUMMARY")

    header = (
        f"{'Symbol':<12} "
        f"{'Trades':>7} "
        f"{'Wins':>6} "
        f"{'Losses':>7} "
        f"{'WinRate':>9} "
        f"{'TotalR':>9} "
        f"{'PF':>7}"
    )
    print(header)
    print("-" * len(header))

    total_closed = 0
    total_wins = 0
    total_losses = 0
    total_r = 0.0

    for symbol in SYMBOLS:
        trades = per_symbol[symbol]
        stats = _summary_stats(trades)

        pf_str = (
            f"{stats['pf']:.2f}"
            if stats["pf"] != float("inf")
            else "inf"
        )

        print(
            f"{symbol:<12} "
            f"{stats['n']:>7} "
            f"{stats['wins']:>6} "
            f"{stats['losses']:>7} "
            f"{stats['win_rate']*100:>8.2f}% "
            f"{stats['total_r']:>+9.2f} "
            f"{pf_str:>7}"
        )

        total_closed += stats["n"]
        total_wins += stats["wins"]
        total_losses += stats["losses"]
        total_r += stats["total_r"]

    print("-" * len(header))

    portfolio_wr = (total_wins / total_closed) if total_closed else 0.0

    print(
        f"{'PORTFOLIO':<12} "
        f"{total_closed:>7} "
        f"{total_wins:>6} "
        f"{total_losses:>7} "
        f"{portfolio_wr*100:>8.2f}% "
        f"{total_r:>+9.2f}"
    )

    # ========================================================
    # PORTFOLIO METRICS
    # ========================================================

    print_section("PORTFOLIO METRICS (all symbols)")

    if all_trades:
        portfolio_metrics = calculate_metrics(all_trades)
        print_metrics(portfolio_metrics)
    else:
        print("No trades across all symbols.")

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print_section("TRAIN / TEST SPLIT (portfolio)")

    print(
        f"Split date: {SPLIT_DATE} "
        f"(Jan-Jun = TRAIN, Jul-Dec = TEST)"
    )

    split_ts = parse_timestamp(SPLIT_DATE)

    train_trades = []
    test_trades = []

    for t in all_trades:
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
        print_metrics(calculate_metrics(train_trades))
    else:
        print("No trades in TRAIN period.")

    print()
    print("--- TEST (Jul-Dec) ---")
    if test_trades:
        print_metrics(calculate_metrics(test_trades))
    else:
        print("No trades in TEST period.")

    # ========================================================
    # HYPOTHESIS CHECK
    # ========================================================

    print_section(
        "HYPOTHESIS CHECK: direction filter "
        "(rule frozen from TRAIN, checked on TEST)"
    )

    print()
    print("Statistical disclaimer:")
    print(f"  Minimum sample for meaningful rule: n >= {MIN_SAMPLE_FOR_RULE}")
    print(f"  Minimum sample for strong evidence: n >= {MIN_SAMPLE_FOR_STRONG}")
    print(f"  With fewer samples, results are UNTESTED (not printed).")
    print()

    train_bull = [
        t for t in train_trades
        if normalize_direction(t.direction) == "bullish"
    ]
    train_bear = [
        t for t in train_trades
        if normalize_direction(t.direction) == "bearish"
    ]

    bull_stats = _summary_stats(train_bull)
    bear_stats = _summary_stats(train_bear)

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

    elif bull_stats["n"] < MIN_SAMPLE_FOR_RULE:
        print()
        print(
            "Frozen rule: keep bearish_only "
            f"(bullish subset has n={bull_stats['n']})."
        )
        chosen_direction = "bearish"

    elif bear_stats["n"] < MIN_SAMPLE_FOR_RULE:
        print()
        print(
            "Frozen rule: keep bullish_only "
            f"(bearish subset has n={bear_stats['n']})."
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

    print()
    print("Applying frozen rule to TEST (out-of-sample, NOT re-tuned):")

    if chosen_direction is None:
        print("  SKIPPED — no frozen rule available.")
    else:
        test_subset = [
            t for t in test_trades
            if normalize_direction(t.direction) == chosen_direction
        ]
        test_stats = _summary_stats(test_subset)

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

    # ========================================================
    # WALK-FORWARD ANALYSIS (4 quarters)
    #
    # The 2-way TRAIN/TEST split above answers "does a rule
    # found in H1 survive in H2". It cannot tell us whether
    # performance is consistently bad/good across the whole
    # year, or whether it swings wildly quarter to quarter
    # (which would point to regime noise rather than a stable
    # signal or a stable absence of one).
    #
    # This section:
    #   1. Reports each calendar quarter's stats independently
    #      (no lookahead issue -- purely descriptive).
    #   2. Runs a ROLLING hypothesis check: for each quarter
    #      Q(i), derive a direction rule using ONLY quarters
    #      1..(i-1) (everything known so far), freeze it, and
    #      test it purely on Q(i) -- the one quarter that rule
    #      has never seen. This is repeated forward through the
    #      year, so we get multiple independent out-of-sample
    #      checks instead of just one.
    # ========================================================

    print_section("WALK-FORWARD ANALYSIS (4 quarters)")

    quarter_bounds = [
        (
            f"Q1",
            parse_timestamp(f"{YEAR}-01-01T00:00:00+00:00"),
            parse_timestamp(f"{YEAR}-04-01T00:00:00+00:00"),
        ),
        (
            f"Q2",
            parse_timestamp(f"{YEAR}-04-01T00:00:00+00:00"),
            parse_timestamp(f"{YEAR}-07-01T00:00:00+00:00"),
        ),
        (
            f"Q3",
            parse_timestamp(f"{YEAR}-07-01T00:00:00+00:00"),
            parse_timestamp(f"{YEAR}-10-01T00:00:00+00:00"),
        ),
        (
            f"Q4",
            parse_timestamp(f"{YEAR}-10-01T00:00:00+00:00"),
            parse_timestamp(f"{YEAR + 1}-01-01T00:00:00+00:00"),
        ),
    ]

    quarter_trades = {}

    for label, start, end in quarter_bounds:

        bucket = []

        for t in all_trades:
            ts = parse_timestamp(t.entry_timestamp)
            if ts is None:
                continue
            if start <= ts < end:
                bucket.append(t)

        quarter_trades[label] = bucket

    # --------------------------------------------------------
    # 1. Independent per-quarter stats
    # --------------------------------------------------------

    print("\nPer-quarter results (independent, descriptive only):\n")

    q_header = (
        f"{'Quarter':<8} "
        f"{'n':>4} "
        f"{'WinRate':>9} "
        f"{'TotalR':>9} "
        f"{'PF':>7}"
    )
    print(q_header)
    print("-" * len(q_header))

    quarter_stats = {}

    for label, _, _ in quarter_bounds:

        stats = _summary_stats(quarter_trades[label])
        quarter_stats[label] = stats

        pf_str = (
            f"{stats['pf']:.2f}"
            if stats["pf"] != float("inf")
            else "inf"
        )

        if stats["n"] < MIN_SAMPLE_FOR_RULE:
            print(
                f"{label:<8} "
                f"{stats['n']:>4} "
                f"{'UNTESTED':>9} "
                f"{'—':>9} "
                f"{'—':>7}"
            )
        else:
            print(
                f"{label:<8} "
                f"{stats['n']:>4} "
                f"{stats['win_rate']*100:>8.2f}% "
                f"{stats['total_r']:>+9.2f} "
                f"{pf_str:>7}"
            )

    # --------------------------------------------------------
    # Stability read: is performance similar across quarters,
    # or does it swing between clearly profitable and clearly
    # unprofitable? This is descriptive, not a statistical test
    # -- with n this small per quarter, treat it as a signal to
    # investigate further, not a verdict.
    # --------------------------------------------------------

    tested_quarters = [
        (label, quarter_stats[label])
        for label, _, _ in quarter_bounds
        if quarter_stats[label]["n"] >= MIN_SAMPLE_FOR_RULE
    ]

    if len(tested_quarters) >= 2:

        r_values = [s["total_r"] for _, s in tested_quarters]
        positive_quarters = sum(1 for r in r_values if r > 0)
        negative_quarters = sum(1 for r in r_values if r < 0)

        print(
            f"\nQuarters with enough data to read: "
            f"{len(tested_quarters)} / 4"
        )
        print(
            f"  Profitable quarters: {positive_quarters}"
        )
        print(
            f"  Losing quarters:     {negative_quarters}"
        )

        if positive_quarters == len(tested_quarters):
            print(
                "  -> Consistently profitable across every "
                "tested quarter. Encouraging, still confirm "
                "with more data before trusting it."
            )
        elif negative_quarters == len(tested_quarters):
            print(
                "  -> Consistently unprofitable across every "
                "tested quarter. This points to the SIGNAL "
                "itself, not a temporary regime -- redesigning "
                "entry logic is more promising than further "
                "filtering."
            )
        else:
            print(
                "  -> Mixed: some quarters profitable, some not. "
                "This is consistent with regime-dependence "
                "(performance tracks market conditions) rather "
                "than either a stable edge or a broken signal."
            )
    else:
        print(
            f"\nOnly {len(tested_quarters)} quarter(s) have "
            f"n >= {MIN_SAMPLE_FOR_RULE}. Not enough tested "
            f"quarters to read a stability pattern."
        )

    # --------------------------------------------------------
    # 2. Rolling hypothesis check
    #
    # For each quarter Q(i) starting at Q2, derive a direction
    # rule from ALL trades strictly before Q(i) (i.e. Q1..Q(i-1)
    # combined), freeze it, and test only on Q(i). This gives up
    # to 3 independent out-of-sample checks (Q2 tested against
    # Q1, Q3 against Q1+Q2, Q4 against Q1+Q2+Q3) instead of the
    # single TRAIN/TEST check above.
    # --------------------------------------------------------

    print("\n" + "-" * 72)
    print("Rolling hypothesis check (direction rule, quarter by quarter)")
    print("-" * 72)

    labels_in_order = [label for label, _, _ in quarter_bounds]

    rolling_results = []

    for i in range(1, len(labels_in_order)):

        test_label = labels_in_order[i]
        history_labels = labels_in_order[:i]

        history_trades = []
        for h_label in history_labels:
            history_trades.extend(quarter_trades[h_label])

        h_bull = [
            t for t in history_trades
            if normalize_direction(t.direction) == "bullish"
        ]
        h_bear = [
            t for t in history_trades
            if normalize_direction(t.direction) == "bearish"
        ]

        h_bull_stats = _summary_stats(h_bull)
        h_bear_stats = _summary_stats(h_bear)

        print(
            f"\n{'+'.join(history_labels)} -> {test_label}:"
        )

        if (
            h_bull_stats["n"] < MIN_SAMPLE_FOR_RULE
            and h_bear_stats["n"] < MIN_SAMPLE_FOR_RULE
        ):
            print(
                f"  History too small "
                f"(bullish n={h_bull_stats['n']}, "
                f"bearish n={h_bear_stats['n']}) "
                f"-- no rule derived."
            )
            continue

        if h_bull_stats["total_r"] >= h_bear_stats["total_r"]:
            rule_direction = "bullish"
        else:
            rule_direction = "bearish"

        print(
            f"  Rule from history: keep {rule_direction}_only "
            f"(bullish R={h_bull_stats['total_r']:+.2f} n={h_bull_stats['n']}, "
            f"bearish R={h_bear_stats['total_r']:+.2f} n={h_bear_stats['n']})"
        )

        test_subset = [
            t for t in quarter_trades[test_label]
            if normalize_direction(t.direction) == rule_direction
        ]
        test_stats = _summary_stats(test_subset)

        if test_stats["n"] < MIN_SAMPLE_FOR_RULE:
            print(
                f"  Applied to {test_label}: UNTESTED "
                f"(n={test_stats['n']} < {MIN_SAMPLE_FOR_RULE})"
            )
            continue

        outcome = (
            "HOLDS"
            if test_stats["total_r"] > 0
            else "FAILS"
        )

        print(
            f"  Applied to {test_label}: n={test_stats['n']}  "
            f"win_rate={test_stats['win_rate']:.2%}  "
            f"total_r={test_stats['total_r']:+.2f}  "
            f"-> rule {outcome}"
        )

        rolling_results.append(outcome)

    if rolling_results:
        holds = rolling_results.count("HOLDS")
        fails = rolling_results.count("FAILS")
        print(
            f"\nRolling summary: {holds} HOLDS / {fails} FAILS "
            f"out of {len(rolling_results)} testable transitions."
        )
        if fails > holds:
            print(
                "  -> The direction rule does not survive "
                "moving forward through the year more often "
                "than it does. Weak basis for a permanent "
                "direction filter."
            )
        elif holds > fails:
            print(
                "  -> The direction rule survived more often "
                "than it failed. Still only a few independent "
                "checks -- treat as suggestive, not proof."
            )
        else:
            print("  -> Inconclusive (tied).")
    else:
        print(
            "\nNo quarter transition had enough data on both "
            "sides to test."
        )

    # ========================================================
    # INTERPRETATION GUIDELINES
    # ========================================================

    print()
    print("Interpretation guidelines:")
    print(f"  n < {MIN_SAMPLE_FOR_RULE}:        UNTESTED")
    print(f"  {MIN_SAMPLE_FOR_RULE} <= n < 10:  WEAK EVIDENCE")
    print(f"  10 <= n < {MIN_SAMPLE_FOR_STRONG}: MODERATE EVIDENCE")
    print(f"  n >= {MIN_SAMPLE_FOR_STRONG}:      STRONG EVIDENCE")

    # ========================================================
    # SAVE CSV
    # ========================================================

    print_section("SAVING RESULTS")

    output_dir = PROJECT_ROOT / "data" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_trades:
        df_trades = trades_to_dataframe(all_trades)
        trades_path = output_dir / "multi_symbol_trades.csv"
        df_trades.to_csv(trades_path, index=False)
        print(f"Saved: {trades_path}")

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_section("FINAL STATUS")

    checks = {
        "At least 1 symbol produced trades":
            len(all_trades) > 0,

        "Portfolio has 100+ trades":
            len(all_trades) >= 100,

        "Portfolio has 30+ closed trades":
            len([
                t for t in all_trades
                if getattr(t, "outcome", None) in {"win", "loss"}
            ]) >= 30,

        "TRAIN has >= 5 trades":
            len(train_trades) >= 5,

        "TEST has >= 5 trades":
            len(test_trades) >= 5,
    }

    for name, passed in checks.items():
        status = "PASS" if passed else "WARN"
        print(f"{status:<6} | {name}")

    print()
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()