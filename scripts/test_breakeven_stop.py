from __future__ import annotations

"""
Breakeven-Stop Backtest

Re-simulates all trades from the multi-symbol pipeline with a
modified exit rule:

  1. Entry at entry_price, initial SL, TP = 2R
  2. When price reaches +breakeven_trigger_R (default 0.5R),
     move SL to entry price (Breakeven).
  3. Exit:
       - at TP if touched first
       - at Breakeven SL if touched after trigger
       - at original SL if touched before trigger

Compares:
  - Baseline (SL/TP fixed) vs Breakeven-Stop
  - Total R, win rate, profit factor, distribution of outcomes

Also does Train/Test split:
  - Jan-Jun = TRAIN
  - Jul-Dec = TEST

Usage:
    python scripts/test_breakeven_stop.py
"""

from pathlib import Path
import sys
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import DATA_DIR, SYMBOLS, YEAR

from src.data.storage import load_year
from src.data.aggregator import resample_ohlcv

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

ACCOUNT_EQUITY = 10_000.0
RISK_PERCENT = 1.0
LEVERAGE = 1.0

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

# Breakeven rule
BREAKEVEN_TRIGGER_R = 0.5    # after +0.5R, move SL to entry

# Train/Test split
SPLIT_DATE = f"{YEAR}-07-01T00:00:00+00:00"


# ============================================================
# HELPERS
# ============================================================

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


def to_utc(value):
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
# CUSTOM TRADE SIMULATOR WITH BREAKEVEN-STOP
# ============================================================

def simulate_breakeven_trade(
    entry_ts,
    entry_price,
    stop_loss,
    take_profit,
    direction,
    df_5m_indexed,
    breakeven_trigger_r=BREAKEVEN_TRIGGER_R,
):
    """
    Simulate one trade with Breakeven-Stop logic.

    Returns dict:
        exit_ts, exit_price, exit_reason, bars_held, r_multiple
    """
    ts_from = to_utc(entry_ts)
    if ts_from is None:
        return None

    idx = df_5m_indexed.index
    i_from = idx.searchsorted(ts_from, side="right")

    if i_from >= len(idx):
        return None

    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    if risk <= 0:
        return None

    # Trigger price for breakeven
    if direction == "bullish":
        trigger_price = entry_price + breakeven_trigger_r * risk
    elif direction == "bearish":
        trigger_price = entry_price - breakeven_trigger_r * risk
    else:
        return None

    # Walk forward
    current_stop = stop_loss
    breakeven_armed = False

    window = df_5m_indexed.iloc[i_from:]
    max_bars = min(len(window), BACKTEST_MAX_BARS)
    window = window.iloc[:max_bars]

    for i in range(len(window)):
        row = window.iloc[i]
        ts = window.index[i]
        high = float(row["high"])
        low = float(row["low"])

        # Conservative: SL first if both touched in same candle
        if direction == "bullish":
            # Check SL (current stop)
            if low <= current_stop:
                r = (current_stop - entry_price) / risk
                return {
                    "exit_ts": ts,
                    "exit_price": current_stop,
                    "exit_reason": "BE-SL" if breakeven_armed else "SL",
                    "bars_held": i + 1,
                    "r_multiple": r,
                }
            # Check TP
            if high >= take_profit:
                r = (take_profit - entry_price) / risk
                return {
                    "exit_ts": ts,
                    "exit_price": take_profit,
                    "exit_reason": "TP",
                    "bars_held": i + 1,
                    "r_multiple": r,
                }
            # Check breakeven trigger
            if not breakeven_armed and high >= trigger_price:
                current_stop = entry_price
                breakeven_armed = True

        elif direction == "bearish":
            if high >= current_stop:
                r = (entry_price - current_stop) / risk
                return {
                    "exit_ts": ts,
                    "exit_price": current_stop,
                    "exit_reason": "BE-SL" if breakeven_armed else "SL",
                    "bars_held": i + 1,
                    "r_multiple": r,
                }
            if low <= take_profit:
                r = (entry_price - take_profit) / risk
                return {
                    "exit_ts": ts,
                    "exit_price": take_profit,
                    "exit_reason": "TP",
                    "bars_held": i + 1,
                    "r_multiple": r,
                }
            if not breakeven_armed and low <= trigger_price:
                current_stop = entry_price
                breakeven_armed = True

    # End of data — still open
    last = window.iloc[-1]
    last_ts = window.index[-1]
    last_close = float(last["close"])

    if direction == "bullish":
        r = (last_close - entry_price) / risk
    else:
        r = (entry_price - last_close) / risk

    return {
        "exit_ts": last_ts,
        "exit_price": last_close,
        "exit_reason": "EOD",
        "bars_held": len(window),
        "r_multiple": r,
    }


# ============================================================
# PIPELINE
# ============================================================

def process_symbol(symbol: str, year: int, results: list):
    print(f"  Processing {symbol} ...")

    try:
        df_5m = load_year(DATA_DIR, symbol, "5m", year)
    except Exception as e:
        print(f"    SKIP: {e}")
        return

    if df_5m is None or len(df_5m) == 0:
        print("    SKIP: empty data")
        return

    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=STRUCTURE_PIVOT_WINDOW_4H,
        right_bars=STRUCTURE_PIVOT_WINDOW_4H,
        timeframe="4h",
    )

    equal_highs = detect_equal_highs(df_4h, atr_period=ATR_PERIOD,
                                      tolerance_atr=EQUAL_TOLERANCE_ATR,
                                      pivot_window=LIQUIDITY_PIVOT_WINDOW)
    equal_lows = detect_equal_lows(df_4h, atr_period=ATR_PERIOD,
                                    tolerance_atr=EQUAL_TOLERANCE_ATR,
                                    pivot_window=LIQUIDITY_PIVOT_WINDOW)
    previous_day_zones = detect_previous_day_liquidity(df_4h)
    swing_zones = detect_swing_liquidity(swings_4h)

    liquidity_zones = (
        equal_highs + equal_lows + previous_day_zones + swing_zones
    )
    liquidity_zones = deduplicate_zones(liquidity_zones, price_tolerance=1e-8)

    sweeps = detect_sweeps(df_1h, liquidity_zones, atr_period=SWEEP_ATR_PERIOD)

    swings_1h = detect_pivots(
        df_1h,
        left_bars=INTERNAL_PIVOT_WINDOW_1H,
        right_bars=INTERNAL_PIVOT_WINDOW_1H,
        timeframe="1h",
    )

    mss_events = detect_mss(df_1h, sweeps, swings_1h,
                             max_bars_after_sweep=MSS_MAX_BARS_AFTER_SWEEP)

    displacement_events = detect_displacements(
        df_1h, mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=DISPLACEMENT_MIN_BODY_ATR,
        bullish_clv=DISPLACEMENT_BULLISH_CLV,
        bearish_clv=DISPLACEMENT_BEARISH_CLV,
        max_bars_after_mss=DISPLACEMENT_MAX_BARS_AFTER_MSS,
    )

    order_blocks = detect_order_blocks(df_1h, displacement_events, lookback=OB_LOOKBACK)
    fvgs = detect_fvgs(df_1h, displacement_events, search_bars=FVG_SEARCH_BARS,
                       atr_period=ATR_PERIOD)

    ob_retests = detect_ob_retests(df_1h, order_blocks,
                                    max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE)
    fvg_retests = detect_fvg_retests(df_1h, fvgs,
                                      max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE)
    all_retests = list(ob_retests) + list(fvg_retests)

    swings_5m = detect_pivots(
        df_5m,
        left_bars=MICRO_PIVOT_WINDOW_5M,
        right_bars=MICRO_PIVOT_WINDOW_5M,
        timeframe="5m",
    )

    import pandas as pd

    df_5m_indexed = df_5m.copy()
    if "timestamp" in df_5m_indexed.columns:
        df_5m_indexed["timestamp"] = pd.to_datetime(df_5m_indexed["timestamp"], utc=True)
        df_5m_indexed = df_5m_indexed.set_index("timestamp").sort_index()
    elif not isinstance(df_5m_indexed.index, pd.DatetimeIndex):
        df_5m_indexed.index = pd.to_datetime(df_5m_indexed.index, utc=True)

    micro_mss_events = detect_micro_mss(
        df_5m_indexed, all_retests, swings_5m,
        max_bars_after_retest=MICRO_MSS_MAX_BARS_AFTER_RETEST,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )
    micro_mss_events = deduplicate_micro_mss(micro_mss_events)

    context = {}
    entry_prices = {}
    for _, row in df_5m.iterrows():
        ts = timestamp_key(row.get("timestamp"))
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
    swings_4h_sorted = sorted(swings_4h, key=lambda s: str(getattr(s, "timestamp", "")))
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
            swing_dir = normalize_direction(getattr(latest_swing, "direction", ""))
            if swing_dir:
                bias_by_timestamp[str(micro_ts)] = swing_dir
    context["bias_by_timestamp"] = bias_by_timestamp
    context["risk_reward"] = RISK_REWARD

    entry_candidates = detect_entry_candidates(
        micro_mss_events, all_retests, mss_events, displacement_events,
        order_blocks, fvgs, context=context, debug=False,
    )

    tradable_entries = [c for c in entry_candidates if getattr(c, "tradable", False)]
    if not tradable_entries:
        return

    df_1h_indexed = df_1h.copy()
    if "timestamp" in df_1h_indexed.columns:
        df_1h_indexed["timestamp"] = pd.to_datetime(df_1h_indexed["timestamp"], utc=True)
        df_1h_indexed = df_1h_indexed.set_index("timestamp").sort_index()

    atr_1h_series = calculate_atr(df_1h_indexed, period=ATR_PERIOD).sort_index()

    def _atr_as_of(value):
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            result = atr_1h_series.asof(ts)
        except (KeyError, ValueError, TypeError):
            return None
        if result is None or pd.isna(result):
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

    risk_zones = list(order_blocks) + list(fvgs)

    risk_plans = build_risk_plans(
        tradable_entries, risk_zones, atr_by_timestamp,
        target_rr=RISK_REWARD,
    )
    valid_risk_plans = get_valid_risk_plans(risk_plans)

    position_sizes = build_position_sizes(
        valid_risk_plans,
        account_equity=ACCOUNT_EQUITY,
        risk_percent=RISK_PERCENT,
        leverage=LEVERAGE,
    )
    valid_position_sizes = get_valid_position_sizes(position_sizes)

    if not valid_position_sizes:
        return

    # Baseline (original backtest)
    baseline_trades = run_backtest(
        valid_position_sizes, df_5m, max_bars=BACKTEST_MAX_BARS,
    )

    # Breakeven simulation
    for ps, baseline in zip(valid_position_sizes, baseline_trades):
        entry_ts = get_attr(ps, "timestamp")
        entry_price = safe_float(get_attr(ps, "entry_price"))
        stop_loss = safe_float(get_attr(ps, "stop_loss"))
        take_profit = safe_float(get_attr(ps, "take_profit"))
        direction = normalize_direction(get_attr(ps, "direction"))

        if None in (entry_price, stop_loss, take_profit):
            continue

        be_result = simulate_breakeven_trade(
            entry_ts=entry_ts,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
            df_5m_indexed=df_5m_indexed,
            breakeven_trigger_r=BREAKEVEN_TRIGGER_R,
        )

        if be_result is None:
            continue

        results.append({
            "symbol": symbol,
            "entry_timestamp": entry_ts,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "baseline_outcome": get_attr(baseline, "outcome"),
            "baseline_r": safe_float(get_attr(baseline, "r_multiple")),
            "be_outcome": (
                "win" if be_result["r_multiple"] > 0
                else "loss" if be_result["r_multiple"] < 0
                else "be"
            ),
            "be_r": be_result["r_multiple"],
            "be_exit_reason": be_result["exit_reason"],
            "be_bars_held": be_result["bars_held"],
        })


# ============================================================
# ANALYSIS
# ============================================================

def summarize(trades, label, r_key="be_r", outcome_key="be_outcome"):
    closed = [
        t for t in trades
        if t[outcome_key] in {"win", "loss", "be"}
        and t[r_key] is not None
    ]

    if not closed:
        print(f"{label}: no trades.")
        return

    r_values = [t[r_key] for t in closed]
    wins = sum(1 for t in closed if t[r_key] > 0)
    losses = sum(1 for t in closed if t[r_key] < 0)
    bes = sum(1 for t in closed if t[r_key] == 0)

    gross_profit = sum(r for r in r_values if r > 0)
    gross_loss = abs(sum(r for r in r_values if r < 0))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    print(f"{label}:")
    print(f"  Total trades:  {len(closed)}")
    print(f"  Wins:          {wins}")
    print(f"  Losses:        {losses}")
    print(f"  Breakeven:     {bes}")
    print(f"  Win rate:      {wins / len(closed) * 100:.2f}%")
    print(f"  Total R:       {sum(r_values):+.2f}")
    print(f"  Profit factor: {pf:.4f}")
    print(f"  Avg R:         {mean(r_values):+.4f}")
    print(f"  Median R:      {median(r_values):+.4f}")
    print()


def analyze(results: list):
    print()
    print("=" * 72)
    print("BREAKEVEN-STOP ANALYSIS")
    print("=" * 72)
    print(f"Trigger: move SL to entry after +{BREAKEVEN_TRIGGER_R}R")
    print(f"Total trades: {len(results)}")
    print()

    # ---- Overall comparison ----
    print("=" * 72)
    print("BASELINE vs BREAKEVEN-STOP")
    print("=" * 72)
    print()

    summarize(results, "BASELINE", r_key="baseline_r",
              outcome_key="baseline_outcome")
    summarize(results, "BREAKEVEN-STOP", r_key="be_r",
              outcome_key="be_outcome")

    # ---- Exit reason distribution ----
    print("=" * 72)
    print("EXIT REASON DISTRIBUTION (Breakeven-Stop)")
    print("=" * 72)
    print()

    from collections import Counter
    reasons = Counter(t["be_exit_reason"] for t in results)
    for reason, cnt in reasons.most_common():
        print(f"  {reason:<8} {cnt:>4}  ({cnt / len(results) * 100:.1f}%)")

    # ---- Train/Test split ----
    print()
    print("=" * 72)
    print("TRAIN / TEST SPLIT")
    print("=" * 72)
    print()

    split_ts = to_utc(SPLIT_DATE)

    train = [
        t for t in results
        if to_utc(t["entry_timestamp"]) is not None
        and to_utc(t["entry_timestamp"]) < split_ts
    ]
    test = [
        t for t in results
        if to_utc(t["entry_timestamp"]) is not None
        and to_utc(t["entry_timestamp"]) >= split_ts
    ]

    print(f"TRAIN trades (Jan-Jun): {len(train)}")
    print(f"TEST trades  (Jul-Dec): {len(test)}")
    print()

    summarize(train, "TRAIN — BASELINE", r_key="baseline_r",
              outcome_key="baseline_outcome")
    summarize(train, "TRAIN — BREAKEVEN-STOP", r_key="be_r",
              outcome_key="be_outcome")
    summarize(test, "TEST — BASELINE", r_key="baseline_r",
              outcome_key="baseline_outcome")
    summarize(test, "TEST — BREAKEVEN-STOP", r_key="be_r",
              outcome_key="be_outcome")

    # ---- Improvement by direction ----
    print("=" * 72)
    print("IMPROVEMENT: BREAKEVEN vs BASELINE")
    print("=" * 72)
    print()

    baseline_total = sum(t["baseline_r"] for t in results if t["baseline_r"] is not None)
    be_total = sum(t["be_r"] for t in results if t["be_r"] is not None)
    diff = be_total - baseline_total

    print(f"  Baseline total R:      {baseline_total:+.2f}")
    print(f"  Breakeven total R:     {be_total:+.2f}")
    print(f"  Difference:            {diff:+.2f}")
    print()

    if diff > 0:
        print("  -> Breakeven-Stop IMPROVES the strategy")
    elif diff < 0:
        print("  -> Breakeven-Stop WORSENS the strategy")
    else:
        print("  -> No change")

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():
    year = YEAR

    print("=" * 72)
    print("BREAKEVEN-STOP BACKTEST")
    print("=" * 72)
    print(f"Year:    {year}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Trigger: +{BREAKEVEN_TRIGGER_R}R -> move SL to entry")
    print()

    results = []

    print("Processing symbols...")
    for symbol in SYMBOLS:
        process_symbol(symbol, year, results)

    analyze(results)


if __name__ == "__main__":
    main()