from __future__ import annotations

"""
Forex Backtest — adapted pipeline for 1H data.

Differences from the crypto test:
    - Input:    1H parquet (from Yahoo Finance)
    - Higher:   4H (resampled from 1H)
    - Macro:    1D (resampled from 1H)
    - Micro-MSS runs on 1H (not 5M)

Everything else (liquidity, sweeps, MSS, displacement, OB, FVG,
retest, entry detector, risk manager, position sizer, backtest)
stays the same.

Usage:
    python scripts/test_backtest_forex.py
"""

from pathlib import Path
import sys
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import DATA_DIR, YEAR

import pandas as pd


# ============================================================
# SYMBOLS + CONFIG
# ============================================================

SYMBOL = "EURUSD"     # change to GBPUSD, XAUUSD, etc.

DATA_FILE = (
    DATA_DIR / "raw" / SYMBOL / f"{SYMBOL}_1h_{YEAR}.parquet"
)

ACCOUNT_EQUITY = 10_000.0
RISK_PERCENT = 1.0
LEVERAGE = 1.0

ATR_PERIOD = 14

# Pivot windows (adapted for 1H)
STRUCTURE_PIVOT_WINDOW_4H = 2
LIQUIDITY_PIVOT_WINDOW = 3
INTERNAL_PIVOT_WINDOW_1H = 2
MICRO_PIVOT_WINDOW_1H = 2       # renamed from 5M

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
BACKTEST_MAX_BARS = 5_000


# ============================================================
# STRATEGY IMPORTS
# ============================================================

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
from src.backtest.backtest_engine import run_backtest, calculate_metrics, print_metrics


# ============================================================
# HELPERS
# ============================================================

def print_section(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


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
    """Resample 1H OHLCV to a higher timeframe."""
    df = df_1h.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    out = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return out.reset_index()


# ============================================================
# MAIN
# ============================================================

def main():
    print_section("FOREX BACKTEST — ADAPTED FOR 1H DATA")
    print(f"Symbol:   {SYMBOL}")
    print(f"Year:     {YEAR}")
    print(f"Data:     {DATA_FILE}")

    if not DATA_FILE.exists():
        print(f"\nERROR: data file not found: {DATA_FILE}")
        print("Run: python scripts/download_forex_data.py")
        return

    # --------------------------------------------------------
    # LOAD 1H
    # --------------------------------------------------------

    df_1h = pd.read_parquet(DATA_FILE)
    df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], utc=True)
    df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)

    print(f"1H candles: {len(df_1h):,}")

    # --------------------------------------------------------
    # RESAMPLE TO 4H and 1D
    # --------------------------------------------------------

    df_4h = resample_from_1h(df_1h, "4h")
    df_1d = resample_from_1h(df_1h, "1D")

    print(f"4H candles: {len(df_4h):,}")
    print(f"1D candles: {len(df_1d):,}")

    # --------------------------------------------------------
    # 4H STRUCTURE
    # --------------------------------------------------------

    print_section("4H STRUCTURE")

    swings_4h = detect_pivots(
        df_4h,
        left_bars=STRUCTURE_PIVOT_WINDOW_4H,
        right_bars=STRUCTURE_PIVOT_WINDOW_4H,
        timeframe="4h",
    )
    print(f"4H swings: {len(swings_4h):,}")

    # --------------------------------------------------------
    # 4H LIQUIDITY
    # --------------------------------------------------------

    print_section("4H LIQUIDITY")

    equal_highs = detect_equal_highs(
        df_4h, atr_period=ATR_PERIOD,
        tolerance_atr=EQUAL_TOLERANCE_ATR,
        pivot_window=LIQUIDITY_PIVOT_WINDOW,
    )
    equal_lows = detect_equal_lows(
        df_4h, atr_period=ATR_PERIOD,
        tolerance_atr=EQUAL_TOLERANCE_ATR,
        pivot_window=LIQUIDITY_PIVOT_WINDOW,
    )
    previous_day_zones = detect_previous_day_liquidity(df_4h)
    swing_zones = detect_swing_liquidity(swings_4h)

    liquidity_zones = (
        equal_highs + equal_lows + previous_day_zones + swing_zones
    )
    liquidity_zones = deduplicate_zones(liquidity_zones, price_tolerance=1e-8)

    print(f"Liquidity zones: {len(liquidity_zones):,}")

    # --------------------------------------------------------
    # SWEEPS (on 1H)
    # --------------------------------------------------------

    print_section("SWEEPS")

    sweeps = detect_sweeps(
        df_1h, liquidity_zones,
        atr_period=SWEEP_ATR_PERIOD,
    )
    print(f"Sweeps: {len(sweeps):,}")

    # --------------------------------------------------------
    # 1H INTERNAL STRUCTURE
    # --------------------------------------------------------

    print_section("1H INTERNAL STRUCTURE")

    swings_1h = detect_pivots(
        df_1h,
        left_bars=INTERNAL_PIVOT_WINDOW_1H,
        right_bars=INTERNAL_PIVOT_WINDOW_1H,
        timeframe="1h",
    )
    print(f"1H internal swings: {len(swings_1h):,}")

    # --------------------------------------------------------
    # MSS
    # --------------------------------------------------------

    print_section("1H MSS")

    mss_events = detect_mss(
        df_1h, sweeps, swings_1h,
        max_bars_after_sweep=MSS_MAX_BARS_AFTER_SWEEP,
    )
    print(f"MSS: {len(mss_events):,}")

    # --------------------------------------------------------
    # DISPLACEMENT
    # --------------------------------------------------------

    print_section("DISPLACEMENT")

    displacement_events = detect_displacements(
        df_1h, mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=DISPLACEMENT_MIN_BODY_ATR,
        bullish_clv=DISPLACEMENT_BULLISH_CLV,
        bearish_clv=DISPLACEMENT_BEARISH_CLV,
        max_bars_after_mss=DISPLACEMENT_MAX_BARS_AFTER_MSS,
    )
    print(f"Displacements: {len(displacement_events):,}")

    # --------------------------------------------------------
    # OB / FVG
    # --------------------------------------------------------

    print_section("ORDER BLOCKS")
    order_blocks = detect_order_blocks(df_1h, displacement_events, lookback=OB_LOOKBACK)
    print(f"Order Blocks: {len(order_blocks):,}")

    print_section("FVG")
    fvgs = detect_fvgs(
        df_1h, displacement_events,
        search_bars=FVG_SEARCH_BARS, atr_period=ATR_PERIOD,
    )
    print(f"FVG: {len(fvgs):,}")

    # --------------------------------------------------------
    # RETESTS
    # --------------------------------------------------------

    print_section("RETESTS")

    ob_retests = detect_ob_retests(
        df_1h, order_blocks,
        max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE,
    )
    fvg_retests = detect_fvg_retests(
        df_1h, fvgs,
        max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE,
    )
    all_retests = list(ob_retests) + list(fvg_retests)
    print(f"Total retests: {len(all_retests):,}")

    # --------------------------------------------------------
    # 1H MICRO STRUCTURE (was 5M)
    # --------------------------------------------------------

    print_section("1H MICRO STRUCTURE")

    swings_micro = detect_pivots(
        df_1h,
        left_bars=MICRO_PIVOT_WINDOW_1H,
        right_bars=MICRO_PIVOT_WINDOW_1H,
        timeframe="1h",
    )
    print(f"Micro swings: {len(swings_micro):,}")

    # --------------------------------------------------------
    # MICRO-MSS on 1H
    # --------------------------------------------------------

    print_section("MICRO-MSS (on 1H)")

    df_1h_indexed = df_1h.copy()
    df_1h_indexed["timestamp"] = pd.to_datetime(df_1h_indexed["timestamp"], utc=True)
    df_1h_indexed = df_1h_indexed.set_index("timestamp").sort_index()

    micro_mss_events = detect_micro_mss(
        df_1h_indexed,
        all_retests,
        swings_micro,
        max_bars_after_retest=MICRO_MSS_MAX_BARS_AFTER_RETEST,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )
    micro_mss_events = deduplicate_micro_mss(micro_mss_events)
    print(f"Micro-MSS: {len(micro_mss_events):,}")

    # --------------------------------------------------------
    # ENTRY DETECTOR
    # --------------------------------------------------------

    print_section("ENTRY DETECTOR")

    # Build context
    context = {}

    entry_prices = {}
    for _, row in df_1h.iterrows():
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
            d = normalize_direction(getattr(latest_swing, "direction", ""))
            if d:
                bias_by_timestamp[str(micro_ts)] = d
    context["bias_by_timestamp"] = bias_by_timestamp
    context["risk_reward"] = RISK_REWARD

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

    tradable_entries = [c for c in entry_candidates if getattr(c, "tradable", False)]

    print(f"Entry candidates: {len(entry_candidates):,}")
    print(f"Tradable:         {len(tradable_entries):,}")

    # --------------------------------------------------------
    # RISK MANAGER
    # --------------------------------------------------------

    print_section("RISK MANAGER")

    risk_zones = list(order_blocks) + list(fvgs)

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
        except Exception:
            return None
        if result is None or pd.isna(result):
            return None
        return float(result)

    atr_by_timestamp = {}
    for candidate in tradable_entries:
        c_ts = getattr(candidate, "timestamp", None)
        if c_ts is None:
            continue
        v = _atr_as_of(c_ts)
        if v is not None:
            atr_by_timestamp[c_ts] = v

    risk_plans = build_risk_plans(
        tradable_entries, risk_zones, atr_by_timestamp,
        target_rr=RISK_REWARD,
    )
    valid_risk_plans = get_valid_risk_plans(risk_plans)

    print(f"Risk plans:       {len(risk_plans):,}")
    print(f"Valid risk plans: {len(valid_risk_plans):,}")

    # --------------------------------------------------------
    # POSITION SIZER
    # --------------------------------------------------------

    print_section("POSITION SIZER")

    position_sizes = build_position_sizes(
        valid_risk_plans,
        account_equity=ACCOUNT_EQUITY,
        risk_percent=RISK_PERCENT,
        leverage=LEVERAGE,
    )
    valid_position_sizes = get_valid_position_sizes(position_sizes)

    print(f"Position sizes: {len(position_sizes):,}")
    print(f"Valid:          {len(valid_position_sizes):,}")

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    print_section("BACKTEST")

    trades = run_backtest(
        valid_position_sizes,
        df_1h,
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
            f"Entry {t.entry_price:,.4f} | "
            f"Exit {t.exit_reason or '—':<6} | "
            f"R {r_str:>6} | "
            f"PnL {pnl_str:>10}"
        )

    metrics = calculate_metrics(trades)
    print_metrics(metrics)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print_section("SAVING RESULTS")

    output_dir = PROJECT_ROOT / "data" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict
    rows = [asdict(t) for t in trades]
    df_out = pd.DataFrame(rows)
    out_path = output_dir / f"backtest_forex_{SYMBOL}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()