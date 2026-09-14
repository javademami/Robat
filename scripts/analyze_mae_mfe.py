from __future__ import annotations

"""
MAE / MFE Analysis

For each closed trade, walk candle-by-candle from entry to exit and
record:

  MAE  = Maximum Adverse Excursion  (worst move against the trade)
  MFE  = Maximum Favorable Excursion (best move in favor of the trade)

Both are expressed:
  - as absolute price distance
  - as a multiple of the SL distance (so 1.0 MAE = touched SL)
  - as a multiple of the TP distance (so 1.0 MFE = touched TP)

Then aggregates:
  - For losers: how far did price go in favor before reversing?
  - For winners: how far did price go against before reaching TP?
  - Distribution of time-to-MAE and time-to-MFE (in bars)
  - % of losers where MFE >= 1.0 (i.e. would have hit a smaller TP)
  - % of winners where MAE >= 0.5 (i.e. almost stopped out)

Usage:
    python scripts/analyze_mae_mfe.py
"""

from pathlib import Path
import sys
import math
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import DATA_DIR, SYMBOLS, YEAR

from src.data.storage import load_year
from src.data.aggregator import resample_ohlcv

# Reuse the full multi-symbol pipeline
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
# CONFIG (must match test_multi_symbol.py)
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


# ============================================================
# MAE / MFE PER TRADE
# ============================================================

def compute_mae_mfe(trade, df_5m_indexed):
    """
    Walk candle-by-candle from entry+1 to exit and compute MAE/MFE.

    Returns dict with:
        mae_price, mfe_price    absolute distances
        mae_r, mfe_r            as multiples of risk_per_unit
        mae_bars, mfe_bars      bars until MAE / MFE first reached
        bars_to_exit
    or None if data missing.
    """

    entry_ts = get_attr(trade, "entry_timestamp")
    exit_ts = get_attr(trade, "exit_timestamp")
    entry = safe_float(get_attr(trade, "entry_price"))
    stop = safe_float(get_attr(trade, "stop_loss"))
    tp = safe_float(get_attr(trade, "take_profit"))
    direction = normalize_direction(get_attr(trade, "direction"))

    if entry is None or stop is None or tp is None:
        return None
    if entry_ts is None or exit_ts is None:
        return None

    try:
        import pandas as pd
        f = pd.Timestamp(entry_ts)
        x = pd.Timestamp(exit_ts)
        if f.tzinfo is None:
            f = f.tz_localize("UTC")
        else:
            f = f.tz_convert("UTC")
        if x.tzinfo is None:
            x = x.tz_localize("UTC")
        else:
            x = x.tz_convert("UTC")
    except Exception:
        return None

    idx = df_5m_indexed.index
    i_from = idx.searchsorted(f, side="right")  # strictly after entry
    i_to = idx.searchsorted(x, side="right")

    if i_from >= i_to:
        return None

    window = df_5m_indexed.iloc[i_from:i_to]

    if len(window) == 0:
        return None

    risk_per_unit = abs(entry - stop)
    reward_per_unit = abs(tp - entry)
    if risk_per_unit <= 0 or reward_per_unit <= 0:
        return None

    highs = window["high"].astype(float).values
    lows = window["low"].astype(float).values
    n = len(window)

    if direction == "bullish":
        # MAE = how far price fell against us (entry - lowest low)
        # MFE = how far price rose in favor (highest high - entry)
        max_adverse = 0.0
        max_favorable = 0.0
        mae_bar = None
        mfe_bar = None
        for i in range(n):
            adverse = entry - lows[i]
            favorable = highs[i] - entry
            if adverse > max_adverse:
                max_adverse = adverse
                mae_bar = i + 1
            if favorable > max_favorable:
                max_favorable = favorable
                mfe_bar = i + 1

    elif direction == "bearish":
        # MAE = how far price rose against us (highest high - entry)
        # MFE = how far price fell in favor (entry - lowest low)
        max_adverse = 0.0
        max_favorable = 0.0
        mae_bar = None
        mfe_bar = None
        for i in range(n):
            adverse = highs[i] - entry
            favorable = entry - lows[i]
            if adverse > max_adverse:
                max_adverse = adverse
                mae_bar = i + 1
            if favorable > max_favorable:
                max_favorable = favorable
                mfe_bar = i + 1

    else:
        return None

    # Floor at zero (shouldn't happen, but safe)
    max_adverse = max(max_adverse, 0.0)
    max_favorable = max(max_favorable, 0.0)

    return {
        "mae_price": max_adverse,
        "mfe_price": max_favorable,
        "mae_r": max_adverse / risk_per_unit,
        "mfe_r": max_favorable / risk_per_unit,
        "mfe_rr": max_favorable / reward_per_unit,   # in units of TP
        "mae_bars": mae_bar if mae_bar is not None else n,
        "mfe_bars": mfe_bar if mfe_bar is not None else n,
        "bars_to_exit": n,
    }


# ============================================================
# PER-SYMBOL PIPELINE
# ============================================================

def process_symbol(symbol: str, year: int, all_trades: list):
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

    trades = run_backtest(valid_position_sizes, df_5m, max_bars=BACKTEST_MAX_BARS)

    for t in trades:
        mae_mfe = compute_mae_mfe(t, df_5m_indexed)
        if mae_mfe is None:
            continue

        all_trades.append({
            "symbol": symbol,
            "entry_timestamp": get_attr(t, "entry_timestamp"),
            "direction": normalize_direction(get_attr(t, "direction")),
            "outcome": get_attr(t, "outcome"),
            "exit_reason": get_attr(t, "exit_reason"),
            "r_multiple": safe_float(get_attr(t, "r_multiple")),
            "entry_price": safe_float(get_attr(t, "entry_price")),
            "stop_loss": safe_float(get_attr(t, "stop_loss")),
            "take_profit": safe_float(get_attr(t, "take_profit")),
            "risk_per_unit": abs(
                (safe_float(get_attr(t, "entry_price")) or 0.0)
                - (safe_float(get_attr(t, "stop_loss")) or 0.0)
            ),
            "reward_per_unit": abs(
                (safe_float(get_attr(t, "take_profit")) or 0.0)
                - (safe_float(get_attr(t, "entry_price")) or 0.0)
            ),
            **mae_mfe,
        })


# ============================================================
# ANALYSIS
# ============================================================

def analyze(all_trades: list):
    closed = [
        t for t in all_trades
        if t["outcome"] in {"win", "loss"}
        and t["r_multiple"] is not None
    ]

    print()
    print("=" * 72)
    print("MAE / MFE ANALYSIS")
    print("=" * 72)
    print(f"Total trades with MAE/MFE:  {len(all_trades)}")
    print(f"Closed trades for analysis: {len(closed)}")
    print()

    if len(closed) < 5:
        print("Not enough closed trades for meaningful analysis.")
        return

    wins = [t for t in closed if t["outcome"] == "win"]
    losses = [t for t in closed if t["outcome"] == "loss"]

    # ---- Overall MAE / MFE ----
    print("=" * 72)
    print("MAE / MFE DISTRIBUTION (in R units, 1.0 = SL distance)")
    print("=" * 72)
    print()

    def _stats(values):
        if not values:
            return "(no data)"
        return (
            f"mean={mean(values):.3f}  "
            f"median={median(values):.3f}  "
            f"min={min(values):.3f}  "
            f"max={max(values):.3f}"
        )

    mae_all = [t["mae_r"] for t in closed]
    mfe_all = [t["mfe_r"] for t in closed]

    print(f"  MAE (all):  {_stats(mae_all)}")
    print(f"  MFE (all):  {_stats(mfe_all)}")
    print()

    mae_w = [t["mae_r"] for t in wins]
    mfe_w = [t["mfe_r"] for t in wins]
    mae_l = [t["mae_r"] for t in losses]
    mfe_l = [t["mfe_r"] for t in losses]

    print(f"  Wins   — MAE: {_stats(mae_w)}")
    print(f"  Wins   — MFE: {_stats(mfe_w)}")
    print()
    print(f"  Losses — MAE: {_stats(mae_l)}")
    print(f"  Losses — MFE: {_stats(mfe_l)}")
    print()

    # ---- MFE as multiple of TP ----
    print("=" * 72)
    print("MFE AS MULTIPLE OF TP (mfe_rr)")
    print("=" * 72)
    print()
    print("  1.0 = touched TP; >1.0 = would have hit a farther TP")
    print()

    mfe_rr_w = [t["mfe_rr"] for t in wins]
    mfe_rr_l = [t["mfe_rr"] for t in losses]

    print(f"  Wins   — MFE/TP: {_stats(mfe_rr_w)}")
    print(f"  Losses — MFE/TP: {_stats(mfe_rr_l)}")
    print()

    # ---- % of losers that reached >= 0.5R, 1.0R, 1.5R favorable ----
    print("=" * 72)
    print("FOR LOSERS: how far did price go in favor before reversing?")
    print("=" * 72)
    print()
    if losses:
        for thresh in (0.25, 0.5, 1.0, 1.5, 2.0):
            cnt = sum(1 for t in losses if t["mfe_r"] >= thresh)
            pct = cnt / len(losses) * 100
            print(f"  losers with MFE >= {thresh:.2f}R:  {cnt:>4} / {len(losses)}  ({pct:>5.1f}%)")
    print()

    # ---- % of winners that almost stopped out ----
    print("=" * 72)
    print("FOR WINNERS: how close did price come to SL before reaching TP?")
    print("=" * 72)
    print()
    if wins:
        for thresh in (0.25, 0.5, 0.75, 0.9, 1.0):
            cnt = sum(1 for t in wins if t["mae_r"] >= thresh)
            pct = cnt / len(wins) * 100
            print(f"  winners with MAE >= {thresh:.2f}R:  {cnt:>4} / {len(wins)}  ({pct:>5.1f}%)")
    print()

    # ---- Time to MAE and MFE ----
    print("=" * 72)
    print("TIME-TO-MFE AND TIME-TO-MAE (in 5M bars)")
    print("=" * 72)
    print()

    mfe_bars_w = [t["mfe_bars"] for t in wins]
    mae_bars_w = [t["mae_bars"] for t in wins]
    mfe_bars_l = [t["mfe_bars"] for t in losses]
    mae_bars_l = [t["mae_bars"] for t in losses]

    print(f"  Wins   — bars to MFE: {_stats(mfe_bars_w)}")
    print(f"  Wins   — bars to MAE: {_stats(mae_bars_w)}")
    print(f"  Losses — bars to MFE: {_stats(mfe_bars_l)}")
    print(f"  Losses — bars to MAE: {_stats(mae_bars_l)}")
    print()

    # ---- Interpretation ----
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print()

    if losses:
        # How many losers reached 1.0R favorable (i.e. would have hit 1R TP)?
        reach_1r = sum(1 for t in losses if t["mfe_r"] >= 1.0)
        pct_reach_1r = reach_1r / len(losses) * 100

        # How many losers went >0.8R against immediately?
        hard_sl = sum(1 for t in losses if t["mae_r"] >= 0.95)
        pct_hard_sl = hard_sl / len(losses) * 100

        print(f"  Losers that went >= 1.0R favorable before reversing: "
              f"{reach_1r}/{len(losses)} ({pct_reach_1r:.1f}%)")
        print(f"  Losers that went straight to SL (MAE >= 0.95R): "
              f"{hard_sl}/{len(losses)} ({pct_hard_sl:.1f}%)")
        print()

        if pct_reach_1r >= 50:
            print("  -> MAJORITY of losers went at least 1R in favor before")
            print("     reversing. This suggests TP is too far, or a trailing")
            print("     stop / partial take-profit would help.")
        elif pct_hard_sl >= 60:
            print("  -> MAJORITY of losers went straight to SL. This suggests")
            print("     entry timing or direction is wrong, not SL placement.")
        else:
            print("  -> Mixed pattern. Neither SL-too-tight nor entry-wrong")
            print("     is dominant. Consider testing both hypotheses.")

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
    print("MAE / MFE ANALYSIS")
    print("=" * 72)
    print(f"Year:    {year}")
    print(f"Symbols: {len(SYMBOLS)}")
    print()

    all_trades = []

    print("Processing symbols...")
    for symbol in SYMBOLS:
        process_symbol(symbol, year, all_trades)

    analyze(all_trades)


if __name__ == "__main__":
    main()