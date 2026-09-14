from __future__ import annotations

"""
Entry Quality Analysis

For every trade produced by the multi-symbol pipeline, extract
Entry-level features and check their correlation with the trade
outcome (R-multiple).

Features extracted per trade:
    - bars_from_mss_to_entry
    - bars_from_displacement_to_entry
    - bars_from_retest_to_entry
    - bars_from_micro_mss_to_entry  (should be 0)
    - penetration_ratio             (from retest)
    - reference_distance_atr        (from micro-mss)
    - body_atr                      (from displacement)
    - entry_score                   (from entry detector)
    - zone_type                     (ob / fvg)
    - direction                     (bullish / bearish)
    - session_hour                  (hour of entry, UTC)
    - day_of_week

Analysis:
    - Correlation between each numeric feature and R.
    - Mean feature values for winning vs losing trades.
    - Distribution of features per month (to catch regime shifts).

Usage:
    python scripts/analyze_entry_quality.py
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


def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def bars_between(df_5m_indexed, ts_from, ts_to) -> int | None:
    """
    Return the number of 5M bars between two timestamps.
    Both normalized to UTC. Returns None if either is missing.
    """
    if ts_from is None or ts_to is None:
        return None
    try:
        f = to_utc(ts_from)
        t = to_utc(ts_to)
        if f is None or t is None:
            return None
        if t <= f:
            return 0
        # Use searchsorted on the index
        idx = df_5m_indexed.index
        i_from = idx.searchsorted(f, side="left")
        i_to = idx.searchsorted(t, side="left")
        return int(i_to - i_from)
    except Exception:
        return None


# ============================================================
# PER-SYMBOL PIPELINE — with feature extraction
# ============================================================

def process_symbol(symbol: str, year: int, features: list):
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
    liquidity_zones = deduplicate_zones(liquidity_zones, price_tolerance=1e-8)

    sweeps = detect_sweeps(
        df_1h,
        liquidity_zones,
        atr_period=SWEEP_ATR_PERIOD,
    )

    swings_1h = detect_pivots(
        df_1h,
        left_bars=INTERNAL_PIVOT_WINDOW_1H,
        right_bars=INTERNAL_PIVOT_WINDOW_1H,
        timeframe="1h",
    )

    mss_events = detect_mss(
        df_1h,
        sweeps,
        swings_1h,
        max_bars_after_sweep=MSS_MAX_BARS_AFTER_SWEEP,
    )

    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=DISPLACEMENT_MIN_BODY_ATR,
        bullish_clv=DISPLACEMENT_BULLISH_CLV,
        bearish_clv=DISPLACEMENT_BEARISH_CLV,
        max_bars_after_mss=DISPLACEMENT_MAX_BARS_AFTER_MSS,
    )

    order_blocks = detect_order_blocks(df_1h, displacement_events, lookback=OB_LOOKBACK)
    fvgs = detect_fvgs(df_1h, displacement_events, search_bars=FVG_SEARCH_BARS, atr_period=ATR_PERIOD)

    ob_retests = detect_ob_retests(df_1h, order_blocks, max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE)
    fvg_retests = detect_fvg_retests(df_1h, fvgs, max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE)
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
        df_5m_indexed,
        all_retests,
        swings_5m,
        max_bars_after_retest=MICRO_MSS_MAX_BARS_AFTER_RETEST,
        min_reference_distance_atr=MICRO_MSS_MIN_REFERENCE_DISTANCE_ATR,
    )
    micro_mss_events = deduplicate_micro_mss(micro_mss_events)

    # ---- Entry Detector context ----
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
            swing_dir = normalize_direction(getattr(latest_swing, "direction", ""))
            if swing_dir:
                bias_by_timestamp[str(micro_ts)] = swing_dir
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
        debug=False,
    )

    tradable_entries = [c for c in entry_candidates if getattr(c, "tradable", False)]
    if not tradable_entries:
        return

    # ---- ATR as-of ----
    df_1h_indexed = df_1h.copy()
    if "timestamp" in df_1h_indexed.columns:
        df_1h_indexed["timestamp"] = pd.to_datetime(
            df_1h_indexed["timestamp"], utc=True,
        )
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
        tradable_entries,
        risk_zones,
        atr_by_timestamp,
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

    # ---- Build index of all events by timestamp for lookup ----
    retests_by_retest_ts = {}
    for r in all_retests:
        rts = get_attr(r, "retest_timestamp")
        if rts is not None:
            retests_by_retest_ts[timestamp_key(rts)] = r

    mss_by_ts = {}
    for m in mss_events:
        mts = get_attr(m, "timestamp")
        if mts is not None:
            mss_by_ts[timestamp_key(mts)] = m

    disp_by_ts = {}
    for d in displacement_events:
        dts = get_attr(d, "timestamp")
        if dts is not None:
            disp_by_ts[timestamp_key(dts)] = d

    micro_by_ts = {}
    for mm in micro_mss_events:
        mmts = get_attr(mm, "timestamp")
        if mmts is not None:
            micro_by_ts[timestamp_key(mmts)] = mm

    ps_by_ts = {}
    for ps in valid_position_sizes:
        pts = get_attr(ps, "timestamp")
        if pts is not None:
            ps_by_ts[timestamp_key(pts)] = ps

    # ---- Extract features per trade ----
    for t in trades:
        entry_ts = get_attr(t, "entry_timestamp")
        entry_ts_key = timestamp_key(entry_ts)

        # Find the corresponding position size and entry candidate
        ps = ps_by_ts.get(entry_ts_key)
        if ps is None:
            continue

        # The entry candidate is the tradable entry at the same timestamp
        # Find it by timestamp
        candidate = None
        for c in tradable_entries:
            cts = get_attr(c, "timestamp")
            if timestamp_key(cts) == entry_ts_key:
                candidate = c
                break

        if candidate is None:
            continue

        # ---- Look up related events via candidate timestamps ----
        mss_ts = get_attr(candidate, "mss_timestamp")
        disp_ts = get_attr(candidate, "displacement_timestamp")
        retest_ts = get_attr(candidate, "retest_timestamp")
        micro_ts = get_attr(candidate, "micro_mss_timestamp")

        # ---- Compute bars ----
        bars_mss = bars_between(df_5m_indexed, mss_ts, entry_ts)
        bars_disp = bars_between(df_5m_indexed, disp_ts, entry_ts)
        bars_retest = bars_between(df_5m_indexed, retest_ts, entry_ts)

        # ---- Look up events ----
        mss_event = mss_by_ts.get(timestamp_key(mss_ts))
        disp_event = disp_by_ts.get(timestamp_key(disp_ts))
        micro_event = micro_by_ts.get(timestamp_key(micro_ts))

        retest_event = None
        if retest_ts is not None:
            retest_event = retests_by_retest_ts.get(timestamp_key(retest_ts))

        # ---- Extract features ----
        body_atr = safe_float(get_attr(disp_event, "body_atr", "body_over_atr")) if disp_event else None
        ref_dist = safe_float(get_attr(micro_event, "reference_distance_atr")) if micro_event else None
        penetration = safe_float(get_attr(retest_event, "penetration_ratio", "penetration")) if retest_event else None
        close_inside = get_attr(retest_event, "close_inside") if retest_event else None
        close_through = get_attr(retest_event, "close_through") if retest_event else None

        entry_score = safe_float(get_attr(candidate, "score"))
        zone_type = get_attr(candidate, "zone_type", default="unknown")
        direction = normalize_direction(get_attr(candidate, "direction"))

        # Session / day
        ts = to_utc(entry_ts)
        session_hour = ts.hour if ts is not None else None
        day_of_week = ts.dayofweek if ts is not None else None  # 0=Mon, 6=Sun

        r_multiple = safe_float(get_attr(t, "r_multiple"))
        outcome = get_attr(t, "outcome")

        if r_multiple is None:
            continue

        features.append({
            "symbol": symbol,
            "entry_timestamp": entry_ts,
            "direction": direction,
            "zone_type": zone_type,
            "r_multiple": r_multiple,
            "outcome": outcome,
            "bars_mss": bars_mss,
            "bars_disp": bars_disp,
            "bars_retest": bars_retest,
            "body_atr": body_atr,
            "ref_dist": ref_dist,
            "penetration": penetration,
            "close_inside": close_inside,
            "close_through": close_through,
            "entry_score": entry_score,
            "session_hour": session_hour,
            "day_of_week": day_of_week,
        })


# ============================================================
# ANALYSIS
# ============================================================

def analyze(features: list):
    # Filter to closed trades only
    closed = [
        f for f in features
        if f["outcome"] in {"win", "loss"}
        and f["r_multiple"] is not None
    ]

    print()
    print("=" * 72)
    print("ENTRY QUALITY ANALYSIS")
    print("=" * 72)
    print(f"Total features extracted:  {len(features)}")
    print(f"Closed trades for analysis: {len(closed)}")
    print()

    if len(closed) < 5:
        print("Not enough closed trades for meaningful analysis.")
        return

    # ---- Correlations ----
    print("=" * 72)
    print("CORRELATION WITH R-MULTIPLE")
    print("=" * 72)
    print()

    numeric_features = [
        ("bars_mss", "bars_mss"),
        ("bars_disp", "bars_disp"),
        ("bars_retest", "bars_retest"),
        ("body_atr", "body_atr"),
        ("ref_dist", "ref_dist"),
        ("penetration", "penetration"),
        ("entry_score", "entry_score"),
        ("session_hour", "session_hour"),
        ("day_of_week", "day_of_week"),
    ]

    r_series = [f["r_multiple"] for f in closed]

    print(f"  {'Feature':<16} {'n':>5} {'r':>9}  Significance")
    print("-" * 60)

    for label, key in numeric_features:
        xs = []
        ys = []
        for f, r in zip(closed, r_series):
            v = f.get(key)
            if v is None:
                continue
            xs.append(float(v))
            ys.append(float(r))

        if len(xs) < 5:
            print(f"  {label:<16} {len(xs):>5}   (insufficient)")
            continue

        r_val = pearson(xs, ys)
        n = len(xs)

        if n >= 30 and abs(r_val) >= 0.361:
            sig = "STRONG (p<0.05)"
        elif n >= 12 and abs(r_val) >= 0.576:
            sig = "STRONG (p<0.05)"
        elif abs(r_val) >= 0.4:
            sig = "moderate"
        else:
            sig = "weak"

        print(f"  {label:<16} {n:>5} {r_val:>+9.3f}  {sig}")

    # ---- Win vs Loss means ----
    print()
    print("=" * 72)
    print("MEAN FEATURE VALUE — WINS vs LOSSES")
    print("=" * 72)
    print()

    wins = [f for f in closed if f["outcome"] == "win"]
    losses = [f for f in closed if f["outcome"] == "loss"]

    print(f"  {'Feature':<16} {'Wins':>10} {'Losses':>10} {'Diff':>10}")
    print("-" * 60)

    for label, key in numeric_features:
        w_vals = [float(f[key]) for f in wins if f.get(key) is not None]
        l_vals = [float(f[key]) for f in losses if f.get(key) is not None]

        if not w_vals or not l_vals:
            continue

        w_mean = mean(w_vals)
        l_mean = mean(l_vals)
        diff = w_mean - l_mean

        print(
            f"  {label:<16} "
            f"{w_mean:>10.3f} "
            f"{l_mean:>10.3f} "
            f"{diff:>+10.3f}"
        )

    # ---- Zone type breakdown ----
    print()
    print("=" * 72)
    print("ZONE TYPE BREAKDOWN")
    print("=" * 72)
    print()

    ob_trades = [f for f in closed if f["zone_type"] == "ob"]
    fvg_trades = [f for f in closed if f["zone_type"] == "fvg"]
    other = [f for f in closed if f["zone_type"] not in {"ob", "fvg"}]

    for name, group in (("OB", ob_trades), ("FVG", fvg_trades), ("Other", other)):
        if not group:
            continue
        wins_g = sum(1 for f in group if f["outcome"] == "win")
        total_r = sum(f["r_multiple"] for f in group)
        print(
            f"  {name:<8} n={len(group):<4} "
            f"win_rate={wins_g/len(group)*100:>6.2f}%  "
            f"total_r={total_r:+.2f}"
        )

    # ---- Direction breakdown ----
    print()
    print("=" * 72)
    print("DIRECTION BREAKDOWN")
    print("=" * 72)
    print()

    bull = [f for f in closed if f["direction"] == "bullish"]
    bear = [f for f in closed if f["direction"] == "bearish"]

    for name, group in (("Bullish", bull), ("Bearish", bear)):
        if not group:
            continue
        wins_g = sum(1 for f in group if f["outcome"] == "win")
        total_r = sum(f["r_multiple"] for f in group)
        print(
            f"  {name:<8} n={len(group):<4} "
            f"win_rate={wins_g/len(group)*100:>6.2f}%  "
            f"total_r={total_r:+.2f}"
        )

    # ---- Session hour breakdown (4 buckets) ----
    print()
    print("=" * 72)
    print("SESSION HOUR BREAKDOWN (UTC)")
    print("=" * 72)
    print()

    buckets = [
        ("00-06 (Asia)", 0, 6),
        ("06-12 (Europe)", 6, 12),
        ("12-18 (US)", 12, 18),
        ("18-24 (US close)", 18, 24),
    ]

    for name, lo, hi in buckets:
        group = [
            f for f in closed
            if f.get("session_hour") is not None
            and lo <= f["session_hour"] < hi
        ]
        if not group:
            continue
        wins_g = sum(1 for f in group if f["outcome"] == "win")
        total_r = sum(f["r_multiple"] for f in group)
        print(
            f"  {name:<20} n={len(group):<4} "
            f"win_rate={wins_g/len(group)*100:>6.2f}%  "
            f"total_r={total_r:+.2f}"
        )

    # ---- Entry score vs outcome ----
    print()
    print("=" * 72)
    print("ENTRY SCORE DISTRIBUTION")
    print("=" * 72)
    print()

    score_wins = [f["entry_score"] for f in wins if f.get("entry_score") is not None]
    score_losses = [f["entry_score"] for f in losses if f.get("entry_score") is not None]

    if score_wins and score_losses:
        print(f"  Wins   — mean score: {mean(score_wins):.2f}  (n={len(score_wins)})")
        print(f"  Losses — mean score: {mean(score_losses):.2f}  (n={len(score_losses)})")
        print(f"  Diff: {mean(score_wins) - mean(score_losses):+.2f}")

    # ---- Bars to entry ----
    print()
    print("=" * 72)
    print("BARS-TO-ENTRY DISTRIBUTION")
    print("=" * 72)
    print()

    for key, label in (
        ("bars_mss", "MSS -> Entry"),
        ("bars_disp", "Displacement -> Entry"),
        ("bars_retest", "Retest -> Entry"),
    ):
        w_vals = [f[key] for f in wins if f.get(key) is not None]
        l_vals = [f[key] for f in losses if f.get(key) is not None]

        if not w_vals or not l_vals:
            continue

        print(
            f"  {label:<24}  "
            f"Wins: mean={mean(w_vals):>7.1f}  median={median(w_vals):>7.1f}  "
            f"Losses: mean={mean(l_vals):>7.1f}  median={median(l_vals):>7.1f}"
        )

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
    print("ENTRY QUALITY ANALYSIS")
    print("=" * 72)
    print(f"Year:    {year}")
    print(f"Symbols: {len(SYMBOLS)}")
    print()

    features = []

    print("Processing symbols...")
    for symbol in SYMBOLS:
        process_symbol(symbol, year, features)

    analyze(features)


if __name__ == "__main__":
    main()