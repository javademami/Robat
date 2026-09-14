from __future__ import annotations

"""
Quarterly Analysis — Market vs Signal vs Performance

For each quarter (Q1-Q4 of YEAR), compute:

  Market metrics:
    - mean ATR_1H
    - mean ATR_4H
    - mean daily range (high-low)/close
    - mean volume

  Signal metrics (per symbol, aggregated):
    - sweeps, MSS, displacements
    - OB, FVG
    - retests
    - micro-MSS
    - tradable entries
    - trades

  Quality metrics:
    - mean reference_distance_atr (Micro-MSS)
    - mean body_atr (Displacement)

  Performance metrics:
    - trades, wins, losses, win_rate, total_r

Then:
  - Prints a per-quarter comparison table.
  - Computes Pearson correlation between each feature and total_r.

Usage:
    python scripts/analyze_quarters.py
"""

from pathlib import Path
import sys
import math
from statistics import mean


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


def quarter_of(year: int, ts):
    """Return 1, 2, 3, or 4 for a given timestamp, else None."""
    if ts is None:
        return None
    try:
        import pandas as pd
        t = pd.Timestamp(ts)
        if t.year != year:
            return None
        return (t.month - 1) // 3 + 1
    except Exception:
        return None


def pearson(xs, ys) -> float:
    """Pearson correlation. Returns 0.0 if undefined."""
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


# ============================================================
# PER-QUARTER BUCKETS
# ============================================================

class QuarterBucket:
    def __init__(self, q: int):
        self.q = q

        # Market metrics
        self.atr_1h_values = []
        self.atr_4h_values = []
        self.daily_range_pct_values = []
        self.volume_values = []

        # Signal metrics
        self.sweeps = 0
        self.mss = 0
        self.displacements = 0
        self.order_blocks = 0
        self.fvgs = 0
        self.retests = 0
        self.micro_mss = 0
        self.tradable_entries = 0

        # Quality metrics
        self.reference_distances = []
        self.body_atr_values = []

        # Performance
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_r = 0.0

    def finalize(self):
        return {
            "quarter": self.q,
            "atr_1h": mean(self.atr_1h_values) if self.atr_1h_values else 0.0,
            "atr_4h": mean(self.atr_4h_values) if self.atr_4h_values else 0.0,
            "daily_range_pct": mean(self.daily_range_pct_values) if self.daily_range_pct_values else 0.0,
            "volume": mean(self.volume_values) if self.volume_values else 0.0,
            "sweeps": self.sweeps,
            "mss": self.mss,
            "displacements": self.displacements,
            "order_blocks": self.order_blocks,
            "fvgs": self.fvgs,
            "retests": self.retests,
            "micro_mss": self.micro_mss,
            "tradable": self.tradable_entries,
            "mean_ref_dist": mean(self.reference_distances) if self.reference_distances else 0.0,
            "mean_body_atr": mean(self.body_atr_values) if self.body_atr_values else 0.0,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": (self.wins / (self.wins + self.losses)) if (self.wins + self.losses) else 0.0,
            "total_r": self.total_r,
        }


# ============================================================
# MARKET METRICS PER QUARTER (per symbol)
# ============================================================

def compute_market_metrics(df_5m, year: int, buckets: dict):
    """
    Compute ATR_1H, ATR_4H, daily range %, volume per quarter.
    """

    import pandas as pd

    # ---- 1H ----
    df_1h = resample_ohlcv(df_5m, "1h").copy()
    df_1h_indexed = df_1h.copy()
    if "timestamp" in df_1h_indexed.columns:
        df_1h_indexed["timestamp"] = pd.to_datetime(
            df_1h_indexed["timestamp"], utc=True,
        )
        df_1h_indexed = df_1h_indexed.set_index("timestamp").sort_index()

    atr_1h = calculate_atr(df_1h_indexed, period=ATR_PERIOD)

    for ts, val in atr_1h.items():
        q = quarter_of(year, ts)
        if q is None or pd.isna(val):
            continue
        buckets[q].atr_1h_values.append(float(val))

    # ---- 4H ----
    df_4h = resample_ohlcv(df_5m, "4h").copy()
    df_4h_indexed = df_4h.copy()
    if "timestamp" in df_4h_indexed.columns:
        df_4h_indexed["timestamp"] = pd.to_datetime(
            df_4h_indexed["timestamp"], utc=True,
        )
        df_4h_indexed = df_4h_indexed.set_index("timestamp").sort_index()

    atr_4h = calculate_atr(df_4h_indexed, period=ATR_PERIOD)

    for ts, val in atr_4h.items():
        q = quarter_of(year, ts)
        if q is None or pd.isna(val):
            continue
        buckets[q].atr_4h_values.append(float(val))

    # ---- Daily range % and volume (manual resample to 1D) ----
    df_daily_source = df_5m.copy()
    if "timestamp" in df_daily_source.columns:
        df_daily_source["timestamp"] = pd.to_datetime(
            df_daily_source["timestamp"], utc=True,
        )
        df_daily_source = (
            df_daily_source.set_index("timestamp").sort_index()
        )
    elif not isinstance(df_daily_source.index, pd.DatetimeIndex):
        df_daily_source.index = pd.to_datetime(
            df_daily_source.index, utc=True,
        )

    df_daily = df_daily_source.resample("1D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    for ts, row in df_daily.iterrows():
        q = quarter_of(year, ts)
        if q is None:
            continue
        high = safe_float(row.get("high"))
        low = safe_float(row.get("low"))
        close = safe_float(row.get("close"))
        volume = safe_float(row.get("volume"))
        if high and low and close and close > 0:
            buckets[q].daily_range_pct_values.append((high - low) / close)
        if volume is not None:
            buckets[q].volume_values.append(volume)


# ============================================================
# PIPELINE PER SYMBOL
# ============================================================

def process_symbol(symbol: str, year: int, buckets: dict):
    print(f"  Processing {symbol} ...")

    try:
        df_5m = load_year(DATA_DIR, symbol, "5m", year)
    except Exception as e:
        print(f"    SKIP: {e}")
        return

    if df_5m is None or len(df_5m) == 0:
        print("    SKIP: empty data")
        return

    # ---- Market metrics ----
    compute_market_metrics(df_5m, year, buckets)

    df_1h = resample_ohlcv(df_5m, "1h")
    df_4h = resample_ohlcv(df_5m, "4h")

    # ---- 4H structure + liquidity ----
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

    # ---- Sweeps ----
    sweeps = detect_sweeps(
        df_1h,
        liquidity_zones,
        atr_period=SWEEP_ATR_PERIOD,
    )

    # ---- 1H structure + MSS ----
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

    # ---- Displacement ----
    displacement_events = detect_displacements(
        df_1h,
        mss_events,
        atr_period=ATR_PERIOD,
        min_body_atr=DISPLACEMENT_MIN_BODY_ATR,
        bullish_clv=DISPLACEMENT_BULLISH_CLV,
        bearish_clv=DISPLACEMENT_BEARISH_CLV,
        max_bars_after_mss=DISPLACEMENT_MAX_BARS_AFTER_MSS,
    )

    # ---- OB / FVG ----
    order_blocks = detect_order_blocks(df_1h, displacement_events, lookback=OB_LOOKBACK)
    fvgs = detect_fvgs(df_1h, displacement_events, search_bars=FVG_SEARCH_BARS, atr_period=ATR_PERIOD)

    # ---- Retests ----
    ob_retests = detect_ob_retests(df_1h, order_blocks, max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE)
    fvg_retests = detect_fvg_retests(df_1h, fvgs, max_bars_after_zone=RETEST_MAX_BARS_AFTER_ZONE)
    all_retests = list(ob_retests) + list(fvg_retests)

    # ---- 5M micro structure + Micro-MSS ----
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

    # ---- Bucket signals ----
    for s in sweeps:
        q = quarter_of(year, get_attr(s, "timestamp", "sweep_timestamp"))
        if q is not None:
            buckets[q].sweeps += 1

    for m in mss_events:
        q = quarter_of(year, get_attr(m, "timestamp"))
        if q is not None:
            buckets[q].mss += 1

    for d in displacement_events:
        q = quarter_of(year, get_attr(d, "timestamp"))
        if q is not None:
            buckets[q].displacements += 1
        body_atr = safe_float(get_attr(d, "body_atr", "body_over_atr"))
        if q is not None and body_atr is not None:
            buckets[q].body_atr_values.append(body_atr)

    for ob in order_blocks:
        q = quarter_of(year, get_attr(ob, "timestamp"))
        if q is not None:
            buckets[q].order_blocks += 1

    for f in fvgs:
        q = quarter_of(year, get_attr(f, "timestamp"))
        if q is not None:
            buckets[q].fvgs += 1

    for r in all_retests:
        q = quarter_of(year, get_attr(r, "retest_timestamp"))
        if q is not None:
            buckets[q].retests += 1

    for mm in micro_mss_events:
        q = quarter_of(year, get_attr(mm, "timestamp"))
        if q is not None:
            buckets[q].micro_mss += 1
        ref_dist = safe_float(get_attr(mm, "reference_distance_atr"))
        if q is not None and ref_dist is not None:
            buckets[q].reference_distances.append(ref_dist)

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

    for c in tradable_entries:
        q = quarter_of(year, getattr(c, "timestamp"))
        if q is not None:
            buckets[q].tradable_entries += 1

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

    # ---- Risk + Size + Backtest ----
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

    for t in trades:
        q = quarter_of(year, getattr(t, "entry_timestamp", None))
        if q is None:
            continue
        buckets[q].trades += 1
        outcome = getattr(t, "outcome", None)
        if outcome == "win":
            buckets[q].wins += 1
            buckets[q].total_r += float(t.r_multiple or 0.0)
        elif outcome == "loss":
            buckets[q].losses += 1
            buckets[q].total_r += float(t.r_multiple or 0.0)


# ============================================================
# MAIN
# ============================================================

def main():
    year = YEAR

    print("=" * 72)
    print("QUARTERLY ANALYSIS — MARKET vs SIGNAL vs PERFORMANCE")
    print("=" * 72)
    print(f"Year:    {year}")
    print(f"Symbols: {len(SYMBOLS)}")
    print()

    buckets = {
        1: QuarterBucket(1),
        2: QuarterBucket(2),
        3: QuarterBucket(3),
        4: QuarterBucket(4),
    }

    print("Processing symbols...")
    for symbol in SYMBOLS:
        process_symbol(symbol, year, buckets)

    print()
    print("=" * 72)
    print("QUARTERLY SUMMARY")
    print("=" * 72)

    rows = [buckets[q].finalize() for q in (1, 2, 3, 4)]

    # ---- Table 1: Market ----
    print()
    print("Market metrics (averaged across symbols):")
    print()
    print(
        f"{'Quarter':<9} "
        f"{'ATR_1H':>10} "
        f"{'ATR_4H':>10} "
        f"{'DailyRange%':>13} "
        f"{'Volume':>14}"
    )
    print("-" * 72)
    for r in rows:
        print(
            f"Q{r['quarter']:<8} "
            f"{r['atr_1h']:>10.2f} "
            f"{r['atr_4h']:>10.2f} "
            f"{r['daily_range_pct']*100:>12.2f}% "
            f"{r['volume']:>14,.0f}"
        )

    # ---- Table 2: Signals ----
    print()
    print("Signal counts (summed across symbols):")
    print()
    print(
        f"{'Quarter':<9} "
        f"{'Sweeps':>8} "
        f"{'MSS':>6} "
        f"{'Disp':>6} "
        f"{'OB':>6} "
        f"{'FVG':>6} "
        f"{'Retests':>9} "
        f"{'MicroMSS':>10} "
        f"{'Tradable':>10}"
    )
    print("-" * 84)
    for r in rows:
        print(
            f"Q{r['quarter']:<8} "
            f"{r['sweeps']:>8} "
            f"{r['mss']:>6} "
            f"{r['displacements']:>6} "
            f"{r['order_blocks']:>6} "
            f"{r['fvgs']:>6} "
            f"{r['retests']:>9} "
            f"{r['micro_mss']:>10} "
            f"{r['tradable']:>10}"
        )

    # ---- Table 3: Quality ----
    print()
    print("Quality metrics (averaged):")
    print()
    print(
        f"{'Quarter':<9} "
        f"{'MeanRefDist':>13} "
        f"{'MeanBodyATR':>13}"
    )
    print("-" * 72)
    for r in rows:
        print(
            f"Q{r['quarter']:<8} "
            f"{r['mean_ref_dist']:>13.3f} "
            f"{r['mean_body_atr']:>13.3f}"
        )

    # ---- Table 4: Performance ----
    print()
    print("Performance:")
    print()
    print(
        f"{'Quarter':<9} "
        f"{'Trades':>8} "
        f"{'Wins':>6} "
        f"{'Losses':>8} "
        f"{'WinRate':>10} "
        f"{'TotalR':>10}"
    )
    print("-" * 72)
    for r in rows:
        print(
            f"Q{r['quarter']:<8} "
            f"{r['trades']:>8} "
            f"{r['wins']:>6} "
            f"{r['losses']:>8} "
            f"{r['win_rate']*100:>9.2f}% "
            f"{r['total_r']:>+10.2f}"
        )

    # ---- Correlations ----
    print()
    print("=" * 72)
    print("CORRELATION WITH TOTAL R")
    print("=" * 72)
    print()
    print("(Pearson correlation between each metric and Total R across Q1-Q4)")
    print("  |r| > 0.7  -> strong candidate for regime filter")
    print("  0.4 < |r| < 0.7 -> moderate")
    print("  |r| < 0.4  -> weak")
    print()

    total_r_series = [r["total_r"] for r in rows]

    features = [
        ("ATR_1H", [r["atr_1h"] for r in rows]),
        ("ATR_4H", [r["atr_4h"] for r in rows]),
        ("DailyRange%", [r["daily_range_pct"] for r in rows]),
        ("Volume", [r["volume"] for r in rows]),
        ("Sweeps", [r["sweeps"] for r in rows]),
        ("MSS", [r["mss"] for r in rows]),
        ("Displacements", [r["displacements"] for r in rows]),
        ("OrderBlocks", [r["order_blocks"] for r in rows]),
        ("FVGs", [r["fvgs"] for r in rows]),
        ("Retests", [r["retests"] for r in rows]),
        ("MicroMSS", [r["micro_mss"] for r in rows]),
        ("Tradable", [r["tradable"] for r in rows]),
        ("MeanRefDist", [r["mean_ref_dist"] for r in rows]),
        ("MeanBodyATR", [r["mean_body_atr"] for r in rows]),
    ]

    results = []
    for name, values in features:
        r_val = pearson(values, total_r_series)
        results.append((name, r_val))

    results.sort(key=lambda x: -abs(x[1]))

    for name, r_val in results:
        if abs(r_val) > 0.7:
            tag = "STRONG"
        elif abs(r_val) > 0.4:
            tag = "moderate"
        else:
            tag = "weak"

        print(f"  {name:<16} r = {r_val:+.3f}   [{tag}]")

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()