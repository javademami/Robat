"""
Risk Manager V1.1
-----------------
Calculates Entry / Stop Loss / Take Profit / Risk-Reward
for validated Entry Candidates.

V1 philosophy:
- No trade without a valid structural invalidation point.
- SL is based on the setup structure, not an arbitrary percentage.
- TP is based on configurable Risk:Reward.
- RR is calculated before allowing a trade.
- No position sizing or leverage in V1.

V1.1 change:
- Added MAX_RISK_ATR cap on stop distance.
- If the structural stop (zone edge ± buffer) is more than
  MAX_RISK_ATR * ATR away from entry, the stop is capped.
- This prevents huge OB/FVG zones from producing stops that
  are 10-22% of price on BTC, which are not tradable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Optional


# ============================================================
# V1 CONFIG
# ============================================================

DEFAULT_RR = 2.0
MIN_RR = 1.5

# Small ATR buffer beyond structural invalidation.
# This is intentionally modest in V1.
SL_BUFFER_ATR = 0.10

# Maximum SL distance from entry, in ATR units.
# Prevents huge zones from producing absurd stops.
# Set to None to disable capping.
MAX_RISK_ATR: Optional[float] = 5.0

ATR_PERIOD = 14


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class RiskPlan:
    timestamp: Any
    direction: str

    entry_price: float

    stop_loss: float
    take_profit: float

    risk_per_unit: float
    reward_per_unit: float
    risk_reward: float

    sl_reference: str
    tp_reference: str

    sl_buffer_atr: float
    atr: Optional[float]

    valid: bool
    rejection_reason: Optional[str] = None

    # Optional source references
    micro_mss_timestamp: Any = None
    retest_timestamp: Any = None
    displacement_timestamp: Any = None
    mss_timestamp: Any = None
    zone_type: Optional[str] = None


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_direction(value: Any) -> str:
    """
    Normalize Enum/string direction to:
        bullish
        bearish
    """
    raw = getattr(value, "value", value)
    return str(raw).lower().strip()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        result = float(value)

        if result != result:  # NaN
            return None

        return result
    except (TypeError, ValueError):
        return None


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period: int = ATR_PERIOD,
):
    """
    Calculate Wilder-style ATR using OHLC columns.

    Returns a pandas Series aligned with df.index.

    Expected columns:
        high
        low
        close
    """

    import pandas as pd

    required = {"high", "low", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"ATR calculation missing columns: {sorted(missing)}"
        )

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder/RMA approximation
    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


# ============================================================
# ZONE HELPERS
# ============================================================

def _get_zone_low(zone: Any) -> Optional[float]:
    return _safe_float(getattr(zone, "zone_low", None))


def _get_zone_high(zone: Any) -> Optional[float]:
    return _safe_float(getattr(zone, "zone_high", None))


def _get_zone_type(zone: Any) -> str:
    """
    Return the zone's type as "ob" or "fvg".

    FIX: OrderBlockEvent / FVGEvent do not carry a `zone_type`
    attribute of their own (only RetestEvent/EntryCandidate do,
    set externally in retest_detector.py). The previous version
    of this helper only checked getattr(zone, "zone_type", ""),
    which was always "" for every real zone object -- silently
    breaking ALL zone matching in build_risk_plans (every
    candidate fell through to "missing_zone", every RiskPlan
    was invalid, regardless of the actual data).

    Same fallback-by-class-name approach already used in
    entry_detector.py.
    """

    zone_type = str(
        getattr(zone, "zone_type", "")
    ).lower().strip()

    if zone_type:
        return zone_type

    class_name = type(zone).__name__.lower()

    if "fvg" in class_name:
        return "fvg"

    if (
        "orderblock" in class_name
        or "order_block" in class_name
    ):
        return "ob"

    return ""


def _get_zone_direction(zone: Any) -> str:
    return _normalize_direction(
        getattr(zone, "direction", "")
    )


# ============================================================
# STRUCTURAL SL
# ============================================================

def calculate_structural_stop(
    entry_price: float,
    direction: Any,
    zone: Any = None,
    atr: Optional[float] = None,
    buffer_atr: float = SL_BUFFER_ATR,
    max_risk_atr: Optional[float] = MAX_RISK_ATR,
) -> tuple[Optional[float], str]:
    """
    Calculate structural stop.

    Bullish:
        SL below zone_low

    Bearish:
        SL above zone_high

    ATR buffer is added outside the zone.

    IMPORTANT (V1.1):
    - If the structural stop (zone edge ± buffer) is more than
      `max_risk_atr * ATR` away from entry, the stop is capped at
      `entry ± max_risk_atr * ATR`.
    - This prevents zones that are hundreds of candles wide from
      producing absurd stop distances (e.g. 10-22% of price on BTC).
    - Set `max_risk_atr=None` to disable capping.

    Returns:
        (stop_loss, reference)
    """

    direction = _normalize_direction(direction)

    entry = _safe_float(entry_price)
    atr_value = _safe_float(atr)

    if entry is None:
        return None, "invalid_entry"

    if direction not in {"bullish", "bearish"}:
        return None, "invalid_direction"

    if zone is None:
        return None, "missing_zone"

    zone_low = _get_zone_low(zone)
    zone_high = _get_zone_high(zone)

    if zone_low is None or zone_high is None:
        return None, "invalid_zone"

    if zone_low >= zone_high:
        return None, "invalid_zone_range"

    # --------------------------------------------------------
    # ATR buffer
    # --------------------------------------------------------

    if atr_value is not None and atr_value > 0:
        buffer = atr_value * buffer_atr
    else:
        buffer = 0.0

    # --------------------------------------------------------
    # Bullish setup
    # --------------------------------------------------------

    if direction == "bullish":

        structural_stop = zone_low - buffer

        if structural_stop >= entry:
            return None, "stop_not_below_entry"

        # ------------------------------------------------
        # Apply ATR cap
        # ------------------------------------------------

        if (
            max_risk_atr is not None
            and atr_value is not None
            and atr_value > 0
        ):
            max_distance = max_risk_atr * atr_value
            structural_distance = entry - structural_stop

            if structural_distance > max_distance:
                capped_stop = entry - max_distance

                return (
                    capped_stop,
                    f"zone_low_capped_at_{max_risk_atr:.1f}atr",
                )

        return structural_stop, "zone_low_minus_atr_buffer"

    # --------------------------------------------------------
    # Bearish setup
    # --------------------------------------------------------

    structural_stop = zone_high + buffer

    if structural_stop <= entry:
        return None, "stop_not_above_entry"

    # ----------------------------------------------------
    # Apply ATR cap
    # ----------------------------------------------------

    if (
        max_risk_atr is not None
        and atr_value is not None
        and atr_value > 0
    ):
        max_distance = max_risk_atr * atr_value
        structural_distance = structural_stop - entry

        if structural_distance > max_distance:
            capped_stop = entry + max_distance

            return (
                capped_stop,
                f"zone_high_capped_at_{max_risk_atr:.1f}atr",
            )

    return structural_stop, "zone_high_plus_atr_buffer"


# ============================================================
# TAKE PROFIT
# ============================================================

def calculate_take_profit(
    entry_price: float,
    stop_loss: float,
    direction: Any,
    target_rr: float = DEFAULT_RR,
) -> tuple[Optional[float], Optional[float]]:
    """
    Calculate TP from fixed Risk:Reward.

    Returns:
        (take_profit, reward_distance)
    """

    entry = _safe_float(entry_price)
    stop = _safe_float(stop_loss)
    rr = _safe_float(target_rr)

    direction = _normalize_direction(direction)

    if entry is None or stop is None:
        return None, None

    if rr is None or rr <= 0:
        return None, None

    risk = abs(entry - stop)

    if risk <= 0:
        return None, None

    reward = risk * rr

    if direction == "bullish":
        tp = entry + reward

    elif direction == "bearish":
        tp = entry - reward

    else:
        return None, None

    return tp, reward


# ============================================================
# RISK / REWARD
# ============================================================

def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> Optional[float]:
    """
    Calculate actual Risk:Reward.
    """

    entry = _safe_float(entry_price)
    stop = _safe_float(stop_loss)
    tp = _safe_float(take_profit)

    if entry is None or stop is None or tp is None:
        return None

    risk = abs(entry - stop)

    if risk <= 0:
        return None

    reward = abs(tp - entry)

    return reward / risk


# ============================================================
# VALIDATION
# ============================================================

def validate_risk_plan(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: Any,
    min_rr: float = MIN_RR,
) -> tuple[bool, Optional[str]]:
    """
    Validate the final risk plan.
    """

    entry = _safe_float(entry_price)
    stop = _safe_float(stop_loss)
    tp = _safe_float(take_profit)

    direction = _normalize_direction(direction)

    if entry is None:
        return False, "invalid_entry"

    if stop is None:
        return False, "invalid_stop"

    if tp is None:
        return False, "invalid_take_profit"

    if direction == "bullish":

        if stop >= entry:
            return False, "bullish_stop_not_below_entry"

        if tp <= entry:
            return False, "bullish_tp_not_above_entry"

    elif direction == "bearish":

        if stop <= entry:
            return False, "bearish_stop_not_above_entry"

        if tp >= entry:
            return False, "bearish_tp_not_below_entry"

    else:
        return False, "invalid_direction"

    rr = calculate_risk_reward(
        entry,
        stop,
        tp,
    )

    if rr is None:
        return False, "invalid_rr"

    if rr < min_rr:
        return False, "rr_below_minimum"

    return True, None


# ============================================================
# BUILD ONE RISK PLAN
# ============================================================

def build_risk_plan(
    entry_candidate: Any,
    zone: Any,
    atr: Optional[float] = None,
    target_rr: float = DEFAULT_RR,
    min_rr: float = MIN_RR,
    buffer_atr: float = SL_BUFFER_ATR,
    max_risk_atr: Optional[float] = MAX_RISK_ATR,
) -> RiskPlan:
    """
    Build a complete RiskPlan from an EntryCandidate.

    EntryCandidate is expected to expose:
        timestamp
        direction
        entry_price
        micro_mss_timestamp
        retest_timestamp
        displacement_timestamp
        mss_timestamp
        zone_type

    Zone is expected to expose:
        zone_low
        zone_high
    """

    timestamp = getattr(entry_candidate, "timestamp", None)

    direction = _normalize_direction(
        getattr(entry_candidate, "direction", "")
    )

    entry_price = _safe_float(
        getattr(entry_candidate, "entry_price", None)
    )

    zone_type = str(
        getattr(
            entry_candidate,
            "zone_type",
            getattr(zone, "zone_type", ""),
        )
    ).lower()

    # --------------------------------------------------------
    # Missing entry
    # --------------------------------------------------------

    if entry_price is None:
        return RiskPlan(
            timestamp=timestamp,
            direction=direction,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            risk_per_unit=0.0,
            reward_per_unit=0.0,
            risk_reward=0.0,
            sl_reference="invalid_entry",
            tp_reference="not_calculated",
            sl_buffer_atr=buffer_atr,
            atr=atr,
            valid=False,
            rejection_reason="missing_entry_price",
            micro_mss_timestamp=getattr(
                entry_candidate,
                "timestamp",
                None,
            ),
            retest_timestamp=getattr(
                entry_candidate,
                "retest_timestamp",
                None,
            ),
            displacement_timestamp=getattr(
                entry_candidate,
                "displacement_timestamp",
                None,
            ),
            mss_timestamp=getattr(
                entry_candidate,
                "mss_timestamp",
                None,
            ),
            zone_type=zone_type,
        )

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    stop_loss, sl_reference = calculate_structural_stop(
        entry_price=entry_price,
        direction=direction,
        zone=zone,
        atr=atr,
        buffer_atr=buffer_atr,
        max_risk_atr=max_risk_atr,
    )

    if stop_loss is None:
        return RiskPlan(
            timestamp=timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_loss=0.0,
            take_profit=0.0,
            risk_per_unit=0.0,
            reward_per_unit=0.0,
            risk_reward=0.0,
            sl_reference=sl_reference,
            tp_reference="not_calculated",
            sl_buffer_atr=buffer_atr,
            atr=atr,
            valid=False,
            rejection_reason=sl_reference,
            micro_mss_timestamp=getattr(
                entry_candidate,
                "timestamp",
                None,
            ),
            retest_timestamp=getattr(
                entry_candidate,
                "retest_timestamp",
                None,
            ),
            displacement_timestamp=getattr(
                entry_candidate,
                "displacement_timestamp",
                None,
            ),
            mss_timestamp=getattr(
                entry_candidate,
                "mss_timestamp",
                None,
            ),
            zone_type=zone_type,
        )

    # --------------------------------------------------------
    # TP
    # --------------------------------------------------------

    take_profit, reward = calculate_take_profit(
        entry_price=entry_price,
        stop_loss=stop_loss,
        direction=direction,
        target_rr=target_rr,
    )

    if take_profit is None or reward is None:
        return RiskPlan(
            timestamp=timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=0.0,
            risk_per_unit=abs(entry_price - stop_loss),
            reward_per_unit=0.0,
            risk_reward=0.0,
            sl_reference=sl_reference,
            tp_reference="invalid_target_rr",
            sl_buffer_atr=buffer_atr,
            atr=atr,
            valid=False,
            rejection_reason="invalid_target_rr",
            micro_mss_timestamp=getattr(
                entry_candidate,
                "timestamp",
                None,
            ),
            retest_timestamp=getattr(
                entry_candidate,
                "retest_timestamp",
                None,
            ),
            displacement_timestamp=getattr(
                entry_candidate,
                "displacement_timestamp",
                None,
            ),
            mss_timestamp=getattr(
                entry_candidate,
                "mss_timestamp",
                None,
            ),
            zone_type=zone_type,
        )

    # --------------------------------------------------------
    # RR
    # --------------------------------------------------------

    rr = calculate_risk_reward(
        entry_price,
        stop_loss,
        take_profit,
    )

    if rr is None:
        rr = 0.0

    valid, rejection_reason = validate_risk_plan(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        direction=direction,
        min_rr=min_rr,
    )

    risk = abs(entry_price - stop_loss)

    return RiskPlan(
        timestamp=timestamp,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_per_unit=risk,
        reward_per_unit=reward,
        risk_reward=rr,
        sl_reference=sl_reference,
        tp_reference=f"fixed_rr_{target_rr:.2f}",
        sl_buffer_atr=buffer_atr,
        atr=atr,
        valid=valid,
        rejection_reason=rejection_reason,
        micro_mss_timestamp=getattr(
            entry_candidate,
            "timestamp",
            None,
        ),
        retest_timestamp=getattr(
            entry_candidate,
            "retest_timestamp",
            None,
        ),
        displacement_timestamp=getattr(
            entry_candidate,
            "displacement_timestamp",
            None,
        ),
        mss_timestamp=getattr(
            entry_candidate,
            "mss_timestamp",
            None,
        ),
        zone_type=zone_type,
    )


# ============================================================
# BATCH PROCESSING
# ============================================================

def build_risk_plans(
    entry_candidates: Iterable[Any],
    zones: Iterable[Any],
    atr_by_timestamp: Optional[dict] = None,
    target_rr: float = DEFAULT_RR,
    min_rr: float = MIN_RR,
    buffer_atr: float = SL_BUFFER_ATR,
    max_risk_atr: Optional[float] = MAX_RISK_ATR,
) -> list[RiskPlan]:
    """
    Build risk plans for multiple EntryCandidates.

    `zones` should contain the corresponding OB/FVG zones.

    Matching priority:
        1. exact timestamp
        2. zone_type + direction
    """

    candidates = list(entry_candidates)
    zone_list = list(zones)

    results: list[RiskPlan] = []

    for candidate in candidates:

        timestamp = getattr(candidate, "timestamp", None)

        direction = _normalize_direction(
            getattr(candidate, "direction", "")
        )

        zone_type = str(
            getattr(candidate, "zone_type", "")
        ).lower()

        zone = None

        # ----------------------------------------------------
        # Exact timestamp
        # ----------------------------------------------------

        for candidate_zone in zone_list:

            z_direction = _get_zone_direction(
                candidate_zone
            )

            z_type = _get_zone_type(
                candidate_zone
            )

            z_timestamp = getattr(
                candidate_zone,
                "timestamp",
                None,
            )

            if (
                z_timestamp == timestamp
                and z_direction == direction
                and (
                    not zone_type
                    or z_type == zone_type
                )
            ):
                zone = candidate_zone
                break

        # ----------------------------------------------------
        # Fallback: matching zone type + direction
        # ----------------------------------------------------

        if zone is None:

            for candidate_zone in zone_list:

                z_direction = _get_zone_direction(
                    candidate_zone
                )

                z_type = _get_zone_type(
                    candidate_zone
                )

                if (
                    z_direction == direction
                    and (
                        not zone_type
                        or z_type == zone_type
                    )
                ):
                    zone = candidate_zone
                    break

        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        atr = None

        if atr_by_timestamp is not None:
            atr = _safe_float(
                atr_by_timestamp.get(timestamp)
            )

        results.append(
            build_risk_plan(
                entry_candidate=candidate,
                zone=zone,
                atr=atr,
                target_rr=target_rr,
                min_rr=min_rr,
                buffer_atr=buffer_atr,
                max_risk_atr=max_risk_atr,
            )
        )

    return results


# ============================================================
# FILTER
# ============================================================

def get_valid_risk_plans(
    plans: Iterable[RiskPlan],
) -> list[RiskPlan]:
    """
    Return only valid risk plans.
    """

    return [
        plan
        for plan in plans
        if plan.valid
    ]


# ============================================================
# DATAFRAME EXPORT
# ============================================================

def risk_plans_to_dataframe(
    plans: Iterable[RiskPlan],
):
    """
    Convert RiskPlan objects to pandas DataFrame.
    """

    import pandas as pd

    rows = [
        asdict(plan)
        for plan in plans
    ]

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


__all__ = [
    "RiskPlan",
    "DEFAULT_RR",
    "MIN_RR",
    "SL_BUFFER_ATR",
    "MAX_RISK_ATR",
    "ATR_PERIOD",
    "calculate_atr",
    "calculate_structural_stop",
    "calculate_take_profit",
    "calculate_risk_reward",
    "validate_risk_plan",
    "build_risk_plan",
    "build_risk_plans",
    "get_valid_risk_plans",
    "risk_plans_to_dataframe",
]