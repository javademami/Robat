from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional
import math


# ============================================================
# CONFIG
# ============================================================

DEFAULT_RISK_PERCENT = 1.0
MIN_RISK_PERCENT = 0.01
MAX_RISK_PERCENT = 5.0


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class PositionSize:
    timestamp: Any

    direction: str

    account_equity: float
    risk_percent: float
    risk_amount: float

    entry_price: float
    stop_loss: float
    take_profit: float

    risk_per_unit: float
    reward_per_unit: float
    risk_reward: float

    position_size: float

    notional_value: float
    margin_required: float

    leverage: float

    valid: bool
    rejection_reason: Optional[str] = None

    score: Optional[float] = None
    grade: Optional[str] = None
    zone_type: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def _normalize_direction(value: Any) -> str:
    """
    Normalize Enum/string direction to:
        bullish
        bearish
    """

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


# ============================================================
# RISK AMOUNT
# ============================================================

def calculate_risk_amount(
    account_equity: float,
    risk_percent: float = DEFAULT_RISK_PERCENT,
) -> float:
    """
    Calculate maximum account-level risk.

    Example:
        equity = 10,000
        risk   = 1%

        risk_amount = 100
    """

    equity = _safe_float(account_equity)
    risk = _safe_float(risk_percent)

    if equity is None or equity <= 0:
        raise ValueError("account_equity must be > 0")

    if risk is None or risk <= 0:
        raise ValueError("risk_percent must be > 0")

    return equity * (risk / 100.0)


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    risk_amount: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    """
    Position size based on fixed account risk.

        position_size =
            risk_amount / abs(entry - stop)
    """

    risk_amount_f = _safe_float(risk_amount)
    entry = _safe_float(entry_price)
    stop = _safe_float(stop_loss)

    if risk_amount_f is None or risk_amount_f <= 0:
        raise ValueError("risk_amount must be > 0")

    if entry is None or entry <= 0:
        raise ValueError("entry_price must be > 0")

    if stop is None or stop <= 0:
        raise ValueError("stop_loss must be > 0")

    risk_per_unit = abs(entry - stop)

    if risk_per_unit <= 0:
        raise ValueError("entry_price and stop_loss cannot be equal")

    return risk_amount_f / risk_per_unit


# ============================================================
# NOTIONAL
# ============================================================

def calculate_notional_value(
    position_size: float,
    entry_price: float,
) -> float:

    size = _safe_float(position_size)
    entry = _safe_float(entry_price)

    if size is None or size <= 0:
        raise ValueError("position_size must be > 0")

    if entry is None or entry <= 0:
        raise ValueError("entry_price must be > 0")

    return size * entry


# ============================================================
# MARGIN
# ============================================================

def calculate_margin_required(
    notional_value: float,
    leverage: float = 1.0,
) -> float:

    notional = _safe_float(notional_value)
    lev = _safe_float(leverage)

    if notional is None or notional < 0:
        raise ValueError("notional_value must be >= 0")

    if lev is None or lev <= 0:
        raise ValueError("leverage must be > 0")

    return notional / lev


# ============================================================
# VALIDATION
# ============================================================

def validate_position_size_inputs(
    direction: str,
    account_equity: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    leverage: float = 1.0,
    min_risk_percent: float = MIN_RISK_PERCENT,
    max_risk_percent: float = MAX_RISK_PERCENT,
) -> tuple[bool, Optional[str]]:

    normalized_direction = _normalize_direction(direction)

    if normalized_direction not in {"bullish", "bearish"}:
        return False, "invalid_direction"

    equity = _safe_float(account_equity)
    risk_pct = _safe_float(risk_percent)
    entry = _safe_float(entry_price)
    stop = _safe_float(stop_loss)
    tp = _safe_float(take_profit)
    lev = _safe_float(leverage)

    if equity is None or equity <= 0:
        return False, "invalid_account_equity"

    if risk_pct is None:
        return False, "invalid_risk_percent"

    if risk_pct < min_risk_percent:
        return False, "risk_percent_below_minimum"

    if risk_pct > max_risk_percent:
        return False, "risk_percent_above_maximum"

    if entry is None or entry <= 0:
        return False, "invalid_entry_price"

    if stop is None or stop <= 0:
        return False, "invalid_stop_loss"

    if tp is None or tp <= 0:
        return False, "invalid_take_profit"

    if lev is None or lev <= 0:
        return False, "invalid_leverage"

    if entry == stop:
        return False, "zero_risk_distance"

    if normalized_direction == "bullish":

        if stop >= entry:
            return False, "bullish_stop_must_be_below_entry"

        if tp <= entry:
            return False, "bullish_tp_must_be_above_entry"

    elif normalized_direction == "bearish":

        if stop <= entry:
            return False, "bearish_stop_must_be_above_entry"

        if tp >= entry:
            return False, "bearish_tp_must_be_below_entry"

    return True, None


# ============================================================
# BUILD ONE POSITION SIZE
# ============================================================

def build_position_size(
    risk_plan: Any,
    account_equity: float,
    risk_percent: float = DEFAULT_RISK_PERCENT,
    leverage: float = 1.0,
) -> PositionSize:
    """
    Convert a valid RiskPlan into an account-risk-based position size.

    This function intentionally does NOT modify RiskPlan.
    """

    timestamp = getattr(risk_plan, "timestamp", None)

    direction = _normalize_direction(
        getattr(risk_plan, "direction", None)
    )

    entry = _safe_float(
        getattr(risk_plan, "entry_price", None)
    )

    stop = _safe_float(
        getattr(risk_plan, "stop_loss", None)
    )

    tp = _safe_float(
        getattr(risk_plan, "take_profit", None)
    )

    rr = _safe_float(
        getattr(risk_plan, "risk_reward", None)
    )

    score = _safe_float(
        getattr(risk_plan, "score", None)
    )

    grade = getattr(risk_plan, "grade", None)

    zone_type = getattr(risk_plan, "zone_type", None)

    valid, rejection = validate_position_size_inputs(
        direction=direction,
        account_equity=account_equity,
        risk_percent=risk_percent,
        entry_price=entry if entry is not None else 0,
        stop_loss=stop if stop is not None else 0,
        take_profit=tp if tp is not None else 0,
        leverage=leverage,
    )

    if not valid:
        return PositionSize(
            timestamp=timestamp,
            direction=direction,
            account_equity=float(account_equity),
            risk_percent=float(risk_percent),
            risk_amount=0.0,
            entry_price=entry or 0.0,
            stop_loss=stop or 0.0,
            take_profit=tp or 0.0,
            risk_per_unit=0.0,
            reward_per_unit=0.0,
            risk_reward=rr or 0.0,
            position_size=0.0,
            notional_value=0.0,
            margin_required=0.0,
            leverage=float(leverage),
            valid=False,
            rejection_reason=rejection,
            score=score,
            grade=grade,
            zone_type=zone_type,
        )

    risk_amount = calculate_risk_amount(
        account_equity,
        risk_percent,
    )

    risk_per_unit = abs(entry - stop)

    reward_per_unit = abs(tp - entry)

    calculated_rr = (
        reward_per_unit / risk_per_unit
        if risk_per_unit > 0
        else 0.0
    )

    position_size = calculate_position_size(
        risk_amount=risk_amount,
        entry_price=entry,
        stop_loss=stop,
    )

    notional_value = calculate_notional_value(
        position_size=position_size,
        entry_price=entry,
    )

    margin_required = calculate_margin_required(
        notional_value=notional_value,
        leverage=leverage,
    )

    return PositionSize(
        timestamp=timestamp,
        direction=direction,

        account_equity=float(account_equity),
        risk_percent=float(risk_percent),
        risk_amount=risk_amount,

        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,

        risk_per_unit=risk_per_unit,
        reward_per_unit=reward_per_unit,
        risk_reward=calculated_rr,

        position_size=position_size,

        notional_value=notional_value,
        margin_required=margin_required,

        leverage=float(leverage),

        valid=True,
        rejection_reason=None,

        score=score,
        grade=grade,
        zone_type=zone_type,
    )


# ============================================================
# BUILD MANY
# ============================================================

def build_position_sizes(
    risk_plans: Iterable[Any],
    account_equity: float,
    risk_percent: float = DEFAULT_RISK_PERCENT,
    leverage: float = 1.0,
) -> list[PositionSize]:

    results: list[PositionSize] = []

    for risk_plan in risk_plans:

        result = build_position_size(
            risk_plan=risk_plan,
            account_equity=account_equity,
            risk_percent=risk_percent,
            leverage=leverage,
        )

        results.append(result)

    return results


# ============================================================
# VALID ONLY
# ============================================================

def get_valid_position_sizes(
    position_sizes: Iterable[PositionSize],
) -> list[PositionSize]:

    return [
        item
        for item in position_sizes
        if item.valid
    ]


# ============================================================
# DATAFRAME EXPORT
# ============================================================

def position_sizes_to_dataframe(
    position_sizes: Iterable[PositionSize],
):
    import pandas as pd

    rows = []

    for item in position_sizes:

        rows.append(
            {
                "timestamp": item.timestamp,
                "direction": item.direction,

                "account_equity": item.account_equity,
                "risk_percent": item.risk_percent,
                "risk_amount": item.risk_amount,

                "entry_price": item.entry_price,
                "stop_loss": item.stop_loss,
                "take_profit": item.take_profit,

                "risk_per_unit": item.risk_per_unit,
                "reward_per_unit": item.reward_per_unit,
                "risk_reward": item.risk_reward,

                "position_size": item.position_size,

                "notional_value": item.notional_value,
                "margin_required": item.margin_required,

                "leverage": item.leverage,

                "valid": item.valid,
                "rejection_reason": item.rejection_reason,

                "score": item.score,
                "grade": item.grade,
                "zone_type": item.zone_type,
            }
        )

    return pd.DataFrame(rows)