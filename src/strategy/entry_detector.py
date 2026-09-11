"""
Entry Detector V2
=================

Pipeline:

4H Structure
    ↓
Liquidity
    ↓
Sweep
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
ENTRY + SCORE

Important:
- No look-ahead.
- Micro-MSS V1.2 reference validation is preserved.
- SL / TP / position sizing are NOT handled here.
- This module only creates EntryCandidate objects.

Score V2 = 100 points:

Structure / Bias       20
Liquidity / Sweep      15
MSS                    15
Displacement           15
OB / FVG Quality       10
Retest Quality         10
Micro-MSS              10
Risk / Reward           5
--------------------------------
TOTAL                  100

Grades:

90-100  A+
80-89   A
70-79   B
60-69   C
<60     D

Minimum tradable score:
70

V2 changes (context enrichment):
    - _find_zone_for_retest now matches on zone_timestamp
      (RetestEvent.zone_timestamp == OB/FVG.timestamp).
    - _score_zone uses class name (OB vs FVG) and reads
      penetration_ratio from the retest.
    - _score_structure supports per-timestamp bias
      via context["bias_by_timestamp"].
    - _score_liquidity supports per-MSS sweep linking
      via context["sweeps_by_mss_timestamp"] and uses
      sweep.depth_atr when available.
    - _score_mss uses real bars_after_sweep thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# ============================================================
# CONFIG
# ============================================================

MIN_ENTRY_SCORE = 70.0


SCORE_WEIGHTS = {
    "structure": 20,
    "liquidity_sweep": 15,
    "mss": 15,
    "displacement": 15,
    "zone_quality": 10,
    "retest_quality": 10,
    "micro_mss": 10,
    "risk_reward": 5,
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class EntryScore:
    """
    Entry score breakdown.
    """

    structure: float = 0.0
    liquidity_sweep: float = 0.0
    mss: float = 0.0
    displacement: float = 0.0
    zone_quality: float = 0.0
    retest_quality: float = 0.0
    micro_mss: float = 0.0
    risk_reward: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.structure
            + self.liquidity_sweep
            + self.mss
            + self.displacement
            + self.zone_quality
            + self.retest_quality
            + self.micro_mss
            + self.risk_reward,
            2,
        )

    @property
    def grade(self) -> str:
        score = self.total

        if score >= 90:
            return "A+"

        if score >= 80:
            return "A"

        if score >= 70:
            return "B"

        if score >= 60:
            return "C"

        return "D"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "structure": self.structure,
            "liquidity_sweep": self.liquidity_sweep,
            "mss": self.mss,
            "displacement": self.displacement,
            "zone_quality": self.zone_quality,
            "retest_quality": self.retest_quality,
            "micro_mss": self.micro_mss,
            "risk_reward": self.risk_reward,
            "total": self.total,
            "grade": self.grade,
        }


@dataclass
class EntryCandidate:
    """
    Final entry candidate produced by Entry Detector.
    """

    timestamp: Any

    direction: str

    entry_price: float

    zone_type: str

    mss_timestamp: Any = None

    displacement_timestamp: Any = None

    retest_timestamp: Any = None

    micro_mss_timestamp: Any = None

    zone_high: Optional[float] = None

    zone_low: Optional[float] = None

    score: float = 0.0

    grade: str = "D"

    tradable: bool = False

    score_breakdown: dict[str, float] = field(
        default_factory=dict
    )

    reasons: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "timestamp": self.timestamp,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "zone_type": self.zone_type,
            "mss_timestamp": self.mss_timestamp,
            "displacement_timestamp": self.displacement_timestamp,
            "retest_timestamp": self.retest_timestamp,
            "micro_mss_timestamp": self.micro_mss_timestamp,
            "zone_high": self.zone_high,
            "zone_low": self.zone_low,
            "score": self.score,
            "grade": self.grade,
            "tradable": self.tradable,
            "score_breakdown": self.score_breakdown,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


# ============================================================
# GENERIC HELPERS
# ============================================================

def _value(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:

    for name in names:

        if isinstance(obj, dict):

            if name in obj:
                return obj[name]

        elif hasattr(obj, name):

            return getattr(obj, name)

    return default


def _normalize_direction(value: Any) -> str:
    """
    Handles both:

        "bullish"
        "bearish"

    and Enum values such as:

        SweepDirection.BULLISH
    """

    if value is None:
        return ""

    raw = getattr(value, "value", value)

    text = str(raw).strip().lower()

    if text.endswith(".bullish"):
        return "bullish"

    if text.endswith(".bearish"):
        return "bearish"

    if text in {"long", "buy"}:
        return "bullish"

    if text in {"short", "sell"}:
        return "bearish"

    return text


def _same_direction(
    a: Any,
    b: Any,
) -> bool:

    da = _normalize_direction(
        _value(a, "direction")
    )

    db = _normalize_direction(
        _value(b, "direction")
    )

    return (
        bool(da)
        and bool(db)
        and da == db
    )


def _timestamp_key(
    value: Any,
) -> str:

    if value is None:
        return ""

    return str(value)


def _as_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def _events(
    events: Any,
) -> list[Any]:

    if events is None:
        return []

    return list(events)


# ============================================================
# FIND RELATED EVENTS
# ============================================================

def _find_retest_for_micro(
    micro: Any,
    retests: list[Any],
) -> Any:

    target_timestamp = _value(
        micro,
        "retest_timestamp",
    )

    direction = _normalize_direction(
        _value(micro, "direction")
    )

    matches = []

    for retest in retests:

        retest_timestamp = _value(
            retest,
            "retest_timestamp",
        )

        retest_direction = _normalize_direction(
            _value(retest, "direction")
        )

        if (
            _timestamp_key(retest_timestamp)
            == _timestamp_key(target_timestamp)
            and retest_direction == direction
        ):

            matches.append(retest)

    if not matches:
        return None

    return matches[-1]


def _find_mss_for_retest(
    retest: Any,
    mss_events: list[Any],
) -> Any:

    target_timestamp = _value(
        retest,
        "mss_timestamp",
    )

    direction = _normalize_direction(
        _value(retest, "direction")
    )

    candidates = []

    for event in mss_events:

        if _normalize_direction(
            _value(event, "direction")
        ) != direction:

            continue

        candidates.append(event)

    if not candidates:
        return None

    if target_timestamp is not None:

        exact = [
            event
            for event in candidates
            if _timestamp_key(
                _value(event, "timestamp")
            )
            == _timestamp_key(target_timestamp)
        ]

        if exact:
            return exact[-1]

    return candidates[-1]


def _find_displacement_for_retest(
    retest: Any,
    displacement_events: list[Any],
) -> Any:

    target_timestamp = _value(
        retest,
        "displacement_timestamp",
    )

    direction = _normalize_direction(
        _value(retest, "direction")
    )

    candidates = []

    for event in displacement_events:

        if _normalize_direction(
            _value(event, "direction")
        ) != direction:

            continue

        candidates.append(event)

    if not candidates:
        return None

    if target_timestamp is not None:

        exact = [
            event
            for event in candidates
            if _timestamp_key(
                _value(event, "timestamp")
            )
            == _timestamp_key(target_timestamp)
        ]

        if exact:
            return exact[-1]

    return candidates[-1]


def _find_zone_for_retest(
    retest: Any,
    zones: list[Any],
) -> Any:
    """
    Find the OB/FVG zone that this retest was built on.

    RetestEvent carries `zone_timestamp`. OB and FVG events
    carry `timestamp` (the zone creation time). Match on that.
    """

    zone_timestamp = _value(
        retest,
        "zone_timestamp",
    )

    if zone_timestamp is None:
        return None

    target_key = _timestamp_key(zone_timestamp)

    for zone in zones:

        zone_ts = _value(
            zone,
            "timestamp",
            "zone_timestamp",
            "created_at",
        )

        if _timestamp_key(zone_ts) == target_key:
            return zone

    return None


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def _validate_chain(
    micro: Any,
    retest: Any,
    displacement: Any,
    mss: Any,
):
    """
    Validate:

        MSS
          <
        Displacement
          <
        Retest
          <
        Micro-MSS

    Also preserves Micro-MSS V1.2 look-ahead rules.
    """

    reasons = []
    warnings = []

    direction = _normalize_direction(
        _value(micro, "direction")
    )

    if direction not in {
        "bullish",
        "bearish",
    }:

        return (
            False,
            reasons,
            ["Invalid Micro-MSS direction"],
        )

    # --------------------------------------------------------
    # Direction consistency
    # --------------------------------------------------------

    for name, event in (
        ("Retest", retest),
        ("Displacement", displacement),
        ("MSS", mss),
    ):

        if event is None:
            continue

        if not _same_direction(
            micro,
            event,
        ):

            return (
                False,
                reasons,
                [
                    f"Micro-MSS and {name} "
                    "directions differ"
                ],
            )

    # --------------------------------------------------------
    # Timestamp sequence
    # --------------------------------------------------------

    mss_timestamp = _value(
        mss,
        "timestamp",
        default=_value(
            micro,
            "mss_timestamp",
        ),
    )

    displacement_timestamp = _value(
        displacement,
        "timestamp",
        default=_value(
            micro,
            "displacement_timestamp",
        ),
    )

    retest_timestamp = _value(
        retest,
        "retest_timestamp",
    )

    micro_timestamp = _value(
        micro,
        "timestamp",
    )

    if all(
        x is not None
        for x in (
            mss_timestamp,
            displacement_timestamp,
            retest_timestamp,
            micro_timestamp,
        )
    ):

        if not (
            mss_timestamp
            < displacement_timestamp
            < retest_timestamp
            < micro_timestamp
        ):

            return (
                False,
                reasons,
                [
                    "Invalid sequence: "
                    "MSS < Displacement < Retest < Micro-MSS"
                ],
            )

    # --------------------------------------------------------
    # Micro-MSS V1.2 look-ahead validation
    # --------------------------------------------------------

    confirmed_at = _value(
        micro,
        "broken_swing_confirmed_at",
        "reference_confirmed_at",
    )

    if (
        confirmed_at is not None
        and retest_timestamp is not None
    ):

        if confirmed_at > retest_timestamp:

            return (
                False,
                reasons,
                [
                    "Reference swing was confirmed "
                    "after Retest opened"
                ],
            )

    # --------------------------------------------------------
    # Minimum reference distance
    # --------------------------------------------------------

    reference_distance = _as_float(
        _value(
            micro,
            "reference_distance_atr",
        )
    )

    if (
        reference_distance is not None
        and reference_distance < 0.25
    ):

        return (
            False,
            reasons,
            [
                "Reference swing distance "
                "is below 0.25 ATR"
            ],
        )

    reasons.append(
        "Valid MSS → Displacement → Retest → Micro-MSS chain"
    )

    if confirmed_at is not None:

        reasons.append(
            "Micro-MSS V1.2 reference "
            "confirmed before Retest"
        )

    if reference_distance is not None:

        reasons.append(
            f"Reference distance = "
            f"{reference_distance:.2f} ATR"
        )

    return (
        True,
        reasons,
        warnings,
    )


# ============================================================
# SCORE: STRUCTURE
# ============================================================

def _score_structure(
    micro: Any,
    context: Optional[dict[str, Any]],
) -> tuple[float, str]:
    """
    Maximum: 20

    Uses context["bias_by_timestamp"][micro_ts] when available.
    Falls back to context["bias"] (global) otherwise.
    """

    context = context or {}

    direction = _normalize_direction(
        _value(micro, "direction")
    )

    micro_ts = _timestamp_key(
        _value(micro, "timestamp")
    )

    bias_map = context.get("bias_by_timestamp", {})

    bias = _normalize_direction(
        bias_map.get(micro_ts)
        or context.get("bias")
        or context.get("market_bias")
    )

    if bias and bias == direction:
        return (
            20.0,
            f"4H bias ({bias}) agrees with entry",
        )

    if bias and bias in {"bullish", "bearish"} and bias != direction:
        return (
            5.0,
            f"4H bias ({bias}) opposes entry",
        )

    return (
        12.0,
        "No clear 4H bias",
    )


# ============================================================
# SCORE: LIQUIDITY
# ============================================================

def _score_liquidity(
    micro: Any,
    context: Optional[dict[str, Any]],
) -> tuple[float, str]:
    """
    Maximum: 15

    Uses context["sweeps_by_mss_timestamp"][mss_ts] when available.
    Falls back to context["sweep"] (global) otherwise.
    """

    context = context or {}

    mss_ts = _timestamp_key(
        _value(micro, "mss_timestamp")
    )

    sweep_map = context.get(
        "sweeps_by_mss_timestamp",
        {},
    )

    sweep = (
        sweep_map.get(mss_ts)
        or context.get("sweep")
    )

    if sweep is not None:

        depth_atr = _as_float(
            _value(sweep, "depth_atr")
        )

        # Deeper sweep = stronger liquidity grab
        if depth_atr is not None:

            if depth_atr >= 0.50:
                return (
                    15.0,
                    f"Strong sweep (depth = {depth_atr:.2f} ATR)",
                )

            if depth_atr >= 0.25:
                return (
                    13.0,
                    f"Normal sweep (depth = {depth_atr:.2f} ATR)",
                )

            return (
                10.0,
                f"Shallow sweep (depth = {depth_atr:.2f} ATR)",
            )

        return (
            15.0,
            "Liquidity sweep linked to this setup",
        )

    sweep_ts = _value(micro, "sweep_timestamp")

    if sweep_ts is not None:
        return (
            15.0,
            "Sweep timestamp present on Micro-MSS",
        )

    return (
        0.0,
        "No liquidity sweep linked",
    )


# ============================================================
# SCORE: MSS
# ============================================================

def _score_mss(
    mss: Any,
) -> tuple[float, str]:
    """
    Maximum: 15
    """

    if mss is None:
        return (
            0.0,
            "MSS could not be linked",
        )

    bars_after_sweep = _as_float(
        _value(
            mss,
            "bars_after_sweep",
        )
    )

    if bars_after_sweep is None:
        return (
            15.0,
            "Valid MSS linked",
        )

    if bars_after_sweep <= 2:
        return (
            15.0,
            "Fast MSS after sweep (<= 2 bars)",
        )

    if bars_after_sweep <= 5:
        return (
            13.0,
            "MSS within confirmation window",
        )

    return (
        10.0,
        "Late MSS within confirmation window",
    )


# ============================================================
# SCORE: DISPLACEMENT
# ============================================================

def _score_displacement(
    displacement: Any,
) -> tuple[float, str]:
    """
    Maximum: 15

    V1 requirements:
        Body / ATR >= 1.0
        Bullish CLV >= 0.70
        Bearish CLV <= 0.30
    """

    if displacement is None:
        return (
            0.0,
            "Displacement could not be linked",
        )

    body_atr = _as_float(
        _value(
            displacement,
            "body_atr",
            "body_over_atr",
        )
    )

    clv = _as_float(
        _value(
            displacement,
            "clv",
        )
    )

    if body_atr is None:
        score = 10.0

    elif body_atr >= 1.75:
        score = 15.0

    elif body_atr >= 1.50:
        score = 14.0

    elif body_atr >= 1.25:
        score = 12.0

    else:  # 1.0 <= body_atr < 1.25
        score = 10.0

    direction = _normalize_direction(
        _value(
            displacement,
            "direction",
        )
    )

    if clv is not None:

        clv_good = (
            (
                direction == "bullish"
                and clv >= 0.70
            )
            or
            (
                direction == "bearish"
                and clv <= 0.30
            )
        )

        if clv_good:
            score = min(
                15.0,
                score + 1.0,
            )

    return (
        score,
        "Displacement satisfies V1 Body/ATR and CLV conditions",
    )


# ============================================================
# SCORE: OB / FVG
# ============================================================

def _score_zone(
    zone: Any,
    retest: Any,
) -> tuple[float, str]:
    """
    Maximum: 10

    Distinguishes OB vs FVG by class name and uses
    penetration_ratio from the retest.
    """

    if zone is None:
        return (
            0.0,
            "OB/FVG zone could not be linked",
        )

    class_name = type(zone).__name__.lower()

    is_fvg = "fvg" in class_name

    is_ob = (
        "orderblock" in class_name
        or "order_block" in class_name
    )

    penetration = _as_float(
        _value(
            retest,
            "penetration_ratio",
            "penetration",
        )
    )

    # --------------------------------------------------------
    # FVG
    # --------------------------------------------------------
    if is_fvg:

        gap_atr = _as_float(
            _value(zone, "gap_atr")
        )

        base = 8.0

        if gap_atr is not None:

            if gap_atr >= 0.25:
                base = 10.0

            elif gap_atr >= 0.15:
                base = 9.0

        # Deep penetration weakens the zone
        if penetration is not None and penetration > 0.75:
            base = max(0.0, base - 1.0)

        return (
            base,
            "FVG linked with quality assessment",
        )

    # --------------------------------------------------------
    # OB
    # --------------------------------------------------------
    if is_ob:

        base = 9.0

        bars_before = _as_float(
            _value(
                zone,
                "bars_before_displacement",
            )
        )

        if bars_before is not None and bars_before <= 1:
            base = 10.0

        # Deep penetration weakens the zone
        if penetration is not None and penetration > 0.75:
            base = max(0.0, base - 1.0)

        return (
            base,
            "OB linked with quality assessment",
        )

    return (
        7.0,
        f"Zone linked but class unknown: {class_name}",
    )


# ============================================================
# SCORE: RETEST
# ============================================================

def _score_retest(
    retest: Any,
) -> tuple[float, str]:
    """
    Maximum: 10
    """

    if retest is None:
        return (
            0.0,
            "Retest missing",
        )

    close_through = _value(
        retest,
        "close_through",
    )

    close_inside = _value(
        retest,
        "close_inside",
    )

    penetration = _as_float(
        _value(
            retest,
            "penetration_ratio",
            "penetration",
        )
    )

    # A candle closing through the zone is a warning sign.
    if close_through is True:

        return (
            3.0,
            "Retest candle closed through the zone",
        )

    if close_inside is True:
        score = 8.0

    else:
        score = 6.0

    if penetration is not None:

        if penetration <= 0.50:
            score += 2.0

        elif penetration <= 0.75:
            score += 1.0

    return (
        min(
            10.0,
            score,
        ),
        "Retest respected the zone",
    )


# ============================================================
# SCORE: MICRO-MSS
# ============================================================

def _score_micro_mss(
    micro: Any,
) -> tuple[float, str]:
    """
    Maximum: 10
    """

    distance = _as_float(
        _value(
            micro,
            "reference_distance_atr",
        )
    )

    bars_after_retest = _as_float(
        _value(
            micro,
            "bars_after_retest",
        )
    )

    if distance is None:
        return (
            8.0,
            "Valid Micro-MSS V1.2",
        )

    score = 8.0

    if distance >= 1.5:
        score += 2.0

    elif distance >= 0.75:
        score += 1.0

    # Very late confirmation is slightly weaker.
    if (
        bars_after_retest is not None
        and bars_after_retest >= 17
    ):
        score -= 1.0

    score = max(
        0.0,
        min(
            10.0,
            score,
        ),
    )

    return (
        score,
        "Micro-MSS V1.2 is structurally valid",
    )


# ============================================================
# SCORE: RISK / REWARD
# ============================================================

def _score_risk_reward(
    context: Optional[dict[str, Any]],
) -> tuple[float, str]:
    """
    Maximum: 5

    V1 does not calculate SL/TP.

    If a precomputed RR is supplied by the caller,
    it can be scored.

    Otherwise neutral = 3/5.
    """

    context = context or {}

    rr = _as_float(
        context.get(
            "risk_reward",
            context.get("rr"),
        )
    )

    if rr is None:
        return (
            3.0,
            "RR not calculated yet; neutral score assigned",
        )

    if rr >= 3.0:
        return (
            5.0,
            f"RR = {rr:.2f}",
        )

    if rr >= 2.0:
        return (
            4.0,
            f"RR = {rr:.2f}",
        )

    if rr >= 1.5:
        return (
            3.0,
            f"RR = {rr:.2f}",
        )

    if rr >= 1.0:
        return (
            1.0,
            f"RR = {rr:.2f}",
        )

    return (
        0.0,
        f"RR = {rr:.2f} below baseline",
    )


# ============================================================
# CALCULATE COMPLETE SCORE
# ============================================================

def calculate_entry_score(
    micro_mss: Any,
    retest: Any,
    mss: Any = None,
    displacement: Any = None,
    zone: Any = None,
    context: Optional[dict[str, Any]] = None,
):
    """
    Calculate complete 100-point Entry Score.
    """

    score = EntryScore()

    reasons = []

    warnings = []

    # Structure
    score.structure, text = _score_structure(
        micro_mss,
        context,
    )

    reasons.append(text)

    # Liquidity
    (
        score.liquidity_sweep,
        text,
    ) = _score_liquidity(
        micro_mss,
        context,
    )

    reasons.append(text)

    # MSS
    score.mss, text = _score_mss(
        mss,
    )

    reasons.append(text)

    # Displacement
    (
        score.displacement,
        text,
    ) = _score_displacement(
        displacement,
    )

    reasons.append(text)

    # OB/FVG
    (
        score.zone_quality,
        text,
    ) = _score_zone(
        zone,
        retest,
    )

    reasons.append(text)

    # Retest
    (
        score.retest_quality,
        text,
    ) = _score_retest(
        retest,
    )

    reasons.append(text)

    # Micro-MSS
    (
        score.micro_mss,
        text,
    ) = _score_micro_mss(
        micro_mss,
    )

    reasons.append(text)

    # RR
    (
        score.risk_reward,
        text,
    ) = _score_risk_reward(
        context,
    )

    reasons.append(text)

    if score.total < MIN_ENTRY_SCORE:

        warnings.append(
            f"Score {score.total:.2f} "
            f"is below minimum tradable score "
            f"{MIN_ENTRY_SCORE:.0f}"
        )

    return (
        score,
        reasons,
        warnings,
    )


# ============================================================
# MAIN ENTRY DETECTOR
# ============================================================

def detect_entry_candidates(
    micro_mss_events: Iterable[Any],
    retests: Iterable[Any],
    mss_events: Iterable[Any],
    displacement_events: Iterable[Any],
    ob_events: Iterable[Any] | None = None,
    fvg_events: Iterable[Any] | None = None,
    *,
    context: Optional[dict[str, Any]] = None,
    min_score: float = MIN_ENTRY_SCORE,
) -> list[EntryCandidate]:
    """
    Build EntryCandidate objects from the locked pipeline.

    Required sequence:

        MSS
        ↓
        Displacement
        ↓
        Retest
        ↓
        Micro-MSS V1.2

    Only valid chains become candidates.
    """

    micro_events = sorted(
        _events(micro_mss_events),
        key=lambda event: _timestamp_key(
            _value(event, "timestamp")
        ),
    )

    retest_events = _events(retests)

    mss_list = _events(mss_events)

    displacement_list = _events(displacement_events)

    zones = (
        _events(ob_events)
        + _events(fvg_events)
    )

    candidates = []

    seen = set()

    for micro in micro_events:

        # ----------------------------------------------------
        # Find Retest
        # ----------------------------------------------------

        retest = _find_retest_for_micro(
            micro,
            retest_events,
        )

        if retest is None:
            continue

        # ----------------------------------------------------
        # Find MSS
        # ----------------------------------------------------

        mss = _find_mss_for_retest(
            retest,
            mss_list,
        )

        # ----------------------------------------------------
        # Find Displacement
        # ----------------------------------------------------

        displacement = (
            _find_displacement_for_retest(
                retest,
                displacement_list,
            )
        )

        # ----------------------------------------------------
        # Find OB / FVG
        # ----------------------------------------------------

        zone = _find_zone_for_retest(
            retest,
            zones,
        )

        # ----------------------------------------------------
        # Validate complete chain
        # ----------------------------------------------------

        (
            valid,
            chain_reasons,
            chain_warnings,
        ) = _validate_chain(
            micro,
            retest,
            displacement,
            mss,
        )

        if not valid:
            continue

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        direction = _normalize_direction(
            _value(micro, "direction")
        )

        # ----------------------------------------------------
        # Zone type
        # ----------------------------------------------------

        zone_type = str(
            _value(
                retest,
                "zone_type",
                default="",
            )
        ).lower()

        if not zone_type and zone is not None:

            class_name = type(zone).__name__.lower()

            if "fvg" in class_name:
                zone_type = "fvg"

            elif (
                "orderblock" in class_name
                or "order_block" in class_name
            ):
                zone_type = "ob"

        if not zone_type:
            zone_type = "unknown"

        # ----------------------------------------------------
        # Dedup
        # ----------------------------------------------------

        micro_timestamp = _value(
            micro,
            "timestamp",
        )

        dedup_key = (
            _timestamp_key(micro_timestamp),
            direction,
            zone_type,
        )

        if dedup_key in seen:
            continue

        seen.add(dedup_key)

        # ----------------------------------------------------
        # Entry price
        # ----------------------------------------------------

        entry_price = _as_float(
            _value(
                micro,
                "entry_price",
                "close",
            )
        )

        if (
            entry_price is None
            and context is not None
        ):

            entry_prices = context.get(
                "entry_prices",
                {},
            )

            entry_price = _as_float(
                entry_prices.get(
                    _timestamp_key(micro_timestamp)
                )
            )

        if entry_price is None:
            continue

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        (
            score,
            score_reasons,
            score_warnings,
        ) = calculate_entry_score(
            micro_mss=micro,
            retest=retest,
            mss=mss,
            displacement=displacement,
            zone=zone,
            context=context,
        )

        # ----------------------------------------------------
        # Build candidate
        # ----------------------------------------------------

        candidate = EntryCandidate(

            timestamp=micro_timestamp,

            direction=direction,

            entry_price=entry_price,

            zone_type=zone_type,

            mss_timestamp=_value(
                mss,
                "timestamp",
                default=_value(
                    micro,
                    "mss_timestamp",
                ),
            ),

            displacement_timestamp=_value(
                displacement,
                "timestamp",
                default=_value(
                    micro,
                    "displacement_timestamp",
                ),
            ),

            retest_timestamp=_value(
                retest,
                "retest_timestamp",
            ),

            micro_mss_timestamp=micro_timestamp,

            zone_high=_as_float(
                _value(
                    retest,
                    "zone_high",
                )
            ),

            zone_low=_as_float(
                _value(
                    retest,
                    "zone_low",
                )
            ),

            score=score.total,

            grade=score.grade,

            tradable=(
                score.total >= min_score
            ),

            score_breakdown=score.to_dict(),

            reasons=(
                chain_reasons
                + score_reasons
            ),

            warnings=(
                chain_warnings
                + score_warnings
            ),
        )

        candidates.append(candidate)

    return candidates


# ============================================================
# TRADABLE FILTER
# ============================================================

def get_tradable_entries(
    candidates: Iterable[EntryCandidate],
    *,
    min_score: float = MIN_ENTRY_SCORE,
) -> list[EntryCandidate]:

    return [
        candidate
        for candidate in candidates
        if (
            candidate.tradable
            and candidate.score >= min_score
        )
    ]


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_entries(
    candidates: Iterable[EntryCandidate],
) -> list[EntryCandidate]:

    result = []

    seen = set()

    for candidate in candidates:

        key = (
            _timestamp_key(candidate.timestamp),
            candidate.direction,
            candidate.zone_type,
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(candidate)

    return result


# ============================================================
# DATAFRAME
# ============================================================

def entries_to_dataframe(
    candidates: Iterable[EntryCandidate],
):
    """
    Convert EntryCandidate objects to pandas DataFrame.
    """

    import pandas as pd

    rows = []

    for candidate in candidates:

        row = candidate.to_dict()

        breakdown = row.pop(
            "score_breakdown",
            {},
        )

        for key, value in breakdown.items():

            row[f"score_{key}"] = value

        row["reasons"] = (
            " | ".join(candidate.reasons)
        )

        row["warnings"] = (
            " | ".join(candidate.warnings)
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "MIN_ENTRY_SCORE",
    "SCORE_WEIGHTS",
    "EntryScore",
    "EntryCandidate",
    "calculate_entry_score",
    "detect_entry_candidates",
    "get_tradable_entries",
    "deduplicate_entries",
    "entries_to_dataframe",
]