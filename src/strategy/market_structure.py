from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategy.swing_detector import (
    SwingPoint,
)


@dataclass(frozen=True)
class StructureEvent:
    """
    Market structure event.
    """

    timestamp: pd.Timestamp
    confirmed_at: pd.Timestamp
    kind: str
    price: float
    previous_price: float | None


@dataclass(frozen=True)
class MarketStructureState:
    """
    Current market structure state.
    """

    bias: str

    last_high: SwingPoint | None
    previous_high: SwingPoint | None

    last_low: SwingPoint | None
    previous_low: SwingPoint | None

    events: tuple[StructureEvent, ...]


def classify_swings(
    swings: list[SwingPoint],
) -> list[StructureEvent]:
    """
    Classify confirmed swing points into:

        HH = Higher High
        LH = Lower High
        HL = Higher Low
        LL = Lower Low

    Highs are compared only against previous confirmed highs.
    Lows are compared only against previous confirmed lows.
    """

    events: list[StructureEvent] = []

    previous_high: SwingPoint | None = None
    previous_low: SwingPoint | None = None

    for swing in swings:

        if swing.kind == "high":

            if previous_high is not None:

                if swing.price > previous_high.price:

                    events.append(
                        StructureEvent(
                            timestamp=swing.timestamp,
                            confirmed_at=swing.confirmed_at,
                            kind="HH",
                            price=swing.price,
                            previous_price=previous_high.price,
                        )
                    )

                elif swing.price < previous_high.price:

                    events.append(
                        StructureEvent(
                            timestamp=swing.timestamp,
                            confirmed_at=swing.confirmed_at,
                            kind="LH",
                            price=swing.price,
                            previous_price=previous_high.price,
                        )
                    )

            previous_high = swing

        elif swing.kind == "low":

            if previous_low is not None:

                if swing.price > previous_low.price:

                    events.append(
                        StructureEvent(
                            timestamp=swing.timestamp,
                            confirmed_at=swing.confirmed_at,
                            kind="HL",
                            price=swing.price,
                            previous_price=previous_low.price,
                        )
                    )

                elif swing.price < previous_low.price:

                    events.append(
                        StructureEvent(
                            timestamp=swing.timestamp,
                            confirmed_at=swing.confirmed_at,
                            kind="LL",
                            price=swing.price,
                            previous_price=previous_low.price,
                        )
                    )

            previous_low = swing

    events.sort(
        key=lambda event: event.confirmed_at
    )

    return events


def determine_bias(
    events: list[StructureEvent],
) -> str:
    """
    Determine broad market bias.

    Bullish:
        latest confirmed high is HH
        AND latest confirmed low is HL

    Bearish:
        latest confirmed high is LH
        AND latest confirmed low is LL

    Otherwise:
        Neutral
    """

    latest_high_event = None
    latest_low_event = None

    for event in events:

        if event.kind in {"HH", "LH"}:
            latest_high_event = event

        elif event.kind in {"HL", "LL"}:
            latest_low_event = event

    if (
        latest_high_event is None
        or latest_low_event is None
    ):
        return "NEUTRAL"

    if (
        latest_high_event.kind == "HH"
        and latest_low_event.kind == "HL"
    ):
        return "BULLISH"

    if (
        latest_high_event.kind == "LH"
        and latest_low_event.kind == "LL"
    ):
        return "BEARISH"

    return "NEUTRAL"


def build_structure_state(
    swings: list[SwingPoint],
    current_time: pd.Timestamp | None = None,
) -> MarketStructureState:
    """
    Build structure state using only swings confirmed
    by current_time.
    """

    if current_time is not None:

        current_time = pd.Timestamp(
            current_time
        )

        if current_time.tzinfo is None:

            current_time = current_time.tz_localize(
                "UTC"
            )

        else:

            current_time = current_time.tz_convert(
                "UTC"
            )

        confirmed_swings = [
            swing
            for swing in swings
            if swing.confirmed_at <= current_time
        ]

    else:

        confirmed_swings = list(swings)

    confirmed_swings.sort(
        key=lambda swing: swing.confirmed_at
    )

    events = classify_swings(
        confirmed_swings
    )

    high_swings = [
        swing
        for swing in confirmed_swings
        if swing.kind == "high"
    ]

    low_swings = [
        swing
        for swing in confirmed_swings
        if swing.kind == "low"
    ]

    last_high = (
        high_swings[-1]
        if high_swings
        else None
    )

    previous_high = (
        high_swings[-2]
        if len(high_swings) >= 2
        else None
    )

    last_low = (
        low_swings[-1]
        if low_swings
        else None
    )

    previous_low = (
        low_swings[-2]
        if len(low_swings) >= 2
        else None
    )

    return MarketStructureState(
        bias=determine_bias(events),
        last_high=last_high,
        previous_high=previous_high,
        last_low=last_low,
        previous_low=previous_low,
        events=tuple(events),
    )
