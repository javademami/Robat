from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_columns(df: pd.DataFrame) -> None:

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )


def validate_numeric(df: pd.DataFrame) -> None:

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise ValueError(
                f"Column {column} is not numeric"
            )


def validate_timestamp(df: pd.DataFrame) -> None:

    if not pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    ):
        raise ValueError(
            "Timestamp column is not datetime"
        )

    if df["timestamp"].dt.tz is None:
        raise ValueError(
            "Timestamp must be timezone-aware"
        )


def validate_ohlc(df: pd.DataFrame) -> None:

    invalid_high = (
        df["high"]
        < df[["open", "close", "low"]].max(axis=1)
    )

    invalid_low = (
        df["low"]
        > df[["open", "close", "high"]].min(axis=1)
    )

    if invalid_high.any():
        raise ValueError(
            "Invalid OHLC: high is below "
            "open/close/low"
        )

    if invalid_low.any():
        raise ValueError(
            "Invalid OHLC: low is above "
            "open/close/high"
        )

    if (df["volume"] < 0).any():
        raise ValueError(
            "Negative volume detected"
        )

    if (df["open"] <= 0).any():
        raise ValueError("Invalid open price")

    if (df["high"] <= 0).any():
        raise ValueError("Invalid high price")

    if (df["low"] <= 0).any():
        raise ValueError("Invalid low price")

    if (df["close"] <= 0).any():
        raise ValueError("Invalid close price")


def validate_sorted(df: pd.DataFrame) -> None:

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not sorted"
        )


def find_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return df[
        df["timestamp"].duplicated(
            keep=False
        )
    ]


def find_missing_candles(
    df: pd.DataFrame,
    interval_minutes: int,
) -> pd.DataFrame:

    if len(df) < 2:
        return pd.DataFrame()

    expected_delta = pd.Timedelta(
        minutes=interval_minutes
    )

    deltas = df["timestamp"].diff().iloc[1:]

    missing = deltas[
        deltas != expected_delta
    ]

    return missing.to_frame(name="delta")


def validate_dataframe(
    df: pd.DataFrame,
    interval_minutes: int = 5,
) -> dict:

    validate_columns(df)

    if df.empty:
        return {
            "rows": 0,
            "duplicates": 0,
            "gaps": 0,
            "start": None,
            "end": None,
            "valid": False,
        }

    validate_timestamp(df)
    validate_numeric(df)
    validate_ohlc(df)
    validate_sorted(df)

    duplicates = find_duplicates(df)

    missing = find_missing_candles(
        df,
        interval_minutes,
    )

    return {
        "rows": len(df),
        "duplicates": len(duplicates),
        "gaps": len(missing),
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "valid": (
            len(duplicates) == 0
            and len(missing) == 0
        ),
    }
