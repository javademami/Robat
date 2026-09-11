from __future__ import annotations

from pathlib import Path
import pandas as pd


OHLCV_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def load_5m_file(path: Path) -> pd.DataFrame:
    """
    Load canonical 5M dataset.
    All timestamps are normalized to UTC.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"5M data file not found: {path}"
        )

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(
            f"5M dataset is empty: {path}"
        )

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """
    Convert 5M candles into a higher timeframe.

    Supported:
        1h
        4h

    The resulting candle is only considered complete
    when its entire timeframe has elapsed.
    """

    if timeframe not in {"1h", "4h"}:
        raise ValueError(
            "Supported timeframes: 1h, 4h"
        )

    data = df.copy()

    data = data.set_index("timestamp")

    result = (
        data[OHLCV_COLUMNS]
        .resample(
            timeframe,
            label="left",
            closed="left",
        )
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
    )

    result = result.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    result = result.reset_index()

    return result


def remove_incomplete_last_candle(
    df: pd.DataFrame,
    source_end: pd.Timestamp,
    timeframe: str,
) -> pd.DataFrame:
    """
    Remove the final higher-timeframe candle if it is
    not fully closed according to the source data.
    """

    if df.empty:
        return df

    duration = pd.Timedelta(timeframe)

    last_start = df["timestamp"].iloc[-1]

    last_close_time = (
        last_start + duration
    )

    if last_close_time > source_end:
        return df.iloc[:-1].reset_index(
            drop=True
        )

    return df


def build_timeframes(
    df_5m: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build 1H and 4H datasets from canonical 5M data.
    """

    if df_5m.empty:
        raise ValueError(
            "Cannot aggregate empty 5M data"
        )

    source_end = df_5m["timestamp"].iloc[-1]

    df_1h = resample_ohlcv(
        df_5m,
        "1h",
    )

    df_4h = resample_ohlcv(
        df_5m,
        "4h",
    )

    df_1h = remove_incomplete_last_candle(
        df_1h,
        source_end,
        "1h",
    )

    df_4h = remove_incomplete_last_candle(
        df_4h,
        source_end,
        "4h",
    )

    return {
        "5m": df_5m,
        "1h": df_1h,
        "4h": df_4h,
    }


def save_timeframes(
    timeframes: dict[str, pd.DataFrame],
    output_dir: Path,
    year: int,
) -> dict[str, Path]:
    """
    Save derived timeframe datasets.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = {}

    for timeframe, df in timeframes.items():

        if timeframe == "5m":
            continue

        path = (
            output_dir
            / f"{year}_{timeframe}.parquet"
        )

        df.to_parquet(
            path,
            index=False,
            engine="pyarrow",
        )

        saved[timeframe] = path

    return saved
