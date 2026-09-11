from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def get_symbol_directory(
    data_dir: Path,
    symbol: str,
    interval: str,
) -> Path:

    directory = data_dir / symbol / interval
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def get_file_path(
    data_dir: Path,
    symbol: str,
    interval: str,
    year: int,
) -> Path:

    directory = get_symbol_directory(
        data_dir,
        symbol,
        interval,
    )

    return directory / f"{year}.parquet"


def save_year(
    df: pd.DataFrame,
    data_dir: Path,
    symbol: str,
    interval: str,
    year: int,
) -> Path:

    if df.empty:
        raise ValueError("Cannot save empty dataframe")

    path = get_file_path(
        data_dir,
        symbol,
        interval,
        year,
    )

    df.to_parquet(
        path,
        index=False,
        engine="pyarrow",
    )

    return path


def load_year(
    data_dir: Path,
    symbol: str,
    interval: str,
    year: int,
) -> pd.DataFrame:

    path = get_file_path(
        data_dir,
        symbol,
        interval,
        year,
    )

    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.read_parquet(path)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

    return df
