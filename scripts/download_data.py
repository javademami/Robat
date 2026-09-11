from datetime import datetime, timezone

from src.config import (
    DATA_DIR,
    DEFAULT_INTERVAL,
)

from src.data.downloader import (
    BinanceDownloader,
)

from src.data.storage import (
    save_year,
)

from src.data.validator import (
    validate_dataframe,
)


def download_symbol_year(
    symbol: str,
    year: int,
):

    start = datetime(
        year,
        1,
        1,
        tzinfo=timezone.utc,
    )

    end = datetime(
        year + 1,
        1,
        1,
        tzinfo=timezone.utc,
    )

    downloader = BinanceDownloader()

    print()
    print("=" * 70)
    print(
        f"Downloading {symbol} "
        f"{DEFAULT_INTERVAL} "
        f"{year}"
    )
    print("=" * 70)

    df = downloader.download_range(
        symbol=symbol,
        interval=DEFAULT_INTERVAL,
        start=start,
        end=end,
    )

    if df.empty:
        print(
            f"No data returned for "
            f"{symbol} {year}"
        )
        return

    report = validate_dataframe(
        df,
        interval_minutes=5,
    )

    print()
    print("Validation")
    print("-" * 40)
    print(f"Rows:       {report['rows']:,}")
    print(f"Duplicates: {report['duplicates']}")
    print(f"Gaps:       {report['gaps']}")
    print(f"Start:      {report['start']}")
    print(f"End:        {report['end']}")
    print(f"Valid:      {report['valid']}")

    if not report["valid"]:
        raise RuntimeError(
            "Dataset validation failed"
        )

    path = save_year(
        df=df,
        data_dir=DATA_DIR,
        symbol=symbol,
        interval=DEFAULT_INTERVAL,
        year=year,
    )

    print()
    print("=" * 70)
    print(f"Saved successfully:")
    print(path)
    print("=" * 70)


if __name__ == "__main__":

    download_symbol_year(
        symbol="BTCUSDT",
        year=2025,
    )
