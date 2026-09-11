from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import (
    BINANCE_FUTURES_URL,
    MAX_LIMIT,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    REQUEST_DELAY_SECONDS,
)


KLINE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class BinanceDownloader:

    def __init__(
        self,
        base_url: str = BINANCE_FUTURES_URL,
    ):

        self.base_url = base_url

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": "Robat-CryptoBot/1.0"
        })


    @staticmethod
    def datetime_to_ms(
        value: datetime,
    ) -> int:

        if value.tzinfo is None:
            value = value.replace(
                tzinfo=timezone.utc
            )

        return int(
            value.timestamp() * 1000
        )


    @staticmethod
    def ms_to_datetime(
        value: int,
    ) -> pd.Timestamp:

        return pd.to_datetime(
            value,
            unit="ms",
            utc=True,
        )


    def fetch_batch(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }

        last_error = None

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            try:

                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 429:

                    wait = min(
                        2 ** attempt,
                        30,
                    )

                    print(
                        f"Rate limited. "
                        f"Waiting {wait}s..."
                    )

                    time.sleep(wait)

                    continue

                response.raise_for_status()

                data = response.json()

                if not isinstance(data, list):
                    raise RuntimeError(
                        f"Unexpected Binance response: {data}"
                    )

                return data

            except (
                requests.RequestException,
                ValueError,
                RuntimeError,
            ) as exc:

                last_error = exc

                wait = min(
                    2 ** attempt,
                    30,
                )

                print(
                    f"Request failed "
                    f"(attempt {attempt}/"
                    f"{MAX_RETRIES}): {exc}"
                )

                time.sleep(wait)

        raise RuntimeError(
            "Failed to download Binance data"
        ) from last_error


    def download_range(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:

        start_ms = self.datetime_to_ms(start)
        end_ms = self.datetime_to_ms(end)

        all_rows = []

        current_start = start_ms

        batch_number = 0

        while current_start < end_ms:

            batch_number += 1

            print(
                f"[{batch_number}] "
                f"{symbol} {interval} "
                f"{self.ms_to_datetime(current_start)}"
            )

            rows = self.fetch_batch(
                symbol=symbol,
                interval=interval,
                start_ms=current_start,
                end_ms=end_ms,
            )

            if not rows:
                print("No more data.")
                break

            all_rows.extend(rows)

            last_open_time = rows[-1][0]

            next_start = last_open_time + 1

            if next_start <= current_start:
                raise RuntimeError(
                    "Downloader cursor did not advance"
                )

            current_start = next_start

            if len(rows) < MAX_LIMIT:
                break

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

        if not all_rows:
            return pd.DataFrame(
                columns=KLINE_COLUMNS
            )

        df = self._to_dataframe(all_rows)

        start_timestamp = pd.to_datetime(
            start,
            utc=True,
        )

        end_timestamp = pd.to_datetime(
            end,
            utc=True,
        )

        df = df[
            (df["timestamp"] >= start_timestamp)
            &
            (df["timestamp"] < end_timestamp)
        ]

        df = (
            df
            .drop_duplicates(
                subset=["timestamp"]
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        return df


    @staticmethod
    def _to_dataframe(
        rows: list,
    ) -> pd.DataFrame:

        data = []

        for row in rows:

            data.append({
                "timestamp": pd.to_datetime(
                    row[0],
                    unit="ms",
                    utc=True,
                ),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })

        return pd.DataFrame(
            data,
            columns=KLINE_COLUMNS,
        )
