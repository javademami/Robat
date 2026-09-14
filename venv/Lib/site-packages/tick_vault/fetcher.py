"""HTTP fetching module for Dukascopy data with comprehensive error handling."""

import asyncio

from httpx import (
    AsyncClient,
    ConnectError,
    NetworkError,
    ProtocolError,
    RequestError,
    TimeoutException,
)

from .config import CONFIG
from .logger import logger


# Custom exceptions for different error categories
class FetchError(Exception):
    """Base exception for fetch errors."""


class RateLimitError(FetchError):
    """Raised when rate limiting is detected (429, 503 with Retry-After)."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ForbiddenError(FetchError):
    """Raised when access is forbidden or blocked (401, 403, 451)."""


class RetryableError(FetchError):
    """Raised for transient errors that should be retried."""


async def _fetch(client: AsyncClient, url: str) -> bytes | None:
    """
    Fetch data from a URL with comprehensive error handling.

    Args:
        client: An httpx AsyncClient instance
        url: The URL to fetch

    Returns:
        bytes: The response content if successful (status 200 with data)
        None: If the resource doesn't exist (status 404)

    Raises:
        RateLimitError: When rate limiting is detected (429 or specific 503)
        ForbiddenError: When access is denied/blocked (401, 403, 451)
        RetryableError: For transient network/server errors
        RuntimeError: For unexpected status codes or errors
    """
    try:
        response = await client.get(url)

        # Success case with data
        if response.status_code == 200:
            logger.debug(
                f"Successfully fetched {len(response.content)} bytes from {url}"
            )
            if response.content:
                return response.content
            else:
                # Empty response with 200 - unexpected but treat as no data
                return None

        # Resource not found - legitimate "no data" case
        elif response.status_code == 404:
            logger.debug(f"No data found (404) for URL: {url}")
            return None

        # Rate limiting
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_seconds = (
                int(retry_after) if retry_after and retry_after.isdigit() else None
            )
            logger.warning(
                f"Rate limited (429) for {url}, retry_after={retry_seconds}s"
            )
            raise RateLimitError(
                f"Rate limited (429) for URL: {url}", retry_after=retry_seconds
            )

        # Forbidden/blocked access
        elif response.status_code in (401, 403, 451):
            logger.error(f"Access forbidden ({response.status_code}) for {url}")
            raise ForbiddenError(
                f"Access forbidden (status {response.status_code}) for URL: {url}. "
                "Possible IP ban or authentication issue."
            )

        # Server errors that might indicate rate limiting or temporary issues
        elif response.status_code == 503:
            # 503 can be rate limiting or temporary unavailability
            retry_after = response.headers.get("Retry-After")
            logger.warning(
                f"Service unavailable (503) for {url}, retry_after={retry_after}"
            )
            if retry_after:
                retry_seconds = int(retry_after) if retry_after.isdigit() else None
                raise RateLimitError(
                    f"Service unavailable with Retry-After (503) for URL: {url}",
                    retry_after=retry_seconds,
                )
            else:
                raise RetryableError(
                    f"Service temporarily unavailable (503) for URL: {url}"
                )

        # Other server errors (500, 502, 504) - potentially retryable
        elif 500 <= response.status_code < 600:
            logger.warning(
                f"Server error ({response.status_code}) for {url}, will retry"
            )
            raise RetryableError(
                f"Server error (status {response.status_code}) for URL: {url}"
            )

        # Client errors (other 4xx) - unexpected, needs investigation
        elif 400 <= response.status_code < 500:
            raise RuntimeError(
                f"Unexpected client error (status {response.status_code}) for URL: {url}"
            )

        # Any other unexpected status code
        else:
            raise RuntimeError(
                f"Unexpected status code {response.status_code} for URL: {url}"
            )

    # Network timeouts (includes ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout)
    except TimeoutException as e:
        logger.warning(f"Timeout while fetching {url}")
        raise RetryableError(f"Request timeout for URL: {url}") from e

    # Connection errors and DNS failures - retryable
    except ConnectError as e:
        logger.warning(f"Network error for {url}: {type(e).__name__}")
        raise RetryableError(f"Connection error for URL: {url}") from e

    # Other network-level errors - retryable
    except NetworkError as e:
        logger.warning(f"Network error for {url}: {type(e).__name__}")
        raise RetryableError(f"Network error for URL: {url}") from e

    # Protocol errors - unexpected
    except ProtocolError as e:
        raise RuntimeError(f"Protocol error for URL: {url}") from e

    # Catch-all for any other httpx exceptions
    except RequestError as e:
        # If we haven't caught it above, treat as unexpected
        raise RuntimeError(f"Unexpected request error for URL: {url}: {e}") from e

    # Catch any other unexpected exceptions
    except Exception as e:
        raise RuntimeError(
            f"Unexpected exception while fetching URL: {url}: {type(e).__name__}: {e}"
        ) from e


async def fetch_with_retry(client: AsyncClient, url: str) -> bytes | None:
    """
    Fetch data from a URL with automatic retry logic for transient failures.

    This function wraps _fetch() and implements exponential backoff retry logic
    for rate limiting and transient network errors. Fatal errors (ForbiddenError)
    are raised immediately without retry attempts.

    Args:
        client: An httpx AsyncClient instance for making HTTP requests
        url: The URL to fetch data from

    Returns:
        bytes: The response content if successful (status 200 with data)
        None: If the resource doesn't exist (status 404)

    Raises:
        ForbiddenError: When access is denied/blocked (401, 403, 451).
            Raised immediately without retry.
        RuntimeError: When max retry attempts are exhausted for RateLimitError
            or RetryableError, or when any unexpected error occurs.

    Retry behavior:
        - RateLimitError: Uses server's Retry-After header if available,
          otherwise applies exponential backoff
        - RetryableError: Applies exponential backoff with base delay from config
        - Max attempts: CONFIG.fetch_max_retry_attempts + 1 (initial + retries)
        - Backoff formula: base_delay * (2 ** attempt)
    """
    logger.debug(f"Fetching URL: {url}")

    for attempt in range(CONFIG.fetch_max_retry_attempts + 1):
        try:
            return await _fetch(client, url)

        except RateLimitError as e:
            if attempt == CONFIG.fetch_max_retry_attempts:
                raise RuntimeError(
                    f"Max retries ({CONFIG.fetch_max_retry_attempts}) exceeded for URL: {url}"
                ) from e

            # Use server-suggested delay or exponential backoff
            delay = (
                e.retry_after
                if e.retry_after
                else CONFIG.fetch_base_retry_delay * (2**attempt)
            )
            logger.info(
                f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{CONFIG.fetch_max_retry_attempts + 1})"
            )
            await asyncio.sleep(delay)
            continue

        except RetryableError as e:
            if attempt == CONFIG.fetch_max_retry_attempts:
                raise RuntimeError(
                    f"Max retries ({CONFIG.fetch_max_retry_attempts}) exceeded for URL: {url}"
                ) from e

            # Exponential backoff
            delay = CONFIG.fetch_base_retry_delay * (2**attempt)
            logger.info(
                f"Retryable error, backing off {delay:.1f}s (attempt {attempt + 1}/{CONFIG.fetch_max_retry_attempts + 1})"
            )
            await asyncio.sleep(delay)
            continue

    # Should never reach here, but for type safety
    logger.error(f"Max retries exhausted for {url}")
    raise RuntimeError(f"Unexpected code path for URL: {url}")
