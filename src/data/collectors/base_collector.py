"""Base collector abstract class and common utilities."""

import time
from abc import ABC, abstractmethod
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry


class CollectorError(Exception):
    """Base exception for collector errors."""

    pass


class RateLimitError(CollectorError):
    """Raised when rate limit is hit."""

    pass


class ParseError(CollectorError):
    """Raised when parsing fails."""

    pass


class BaseCollector(ABC):
    """Abstract base class for all data collectors.

    Provides common functionality:
    - HTTP session with retry logic
    - Rate limiting
    - Error handling
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: int = 30,
        rate_limit_per_minute: int = 30,
    ) -> None:
        """Initialize the collector.

        Args:
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
            rate_limit_per_minute: Maximum requests per minute
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.rate_limit_per_minute = rate_limit_per_minute

        # Configure session with retry strategy
        self.session = self._create_session()

        # Rate limiting
        self._request_times: list[float] = []
        self._burst_tokens: int = 5  # Allow burst of 5 requests
        self._burst_last_refill: float = time.time()

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry configuration.

        Returns:
            Configured requests Session
        """
        session = requests.Session()

        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting.

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        now = time.time()

        # Refill burst tokens (1 token per second)
        elapsed = now - self._burst_last_refill
        self._burst_tokens = min(5, self._burst_tokens + int(elapsed))
        self._burst_last_refill = now

        if self._burst_tokens <= 0:
            raise RateLimitError("Rate limit exceeded: burst tokens exhausted")

        # Clean old request times (older than 1 minute)
        self._request_times = [t for t in self._request_times if now - t < 60]

        # Check per-minute limit
        if len(self._request_times) >= self.rate_limit_per_minute:
            raise RateLimitError(
                f"Rate limit exceeded: {len(self._request_times)} "
                f"requests in last 60 seconds (limit: {self.rate_limit_per_minute})"
            )

        # Use a burst token
        self._burst_tokens -= 1

    def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Make HTTP request with error handling and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            CollectorError: If request fails after retries
            RateLimitError: If rate limit is exceeded
        """
        self._check_rate_limit()

        # Set default timeout
        kwargs.setdefault("timeout", self.timeout)

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()

                # Record successful request
                self._request_times.append(time.time())

                return response

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))

            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response is not None and e.response.status_code == 429:
                    raise RateLimitError(f"Rate limit error from {url}") from e
                if attempt < self.max_retries and e.response is not None and e.response.status_code >= 500:
                    time.sleep(self.retry_delay * (attempt + 1))

            except RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise CollectorError(
            f"Request failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Make GET request.

        Args:
            url: Request URL
            **kwargs: Additional arguments

        Returns:
            Response object
        """
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """Make POST request.

        Args:
            url: Request URL
            **kwargs: Additional arguments

        Returns:
            Response object
        """
        return self._request("POST", url, **kwargs)

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self) -> "BaseCollector":
        """Context manager entry."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Context manager exit."""
        self.close()


def clean_text(text: str) -> str:
    """Clean and normalize text content.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    import re

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_stock_codes(text: str) -> list[str]:
    """Extract stock codes from text.

    Args:
        text: Input text

    Returns:
        List of stock codes (e.g., ["600519.SH", "000858.SZ"])
    """
    import re

    codes: list[str] = []

    # Match patterns like 600519.SH, 000858.SZ
    pattern = r"\b(\d{6})\.(SH|SZ)\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    for code, exchange in matches:
        codes.append(f"{code}.{exchange.upper()}")

    # Match standalone 6-digit codes (need to determine exchange)
    pattern2 = r"\b(\d{6})\b"
    matches2 = re.findall(pattern2, text)

    # A-share code ranges:
    # SH: 600xxx, 601xxx, 603xxx, 605xxx, 688xxx (STAR)
    # SZ: 000xxx, 001xxx, 002xxx, 003xxx, 300xxx (ChiNext)
    for code in matches2:
        if code.startswith(("600", "601", "603", "605", "688")):
            codes.append(f"{code}.SH")
        elif code.startswith(("000", "001", "002", "003", "300")):
            codes.append(f"{code}.SZ")

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            result.append(code)

    return result
