"""
HTTP fetching for Genius lyrics pages.

This module intentionally contains only network logic:
- requests
- retries
- backoff

No parsing. No heuristics. No Genius-specific semantics.
"""

import time
import random
from typing import Optional
import requests  # type: ignore[import-untyped]

DEFAULT_TIMEOUT = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_SLEEP = 1.0


def fetch_html(
    url: str,
    *,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: float = DEFAULT_RETRY_SLEEP,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """
    Fetch a Genius song page and return raw HTML.

    Returns None on failure.
    """
    sess = session or requests
    headers = headers or {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/143.0.0.0 Safari/537.36"
    }

    delay = retry_sleep
    for attempt in range(max_retries):
        try:
            resp = sess.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception:
            time.sleep(delay + random.random())
            delay = min(delay * 2, 30.0)

    return None


def fetch_json(
    url: str,
    *,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: float = DEFAULT_RETRY_SLEEP,
    session: Optional[requests.Session] = None,
) -> Optional[dict]:
    """
    Fetch JSON from a URL and return as a dict.

    Returns None on failure.
    """
    sess = session or requests
    headers = headers or {}

    delay = retry_sleep
    for attempt in range(max_retries):
        try:
            resp = sess.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            time.sleep(delay + random.random())
            delay = min(delay * 2, 30.0)

    return None
