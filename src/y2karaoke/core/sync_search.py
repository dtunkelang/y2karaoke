"""Provider search/retry helpers for synced lyrics fetching."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Callable, Dict, Optional, Tuple


@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr to hide noisy library output."""
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr


def search_single_provider(
    search_fn: Callable[..., Optional[str]],
    search_term: str,
    provider: str,
    failed_providers: Dict[str, int],
    failure_threshold: int,
    logger,
    synced_only: bool = True,
    enhanced: bool = False,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> Optional[str]:
    """Search a single provider with retry logic."""
    if failed_providers.get(provider, 0) >= failure_threshold:
        logger.debug(f"Skipping {provider} due to repeated failures")
        return None

    for attempt in range(max_retries + 1):
        try:
            with suppress_stderr():
                lrc = search_fn(
                    search_term,
                    providers=[provider],
                    synced_only=synced_only,
                    enhanced=enhanced,
                )
            if lrc:
                failed_providers[provider] = 0
                return lrc
            return None
        except Exception as e:
            error_msg = str(e).lower()
            is_transient = any(
                x in error_msg
                for x in [
                    "connection",
                    "timeout",
                    "temporarily",
                    "rate limit",
                    "remote end closed",
                    "429",
                    "503",
                    "502",
                ]
            )
            if is_transient and attempt < max_retries:
                delay = retry_delay * (2**attempt)
                logger.debug(
                    f"{provider} transient error, retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
                continue
            failed_providers[provider] = failed_providers.get(provider, 0) + 1
            logger.debug(f"{provider} failed (attempt {attempt + 1}): {e}")
            return None
    return None


def search_with_fallback(
    search_term: str,
    provider_order,
    search_single_provider_fn: Callable[..., Optional[str]],
    search_cache: Dict[str, Tuple[Optional[str], str]],
    disk_cache: Dict,
    disk_cache_enabled: bool,
    load_disk_cache_fn: Callable[[], None],
    save_disk_cache_fn: Callable[[], None],
    logger,
    synced_only: bool = True,
    enhanced: bool = False,
) -> Tuple[Optional[str], str]:
    """Search across providers with fallback and cache support."""
    if disk_cache_enabled:
        load_disk_cache_fn()
    cache_key = f"{search_term.lower()}:{synced_only}:{enhanced}"
    if cache_key in search_cache:
        logger.debug(f"Using cached search result for: {search_term}")
        return search_cache[cache_key]
    if disk_cache_enabled:
        disk_search = disk_cache.get("search_cache", {})
        if cache_key in disk_search:
            cached = tuple(disk_search[cache_key])
            search_cache[cache_key] = cached
            logger.debug(f"Using cached search result for: {search_term}")
            return cached

    for provider in provider_order:
        logger.debug(f"Trying {provider} for: {search_term}")
        lrc = search_single_provider_fn(
            search_term,
            provider,
            synced_only=synced_only,
            enhanced=enhanced,
        )
        if lrc:
            logger.debug(f"Found lyrics from {provider}")
            found_result: Tuple[Optional[str], str] = (lrc, provider)
            search_cache[cache_key] = found_result
            if disk_cache_enabled:
                disk_cache.setdefault("search_cache", {})[cache_key] = [
                    found_result[0],
                    found_result[1],
                ]
                save_disk_cache_fn()
            return found_result
        time.sleep(0.3)

    empty_result: Tuple[Optional[str], str] = (None, "")
    search_cache[cache_key] = empty_result
    if disk_cache_enabled:
        disk_cache.setdefault("search_cache", {})[cache_key] = [
            empty_result[0],
            empty_result[1],
        ]
        save_disk_cache_fn()
    return empty_result
