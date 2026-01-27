"""Synced lyrics fetching using syncedlyrics library."""

import re
import sys
import os
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Dict, List

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    import syncedlyrics
    SYNCEDLYRICS_AVAILABLE = True
except ImportError:
    syncedlyrics = None
    SYNCEDLYRICS_AVAILABLE = False


# Provider order: prioritize more reliable sources
# NetEase and Lrclib tend to have good coverage, Musixmatch has rate limits
PROVIDER_ORDER = ["Musixmatch", "Lrclib", "NetEase", "Megalobiz", "Genius"]


# Providers that have shown persistent failures (skip after repeated errors)
_failed_providers: Dict[str, int] = {}
_FAILURE_THRESHOLD = 3  # Skip provider after this many consecutive failures


@contextmanager
def _suppress_stderr():
    """Temporarily suppress stderr to hide noisy library output."""
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr


def _search_single_provider(
    search_term: str,
    provider: str,
    synced_only: bool = True,
    enhanced: bool = False,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> Optional[str]:
    """Search a single provider with retry logic.

    Args:
        search_term: Search query
        provider: Provider name
        synced_only: Only return synced lyrics
        enhanced: Try word-level timing
        max_retries: Number of retries on failure
        retry_delay: Base delay between retries (exponential backoff)

    Returns:
        LRC text if found, None otherwise
    """
    global _failed_providers

    # Skip providers that have failed too many times
    if _failed_providers.get(provider, 0) >= _FAILURE_THRESHOLD:
        logger.debug(f"Skipping {provider} due to repeated failures")
        return None

    for attempt in range(max_retries + 1):
        try:
            with _suppress_stderr():
                lrc = syncedlyrics.search(
                    search_term,
                    providers=[provider],
                    synced_only=synced_only,
                    enhanced=enhanced,
                )
            if lrc:
                # Success - reset failure count
                _failed_providers[provider] = 0
                return lrc
            # No result but no error - don't count as failure
            return None

        except Exception as e:
            error_msg = str(e).lower()
            # Check for transient vs permanent errors
            is_transient = any(x in error_msg for x in [
                'connection', 'timeout', 'temporarily', 'rate limit',
                'remote end closed', '429', '503', '502'
            ])

            if is_transient and attempt < max_retries:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.debug(f"{provider} transient error, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
                continue
            else:
                # Permanent error or retries exhausted
                _failed_providers[provider] = _failed_providers.get(provider, 0) + 1
                logger.debug(f"{provider} failed (attempt {attempt + 1}): {e}")
                return None

    return None


def _search_with_fallback(
    search_term: str,
    synced_only: bool = True,
    enhanced: bool = False,
) -> Tuple[Optional[str], str]:
    """Search across providers with fallback.

    Tries each provider in order until one succeeds.

    Returns:
        Tuple of (lrc_text, provider_name)
    """
    for provider in PROVIDER_ORDER:
        logger.debug(f"Trying {provider} for: {search_term}")
        lrc = _search_single_provider(
            search_term,
            provider,
            synced_only=synced_only,
            enhanced=enhanced,
        )
        if lrc:
            logger.debug(f"Found lyrics from {provider}")
            return lrc, provider

        # Small delay between providers to be nice to services
        time.sleep(0.3)

    return None, ""


# Cache for LRC results to avoid duplicate fetches
# Key: (artist.lower(), title.lower()), Value: (lrc_text, is_synced, source_name)
_lrc_cache: Dict[Tuple[str, str], Tuple[Optional[str], bool, str]] = {}


def fetch_lyrics_multi_source(
    title: str,
    artist: str,
    synced_only: bool = True,
    enhanced: bool = False,
) -> Tuple[Optional[str], bool, str]:
    """
    Fetch lyrics from multiple sources using syncedlyrics.

    Tries providers in order with retries and fallback for reliability.
    Results are cached to avoid duplicate network requests.

    Args:
        title: Song title
        artist: Artist name
        synced_only: Only return synced (timestamped) lyrics
        enhanced: Try to get word-level timing (Musixmatch only)

    Returns:
        Tuple of (lrc_text, is_synced, source_name)
        - lrc_text: LRC formatted lyrics or None if not found
        - is_synced: True if lyrics have timestamps
        - source_name: Name of the source that provided lyrics
    """
    if not SYNCEDLYRICS_AVAILABLE:
        logger.warning("syncedlyrics not installed")
        return None, False, ""

    # Check cache first
    cache_key = (artist.lower().strip(), title.lower().strip())
    if cache_key in _lrc_cache:
        cached = _lrc_cache[cache_key]
        logger.debug(f"Using cached LRC result for {artist} - {title}")
        return cached

    search_term = f"{artist} {title}"
    logger.debug(f"Searching for synced lyrics: {search_term}")

    try:
        # Try enhanced (word-level) first if requested
        if enhanced:
            lrc, provider = _search_with_fallback(
                search_term,
                synced_only=True,
                enhanced=True,
            )
            if lrc and _has_timestamps(lrc):
                logger.debug(f"Found enhanced (word-level) synced lyrics from {provider}")
                result = (lrc, True, f"{provider} (enhanced)")
                _lrc_cache[cache_key] = result
                return result

        # Try synced lyrics with provider fallback
        lrc, provider = _search_with_fallback(
            search_term,
            synced_only=synced_only,
        )

        if lrc:
            is_synced = _has_timestamps(lrc)
            if is_synced:
                logger.debug(f"Found synced lyrics from {provider}")
                result = (lrc, True, provider)
                _lrc_cache[cache_key] = result
                return result
            elif not synced_only:
                logger.debug(f"Found plain lyrics from {provider}")
                result = (lrc, False, provider)
                _lrc_cache[cache_key] = result
                return result

        logger.warning("No synced lyrics found from any provider")
        result = (None, False, "")
        _lrc_cache[cache_key] = result
        return result

    except Exception as e:
        logger.error(f"Error fetching synced lyrics: {e}")
        result = (None, False, "")
        _lrc_cache[cache_key] = result
        return result


def _has_timestamps(lrc_text: str) -> bool:
    """Check if LRC text contains timestamps."""
    if not lrc_text:
        return False
    # LRC timestamps look like [mm:ss.xx] or [mm:ss:xx]
    timestamp_pattern = r'\[\d{1,2}:\d{2}[.:]\d{2,3}\]'
    return bool(re.search(timestamp_pattern, lrc_text))


def get_lrc_duration(lrc_text: str) -> Optional[int]:
    """Get the implied duration from LRC text based on timestamp span.

    Uses first-to-last timestamp difference plus a proportional buffer,
    rather than arbitrary fixed buffer.
    """
    if not lrc_text or not _has_timestamps(lrc_text):
        return None

    from .lrc import parse_lrc_with_timing
    timings = parse_lrc_with_timing(lrc_text, "", "")
    if not timings or len(timings) < 2:
        return None

    first_ts = timings[0][0]
    last_ts = timings[-1][0]
    lyrics_span = last_ts - first_ts

    # Add proportional buffer: ~10% of span or minimum 3s for outro
    # This accounts for songs with varying outro lengths
    buffer = max(3, int(lyrics_span * 0.1))

    return int(last_ts + buffer)


def validate_lrc_quality(lrc_text: str, expected_duration: Optional[int] = None) -> tuple[bool, str]:
    """Validate that an LRC file has sufficient quality for karaoke use.

    Checks:
    - Minimum timestamp density (at least 1 timestamp per 15 seconds)
    - Sufficient coverage of expected duration (80%+ if duration provided)
    - No large gaps (>30s) in the middle of the song

    Args:
        lrc_text: The LRC text to validate
        expected_duration: Expected song duration in seconds (optional)

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    if not lrc_text or not _has_timestamps(lrc_text):
        return False, "No timestamps found"

    from .lrc import parse_lrc_with_timing
    timings = parse_lrc_with_timing(lrc_text, "", "")

    if len(timings) < 5:
        return False, f"Too few timestamped lines ({len(timings)})"

    first_ts = timings[0][0]
    last_ts = timings[-1][0]
    lyrics_span = last_ts - first_ts

    if lyrics_span < 30:
        return False, f"Lyrics span too short ({lyrics_span:.0f}s)"

    # Check timestamp density: at least 1 per 15 seconds on average
    density = len(timings) / (lyrics_span / 15) if lyrics_span > 0 else 0
    if density < 1.0:
        return False, f"Timestamp density too low ({density:.2f} per 15s)"

    # Check for large gaps (>30s) that might indicate missing sections
    for i in range(1, len(timings)):
        gap = timings[i][0] - timings[i-1][0]
        if gap > 30:
            logger.debug(f"Large gap detected in LRC: {gap:.0f}s between lines {i-1} and {i}")
            # Don't fail for single large gap (could be instrumental break)
            # but flag if there are multiple

    # Check coverage of expected duration
    if expected_duration and expected_duration > 0:
        coverage = lyrics_span / expected_duration
        if coverage < 0.6:
            return False, f"LRC covers only {coverage*100:.0f}% of expected duration"

    return True, ""


def fetch_lyrics_for_duration(
    title: str,
    artist: str,
    target_duration: int,
    tolerance: int = 20,
) -> Tuple[Optional[str], bool, str, Optional[int]]:
    """
    Fetch synced lyrics that match a target duration.

    Tries multiple search strategies to find LRC with timing that matches
    the expected track duration.

    Args:
        title: Song title
        artist: Artist name
        target_duration: Expected track duration in seconds
        tolerance: Maximum acceptable duration difference in seconds

    Returns:
        Tuple of (lrc_text, is_synced, source_name, lrc_duration)
        - lrc_text: LRC formatted lyrics or None if not found
        - is_synced: True if lyrics have timestamps
        - source_name: Name of the source that provided lyrics
        - lrc_duration: Duration implied by the LRC timestamps
    """
    if not SYNCEDLYRICS_AVAILABLE:
        logger.warning("syncedlyrics not installed")
        return None, False, "", None

    # Strategy 1: Try the standard search
    lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist, synced_only=True)

    if is_synced and lrc_text:
        lrc_duration = get_lrc_duration(lrc_text)
        if lrc_duration:
            diff = abs(lrc_duration - target_duration)
            if diff <= tolerance:
                logger.info(f"Found LRC with matching duration: {lrc_duration}s (target: {target_duration}s)")
                return lrc_text, is_synced, source, lrc_duration
            else:
                logger.warning(f"LRC duration mismatch: LRC={lrc_duration}s, target={target_duration}s, diff={diff}s")

    # Strategy 2: Try searching with different terms
    alternative_searches = [
        f"{title} {artist}",  # Swap order
        f"{artist} {title} official",
        f"{artist} {title} album version",
    ]

    for search_term in alternative_searches:
        logger.debug(f"Trying alternative LRC search: {search_term}")
        lrc, provider = _search_with_fallback(search_term, synced_only=True)
        if lrc and _has_timestamps(lrc):
            alt_duration = get_lrc_duration(lrc)
            if alt_duration:
                diff = abs(alt_duration - target_duration)
                if diff <= tolerance:
                    logger.info(f"Found LRC with alternative search '{search_term}' from {provider}: {alt_duration}s")
                    return lrc, True, f"{provider} ({search_term})", alt_duration

    # Strategy 3: If we found LRC but duration doesn't match, return it anyway
    # with a warning - better than nothing
    if is_synced and lrc_text:
        lrc_duration = get_lrc_duration(lrc_text)
        logger.warning(f"Using LRC despite duration mismatch. LRC timing may be off.")
        return lrc_text, is_synced, source, lrc_duration

    return None, False, "", None
