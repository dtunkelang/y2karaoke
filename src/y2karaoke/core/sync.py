"""Synced lyrics fetching using syncedlyrics and lyriq libraries."""

import re
import sys
import os
import time
import json
from contextlib import contextmanager
from typing import Optional, Tuple, Dict, List, Any

from ..utils.logging import get_logger
from ..config import get_cache_dir

logger = get_logger(__name__)

try:
    import syncedlyrics

    SYNCEDLYRICS_AVAILABLE = True
except ImportError:
    syncedlyrics = None
    SYNCEDLYRICS_AVAILABLE = False

try:
    from lyriq import get_lyrics as lyriq_get_lyrics

    LYRIQ_AVAILABLE = True
except ImportError:
    lyriq_get_lyrics = None
    LYRIQ_AVAILABLE = False


# Provider order for syncedlyrics: prioritize more reliable sources
# Note: lyriq (LRCLib) is tried first before syncedlyrics providers
# Musixmatch: Best quality but has rate limits
# NetEase: Good coverage, especially for Asian music
# Megalobiz: Less reliable but can have unique content
# Lrclib: Moved to end since lyriq already uses LRCLib API with potentially better search
# Genius: Plain text only (not useful for synced lyrics, kept as last resort)
PROVIDER_ORDER = ["Musixmatch", "NetEase", "Megalobiz", "Lrclib", "Genius"]


# Providers that have shown persistent failures (skip after repeated errors)
_failed_providers: Dict[str, int] = {}
_FAILURE_THRESHOLD = 3  # Skip provider after this many consecutive failures


@contextmanager
def _suppress_stderr():
    """Temporarily suppress stderr to hide noisy library output."""
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
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
                delay = retry_delay * (2**attempt)  # Exponential backoff
                logger.debug(
                    f"{provider} transient error, retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
                continue
            else:
                # Permanent error or retries exhausted
                _failed_providers[provider] = _failed_providers.get(provider, 0) + 1
                logger.debug(f"{provider} failed (attempt {attempt + 1}): {e}")
                return None

    return None


# Cache for search term results
_search_cache: Dict[str, Tuple[Optional[str], str]] = {}
_disk_cache: Dict[str, Any] = {}
_disk_cache_loaded = False


def _search_with_fallback(
    search_term: str,
    synced_only: bool = True,
    enhanced: bool = False,
) -> Tuple[Optional[str], str]:
    """Search across providers with fallback.

    Tries each provider in order until one succeeds.
    Results are cached by search term.

    Returns:
        Tuple of (lrc_text, provider_name)
    """
    disk_cache_enabled = _disk_cache_enabled()
    if disk_cache_enabled:
        _load_disk_cache()
    cache_key = f"{search_term.lower()}:{synced_only}:{enhanced}"
    if cache_key in _search_cache:
        logger.debug(f"Using cached search result for: {search_term}")
        return _search_cache[cache_key]
    if disk_cache_enabled:
        disk_search = _disk_cache.get("search_cache", {})
        if cache_key in disk_search:
            cached = tuple(disk_search[cache_key])
            _search_cache[cache_key] = cached
            logger.debug(f"Using cached search result for: {search_term}")
            return cached

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
            found_result: Tuple[Optional[str], str] = (lrc, provider)
            _search_cache[cache_key] = found_result
            if disk_cache_enabled:
                _disk_cache.setdefault("search_cache", {})[cache_key] = [
                    found_result[0],
                    found_result[1],
                ]
                _save_disk_cache()
            return found_result

        # Small delay between providers to be nice to services
        time.sleep(0.3)

    empty_result: Tuple[Optional[str], str] = (None, "")
    _search_cache[cache_key] = empty_result
    if disk_cache_enabled:
        _disk_cache.setdefault("search_cache", {})[cache_key] = [
            empty_result[0],
            empty_result[1],
        ]
        _save_disk_cache()
    return empty_result


# Cache for LRC results to avoid duplicate fetches
# Key: (artist.lower(), title.lower()), Value: (lrc_text, is_synced, source_name, lrc_duration)
_lrc_cache: Dict[Tuple[str, str], Tuple[Optional[str], bool, str, Optional[int]]] = {}

# Cache for lyriq results
_lyriq_cache: Dict[Tuple[str, str], Optional[str]] = {}


def _get_disk_cache_path() -> "Path":
    return get_cache_dir() / "lyrics_cache.json"


def _disk_cache_enabled() -> bool:
    # Disable caches for tests to keep results deterministic and isolated.
    return "PYTEST_CURRENT_TEST" not in os.environ


def _load_disk_cache() -> None:
    global _disk_cache_loaded, _disk_cache
    if _disk_cache_loaded:
        return
    _disk_cache_loaded = True
    if not _disk_cache_enabled():
        _disk_cache = {"search_cache": {}, "lrc_cache": {}, "lyriq_cache": {}}
        return
    cache_path = _get_disk_cache_path()
    if not cache_path.exists():
        _disk_cache = {"search_cache": {}, "lrc_cache": {}, "lyriq_cache": {}}
        return
    try:
        _disk_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        _disk_cache = {"search_cache": {}, "lrc_cache": {}, "lyriq_cache": {}}


def _save_disk_cache() -> None:
    if not _disk_cache_enabled():
        return
    cache_path = _get_disk_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_path.write_text(
            json.dumps(_disk_cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        logger.debug("Failed to write lyrics cache to disk")


def _set_lrc_cache(
    cache_key: Tuple[str, str],
    value: Tuple[Optional[str], bool, str, Optional[int]],
) -> None:
    _lrc_cache[cache_key] = value
    if _disk_cache_enabled():
        disk_key = f"{cache_key[0]}|{cache_key[1]}"
        _disk_cache.setdefault("lrc_cache", {})[disk_key] = list(value)
        _save_disk_cache()


def _fetch_from_lyriq(
    title: str,
    artist: str,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> Optional[str]:
    """Fetch lyrics from lyriq (LRCLib API).

    lyriq provides a different search/matching algorithm than syncedlyrics'
    Lrclib provider, so it may find different results.

    Args:
        title: Song title
        artist: Artist name
        max_retries: Number of retries on failure
        retry_delay: Base delay between retries (exponential backoff)

    Returns:
        LRC text if found, None otherwise
    """
    disk_cache_enabled = _disk_cache_enabled()
    if disk_cache_enabled:
        _load_disk_cache()
    if not LYRIQ_AVAILABLE:
        return None

    # Check cache
    cache_key = (artist.lower().strip(), title.lower().strip())
    disk_key = f"{cache_key[0]}|{cache_key[1]}"
    if cache_key in _lyriq_cache:
        return _lyriq_cache[cache_key]
    if disk_cache_enabled:
        disk_lyriq = _disk_cache.get("lyriq_cache", {})
        if disk_key in disk_lyriq:
            cached = disk_lyriq[disk_key]
            _lyriq_cache[cache_key] = cached
            return cached

    for attempt in range(max_retries + 1):
        try:
            with _suppress_stderr():
                # lyriq.get_lyrics returns a Lyrics object or None
                lyrics_obj = lyriq_get_lyrics(title, artist)

            if lyrics_obj is None:
                _lyriq_cache[cache_key] = None
                if disk_cache_enabled:
                    _disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                    _save_disk_cache()
                return None

            # lyriq returns a Lyrics object with synced_lyrics and plain_lyrics attributes
            synced = getattr(lyrics_obj, "synced_lyrics", None)
            if synced and _has_timestamps(synced):
                logger.debug("Found synced lyrics from lyriq (LRCLib)")
                _lyriq_cache[cache_key] = synced
                if disk_cache_enabled:
                    _disk_cache.setdefault("lyriq_cache", {})[disk_key] = synced
                    _save_disk_cache()
                return synced

            # Fall back to plain lyrics if no synced available
            plain = getattr(lyrics_obj, "plain_lyrics", None)
            if plain:
                logger.debug("Found plain lyrics from lyriq (no timestamps)")
                # Don't cache plain for synced requests
                _lyriq_cache[cache_key] = None
                if disk_cache_enabled:
                    _disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                    _save_disk_cache()
                return None

            _lyriq_cache[cache_key] = None
            if disk_cache_enabled:
                _disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                _save_disk_cache()
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
                logger.debug(f"lyriq transient error, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
                continue
            else:
                logger.debug(f"lyriq failed (attempt {attempt + 1}): {e}")
                _lyriq_cache[cache_key] = None
                if disk_cache_enabled:
                    _disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                    _save_disk_cache()
                return None

    _lyriq_cache[cache_key] = None
    if disk_cache_enabled:
        _disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
        _save_disk_cache()
    return None


def fetch_lyrics_multi_source(
    title: str,
    artist: str,
    synced_only: bool = True,
    enhanced: bool = False,
    target_duration: Optional[int] = None,
    duration_tolerance: int = 20,
) -> Tuple[Optional[str], bool, str]:
    """
    Fetch lyrics from multiple sources using lyriq and syncedlyrics.

    Tries lyriq first (LRCLib with potentially better search matching),
    then falls back to syncedlyrics providers in order with retries.
    Results are cached to avoid duplicate network requests.

    Args:
        title: Song title
        artist: Artist name
        synced_only: Only return synced (timestamped) lyrics
        enhanced: Try to get word-level timing (Musixmatch only)
        target_duration: Expected track duration in seconds (for cache validation)
        duration_tolerance: Maximum acceptable duration difference (seconds)

    Returns:
        Tuple of (lrc_text, is_synced, source_name)
        - lrc_text: LRC formatted lyrics or None if not found
        - is_synced: True if lyrics have timestamps
        - source_name: Name of the source that provided lyrics
    """
    disk_cache_enabled = _disk_cache_enabled()
    if disk_cache_enabled:
        _load_disk_cache()
    cache_key = (artist.lower().strip(), title.lower().strip())
    if cache_key in _lrc_cache:
        cached = _lrc_cache[cache_key]
        cached_duration = cached[3] if len(cached) > 3 else None
        # If target_duration specified, validate cached duration matches
        if target_duration and cached_duration:
            if abs(cached_duration - target_duration) > duration_tolerance:
                logger.debug(
                    f"Cached LRC duration ({cached_duration}s) doesn't match target ({target_duration}s), re-fetching"
                )
            else:
                logger.debug(f"Using cached LRC result for {artist} - {title}")
                return (cached[0], cached[1], cached[2])
        else:
            logger.debug(f"Using cached LRC result for {artist} - {title}")
            return (cached[0], cached[1], cached[2])

    if disk_cache_enabled:
        disk_key = f"{cache_key[0]}|{cache_key[1]}"
        disk_lrc = _disk_cache.get("lrc_cache", {})
        if disk_key in disk_lrc:
            cached = disk_lrc[disk_key]
            cached_duration = cached[3] if len(cached) > 3 else None
            if target_duration and cached_duration:
                if abs(cached_duration - target_duration) > duration_tolerance:
                    logger.debug(
                        f"Cached LRC duration ({cached_duration}s) doesn't match target ({target_duration}s), re-fetching"
                    )
                else:
                    _lrc_cache[cache_key] = tuple(cached)
                    logger.debug(f"Using cached LRC result for {artist} - {title}")
                    return (cached[0], cached[1], cached[2])
            else:
                _lrc_cache[cache_key] = tuple(cached)
                logger.debug(f"Using cached LRC result for {artist} - {title}")
                return (cached[0], cached[1], cached[2])

    search_term = f"{artist} {title}"
    logger.debug(f"Searching for synced lyrics: {search_term}")

    try:
        # Try lyriq first (LRCLib with potentially better search matching)
        if LYRIQ_AVAILABLE:
            logger.debug(f"Trying lyriq for: {title} - {artist}")
            lrc = _fetch_from_lyriq(title, artist)
            if lrc and _has_timestamps(lrc):
                logger.debug("Found synced lyrics from lyriq (LRCLib)")
                lrc_duration = get_lrc_duration(lrc)
                result = (lrc, True, "lyriq (LRCLib)", lrc_duration)
                _set_lrc_cache(cache_key, result)
                return (lrc, True, "lyriq (LRCLib)")

        if not SYNCEDLYRICS_AVAILABLE:
            logger.warning("syncedlyrics not installed")
            no_sync_result: Tuple[Optional[str], bool, str, Optional[int]] = (
                None,
                False,
                "",
                None,
            )
            _set_lrc_cache(cache_key, no_sync_result)
            return (None, False, "")

        # Try enhanced (word-level) first if requested
        if enhanced:
            lrc, provider = _search_with_fallback(
                search_term,
                synced_only=True,
                enhanced=True,
            )
            if lrc and _has_timestamps(lrc):
                logger.debug(
                    f"Found enhanced (word-level) synced lyrics from {provider}"
                )
                lrc_duration = get_lrc_duration(lrc)
                result = (lrc, True, f"{provider} (enhanced)", lrc_duration)
                _set_lrc_cache(cache_key, result)
                return (lrc, True, f"{provider} (enhanced)")

        # Try synced lyrics with provider fallback
        lrc, provider = _search_with_fallback(
            search_term,
            synced_only=synced_only,
        )

        if lrc:
            is_synced = _has_timestamps(lrc)
            if is_synced:
                logger.debug(f"Found synced lyrics from {provider}")
                lrc_duration = get_lrc_duration(lrc)
                result = (lrc, True, provider, lrc_duration)
                _set_lrc_cache(cache_key, result)
                return (lrc, True, provider)
            elif not synced_only:
                logger.debug(f"Found plain lyrics from {provider}")
                result = (lrc, False, provider, None)
                _set_lrc_cache(cache_key, result)
                return (lrc, False, provider)

        logger.warning("No synced lyrics found from any provider")
        not_found: Tuple[Optional[str], bool, str, Optional[int]] = (
            None,
            False,
            "",
            None,
        )
        _set_lrc_cache(cache_key, not_found)
        return (None, False, "")

    except Exception as e:
        logger.error(f"Error fetching synced lyrics: {e}")
        error_result: Tuple[Optional[str], bool, str, Optional[int]] = (
            None,
            False,
            "",
            None,
        )
        _set_lrc_cache(cache_key, error_result)
        return (None, False, "")


def _has_timestamps(lrc_text: str) -> bool:
    """Check if LRC text contains timestamps."""
    if not lrc_text:
        return False
    # LRC timestamps look like [mm:ss.xx] or [mm:ss:xx]
    timestamp_pattern = r"\[\d{1,2}:\d{2}[.:]\d{2,3}\]"
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


def validate_lrc_quality(
    lrc_text: str, expected_duration: Optional[int] = None
) -> tuple[bool, str]:
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
        gap = timings[i][0] - timings[i - 1][0]
        if gap > 30:
            logger.debug(
                f"Large gap detected in LRC: {gap:.0f}s between lines {i-1} and {i}"
            )
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
    if not SYNCEDLYRICS_AVAILABLE and not LYRIQ_AVAILABLE:
        logger.warning("Neither syncedlyrics nor lyriq installed")
        return None, False, "", None

    # Strategy 1: Try the standard search (tries lyriq first, then syncedlyrics providers)
    # Pass target_duration to enable cache validation
    lrc_text, is_synced, source = fetch_lyrics_multi_source(
        title,
        artist,
        synced_only=True,
        target_duration=target_duration,
        duration_tolerance=tolerance,
    )

    if is_synced and lrc_text:
        lrc_duration = get_lrc_duration(lrc_text)
        if lrc_duration:
            diff = abs(lrc_duration - target_duration)
            if diff <= tolerance:
                logger.info(
                    f"Found LRC with matching duration: {lrc_duration}s (target: {target_duration}s)"
                )
                return lrc_text, is_synced, source, lrc_duration
            else:
                logger.warning(
                    f"LRC duration mismatch: LRC={lrc_duration}s, target={target_duration}s, diff={diff}s"
                )

    # Strategy 2: Try searching with different terms (syncedlyrics only)
    if SYNCEDLYRICS_AVAILABLE:
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
                        logger.info(
                            f"Found LRC with alternative search '{search_term}' from {provider}: {alt_duration}s"
                        )
                        return lrc, True, f"{provider} ({search_term})", alt_duration

    # Strategy 3: If we found LRC but duration doesn't match, return it anyway
    # with a warning - better than nothing
    if is_synced and lrc_text:
        lrc_duration = get_lrc_duration(lrc_text)
        logger.warning("Using LRC despite duration mismatch. LRC timing may be off.")
        return lrc_text, is_synced, source, lrc_duration

    return None, False, "", None


def _count_large_gaps(timings: List[Tuple[float, str]], threshold: float = 30.0) -> int:
    """Count gaps between consecutive timestamps exceeding threshold."""
    return sum(
        1
        for i in range(1, len(timings))
        if timings[i][0] - timings[i - 1][0] > threshold
    )


def _calculate_quality_score(report: Dict[str, Any], num_timings: int) -> float:
    """Calculate quality score based on metrics in report."""
    score = 100.0
    issues: List[str] = report["issues"]

    # Coverage penalties
    if report["coverage"] < 0.6:
        score -= 30
        issues.append(f"Low coverage ({report['coverage']*100:.0f}%)")
    elif report["coverage"] < 0.8:
        score -= 15

    # Density penalties
    if report["timestamp_density"] < 1.5:
        score -= 20
        issues.append(f"Low timestamp density ({report['timestamp_density']:.1f}/10s)")
    elif report["timestamp_density"] < 2.0:
        score -= 10

    # Duration mismatch penalty
    if not report["duration_match"]:
        score -= 20

    # Few lines penalty
    if num_timings < 10:
        score -= 15
        issues.append(f"Only {num_timings} lines")

    return score


def get_lyrics_quality_report(
    lrc_text: str,
    source: str,
    target_duration: Optional[int] = None,
    sources_tried: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a quality report for fetched LRC lyrics.

    Args:
        lrc_text: The LRC text to analyze
        source: The source that provided the lyrics
        target_duration: Expected track duration for comparison
        sources_tried: List of sources that were attempted

    Returns:
        Dict with quality metrics:
        - quality_score: 0-100 overall quality
        - source: Provider that succeeded
        - sources_tried: Providers attempted
        - coverage: Fraction of duration covered (0-1)
        - timestamp_density: Lines per 10 seconds
        - duration: Implied duration from LRC
        - duration_match: Whether duration matches target
        - issues: List of quality concerns
    """
    from .lrc import parse_lrc_with_timing

    issues: List[str] = []
    report: Dict[str, Any] = {
        "quality_score": 0.0,
        "source": source,
        "sources_tried": sources_tried or [],
        "coverage": 0.0,
        "timestamp_density": 0.0,
        "duration": None,
        "duration_match": True,
        "issues": issues,
    }

    if not lrc_text or not _has_timestamps(lrc_text):
        issues.append("No synced lyrics found")
        return report

    # Parse timings
    timings = parse_lrc_with_timing(lrc_text, "", "")
    if not timings or len(timings) < 2:
        report["quality_score"] = 20.0
        issues.append("Too few timestamped lines")
        return report

    # Calculate metrics
    lyrics_span = timings[-1][0] - timings[0][0]
    lrc_duration = get_lrc_duration(lrc_text)
    report["duration"] = lrc_duration

    # Coverage (relative to target or LRC duration)
    reference_duration = target_duration or lrc_duration or lyrics_span
    if reference_duration > 0:
        report["coverage"] = min(1.0, lyrics_span / reference_duration)

    # Timestamp density (lines per 10 seconds)
    if lyrics_span > 0:
        report["timestamp_density"] = len(timings) / (lyrics_span / 10.0)

    # Duration match check
    if target_duration and lrc_duration:
        diff = abs(lrc_duration - target_duration)
        report["duration_match"] = diff <= 20
        if diff > 20:
            issues.append(
                f"Duration mismatch: LRC={lrc_duration}s, target={target_duration}s"
            )

    # Calculate base score
    score = _calculate_quality_score(report, len(timings))

    # Gap penalty
    large_gaps = _count_large_gaps(timings)
    if large_gaps > 2:
        score -= 10
        issues.append(f"{large_gaps} large gaps (>30s)")

    report["quality_score"] = max(0.0, min(100.0, score))
    return report


def fetch_from_all_sources(
    title: str,
    artist: str,
) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    """
    Fetch lyrics from all available sources for comparison.

    This function fetches lyrics from all providers without caching,
    useful for comparing timing quality across sources.

    Args:
        title: Song title
        artist: Artist name

    Returns:
        Dict mapping source name to (lrc_text, lrc_duration) tuple
    """
    results: Dict[str, Tuple[Optional[str], Optional[int]]] = {}

    # Try lyriq
    if LYRIQ_AVAILABLE:
        try:
            with _suppress_stderr():
                lyrics_obj = lyriq_get_lyrics(title, artist)
            if lyrics_obj:
                synced = getattr(lyrics_obj, "synced_lyrics", None)
                if synced and _has_timestamps(synced):
                    duration = get_lrc_duration(synced)
                    results["lyriq (LRCLib)"] = (synced, duration)
        except Exception as e:
            logger.debug(f"lyriq fetch failed for comparison: {e}")

    # Try each syncedlyrics provider individually
    if SYNCEDLYRICS_AVAILABLE:
        search_term = f"{artist} {title}"
        for provider in PROVIDER_ORDER:
            if provider == "Genius":  # Skip Genius as it only provides plain text
                continue
            try:
                with _suppress_stderr():
                    lrc = syncedlyrics.search(
                        search_term,
                        providers=[provider],
                        synced_only=True,
                    )
                if lrc and _has_timestamps(lrc):
                    duration = get_lrc_duration(lrc)
                    results[provider] = (lrc, duration)
            except Exception as e:
                logger.debug(f"{provider} fetch failed for comparison: {e}")

    return results
