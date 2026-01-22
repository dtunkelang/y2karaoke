"""Synced lyrics fetching using syncedlyrics library."""

import re
from typing import Optional, Tuple, Dict

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    import syncedlyrics
    SYNCEDLYRICS_AVAILABLE = True
except ImportError:
    syncedlyrics = None
    SYNCEDLYRICS_AVAILABLE = False

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

    Results are cached to avoid duplicate network requests when checking
    LRC availability during search and then fetching lyrics later.

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
            lrc = syncedlyrics.search(
                search_term,
                synced_only=True,
                enhanced=True,
            )
            if lrc and _has_timestamps(lrc):
                logger.debug("Found enhanced (word-level) synced lyrics")
                result = (lrc, True, "syncedlyrics (enhanced)")
                _lrc_cache[cache_key] = result
                return result

        # Try synced lyrics
        lrc = syncedlyrics.search(
            search_term,
            synced_only=synced_only,
        )

        if lrc:
            is_synced = _has_timestamps(lrc)
            if is_synced:
                logger.debug("Found synced lyrics with line timing")
                result = (lrc, True, "syncedlyrics")
                _lrc_cache[cache_key] = result
                return result
            elif not synced_only:
                logger.debug("Found plain lyrics (no timing)")
                result = (lrc, False, "syncedlyrics")
                _lrc_cache[cache_key] = result
                return result

        logger.warning("No synced lyrics found")
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
