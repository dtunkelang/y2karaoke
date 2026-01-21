"""Synced lyrics fetching using syncedlyrics library."""

import re
from typing import Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    import syncedlyrics
    SYNCEDLYRICS_AVAILABLE = True
except ImportError:
    syncedlyrics = None
    SYNCEDLYRICS_AVAILABLE = False


def fetch_lyrics_multi_source(
    title: str,
    artist: str,
    synced_only: bool = True,
    enhanced: bool = False,
) -> Tuple[Optional[str], bool, str]:
    """
    Fetch lyrics from multiple sources using syncedlyrics.

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

    search_term = f"{artist} {title}"
    logger.info(f"Searching for synced lyrics: {search_term}")

    try:
        # Try enhanced (word-level) first if requested
        if enhanced:
            lrc = syncedlyrics.search(
                search_term,
                synced_only=True,
                enhanced=True,
            )
            if lrc and _has_timestamps(lrc):
                logger.info("Found enhanced (word-level) synced lyrics")
                return lrc, True, "syncedlyrics (enhanced)"

        # Try synced lyrics
        lrc = syncedlyrics.search(
            search_term,
            synced_only=synced_only,
        )

        if lrc:
            is_synced = _has_timestamps(lrc)
            if is_synced:
                logger.info("Found synced lyrics with line timing")
                return lrc, True, "syncedlyrics"
            elif not synced_only:
                logger.info("Found plain lyrics (no timing)")
                return lrc, False, "syncedlyrics"

        logger.warning("No synced lyrics found")
        return None, False, ""

    except Exception as e:
        logger.error(f"Error fetching synced lyrics: {e}")
        return None, False, ""


def _has_timestamps(lrc_text: str) -> bool:
    """Check if LRC text contains timestamps."""
    if not lrc_text:
        return False
    # LRC timestamps look like [mm:ss.xx] or [mm:ss:xx]
    timestamp_pattern = r'\[\d{1,2}:\d{2}[.:]\d{2,3}\]'
    return bool(re.search(timestamp_pattern, lrc_text))
