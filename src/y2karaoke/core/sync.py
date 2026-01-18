"""Synced lyrics fetching using syncedlyrics library."""

import re
from typing import Optional, Tuple, List

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


def fetch_enhanced_lyrics(
    title: str,
    artist: str,
) -> Tuple[Optional[str], bool]:
    """
    Fetch word-level synced lyrics if available.

    This specifically tries to get enhanced (word-by-word) timing
    from Musixmatch via syncedlyrics.

    Args:
        title: Song title
        artist: Artist name

    Returns:
        Tuple of (lrc_text, has_word_timing)
    """
    lrc, is_synced, source = fetch_lyrics_multi_source(
        title, artist, synced_only=True, enhanced=True
    )

    if not lrc or not is_synced:
        return None, False

    # Check if it has word-level timing (enhanced LRC format)
    # Enhanced format has timestamps within lines like: [00:05.00] <00:05.10> word <00:05.50> word
    has_word_timing = bool(re.search(r'<\d{2}:\d{2}\.\d{2}>', lrc))

    return lrc, has_word_timing


def get_lrc_from_youtube(url: str) -> Optional[str]:
    """
    Placeholder for getting LRC from YouTube captions.

    This could be implemented to extract timing from YouTube's
    auto-generated or manual captions.

    Args:
        url: YouTube video URL

    Returns:
        LRC formatted text or None
    """
    # TODO: Implement YouTube caption extraction
    # Could use yt-dlp to get subtitles and convert to LRC
    logger.debug("YouTube LRC extraction not yet implemented")
    return None


def parse_enhanced_lrc(lrc_text: str) -> List[Tuple[float, str, List[Tuple[float, str]]]]:
    """
    Parse enhanced LRC format with word-level timing.

    Enhanced LRC format:
    [00:05.00] <00:05.10> Hello <00:05.50> world

    Args:
        lrc_text: Enhanced LRC formatted text

    Returns:
        List of (line_timestamp, line_text, [(word_timestamp, word), ...])
    """
    results = []
    line_pattern = re.compile(r'\[(\d{2}):(\d{2})\.(\d{2,3})\](.+)')
    word_pattern = re.compile(r'<(\d{2}):(\d{2})\.(\d{2})>\s*(\S+)')

    for line in lrc_text.strip().split('\n'):
        line_match = line_pattern.match(line)
        if not line_match:
            continue

        # Parse line timestamp
        mins, secs, ms = line_match.groups()[:3]
        line_content = line_match.group(4)
        line_ts = int(mins) * 60 + int(secs) + int(ms) / (1000 if len(ms) == 3 else 100)

        # Parse word timestamps
        words = []
        for word_match in word_pattern.finditer(line_content):
            w_mins, w_secs, w_ms, word = word_match.groups()
            word_ts = int(w_mins) * 60 + int(w_secs) + int(w_ms) / 100
            words.append((word_ts, word))

        # If no word-level timing, extract plain text
        if not words:
            plain_text = re.sub(r'<\d{2}:\d{2}\.\d{2}>\s*', '', line_content).strip()
            if plain_text:
                results.append((line_ts, plain_text, []))
        else:
            full_text = ' '.join(w[1] for w in words)
            results.append((line_ts, full_text, words))

    return results
