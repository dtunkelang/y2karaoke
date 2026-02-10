"""Track identification pipeline for y2karaoke.

This module handles two distinct paths for identifying track information:
- Path A (search string): Query -> MusicBrainz/syncedlyrics -> canonical track -> YouTube
- Path B (YouTube URL): URL -> YouTube duration -> best LRC match by duration
"""

from typing import Optional, Dict
import musicbrainzngs  # noqa: F401 - re-exported for compatibility tests

from ....utils.logging import get_logger
from ....exceptions import Y2KaraokeError
from ...text_utils import normalize_title
from ...models import TrackInfo
from .helpers import TrackIdentifierHelpers
from .youtube import YouTubeSearcher
from .musicbrainz import MusicBrainzClient
from .parser import QueryParser

logger = get_logger(__name__)

__all__ = [
    "TrackIdentifier",
    "TrackInfo",
    "YouTubeSearcher",
    "MusicBrainzClient",
    "QueryParser",
    "normalize_title",
    "Y2KaraokeError",
]


class TrackIdentifier(
    YouTubeSearcher, MusicBrainzClient, QueryParser, TrackIdentifierHelpers
):
    """Identifies track information from search queries or YouTube URLs."""

    def __init__(self):
        self._lrc_cache: Dict[tuple, tuple] = {}
        self._http_get = None
        self._mb_search_recordings = None
        self._sleep = None

    def _try_direct_lrc_search(self, query: str) -> Optional[TrackInfo]:  # noqa: C901
        """Try to find track by searching LRC providers directly.

        This is the simplest approach - just search for the query and see if
        LRC providers can find it. Works well for queries like "artist title"
        or "title artist" without needing complex parsing.

        Returns:
            TrackInfo if found, None otherwise
        """
        from ...sync import fetch_lyrics_multi_source, get_lrc_duration

        logger.debug(f"Trying direct LRC search: {query}")

        # Try searching with the query as-is
        lrc_text, is_synced, source = fetch_lyrics_multi_source(
            query, "", synced_only=True  # Empty artist, let provider search freely
        )

        if not is_synced or not lrc_text:
            # Also try with query parts swapped (in case order is wrong)
            parts = query.split()
            if len(parts) >= 2:
                # Try "last_part first_parts"
                swapped = f"{parts[-1]} {' '.join(parts[:-1])}"
                lrc_text, is_synced, source = fetch_lyrics_multi_source(
                    swapped, "", synced_only=True
                )

        if not is_synced or not lrc_text:
            return None

        # Get duration from LRC
        lrc_duration = get_lrc_duration(lrc_text)
        if not lrc_duration or lrc_duration < 60:
            return None

        logger.info(
            f"Direct LRC search found lyrics from {source} (duration: {lrc_duration}s)"
        )

        # Try to extract artist/title from LRC metadata tags
        artist, title = self._extract_lrc_metadata(lrc_text)
        derived_from_lrc = bool(artist or title)

        # Fall back to parsing the query
        if not artist or not title:
            artist_hint, title_hint = self._parse_query(query)
            if not artist:
                artist = artist_hint
            if not title:
                title = title_hint or query

        # If still no artist, try MusicBrainz to identify properly
        if not artist or artist == "Unknown":
            mb_artist, mb_title = self._lookup_musicbrainz_for_query(
                query, lrc_duration
            )
            if mb_artist:
                artist = mb_artist
            if mb_title:
                title = mb_title

        # Last resort: try to infer from query
        if not artist:
            artist = self._infer_artist_from_query(query, title)

        # Try split-based search to resolve artist/title when LRC metadata looks like a cover
        # or doesn't clearly match the query.
        split_best = None
        if derived_from_lrc or not artist or not title or artist == "Unknown":
            split_best = self._try_split_search(query)
        if split_best:
            _, split_artist, split_title = split_best
            query_words = set(normalize_title(query, remove_stopwords=True).split())
            lrc_artist_words = set(
                normalize_title(artist or "", remove_stopwords=True).split()
            )
            split_artist_words = set(
                normalize_title(split_artist or "", remove_stopwords=True).split()
            )

            lrc_artist_overlap = len(query_words & lrc_artist_words)
            split_artist_overlap = len(query_words & split_artist_words)

            # Prefer split result when it better matches the query's artist tokens
            if split_artist_overlap > lrc_artist_overlap:
                artist = split_artist or artist
                title = split_title or title

        logger.info(f"Identified from LRC: {artist or 'Unknown'} - {title}")

        # Search YouTube for matching video using identified artist/title
        # This ensures we find the right version, not a cover/tribute
        search_query = f"{artist} {title}" if artist and artist != "Unknown" else query
        youtube_result = self._search_youtube_verified(
            search_query, lrc_duration, artist, title
        )
        if not youtube_result:
            # Fallback to original query if artist-specific search fails
            youtube_result = self._search_youtube_verified(
                query, lrc_duration, artist, title
            )
        if not youtube_result:
            return None

        return TrackInfo(
            artist=artist or "Unknown",
            title=title,
            duration=lrc_duration,
            youtube_url=youtube_result["url"],
            youtube_duration=youtube_result["duration"] or lrc_duration,
            source="syncedlyrics",
            lrc_duration=lrc_duration,
            lrc_validated=True,
        )

    # -------------------------
    # Path A: Search String -> Track
    # -------------------------

    def identify_from_search(self, query: str) -> TrackInfo:
        """Identify track from a search string (not a URL).

        Flow:
        1. Try direct LRC search with the query (simplest approach)
        2. Parse query for artist/title hints
        3. Query MusicBrainz for candidate recordings
        4. If no separator found, try different artist/title splits
        5. Check LRC availability and duration for candidates
        6. Filter out live/remix versions
        7. Search YouTube for videos matching canonical duration

        Args:
            query: Search string like "Artist - Title" or just "Title"

        Returns:
            TrackInfo with canonical artist, title, duration, and YouTube URL
        """
        logger.info(f"Identifying track from search: {query}")

        direct_result = self._try_direct_lrc_search(query)
        if direct_result:
            return direct_result

        artist_hint, title_hint = self._parse_query(query)
        logger.debug(f"Parsed hints: artist='{artist_hint}', title='{title_hint}'")

        recordings = self._query_musicbrainz(query, artist_hint, title_hint)

        if not recordings:
            logger.warning("No MusicBrainz results, falling back to YouTube search")
            return self._fallback_youtube_search(query)

        best = None
        if artist_hint:
            best = self._find_best_with_artist_hint(recordings, query, artist_hint)
        else:
            best = self._find_best_title_only(recordings, title_hint)

        if not best and not artist_hint:
            logger.debug(
                "No match with title-only search, trying artist/title splits..."
            )
            best = self._try_split_search(query)

        if not best:
            logger.warning("No suitable MusicBrainz candidate, falling back to YouTube")
            return self._fallback_youtube_search(query)

        canonical_duration, canonical_artist, canonical_title = best
        logger.info(
            f"Canonical track: {canonical_artist} - {canonical_title} ({canonical_duration}s)"
        )

        from ...sync import fetch_lyrics_for_duration

        _, _, _, lrc_duration = fetch_lyrics_for_duration(
            canonical_title, canonical_artist, canonical_duration, tolerance=8
        )
        lrc_validated = (
            lrc_duration is not None and abs(lrc_duration - canonical_duration) <= 8
        )

        if lrc_duration and not lrc_validated:
            logger.warning(
                f"LRC duration ({lrc_duration}s) doesn't match canonical ({canonical_duration}s) - lyrics timing may be off"
            )

        youtube_result = self._search_youtube_by_duration(
            f"{canonical_artist} {canonical_title}", canonical_duration
        )

        if not youtube_result:
            youtube_result = self._search_youtube_by_duration(query, canonical_duration)

        if not youtube_result:
            raise Y2KaraokeError(f"No YouTube results found for: {query}")

        return TrackInfo(
            artist=canonical_artist,
            title=canonical_title,
            duration=canonical_duration,
            youtube_url=youtube_result["url"],
            youtube_duration=youtube_result["duration"],
            source="musicbrainz",
            lrc_duration=lrc_duration,
            lrc_validated=lrc_validated,
        )

    # -------------------------
    # Path B: YouTube URL -> Track
    # -------------------------

    def identify_from_url(
        self,
        url: str,
        artist_hint: Optional[str] = None,
        title_hint: Optional[str] = None,
    ) -> TrackInfo:
        """Identify track from a YouTube URL.

        Flow:
        1. Get YouTube video metadata (title, uploader, duration)
        2. If artist_hint/title_hint provided, use those directly for LRC search
        3. Otherwise, parse video title for artist/title hints
        4. Query MusicBrainz for candidates
        5. Check LRC for each candidate, score by duration match
        6. Return candidate with best duration match

        Args:
            url: YouTube URL
            artist_hint: Explicit artist name (overrides YouTube metadata parsing)
            title_hint: Explicit song title (overrides YouTube metadata parsing)

        Returns:
            TrackInfo with canonical artist, title, and the given YouTube URL
        """
        logger.info(f"Identifying track from URL: {url}")

        yt_title, yt_uploader, yt_duration = self._get_youtube_metadata(url)
        logger.info(f"YouTube: '{yt_title}' by {yt_uploader} ({yt_duration}s)")

        parsed_artist: Optional[str]
        parsed_title: str

        if artist_hint and title_hint:
            logger.info(f"Using provided artist/title: {artist_hint} - {title_hint}")
            parsed_artist = artist_hint
            parsed_title = title_hint
        else:
            if self._is_likely_non_studio(yt_title):
                logger.warning(
                    f"YouTube video appears to be non-studio version: '{yt_title}'"
                )
                logger.warning(
                    "Lyrics timing may not match. Consider using a studio version URL."
                )

            parsed_artist, parsed_title = self._parse_youtube_title(yt_title)
            logger.debug(
                f"Parsed from title: artist='{parsed_artist}', title='{parsed_title}'"
            )

        search_query = (
            f"{parsed_artist} {parsed_title}" if parsed_artist else parsed_title
        )
        recordings = self._query_musicbrainz(search_query, parsed_artist, parsed_title)

        unique_candidates = self._build_url_candidates(
            yt_uploader, parsed_artist, parsed_title, recordings
        )
        logger.debug(f"Found {len(unique_candidates)} unique candidates to check")

        best_match = self._find_best_lrc_by_duration(
            unique_candidates, yt_duration, parsed_title or yt_title
        )

        if best_match:
            artist, title, lrc_duration = best_match
            lrc_validated = abs(lrc_duration - yt_duration) <= 8

            if self._is_likely_non_studio(yt_title) and not lrc_validated:
                logger.warning(
                    f"Non-studio YouTube video with mismatched LRC duration "
                    f"(YT: {yt_duration}s, LRC: {lrc_duration}s)"
                )

            if not lrc_validated and abs(lrc_duration - yt_duration) > 8:
                alt_result = self._search_matching_youtube_video(
                    artist, title, lrc_duration, yt_duration
                )
                if alt_result:
                    return alt_result

            logger.info(
                f"Best LRC match: {artist} - {title} (LRC duration: {lrc_duration}s, validated: {lrc_validated})"
            )
            return TrackInfo(
                artist=artist,
                title=title,
                duration=yt_duration,
                youtube_url=url,
                youtube_duration=yt_duration,
                source="syncedlyrics",
                lrc_duration=lrc_duration,
                lrc_validated=lrc_validated,
            )

        fallback_artist, fallback_title = self._find_fallback_artist_title(
            unique_candidates, yt_uploader, parsed_artist, parsed_title, yt_title
        )

        logger.warning(
            f"No LRC match found, using: {fallback_artist} - {fallback_title}"
        )

        return TrackInfo(
            artist=fallback_artist,
            title=fallback_title,
            duration=yt_duration,
            youtube_url=url,
            youtube_duration=yt_duration,
            source="youtube",
        )

    # Explicit override so static analysis sees this method on TrackIdentifier.
    def _check_lrc_and_duration(
        self, title: str, artist: str, expected_duration: Optional[int] = None
    ) -> tuple[bool, Optional[int]]:
        return TrackIdentifierHelpers._check_lrc_and_duration(
            self, title, artist, expected_duration
        )
