"""Track identification pipeline for y2karaoke.

This module handles two distinct paths for identifying track information:
- Path A (search string): Query -> MusicBrainz/syncedlyrics -> canonical track -> YouTube
- Path B (YouTube URL): URL -> YouTube duration -> best LRC match by duration
"""

import re
from dataclasses import dataclass
from collections import Counter
from typing import Optional, List, Dict, Any, Tuple

import musicbrainzngs

from ..utils.logging import get_logger
from ..exceptions import Y2KaraokeError

logger = get_logger(__name__)

# Stopwords for multiple languages - filtered during title/artist matching
STOP_WORDS = {
    # English
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "with",
    "in",
    "to",
    "for",
    "by",
    "&",
    "+",
    # Spanish
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "y",
    "de",
    "del",
    "con",
    # French
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "et",
    "de",
    "du",
    "au",
    "aux",
    # German
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "und",
    "von",
    "mit",
    # Italian
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "uno",
    "una",
    "e",
    "di",
    "del",
    "della",
    # Portuguese
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "uns",
    "umas",
    "e",
    "de",
    "do",
    "da",
    "dos",
    "das",
}

# Initialize MusicBrainz
musicbrainzngs.set_useragent(
    "y2karaoke", "1.0", "https://github.com/dtunkelang/y2karaoke"
)


@dataclass
class TrackInfo:
    """Canonical track information from identification pipeline."""

    artist: str
    title: str
    duration: int  # canonical duration in seconds
    youtube_url: str
    youtube_duration: int  # actual YouTube video duration
    source: str  # "musicbrainz", "syncedlyrics", "youtube"
    lrc_duration: Optional[int] = None  # duration implied by LRC lyrics (if found)
    lrc_validated: bool = False  # True if LRC duration matches canonical duration
    # Quality reporting fields
    identification_quality: float = 100.0  # 0-100 confidence score
    quality_issues: Optional[List[str]] = None  # List of quality concerns
    sources_tried: Optional[List[str]] = None  # List of sources attempted
    fallback_used: bool = False  # True if had to fall back from primary source

    def __post_init__(self):
        if self.quality_issues is None:
            object.__setattr__(self, "quality_issues", [])
        if self.sources_tried is None:
            object.__setattr__(self, "sources_tried", [])


class TrackIdentifier:
    """Identifies track information from search queries or YouTube URLs."""

    def __init__(self):
        self._lrc_cache: Dict[tuple, tuple] = {}

    def _try_direct_lrc_search(self, query: str) -> Optional[TrackInfo]:
        """Try to find track by searching LRC providers directly.

        This is the simplest approach - just search for the query and see if
        LRC providers can find it. Works well for queries like "artist title"
        or "title artist" without needing complex parsing.

        Returns:
            TrackInfo if found, None otherwise
        """
        from .sync import fetch_lyrics_multi_source, get_lrc_duration

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

    def _extract_lrc_metadata(
        self, lrc_text: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract artist and title from LRC metadata tags.

        LRC files often have tags like:
        [ar:Artist Name]
        [ti:Song Title]
        """
        artist = None
        title = None

        for line in lrc_text.split("\n")[:20]:  # Check first 20 lines
            line = line.strip()
            # Artist tag
            match = re.match(r"\[ar:(.+)\]", line, re.IGNORECASE)
            if match:
                artist = match.group(1).strip()
                continue
            # Title tag
            match = re.match(r"\[ti:(.+)\]", line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                continue

        return artist, title

    def _infer_artist_from_query(
        self, query: str, title: Optional[str]
    ) -> Optional[str]:
        """Try to infer artist name from query by removing title words."""
        if not title:
            return None

        query_lower = query.lower()
        title_lower = title.lower()

        # Remove title words from query to get potential artist
        remaining = query_lower
        for word in title_lower.split():
            remaining = remaining.replace(word, "", 1)

        # Clean up and capitalize
        remaining = " ".join(remaining.split())
        if remaining and len(remaining) > 2:
            # Title case the result
            return remaining.title()

        return None

    def _lookup_musicbrainz_for_query(
        self, query: str, expected_duration: int
    ) -> tuple[Optional[str], Optional[str]]:
        """Look up MusicBrainz to identify artist/title from a query.

        Uses the expected duration AND query word matching to identify the correct recording.
        """
        query_words = set(self._normalize_title(query).split())

        try:
            # Search MusicBrainz with the full query
            results = musicbrainzngs.search_recordings(recording=query, limit=15)
            recordings = results.get("recording-list", [])

            best_match = None
            best_score = 0

            for rec in recordings:
                length = rec.get("length")
                if not length:
                    continue

                duration_sec = int(length) // 1000

                # Check if duration matches (within 30s)
                if abs(duration_sec - expected_duration) > 30:
                    continue

                # Skip unreasonably long recordings
                if duration_sec > 720:
                    continue

                artist_credits = rec.get("artist-credit", [])
                artists = [a["artist"]["name"] for a in artist_credits if "artist" in a]
                artist_name = " & ".join(artists) if artists else None
                title = rec.get("title")

                if not artist_name or not title:
                    continue

                # Score by how many query words match artist+title
                result_text = f"{artist_name} {title}".lower()
                result_words = set(self._normalize_title(result_text).split())
                matching_words = query_words & result_words
                score = len(matching_words)

                # Require at least 2 matching words (to avoid false positives)
                if score >= 2 and score > best_score:
                    best_score = score
                    best_match = (artist_name, title)
                    logger.debug(
                        f"MusicBrainz candidate: {artist_name} - {title} (score={score})"
                    )

            if best_match:
                logger.debug(
                    f"MusicBrainz identified: {best_match[0]} - {best_match[1]}"
                )
                return best_match

        except Exception as e:
            logger.debug(f"MusicBrainz lookup failed: {e}")

        return None, None

    # -------------------------
    # Path A: Search String -> Track
    # -------------------------

    def _score_split_candidate(
        self, candidate: tuple, split_artist: str, split_title: str, query: str
    ) -> Optional[int]:
        """Score a split candidate for matching quality. Returns None to reject."""
        duration, artist, title = candidate
        query_words = set(query.lower().split())
        query_lower = query.lower()

        if duration > 720:
            logger.debug(
                f"Skipping candidate with unreasonable duration: {artist} - {title} ({duration}s)"
            )
            return None

        has_lrc, _ = self._check_lrc_and_duration(title, artist)

        score = 10 if has_lrc else 0

        # Artist match bonus
        if (
            split_artist.lower() in artist.lower()
            or artist.lower() in split_artist.lower()
        ):
            score += 20

        # Artist in query check
        artist_words = set(w for w in artist.lower().split() if len(w) > 2)
        if any(w in query_lower for w in artist_words):
            score += 15
        else:
            score -= 30

        # Title matching
        split_title_norm = self._normalize_title(split_title, remove_stopwords=True)
        result_title_norm = self._normalize_title(title, remove_stopwords=True)
        title_words = set(self._normalize_title(title).split())
        split_title_words = set(self._normalize_title(split_title).split())
        title_overlap = split_title_words & title_words

        if not title_overlap:
            logger.debug(
                f"Skipping candidate with no title overlap: {artist} - {title}"
            )
            return None

        if split_title_norm == result_title_norm:
            score += 50
        elif (
            split_title_norm in result_title_norm
            or result_title_norm in split_title_norm
        ):
            score += 25
        else:
            score += len(title_overlap) * 10

        # Query word matching bonus
        result_words = set(f"{artist} {title}".lower().split())
        score += len(query_words & result_words) * 3

        # Penalty for artist names that look like song titles
        song_title_indicators = ["rhapsody", "symphony", "concerto", "song", "ballad"]
        if any(ind in artist.lower() for ind in song_title_indicators):
            score -= 15

        return score

    def _try_split_search(self, query: str) -> Optional[tuple]:
        """Try artist/title splits to find best candidate."""
        splits = self._try_artist_title_splits(query)
        split_candidates = []

        for split_artist, split_title in splits:
            logger.debug(
                f"Trying split: artist='{split_artist}', title='{split_title}'"
            )
            split_recordings = self._query_musicbrainz(
                f"{split_artist} {split_title}", split_artist, split_title
            )
            if not split_recordings:
                continue

            candidate = self._find_best_with_artist_hint(
                split_recordings, query, split_artist
            )
            if not candidate:
                continue

            score = self._score_split_candidate(
                candidate, split_artist, split_title, query
            )
            if score is not None:
                duration, artist, title = candidate
                has_lrc, _ = self._check_lrc_and_duration(title, artist)
                logger.debug(
                    f"Split candidate: {artist} - {title} (score={score}, has_lrc={has_lrc})"
                )
                split_candidates.append((score, candidate))

        if split_candidates:
            split_candidates.sort(key=lambda x: x[0], reverse=True)
            best = split_candidates[0][1]
            logger.info(f"Selected best split match: {best[1]} - {best[2]}")
            return best

        return None

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

        from .sync import fetch_lyrics_for_duration

        _, _, _, lrc_duration = fetch_lyrics_for_duration(
            canonical_title, canonical_artist, canonical_duration, tolerance=20
        )
        lrc_validated = (
            lrc_duration is not None and abs(lrc_duration - canonical_duration) <= 20
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

    def _build_url_candidates(
        self,
        yt_uploader: str,
        parsed_artist: Optional[str],
        parsed_title: str,
        recordings: List[Dict],
    ) -> List[Dict]:
        """Build deduplicated candidate list from various sources."""
        candidates = []

        if yt_uploader and parsed_title:
            candidates.append({"artist": yt_uploader, "title": parsed_title})

        if parsed_artist and parsed_title:
            if not yt_uploader or parsed_artist.lower() != yt_uploader.lower():
                candidates.append({"artist": parsed_artist, "title": parsed_title})

        for rec in recordings:
            artist_credits = rec.get("artist-credit", [])
            artists = [a["artist"]["name"] for a in artist_credits if "artist" in a]
            artist_name = " & ".join(artists) if artists else None
            title = rec.get("title")
            if artist_name and title:
                candidates.append({"artist": artist_name, "title": title})

        seen = set()
        unique = []
        for c in candidates:
            key = (c["artist"].lower(), c["title"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def _search_matching_youtube_video(
        self, artist: str, title: str, lrc_duration: int, yt_duration: int
    ) -> Optional[TrackInfo]:
        """Search for a YouTube video matching LRC duration."""
        logger.info(
            f"LRC duration ({lrc_duration}s) differs from YouTube ({yt_duration}s)"
        )
        logger.info("Searching for YouTube video matching LRC duration...")

        search_queries = [
            f"{artist} {title} official audio",
            f"{artist} {title} audio",
            f"{artist} {title}",
        ]

        for search_query in search_queries:
            alt_youtube = self._search_youtube_verified(
                search_query, lrc_duration, artist, title
            )
            if alt_youtube and alt_youtube["duration"]:
                if abs(alt_youtube["duration"] - lrc_duration) <= 15:
                    logger.info(
                        f"Found matching YouTube video: {alt_youtube['url']} ({alt_youtube['duration']}s)"
                    )
                    return TrackInfo(
                        artist=artist,
                        title=title,
                        duration=lrc_duration,
                        youtube_url=alt_youtube["url"],
                        youtube_duration=alt_youtube["duration"],
                        source="syncedlyrics",
                        lrc_duration=lrc_duration,
                        lrc_validated=True,
                    )

        logger.warning(
            f"Could not find clean YouTube video matching LRC duration ({lrc_duration}s)"
        )

        if abs(lrc_duration - yt_duration) > 20:
            alt_youtube = self._search_youtube_verified(
                f"{artist} {title} radio edit", lrc_duration, artist, title
            )
            if (
                alt_youtube
                and alt_youtube["duration"]
                and abs(alt_youtube["duration"] - lrc_duration) <= 15
            ):
                logger.info(
                    f"Found radio edit: {alt_youtube['url']} ({alt_youtube['duration']}s)"
                )
                return TrackInfo(
                    artist=artist,
                    title=title,
                    duration=lrc_duration,
                    youtube_url=alt_youtube["url"],
                    youtube_duration=alt_youtube["duration"],
                    source="syncedlyrics",
                    lrc_duration=lrc_duration,
                    lrc_validated=True,
                )

        return None

    def _find_fallback_artist_title(
        self,
        unique_candidates: List[Dict],
        yt_uploader: str,
        parsed_artist: Optional[str],
        parsed_title: str,
        yt_title: str,
    ) -> tuple[str, str]:
        """Find fallback artist/title when no LRC match is found."""
        fallback_artist = None
        fallback_title = None

        if unique_candidates:
            for c in unique_candidates:
                if c["artist"] != yt_uploader:
                    fallback_artist = c["artist"]
                    fallback_title = c["title"]
                    break

        if not fallback_artist:
            fallback_artist = parsed_artist
            fallback_title = parsed_title

        if not fallback_artist and parsed_title:
            splits = self._try_artist_title_splits(parsed_title)
            for split_artist, split_title in splits[:3]:
                has_lrc, _ = self._check_lrc_and_duration(split_title, split_artist)
                if has_lrc:
                    fallback_artist = split_artist
                    fallback_title = split_title
                    logger.info(
                        f"Found LRC with title split: {split_artist} - {split_title}"
                    )
                    break

        if not fallback_artist:
            fallback_artist = yt_uploader or "Unknown"
            fallback_title = parsed_title or yt_title

        # Ensure we always return strings
        final_artist: str = fallback_artist if fallback_artist else "Unknown"
        final_title: str = fallback_title if fallback_title else yt_title

        return final_artist, final_title

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
            lrc_validated = abs(lrc_duration - yt_duration) <= 15

            if self._is_likely_non_studio(yt_title) and not lrc_validated:
                logger.warning(
                    f"Non-studio YouTube video with mismatched LRC duration "
                    f"(YT: {yt_duration}s, LRC: {lrc_duration}s)"
                )

            if not lrc_validated and abs(lrc_duration - yt_duration) > 15:
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

    # -------------------------
    # Helper Methods
    # -------------------------

    def _parse_query(self, query: str) -> tuple[Optional[str], str]:
        """Parse a search query for artist/title hints.

        Supports multiple separator formats:
        - "Artist - Title" (hyphen)
        - "Artist – Title" (en-dash)
        - "Artist — Title" (em-dash)
        - "Title by Artist" format
        - "Artist: Title" (colon)
        """
        query_clean = query.strip()

        # Try various separators in order of preference
        separators = [" - ", " – ", " — ", ": "]
        for sep in separators:
            if sep in query_clean:
                parts = query_clean.split(sep, 1)
                return parts[0].strip(), parts[1].strip()

        # Try "Title by Artist" format
        by_match = re.match(r"^(.+?)\s+by\s+(.+)$", query_clean, re.IGNORECASE)
        if by_match:
            title = by_match.group(1).strip()
            artist = by_match.group(2).strip()
            return artist, title

        return None, query_clean

    def _try_artist_title_splits(self, query: str) -> List[tuple[str, str]]:
        """Generate possible artist/title splits from a query without separators.

        For queries like "beatles yesterday", tries splitting at each word
        boundary to find potential artist/title combinations.

        Returns:
            List of (artist, title) tuples to try, ordered by likelihood
        """
        words = query.strip().split()
        if len(words) < 2:
            return []

        splits = []

        # Most common pattern: first word(s) = artist, rest = title
        # Try shorter artists first (single word artists are common)
        for i in range(1, len(words)):
            artist = " ".join(words[:i])
            title = " ".join(words[i:])
            splits.append((artist, title))

        # Less common: last word(s) = artist, beginning = title
        # (e.g., "yesterday beatles" -> "beatles", "yesterday")
        for i in range(len(words) - 1, 0, -1):
            title = " ".join(words[:i])
            artist = " ".join(words[i:])
            # Only add if not already present
            if (artist, title) not in splits:
                splits.append((artist, title))

        return splits

    def _parse_youtube_title(self, title: str) -> tuple[Optional[str], str]:
        """Parse a YouTube video title for artist/title."""
        # Remove common suffixes
        cleaned = title
        for suffix in [
            "Official Music Video",
            "Official Video",
            "Official Audio",
            "Lyric Video",
            "Lyrics",
            "HD",
            "4K",
            "MV",
            "(Official)",
        ]:
            cleaned = re.sub(
                rf"\s*[\(\[]?\s*{re.escape(suffix)}\s*[\)\]]?\s*$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        # Only remove parenthetical/bracket content that looks like metadata
        # (e.g., "Remastered 2023", "feat. Artist", "Live") - NOT actual title parts like "(Don't Fear)"
        metadata_patterns = [
            r"\s*[\(\[](?:remaster(?:ed)?|remix|live|acoustic|demo|edit|version|ver\.?|"
            r"feat\.?\s+[^)\]]+|ft\.?\s+[^)\]]+|with\s+[^)\]]+|"
            r"\d{4}(?:\s+remaster)?|single|album|ep|radio|extended|original|"
            r"bonus\s+track|deluxe|explicit|clean)[\)\]]\s*",
        ]
        for pattern in metadata_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Try to split on " - " or " – " or " — "
        for sep in [" - ", " – ", " — ", " | "]:
            if sep in cleaned:
                parts = cleaned.split(sep, 1)
                return parts[0].strip(), parts[1].strip()

        # No separator found - try to detect "Artist Name song title" pattern
        # by querying MusicBrainz with different word splits
        parsed = self._try_parse_artist_from_title(cleaned)
        if parsed:
            return parsed

        return None, cleaned

    def _try_parse_artist_from_title(self, title: str) -> Optional[tuple[str, str]]:
        """Try to extract artist from a title without separators.

        For titles like "The White Stripes fell in love with a girl",
        tries splitting at word boundaries and checking MusicBrainz for valid artists.
        """
        words = title.split()
        if len(words) < 3:
            return None

        # Try different splits - common patterns are 2-4 word artist names
        # Start from 2 words to avoid matching single words like "The" to "The Beatles"
        for split_point in range(2, min(5, len(words))):
            potential_artist = " ".join(words[:split_point])
            potential_title = " ".join(words[split_point:])

            # Skip if potential title is too short
            if len(potential_title.split()) < 2:
                continue

            # Skip very short potential artists (avoid "The" matching "The Beatles")
            if len(potential_artist) < 5:
                continue

            # Quick check: query MusicBrainz for this artist
            try:
                results = musicbrainzngs.search_artists(
                    artist=potential_artist, limit=3
                )
                artists = results.get("artist-list", [])

                for artist in artists:
                    artist_name = artist.get("name", "")
                    # Check for exact match (case-insensitive)
                    if artist_name.lower() == potential_artist.lower():
                        logger.debug(
                            f"Detected artist '{artist_name}' from title: '{title}'"
                        )
                        return artist_name, potential_title

                    # Accept if potential_artist contains most of artist_name
                    # (e.g., "White Stripes" matches "The White Stripes")
                    # But NOT if potential_artist is just a small part of artist_name
                    artist_name_lower = artist_name.lower()
                    potential_lower = potential_artist.lower()

                    # Check if one contains the other with significant overlap
                    if potential_lower in artist_name_lower:
                        # potential is substring of artist - check it's significant
                        if len(potential_lower) >= len(artist_name_lower) * 0.6:
                            logger.debug(
                                f"Detected artist '{artist_name}' from title: '{title}'"
                            )
                            return artist_name, potential_title
                    elif artist_name_lower in potential_lower:
                        # artist is substring of potential - check it's significant
                        if len(artist_name_lower) >= len(potential_lower) * 0.6:
                            logger.debug(
                                f"Detected artist '{artist_name}' from title: '{title}'"
                            )
                            return artist_name, potential_title

            except Exception as e:
                logger.debug(f"Artist detection query failed: {e}")
                continue

        return None

    def _query_musicbrainz(
        self,
        query: str,
        artist_hint: Optional[str],
        title_hint: str,
        max_retries: int = 3,
    ) -> List[Dict]:
        """Query MusicBrainz for recordings matching the query.

        Prioritizes studio recordings by:
        1. Including release information to check for album vs. compilation
        2. Filtering and sorting by recording attributes
        3. Boosting recordings with titles matching the search hint

        Includes retry logic for transient network errors.
        """
        import time

        for attempt in range(max_retries + 1):
            try:
                # Include release info to check release type
                if artist_hint:
                    results = musicbrainzngs.search_recordings(
                        recording=title_hint, artist=artist_hint, limit=25
                    )
                else:
                    results = musicbrainzngs.search_recordings(
                        recording=query, limit=25
                    )

                recordings = results.get("recording-list", [])

                # Score and sort recordings to prioritize studio versions and title matches
                scored = []
                for rec in recordings:
                    score = self._score_recording_studio_likelihood(rec)

                    # Bonus for title match (when user explicitly provides title)
                    if title_hint:
                        rec_title = rec.get("title", "")

                        # First check exact match (with stopwords retained)
                        title_hint_norm = self._normalize_title(
                            title_hint, remove_stopwords=False
                        )
                        rec_title_norm = self._normalize_title(
                            rec_title, remove_stopwords=False
                        )

                        if rec_title_norm == title_hint_norm:
                            score += 100  # Strong bonus for exact match - should outweigh album release bonus
                        else:
                            # Check match with stopwords removed (looser matching)
                            title_hint_no_stop = self._normalize_title(
                                title_hint, remove_stopwords=True
                            )
                            rec_title_no_stop = self._normalize_title(
                                rec_title, remove_stopwords=True
                            )

                            if rec_title_no_stop == title_hint_no_stop:
                                score += (
                                    30  # Moderate bonus for stopword-invariant match
                                )
                            elif (
                                title_hint_no_stop in rec_title_no_stop
                                or rec_title_no_stop in title_hint_no_stop
                            ):
                                score += 15  # Small bonus for partial match

                    scored.append((score, rec))

                # Sort by score (highest first) and take top 15
                scored.sort(key=lambda x: x[0], reverse=True)
                return [rec for _, rec in scored[:15]]

            except Exception as e:
                error_msg = str(e).lower()
                is_transient = any(
                    x in error_msg
                    for x in [
                        "connection",
                        "timeout",
                        "reset",
                        "temporarily",
                        "urlopen error",
                    ]
                )

                if is_transient and attempt < max_retries:
                    delay = 1.0 * (2**attempt)
                    logger.debug(
                        f"MusicBrainz transient error, retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(f"MusicBrainz search failed: {e}")
                    return []

        return []

    def _score_recording_studio_likelihood(self, recording: Dict) -> int:
        """Score a MusicBrainz recording by how likely it is to be a studio version.

        Higher score = more likely to be the canonical studio recording.

        Checks:
        - Disambiguation field for live/remix/demo indicators
        - Release list for album vs. compilation appearances
        - Title patterns indicating non-studio versions
        """
        score = 100  # Base score
        title = recording.get("title", "")
        disambiguation = recording.get("disambiguation", "")

        # Check disambiguation field (MusicBrainz's own classification)
        disambig_lower = disambiguation.lower()
        non_studio_disambig = [
            "live",
            "demo",
            "remix",
            "acoustic",
            "radio edit",
            "single version",
            "alternate",
            "instrumental",
            "a]cappella",
            "karaoke",
            "cover",
            "session",
            "bootleg",
            "rehearsal",
            "unplugged",
            "stripped",
        ]
        for term in non_studio_disambig:
            if term in disambig_lower:
                score -= 80
                logger.debug(
                    f"Recording '{title}' penalized for disambiguation: '{disambiguation}'"
                )
                break

        # Check title for non-studio indicators
        if self._is_likely_non_studio(title):
            score -= 60

        # Check release list if available
        releases = recording.get("release-list", [])
        has_album_release = False
        has_compilation_only = True

        for release in releases:
            release_group = release.get("release-group", {})
            primary_type = release_group.get("primary-type", "").lower()
            secondary_types = [
                t.lower() for t in release_group.get("secondary-type-list", [])
            ]

            # Prefer recordings that appear on actual albums
            if primary_type == "album" and "compilation" not in secondary_types:
                has_album_release = True
                has_compilation_only = False
            elif primary_type in ["single", "ep"]:
                has_compilation_only = False

        if has_album_release:
            score += 20  # Bonus for appearing on studio album
        elif has_compilation_only and releases:
            score -= 30  # Penalty for only appearing on compilations

        return score

    def _find_best_with_artist_hint(
        self, recordings: List[Dict], query: str, artist_hint: str
    ) -> Optional[tuple]:
        """Find best recording when artist hint is available."""
        # Use artist_hint words for matching, not the full query
        # Require significant overlap to prevent "Hamlet Express" matching "aucun express"
        artist_hint_words = set(w.lower() for w in artist_hint.split() if len(w) > 2)
        artist_hint_lower = self._normalize_title(artist_hint)
        matches = []

        for rec in recordings:
            length = rec.get("length")
            if not length:
                continue

            duration_sec = int(length) // 1000

            # Skip recordings with unreasonable durations (> 12 minutes)
            # These are likely albums, compilations, or misidentified entries
            if duration_sec > 720:
                continue

            artist_credits = rec.get("artist-credit", [])
            artists = [a["artist"]["name"] for a in artist_credits if "artist" in a]
            artist_str_lower = self._normalize_title(" ".join(artists))
            artist_words = set(artist_str_lower.split())

            # Check if artist matches the hint with substantial overlap
            # Require either:
            # 1. Full containment (one is substring of the other)
            # 2. Majority word overlap (> 50% for multi-word, or exact match for single word)
            if (
                artist_hint_lower in artist_str_lower
                or artist_str_lower in artist_hint_lower
            ):
                artist_matches = True
            elif artist_hint_words and artist_words:
                # Calculate word overlap
                overlap = artist_hint_words & artist_words
                if len(artist_hint_words) == 1:
                    # Single word hint - require exact match
                    artist_matches = (
                        artist_hint_words == artist_words
                        or artist_hint_lower == artist_str_lower
                    )
                else:
                    # Multi-word hint - require > 50% overlap (strictly greater)
                    overlap_ratio = len(overlap) / len(artist_hint_words)
                    artist_matches = overlap_ratio > 0.5
            else:
                artist_matches = False

            if artist_matches:
                artist_name = " & ".join(artists) if artists else None
                title = rec.get("title")
                matches.append((duration_sec, artist_name, title))

        if not matches:
            return None

        return self._score_and_select_best(matches)

    def _is_likely_cover_recording(
        self, title: str, artist_name: str, query_words: set
    ) -> bool:
        """Check if a recording is likely a cover based on parenthetical credits.

        Pattern: Title has "(Artist)" crediting original artist, but performing artist is different.
        e.g., "Bohemian Rhapsody (Queen)" by "Doctor Octoroc" is likely a cover.
        """
        paren_match = re.search(r"\(([^)]+)\)", title)
        if not paren_match:
            return False

        paren_content = paren_match.group(1).lower().strip()
        paren_words = set(paren_content.split())
        artist_words = set(artist_name.lower().split())
        query_matches_paren = any(
            word in paren_content for word in query_words if len(word) > 3
        )

        if query_matches_paren and not (paren_words & artist_words):
            logger.debug(
                f"Skipping likely cover (credits original artist): {artist_name} - {title}"
            )
            return True
        return False

    def _select_best_from_artist_matches(
        self,
        artist_matches: List[Dict],
        lrc_duration: Optional[int],
        mb_consensus: Optional[int],
    ) -> tuple:
        """Select best recording from matches for a given artist, using duration matching."""
        if lrc_duration and mb_consensus:
            lrc_mb_diff = abs(lrc_duration - mb_consensus)
            if lrc_mb_diff > 30:
                logger.debug(
                    f"LRC duration ({lrc_duration}s) differs from MB"
                    f" consensus ({mb_consensus}s) by {lrc_mb_diff}s - using MB consensus"
                )
                artist_matches.sort(key=lambda c: abs(c["duration"] - mb_consensus))
            else:
                artist_matches.sort(key=lambda c: abs(c["duration"] - lrc_duration))
        elif lrc_duration:
            artist_matches.sort(key=lambda c: abs(c["duration"] - lrc_duration))

        best = artist_matches[0]
        return (best["duration"], best["artist"], best["title"])

    def _find_best_title_only(
        self, recordings: List[Dict], title_hint: str
    ) -> Optional[tuple]:
        """Find best recording for title-only searches using consensus and LRC preference.

        Note: This method is conservative - it requires an exact normalized title match.
        If the query contains what looks like an artist name (e.g., "bohemian rhapsody queen"),
        the caller should also try artist/title splits.
        """
        title_normalized = self._normalize_title(title_hint)
        title_normalized_no_stop = self._normalize_title(
            title_hint, remove_stopwords=True
        )
        query_words = set(title_hint.lower().split())
        candidates = []

        for rec in recordings:
            title = rec.get("title", "")
            length = rec.get("length")
            if not length:
                continue

            duration_sec = int(length) // 1000
            if duration_sec > 720:
                continue

            rec_title_normalized = self._normalize_title(title)
            rec_title_no_stop = self._normalize_title(title, remove_stopwords=True)

            if (
                rec_title_normalized != title_normalized
                and rec_title_no_stop != title_normalized_no_stop
            ):
                continue

            artist_credits = rec.get("artist-credit", [])
            artists = [a["artist"]["name"] for a in artist_credits if "artist" in a]
            artist_name = " & ".join(artists) if artists else None

            if not artist_name:
                continue

            if self._is_likely_cover_recording(title, artist_name, query_words):
                continue

            candidates.append(
                {
                    "duration": duration_sec,
                    "artist": artist_name,
                    "title": title,
                }
            )

        if not candidates:
            return None

        artist_counts = Counter(c["artist"] for c in candidates)
        all_artists = artist_counts.most_common()

        all_durations = [c["duration"] for c in candidates]
        rounded_durations = [d // 5 * 5 for d in all_durations]
        mb_consensus_duration = (
            Counter(rounded_durations).most_common(1)[0][0]
            if rounded_durations
            else None
        )

        # First pass: find artist with LRC available
        for artist_name, count in all_artists:
            artist_matches = [c for c in candidates if c["artist"] == artist_name]
            sample_title = artist_matches[0]["title"]

            lrc_available, lrc_duration = self._check_lrc_and_duration(
                sample_title, artist_name
            )
            if lrc_available:
                return self._select_best_from_artist_matches(
                    artist_matches, lrc_duration, mb_consensus_duration
                )

        # Fallback: use most common artist with duration consensus
        if len(candidates) >= 2:
            most_common_artist, artist_count = all_artists[0]
            if artist_count >= 2 and artist_count / len(candidates) >= 0.4:
                artist_matches = [
                    c for c in candidates if c["artist"] == most_common_artist
                ]
                durations = [c["duration"] for c in artist_matches]
                rounded = [d // 5 * 5 for d in durations]
                duration_counts = Counter(rounded)
                if duration_counts.most_common(1)[0][1] >= 2:
                    best = artist_matches[0]
                    return (best["duration"], best["artist"], best["title"])

        return None

    def _normalize_title(self, title: str, remove_stopwords: bool = False) -> str:
        """Normalize title for comparison.

        Args:
            title: Title string to normalize
            remove_stopwords: If True, remove common stopwords (the, el, los, etc.)
        """
        normalized = re.sub(r"[,.\-:;\'\"!?()]", " ", title.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()

        if remove_stopwords:
            words = [w for w in normalized.split() if w not in STOP_WORDS]
            normalized = " ".join(words)

        return normalized

    def _score_and_select_best(self, matches: List[tuple]) -> Optional[tuple]:
        """Score matches and select the best one, penalizing live/remix versions.

        Prefers shorter durations when scores are equal, as original studio versions
        are typically shorter than extended/live versions.
        """
        if not matches:
            return None

        def score_match(m):
            duration, artist, title = m
            score = 0

            # Penalize parenthetical suffixes (live, remix, demo, etc.)
            paren_match = re.search(r"\([^)]+\)\s*$", title)
            if paren_match:
                paren_content = paren_match.group().lower()
                if any(
                    word in paren_content
                    for word in [
                        "live",
                        "remix",
                        "demo",
                        "acoustic",
                        "radio",
                        "edit",
                        "version",
                    ]
                ):
                    score -= 100
                else:
                    score -= 50

            if re.search(r"\[[^\]]+\]\s*$", title):
                score -= 50

            return score

        # Find most common duration among clean matches
        clean_matches = [m for m in matches if score_match(m) >= 0]
        if clean_matches:
            durations = [m[0] for m in clean_matches]
        else:
            durations = [m[0] for m in matches]

        # Find duration clusters and prefer the shorter common duration
        # (original album versions are typically shorter than extended/live)
        rounded = [d // 10 * 10 for d in durations]  # Round to 10s for clustering
        duration_counts = Counter(rounded)

        # Get durations with at least 2 occurrences, sorted by duration (shortest first)
        common_durations = sorted([d for d, c in duration_counts.items() if c >= 2])
        if common_durations:
            # Prefer the shortest common duration (likely original version)
            target_duration = common_durations[0]
        else:
            # Fallback to most common single duration
            target_duration = duration_counts.most_common(1)[0][0]

        def final_score(m):
            title_score = score_match(m)
            duration_diff = abs(m[0] - target_duration)
            # Use negative duration as tiebreaker (prefer shorter)
            return (title_score, -duration_diff, -m[0])

        return max(matches, key=final_score)

    def _check_lrc_and_duration(
        self, title: str, artist: str, expected_duration: Optional[int] = None
    ) -> tuple[bool, Optional[int]]:
        """Check if synced LRC lyrics are available, valid, and get implied duration.

        Uses improved LRC validation to ensure lyrics have sufficient quality:
        - Minimum timestamp density
        - Reasonable coverage of song duration
        - No critical gaps
        """
        cache_key = (artist.lower(), title.lower())
        if cache_key in self._lrc_cache:
            return self._lrc_cache[cache_key]

        try:
            from .sync import (
                fetch_lyrics_multi_source,
                get_lrc_duration,
                validate_lrc_quality,
                SYNCEDLYRICS_AVAILABLE,
            )

            if not SYNCEDLYRICS_AVAILABLE:
                no_sync: Tuple[bool, Optional[int]] = (False, None)
                self._lrc_cache[cache_key] = no_sync
                return no_sync

            lrc_text, is_synced, _ = fetch_lyrics_multi_source(
                title, artist, synced_only=True
            )
            if not is_synced or not lrc_text:
                not_synced: Tuple[bool, Optional[int]] = (False, None)
                self._lrc_cache[cache_key] = not_synced
                return not_synced

            # Validate LRC quality
            is_valid, reason = validate_lrc_quality(lrc_text, expected_duration)
            if not is_valid:
                logger.debug(f"LRC for {artist} - {title} failed validation: {reason}")
                invalid: Tuple[bool, Optional[int]] = (False, None)
                self._lrc_cache[cache_key] = invalid
                return invalid

            # Get duration using improved calculation
            implied_duration = get_lrc_duration(lrc_text)
            found: Tuple[bool, Optional[int]] = (True, implied_duration)

            self._lrc_cache[cache_key] = found
            return found

        except Exception as e:
            logger.debug(f"LRC check failed for {artist} - {title}: {e}")
            error: Tuple[bool, Optional[int]] = (False, None)
            self._lrc_cache[cache_key] = error
            return error

    def _find_best_lrc_by_duration(
        self,
        candidates: List[Dict],
        target_duration: int,
        title_hint: str = "",
        tolerance: int = 15,
    ) -> Optional[tuple[str, str, int]]:
        """Find the candidate whose LRC duration and title best match.

        Scores candidates by both duration match AND title similarity to
        ensure we pick the right song, not just any song by the same artist.

        Args:
            candidates: List of dicts with 'artist' and 'title' keys
            target_duration: Target duration in seconds (from YouTube)
            title_hint: The expected title (from YouTube or parsed)
            tolerance: Maximum acceptable duration difference in seconds

        Returns:
            Tuple of (artist, title, lrc_duration) or None if no match found
        """
        from difflib import SequenceMatcher

        scored_matches = []
        fallback_match = None

        # Normalize title hint for comparison
        title_hint_words = set(
            self._normalize_title(title_hint, remove_stopwords=True).split()
        )

        for candidate in candidates:
            artist = candidate["artist"]
            title = candidate["title"]

            # Pass expected duration to help with validation
            lrc_available, lrc_duration = self._check_lrc_and_duration(
                title, artist, expected_duration=target_duration
            )

            if not lrc_available:
                continue

            if lrc_duration is None:
                # LRC available but couldn't determine duration
                # Keep as fallback if nothing else works
                if fallback_match is None:
                    fallback_match = (artist, title, target_duration)
                continue

            # Score by both duration and title similarity
            duration_diff = abs(lrc_duration - target_duration)

            # Title similarity: check word overlap and sequence match
            candidate_title_normalized = self._normalize_title(
                title, remove_stopwords=True
            )
            candidate_title_words = set(candidate_title_normalized.split())

            # Word overlap score (0-1)
            if title_hint_words:
                word_overlap = len(title_hint_words & candidate_title_words) / len(
                    title_hint_words
                )
            else:
                word_overlap = 0

            # Sequence similarity (0-1)
            title_hint_norm = self._normalize_title(title_hint, remove_stopwords=False)
            seq_similarity = SequenceMatcher(
                None, title_hint_norm, candidate_title_normalized
            ).ratio()

            # Combined title score (higher is better)
            title_score = (word_overlap * 0.6) + (seq_similarity * 0.4)

            # Duration score (0-1, 1 = perfect match)
            duration_score = max(0, 1 - (duration_diff / 60))  # Linear decay over 60s

            # Combined score: title match is more important than duration
            # A perfect title match with okay duration beats okay title with perfect duration
            combined_score = (title_score * 0.7) + (duration_score * 0.3)

            logger.debug(
                f"LRC candidate: {artist} - {title}, "
                f"duration={lrc_duration}s (diff={duration_diff}s), "
                f"title_score={title_score:.2f}, combined={combined_score:.2f}"
            )

            scored_matches.append(
                (combined_score, duration_diff, artist, title, lrc_duration)
            )

        if not scored_matches:
            if fallback_match:
                logger.warning("Using LRC with unknown duration as fallback")
                return fallback_match
            return None

        # Sort by combined score (highest first), then by duration diff (lowest first)
        scored_matches.sort(key=lambda x: (-x[0], x[1]))

        best = scored_matches[0]
        combined_score, duration_diff, artist, title, lrc_duration = best

        # Warn if title match is poor
        if combined_score < 0.3:
            logger.warning(
                f"Best LRC match has low title similarity ({combined_score:.2f})"
            )

        if duration_diff <= tolerance:
            return (artist, title, lrc_duration)
        else:
            logger.warning(
                f"Best LRC match has {duration_diff}s duration difference (tolerance: {tolerance}s)"
            )
            return (artist, title, lrc_duration)

    def _is_likely_non_studio(self, title: str) -> bool:
        """Check if a title suggests a non-studio version.

        Comprehensive detection of live, remix, acoustic, cover, and other
        non-canonical versions to help select studio recordings.

        Note: "single edit", "radio edit", "album version" etc. ARE studio
        recordings - they're just different cuts of the studio recording.
        """
        title_lower = title.lower()

        # First check for studio edit/version indicators - these are NOT non-studio
        # Patterns like "single edit", "radio version", "album edit" indicate official releases
        studio_edit_pattern = r"\b(single|radio|album)\s*(edit|version)?\b"
        if re.search(studio_edit_pattern, title_lower):
            # Check if this is actually a radio SHOW (non-studio) vs radio EDIT (studio)
            # "radio edit" = studio, "radio 1 session" = non-studio
            if "radio" in title_lower:
                # Radio show indicators
                radio_show_patterns = [
                    r"radio\s*\d",  # "radio 1", "radio 2"
                    r"radio\s+session",
                    r"bbc\s+radio",
                    r"radio\s+show",
                ]
                is_radio_show = any(
                    re.search(p, title_lower) for p in radio_show_patterns
                )
                if not is_radio_show:
                    # This looks like "radio edit" or "radio version" - it's studio
                    return False
            else:
                # "single edit", "album version" etc. - definitely studio
                return False

        # Terms that indicate live/alternate versions
        non_studio_terms = [
            # Live performances
            "live",
            "concert",
            "performance",
            "performs",
            "performing",
            "tour",
            "in concert",
            # Alternate versions (but NOT radio/single/album edits - handled above)
            "acoustic",
            "unplugged",
            "stripped",
            "piano version",
            "remix",
            "extended",
            "extended mix",
            "demo",
            "rehearsal",
            "bootleg",
            "outtake",
            "alternate take",
            # Other artists
            "cover",
            "tribute",
            "karaoke",
            "instrumental",
            # Not music
            "reaction",
            "tutorial",
            "lesson",
            "how to play",
            "guitar lesson",
            # Audio effects
            "slowed",
            "sped up",
            "reverb",
            "8d audio",
            "nightcore",
            "bass boosted",
            # Sessions (these are typically live/alternate recordings)
            "session",
            "sessions",
            "bbc session",
            "peel session",
            "maida vale",
            # Parody
            "parody",
            "weird al",
        ]

        # Check for any of these terms
        for term in non_studio_terms:
            if term in title_lower:
                return True

        # Check for common live venue patterns
        live_patterns = [
            # General venue patterns
            r"\bat\b.*\b(show|festival|arena|stadium|hall|theater|theatre|club|center|centre)\b",
            r"\blive\s+(at|from|in)\b",
            # TV shows
            r"\b(snl|saturday night live|letterman|fallon|kimmel|conan|ellen|tonight show)\b",
            r"\b(jools holland|later with|top of the pops|totp|graham norton)\b",
            r"\b(tiny desk|npr|kexp|colors?\s*show|a]colors)\b",
            # Festivals
            r"\b(glastonbury|coachella|lollapalooza|reading|leeds|bonnaroo)\b",
            r"\b(rock am ring|download|download festival|wacken|hellfest)\b",
            r"\b(south by southwest|sxsw|austin city limits|acl)\b",
            # Awards shows
            r"\b(mtv|vma|grammy|grammys|brit awards?|ama|american music)\b",
            r"\b(billboard|bet awards|iheartradio)\b",
            # Unplugged/session series
            r"\b(unplugged|stripped|acoustic sessions?)\b",
            r"\b(spotify\s*sessions?|apple\s*music\s*sessions?)\b",
            # Year indicators often mean live recordings
            r"\b(19|20)\d{2}\s+(tour|live|concert|performance)\b",
        ]
        for pattern in live_patterns:
            if re.search(pattern, title_lower):
                return True

        # Check for parenthetical indicators - but allow studio edits
        paren_match = re.search(r"\(([^)]+)\)", title_lower)
        if paren_match:
            paren_content = paren_match.group(1)
            # Skip if it's a studio edit indicator
            if re.search(r"\b(single|radio|album)\s*(edit|version)?\b", paren_content):
                return False
            # Check for non-studio indicators
            if any(
                term in paren_content
                for term in [
                    "live",
                    "acoustic",
                    "remix",
                    "demo",
                    "cover",
                    "session",
                    "unplugged",
                    "stripped",
                ]
            ):
                return True

        return False

    def _is_preferred_audio_title(self, title: str) -> bool:
        """Check if a title explicitly indicates an audio-only upload."""
        title_lower = title.lower()
        audio_terms = [
            "official audio",
            "audio only",
            "full audio",
            "audio version",
        ]
        if any(term in title_lower for term in audio_terms):
            return True
        # Catch common suffix/prefix patterns like "Artist - Song (Audio)"
        if re.search(r"\b(audio)\b", title_lower):
            video_terms = [
                "official video",
                "music video",
                "mv",
                "lyric video",
                "lyrics",
                "visualizer",
                "visualiser",
            ]
            if not any(term in title_lower for term in video_terms):
                return True
        return False

    def _search_youtube_verified(
        self,
        query: str,
        target_duration: int,
        expected_artist: Optional[str],
        expected_title: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Search YouTube and return a video that matches artist/title.

        Verifies that the video title contains the expected artist or title
        to avoid covers, tributes, and mismatches.

        Args:
            query: Search query
            target_duration: Target duration in seconds
            expected_artist: Expected artist name to match
            expected_title: Expected song title to match

        Returns:
            Dict with 'url', 'duration', and 'title' keys, or None if not found
        """
        try:
            import requests
        except ImportError:
            return self._search_youtube_by_duration(query, target_duration)

        search_query = query.replace(" ", "+")
        search_url = f"https://www.youtube.com/results?search_query={search_query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            candidates = self._extract_youtube_candidates(response.text)

            if not candidates:
                return None

            # Prepare matching criteria
            artist_words = set(self._normalize_title(expected_artist or "").split())
            title_words = set(self._normalize_title(expected_title or "").split())

            # Filter and score candidates
            scored = []
            tolerance = (
                max(20, int(target_duration * 0.15)) if target_duration > 0 else 30
            )

            for c in candidates:
                if c["duration"] is None:
                    continue

                duration_diff = abs(c["duration"] - target_duration)
                if duration_diff > tolerance:
                    continue

                # Skip non-studio versions
                if self._is_likely_non_studio(c["title"]):
                    logger.debug(f"Skipping non-studio: {c['title']}")
                    continue

                # Score by how well the title matches expected artist/title
                video_title_normalized = self._normalize_title(c["title"])
                video_words = set(video_title_normalized.split())

                artist_match = len(artist_words & video_words) if artist_words else 0
                title_match = len(title_words & video_words) if title_words else 0
                total_match = artist_match + title_match

                # Require at least some match
                if total_match == 0:
                    logger.debug(f"Skipping no-match: {c['title']}")
                    continue

                # Score: higher match is better, lower duration diff is better
                score = (total_match * 10) - duration_diff
                if self._is_preferred_audio_title(c["title"]):
                    score += 5
                scored.append((score, c))
                logger.debug(
                    f"YouTube candidate: {c['title']} (match={total_match}, diff={duration_diff}s, score={score})"
                )

            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                best = scored[0][1]
                logger.info(
                    f"Found: https://www.youtube.com/watch?v={best['video_id']}"
                )
                return {
                    "url": f"https://www.youtube.com/watch?v={best['video_id']}",
                    "duration": best["duration"],
                    "title": best["title"],
                }

        except Exception as e:
            logger.warning(f"YouTube verified search failed: {e}")

        return None

    def _search_youtube_by_duration(
        self, query: str, target_duration: int
    ) -> Optional[Dict[str, Any]]:
        """Search YouTube and return the video with closest duration match.

        Uses a two-stage approach: first searches without modification,
        then falls back to adding "audio" if no good duration match is found.

        Args:
            query: Search query
            target_duration: Target duration in seconds

        Returns:
            Dict with 'url' and 'duration' keys, or None if not found
        """
        # First try without "lyrics"
        result = self._search_youtube_single(query, target_duration)

        if result:
            # Check if this is a good duration match
            if target_duration > 0 and result["duration"]:
                tolerance = max(20, int(target_duration * 0.15))
                diff = abs(result["duration"] - target_duration)
                if diff <= tolerance:
                    return result

                # Not a good match - try with "audio" as fallback
                logger.debug(
                    f"Initial search found video with duration diff={diff}s, trying 'audio' search"
                )
                audio_result = self._search_youtube_single(
                    f"{query} audio", target_duration
                )

                if audio_result and audio_result["duration"]:
                    audio_diff = abs(audio_result["duration"] - target_duration)
                    if audio_diff < diff:
                        logger.debug(
                            f"'audio' search found better match: diff={audio_diff}s vs {diff}s"
                        )
                        return audio_result

                # Original was still better (or audio search found nothing)
                return result
            else:
                return result

        # First search found nothing, try with "audio"
        logger.debug("Initial search found no results, trying 'audio' search")
        return self._search_youtube_single(f"{query} audio", target_duration)

    def _search_youtube_single(
        self, query: str, target_duration: int
    ) -> Optional[Dict[str, Any]]:
        """Execute a single YouTube search and return the best match.

        Args:
            query: Search query (used as-is)
            target_duration: Target duration in seconds

        Returns:
            Dict with 'url' and 'duration' keys, or None if not found
        """
        search_query = query.replace(" ", "+")
        search_url = f"https://www.youtube.com/results?search_query={search_query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        try:
            candidates = self._fetch_youtube_candidates(
                search_url=search_url, headers=headers
            )
            if not candidates:
                return None

            query_wants_non_studio = self._query_wants_non_studio(query)
            with_duration = [c for c in candidates if c["duration"] is not None]

            if not with_duration:
                return self._pick_first_candidate(candidates, query_wants_non_studio)

            if not query_wants_non_studio:
                with_duration = self._filter_studio_candidates(with_duration)

            tolerance = self._youtube_duration_tolerance(target_duration)
            return self._pick_best_duration_candidate(
                with_duration, target_duration, tolerance
            )

        except Exception as e:
            logger.warning(f"YouTube search failed: {e}")
            return None

    def _fetch_youtube_candidates(
        self, search_url: str, headers: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        try:
            import requests
        except ImportError:
            raise Y2KaraokeError("requests required for YouTube search")

        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        return self._extract_youtube_candidates(response.text)

    def _query_wants_non_studio(self, query: str) -> bool:
        query_lower = query.lower()
        return any(
            term in query_lower
            for term in ["live", "concert", "acoustic", "remix", "cover", "karaoke"]
        )

    def _pick_first_candidate(
        self, candidates: List[Dict[str, Any]], query_wants_non_studio: bool
    ) -> Optional[Dict[str, Any]]:
        if not query_wants_non_studio:
            filtered = [
                c for c in candidates if not self._is_likely_non_studio(c["title"])
            ]
            if filtered:
                candidates = filtered

        if not candidates:
            return None

        return {
            "url": f"https://www.youtube.com/watch?v={candidates[0]['video_id']}",
            "duration": None,
        }

    def _filter_studio_candidates(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        studio_candidates = [
            c for c in candidates if not self._is_likely_non_studio(c["title"])
        ]
        if studio_candidates:
            logger.debug(
                f"Filtered to {len(studio_candidates)} studio version candidates"
            )
            return studio_candidates
        return candidates

    def _youtube_duration_tolerance(self, target_duration: int) -> int:
        return max(20, int(target_duration * 0.15)) if target_duration > 0 else 30

    def _pick_best_duration_candidate(
        self,
        with_duration: List[Dict[str, Any]],
        target_duration: int,
        tolerance: int,
    ) -> Optional[Dict[str, Any]]:
        with_duration.sort(key=lambda c: abs(c["duration"] - target_duration))
        scored_candidates = self._score_duration_candidates(
            with_duration, target_duration, tolerance
        )
        if scored_candidates:
            scored_candidates.sort(key=lambda item: item[0])
            best = scored_candidates[0][1]
            return {
                "url": f"https://www.youtube.com/watch?v={best['video_id']}",
                "duration": best["duration"],
            }

        if with_duration:
            best = with_duration[0]
            diff = abs(best["duration"] - target_duration) if target_duration else 0
            logger.warning(
                f"No YouTube video within {tolerance}s of target ({target_duration}s). "
                f"Best match: '{best['title']}' ({best['duration']}s, diff={diff}s)"
            )
            return {
                "url": f"https://www.youtube.com/watch?v={best['video_id']}",
                "duration": best["duration"],
            }

        return None

    def _score_duration_candidates(
        self,
        with_duration: List[Dict[str, Any]],
        target_duration: int,
        tolerance: int,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        scored_candidates = []
        for candidate in with_duration:
            diff = abs(candidate["duration"] - target_duration)
            logger.debug(
                f"YouTube candidate: '{candidate['title']}' duration={candidate['duration']}s, diff={diff}s"
            )

            if diff <= tolerance:
                audio_bonus = (
                    5 if self._is_preferred_audio_title(candidate["title"]) else 0
                )
                scored_candidates.append((diff - audio_bonus, candidate))

        return scored_candidates

    def _extract_youtube_candidates(self, response_text: str) -> List[Dict]:
        """Extract video candidates from YouTube response."""
        candidates = []

        video_pattern = re.compile(
            r'"videoRenderer":\{"videoId":"([^"]{11})".{0,800}?"title":\{"runs":\[\{"text":"([^"]+)"',
            re.DOTALL,
        )

        for match in video_pattern.finditer(response_text):
            video_id = match.group(1)
            title = match.group(2)

            duration_pattern = rf'"videoRenderer":\{{"videoId":"{video_id}".{{0,2000}}?"simpleText":"(\d+:\d+(?::\d+)?)"'
            duration_match = re.search(duration_pattern, response_text, re.DOTALL)

            duration_sec = None
            if duration_match:
                time_str = duration_match.group(1)
                parts = time_str.split(":")
                if len(parts) == 2:
                    duration_sec = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    duration_sec = (
                        int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    )

            candidates.append(
                {"video_id": video_id, "title": title, "duration": duration_sec}
            )

        return candidates

    def _get_youtube_metadata(self, url: str) -> tuple[str, str, int]:
        """Get YouTube video metadata without downloading.

        Returns:
            Tuple of (title, uploader, duration_seconds)
        """
        try:
            import yt_dlp

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "skip_download": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                title = info.get("title", "Unknown")
                uploader = info.get("uploader", info.get("channel", "Unknown"))
                duration = int(info.get("duration", 0))

                return title, uploader, duration

        except Exception as e:
            logger.error(f"Failed to get YouTube metadata: {e}")
            raise Y2KaraokeError(f"Failed to get YouTube metadata: {e}")

    def _fallback_youtube_search(self, query: str) -> TrackInfo:
        """Fallback when MusicBrainz fails: just search YouTube."""
        youtube_result = self._search_youtube_by_duration(query, target_duration=0)

        if not youtube_result:
            raise Y2KaraokeError(f"No YouTube results found for: {query}")

        # Parse query for artist/title
        artist_hint, title_hint = self._parse_query(query)

        return TrackInfo(
            artist=artist_hint or "Unknown",
            title=title_hint,
            duration=youtube_result["duration"] or 0,
            youtube_url=youtube_result["url"],
            youtube_duration=youtube_result["duration"] or 0,
            source="youtube",
        )
