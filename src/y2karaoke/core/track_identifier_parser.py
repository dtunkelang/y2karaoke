"""Query and YouTube title parsing logic."""

import re
import musicbrainzngs
from typing import Optional, List, Dict
from ..utils.logging import get_logger
from .text_utils import normalize_title

logger = get_logger(__name__)


class QueryParser:
    """Handles parsing track metadata from queries and titles."""

    def _normalize_title(self, title: str, remove_stopwords: bool = False) -> str:
        """Delegate to text_utils.normalize_title."""
        return normalize_title(title, remove_stopwords=remove_stopwords)

    def _query_musicbrainz(
        self,
        query: str,
        artist_hint: Optional[str],
        title_hint: str,
        max_retries: int = 3,
    ) -> List[Dict]:
        """Delegate to MusicBrainzClient (implemented via multiple inheritance)."""
        raise NotImplementedError("Subclasses must implement _query_musicbrainz")
    
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

