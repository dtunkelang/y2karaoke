"""Track identification pipeline for y2karaoke.

This module handles two distinct paths for identifying track information:
- Path A (search string): Query -> MusicBrainz/syncedlyrics -> canonical track -> YouTube
- Path B (YouTube URL): URL -> YouTube duration -> best LRC match by duration
"""

import re
from dataclasses import dataclass
from collections import Counter
from typing import Optional, List, Dict, Any

import musicbrainzngs

from ..utils.logging import get_logger
from ..exceptions import Y2KaraokeError

logger = get_logger(__name__)

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


class TrackIdentifier:
    """Identifies track information from search queries or YouTube URLs."""

    def __init__(self):
        self._lrc_cache: Dict[tuple, tuple] = {}

    # -------------------------
    # Path A: Search String -> Track
    # -------------------------

    def identify_from_search(self, query: str) -> TrackInfo:
        """Identify track from a search string (not a URL).

        Flow:
        1. Parse query for artist/title hints
        2. Query MusicBrainz for candidate recordings
        3. Check LRC availability and duration for candidates
        4. Filter out live/remix versions
        5. Search YouTube for videos matching canonical duration

        Args:
            query: Search string like "Artist - Title" or just "Title"

        Returns:
            TrackInfo with canonical artist, title, duration, and YouTube URL
        """
        logger.info(f"Identifying track from search: {query}")

        # Parse query for artist/title hints
        artist_hint, title_hint = self._parse_query(query)
        logger.debug(f"Parsed hints: artist='{artist_hint}', title='{title_hint}'")

        # Query MusicBrainz
        recordings = self._query_musicbrainz(query, artist_hint, title_hint)

        if not recordings:
            logger.warning("No MusicBrainz results, falling back to YouTube search")
            return self._fallback_youtube_search(query)

        # Find best candidate with LRC preference
        if artist_hint:
            best = self._find_best_with_artist_hint(recordings, query, artist_hint)
        else:
            best = self._find_best_title_only(recordings, title_hint)

        if not best:
            logger.warning("No suitable MusicBrainz candidate, falling back to YouTube")
            return self._fallback_youtube_search(query)

        canonical_duration, canonical_artist, canonical_title = best
        logger.info(f"Canonical track: {canonical_artist} - {canonical_title} ({canonical_duration}s)")

        # Validate LRC duration matches canonical duration
        from .sync import fetch_lyrics_for_duration
        _, _, _, lrc_duration = fetch_lyrics_for_duration(
            canonical_title, canonical_artist, canonical_duration, tolerance=20
        )
        lrc_validated = lrc_duration is not None and abs(lrc_duration - canonical_duration) <= 20

        if lrc_duration and not lrc_validated:
            logger.warning(f"LRC duration ({lrc_duration}s) doesn't match canonical ({canonical_duration}s) - lyrics timing may be off")

        # Search YouTube with duration matching
        youtube_result = self._search_youtube_by_duration(
            f"{canonical_artist} {canonical_title}",
            canonical_duration
        )

        if not youtube_result:
            # Try with original query
            youtube_result = self._search_youtube_by_duration(query, canonical_duration)

        if not youtube_result:
            raise Y2KaraokeError(f"No YouTube results found for: {query}")

        return TrackInfo(
            artist=canonical_artist,
            title=canonical_title,
            duration=canonical_duration,
            youtube_url=youtube_result['url'],
            youtube_duration=youtube_result['duration'],
            source="musicbrainz",
            lrc_duration=lrc_duration,
            lrc_validated=lrc_validated
        )

    # -------------------------
    # Path B: YouTube URL -> Track
    # -------------------------

    def identify_from_url(self, url: str) -> TrackInfo:
        """Identify track from a YouTube URL.

        Flow:
        1. Get YouTube video metadata (title, uploader, duration)
        2. Parse video title for artist/title hints
        3. Query MusicBrainz for candidates
        4. Check LRC for each candidate, score by duration match
        5. Return candidate with best duration match

        Args:
            url: YouTube URL

        Returns:
            TrackInfo with canonical artist, title, and the given YouTube URL
        """
        logger.info(f"Identifying track from URL: {url}")

        # Get YouTube metadata
        yt_title, yt_uploader, yt_duration = self._get_youtube_metadata(url)
        logger.info(f"YouTube: '{yt_title}' by {yt_uploader} ({yt_duration}s)")

        # Parse video title for hints
        parsed_artist, parsed_title = self._parse_youtube_title(yt_title)
        logger.debug(f"Parsed from title: artist='{parsed_artist}', title='{parsed_title}'")

        # Query MusicBrainz for candidates
        search_query = f"{parsed_artist} {parsed_title}" if parsed_artist else parsed_title
        recordings = self._query_musicbrainz(search_query, parsed_artist, parsed_title)

        # Build candidate list with (artist, title) pairs
        candidates = []
        for rec in recordings:
            artist_credits = rec.get('artist-credit', [])
            artists = [a['artist']['name'] for a in artist_credits if 'artist' in a]
            artist_name = ' & '.join(artists) if artists else None
            title = rec.get('title')
            if artist_name and title:
                candidates.append({'artist': artist_name, 'title': title})

        # Add parsed hint as fallback candidate
        if parsed_artist and parsed_title:
            candidates.append({'artist': parsed_artist, 'title': parsed_title})
        elif yt_uploader and parsed_title:
            candidates.append({'artist': yt_uploader, 'title': parsed_title})

        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = (c['artist'].lower(), c['title'].lower())
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        logger.debug(f"Found {len(unique_candidates)} unique candidates to check")

        # Find best LRC match by duration
        best_match = self._find_best_lrc_by_duration(unique_candidates, yt_duration)

        if best_match:
            artist, title, lrc_duration = best_match
            lrc_validated = abs(lrc_duration - yt_duration) <= 15
            logger.info(f"Best LRC match: {artist} - {title} (LRC duration: {lrc_duration}s, validated: {lrc_validated})")
            return TrackInfo(
                artist=artist,
                title=title,
                duration=lrc_duration,
                youtube_url=url,
                youtube_duration=yt_duration,
                source="syncedlyrics",
                lrc_duration=lrc_duration,
                lrc_validated=lrc_validated
            )

        # Fallback: use parsed info without LRC validation
        fallback_artist = parsed_artist or yt_uploader or "Unknown"
        fallback_title = parsed_title or yt_title
        logger.warning(f"No LRC match found, using: {fallback_artist} - {fallback_title}")

        return TrackInfo(
            artist=fallback_artist,
            title=fallback_title,
            duration=yt_duration,
            youtube_url=url,
            youtube_duration=yt_duration,
            source="youtube"
        )

    # -------------------------
    # Helper Methods
    # -------------------------

    def _parse_query(self, query: str) -> tuple[Optional[str], str]:
        """Parse a search query for artist/title hints."""
        query_clean = query.strip()

        if ' - ' in query_clean:
            parts = query_clean.split(' - ', 1)
            return parts[0].strip(), parts[1].strip()

        return None, query_clean

    def _parse_youtube_title(self, title: str) -> tuple[Optional[str], str]:
        """Parse a YouTube video title for artist/title."""
        # Remove common suffixes
        cleaned = title
        for suffix in ["Official Music Video", "Official Video", "Official Audio",
                       "Lyric Video", "Lyrics", "HD", "4K", "MV", "(Official)"]:
            cleaned = re.sub(rf'\s*[\(\[]?\s*{re.escape(suffix)}\s*[\)\]]?\s*$', '', cleaned, flags=re.IGNORECASE)

        # Remove parenthetical/bracket content
        cleaned = re.sub(r'\s*[\(\[].*?[\)\]]\s*', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Try to split on " - " or " – " or " — "
        for sep in [' - ', ' – ', ' — ', ' | ']:
            if sep in cleaned:
                parts = cleaned.split(sep, 1)
                return parts[0].strip(), parts[1].strip()

        return None, cleaned

    def _query_musicbrainz(self, query: str, artist_hint: Optional[str], title_hint: str) -> List[Dict]:
        """Query MusicBrainz for recordings matching the query."""
        try:
            if artist_hint:
                results = musicbrainzngs.search_recordings(
                    recording=title_hint, artist=artist_hint, limit=15
                )
            else:
                results = musicbrainzngs.search_recordings(recording=query, limit=15)

            return results.get('recording-list', [])

        except Exception as e:
            logger.warning(f"MusicBrainz search failed: {e}")
            return []

    def _find_best_with_artist_hint(self, recordings: List[Dict], query: str, artist_hint: str) -> Optional[tuple]:
        """Find best recording when artist hint is available."""
        query_words = set(query.lower().split())
        matches = []

        for rec in recordings:
            length = rec.get('length')
            if not length:
                continue

            artist_credits = rec.get('artist-credit', [])
            artists = [a['artist']['name'] for a in artist_credits if 'artist' in a]
            artist_str_lower = ' '.join(a.lower() for a in artists)

            if any(word in artist_str_lower for word in query_words):
                artist_name = ' & '.join(artists) if artists else None
                title = rec.get('title')
                matches.append((int(length) // 1000, artist_name, title))

        if not matches:
            return None

        return self._score_and_select_best(matches)

    def _find_best_title_only(self, recordings: List[Dict], title_hint: str) -> Optional[tuple]:
        """Find best recording for title-only searches using consensus and LRC preference."""
        title_normalized = self._normalize_title(title_hint)
        candidates = []

        for rec in recordings:
            title = rec.get('title', '')
            length = rec.get('length')
            if not length:
                continue

            if self._normalize_title(title) != title_normalized:
                continue

            artist_credits = rec.get('artist-credit', [])
            artists = [a['artist']['name'] for a in artist_credits if 'artist' in a]
            artist_name = ' & '.join(artists) if artists else None

            if artist_name:
                candidates.append({
                    'duration': int(length) // 1000,
                    'artist': artist_name,
                    'title': title,
                })

        if not candidates:
            return None

        # Find unique artists sorted by frequency
        artist_counts = Counter(c['artist'] for c in candidates)
        all_artists = artist_counts.most_common()

        # Calculate MusicBrainz duration consensus (most common duration cluster)
        all_durations = [c['duration'] for c in candidates]
        rounded_durations = [d // 5 * 5 for d in all_durations]
        mb_consensus_duration = Counter(rounded_durations).most_common(1)[0][0] if rounded_durations else None

        # First pass: find artist with LRC available, but validate against MB consensus
        for artist_name, count in all_artists:
            artist_matches = [c for c in candidates if c['artist'] == artist_name]
            sample_title = artist_matches[0]['title']

            lrc_available, lrc_duration = self._check_lrc_and_duration(sample_title, artist_name)
            if lrc_available:
                # Check if LRC duration is reasonable compared to MusicBrainz consensus
                if lrc_duration and mb_consensus_duration:
                    lrc_mb_diff = abs(lrc_duration - mb_consensus_duration)
                    # If LRC duration differs significantly from MB consensus (>30s),
                    # prefer MB consensus - the LRC might be for a different version
                    if lrc_mb_diff > 30:
                        logger.debug(f"LRC duration ({lrc_duration}s) differs from MB consensus ({mb_consensus_duration}s) by {lrc_mb_diff}s - using MB consensus")
                        # Sort by proximity to MB consensus instead
                        artist_matches.sort(key=lambda c: abs(c['duration'] - mb_consensus_duration))
                    else:
                        # LRC duration is reasonable, use it
                        artist_matches.sort(key=lambda c: abs(c['duration'] - lrc_duration))
                elif lrc_duration:
                    artist_matches.sort(key=lambda c: abs(c['duration'] - lrc_duration))

                best = artist_matches[0]
                return (best['duration'], best['artist'], best['title'])

        # Fallback: use most common artist with duration consensus
        if len(candidates) >= 2:
            most_common_artist, artist_count = all_artists[0]
            if artist_count >= 2 and artist_count / len(candidates) >= 0.4:
                artist_matches = [c for c in candidates if c['artist'] == most_common_artist]
                durations = [c['duration'] for c in artist_matches]
                rounded = [d // 5 * 5 for d in durations]
                duration_counts = Counter(rounded)
                if duration_counts.most_common(1)[0][1] >= 2:
                    best = artist_matches[0]
                    return (best['duration'], best['artist'], best['title'])

        return None

    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        normalized = re.sub(r'[,.\-:;\'\"!?()]', ' ', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _score_and_select_best(self, matches: List[tuple]) -> Optional[tuple]:
        """Score matches and select the best one, penalizing live/remix versions."""
        if not matches:
            return None

        def score_match(m):
            duration, artist, title = m
            score = 0

            # Penalize parenthetical suffixes (live, remix, demo, etc.)
            paren_match = re.search(r'\([^)]+\)\s*$', title)
            if paren_match:
                paren_content = paren_match.group().lower()
                if any(word in paren_content for word in ['live', 'remix', 'demo', 'acoustic', 'radio', 'edit', 'version']):
                    score -= 100
                else:
                    score -= 50

            if re.search(r'\[[^\]]+\]\s*$', title):
                score -= 50

            return score

        # Find most common duration among clean matches
        clean_matches = [m for m in matches if score_match(m) >= 0]
        if clean_matches:
            durations = [m[0] for m in clean_matches]
        else:
            durations = [m[0] for m in matches]

        rounded = [d // 3 * 3 for d in durations]
        most_common_rounded = Counter(rounded).most_common(1)[0][0]

        def final_score(m):
            title_score = score_match(m)
            duration_diff = abs(m[0] - most_common_rounded)
            return (title_score, -duration_diff)

        return max(matches, key=final_score)

    def _check_lrc_and_duration(self, title: str, artist: str) -> tuple[bool, Optional[int]]:
        """Check if synced LRC lyrics are available and get implied duration."""
        cache_key = (artist.lower(), title.lower())
        if cache_key in self._lrc_cache:
            return self._lrc_cache[cache_key]

        try:
            from .sync import fetch_lyrics_multi_source, SYNCEDLYRICS_AVAILABLE
            from .lrc import parse_lrc_with_timing

            if not SYNCEDLYRICS_AVAILABLE:
                result = (False, None)
                self._lrc_cache[cache_key] = result
                return result

            lrc_text, is_synced, _ = fetch_lyrics_multi_source(title, artist, synced_only=True)
            if not is_synced or not lrc_text:
                result = (False, None)
                self._lrc_cache[cache_key] = result
                return result

            timings = parse_lrc_with_timing(lrc_text, title, artist)
            if timings:
                last_ts = timings[-1][0]
                implied_duration = int(last_ts + 5)
                result = (True, implied_duration)
            else:
                result = (True, None)

            self._lrc_cache[cache_key] = result
            return result

        except Exception as e:
            logger.debug(f"LRC check failed for {artist} - {title}: {e}")
            result = (False, None)
            self._lrc_cache[cache_key] = result
            return result

    def _find_best_lrc_by_duration(
        self,
        candidates: List[Dict],
        target_duration: int,
        tolerance: int = 15
    ) -> Optional[tuple[str, str, int]]:
        """Find the candidate whose LRC duration best matches the target duration.

        Args:
            candidates: List of dicts with 'artist' and 'title' keys
            target_duration: Target duration in seconds (from YouTube)
            tolerance: Maximum acceptable duration difference in seconds

        Returns:
            Tuple of (artist, title, lrc_duration) or None if no match found
        """
        best_match = None
        best_diff = float('inf')

        for candidate in candidates:
            artist = candidate['artist']
            title = candidate['title']

            lrc_available, lrc_duration = self._check_lrc_and_duration(title, artist)

            if not lrc_available:
                continue

            if lrc_duration is None:
                # LRC available but couldn't determine duration
                # Keep as fallback if nothing else works
                if best_match is None:
                    best_match = (artist, title, target_duration)
                continue

            diff = abs(lrc_duration - target_duration)
            logger.debug(f"LRC candidate: {artist} - {title}, duration={lrc_duration}s, diff={diff}s")

            if diff < best_diff:
                best_diff = diff
                best_match = (artist, title, lrc_duration)

        if best_match and best_diff <= tolerance:
            return best_match
        elif best_match:
            logger.warning(f"Best LRC match has {best_diff}s duration difference (tolerance: {tolerance}s)")
            return best_match

        return None

    def _is_likely_non_studio(self, title: str) -> bool:
        """Check if a YouTube title suggests a non-studio version."""
        title_lower = title.lower()

        # Terms that indicate live/alternate versions
        non_studio_terms = [
            'live', 'concert', 'performance', 'session', 'acoustic',
            'remix', 'extended', 'cover', 'karaoke', 'instrumental',
            'demo', 'rehearsal', 'bootleg', 'tribute', 'parody',
            'reaction', 'tutorial', 'lesson', 'how to play',
            'slowed', 'sped up', 'reverb', '8d audio', 'nightcore',
        ]

        # Check for any of these terms
        for term in non_studio_terms:
            if term in title_lower:
                return True

        # Check for common live venue patterns
        live_patterns = [
            r'\bat\b.*\b(show|festival|arena|stadium|hall|theater|theatre|club)\b',
            r'\b(snl|letterman|fallon|kimmel|conan|ellen|tonight show)\b',
            r'\b(glastonbury|coachella|lollapalooza|reading|leeds)\b',
            r'\b(mtv|vma|grammy|brit|award)\b',
            r'\b(unplugged|stripped|sessions?)\b',
        ]
        for pattern in live_patterns:
            if re.search(pattern, title_lower):
                return True

        return False

    def _search_youtube_by_duration(
        self,
        query: str,
        target_duration: int
    ) -> Optional[Dict[str, Any]]:
        """Search YouTube and return the video with closest duration match.

        Args:
            query: Search query
            target_duration: Target duration in seconds

        Returns:
            Dict with 'url' and 'duration' keys, or None if not found
        """
        try:
            import requests
        except ImportError:
            raise Y2KaraokeError("requests required for YouTube search")

        search_query = query.replace(" ", "+")
        search_url = f"https://www.youtube.com/results?search_query={search_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            candidates = self._extract_youtube_candidates(response.text)

            if not candidates:
                return None

            # Check if query explicitly asks for a non-studio version
            query_lower = query.lower()
            query_wants_non_studio = any(term in query_lower for term in
                ['live', 'concert', 'acoustic', 'remix', 'cover', 'karaoke'])

            # Filter candidates with duration info
            with_duration = [c for c in candidates if c['duration'] is not None]

            if not with_duration:
                # No duration info available, just filter out non-studio and take first
                if not query_wants_non_studio:
                    filtered = [c for c in candidates if not self._is_likely_non_studio(c['title'])]
                    if filtered:
                        candidates = filtered
                if candidates:
                    return {
                        'url': f"https://www.youtube.com/watch?v={candidates[0]['video_id']}",
                        'duration': None
                    }
                return None

            # Filter out non-studio versions first (unless query asks for them)
            if not query_wants_non_studio:
                studio_candidates = [c for c in with_duration if not self._is_likely_non_studio(c['title'])]
                if studio_candidates:
                    with_duration = studio_candidates
                    logger.debug(f"Filtered to {len(studio_candidates)} studio version candidates")

            # Use proportional tolerance: max of 20s or 15% of target duration
            tolerance = max(20, int(target_duration * 0.15)) if target_duration > 0 else 30

            # Sort by duration difference from target
            with_duration.sort(key=lambda c: abs(c['duration'] - target_duration))

            for candidate in with_duration:
                diff = abs(candidate['duration'] - target_duration)
                logger.debug(f"YouTube candidate: '{candidate['title']}' duration={candidate['duration']}s, diff={diff}s")

                if diff <= tolerance:
                    return {
                        'url': f"https://www.youtube.com/watch?v={candidate['video_id']}",
                        'duration': candidate['duration']
                    }

            # No good duration match found - take the best available from filtered list
            # but warn about the mismatch
            if with_duration:
                best = with_duration[0]
                diff = abs(best['duration'] - target_duration) if target_duration else 0
                logger.warning(f"No YouTube video within {tolerance}s of target ({target_duration}s). "
                             f"Best match: '{best['title']}' ({best['duration']}s, diff={diff}s)")
                return {
                    'url': f"https://www.youtube.com/watch?v={best['video_id']}",
                    'duration': best['duration']
                }

            return None

        except Exception as e:
            logger.warning(f"YouTube search failed: {e}")
            return None

    def _extract_youtube_candidates(self, response_text: str) -> List[Dict]:
        """Extract video candidates from YouTube response."""
        candidates = []

        video_pattern = re.compile(
            r'"videoRenderer":\{"videoId":"([^"]{11})".{0,800}?"title":\{"runs":\[\{"text":"([^"]+)"',
            re.DOTALL
        )

        for match in video_pattern.finditer(response_text):
            video_id = match.group(1)
            title = match.group(2)

            duration_pattern = rf'"videoRenderer":\{{"videoId":"{video_id}".{{0,2000}}?"simpleText":"(\d+:\d+(?::\d+)?)"'
            duration_match = re.search(duration_pattern, response_text, re.DOTALL)

            duration_sec = None
            if duration_match:
                time_str = duration_match.group(1)
                parts = time_str.split(':')
                if len(parts) == 2:
                    duration_sec = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

            candidates.append({
                'video_id': video_id,
                'title': title,
                'duration': duration_sec
            })

        return candidates

    def _get_youtube_metadata(self, url: str) -> tuple[str, str, int]:
        """Get YouTube video metadata without downloading.

        Returns:
            Tuple of (title, uploader, duration_seconds)
        """
        try:
            import yt_dlp

            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                title = info.get('title', 'Unknown')
                uploader = info.get('uploader', info.get('channel', 'Unknown'))
                duration = int(info.get('duration', 0))

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
            duration=youtube_result['duration'] or 0,
            youtube_url=youtube_result['url'],
            youtube_duration=youtube_result['duration'] or 0,
            source="youtube"
        )
