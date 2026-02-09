"""YouTube search and metadata extraction logic."""

import re
import json
from typing import Optional, List, Dict, Any, Tuple
from ..utils.logging import get_logger
from ..exceptions import Y2KaraokeError
from .models import TrackInfo

logger = get_logger(__name__)


class YouTubeSearcher:
    """Handles searching YouTube and extracting track metadata."""

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
            import requests  # type: ignore[import-untyped]
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

    def _search_youtube_single(  # noqa: C901
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
            import requests  # type: ignore[import-untyped]
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
            logger.warning(f"Failed to get YouTube metadata: {e}")
            cached = self._get_cached_youtube_metadata(url)
            if cached is not None:
                logger.info("Using cached YouTube metadata")
                return cached
            logger.error(f"Failed to get YouTube metadata: {e}")
            raise Y2KaraokeError(f"Failed to get YouTube metadata: {e}")

    def _get_cached_youtube_metadata(self, url: str) -> Optional[tuple[str, str, int]]:
        """Fallback to cached metadata/audio when YouTube is unreachable."""
        try:
            import wave

            from ..config import get_cache_dir
            from .youtube_metadata import extract_video_id

            video_id = extract_video_id(url)
            cache_dir = get_cache_dir() / video_id
            if not cache_dir.exists():
                return None

            title = "Unknown"
            artist = "Unknown"
            metadata_path = cache_dir / "metadata.json"
            if metadata_path.exists():
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                title = data.get("title", title) or title
                artist = data.get("artist", artist) or artist

            duration = 0
            candidates = [
                p
                for p in cache_dir.glob("*.wav")
                if "Vocals" not in p.stem and "instrumental" not in p.stem
            ]
            wav_path = candidates[0] if candidates else None
            if wav_path and wav_path.exists():
                with wave.open(str(wav_path), "rb") as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = int(frames / rate) if rate else 0

            uploader = artist if artist != "Unknown" else "Unknown"
            return title, uploader, duration
        except Exception:
            return None

    def get_cached_youtube_metadata(self, url: str) -> Optional[tuple[str, str, int]]:
        """Public wrapper for cached metadata lookup (offline support)."""
        cached = self._get_cached_youtube_metadata(url)
        if cached is not None:
            return cached
        try:
            from ..config import get_cache_dir
            from .youtube_metadata import extract_video_id

            video_id = extract_video_id(url)
            cache_dir = get_cache_dir() / video_id
            if not cache_dir.exists():
                return None

            candidates = [
                p
                for p in cache_dir.glob("*.wav")
                if "Vocals" not in p.stem and "instrumental" not in p.stem
            ]
            wav_path = candidates[0] if candidates else None
            if not wav_path:
                return None
            return "Unknown", "Unknown", 0
        except Exception:
            return None

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
