"""YouTube search and metadata extraction logic."""

import json
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from ....utils.logging import get_logger
from ....exceptions import Y2KaraokeError
from ...models import TrackInfo
from ...text_utils import normalize_title
from . import youtube_rules as yt_rules

if TYPE_CHECKING:
    from .parser import QueryParser

    _Base = QueryParser
else:
    _Base = object

logger = get_logger(__name__)


class YouTubeSearcher(_Base):
    """Handles searching YouTube and extracting track metadata."""

    def _normalize_title(self, title: str, remove_stopwords: bool = False) -> str:
        """Delegate to text_utils.normalize_title."""
        return normalize_title(title, remove_stopwords=remove_stopwords)

    def _is_likely_non_studio(self, title: str) -> bool:
        return yt_rules.is_likely_non_studio(title)

    def _is_preferred_audio_title(self, title: str) -> bool:
        return yt_rules.is_preferred_audio_title(title)

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

        except Y2KaraokeError:
            return self._search_youtube_by_duration(query, target_duration)
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

        http_get = getattr(self, "_http_get", None)
        if not callable(http_get):
            http_get = requests.get
        response = http_get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        return self._extract_youtube_candidates(response.text)

    def _query_wants_non_studio(self, query: str) -> bool:
        return yt_rules.query_wants_non_studio(query)

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
        return yt_rules.youtube_duration_tolerance(target_duration)

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
        return yt_rules.extract_youtube_candidates(response_text)

    def _load_yt_dlp_module(self):
        try:
            import yt_dlp

            return yt_dlp
        except ImportError as e:
            raise Y2KaraokeError("yt_dlp required for YouTube metadata") from e

    def _get_youtube_metadata(self, url: str) -> tuple[str, str, int]:
        """Get YouTube video metadata without downloading.

        Returns:
            Tuple of (title, uploader, duration_seconds)
        """
        try:
            yt_dlp = self._load_yt_dlp_module()

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

            from ....config import get_cache_dir
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
        except Exception as e:
            logger.debug(f"Cached YouTube metadata lookup failed: {e}")
            return None

    def get_cached_youtube_metadata(self, url: str) -> Optional[tuple[str, str, int]]:
        """Public wrapper for cached metadata lookup (offline support)."""
        cached = self._get_cached_youtube_metadata(url)
        if cached is not None:
            return cached
        try:
            from ....config import get_cache_dir
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
        except Exception as e:
            logger.debug(f"Public cached metadata lookup failed: {e}")
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
