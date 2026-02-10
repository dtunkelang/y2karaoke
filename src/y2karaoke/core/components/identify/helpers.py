"""Helper mixin for TrackIdentifier implementation."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import musicbrainzngs

from ....utils.logging import get_logger
from ...models import TrackInfo
from ...text_utils import normalize_title

logger = get_logger(__name__)


class TrackIdentifierHelpers:
    """Shared helper methods for track identification flows."""

    _lrc_cache: Dict[tuple, tuple]

    def _extract_lrc_metadata(
        self, lrc_text: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract artist and title from LRC metadata tags."""
        artist = None
        title = None
        for line in lrc_text.split("\n")[:20]:
            line = line.strip()
            match = re.match(r"\[ar:(.+)\]", line, re.IGNORECASE)
            if match:
                artist = match.group(1).strip()
                continue
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
        remaining = query.lower()
        for word in title.lower().split():
            remaining = remaining.replace(word, "", 1)
        remaining = " ".join(remaining.split())
        if remaining and len(remaining) > 2:
            return remaining.title()
        return None

    def _lookup_musicbrainz_for_query(
        self, query: str, expected_duration: int
    ) -> tuple[Optional[str], Optional[str]]:
        """Look up MusicBrainz to identify artist/title from a query."""
        query_words = set(normalize_title(query).split())
        search_recordings = getattr(self, "_mb_search_recordings", None)
        if not callable(search_recordings):
            search_recordings = musicbrainzngs.search_recordings
        try:
            results = search_recordings(recording=query, limit=15)
            recordings = results.get("recording-list", [])
            best_match = None
            best_score = 0
            for rec in recordings:
                length = rec.get("length")
                if not length:
                    continue
                duration_sec = int(length) // 1000
                if abs(duration_sec - expected_duration) > 30 or duration_sec > 720:
                    continue
                artist_credits = rec.get("artist-credit", [])
                artists = [a["artist"]["name"] for a in artist_credits if "artist" in a]
                artist_name = " & ".join(artists) if artists else None
                title = rec.get("title")
                if not artist_name or not title:
                    continue
                result_text = f"{artist_name} {title}".lower()
                result_words = set(normalize_title(result_text).split())
                score = len(query_words & result_words)
                if score >= 2 and score > best_score:
                    best_score = score
                    best_match = (artist_name, title)
                    logger.debug(
                        "MusicBrainz candidate: %s - %s (score=%s)",
                        artist_name,
                        title,
                        score,
                    )
            if best_match:
                logger.debug(
                    "MusicBrainz identified: %s - %s", best_match[0], best_match[1]
                )
                return best_match
        except Exception as e:
            logger.debug(f"MusicBrainz lookup failed: {e}")
        return None, None

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

        if (
            split_artist.lower() in artist.lower()
            or artist.lower() in split_artist.lower()
        ):
            score += 20

        artist_words = set(w for w in artist.lower().split() if len(w) > 2)
        if any(w in query_lower for w in artist_words):
            score += 15
        else:
            score -= 30

        split_title_norm = normalize_title(split_title, remove_stopwords=True)
        result_title_norm = normalize_title(title, remove_stopwords=True)
        title_words = set(normalize_title(title).split())
        split_title_words = set(normalize_title(split_title).split())
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

        result_words = set(f"{artist} {title}".lower().split())
        score += len(query_words & result_words) * 3

        if any(
            ind in artist.lower()
            for ind in ["rhapsody", "symphony", "concerto", "song", "ballad"]
        ):
            score -= 15
        return score

    def _try_split_search(self, query: str) -> Optional[tuple]:
        """Try artist/title splits to find best candidate."""
        splits = self._try_artist_title_splits(query)  # type: ignore[attr-defined]
        split_candidates = []
        for split_artist, split_title in splits:
            logger.debug(
                f"Trying split: artist='{split_artist}', title='{split_title}'"
            )
            split_recordings = self._query_musicbrainz(  # type: ignore[attr-defined]
                f"{split_artist} {split_title}", split_artist, split_title
            )
            if not split_recordings:
                continue
            candidate = self._find_best_with_artist_hint(  # type: ignore[attr-defined]
                split_recordings, query, split_artist
            )
            if not candidate:
                continue
            score = self._score_split_candidate(
                candidate, split_artist, split_title, query
            )
            if score is None:
                continue
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
        for candidate in candidates:
            key = (candidate["artist"].lower(), candidate["title"].lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _search_matching_youtube_video(
        self, artist: str, title: str, lrc_duration: int, yt_duration: int
    ) -> Optional[TrackInfo]:
        """Search for a YouTube video matching LRC duration."""
        logger.info(
            f"LRC duration ({lrc_duration}s) differs from YouTube ({yt_duration}s)"
        )
        logger.info("Searching for YouTube video matching LRC duration...")
        for search_query in [
            f"{artist} {title} official audio",
            f"{artist} {title} audio",
            f"{artist} {title}",
        ]:
            alt_youtube = self._search_youtube_verified(  # type: ignore[attr-defined]
                search_query, lrc_duration, artist, title
            )
            if alt_youtube and alt_youtube["duration"]:
                if abs(alt_youtube["duration"] - lrc_duration) <= 8:
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
            alt_youtube = self._search_youtube_verified(  # type: ignore[attr-defined]
                f"{artist} {title} radio edit", lrc_duration, artist, title
            )
            if (
                alt_youtube
                and alt_youtube["duration"]
                and abs(alt_youtube["duration"] - lrc_duration) <= 8
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
            for candidate in unique_candidates:
                if candidate["artist"] != yt_uploader:
                    fallback_artist = candidate["artist"]
                    fallback_title = candidate["title"]
                    break
        if not fallback_artist:
            fallback_artist = parsed_artist
            fallback_title = parsed_title
        if not fallback_artist and parsed_title:
            splits = self._try_artist_title_splits(parsed_title)  # type: ignore[attr-defined]
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
        final_artist: str = fallback_artist if fallback_artist else "Unknown"
        final_title: str = fallback_title if fallback_title else yt_title
        return final_artist, final_title

    def _check_lrc_and_duration(
        self, title: str, artist: str, expected_duration: Optional[int] = None
    ) -> tuple[bool, Optional[int]]:
        """Check if synced LRC lyrics are available and valid, and get duration."""
        cache_key = (artist.lower(), title.lower())
        if cache_key in self._lrc_cache:
            return self._lrc_cache[cache_key]
        try:
            from ...sync import (
                SYNCEDLYRICS_AVAILABLE,
                fetch_lyrics_multi_source,
                get_lrc_duration,
                validate_lrc_quality,
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
            is_valid, reason = validate_lrc_quality(lrc_text, expected_duration)
            if not is_valid:
                logger.debug(f"LRC for {artist} - {title} failed validation: {reason}")
                invalid: Tuple[bool, Optional[int]] = (False, None)
                self._lrc_cache[cache_key] = invalid
                return invalid
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
        """Find candidate whose LRC duration and title best match."""
        scored_matches = []
        fallback_match = None
        title_hint_words = set(
            normalize_title(title_hint, remove_stopwords=True).split()
        )
        for candidate in candidates:
            artist = candidate["artist"]
            title = candidate["title"]
            lrc_available, lrc_duration = self._check_lrc_and_duration(
                title, artist, expected_duration=target_duration
            )
            if not lrc_available:
                continue
            if lrc_duration is None:
                if fallback_match is None:
                    fallback_match = (artist, title, target_duration)
                continue

            duration_diff = abs(lrc_duration - target_duration)
            candidate_title_normalized = normalize_title(title, remove_stopwords=True)
            candidate_title_words = set(candidate_title_normalized.split())
            if title_hint_words:
                word_overlap = len(title_hint_words & candidate_title_words) / len(
                    title_hint_words
                )
            else:
                word_overlap = 0
            title_hint_norm = normalize_title(title_hint, remove_stopwords=False)
            seq_similarity = SequenceMatcher(
                None, title_hint_norm, candidate_title_normalized
            ).ratio()
            title_score = (word_overlap * 0.6) + (seq_similarity * 0.4)
            duration_score = max(0, 1 - (duration_diff / 60))
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

        scored_matches.sort(key=lambda x: (-x[0], x[1]))
        combined_score, duration_diff, artist, title, lrc_duration = scored_matches[0]
        if combined_score < 0.3:
            logger.warning(
                f"Best LRC match has low title similarity ({combined_score:.2f})"
            )
        if duration_diff > tolerance:
            logger.warning(
                f"Best LRC match has {duration_diff}s duration difference (tolerance: {tolerance}s)"
            )
        return (artist, title, lrc_duration)
