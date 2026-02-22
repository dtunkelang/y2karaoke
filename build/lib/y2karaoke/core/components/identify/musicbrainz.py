"""MusicBrainz lookup and scoring logic."""

import re
from collections import Counter
from typing import Optional, List, Dict, TYPE_CHECKING
import musicbrainzngs
from ....utils.logging import get_logger
from ...text_utils import normalize_title

if TYPE_CHECKING:
    from typing import Protocol

    class _TrackIdentifierMixin(Protocol):
        def _is_likely_non_studio(self, title: str) -> bool:
            pass

        def _check_lrc_and_duration(
            self, title: str, artist: str, expected_duration: Optional[int] = None
        ) -> tuple[bool, Optional[int]]:
            pass

    _Base = _TrackIdentifierMixin
else:
    _Base = object

logger = get_logger(__name__)


# Initialize MusicBrainz
musicbrainzngs.set_useragent(
    "y2karaoke", "1.0", "https://github.com/dtunkelang/y2karaoke"
)


class MusicBrainzClient(_Base):
    """Handles querying MusicBrainz and scoring track matches."""

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

        search_recordings = getattr(self, "_mb_search_recordings", None)
        if not callable(search_recordings):
            search_recordings = musicbrainzngs.search_recordings
        sleep_fn = getattr(self, "_sleep", None)
        if not callable(sleep_fn):
            sleep_fn = time.sleep

        for attempt in range(max_retries + 1):
            try:
                # Include release info to check release type
                if artist_hint:
                    results = search_recordings(
                        recording=title_hint, artist=artist_hint, limit=25
                    )
                else:
                    results = search_recordings(recording=query, limit=25)

                recordings = results.get("recording-list", [])

                # Score and sort recordings to prioritize studio versions and title matches
                scored = []
                score_recording = getattr(
                    self, "_score_recording_studio_likelihood_fn", None
                )
                if not callable(score_recording):
                    score_recording = self._score_recording_studio_likelihood
                for rec in recordings:
                    score = score_recording(rec)

                    # Bonus for title match (when user explicitly provides title)
                    if title_hint:
                        rec_title = rec.get("title", "")

                        # First check exact match (with stopwords retained)
                        title_hint_norm = normalize_title(
                            title_hint, remove_stopwords=False
                        )
                        rec_title_norm = normalize_title(
                            rec_title, remove_stopwords=False
                        )

                        if rec_title_norm == title_hint_norm:
                            score += 100  # Strong bonus for exact match - should outweigh album release bonus
                        else:
                            # Check match with stopwords removed (looser matching)
                            title_hint_no_stop = normalize_title(
                                title_hint, remove_stopwords=True
                            )
                            rec_title_no_stop = normalize_title(
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
                    sleep_fn(delay)
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
        artist_hint_lower = normalize_title(artist_hint)
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
            artist_str_lower = normalize_title(" ".join(artists))
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
        title_normalized = normalize_title(title_hint)
        title_normalized_no_stop = normalize_title(title_hint, remove_stopwords=True)
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

            rec_title_normalized = normalize_title(title)
            rec_title_no_stop = normalize_title(title, remove_stopwords=True)

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
        """Delegate to text_utils.normalize_title."""
        return normalize_title(title, remove_stopwords=remove_stopwords)

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
