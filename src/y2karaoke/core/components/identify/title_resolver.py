"""Resolve canonical artist/title for a song, robust to YouTube titles and MusicBrainz results."""

import re
from difflib import SequenceMatcher
import logging
import musicbrainzngs
from typing import Optional

logger = logging.getLogger(__name__)

STOP_WORDS = {"the", "a", "an", "&", "and", "of", "with", "in", "+"}

# Initialize MusicBrainz once
musicbrainzngs.set_useragent(
    "y2karaoke", "1.0", "https://github.com/dtunkelang/y2karaoke"
)


def normalize_string(s: str) -> str:
    """Normalize string for matching: lowercase, remove non-alphanum, strip stop words."""
    s = s.casefold()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    tokens = [t for t in s.split() if t not in STOP_WORDS]
    return " ".join(tokens).strip()


def _strip_parentheses(title: str) -> str:
    """Remove parenthetical expressions from a title for canonical matching."""
    return re.sub(r"\s*[\(\[].*?[\)\]]\s*", "", title).strip()


def guess_musicbrainz_candidates(
    prompt: str, fallback_artist: str = "", fallback_title: str = ""
) -> list[dict]:
    """Query MusicBrainz and return a list of candidate (artist, title) pairs with scores."""
    from musicbrainzngs import search_recordings

    logger.debug(f"Querying MusicBrainz for prompt: '{prompt}'")
    candidates = []

    try:
        results = search_recordings(recording=prompt, limit=15).get(
            "recording-list", []
        )
    except Exception as e:
        logger.warning(f"MusicBrainz search failed: {e}")
        results = []

    prompt_words = set(normalize_string(prompt).split())

    for r in results:
        # Skip if r is not a dict
        if not isinstance(r, dict):
            logger.warning(
                f"Skipping MusicBrainz result because it is not a dict: {r!r}"
            )
            continue

        artist_list = r.get("artist-credit", [])
        title = r.get("title", "")
        if not title or not artist_list:
            continue

        artist = " & ".join(
            [
                a["artist"]["name"]
                for a in artist_list
                if "artist" in a and "name" in a["artist"]
            ]
        )
        candidate_words = set(normalize_string(f"{artist} {title}").split())
        extra_words = candidate_words - prompt_words
        score = len(candidate_words & prompt_words) - 0.5 * len(extra_words)

        # Clip score to [0, 1] for logging clarity
        log_score = max(0.0, min(1.0, score))

        candidates.append(
            {
                "artist": artist,
                "title": _strip_parentheses(title),
                "score": score,  # keep raw score for internal use
                "extra_words": extra_words,
            }
        )

        logger.debug(
            f"Candidate: artist='{artist}', title='{title}', score={log_score:.3f}, extra_words={extra_words}"
        )

    # Always include fallback
    if fallback_artist or fallback_title:
        candidates.append(
            {
                "artist": fallback_artist,
                "title": fallback_title,
                "score": 0.0,
                "extra_words": set(),
            }
        )

    return candidates


def resolve_artist_title_from_youtube(
    youtube_title: str,
    youtube_artist: Optional[str] = None,
    fallback_artist: str = "",
    fallback_title: str = "",
) -> tuple[str, str]:
    """
    Pick the best artist/title from YouTube using MusicBrainz candidates.
    """

    # 1. Clean YouTube title
    cleaned = youtube_title
    cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", cleaned)
    for s in ["Official Music Video", "Official Audio", "Lyric Video", "HD", "4K"]:
        cleaned = cleaned.replace(s, "")
    cleaned = cleaned.strip()
    logger.debug(f"Cleaned YouTube title: '{cleaned}'")

    # 2. Fetch MusicBrainz candidates
    candidates = guess_musicbrainz_candidates(
        cleaned, fallback_artist=fallback_artist, fallback_title=fallback_title
    )
    if not candidates:
        logger.debug("No MusicBrainz candidates found; using fallback")
        return fallback_artist or "Unknown", fallback_title or "Unknown"

    # Normalize YouTube info
    yt_norm = normalize_string(cleaned)
    yt_words = set(yt_norm.split())
    youtube_artist_norm = normalize_string(youtube_artist or "")

    best_score = -999.0
    best_candidate = {"artist": fallback_artist, "title": fallback_title}

    for c in candidates:
        # Guard against malformed candidate
        if not isinstance(c, dict):
            logger.warning(f"Skipping candidate because it is not a dict: {c!r}")
            continue
        if "artist" not in c or "title" not in c:
            logger.warning(f"Skipping candidate because missing keys: {c!r}")
            continue

        # Normalize candidate artist/title
        artist_norm = normalize_string(c["artist"])
        title_norm = normalize_string(c["title"])
        combined_norm = f"{artist_norm} {title_norm}"
        combined_words = set(combined_norm.split())
        artist_words = set(artist_norm.split())

        # Base similarity
        score = SequenceMatcher(None, yt_norm, combined_norm).ratio()

        # Penalize extra words
        extra_words = combined_words - yt_words
        score -= 0.05 * len(extra_words)

        # Penalize missing artist words
        missing_artist_words = artist_words - yt_words
        score -= 0.2 * len(missing_artist_words)

        # Boost if candidate artist matches YouTube uploader
        if youtube_artist_norm and youtube_artist_norm == artist_norm:
            score += 0.5

        # Slight bonus for early artist appearance in title
        artist_pos = yt_norm.find(artist_norm)
        if artist_pos >= 0:
            score += max(0, 0.1 - artist_pos / 1000)

        # Slight boost for shorter titles
        score += 0.001 / max(len(title_norm.split()), 1)

        # Log a cleaner version for clarity
        log_extra = f", extra_words={extra_words}" if extra_words else ""
        log_missing = (
            f", missing_artist_words={missing_artist_words}"
            if missing_artist_words
            else ""
        )
        logger.debug(
            f"Candidate: artist='{c['artist']}', title='{c['title']}', "
            f"score={max(0.0, score):.3f}{log_extra}{log_missing}, artist_pos={artist_pos}"
        )

        if score > best_score:
            best_score = score
            best_candidate = c

    # 3. Strip parentheticals from final title for Genius
    final_artist = best_candidate.get("artist", fallback_artist) or "Unknown"
    final_title = _strip_parentheses(
        best_candidate.get("title", fallback_title) or "Unknown"
    )

    # Use a non-negative score for logging
    log_best_score = max(0.0, best_score)
    logger.info(
        f"Selected candidate: '{final_artist}' - '{final_title}' (score={log_best_score:.3f})"
    )

    return final_artist, final_title
