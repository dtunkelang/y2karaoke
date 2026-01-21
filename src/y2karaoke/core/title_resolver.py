"""Resolve canonical artist/title for a song, robust to YouTube titles and MusicBrainz results."""

import re
from difflib import SequenceMatcher
import logging
import musicbrainzngs

logger = logging.getLogger(__name__)

STOP_WORDS = {"the", "a", "an", "&", "and", "of", "with", "in", "+"}

# Initialize MusicBrainz once
musicbrainzngs.set_useragent(
    "y2karaoke", "1.0", "https://github.com/dtunkelang/y2karaoke"
)


def _normalize_for_matching(s: str) -> str:
    s = s.casefold()
    s = "".join(c for c in s if c.isalnum() or c.isspace())
    tokens = [t for t in s.split() if t not in STOP_WORDS]
    return " ".join(tokens).strip()


def guess_musicbrainz_candidates(prompt: str, fallback_artist: str = "", fallback_title: str = "") -> list[dict]:
    """Query MusicBrainz and return a list of candidate (artist, title) pairs with scores."""
    from musicbrainzngs import search_recordings

    logger.info(f"Querying MusicBrainz for prompt: '{prompt}'")
    candidates = []

    try:
        results = search_recordings(recording=prompt, limit=15).get("recording-list", [])
    except Exception as e:
        logger.warning(f"MusicBrainz search failed: {e}")
        results = []

    prompt_words = set(_normalize_for_matching(prompt).split())

    for r in results:
        artist_list = r.get("artist-credit", [])
        title = r.get("title", "")
        if not title or not artist_list:
            continue
        artist = " & ".join([a["artist"]["name"] for a in artist_list])
        candidate_words = set(_normalize_for_matching(f"{artist} {title}").split())
        extra_words = candidate_words - prompt_words
        score = len(candidate_words & prompt_words) - 0.5 * len(extra_words)
        candidates.append({
            "artist": artist,
            "title": _strip_parentheses(title),
            "score": score,
            "extra_words": extra_words
        })
        logger.info(f"Candidate: artist='{artist}', title='{title}', score={score:.3f}, extra_words={extra_words}")

    # Always include fallback
    if fallback_artist or fallback_title:
        candidates.append({
            "artist": fallback_artist,
            "title": fallback_title,
            "score": 0.0,
            "extra_words": set()
        })

    return candidates


def _strip_parentheses(title: str) -> str:
    """Remove parenthetical expressions from a title for canonical matching."""
    return re.sub(r"\s*\(.*?\)\s*", "", title).strip()


def resolve_artist_title_from_youtube(youtube_title: str, fallback_artist: str = "", fallback_title: str = "") -> tuple[str, str]:
    """Resolve the most likely canonical artist/title from a YouTube title."""
    # Clean YouTube title
    cleaned = youtube_title
    cleaned = re.sub(r'[\(\[\{].*?[\)\]\}]', '', cleaned)
    for s in ["Official Music Video", "Official Audio", "Lyric Video", "HD", "4K"]:
        cleaned = cleaned.replace(s, '')
    parts = cleaned.split(' - ')
    if len(parts) > 2 and parts[0].strip().lower() == parts[1].strip().lower():
        cleaned = " - ".join([parts[0].strip()] + parts[2:])
    cleaned = cleaned.strip()
    logger.info(f"Cleaned YouTube title: '{cleaned}'")

    # Fetch candidates
    candidates = guess_musicbrainz_candidates(cleaned, fallback_artist, fallback_title)
    if not candidates:
        return fallback_artist or "Unknown", fallback_title or "Unknown"

    yt_norm = re.sub(r'[^a-z0-9]+', ' ', cleaned.lower()).strip()
    yt_words = set(yt_norm.split())

    best_score = -999
    best_candidate = (fallback_artist, fallback_title)

    for c in candidates:
        artist_norm = re.sub(r'[^a-z0-9]+', ' ', c["artist"].lower()).strip()
        title_norm = re.sub(r'[^a-z0-9]+', ' ', c["title"].lower()).strip()
        combined_norm = f"{artist_norm} {title_norm}"
        combined_words = set(combined_norm.split())

        score = SequenceMatcher(None, yt_norm, combined_norm).ratio()

        # Penalize extra words
        extra_words = combined_words - yt_words
        score -= 0.05 * len(extra_words)

        # Heavily penalize missing artist words
        artist_words = set(artist_norm.split())
        missing_artist_words = artist_words - yt_words
        score -= 0.2 * len(missing_artist_words)

        # Bonus if artist appears early
        artist_pos = yt_norm.find(artist_norm)
        if artist_pos >= 0:
            score += max(0, 0.1 - artist_pos / 1000)

        score += 0.001 / max(len(title_norm.split()), 1)

        if score > best_score:
            best_score = score
            best_candidate = (c["artist"], c["title"])

    logger.info(f"Selected candidate: '{best_candidate[0]}' - '{best_candidate[1]}' (score={best_score:.3f})")
    return best_candidate
