"""Pure helper rules for YouTube track-identification heuristics."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def is_likely_non_studio(title: str) -> bool:
    """Check if a title suggests a non-studio version."""
    title_lower = title.lower()
    studio_edit_pattern = r"\b(single|radio|album)\s*(edit|version)?\b"
    if re.search(studio_edit_pattern, title_lower):
        if "radio" in title_lower:
            radio_show_patterns = [
                r"radio\s*\d",
                r"radio\s+session",
                r"bbc\s+radio",
                r"radio\s+show",
            ]
            if not any(re.search(p, title_lower) for p in radio_show_patterns):
                return False
        else:
            return False

    non_studio_terms = [
        "live",
        "concert",
        "performance",
        "performs",
        "performing",
        "tour",
        "in concert",
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
        "cover",
        "tribute",
        "karaoke",
        "instrumental",
        "reaction",
        "tutorial",
        "lesson",
        "how to play",
        "guitar lesson",
        "slowed",
        "sped up",
        "reverb",
        "8d audio",
        "nightcore",
        "bass boosted",
        "session",
        "sessions",
        "bbc session",
        "peel session",
        "maida vale",
        "parody",
        "weird al",
    ]
    if any(term in title_lower for term in non_studio_terms):
        return True

    live_patterns = [
        r"\bat\b.*\b(show|festival|arena|stadium|hall|theater|theatre|club|center|centre)\b",
        r"\blive\s+(at|from|in)\b",
        r"\b(snl|saturday night live|letterman|fallon|kimmel|conan|ellen|tonight show)\b",
        r"\b(jools holland|later with|top of the pops|totp|graham norton)\b",
        r"\b(tiny desk|npr|kexp|colors?\s*show|a]colors)\b",
        r"\b(glastonbury|coachella|lollapalooza|reading|leeds|bonnaroo)\b",
        r"\b(rock am ring|download|download festival|wacken|hellfest)\b",
        r"\b(south by southwest|sxsw|austin city limits|acl)\b",
        r"\b(mtv|vma|grammy|grammys|brit awards?|ama|american music)\b",
        r"\b(billboard|bet awards|iheartradio)\b",
        r"\b(unplugged|stripped|acoustic sessions?)\b",
        r"\b(spotify\s*sessions?|apple\s*music\s*sessions?)\b",
        r"\b(19|20)\d{2}\s+(tour|live|concert|performance)\b",
    ]
    if any(re.search(pattern, title_lower) for pattern in live_patterns):
        return True

    paren_match = re.search(r"\(([^)]+)\)", title_lower)
    if paren_match:
        paren_content = paren_match.group(1)
        if re.search(r"\b(single|radio|album)\s*(edit|version)?\b", paren_content):
            return False
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


def is_preferred_audio_title(title: str) -> bool:
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


def query_wants_non_studio(query: str) -> bool:
    query_lower = query.lower()
    return any(
        term in query_lower
        for term in ["live", "concert", "acoustic", "remix", "cover", "karaoke"]
    )


def youtube_duration_tolerance(target_duration: int) -> int:
    return max(20, int(target_duration * 0.15)) if target_duration > 0 else 30


def extract_youtube_candidates(response_text: str) -> List[Dict[str, Any]]:
    """Extract video candidates from YouTube search HTML."""
    candidates: List[Dict[str, Any]] = []
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
            parts = duration_match.group(1).split(":")
            if len(parts) == 2:
                duration_sec = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        candidates.append(
            {"video_id": video_id, "title": title, "duration": duration_sec}
        )
    return candidates
