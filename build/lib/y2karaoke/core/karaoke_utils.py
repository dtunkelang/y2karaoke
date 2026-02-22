"""Utility helpers used by KaraokeGenerator."""

from typing import Any, Dict, List, Tuple

from .models import Line, Word


def apply_splash_offset(lines, min_start: float = 3.5):
    """Shift all lines forward so first line starts no earlier than min_start."""
    if not lines or lines[0].start_time >= min_start:
        return lines
    splash_offset = min_start - lines[0].start_time
    offset_lines = []
    for line in lines:
        offset_words = [
            Word(
                text=w.text,
                start_time=w.start_time + splash_offset,
                end_time=w.end_time + splash_offset,
                singer=w.singer,
            )
            for w in line.words
        ]
        offset_lines.append(Line(words=offset_words, singer=line.singer))
    return offset_lines


def scale_lyrics_timing(lines, tempo_multiplier: float):
    """Scale all line/word timings to match tempo changes."""
    if tempo_multiplier == 1.0:
        return lines
    scaled_lines = []
    for line in lines:
        scaled_words = [
            Word(
                text=w.text,
                start_time=w.start_time / tempo_multiplier,
                end_time=w.end_time / tempo_multiplier,
                singer=w.singer,
            )
            for w in line.words
        ]
        scaled_lines.append(Line(words=scaled_words, singer=line.singer))
    return scaled_lines


def summarize_quality(
    lyrics_result: Dict[str, Any],
) -> Tuple[float, List[str], str, str]:
    """Compute user-facing quality summary tuple."""
    lyrics_quality = lyrics_result.get("quality", {})
    quality_score = lyrics_quality.get("overall_score", 50.0)
    quality_issues = lyrics_quality.get("issues", [])

    if quality_score >= 80:
        quality_emoji = "✅"
        quality_level = "high"
    elif quality_score >= 50:
        quality_emoji = "⚠️"
        quality_level = "medium"
    else:
        quality_emoji = "❌"
        quality_level = "low"

    return quality_score, quality_issues, quality_level, quality_emoji
