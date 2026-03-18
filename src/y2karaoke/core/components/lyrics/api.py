"""Lyrics public API.

Keep the public surface small and explicit. Older tests and callers may still
reach into this module for private helper access, so compatibility resolution is
kept lazy via ``__getattr__`` instead of eagerly re-exporting internals.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import List, Optional, Tuple

from ...models import Line, SongMetadata, Word
from .helpers import romanize_line
from .lrc import (
    create_lines_from_lrc,
    create_lines_from_lrc_timings,
    parse_lrc_timestamp,
    parse_lrc_with_timing,
    split_long_lines,
)
from .lyrics_whisper import get_lyrics_simple, get_lyrics_with_quality

__all__ = [
    "LyricsProcessor",
    "get_lyrics",
    "get_lyrics_simple",
    "get_lyrics_with_quality",
    "parse_lrc_timestamp",
    "parse_lrc_with_timing",
    "create_lines_from_lrc",
    "create_lines_from_lrc_timings",
    "split_long_lines",
    "romanize_line",
    "Word",
    "Line",
    "SongMetadata",
]

_LEGACY_EXPORTS = {
    "_apply_lrc_weighted_timing": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_apply_lrc_weighted_timing",
    ),
    "_apply_singer_info": (
        "y2karaoke.core.components.lyrics.lyrics_whisper",
        "_apply_singer_info",
    ),
    "_apply_timing_to_lines": (
        "y2karaoke.core.components.lyrics.helpers",
        "_apply_timing_to_lines",
    ),
    "_apply_weighted_slots_to_line": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_apply_weighted_slots_to_line",
    ),
    "_apply_whisper_alignment": (
        "y2karaoke.core.components.lyrics.helpers",
        "_apply_whisper_alignment",
    ),
    "_apply_whisper_with_quality": (
        "y2karaoke.core.components.lyrics.lyrics_whisper",
        "_apply_whisper_with_quality",
    ),
    "_build_whisper_word_list": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_build_whisper_word_list",
    ),
    "_calculate_quality_score": (
        "y2karaoke.core.components.lyrics.lyrics_whisper",
        "_calculate_quality_score",
    ),
    "_clamp_line_end_to_next_start": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_clamp_line_end_to_next_start",
    ),
    "_clean_text_lines": (
        "y2karaoke.core.components.lyrics.helpers",
        "_clean_text_lines",
    ),
    "_compress_spurious_lrc_gaps": (
        "y2karaoke.core.components.lyrics.helpers",
        "_compress_spurious_lrc_gaps",
    ),
    "_create_lines_from_plain_text": (
        "y2karaoke.core.components.lyrics.helpers",
        "_create_lines_from_plain_text",
    ),
    "_create_lines_from_whisper": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_create_lines_from_whisper",
    ),
    "_create_no_lyrics_placeholder": (
        "y2karaoke.core.components.lyrics.helpers",
        "_create_no_lyrics_placeholder",
    ),
    "_detect_and_apply_offset": (
        "y2karaoke.core.components.lyrics.helpers",
        "_detect_and_apply_offset",
    ),
    "_detect_offset_with_issues": (
        "y2karaoke.core.components.lyrics.lyrics_whisper",
        "_detect_offset_with_issues",
    ),
    "_distribute_word_timing_in_line": (
        "y2karaoke.core.components.lyrics.helpers",
        "_distribute_word_timing_in_line",
    ),
    "_estimate_singing_duration": (
        "y2karaoke.core.components.lyrics.helpers",
        "_estimate_singing_duration",
    ),
    "_extract_text_lines_from_lrc": (
        "y2karaoke.core.components.lyrics.helpers",
        "_extract_text_lines_from_lrc",
    ),
    "_fetch_genius_with_quality_tracking": (
        "y2karaoke.core.components.lyrics.lyrics_whisper",
        "_fetch_genius_with_quality_tracking",
    ),
    "_fetch_lrc_text_and_timings": (
        "y2karaoke.core.components.lyrics.lyrics_whisper",
        "_fetch_lrc_text_and_timings",
    ),
    "_find_best_segment_for_line": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_find_best_segment_for_line",
    ),
    "_line_duration_from_lrc": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_line_duration_from_lrc",
    ),
    "_load_lyrics_file": (
        "y2karaoke.core.components.lyrics.helpers",
        "_load_lyrics_file",
    ),
    "_map_line_words_to_segment": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_map_line_words_to_segment",
    ),
    "_map_lrc_lines_to_whisper_segments": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_map_lrc_lines_to_whisper_segments",
    ),
    "_norm_token": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_norm_token",
    ),
    "_refine_timing_with_audio": (
        "y2karaoke.core.components.lyrics.helpers",
        "_refine_timing_with_audio",
    ),
    "_refine_timing_with_quality": (
        "y2karaoke.core.components.lyrics.lyrics_whisper",
        "_refine_timing_with_quality",
    ),
    "_resample_slots_to_line": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_resample_slots_to_line",
    ),
    "_romanize_lines": (
        "y2karaoke.core.components.lyrics.helpers",
        "_romanize_lines",
    ),
    "_score_from_dtw_metrics": (
        "y2karaoke.core.components.lyrics.lyrics_whisper",
        "_score_from_dtw_metrics",
    ),
    "_select_window_sequence": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_select_window_sequence",
    ),
    "_select_window_words_for_line": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_select_window_words_for_line",
    ),
    "_shift_words": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_shift_words",
    ),
    "_slots_from_sequence": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_slots_from_sequence",
    ),
    "_whisper_durations_for_line": (
        "y2karaoke.core.components.lyrics.lyrics_whisper_map",
        "_whisper_durations_for_line",
    ),
}


class LyricsProcessor:
    """High-level lyrics processor with caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "karaoke")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_lyrics(
        self,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        romanize: bool = True,
        **kwargs,
    ) -> Tuple[List[Line], Optional[SongMetadata]]:
        """Get lyrics for a song."""
        if not title or not artist:
            placeholder_line = Line(words=[])
            placeholder_metadata = SongMetadata(
                singers=[],
                is_duet=False,
                title=title or "Unknown",
                artist=artist or "Unknown",
            )
            return [placeholder_line], placeholder_metadata

        lines, metadata = get_lyrics_simple(
            title=title,
            artist=artist,
            vocals_path=kwargs.get("vocals_path"),
            cache_dir=str(self.cache_dir),
            lyrics_offset=kwargs.get("lyrics_offset"),
            romanize=romanize,
        )
        return lines, metadata


def get_lyrics(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Get lyrics for a song (convenience function)."""
    return get_lyrics_simple(
        title=title,
        artist=artist,
        vocals_path=vocals_path,
        cache_dir=cache_dir,
    )


def __getattr__(name: str):
    if name in _LEGACY_EXPORTS:
        module_name, attr_name = _LEGACY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LEGACY_EXPORTS.keys()))
