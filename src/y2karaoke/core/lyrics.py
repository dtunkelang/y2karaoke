"""Lyrics public API.

This module provides the main interface for lyrics fetching and processing:
- Fetches lyrics from Genius (canonical text + singer info)
- Gets LRC timing from syncedlyrics
- Aligns text to audio for word-level timing
- Applies romanization for non-Latin scripts
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .models import Line, SongMetadata, Word
from .lrc import (
    parse_lrc_timestamp,
    parse_lrc_with_timing,
    create_lines_from_lrc,
    create_lines_from_lrc_timings,
    split_long_lines,
)
from .lyrics_whisper import (
    get_lyrics_simple,
    get_lyrics_with_quality,
    _apply_singer_info,
    _detect_offset_with_issues,
    _refine_timing_with_quality,
    _calculate_quality_score,
    _score_from_dtw_metrics,
    _fetch_lrc_text_and_timings,
    _fetch_genius_with_quality_tracking,
    _apply_whisper_with_quality,
)
from .lyrics_whisper_map import (
    _norm_token,
    _build_whisper_word_list,
    _select_window_sequence,
    _slots_from_sequence,
    _resample_slots_to_line,
    _whisper_durations_for_line,
    _apply_weighted_slots_to_line,
    _shift_words,
    _find_best_segment_for_line,
    _map_line_words_to_segment,
    _select_window_words_for_line,
    _line_duration_from_lrc,
    _apply_lrc_weighted_timing,
    _clamp_line_end_to_next_start,
    _create_lines_from_whisper,
    _map_lrc_lines_to_whisper_segments,
)
from .lyrics_helpers import (
    _estimate_singing_duration,
    _extract_text_lines_from_lrc,
    _create_lines_from_plain_text,
    _clean_text_lines,
    _load_lyrics_file,
    _create_no_lyrics_placeholder,
    _detect_and_apply_offset,
    _distribute_word_timing_in_line,
    _apply_timing_to_lines,
    _refine_timing_with_audio,
    _compress_spurious_lrc_gaps,
    _apply_whisper_alignment,
    _romanize_lines,
    romanize_line,
)

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
    "Word",
    "Line",
    "SongMetadata",
    "romanize_line",
    "_norm_token",
    "_build_whisper_word_list",
    "_select_window_sequence",
    "_slots_from_sequence",
    "_resample_slots_to_line",
    "_whisper_durations_for_line",
    "_apply_weighted_slots_to_line",
    "_shift_words",
    "_find_best_segment_for_line",
    "_map_line_words_to_segment",
    "_select_window_words_for_line",
    "_line_duration_from_lrc",
    "_apply_lrc_weighted_timing",
    "_clamp_line_end_to_next_start",
    "_create_lines_from_whisper",
    "_map_lrc_lines_to_whisper_segments",
    "_apply_singer_info",
    "_detect_offset_with_issues",
    "_refine_timing_with_quality",
    "_calculate_quality_score",
    "_score_from_dtw_metrics",
    "_fetch_lrc_text_and_timings",
    "_fetch_genius_with_quality_tracking",
    "_apply_whisper_with_quality",
    "_estimate_singing_duration",
    "_extract_text_lines_from_lrc",
    "_create_lines_from_plain_text",
    "_clean_text_lines",
    "_load_lyrics_file",
    "_create_no_lyrics_placeholder",
    "_detect_and_apply_offset",
    "_distribute_word_timing_in_line",
    "_apply_timing_to_lines",
    "_refine_timing_with_audio",
    "_compress_spurious_lrc_gaps",
    "_apply_whisper_alignment",
    "_romanize_lines",
]


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
        """Get lyrics for a song.

        Args:
            title: Song title
            artist: Artist name
            romanize: Whether to romanize non-Latin scripts
            **kwargs: Additional options (vocals_path, lyrics_offset)

        Returns:
            Tuple of (lines, metadata)
        """
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
    """Get lyrics for a song (convenience function).

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio (optional)
        cache_dir: Cache directory (optional)

    Returns:
        Tuple of (lines, metadata)
    """
    return get_lyrics_simple(
        title=title,
        artist=artist,
        vocals_path=vocals_path,
        cache_dir=cache_dir,
    )