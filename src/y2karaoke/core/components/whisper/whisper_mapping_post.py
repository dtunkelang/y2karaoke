"""Post-processing helpers for Whisper mapping output."""

import os

from typing import Dict, List, Set, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_mapping_post_text import (
    _interjection_similarity,
    _is_interjection_line,
    _is_placeholder_whisper_token,
    _normalize_interjection_token,
    _normalize_match_token,
    _soft_token_match,
    _soft_token_overlap_ratio,
)
from .whisper_mapping_post_repetition import (
    _extend_line_to_trailing_whisper_matches,
)
from .whisper_mapping_post_interjections import (
    _retime_short_interjection_lines as _retime_short_interjection_lines_impl,
)
from .whisper_mapping_post_overlaps import (
    _resolve_line_overlaps as _resolve_line_overlaps_impl,
)
from .whisper_mapping_post_segment_pull import (
    _pull_late_lines_to_matching_segments as _pull_late_lines_to_matching_segments_impl,
)
from . import whisper_mapping_post_repeat_shift as _repeat_shift_helpers
from . import whisper_mapping_post_onset as _onset_helpers


def _build_word_assignments_from_phoneme_path(
    path: List[Tuple[int, int]],
    lrc_phonemes: List[Dict],
    whisper_phonemes: List[Dict],
) -> Dict[int, List[int]]:
    """Convert phoneme-level DTW matches back to word-level assignments."""
    assignments: Dict[int, Set[int]] = {}
    for lpc_idx, wpc_idx in path:
        lrc_token = lrc_phonemes[lpc_idx]
        whisper_token = whisper_phonemes[wpc_idx]
        word_idx = lrc_token["word_idx"]
        whisper_word_idx = whisper_token["parent_idx"]
        assignments.setdefault(word_idx, set()).add(whisper_word_idx)
    return {
        word_idx: sorted(list(indices)) for word_idx, indices in assignments.items()
    }


def _shift_repeated_lines_to_next_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
) -> List[models.Line]:
    if os.getenv("Y2K_WHISPER_DISABLE_REPEAT_SHIFT") == "1":
        return mapped_lines
    return _repeat_shift_helpers._shift_repeated_lines_to_next_whisper(
        mapped_lines,
        all_words,
        is_placeholder_whisper_token_fn=_is_placeholder_whisper_token,
    )


def _enforce_monotonic_line_starts_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
) -> List[models.Line]:
    if os.getenv("Y2K_WHISPER_DISABLE_MONOTONIC_START_ENFORCE") == "1":
        return mapped_lines
    return _repeat_shift_helpers._enforce_monotonic_line_starts_whisper(
        mapped_lines, all_words
    )


def _resolve_line_overlaps(lines: List[models.Line]) -> List[models.Line]:  # noqa: C901
    return _resolve_line_overlaps_impl(lines)


def _pull_late_lines_to_matching_segments(  # noqa: C901
    mapped_lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    language: str,  # retained for call compatibility
    min_similarity: float = 0.4,
    min_late: float = 1.0,
    max_late: float = 3.0,
    strong_match_max_late: float = 6.0,
    min_early: float = 0.8,
    max_early: float = 3.5,
    max_early_push: float = 2.5,
    early_min_similarity: float = 0.2,
    contain_similarity_margin: float = 0.1,
    min_start_gain: float = 0.5,
    min_gap: float = 0.05,
    max_time_window: float = 15.0,
) -> List[models.Line]:
    return _pull_late_lines_to_matching_segments_impl(
        mapped_lines,
        segments,
        language=language,
        min_similarity=min_similarity,
        min_late=min_late,
        max_late=max_late,
        strong_match_max_late=strong_match_max_late,
        min_early=min_early,
        max_early=max_early,
        max_early_push=max_early_push,
        early_min_similarity=early_min_similarity,
        contain_similarity_margin=contain_similarity_margin,
        min_start_gain=min_start_gain,
        min_gap=min_gap,
        max_time_window=max_time_window,
    )


def _retime_short_interjection_lines(
    mapped_lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    min_similarity: float = 0.8,
    max_shift: float = 8.0,
    min_gap: float = 0.05,
) -> List[models.Line]:
    return _retime_short_interjection_lines_impl(
        mapped_lines,
        segments,
        is_interjection_line_fn=_is_interjection_line,
        interjection_similarity_fn=_interjection_similarity,
        min_similarity=min_similarity,
        max_shift=max_shift,
        min_gap=min_gap,
    )


def _snap_first_word_to_whisper_onset(  # noqa: C901
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    early_threshold: float = 0.12,
    max_shift: float = 0.8,
    min_gap: float = 0.05,
) -> List[models.Line]:
    return _onset_helpers._snap_first_word_to_whisper_onset(
        mapped_lines,
        all_words,
        normalize_interjection_token_fn=_normalize_interjection_token,
        normalize_match_token_fn=_normalize_match_token,
        soft_token_match_fn=_soft_token_match,
        soft_token_overlap_ratio_fn=_soft_token_overlap_ratio,
        early_threshold=early_threshold,
        max_shift=max_shift,
        min_gap=min_gap,
    )


__all__ = [
    "_build_word_assignments_from_phoneme_path",
    "_enforce_monotonic_line_starts_whisper",
    "_extend_line_to_trailing_whisper_matches",
    "_pull_late_lines_to_matching_segments",
    "_resolve_line_overlaps",
    "_retime_short_interjection_lines",
    "_shift_repeated_lines_to_next_whisper",
    "_snap_first_word_to_whisper_onset",
]
