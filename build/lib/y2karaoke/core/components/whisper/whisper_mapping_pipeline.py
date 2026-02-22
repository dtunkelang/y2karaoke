"""Core mapping pipeline from LRC words to Whisper words."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ....utils.logging import get_logger
from ... import models, phonetic_utils
from ..alignment import timing_models
from . import whisper_utils
from .whisper_dtw import _LineMappingContext
from . import whisper_mapping_pipeline_assembly as _assembly_helpers
from . import whisper_mapping_pipeline_candidates as _candidate_helpers
from . import whisper_mapping_pipeline_line_context as _line_context_helpers
from . import whisper_mapping_pipeline_matching as _matching_helpers
from . import whisper_mapping_pipeline_orchestration as _orchestration_helpers

logger = get_logger(__name__)
_TIME_DRIFT_THRESHOLD = 0.8
_MAX_ANCHOR_DRIFT_FROM_LRC = 6.0
_MAX_MATCHED_START_DRIFT_FORWARD = 8.0
_MAX_MATCHED_START_DRIFT_BACKWARD = 4.0
_MAX_LINE_FORWARD_SHIFT_FROM_LRC = 8.0
_MAX_LINE_BACKWARD_SHIFT_FROM_LRC = 5.0
_MAX_LINE_DURATION_SCALE_FROM_LRC = 1.6


def _fallback_unmatched_line_duration(line: "models.Line") -> float:
    return _line_context_helpers._fallback_unmatched_line_duration(line)


def _register_word_match(
    ctx: _LineMappingContext,
    line_idx: int,
    word: "models.Word",
    best_word: timing_models.TranscriptionWord,
    best_idx: int,
    candidates: List[Tuple[timing_models.TranscriptionWord, int]],
    line_segment: Optional[int],
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    word_idx: int,
    line_last_idx_ref: List[Optional[int]],
) -> None:
    return _candidate_helpers._register_word_match(
        ctx,
        line_idx,
        word,
        best_word,
        best_idx,
        candidates,
        line_segment,
        line_matches,
        line_match_intervals,
        word_idx,
        line_last_idx_ref,
        phonetic_similarity_fn=phonetic_utils._phonetic_similarity,
    )


def _select_best_candidate(
    ctx: _LineMappingContext,
    whisper_candidates: List[Tuple[timing_models.TranscriptionWord, int]],
    word: "models.Word",
    line_shift: float,
    line_segment: Optional[int],
    line_anchor_time: float,
    lrc_idx_opt: Optional[int],
) -> Tuple[timing_models.TranscriptionWord, int]:
    return _candidate_helpers._select_best_candidate(
        ctx,
        whisper_candidates,
        word,
        line_shift,
        line_segment,
        line_anchor_time,
        lrc_idx_opt,
        time_drift_threshold=_TIME_DRIFT_THRESHOLD,
    )


def _filter_and_order_candidates(
    ctx: _LineMappingContext,
    candidate_indices: Set[int],
) -> List[Tuple[timing_models.TranscriptionWord, int]]:
    return _candidate_helpers._filter_and_order_candidates(ctx, candidate_indices)


def _prepare_line_context(
    ctx: _LineMappingContext,
    line: "models.Line",
) -> Tuple[Optional[int], float, float]:
    return _line_context_helpers._prepare_line_context(
        ctx,
        line,
        max_anchor_drift_from_lrc=_MAX_ANCHOR_DRIFT_FROM_LRC,
    )


def _should_override_line_segment(
    *,
    current_segment: Optional[int],
    override_segment: int,
    override_hits: int,
    line_word_count: int,
    line_anchor_time: float,
    segments: Sequence[Any],
    max_local_jump_seconds: float = 8.0,
    max_strong_jump_seconds: float = 18.0,
    max_anchor_jump_seconds: float = 14.0,
    max_anchor_strong_jump_seconds: float = 20.0,
) -> bool:
    return _line_context_helpers._should_override_line_segment(
        current_segment=current_segment,
        override_segment=override_segment,
        override_hits=override_hits,
        line_word_count=line_word_count,
        line_anchor_time=line_anchor_time,
        segments=segments,
        max_local_jump_seconds=max_local_jump_seconds,
        max_strong_jump_seconds=max_strong_jump_seconds,
        max_anchor_jump_seconds=max_anchor_jump_seconds,
        max_anchor_strong_jump_seconds=max_anchor_strong_jump_seconds,
    )


def _clamp_match_window_to_anchor(
    actual_start: float,
    actual_end: float,
    line_anchor_time: float,
    *,
    max_forward: float = _MAX_MATCHED_START_DRIFT_FORWARD,
    max_backward: float = _MAX_MATCHED_START_DRIFT_BACKWARD,
) -> Tuple[float, float]:
    return _line_context_helpers._clamp_match_window_to_anchor(
        actual_start,
        actual_end,
        line_anchor_time,
        max_forward=max_forward,
        max_backward=max_backward,
    )


def _clamp_line_shift_vs_original(
    mapped_line: "models.Line",
    original_line: "models.Line",
    *,
    max_forward: float = _MAX_LINE_FORWARD_SHIFT_FROM_LRC,
    max_backward: float = _MAX_LINE_BACKWARD_SHIFT_FROM_LRC,
) -> "models.Line":
    return _line_context_helpers._clamp_line_shift_vs_original(
        mapped_line,
        original_line,
        max_forward=max_forward,
        max_backward=max_backward,
    )


def _clamp_line_duration_vs_original(
    mapped_line: "models.Line",
    original_line: "models.Line",
    next_original_start: Optional[float],
    *,
    max_scale: float = _MAX_LINE_DURATION_SCALE_FROM_LRC,
    slack_seconds: float = 0.9,
) -> "models.Line":
    return _line_context_helpers._clamp_line_duration_vs_original(
        mapped_line,
        original_line,
        next_original_start,
        max_scale=max_scale,
        slack_seconds=slack_seconds,
    )


def _match_assigned_words(
    ctx: _LineMappingContext,
    line_idx: int,
    line: "models.Line",
    lrc_index_by_loc: Dict[Tuple[int, int], int],
    lrc_assignments: Dict[int, List[int]],
    line_segment: Optional[int],
    line_anchor_time: float,
    line_shift: float,
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    line_last_idx_ref: List[Optional[int]],
) -> None:
    return _matching_helpers._match_assigned_words(
        ctx,
        line_idx,
        line,
        lrc_index_by_loc,
        lrc_assignments,
        line_segment,
        line_anchor_time,
        line_shift,
        line_matches,
        line_match_intervals,
        line_last_idx_ref,
        filter_and_order_candidates_fn=_filter_and_order_candidates,
        select_best_candidate_fn=_select_best_candidate,
        register_word_match_fn=_register_word_match,
    )


def _compute_gap_window(
    line: "models.Line",
    word_idx: int,
    line_match_intervals: Dict[int, Tuple[float, float]],
    prev_line_end: float,
) -> Tuple[float, float]:
    return _matching_helpers._compute_gap_window(
        line, word_idx, line_match_intervals, prev_line_end
    )


def _fill_unmatched_gaps(
    ctx: _LineMappingContext,
    line_idx: int,
    line: "models.Line",
    lrc_index_by_loc: Dict[Tuple[int, int], int],
    lrc_assignments: Dict[int, List[int]],
    line_segment: Optional[int],
    line_anchor_time: float,
    line_shift: float,
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    line_last_idx_ref: List[Optional[int]],
) -> None:
    return _matching_helpers._fill_unmatched_gaps(
        ctx,
        line_idx,
        line,
        lrc_index_by_loc,
        lrc_assignments,
        line_segment,
        line_anchor_time,
        line_shift,
        line_matches,
        line_match_intervals,
        line_last_idx_ref,
        compute_gap_window_fn=_compute_gap_window,
        select_best_candidate_fn=_select_best_candidate,
        register_word_match_fn=_register_word_match,
    )


def _assemble_mapped_line(
    ctx: _LineMappingContext,
    line_idx: int,
    line: "models.Line",
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    line_anchor_time: float,
    line_segment: Optional[int],
    line_last_idx_ref: List[Optional[int]],
    next_original_start: Optional[float],
) -> "models.Line":
    return _assembly_helpers._assemble_mapped_line(
        ctx,
        line_idx,
        line,
        line_matches,
        line_match_intervals,
        line_anchor_time,
        line_segment,
        line_last_idx_ref,
        next_original_start,
        clamp_match_window_to_anchor_fn=_clamp_match_window_to_anchor,
        fallback_unmatched_line_duration_fn=_fallback_unmatched_line_duration,
        redistribute_word_timings_to_line_fn=whisper_utils._redistribute_word_timings_to_line,
        clamp_line_shift_vs_original_fn=_clamp_line_shift_vs_original,
        clamp_line_duration_vs_original_fn=_clamp_line_duration_vs_original,
        logger=logger,
    )


def _map_lrc_words_to_whisper(
    lines: List[models.Line],
    lrc_words: List[Dict],
    all_words: List[timing_models.TranscriptionWord],
    lrc_assignments: Dict[int, List[int]],
    language: str,
    segments: Sequence[Any],
) -> Tuple[List[models.Line], int, float, set]:
    return _orchestration_helpers._map_lrc_words_to_whisper(
        lines,
        lrc_words,
        all_words,
        lrc_assignments,
        language,
        segments,
        prepare_line_context_fn=_prepare_line_context,
        should_override_line_segment_fn=_should_override_line_segment,
        match_assigned_words_fn=_match_assigned_words,
        fill_unmatched_gaps_fn=_fill_unmatched_gaps,
        assemble_mapped_line_fn=_assemble_mapped_line,
        logger=logger,
    )
