"""Assigned-word and gap-fill matching helpers for Whisper mapping pipeline."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ... import models
from .whisper_dtw import _LineMappingContext
from .whisper_mapping_helpers import (
    _collect_unused_words_in_window,
    _collect_unused_words_near_line,
    _segment_word_indices,
)
from . import whisper_utils


def _should_skip_collapsed_assigned_match(
    *,
    ctx: _LineMappingContext,
    line_idx: int,
    line: "models.Line",
    lrc_index_by_loc: Dict[Tuple[int, int], int],
    lrc_assignments: Dict[int, List[int]],
    min_words: int = 6,
    max_unique_assigned_words: int = 3,
    min_late_start_sec: float = 2.5,
) -> bool:
    if len(line.words) < min_words:
        return False
    assigned_indices: set[int] = set()
    for word_idx in range(len(line.words)):
        lrc_idx_opt = lrc_index_by_loc.get((line_idx, word_idx))
        if lrc_idx_opt is None:
            continue
        assigned_indices.update(lrc_assignments.get(lrc_idx_opt, []))
    if not assigned_indices or len(assigned_indices) > max_unique_assigned_words:
        return False
    valid_indices = [idx for idx in assigned_indices if 0 <= idx < len(ctx.all_words)]
    if len(valid_indices) != len(assigned_indices):
        return False
    earliest_assigned_start = min(ctx.all_words[idx].start for idx in valid_indices)
    return (earliest_assigned_start - line.start_time) >= min_late_start_sec


def _compute_gap_window(
    line: "models.Line",
    word_idx: int,
    line_match_intervals: Dict[int, Tuple[float, float]],
    prev_line_end: float,
) -> Tuple[float, float]:
    """Compute the time window for an unmatched word based on neighbors."""
    prev_matches = [idx for idx in line_match_intervals if idx < word_idx]
    next_matches = [idx for idx in line_match_intervals if idx > word_idx]
    gap_start = line.start_time
    if prev_matches:
        gap_start = max(gap_start, line_match_intervals[max(prev_matches)][1])
    gap_start = max(gap_start, prev_line_end)
    gap_end = line.end_time
    if next_matches:
        gap_end = min(gap_end, line_match_intervals[min(next_matches)][0])
    return gap_start, gap_end


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
    *,
    filter_and_order_candidates_fn: Callable[..., Any],
    select_best_candidate_fn: Callable[..., Any],
    register_word_match_fn: Callable[..., None],
) -> None:
    """Phase 1: match words using DTW assignments."""
    if _should_skip_collapsed_assigned_match(
        ctx=ctx,
        line_idx=line_idx,
        line=line,
        lrc_index_by_loc=lrc_index_by_loc,
        lrc_assignments=lrc_assignments,
    ):
        return
    for word_idx, word in enumerate(line.words):
        lrc_idx_opt = lrc_index_by_loc.get((line_idx, word_idx))
        if lrc_idx_opt is None:
            continue
        assigned = lrc_assignments.get(lrc_idx_opt)
        if not assigned:
            continue
        fallback_used = False
        candidate_indices: Set[int] = set(assigned)
        segment_indices: Set[int] = set()
        if line_segment is not None:
            segment_indices = set(
                _segment_word_indices(ctx.all_words, ctx.word_segment_idx, line_segment)
            )
            candidate_indices.update(segment_indices)
        whisper_candidates = filter_and_order_candidates_fn(ctx, candidate_indices)
        if not whisper_candidates:
            fallback_indices = _collect_unused_words_near_line(
                ctx.all_words,
                line,
                ctx.used_word_indices,
                ctx.next_word_idx_start,
                ctx.prev_line_end,
            )
            fallback_used = True
            whisper_candidates = filter_and_order_candidates_fn(
                ctx, set(fallback_indices)
            )
        if not whisper_candidates:
            continue
        best_word, best_idx = select_best_candidate_fn(
            ctx,
            whisper_candidates,
            word,
            line_shift,
            line_segment,
            line_anchor_time,
            lrc_idx_opt,
            trace_context={
                "phase": "assigned_words",
                "assigned_count": len(assigned),
                "segment_count": len(segment_indices),
                "fallback_used": fallback_used,
            },
        )
        register_word_match_fn(
            ctx,
            line_idx,
            word,
            best_word,
            best_idx,
            whisper_candidates,
            line_segment,
            line_matches,
            line_match_intervals,
            word_idx,
            line_last_idx_ref,
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
    *,
    compute_gap_window_fn: Callable[..., Tuple[float, float]],
    select_best_candidate_fn: Callable[..., Any],
    register_word_match_fn: Callable[..., None],
) -> None:
    """Phase 2: fill unmatched words using time-window search."""
    unmatched = [
        idx for idx in range(len(line.words)) if idx not in line_match_intervals
    ]
    for word_idx in unmatched:
        word = line.words[word_idx]
        gap_start, gap_end = compute_gap_window_fn(
            line, word_idx, line_match_intervals, ctx.prev_line_end
        )
        if gap_end <= gap_start:
            continue
        lrc_idx_opt = lrc_index_by_loc.get((line_idx, word_idx))
        assigned = (
            lrc_assignments.get(lrc_idx_opt, []) if lrc_idx_opt is not None else []
        )
        candidate_indices: Set[int] = set(assigned)
        segment_indices: Set[int] = set()
        if line_segment is not None:
            segment_indices = set(
                _segment_word_indices(ctx.all_words, ctx.word_segment_idx, line_segment)
            )
            candidate_indices.update(segment_indices)
        window_start, window_end, gap_min_idx = _gap_window_bounds(
            ctx=ctx,
            gap_start=gap_start,
            gap_end=gap_end,
        )
        window_candidates = _collect_unused_words_in_window(
            ctx.all_words,
            ctx.used_word_indices,
            gap_min_idx,
            window_start,
            window_end,
        )
        candidate_indices.update(window_candidates)
        whisper_candidates = _build_gap_candidates(
            ctx=ctx,
            candidate_indices=candidate_indices,
            gap_min_idx=gap_min_idx,
            gap_start=gap_start,
            window_end=window_end,
        )
        ordered = [
            pair for pair in whisper_candidates if pair[0].start >= ctx.last_line_start
        ]
        if ordered:
            whisper_candidates = ordered
        whisper_candidates = _restrict_gap_candidates_to_blocks(ctx, whisper_candidates)
        if not whisper_candidates:
            continue
        best_word, best_idx = select_best_candidate_fn(
            ctx,
            whisper_candidates,
            word,
            line_shift,
            line_segment,
            line_anchor_time,
            lrc_idx_opt,
            trace_context={
                "phase": "gap_fill",
                "assigned_count": len(assigned),
                "segment_count": len(segment_indices),
                "window_candidate_count": len(window_candidates),
                "gap_start": round(gap_start, 3),
                "gap_end": round(gap_end, 3),
                "window_start": round(window_start, 3),
                "window_end": round(window_end, 3),
            },
        )
        register_word_match_fn(
            ctx,
            line_idx,
            word,
            best_word,
            best_idx,
            whisper_candidates,
            line_segment,
            line_matches,
            line_match_intervals,
            word_idx,
            line_last_idx_ref,
        )


def _gap_window_bounds(
    *,
    ctx: _LineMappingContext,
    gap_start: float,
    gap_end: float,
) -> tuple[float, float, int]:
    window_start = max(gap_start - 0.25, 0.0)
    window_end = gap_end + 0.25
    if ctx.speech_blocks and ctx.current_block < len(ctx.speech_blocks):
        blk_start_t, blk_end_t = whisper_utils._block_time_range(
            ctx.current_block, ctx.speech_blocks, ctx.all_words
        )
        window_start = max(window_start, blk_start_t - 0.25)
        window_end = min(window_end, blk_end_t + 0.25)
        gap_min_idx = min(
            ctx.next_word_idx_start, ctx.speech_blocks[ctx.current_block][0]
        )
        return window_start, window_end, gap_min_idx
    return window_start, window_end, ctx.next_word_idx_start


def _build_gap_candidates(
    *,
    ctx: _LineMappingContext,
    candidate_indices: Set[int],
    gap_min_idx: int,
    gap_start: float,
    window_end: float,
) -> list[tuple[Any, int]]:
    sorted_indices = [
        idx
        for idx in sorted(candidate_indices, key=lambda idx: ctx.all_words[idx].start)
        if idx >= gap_min_idx and idx not in ctx.used_word_indices
    ]
    filtered_indices = [
        idx
        for idx in sorted_indices
        if gap_start <= ctx.all_words[idx].start <= window_end
    ]
    return [(ctx.all_words[idx], idx) for idx in filtered_indices]


def _restrict_gap_candidates_to_blocks(
    ctx: _LineMappingContext,
    whisper_candidates: list[tuple[Any, int]],
) -> list[tuple[Any, int]]:
    if not (ctx.speech_blocks and whisper_candidates):
        return whisper_candidates
    cur_blk = ctx.current_block
    in_block = [
        (w, idx)
        for w, idx in whisper_candidates
        if whisper_utils._word_idx_to_block(idx, ctx.speech_blocks) == cur_blk
    ]
    if in_block:
        return in_block
    in_next = [
        (w, idx)
        for w, idx in whisper_candidates
        if whisper_utils._word_idx_to_block(idx, ctx.speech_blocks) == cur_blk + 1
    ]
    if in_next:
        return in_next
    return whisper_candidates
