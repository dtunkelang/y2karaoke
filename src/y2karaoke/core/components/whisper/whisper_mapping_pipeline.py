"""Core mapping pipeline from LRC words to Whisper words."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ....utils.logging import get_logger
from ... import models, phonetic_utils
from ..alignment import timing_models
from . import whisper_utils
from .whisper_dtw import _LineMappingContext
from . import whisper_mapping_pipeline_line_context as _line_context_helpers
from .whisper_mapping_helpers import (
    _SPEECH_BLOCK_GAP,
    _build_word_to_segment_index,
    _collect_unused_words_in_window,
    _collect_unused_words_near_line,
    _find_nearest_word_in_segment,
    _segment_word_indices,
    _word_match_score,
)

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
    """Register a word match and update tracking state."""
    start, end = best_word.start, best_word.end
    line_matches.append((word_idx, (start, end)))
    line_match_intervals[word_idx] = (start, end)
    ctx.mapped_count += 1
    ctx.mapped_lines_set.add(line_idx)
    best_sim = max(
        phonetic_utils._phonetic_similarity(word.text, ww.text, ctx.language)
        for ww, _ in candidates
    )
    ctx.total_similarity += best_sim
    ctx.used_word_indices.add(best_idx)
    best_seg = ctx.word_segment_idx.get(best_idx)
    if best_seg is not None:
        ctx.current_segment = max(ctx.current_segment, best_seg)
    elif line_segment is not None:
        ctx.current_segment = max(ctx.current_segment, line_segment)
    if ctx.speech_blocks:
        match_blk = whisper_utils._word_idx_to_block(best_idx, ctx.speech_blocks)
        if match_blk > ctx.current_block:
            ctx.current_block = match_blk
    if line_last_idx_ref[0] is None or best_idx > line_last_idx_ref[0]:
        line_last_idx_ref[0] = best_idx


def _select_best_candidate(
    ctx: _LineMappingContext,
    whisper_candidates: List[Tuple[timing_models.TranscriptionWord, int]],
    word: "models.Word",
    line_shift: float,
    line_segment: Optional[int],
    line_anchor_time: float,
    lrc_idx_opt: Optional[int],
) -> Tuple[timing_models.TranscriptionWord, int]:
    """Score candidates, pick the best, and apply drift fallback."""
    best_word, best_idx = min(
        whisper_candidates,
        key=lambda pair: _word_match_score(
            pair[0].start,
            word.start_time + line_shift,
            ctx.word_segment_idx.get(pair[1]),
            line_segment,
            ctx.segments,
            line_anchor_time,
            lrc_idx=lrc_idx_opt,
            candidate_idx=pair[1],
            total_lrc_words=ctx.total_lrc_words,
            total_whisper_words=ctx.total_whisper_words,
        ),
    )
    drift = abs(best_word.start - word.start_time)
    if drift > _TIME_DRIFT_THRESHOLD:
        fallback = _find_nearest_word_in_segment(
            whisper_candidates,
            word.start_time,
            line_segment,
            ctx.word_segment_idx,
        )
        if fallback and abs(fallback[0].start - word.start_time) < drift:
            best_word, best_idx = fallback
    return best_word, best_idx


def _filter_and_order_candidates(
    ctx: _LineMappingContext,
    candidate_indices: Set[int],
) -> List[Tuple[timing_models.TranscriptionWord, int]]:
    """Sort candidates by time, filter used, and prefer ordered ones."""
    if ctx.speech_blocks:
        cur_blk = ctx.current_block
        cur_blk_start = (
            ctx.speech_blocks[cur_blk][0] if cur_blk < len(ctx.speech_blocks) else 0
        )
        min_idx = min(ctx.next_word_idx_start, cur_blk_start)
    else:
        min_idx = ctx.next_word_idx_start

    sorted_indices = [
        idx
        for idx in sorted(candidate_indices, key=lambda idx: ctx.all_words[idx].start)
        if idx >= min_idx and idx not in ctx.used_word_indices
    ]
    whisper_candidates = [(ctx.all_words[i], i) for i in sorted_indices]
    ordered = [
        pair for pair in whisper_candidates if pair[0].start >= ctx.last_line_start
    ]
    if ordered:
        whisper_candidates = ordered

    if ctx.speech_blocks and whisper_candidates:
        in_block = [
            (w, idx)
            for w, idx in whisper_candidates
            if whisper_utils._word_idx_to_block(idx, ctx.speech_blocks) == cur_blk
        ]
        if in_block:
            whisper_candidates = in_block
        else:
            next_blk = cur_blk + 1
            in_next = [
                (w, idx)
                for w, idx in whisper_candidates
                if whisper_utils._word_idx_to_block(idx, ctx.speech_blocks) == next_blk
            ]
            if in_next:
                whisper_candidates = in_next

    return whisper_candidates


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
    """Phase 1: match words using DTW assignments."""
    for word_idx, word in enumerate(line.words):
        lrc_idx_opt = lrc_index_by_loc.get((line_idx, word_idx))
        if lrc_idx_opt is None:
            continue
        assigned = lrc_assignments.get(lrc_idx_opt)
        if not assigned:
            continue
        candidate_indices = set(assigned)
        if line_segment is not None:
            candidate_indices.update(
                _segment_word_indices(ctx.all_words, ctx.word_segment_idx, line_segment)
            )
        whisper_candidates = _filter_and_order_candidates(ctx, candidate_indices)
        if not whisper_candidates:
            fallback_indices = _collect_unused_words_near_line(
                ctx.all_words,
                line,
                ctx.used_word_indices,
                ctx.next_word_idx_start,
                ctx.prev_line_end,
            )
            whisper_candidates = _filter_and_order_candidates(
                ctx, set(fallback_indices)
            )
        if not whisper_candidates:
            continue
        best_word, best_idx = _select_best_candidate(
            ctx,
            whisper_candidates,
            word,
            line_shift,
            line_segment,
            line_anchor_time,
            lrc_idx_opt,
        )
        _register_word_match(
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
    """Phase 2: fill unmatched words using time-window search."""
    unmatched = [
        idx for idx in range(len(line.words)) if idx not in line_match_intervals
    ]
    for word_idx in unmatched:
        word = line.words[word_idx]
        gap_start, gap_end = _compute_gap_window(
            line, word_idx, line_match_intervals, ctx.prev_line_end
        )
        if gap_end <= gap_start:
            continue
        lrc_idx_opt = lrc_index_by_loc.get((line_idx, word_idx))
        assigned = (
            lrc_assignments.get(lrc_idx_opt, []) if lrc_idx_opt is not None else []
        )
        candidate_indices = set(assigned)
        if line_segment is not None:
            candidate_indices.update(
                _segment_word_indices(ctx.all_words, ctx.word_segment_idx, line_segment)
            )
        window_start = max(gap_start - 0.25, 0.0)
        window_end = gap_end + 0.25
        if ctx.speech_blocks and ctx.current_block < len(ctx.speech_blocks):
            blk_start_t, blk_end_t = whisper_utils._block_time_range(
                ctx.current_block, ctx.speech_blocks, ctx.all_words
            )
            window_start = max(window_start, blk_start_t - 0.25)
            window_end = min(window_end, blk_end_t + 0.25)
        if ctx.speech_blocks and ctx.current_block < len(ctx.speech_blocks):
            gap_min_idx = min(
                ctx.next_word_idx_start,
                ctx.speech_blocks[ctx.current_block][0],
            )
        else:
            gap_min_idx = ctx.next_word_idx_start
        window_candidates = _collect_unused_words_in_window(
            ctx.all_words,
            ctx.used_word_indices,
            gap_min_idx,
            window_start,
            window_end,
        )
        candidate_indices.update(window_candidates)
        sorted_indices = [
            idx
            for idx in sorted(
                candidate_indices,
                key=lambda idx: ctx.all_words[idx].start,
            )
            if idx >= gap_min_idx and idx not in ctx.used_word_indices
        ]
        filtered_indices = [
            idx
            for idx in sorted_indices
            if ctx.all_words[idx].start >= gap_start
            and ctx.all_words[idx].start <= window_end
        ]
        whisper_candidates = [(ctx.all_words[idx], idx) for idx in filtered_indices]
        ordered = [
            pair for pair in whisper_candidates if pair[0].start >= ctx.last_line_start
        ]
        if ordered:
            whisper_candidates = ordered
        if ctx.speech_blocks and whisper_candidates:
            cur_blk = ctx.current_block
            in_block = [
                (w, idx)
                for w, idx in whisper_candidates
                if whisper_utils._word_idx_to_block(idx, ctx.speech_blocks) == cur_blk
            ]
            if in_block:
                whisper_candidates = in_block
            else:
                in_next = [
                    (w, idx)
                    for w, idx in whisper_candidates
                    if whisper_utils._word_idx_to_block(idx, ctx.speech_blocks)
                    == cur_blk + 1
                ]
                if in_next:
                    whisper_candidates = in_next
        if not whisper_candidates:
            continue
        best_word, best_idx = _select_best_candidate(
            ctx,
            whisper_candidates,
            word,
            line_shift,
            line_segment,
            line_anchor_time,
            lrc_idx_opt,
        )
        _register_word_match(
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
    """Build a mapped models.Line from match intervals and update tracking state."""
    from ...models import Line as LineModel

    mapped_words: List[models.Word] = []
    for word_idx, word in enumerate(line.words):
        interval = line_match_intervals.get(word_idx)
        if interval:
            start, end = interval
        else:
            start, end = word.start_time, word.end_time
        mapped_words.append(
            models.Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )

    mapped_line = LineModel(words=mapped_words, singer=line.singer)
    if line_matches:
        actual_start = min(iv[0] for _, iv in line_matches)
        actual_end = max(iv[1] for _, iv in line_matches)
        actual_start, actual_end = _clamp_match_window_to_anchor(
            actual_start,
            actual_end,
            line_anchor_time,
        )
    else:
        original_duration = _fallback_unmatched_line_duration(line)
        actual_start = line_anchor_time
        actual_end = actual_start + original_duration
    target_duration = max(actual_end - actual_start, 0.0)
    mapped_line = whisper_utils._redistribute_word_timings_to_line(
        mapped_line,
        line_matches,
        target_duration=target_duration,
        min_word_duration=0.05,
        line_start=actual_start,
    )
    mapped_line = _clamp_line_shift_vs_original(mapped_line, line)
    mapped_line = _clamp_line_duration_vs_original(
        mapped_line, line, next_original_start
    )
    logger.debug(
        "Mapped line %d start=%.2f end=%.2f matches=%d",
        line_idx + 1,
        mapped_line.start_time,
        mapped_line.end_time,
        len(line_matches),
    )
    if mapped_line.words:
        ctx.last_line_start = mapped_line.start_time
        ctx.prev_line_end = mapped_line.end_time
    if line_segment is not None:
        ctx.used_segments.add(line_segment)
    if line_last_idx_ref[0] is not None:
        ctx.next_word_idx_start = line_last_idx_ref[0] + 1
    return mapped_line


def _map_lrc_words_to_whisper(
    lines: List[models.Line],
    lrc_words: List[Dict],
    all_words: List[timing_models.TranscriptionWord],
    lrc_assignments: Dict[int, List[int]],
    language: str,
    segments: Sequence[Any],
) -> Tuple[List[models.Line], int, float, set]:
    """Build mapped lines using Whisper timings based on LRC assignments."""
    segments = segments or []
    speech_blocks = whisper_utils._compute_speech_blocks(all_words)
    if speech_blocks:
        logger.debug(
            "Speech blocks: %d (gaps >= %.1fs)",
            len(speech_blocks),
            _SPEECH_BLOCK_GAP,
        )
    ctx = _LineMappingContext(
        all_words=all_words,
        segments=segments,
        word_segment_idx=_build_word_to_segment_index(all_words, segments),
        language=language,
        total_lrc_words=len(lrc_words),
        total_whisper_words=len(all_words),
        speech_blocks=speech_blocks,
    )
    lrc_index_by_loc = {
        (lw["line_idx"], lw["word_idx"]): idx for idx, lw in enumerate(lrc_words)
    }
    mapped_lines: List[models.Line] = []

    for line_idx, line in enumerate(lines):
        if not line.words:
            mapped_lines.append(line)
            continue

        line_segment, line_anchor_time, line_shift = _prepare_line_context(ctx, line)

        assigned_segs: Dict[int, int] = {}
        for word_idx in range(len(line.words)):
            lrc_idx = lrc_index_by_loc.get((line_idx, word_idx))
            if lrc_idx is not None:
                for wi in lrc_assignments.get(lrc_idx, []):
                    si = ctx.word_segment_idx.get(wi)
                    if si is not None:
                        assigned_segs[si] = assigned_segs.get(si, 0) + 1
        if assigned_segs:
            override_seg = max(assigned_segs, key=assigned_segs.get)  # type: ignore[arg-type]
            if line_segment != override_seg and _should_override_line_segment(
                current_segment=line_segment,
                override_segment=override_seg,
                override_hits=assigned_segs[override_seg],
                line_word_count=len(line.words),
                line_anchor_time=line_anchor_time,
                segments=ctx.segments,
            ):
                line_segment = override_seg
                if ctx.segments and override_seg < len(ctx.segments):
                    seg_start = whisper_utils._segment_start(ctx.segments[override_seg])
                    line_anchor_time = max(seg_start, ctx.prev_line_end)
                    line_shift = line_anchor_time - line.start_time

        line_matches: List[Tuple[int, Tuple[float, float]]] = []
        line_match_intervals: Dict[int, Tuple[float, float]] = {}
        line_last_idx_ref: List[Optional[int]] = [None]

        _match_assigned_words(
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
        )
        _fill_unmatched_gaps(
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
        )
        mapped_line = _assemble_mapped_line(
            ctx,
            line_idx,
            line,
            line_matches,
            line_match_intervals,
            line_anchor_time,
            line_segment,
            line_last_idx_ref,
            lines[line_idx + 1].start_time if line_idx + 1 < len(lines) else None,
        )
        mapped_lines.append(mapped_line)

    return mapped_lines, ctx.mapped_count, ctx.total_similarity, ctx.mapped_lines_set
