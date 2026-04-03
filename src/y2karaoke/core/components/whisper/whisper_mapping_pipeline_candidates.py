"""Candidate scoring, ordering, and registration helpers for mapping pipeline."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Set, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_dtw import _LineMappingContext
from . import whisper_utils
from .whisper_mapping_helpers import _find_nearest_word_in_segment, _word_match_score
from .whisper_mapping_post_text import (
    _is_placeholder_whisper_token,
    _normalize_match_token,
    _soft_token_match,
)
from .whisper_mapping_runtime_config import load_whisper_mapping_trace_config

_MAPPER_CANDIDATE_TRACE_ROWS: list[dict[str, object]] = []


def _append_candidate_trace_row(row: dict[str, object]) -> None:
    trace_path = load_whisper_mapping_trace_config().mapper_candidates_path
    if not trace_path:
        return
    _MAPPER_CANDIDATE_TRACE_ROWS.append(row)
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": _MAPPER_CANDIDATE_TRACE_ROWS}, fh, indent=2)


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
    line_match_word_indices: Dict[int, int],
    word_idx: int,
    line_last_idx_ref: List[Optional[int]],
    *,
    phonetic_similarity_fn: Callable[[str, str, str], float],
) -> None:
    """Register a word match and update tracking state."""
    start, end = best_word.start, best_word.end
    line_matches.append((word_idx, (start, end)))
    line_match_intervals[word_idx] = (start, end)
    line_match_word_indices[word_idx] = best_idx
    ctx.mapped_count += 1
    ctx.mapped_lines_set.add(line_idx)
    best_sim = max(
        phonetic_similarity_fn(word.text, ww.text, ctx.language) for ww, _ in candidates
    )
    ctx.total_similarity += best_sim
    ctx.used_word_indices.add(best_idx)
    best_seg = ctx.word_segment_idx.get(best_idx)
    segment_updates = [ctx.current_segment]
    if best_seg is not None:
        segment_updates.append(best_seg)
    if line_segment is not None:
        segment_updates.append(line_segment)
    ctx.current_segment = max(segment_updates)
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
    prior_matched_word_idx: Optional[int] = None,
    line_word_count: int = 0,
    *,
    trace_context: Optional[dict[str, object]] = None,
    time_drift_threshold: float,
    phonetic_similarity_fn: Callable[[str, str, str], float],
) -> Tuple[timing_models.TranscriptionWord, int]:
    """Score candidates, pick the best, and apply drift fallback."""
    target_token = _normalize_match_token(word.text)
    plausible_candidates = [
        pair
        for pair in whisper_candidates
        if _candidate_is_lexically_plausible(
            target_token=target_token,
            candidate_text=pair[0].text,
            phonetic_similarity_fn=phonetic_similarity_fn,
            target_text=word.text,
            language=ctx.language,
        )
    ]
    candidate_pool = plausible_candidates or whisper_candidates
    monotonic_backtrack_penalty = _dense_line_backtrack_penalty(
        prior_matched_word_idx=prior_matched_word_idx,
        line_word_count=line_word_count,
        candidate_pool_size=len(candidate_pool),
    )
    scored_candidates = [
        (
            pair[0],
            pair[1],
            _word_match_score(
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
            )
            + monotonic_backtrack_penalty(pair[1]),
        )
        for pair in candidate_pool
    ]
    best_word, best_idx, best_score = min(scored_candidates, key=lambda row: row[2])
    drift = abs(best_word.start - word.start_time)
    fallback_used = False
    if drift > time_drift_threshold:
        fallback = _find_nearest_word_in_segment(
            candidate_pool,
            word.start_time,
            line_segment,
            ctx.word_segment_idx,
        )
        if fallback and abs(fallback[0].start - word.start_time) < drift:
            best_word, best_idx = fallback
            fallback_used = True
            best_score = _word_match_score(
                best_word.start,
                word.start_time + line_shift,
                ctx.word_segment_idx.get(best_idx),
                line_segment,
                ctx.segments,
                line_anchor_time,
                lrc_idx=lrc_idx_opt,
                candidate_idx=best_idx,
                total_lrc_words=ctx.total_lrc_words,
                total_whisper_words=ctx.total_whisper_words,
            )
    _append_candidate_trace_row(
        {
            "target_word": word.text,
            "target_start": round(word.start_time, 3),
            "target_token": target_token,
            "line_shift": round(line_shift, 3),
            "line_anchor_time": round(line_anchor_time, 3),
            "line_segment": line_segment,
            "lrc_index": lrc_idx_opt,
            "prior_matched_word_idx": prior_matched_word_idx,
            "trace_context": trace_context or {},
            "lexical_candidate_count": len(plausible_candidates),
            "chosen_index": best_idx,
            "chosen_word": best_word.text,
            "chosen_start": round(best_word.start, 3),
            "chosen_score": round(best_score, 4),
            "fallback_used": fallback_used,
            "candidates": [
                {
                    "index": idx,
                    "word": candidate_word.text,
                    "start": round(candidate_word.start, 3),
                    "score": round(score, 4),
                    "segment": ctx.word_segment_idx.get(idx),
                }
                for candidate_word, idx, score in scored_candidates[:12]
            ],
        }
    )
    return best_word, best_idx


def _dense_line_backtrack_penalty(
    *,
    prior_matched_word_idx: Optional[int],
    line_word_count: int,
    candidate_pool_size: int,
    min_line_words: int = 8,
    min_candidate_pool_size: int = 12,
    grace_words: int = 1,
    base_penalty: float = 0.55,
    step_penalty: float = 0.14,
) -> Callable[[int], float]:
    if (
        prior_matched_word_idx is None
        or line_word_count < min_line_words
        or candidate_pool_size < min_candidate_pool_size
    ):
        return lambda _candidate_idx: 0.0

    def _penalty(candidate_idx: int) -> float:
        if candidate_idx >= prior_matched_word_idx - grace_words:
            return 0.0
        backtrack = (prior_matched_word_idx - grace_words) - candidate_idx
        return base_penalty + backtrack * step_penalty

    return _penalty


def _candidate_is_lexically_plausible(
    *,
    target_token: str,
    candidate_text: str,
    phonetic_similarity_fn: Callable[[str, str, str], float],
    target_text: str,
    language: str,
    min_phonetic_similarity: float = 0.55,
) -> bool:
    if not target_token:
        return True
    if _is_placeholder_whisper_token(candidate_text):
        return False
    candidate_token = _normalize_match_token(candidate_text)
    if not candidate_token:
        return False
    if _soft_token_match(target_token, candidate_token):
        return True
    return (
        phonetic_similarity_fn(target_text, candidate_text, language)
        >= min_phonetic_similarity
    )


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
