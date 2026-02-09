"""Whisper-based mapping of LRC words to transcription timings."""

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, Dict, Any, Set, Sequence, Iterable

from ..utils.logging import get_logger
from . import models
from . import timing_models
from . import phonetic_utils
from . import whisper_utils
from .whisper_dtw import _LineMappingContext

logger = get_logger(__name__)

_SPEECH_BLOCK_GAP = 5.0
_TIME_DRIFT_THRESHOLD = 0.8
_SEGMENT_TIME_WEIGHT = 0.18
_SEGMENT_INDEX_WEIGHT = 0.01
_SEGMENT_MAX_TIME_DIFF = 5.0
_SEGMENT_MIN_TEXT_SCORE = 0.25
_MAX_SEGMENT_LOOKAHEAD = 4.0
_ORDER_POSITION_WEIGHT = 0.35
_BACKWARD_START_PENALTY = 0.5
_MIN_DUPLICATE_SEGMENT_DURATION = 0.25


def _dedupe_whisper_words(
    words: List[timing_models.TranscriptionWord],
) -> List[timing_models.TranscriptionWord]:
    """Remove Whisper words that share the same start time and text."""
    deduped: List[timing_models.TranscriptionWord] = []
    seen: Set[Tuple[int, str]] = set()
    for word in words:
        key = (round(word.start * 1000), word.text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(word)
    return deduped


def _dedupe_whisper_segments(
    segments: List[timing_models.TranscriptionSegment],
) -> List[timing_models.TranscriptionSegment]:
    if not segments:
        return segments
    cleaned: List[timing_models.TranscriptionSegment] = []
    for seg in segments:
        duration = whisper_utils._segment_end(seg) - whisper_utils._segment_start(seg)
        norm_text = _normalize_line_text(whisper_utils._get_segment_text(seg))
        if cleaned:
            prev = cleaned[-1]
            prev_norm = _normalize_line_text(whisper_utils._get_segment_text(prev))
            if (
                duration <= _MIN_DUPLICATE_SEGMENT_DURATION
                and norm_text
                and prev_norm == norm_text
            ):
                prev.end = max(prev.end, whisper_utils._segment_end(seg))
                continue
        cleaned.append(seg)
    return cleaned


def _build_word_to_segment_index(
    all_words: List[timing_models.TranscriptionWord], segments: Sequence[Any]
) -> Dict[int, int]:
    """Map each Whisper word index to its enclosing segment."""
    mapping: Dict[int, int] = {}
    if not segments:
        return mapping
    seg_idx = 0
    for idx, word in enumerate(all_words):
        while seg_idx + 1 < len(segments) and word.start > whisper_utils._segment_end(
            segments[seg_idx]
        ):
            seg_idx += 1
        seg = segments[seg_idx]
        if (
            whisper_utils._segment_start(seg)
            <= word.start
            <= whisper_utils._segment_end(seg)
        ):
            mapping[idx] = seg_idx
    return mapping


def _find_segment_for_time(
    time: float,
    segments: Sequence[Any],
    start_idx: int = 0,
    excluded_segments: Optional[Set[int]] = None,
) -> Optional[int]:
    """Find the segment index whose start is closest to the given time."""
    if not segments:
        return None
    best_idx = None
    best_diff = float("inf")
    excluded_segments = excluded_segments or set()
    for idx in range(start_idx, len(segments)):
        if idx in excluded_segments:
            continue
        seg = segments[idx]
        diff = abs(whisper_utils._segment_start(seg) - time)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
        if whisper_utils._segment_start(seg) > time and diff > best_diff:
            break
    return best_idx


def _word_match_score(
    word_start: float,
    target_start: float,
    word_seg: Optional[int],
    target_seg: Optional[int],
    segments: Sequence[Any],
    line_anchor: Optional[float] = None,
    lrc_idx: Optional[int] = None,
    candidate_idx: Optional[int] = None,
    total_lrc_words: Optional[int] = None,
    total_whisper_words: Optional[int] = None,
) -> float:
    """Score a word match by time difference with a small segment penalty."""
    score = abs(word_start - target_start)
    if target_seg is None or word_seg is None:
        if line_anchor is not None and word_start < line_anchor:
            score += (line_anchor - word_start) * _BACKWARD_START_PENALTY
        return score
    if word_seg != target_seg:
        seg_diff = abs(
            whisper_utils._segment_start(segments[word_seg])
            - whisper_utils._segment_start(segments[target_seg])
        )
        score += 0.4 + 0.01 * seg_diff
    if line_anchor is not None and word_start < line_anchor:
        score += (line_anchor - word_start) * _BACKWARD_START_PENALTY

    if (
        lrc_idx is not None
        and candidate_idx is not None
        and total_lrc_words
        and total_whisper_words
    ):
        lrc_pct = lrc_idx / max(1, total_lrc_words - 1)
        cand_pct = candidate_idx / max(1, total_whisper_words - 1)
        score += abs(lrc_pct - cand_pct) * _ORDER_POSITION_WEIGHT

    return score


def _find_nearest_word_in_segment(
    candidates: Iterable[Tuple[timing_models.TranscriptionWord, int]],
    target_start: float,
    segment_idx: Optional[int],
    word_segment_idx: Dict[int, int],
) -> Optional[Tuple[timing_models.TranscriptionWord, int]]:
    if segment_idx is None:
        return None
    best = None
    best_diff = float("inf")
    for word, idx in candidates:
        if word_segment_idx.get(idx) != segment_idx:
            continue
        diff = abs(word.start - target_start)
        if diff < best_diff:
            best_diff = diff
            best = (word, idx)
    return best


def _normalize_line_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def _trim_whisper_transcription_by_lyrics(
    segments: List[timing_models.TranscriptionSegment],
    words: List[timing_models.TranscriptionWord],
    lyric_texts: List[str],
    min_similarity: float = 0.45,
) -> Tuple[
    List[timing_models.TranscriptionSegment],
    List[timing_models.TranscriptionWord],
    Optional[float],
]:
    """Keep only the portion of the Whisper transcript that matches the provided lyrics."""
    normalized_lyrics = [
        _normalize_line_text(text) for text in lyric_texts if text.strip()
    ]
    if not normalized_lyrics or not segments:
        return segments, words, None

    last_match_end: Optional[float] = None
    for seg in reversed(segments):
        seg_norm = _normalize_line_text(seg.text)
        if not seg_norm:
            continue
        best_ratio = max(
            SequenceMatcher(None, seg_norm, lyric).ratio()
            for lyric in normalized_lyrics
        )
        if best_ratio >= min_similarity:
            last_match_end = seg.end
            break

    if last_match_end is None:
        return segments, words, None

    trimmed_segments = [seg for seg in segments if seg.end <= last_match_end]
    trimmed_words = [word for word in words if word.start <= last_match_end]
    return trimmed_segments, trimmed_words, last_match_end


def _choose_segment_for_line(
    line: models.Line,
    segments: Sequence[Any],
    start_idx: int = 0,
    min_start: float = 0.0,
    excluded_segments: Optional[Set[int]] = None,
) -> Optional[int]:
    """Greedily pick the best future segment that matches the line text."""
    if not segments:
        return None
    norm_line = _normalize_line_text(line.text)
    best_idx = None
    best_score = float("-inf")
    excluded_segments = excluded_segments or set()
    for idx in range(start_idx, len(segments)):
        if idx in excluded_segments:
            continue
        segment = segments[idx]
        seg_text = whisper_utils._get_segment_text(segment)
        if not seg_text:
            continue
        norm_seg = _normalize_line_text(seg_text)
        text_score = SequenceMatcher(None, norm_line, norm_seg).ratio()
        if text_score < _SEGMENT_MIN_TEXT_SCORE:
            continue
        seg_start = whisper_utils._segment_start(segment)
        if seg_start < min_start:
            continue
        if seg_start - line.start_time > _MAX_SEGMENT_LOOKAHEAD:
            break
        if seg_start < min_start:
            continue
        time_diff = abs(seg_start - line.start_time)
        time_penalty = min(time_diff, _SEGMENT_MAX_TIME_DIFF) * _SEGMENT_TIME_WEIGHT
        index_penalty = (idx - start_idx) * _SEGMENT_INDEX_WEIGHT
        score = text_score - time_penalty - index_penalty
        if score > best_score:
            best_score = score
            best_idx = idx
        if time_diff > _SEGMENT_MAX_TIME_DIFF and text_score < 0.5:
            break
    return best_idx


def _segment_word_indices(
    all_words: List[timing_models.TranscriptionWord],
    word_segment_idx: Dict[int, int],
    segment_idx: int,
) -> List[int]:
    """Return word indices belonging to the given Whisper segment."""
    indices = [idx for idx, seg in word_segment_idx.items() if seg == segment_idx]
    return sorted(indices, key=lambda i: all_words[i].start)


def _collect_unused_words_near_line(
    all_words: List[timing_models.TranscriptionWord],
    line: models.Line,
    used_indices: Set[int],
    start_idx: int,
    min_start: float = 0.0,
    lookahead: float = 1.0,
) -> List[int]:
    """Return unused Whisper word indices near the given line timing."""
    window_start = max(min_start, line.start_time - 0.5)
    window_end = line.end_time + lookahead
    collected: List[int] = []
    for idx in range(start_idx, len(all_words)):
        if idx in used_indices:
            continue
        word = all_words[idx]
        if word.start < window_start:
            continue
        if word.start > window_end:
            break
        collected.append(idx)
        if len(collected) >= max(1, len(line.words)):
            break
    return collected


def _collect_unused_words_in_window(
    all_words: List[timing_models.TranscriptionWord],
    used_indices: Set[int],
    start_idx: int,
    window_start: float,
    window_end: float,
) -> List[int]:
    """Return unused Whisper word indices within the specified time window."""
    if window_end <= window_start:
        return []
    collected: List[int] = []
    for idx in range(start_idx, len(all_words)):
        if idx in used_indices:
            continue
        word = all_words[idx]
        if word.start < window_start:
            continue
        if word.start > window_end:
            break
        collected.append(idx)
    return collected


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
    # Advance speech block tracker when a match lands in a later block
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
    """Sort candidates by time, filter used, and prefer ordered ones.

    Speech-block awareness: candidates in the current block are preferred.
    A candidate from a later block is only accepted when the current block
    has no viable candidates left.
    """
    # When speech blocks are available, relax the forward-only constraint for
    # candidates in the current block.  The constraint still applies across
    # block boundaries to prevent jumping backwards to a previous block.
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

    # --- speech-block filtering ---
    if ctx.speech_blocks and whisper_candidates:
        in_block = [
            (w, idx)
            for w, idx in whisper_candidates
            if whisper_utils._word_idx_to_block(idx, ctx.speech_blocks) == cur_blk
        ]
        if in_block:
            whisper_candidates = in_block
        else:
            # Current block exhausted – allow next block only (no skipping)
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
    """Determine segment, anchor time, and shift for a line."""
    line_segment = _choose_segment_for_line(
        line,
        ctx.segments,
        ctx.current_segment,
        min_start=ctx.last_line_start,
        excluded_segments=ctx.used_segments,
    )
    if line_segment is None:
        line_segment = _find_segment_for_time(
            line.start_time,
            ctx.segments,
            ctx.current_segment,
            excluded_segments=ctx.used_segments,
        )
    if (
        line_segment is not None
        and ctx.segments
        and whisper_utils._segment_start(ctx.segments[line_segment])
        < ctx.last_line_start
    ):
        line_segment = None
    line_anchor_time = max(line.start_time, ctx.last_line_start, ctx.prev_line_end)
    line_shift = line_anchor_time - line.start_time
    return line_segment, line_anchor_time, line_shift


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
        # Clip window to current speech block so we don't reach across silence
        if ctx.speech_blocks and ctx.current_block < len(ctx.speech_blocks):
            blk_start_t, blk_end_t = whisper_utils._block_time_range(
                ctx.current_block, ctx.speech_blocks, ctx.all_words
            )
            window_start = max(window_start, blk_start_t - 0.25)
            window_end = min(window_end, blk_end_t + 0.25)
        # Use block-aware min index (same as _filter_and_order_candidates)
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
        # Apply speech-block preference (same logic as _filter_and_order_candidates)
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


def _assemble_mapped_line(
    ctx: _LineMappingContext,
    line_idx: int,
    line: "models.Line",
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    line_anchor_time: float,
    line_segment: Optional[int],
    line_last_idx_ref: List[Optional[int]],
) -> "models.Line":
    """Build a mapped models.Line from match intervals and update tracking state."""
    from .models import Line as LineModel

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
    else:
        original_duration = line.end_time - line.start_time
        if original_duration <= 0:
            original_duration = len(line.words) * 0.6
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

        # Override line_segment with the segment determined by the
        # text-overlap assignment.  _prepare_line_context uses the
        # line's original (placeholder) timing which may be far from
        # the correct position; the assignments already encode the
        # right segment.
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
            if line_segment != override_seg:
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
        )
        mapped_lines.append(mapped_line)

    return mapped_lines, ctx.mapped_count, ctx.total_similarity, ctx.mapped_lines_set


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
    """Ensure repeated lines reserve later Whisper words when they reappear."""

    adjusted_lines: List[models.Line] = []
    last_idx_by_text: Dict[str, int] = {}
    last_end_time: Dict[str, float] = {}

    for line in mapped_lines:
        if not line.words:
            adjusted_lines.append(line)
            continue

        text_norm = line.text.strip().lower() if getattr(line, "text", "") else ""
        prev_idx = last_idx_by_text.get(text_norm)
        prev_end = last_end_time.get(text_norm)
        assigned_end_idx: Optional[int] = None

        if prev_idx is not None and prev_end is not None:
            required_time = max(prev_end + 0.4, line.start_time)
            start_idx = next(
                (
                    idx
                    for idx, ww in enumerate(all_words)
                    if idx > prev_idx and ww.start >= required_time
                ),
                None,
            )
            if start_idx is None:
                start_idx = next(
                    (idx for idx, _ww in enumerate(all_words) if idx > prev_idx),
                    None,
                )
            if start_idx is not None and (all_words[start_idx].start - prev_end > 10.0):
                start_idx = None
            if start_idx is not None:
                adjusted_words: List[models.Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words.append(
                        models.Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = models.Line(words=adjusted_words, singer=line.singer)
                assigned_end_idx = min(
                    start_idx + len(line.words) - 1, len(all_words) - 1
                )

        adjusted_lines.append(line)
        if line.words:
            if assigned_end_idx is None:
                assigned_end_idx = next(
                    (
                        idx
                        for idx, ww in enumerate(all_words)
                        if abs(ww.start - line.words[-1].start_time) < 0.05
                    ),
                    len(all_words) - 1,
                )
            last_idx_by_text[text_norm] = assigned_end_idx
            last_end_time[text_norm] = line.end_time

    return adjusted_lines


def _enforce_monotonic_line_starts_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
) -> List[models.Line]:
    """Ensure line starts are monotonic by shifting backwards lines forward."""

    prev_start = None
    prev_end = None
    monotonic_lines: List[models.Line] = []
    for line in mapped_lines:
        if not line.words:
            monotonic_lines.append(line)
            continue

        if prev_start is not None and line.start_time < prev_start:
            required_time = (prev_end or line.start_time) + 0.01
            start_idx = next(
                (idx for idx, ww in enumerate(all_words) if ww.start >= required_time),
                None,
            )
            if start_idx is not None and (
                all_words[start_idx].start - required_time <= 10.0
            ):
                adjusted_words_2: List[models.Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words_2.append(
                        models.Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = models.Line(words=adjusted_words_2, singer=line.singer)
            else:
                shift = required_time - line.start_time
                shifted_words: List[models.Word] = [
                    models.Word(
                        text=w.text,
                        start_time=w.start_time + shift,
                        end_time=w.end_time + shift,
                        singer=w.singer,
                    )
                    for w in line.words
                ]
                line = models.Line(words=shifted_words, singer=line.singer)

        monotonic_lines.append(line)
        if line.words:
            prev_start = line.start_time
            prev_end = line.end_time

    return monotonic_lines


def _resolve_line_overlaps(lines: List[models.Line]) -> List[models.Line]:
    """Ensure consecutive lines never overlap in time."""
    from .models import Line as LineModel

    resolved: List[models.Line] = list(lines)
    for i in range(len(resolved) - 1):
        cur = resolved[i]
        nxt = resolved[i + 1]
        if not cur.words or not nxt.words:
            continue
        if cur.end_time > nxt.start_time:
            gap_point = nxt.start_time
            new_words: List[models.Word] = []
            for w in cur.words:
                if w.start_time >= gap_point:
                    new_words.append(
                        models.Word(
                            text=w.text,
                            start_time=gap_point - 0.01,
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                elif w.end_time > gap_point:
                    new_words.append(
                        models.Word(
                            text=w.text,
                            start_time=w.start_time,
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                else:
                    new_words.append(w)
            resolved[i] = LineModel(words=new_words, singer=cur.singer)
    return resolved
