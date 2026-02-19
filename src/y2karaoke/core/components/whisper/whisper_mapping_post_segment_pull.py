"""Segment-pull retiming for Whisper-mapped lyric lines."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_mapping_post_repetition import (
    _pull_adjacent_similar_lines_across_long_gaps,
    _realign_repetitive_runs_to_matching_segments,
    _rebalance_short_question_pairs,
    _retime_repetitive_question_runs_to_segment_windows,
    _smooth_adjacent_duplicate_line_cadence,
)
from .whisper_mapping_post_text import (
    _contains_token_sequence,
    _light_text_similarity,
    _max_contiguous_soft_match_run,
    _normalize_match_token,
    _normalize_text_tokens,
    _overlap_suffix_prefix,
    _soft_text_similarity,
    _soft_token_overlap_ratio,
)


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
    """Pull late line starts toward matching Whisper segments within neighbor bounds."""
    _ = language
    if not mapped_lines or not segments:
        return mapped_lines

    adjusted = list(mapped_lines)
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx, line in enumerate(adjusted):
        if not line.words or not line.text.strip():
            continue

        cur_tokens_all = [
            _normalize_match_token(w.text)
            for w in line.words
            if _normalize_match_token(w.text)
        ]
        prev_overlap = 0
        next_overlap = 0
        if idx > 0 and adjusted[idx - 1].words:
            prev_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx - 1].words[-3:]
                if _normalize_match_token(w.text)
            ]
            prev_overlap = _overlap_suffix_prefix(prev_tokens, cur_tokens_all, 3)
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx + 1].words[:3]
                if _normalize_match_token(w.text)
            ]
            next_overlap = _overlap_suffix_prefix(cur_tokens_all, next_tokens, 3)
        if idx > 0 and adjusted[idx - 1].words and len(line.words) <= 4:
            if prev_overlap >= 2:
                continue

        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )

        best_seg: Optional[timing_models.TranscriptionSegment] = None
        best_sim = 0.0
        best_contains = False
        best_rank: Tuple[float, float] = (-1.0, float("-inf"))
        best_contain_seg: Optional[timing_models.TranscriptionSegment] = None
        best_contain_sim = -1.0
        best_contain_dist = float("inf")
        repetitive_in_first_pass = (
            prev_overlap >= 2
            or next_overlap >= 2
            or (len(cur_tokens_all) >= 5 and len(set(cur_tokens_all)) <= 3)
        )
        for seg in sorted_segments:
            if abs(seg.start - line.start_time) > max_time_window:
                continue
            if (
                repetitive_in_first_pass
                and prev_end is not None
                and seg.start <= (prev_end + min_gap)
            ):
                continue
            seg_tokens = _normalize_text_tokens(seg.text)
            sim = max(
                _light_text_similarity(line.text, seg.text),
                _soft_text_similarity(line.text, seg.text),
            )
            if len(cur_tokens_all) >= 3 and len(seg_tokens) <= 1 and sim < 0.55:
                continue
            contains = _contains_token_sequence(line.text, seg.text)
            if not contains and len(line.words) <= 6:
                contains = _max_contiguous_soft_match_run(line.text, seg.text) >= 3
            dist = abs(seg.start - line.start_time)
            rank = (sim, -dist)
            if rank > best_rank:
                best_rank = rank
                best_sim = sim
                best_seg = seg
                best_contains = contains
            if contains and (
                sim > best_contain_sim
                or (sim == best_contain_sim and dist < best_contain_dist)
            ):
                best_contain_seg = seg
                best_contain_sim = sim
                best_contain_dist = dist

        if (
            best_seg is not None
            and best_contain_seg is not None
            and best_contain_sim >= (best_sim - contain_similarity_margin)
        ):
            best_seg = best_contain_seg
            best_sim = best_contain_sim
            best_contains = True

        min_sim_required = min_similarity
        if best_contains and len(line.words) <= 5:
            min_sim_required = min(min_similarity, 0.2)
        if best_seg is None or best_sim < min_sim_required:
            continue
        best_seg_tokens = _normalize_text_tokens(best_seg.text)
        best_token_overlap = _soft_token_overlap_ratio(cur_tokens_all, best_seg_tokens)
        if (
            len(line.words) <= 6
            and not repetitive_in_first_pass
            and not best_contains
            and best_token_overlap < 0.35
            and best_sim < 0.65
        ):
            continue

        late_by = line.start_time - best_seg.start
        allowed_max_late = strong_match_max_late if best_contains else max_late
        if best_contains and len(line.words) <= 5:
            allowed_max_late += 2.0
        if late_by < min_late or late_by > allowed_max_late:
            continue

        window_start = best_seg.start
        if prev_end is not None:
            window_start = max(window_start, prev_end + min_gap)

        window_end = best_seg.end
        if next_start is not None:
            window_end = min(window_end, next_start - min_gap)

        if window_end - window_start <= 0.1:
            continue

        shift = window_start - line.start_time
        if shift >= -0.2:
            continue
        if (line.start_time - window_start) < min_start_gain:
            continue

        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        shifted_line = models.Line(words=shifted_words, singer=line.singer)

        if shifted_line.end_time > window_end:
            total_duration = max(window_end - window_start, 0.2)
            spacing = total_duration / len(shifted_line.words)
            fitted_words = []
            for word_idx, w in enumerate(shifted_line.words):
                start = window_start + word_idx * spacing
                end = start + spacing * 0.9
                fitted_words.append(
                    models.Word(
                        text=w.text,
                        start_time=start,
                        end_time=end,
                        singer=w.singer,
                    )
                )
            shifted_line = models.Line(words=fitted_words, singer=line.singer)

        adjusted[idx] = shifted_line

        continue

    for idx, line in enumerate(adjusted):
        if not line.words or not line.text.strip():
            continue

        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )

        cur_tokens_all = [
            _normalize_match_token(w.text)
            for w in line.words
            if _normalize_match_token(w.text)
        ]
        prev_overlap = 0
        next_overlap = 0
        if idx > 0 and adjusted[idx - 1].words:
            prev_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx - 1].words[-3:]
                if _normalize_match_token(w.text)
            ]
            prev_overlap = _overlap_suffix_prefix(prev_tokens, cur_tokens_all, 3)
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx + 1].words[:3]
                if _normalize_match_token(w.text)
            ]
            next_overlap = _overlap_suffix_prefix(cur_tokens_all, next_tokens, 3)
        prev_set_overlap = 0.0
        next_set_overlap = 0.0
        if idx > 0 and adjusted[idx - 1].words and cur_tokens_all:
            prev_tokens_all = [
                _normalize_match_token(w.text)
                for w in adjusted[idx - 1].words
                if _normalize_match_token(w.text)
            ]
            prev_set_overlap = _soft_token_overlap_ratio(
                cur_tokens_all, prev_tokens_all
            )
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words and cur_tokens_all:
            next_tokens_all = [
                _normalize_match_token(w.text)
                for w in adjusted[idx + 1].words
                if _normalize_match_token(w.text)
            ]
            next_set_overlap = _soft_token_overlap_ratio(
                cur_tokens_all, next_tokens_all
            )
        is_repetitive_phrase = (
            prev_overlap >= 2
            or next_overlap >= 2
            or prev_set_overlap >= 0.5
            or next_set_overlap >= 0.5
            or (len(cur_tokens_all) >= 5 and len(set(cur_tokens_all)) <= 3)
        )

        early_best_seg: Optional[timing_models.TranscriptionSegment] = None
        early_best_sim = 0.0
        early_best_rank: Tuple[float, float] = (-1.0, float("-inf"))
        for seg in sorted_segments:
            if abs(seg.start - line.start_time) > max_time_window:
                continue
            if (
                is_repetitive_phrase
                and prev_end is not None
                and seg.start <= (prev_end + min_gap)
            ):
                continue
            seg_tokens = _normalize_text_tokens(seg.text)
            sim = max(
                _light_text_similarity(line.text, seg.text),
                _soft_text_similarity(line.text, seg.text),
            )
            if len(cur_tokens_all) >= 3 and len(seg_tokens) <= 1 and sim < 0.55:
                continue
            dist = abs(seg.start - line.start_time)
            rank = (sim, -dist)
            if rank > early_best_rank:
                early_best_rank = rank
                early_best_sim = sim
                early_best_seg = seg

        if early_best_seg is None or early_best_sim < early_min_similarity:
            continue
        early_seg_tokens = _normalize_text_tokens(early_best_seg.text)
        early_token_overlap = _soft_token_overlap_ratio(
            cur_tokens_all, early_seg_tokens
        )
        if (
            len(line.words) <= 6
            and not is_repetitive_phrase
            and early_token_overlap < 0.35
            and early_best_sim < 0.65
        ):
            continue

        early_by = early_best_seg.start - line.start_time
        if early_by < min_early or early_by > max_early:
            continue
        if not (is_repetitive_phrase or len(line.words) <= 4):
            continue

        line_duration = max(0.12, line.end_time - line.start_time)
        min_line_duration = max(0.24, min(0.8, 0.12 * len(line.words)))
        upper_start = line.start_time + max_early_push
        if next_start is not None:
            upper_start = min(upper_start, next_start - min_gap - min_line_duration)
        target_start = min(early_best_seg.start, upper_start)
        if prev_end is not None:
            target_start = max(target_start, prev_end + min_gap)
        if target_start <= line.start_time + 0.2:
            continue

        shift = target_start - line.start_time
        target_end = line.end_time + shift
        if next_start is not None:
            target_end = min(target_end, next_start - min_gap)
        if is_repetitive_phrase:
            target_end = min(target_end, early_best_seg.end)
        if target_end <= target_start + min_line_duration:
            continue

        if (target_end - target_start) < (line_duration * 0.85):
            spacing = (target_end - target_start) / len(line.words)
            compressed_words = []
            for word_idx, w in enumerate(line.words):
                ws = target_start + word_idx * spacing
                we = ws + spacing * 0.9
                if word_idx == len(line.words) - 1:
                    we = target_end
                compressed_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx] = models.Line(words=compressed_words, singer=line.singer)
        else:
            shifted_words = [
                models.Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in line.words
            ]
            adjusted[idx] = models.Line(words=shifted_words, singer=line.singer)

    adjusted = _realign_repetitive_runs_to_matching_segments(adjusted, sorted_segments)
    adjusted = _smooth_adjacent_duplicate_line_cadence(adjusted)
    adjusted = _rebalance_short_question_pairs(adjusted)
    adjusted = _retime_repetitive_question_runs_to_segment_windows(
        adjusted, sorted_segments
    )
    return _pull_adjacent_similar_lines_across_long_gaps(adjusted)
