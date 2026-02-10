"""Helper utilities for Whisper-based mapping of LRC words to transcription timings."""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import models, timing_models, whisper_utils

_SPEECH_BLOCK_GAP = 5.0
_SEGMENT_TIME_WEIGHT = 0.18
_SEGMENT_INDEX_WEIGHT = 0.01
_SEGMENT_MAX_TIME_DIFF = 5.0
_SEGMENT_MIN_TEXT_SCORE = 0.25
_MAX_SEGMENT_LOOKAHEAD = 4.0
_BACKWARD_START_PENALTY = 0.5
_ORDER_POSITION_WEIGHT = 0.35
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


def _normalize_line_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


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
