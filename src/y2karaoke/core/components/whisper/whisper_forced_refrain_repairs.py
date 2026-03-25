"""Refrain repair helpers for transcript-constrained WhisperX fallback."""

from __future__ import annotations

import re
from typing import Any, List

from ... import models
from ..alignment import timing_models

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _normalize_token(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def _normalize_line_text(text: str) -> str:
    return " ".join(
        _normalize_token(part) for part in text.split() if _normalize_token(part)
    )


def _coerce_forced_segments(
    segments: Any,
) -> List[timing_models.TranscriptionSegment]:
    if not isinstance(segments, list):
        return []
    coerced: List[timing_models.TranscriptionSegment] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        end = seg.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        coerced.append(
            timing_models.TranscriptionSegment(
                start=float(start),
                end=float(end),
                text=str(seg.get("text") or ""),
                words=[],
            )
        )
    return coerced


def _extract_aligned_word_spans(
    aligned_segments: Any,
) -> list[tuple[str, float, float]]:
    if not isinstance(aligned_segments, list):
        return []
    aligned_words: list[tuple[str, float, float]] = []
    for seg in aligned_segments:
        if not isinstance(seg, dict):
            continue
        for word in seg.get("words", []) or []:
            start = word.get("start")
            end = word.get("end")
            text = str(word.get("word") or word.get("text") or "").strip()
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            token = _normalize_token(text)
            if token:
                aligned_words.append((token, float(start), float(end)))
    return aligned_words


def _neighbor_line_bounds(
    adjusted: List[models.Line], idx: int
) -> tuple[float | None, float | None]:
    prev_end = (
        adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
    )
    next_start = (
        adjusted[idx + 1].start_time
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words
        else None
    )
    return prev_end, next_start


def _best_aligned_word_sequence_match(
    *,
    aligned_words: list[tuple[str, float, float]],
    target_tokens: list[str],
    line_start: float,
    prev_end: float | None,
    next_start: float | None,
    min_gap: float,
    min_late_shift: float,
    max_late_shift: float,
) -> tuple[float, float] | None:
    best_match: tuple[float, float] | None = None
    best_delta: float | None = None
    for word_idx in range(0, len(aligned_words) - len(target_tokens) + 1):
        candidate = aligned_words[word_idx : word_idx + len(target_tokens)]
        if [token for token, _start, _end in candidate] != target_tokens:
            continue
        match_start = candidate[0][1]
        match_end = candidate[-1][2]
        delta = match_start - line_start
        if delta < min_late_shift or delta > max_late_shift:
            continue
        if prev_end is not None and match_start <= prev_end + min_gap:
            continue
        if next_start is not None and match_end >= next_start - min_gap:
            continue
        if best_delta is None or delta < best_delta:
            best_match = (match_start, match_end)
            best_delta = delta
    return best_match


def _rebuild_line_to_match_span(
    line: models.Line, *, target_start: float, target_end: float
) -> models.Line:
    spacing = (target_end - target_start) / len(line.words)
    rebuilt_words: list[models.Word] = []
    for word_idx, word in enumerate(line.words):
        start = target_start + word_idx * spacing
        end = start + spacing * 0.9
        if word_idx == len(line.words) - 1:
            end = target_end
        rebuilt_words.append(
            models.Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return models.Line(words=rebuilt_words, singer=line.singer)


def _restore_short_refrains_from_aligned_segment_words(
    forced_lines: List[models.Line],
    aligned_segments: Any,
    *,
    min_gap: float = 0.05,
    max_words: int = 4,
    min_late_shift: float = 0.8,
    max_late_shift: float = 3.5,
) -> tuple[List[models.Line], int]:
    aligned_words = _extract_aligned_word_spans(aligned_segments)
    if not aligned_words:
        return forced_lines, 0

    adjusted = list(forced_lines)
    restored = 0
    for idx, line in enumerate(adjusted):
        if not line.words or len(line.words) > max_words:
            continue
        target_tokens = [_normalize_token(word.text) for word in line.words]
        if not target_tokens or any(not token for token in target_tokens):
            continue

        prev_end, next_start = _neighbor_line_bounds(adjusted, idx)
        best_match = _best_aligned_word_sequence_match(
            aligned_words=aligned_words,
            target_tokens=target_tokens,
            line_start=line.start_time,
            prev_end=prev_end,
            next_start=next_start,
            min_gap=min_gap,
            min_late_shift=min_late_shift,
            max_late_shift=max_late_shift,
        )
        if best_match is None:
            continue

        adjusted[idx] = _rebuild_line_to_match_span(
            line,
            target_start=best_match[0],
            target_end=best_match[1],
        )
        restored += 1

    return adjusted, restored


def _repeated_refrain_counts(
    forced_lines: List[models.Line],
) -> dict[str, int]:
    normalized_counts: dict[str, int] = {}
    for line in forced_lines:
        if not line.words or not line.text.strip():
            continue
        key = _normalize_line_text(line.text)
        if key:
            normalized_counts[key] = normalized_counts.get(key, 0) + 1
    return normalized_counts


def _followup_refrain_window(
    *,
    adjusted: List[models.Line],
    idx: int,
    line: models.Line,
    prev: models.Line,
    min_current_gap: float,
    min_previous_words: int,
    min_previous_duration: float,
    min_gap_base: float,
    min_gap_per_word: float,
    min_shift: float,
    inter_line_gap: float,
    max_refrain_duration: float,
) -> tuple[float, float] | None:
    prev_duration = prev.end_time - prev.start_time
    if len(prev.words) < min_previous_words and prev_duration < min_previous_duration:
        return None
    current_gap = line.start_time - prev.end_time
    if current_gap > min_current_gap:
        return None

    target_start = prev.end_time + min_gap_base + min_gap_per_word * len(line.words)
    target_duration = min(
        line.end_time - line.start_time,
        max(1.0, min(max_refrain_duration, 0.8 + 0.25 * len(line.words))),
    )
    if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
        next_start = adjusted[idx + 1].start_time
        max_duration = next_start - inter_line_gap - target_start
        if max_duration <= 0.2:
            return None
        target_duration = min(target_duration, max_duration)
    shift = target_start - line.start_time
    if shift < min_shift or target_duration <= 0.2:
        return None
    return target_start, target_duration


def _enforce_repeated_short_refrain_followup_gap(
    forced_lines: List[models.Line],
    *,
    min_current_gap: float = 0.35,
    min_previous_words: int = 6,
    min_previous_duration: float = 2.5,
    min_gap_base: float = 0.9,
    min_gap_per_word: float = 0.15,
    min_shift: float = 0.5,
    inter_line_gap: float = 0.05,
    max_refrain_duration: float = 1.8,
) -> tuple[List[models.Line], int]:
    if not forced_lines:
        return forced_lines, 0

    normalized_counts = _repeated_refrain_counts(forced_lines)
    adjusted = list(forced_lines)
    shifted = 0
    for idx in range(1, len(adjusted)):
        prev = adjusted[idx - 1]
        line = adjusted[idx]
        if not prev.words or not line.words or len(line.words) > 4:
            continue
        key = _normalize_line_text(line.text)
        if not key or normalized_counts.get(key, 0) < 2:
            continue
        target_window = _followup_refrain_window(
            adjusted=adjusted,
            idx=idx,
            line=line,
            prev=prev,
            min_current_gap=min_current_gap,
            min_previous_words=min_previous_words,
            min_previous_duration=min_previous_duration,
            min_gap_base=min_gap_base,
            min_gap_per_word=min_gap_per_word,
            min_shift=min_shift,
            inter_line_gap=inter_line_gap,
            max_refrain_duration=max_refrain_duration,
        )
        if target_window is None:
            continue
        target_start, target_duration = target_window
        spacing = target_duration / len(line.words)
        rebuilt_words: list[models.Word] = []
        for word_idx, word in enumerate(line.words):
            start = target_start + word_idx * spacing
            end = start + spacing * 0.9
            if word_idx == len(line.words) - 1:
                end = target_start + target_duration
            rebuilt_words.append(
                models.Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
        adjusted[idx] = models.Line(words=rebuilt_words, singer=line.singer)
        shifted += 1

    return adjusted, shifted
