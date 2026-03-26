"""Exact-prefix repairs for forced-alignment fallback lines."""

from __future__ import annotations

from typing import Callable, List

from ... import models
from ..alignment import timing_models


def find_exact_whisper_sequence_match(
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    tokens: List[str],
    search_start: float,
    search_end: float,
    normalize_token_fn: Callable[[str], str],
) -> tuple[float, float] | None:
    indexed_match = find_exact_whisper_sequence_match_with_index(
        whisper_words,
        tokens=tokens,
        search_start=search_start,
        search_end=search_end,
        normalize_token_fn=normalize_token_fn,
    )
    if indexed_match is None:
        return None
    match_start, match_end, _match_idx = indexed_match
    return match_start, match_end


def find_exact_whisper_sequence_match_with_index(
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    tokens: List[str],
    search_start: float,
    search_end: float,
    normalize_token_fn: Callable[[str], str],
) -> tuple[float, float, int] | None:
    if not whisper_words or not tokens:
        return None
    normalized_words = [
        (normalize_token_fn(word.text), float(word.start), float(word.end))
        for word in whisper_words
    ]
    best_match: tuple[float, float, int] | None = None
    best_start: float | None = None
    for idx in range(0, len(normalized_words) - len(tokens) + 1):
        candidate = normalized_words[idx : idx + len(tokens)]
        if [token for token, _start, _end in candidate] != tokens:
            continue
        match_start = candidate[0][1]
        match_end = candidate[-1][2]
        if match_end < search_start or match_start > search_end:
            continue
        if best_start is None or match_start < best_start:
            best_match = (match_start, match_end, idx)
            best_start = match_start
    return best_match


def rescale_line_to_new_start_preserving_end(
    line: models.Line, target_start: float
) -> models.Line:
    old_duration = line.end_time - line.start_time
    new_duration = line.end_time - target_start
    span = old_duration if old_duration > 0 else 1.0
    return models.Line(
        words=[
            models.Word(
                text=word.text,
                start_time=target_start
                + ((word.start_time - line.start_time) / span) * new_duration,
                end_time=target_start
                + ((word.end_time - line.start_time) / span) * new_duration,
                singer=word.singer,
            )
            for word in line.words
        ],
        singer=line.singer,
    )


def _prefix_tokens_for_line(
    line: models.Line,
    *,
    prefix_token_count: int,
    normalize_token_fn: Callable[[str], str],
) -> list[str] | None:
    prefix_tokens = [
        normalize_token_fn(word.text) for word in line.words[:prefix_token_count]
    ]
    if len(prefix_tokens) < prefix_token_count or any(
        not token for token in prefix_tokens
    ):
        return None
    return prefix_tokens


def _earlier_prefix_match_for_line(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    prefix_token_count: int,
    min_shift_sec: float,
    max_shift_sec: float,
    normalize_token_fn: Callable[[str], str],
) -> tuple[float, float, int] | None:
    prefix_tokens = _prefix_tokens_for_line(
        line,
        prefix_token_count=prefix_token_count,
        normalize_token_fn=normalize_token_fn,
    )
    if prefix_tokens is None:
        return None
    return find_exact_whisper_sequence_match_with_index(
        whisper_words,
        tokens=prefix_tokens,
        search_start=line.start_time - max_shift_sec,
        search_end=line.start_time - min_shift_sec,
        normalize_token_fn=normalize_token_fn,
    )


def _boundary_matches_prior_line(
    *,
    adjusted: List[models.Line],
    idx: int,
    match_start: float,
    match_index: int,
    whisper_words: List[timing_models.TranscriptionWord],
    max_boundary_gap_sec: float,
    normalize_token_fn: Callable[[str], str],
) -> bool:
    if idx == 0 or not adjusted[idx - 1].words or match_index == 0:
        return False
    prior_line_last_token = normalize_token_fn(adjusted[idx - 1].words[-1].text)
    if not prior_line_last_token:
        return False
    boundary_token = normalize_token_fn(whisper_words[match_index - 1].text)
    if boundary_token != prior_line_last_token:
        return False
    return (match_start - adjusted[idx - 1].end_time) <= max_boundary_gap_sec


def reanchor_medium_lines_to_earlier_exact_prefixes(
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    *,
    normalize_token_fn: Callable[[str], str],
    can_apply_reanchored_line_fn: Callable[[List[models.Line], int, models.Line], bool],
    min_words: int = 7,
    max_words: int = 9,
    prefix_token_count: int = 3,
    min_shift_sec: float = 0.45,
    max_shift_sec: float = 0.8,
    max_applied_shift_sec: float = 0.5,
    max_boundary_gap_sec: float = 0.15,
    min_gap: float = 0.05,
) -> tuple[List[models.Line], int]:
    if not whisper_words:
        return forced_lines, 0

    adjusted = list(forced_lines)
    shifted = 0
    for idx in range(len(adjusted) - 1, -1, -1):
        line = adjusted[idx]
        if len(line.words) < min_words or len(line.words) > max_words:
            continue
        prefix_match = _earlier_prefix_match_for_line(
            line,
            whisper_words,
            prefix_token_count=prefix_token_count,
            min_shift_sec=min_shift_sec,
            max_shift_sec=max_shift_sec,
            normalize_token_fn=normalize_token_fn,
        )
        if prefix_match is None:
            continue
        match_start, _match_end, match_index = prefix_match
        if not _boundary_matches_prior_line(
            adjusted=adjusted,
            idx=idx,
            match_start=match_start,
            match_index=match_index,
            whisper_words=whisper_words,
            max_boundary_gap_sec=max_boundary_gap_sec,
            normalize_token_fn=normalize_token_fn,
        ):
            continue
        target_start = max(match_start, line.start_time - max_applied_shift_sec)
        shifted_line = rescale_line_to_new_start_preserving_end(line, target_start)
        if shifted_line.start_time < adjusted[idx - 1].end_time + min_gap:
            continue
        if not can_apply_reanchored_line_fn(adjusted, idx, shifted_line):
            continue
        adjusted[idx] = shifted_line
        shifted += 1
    return adjusted, shifted
