"""Helpers for transcript-constrained WhisperX fallback alignment."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ... import models
from ..alignment import timing_models

_LIGHT_LEADING_TOKENS = {"the", "a", "an"}
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _normalize_token(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def _shift_line(line: models.Line, delta: float) -> models.Line:
    return models.Line(
        words=[
            models.Word(
                text=word.text,
                start_time=word.start_time + delta,
                end_time=word.end_time + delta,
                singer=word.singer,
            )
            for word in line.words
        ],
        singer=line.singer,
    )


def _count_leading_light_tokens(normalized_tokens: List[str]) -> int:
    count = 0
    for token in normalized_tokens:
        if token in _LIGHT_LEADING_TOKENS:
            count += 1
            continue
        break
    return count


def _find_local_content_anchor_start(
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    content_token: str,
    line: models.Line,
    lookback_sec: float,
    lookahead_sec: float,
) -> float | None:
    nearby_matches = [
        word.start
        for word in whisper_words
        if _normalize_token(word.text) == content_token
        and line.start_time - lookback_sec
        <= word.start
        <= line.end_time + lookahead_sec
    ]
    if not nearby_matches:
        return None
    return min(nearby_matches)


def _reanchor_delta_for_line(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_shift_sec: float,
    max_shift_sec: float,
    lookback_sec: float,
    lookahead_sec: float,
) -> float | None:
    if len(line.words) < 2:
        return None
    normalized = [_normalize_token(word.text) for word in line.words]
    leading_count = _count_leading_light_tokens(normalized)
    if leading_count == 0 or leading_count >= len(line.words):
        return None

    content_token = normalized[leading_count]
    if not content_token:
        return None

    target_start = _find_local_content_anchor_start(
        whisper_words,
        content_token=content_token,
        line=line,
        lookback_sec=lookback_sec,
        lookahead_sec=lookahead_sec,
    )
    if target_start is None:
        return None

    delta = target_start - line.words[leading_count].start_time
    if delta < min_shift_sec or delta > max_shift_sec:
        return None
    return delta


def _can_apply_reanchored_line(
    adjusted: List[models.Line], idx: int, shifted_line: models.Line
) -> bool:
    if idx + 1 >= len(adjusted) or not adjusted[idx + 1].words:
        return True
    next_start = adjusted[idx + 1].start_time
    return shifted_line.end_time <= next_start - 0.04


def _reanchor_forced_lines_to_local_content_words(
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    *,
    min_shift_sec: float = 1.0,
    max_shift_sec: float = 3.5,
    lookback_sec: float = 1.0,
    lookahead_sec: float = 4.0,
) -> tuple[List[models.Line], int]:
    if not whisper_words:
        return forced_lines, 0

    adjusted = list(forced_lines)
    shifted = 0
    for idx in range(len(adjusted) - 1, -1, -1):
        line = adjusted[idx]
        delta = _reanchor_delta_for_line(
            line,
            whisper_words,
            min_shift_sec=min_shift_sec,
            max_shift_sec=max_shift_sec,
            lookback_sec=lookback_sec,
            lookahead_sec=lookahead_sec,
        )
        if delta is None:
            continue

        shifted_line = _shift_line(line, delta)
        if not _can_apply_reanchored_line(adjusted, idx, shifted_line):
            continue
        adjusted[idx] = shifted_line
        shifted += 1
    return adjusted, shifted


def attempt_whisperx_forced_alignment(
    *,
    lines: List[models.Line],
    baseline_lines: List[models.Line],
    vocals_path: str,
    language: str | None,
    detected_lang: str | None,
    logger: Any,
    used_model: str,
    reason: str,
    align_lines_with_whisperx_fn: Callable[..., Any],
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    whisper_words: List[timing_models.TranscriptionWord] | None = None,
    normalize_line_word_timings_fn: Callable[..., Any] | None = None,
    enforce_monotonic_line_starts_fn: Callable[..., Any] | None = None,
    enforce_non_overlapping_lines_fn: Callable[..., Any] | None = None,
    min_forced_word_coverage: float = 0.2,
    min_forced_line_coverage: float = 0.2,
) -> Optional[Tuple[List[models.Line], List[str], Dict[str, Any]]]:
    forced_language = language or detected_lang
    forced = align_lines_with_whisperx_fn(lines, vocals_path, forced_language, logger)
    if forced is None:
        return None
    forced_lines, forced_metrics = forced
    forced_word_coverage = float(forced_metrics.get("forced_word_coverage", 0.0))
    forced_line_coverage = float(forced_metrics.get("forced_line_coverage", 0.0))
    if (
        forced_word_coverage < min_forced_word_coverage
        or forced_line_coverage < min_forced_line_coverage
    ):
        logger.warning(
            (
                "Discarded WhisperX forced alignment due to low forced coverage "
                "(word=%.2f line=%.2f)"
            ),
            forced_word_coverage,
            forced_line_coverage,
        )
        return None

    rollback, short_before, short_after = should_rollback_short_line_degradation_fn(
        baseline_lines, forced_lines
    )
    if rollback:
        repaired_lines, restored_count = restore_implausibly_short_lines_fn(
            baseline_lines, forced_lines
        )
        repaired_rollback, _, repaired_after = (
            should_rollback_short_line_degradation_fn(baseline_lines, repaired_lines)
        )
        if restored_count > 0 and not repaired_rollback:
            logger.info(
                "Kept WhisperX forced alignment after restoring %d short baseline line(s) (%d -> %d)",
                restored_count,
                short_after,
                repaired_after,
            )
            forced_lines = repaired_lines
            rollback = False
    if rollback:
        logger.warning(
            "Discarded WhisperX forced alignment due to short-line degradation (%d -> %d)",
            short_before,
            short_after,
        )
        return None

    forced_lines, reanchored_count = _reanchor_forced_lines_to_local_content_words(
        forced_lines,
        whisper_words,
    )
    if reanchored_count:
        logger.info(
            "Reanchored %d forced-aligned line(s) to local content-word Whisper anchors",
            reanchored_count,
        )

    if normalize_line_word_timings_fn is not None:
        forced_lines = normalize_line_word_timings_fn(forced_lines)
    if enforce_monotonic_line_starts_fn is not None:
        forced_lines = enforce_monotonic_line_starts_fn(forced_lines)
    if enforce_non_overlapping_lines_fn is not None:
        forced_lines = enforce_non_overlapping_lines_fn(forced_lines)

    forced_payload: Dict[str, Any] = {
        "matched_ratio": forced_word_coverage,
        "word_coverage": forced_word_coverage,
        "avg_similarity": 1.0,
        "line_coverage": forced_line_coverage,
        "phonetic_similarity_coverage": forced_word_coverage,
        "high_similarity_ratio": 1.0,
        "exact_match_ratio": 0.0,
        "unmatched_ratio": 1.0 - forced_word_coverage,
        "dtw_used": 0.0,
        "dtw_mode": 0.0,
        "whisperx_forced": 1.0,
        "whisper_model": used_model,
    }
    return (
        forced_lines,
        [f"Applied WhisperX transcript-constrained forced alignment due to {reason}"],
        forced_payload,
    )
