"""Helpers for transcript-constrained WhisperX fallback alignment."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ... import models


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
