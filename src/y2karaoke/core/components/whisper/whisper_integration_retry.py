"""Retry arbitration helpers for Whisper/LRC DTW alignment."""

from typing import Dict

from ..alignment.alignment_policy import should_retry_aggressive_whisper_dtw_map


def should_retry_with_aggressive_whisper(
    *,
    line_count: int,
    aggressive: bool,
    metrics: Dict[str, float],
) -> bool:
    return should_retry_aggressive_whisper_dtw_map(
        line_count=line_count,
        aggressive_already_enabled=aggressive,
        metrics=metrics,
    )


def retry_improves_alignment(
    baseline_metrics: Dict[str, float],
    retry_metrics: Dict[str, float],
) -> bool:
    base_matched = float(baseline_metrics.get("matched_ratio", 0.0) or 0.0)
    base_line = float(baseline_metrics.get("line_coverage", 0.0) or 0.0)
    base_phonetic = float(
        baseline_metrics.get("phonetic_similarity_coverage", 0.0) or 0.0
    )
    retry_matched = float(retry_metrics.get("matched_ratio", 0.0) or 0.0)
    retry_line = float(retry_metrics.get("line_coverage", 0.0) or 0.0)
    retry_phonetic = float(
        retry_metrics.get("phonetic_similarity_coverage", 0.0) or 0.0
    )
    return (
        (retry_matched >= base_matched + 0.03 and retry_line >= base_line - 0.02)
        or (retry_line >= base_line + 0.05 and retry_matched >= base_matched - 0.01)
        or (retry_phonetic >= base_phonetic + 0.05 and retry_matched >= base_matched)
    )
