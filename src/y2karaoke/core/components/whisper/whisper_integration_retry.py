"""Retry arbitration helpers for Whisper/LRC DTW alignment."""

import os
from dataclasses import dataclass
from typing import Dict

from ..alignment.alignment_policy import should_retry_aggressive_whisper_dtw_map


@dataclass(frozen=True)
class _RetryImproveDecisionConfig:
    min_matched_gain_with_line_tolerance: float
    max_line_drop_with_matched_gain: float
    min_line_gain_with_matched_tolerance: float
    max_matched_drop_with_line_gain: float
    min_phonetic_gain: float


def _default_retry_improve_config() -> _RetryImproveDecisionConfig:
    profile = os.getenv("Y2K_WHISPER_PROFILE", "default").strip().lower()
    if profile == "safe":
        return _RetryImproveDecisionConfig(
            min_matched_gain_with_line_tolerance=0.04,
            max_line_drop_with_matched_gain=0.01,
            min_line_gain_with_matched_tolerance=0.06,
            max_matched_drop_with_line_gain=0.005,
            min_phonetic_gain=0.06,
        )
    if profile == "aggressive":
        return _RetryImproveDecisionConfig(
            min_matched_gain_with_line_tolerance=0.02,
            max_line_drop_with_matched_gain=0.03,
            min_line_gain_with_matched_tolerance=0.04,
            max_matched_drop_with_line_gain=0.02,
            min_phonetic_gain=0.04,
        )
    return _RetryImproveDecisionConfig(
        min_matched_gain_with_line_tolerance=0.03,
        max_line_drop_with_matched_gain=0.02,
        min_line_gain_with_matched_tolerance=0.05,
        max_matched_drop_with_line_gain=0.01,
        min_phonetic_gain=0.05,
    )


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
    config = _default_retry_improve_config()
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
        (
            retry_matched >= base_matched + config.min_matched_gain_with_line_tolerance
            and retry_line >= base_line - config.max_line_drop_with_matched_gain
        )
        or (
            retry_line >= base_line + config.min_line_gain_with_matched_tolerance
            and retry_matched >= base_matched - config.max_matched_drop_with_line_gain
        )
        or (
            retry_phonetic >= base_phonetic + config.min_phonetic_gain
            and retry_matched >= base_matched
        )
    )
