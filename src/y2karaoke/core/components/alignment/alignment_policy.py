"""Centralized policy decisions for alignment-time heuristics.

This module is intentionally small and deterministic. It exists to keep
high-level alignment decisions out of deeply nested pipeline code.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class LrcTimingTrustDecision:
    keep_lrc_timings: bool
    mode: str
    reason: str


def decide_lrc_timing_trust(
    *,
    lrc_duration_mismatch_sec: float,
    can_recover_with_audio_alignment: bool,
    likely_outro_padding: bool,
    severe_mismatch_threshold_sec: float = 12.0,
) -> LrcTimingTrustDecision:
    """Decide whether provider LRC line timings should be retained.

    The policy is conservative:
    - keep line timings when mismatch is mild/moderate,
    - keep line timings on likely trailing-outro padding,
    - drop line timings only when mismatch is severe and recovery is available.
    """
    if lrc_duration_mismatch_sec < severe_mismatch_threshold_sec:
        return LrcTimingTrustDecision(
            keep_lrc_timings=True,
            mode="degraded_duration_mismatch",
            reason="duration_mismatch_not_severe",
        )
    if likely_outro_padding:
        return LrcTimingTrustDecision(
            keep_lrc_timings=True,
            mode="degraded_outro_padding",
            reason="likely_trailing_outro_padding",
        )
    if can_recover_with_audio_alignment:
        return LrcTimingTrustDecision(
            keep_lrc_timings=False,
            mode="dropped_duration_mismatch",
            reason="severe_mismatch_with_audio_recovery",
        )
    return LrcTimingTrustDecision(
        keep_lrc_timings=True,
        mode="degraded_duration_mismatch",
        reason="severe_mismatch_without_audio_recovery",
    )


def should_retry_aggressive_whisper_dtw_map(
    *,
    line_count: int,
    aggressive_already_enabled: bool,
    metrics: Dict[str, float],
) -> bool:
    """Whether to attempt a second DTW map pass with aggressive Whisper settings."""
    if aggressive_already_enabled or line_count < 25:
        return False
    matched_ratio = float(metrics.get("matched_ratio", 0.0) or 0.0)
    line_coverage = float(metrics.get("line_coverage", 0.0) or 0.0)
    phonetic_coverage = float(metrics.get("phonetic_similarity_coverage", 0.0) or 0.0)
    return (
        0.78 <= matched_ratio <= 0.88
        and 0.80 <= line_coverage <= 0.92
        and phonetic_coverage >= 0.35
    )
