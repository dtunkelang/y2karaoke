"""Offset and quality heuristics for lyrics alignment."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ...models import Line
from .helpers import _select_offset_anchor_timing

logger = logging.getLogger(__name__)


def _detect_offset_with_issues(
    vocals_path: str,
    line_timings: List[Tuple[float, str]],
    lyrics_offset: Optional[float],
    issues: List[str],
    *,
    auto_offset_scale: float = 1.0,
    scaled_offset_min_abs_sec: float = 0.0,
    scaled_offset_max_abs_sec: float = float("inf"),
    scale_large_negative_offsets: bool = False,
    allow_suspicious_positive_offset: bool = False,
    suppress_moderate_negative_offset: bool = False,
) -> Tuple[List[Tuple[float, str]], float]:
    """Detect vocal offset, track issues for quality report."""
    from ..alignment.alignment import detect_song_start

    detected_vocal_start = detect_song_start(vocals_path)
    anchor_time, _anchor_text, used_alternate_anchor = _select_offset_anchor_timing(
        line_timings
    )
    delta = detected_vocal_start - anchor_time

    logger.info(
        f"Vocal timing: audio_start={detected_vocal_start:.2f}s, "
        f"LRC_start={anchor_time:.2f}s, delta={delta:+.2f}s"
    )

    offset = _compute_auto_offset(
        delta=delta,
        lyrics_offset=lyrics_offset,
        used_alternate_anchor=used_alternate_anchor,
        issues=issues,
        auto_offset_scale=auto_offset_scale,
        scaled_offset_min_abs_sec=scaled_offset_min_abs_sec,
        scaled_offset_max_abs_sec=scaled_offset_max_abs_sec,
        scale_large_negative_offsets=scale_large_negative_offsets,
        allow_suspicious_positive_offset=allow_suspicious_positive_offset,
        suppress_moderate_negative_offset=suppress_moderate_negative_offset,
    )

    if offset != 0.0:
        line_timings = [(ts + offset, text) for ts, text in line_timings]

    return line_timings, offset


def _should_skip_moderate_negative_offset(
    *,
    delta: float,
    suppress_moderate_negative_offset: bool,
    used_alternate_anchor: bool,
    auto_offset_scale: float,
    scaled_offset_min_abs_sec: float,
    scaled_offset_max_abs_sec: float,
    scale_large_negative_offsets: bool,
) -> bool:
    if not suppress_moderate_negative_offset or delta >= 0.0:
        return False
    scale = 0.6 if used_alternate_anchor else 1.0
    if scaled_offset_min_abs_sec <= abs(delta) <= scaled_offset_max_abs_sec:
        scale *= max(0.0, auto_offset_scale)
    elif scale_large_negative_offsets and abs(delta) >= scaled_offset_min_abs_sec:
        scale *= max(0.0, auto_offset_scale)
    effective_offset = abs(delta * scale)
    return effective_offset <= 1.4


def _compute_auto_offset(
    *,
    delta: float,
    lyrics_offset: Optional[float],
    used_alternate_anchor: bool,
    issues: List[str],
    auto_offset_scale: float,
    scaled_offset_min_abs_sec: float,
    scaled_offset_max_abs_sec: float,
    scale_large_negative_offsets: bool,
    allow_suspicious_positive_offset: bool,
    suppress_moderate_negative_offset: bool,
) -> float:
    auto_offset_max_abs_sec = 5.0
    if lyrics_offset is not None:
        return lyrics_offset
    if allow_suspicious_positive_offset and 2.5 < delta <= auto_offset_max_abs_sec:
        scale = 0.6 if used_alternate_anchor else 1.0
        offset = delta * scale
        logger.info(
            "Auto-applying suspicious positive vocal offset under disagreement guard: %+.2fs",
            offset,
        )
        return offset
    if 2.5 < abs(delta) <= auto_offset_max_abs_sec:
        logger.warning(
            "Detected vocal offset (%+.2fs) matches suspicious range (2.5-5.0s) - NOT auto-applying.",
            delta,
        )
        return 0.0
    if _should_skip_moderate_negative_offset(
        delta=delta,
        suppress_moderate_negative_offset=suppress_moderate_negative_offset,
        used_alternate_anchor=used_alternate_anchor,
        auto_offset_scale=auto_offset_scale,
        scaled_offset_min_abs_sec=scaled_offset_min_abs_sec,
        scaled_offset_max_abs_sec=scaled_offset_max_abs_sec,
        scale_large_negative_offsets=scale_large_negative_offsets,
    ):
        logger.info(
            "Skipping moderate negative vocal offset under disagreement guard: %+.2fs",
            delta,
        )
        return 0.0
    if 0.3 < abs(delta) <= 2.5:
        scale = 0.6 if used_alternate_anchor else 1.0
        if scaled_offset_min_abs_sec <= abs(delta) <= scaled_offset_max_abs_sec:
            scale *= max(0.0, auto_offset_scale)
        elif (
            scale_large_negative_offsets
            and delta < 0.0
            and abs(delta) >= scaled_offset_min_abs_sec
        ):
            scale *= max(0.0, auto_offset_scale)
        offset = delta * scale
        logger.info(f"Auto-applying vocal offset: {offset:+.2f}s")
        return offset
    if abs(delta) > auto_offset_max_abs_sec:
        logger.warning(
            "Large timing delta (%+.2fs) exceeds auto-offset clamp (%.1fs) - "
            "not auto-applying.",
            delta,
            auto_offset_max_abs_sec,
        )
        issues.append(
            f"Large timing delta ({delta:+.2f}s) exceeded auto-offset clamp and was not applied"
        )
    return 0.0


def _refine_timing_with_quality(
    lines: List[Line],
    vocals_path: str,
    line_timings: List[Tuple[float, str]],
    lrc_text: str,
    target_duration: Optional[int],
    issues: List[str],
) -> Tuple[List[Line], str]:
    """Refine timing and track issues. Returns (lines, alignment_method)."""
    from ...refine import refine_word_timing
    from ..alignment.alignment import adjust_timing_for_duration_mismatch
    from .sync import get_lrc_duration

    lines = refine_word_timing(lines, vocals_path)
    alignment_method = "onset_refined"
    logger.debug("Word-level timing refined using vocals")

    lrc_duration = get_lrc_duration(lrc_text)
    if target_duration and lrc_duration and abs(target_duration - lrc_duration) > 8:
        logger.info(f"Duration mismatch: LRC={lrc_duration}s, audio={target_duration}s")
        issues.append(
            f"Duration mismatch: LRC={lrc_duration}s vs audio={target_duration}s"
        )
        lines = adjust_timing_for_duration_mismatch(
            lines,
            line_timings,
            vocals_path,
            lrc_duration=lrc_duration,
            audio_duration=target_duration,
        )

    return lines, alignment_method


def _calculate_quality_score(quality_report: dict) -> float:
    """Calculate overall quality score from report components."""
    if quality_report["lyrics_quality"]:
        base_score = quality_report["lyrics_quality"].get("quality_score", 50.0)
    elif quality_report.get("dtw_metrics"):
        base_score = _score_from_dtw_metrics(quality_report["dtw_metrics"])
    else:
        base_score = 30.0

    method_bonus = {
        "whisper_hybrid": 10,
        "onset_refined": 5,
        "lrc_only": 0,
        "genius_fallback": -20,
        "none": -50,
    }
    base_score += method_bonus.get(quality_report["alignment_method"], 0)
    base_score -= len(quality_report["issues"]) * 5

    return max(0.0, min(100.0, base_score))


def _score_from_dtw_metrics(metrics: dict) -> float:
    """Heuristic score derived from Whisper/DTW alignment metrics."""
    matched_ratio = float(metrics.get("matched_ratio", 0.0))
    avg_similarity = float(metrics.get("avg_similarity", 0.0))
    line_coverage = float(metrics.get("line_coverage", 0.0))
    phonetic_similarity_coverage = float(
        metrics.get("phonetic_similarity_coverage", matched_ratio * avg_similarity)
    )
    high_similarity_ratio = float(metrics.get("high_similarity_ratio", avg_similarity))
    exact_match_ratio = float(metrics.get("exact_match_ratio", 0.0))

    score = 40.0
    score += matched_ratio * 20.0
    score += avg_similarity * 15.0
    score += line_coverage * 10.0
    score += phonetic_similarity_coverage * 10.0
    score += high_similarity_ratio * 3.0
    score += exact_match_ratio * 2.0
    return max(20.0, min(100.0, score))
