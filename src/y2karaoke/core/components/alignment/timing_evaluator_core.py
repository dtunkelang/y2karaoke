"""Core evaluation logic for lyrics timing."""

import logging
from typing import List, Optional, Tuple
import numpy as np

from ...models import Line
from .timing_models import (
    TimingIssue,
    AudioFeatures,
    TimingReport,
)
from .timing_evaluator_intervals import (
    clip_intervals as _clip_intervals_impl,
    complement_intervals as _complement_intervals_impl,
    compute_audio_coverage_metrics as _compute_audio_coverage_metrics_impl,
    intersect_intervals as _intersect_intervals_impl,
    merge_intervals as _merge_intervals_impl,
    total_duration as _total_duration_impl,
)
from .timing_evaluator_scoring import (
    calculate_line_alignment_score as _calculate_line_alignment_score_impl,
    find_closest_onset as _find_closest_onset_impl,
    generate_summary as _generate_summary_impl,
    identify_ordering_issues as _identify_ordering_issues_impl,
)
from .timing_evaluator_pauses import (
    append_gap_issues as _append_gap_issues_impl,
    append_line_spans_silence_issues as _append_line_spans_silence_issues_impl,
    append_unexpected_pause_issues as _append_unexpected_pause_issues_impl,
    calculate_pause_score as _calculate_pause_score_impl,
    calculate_pause_score_with_stats as _calculate_pause_score_with_stats_impl,
    check_pause_alignment as _check_pause_alignment_impl,
)
from ...audio_analysis import (
    _check_vocal_activity_in_range,
    _check_for_silence_in_range,
)

logger = logging.getLogger(__name__)


def evaluate_timing(
    lines: List[Line],
    audio_features: AudioFeatures,
    source_name: str = "unknown",
) -> TimingReport:
    """Evaluate lyrics timing against audio features."""
    issues: List[TimingIssue] = []

    # 1. Check logical ordering
    issues.extend(_identify_ordering_issues(lines))

    # 2. Check alignment with onsets
    (
        line_alignment_score,
        matched_count,
        line_issues,
        line_offsets,
    ) = _calculate_line_alignment_score(lines, audio_features.onset_times)
    issues.extend(line_issues)

    # 3. Check alignment with pauses
    pause_issues = _check_pause_alignment(lines, audio_features)
    issues.extend(pause_issues)
    pause_score, matched_silences, total_silences = _calculate_pause_score_with_stats(
        lines, audio_features
    )

    # 4. Check vocal coverage
    (
        audio_coverage_score,
        coverage_issues,
        lyric_on_vocal_ratio,
        vocal_covered_by_lyrics_ratio,
        lyric_in_silence_ratio,
    ) = _calculate_audio_coverage_score(lines, audio_features)
    issues.extend(coverage_issues)

    # Calculate overall score
    total_lines = len([line for line in lines if line.words])
    line_confidence = (len(line_offsets) / total_lines) if total_lines > 0 else 1.0
    pause_confidence = (
        (matched_silences / total_silences) if total_silences > 0 else 1.0
    )
    confidence = 0.7 * line_confidence + 0.3 * pause_confidence

    overall_score = (
        0.5 * line_alignment_score + 0.2 * pause_score + 0.3 * audio_coverage_score
    ) * confidence

    # Statistics
    avg_offset = float(np.mean(line_offsets)) if line_offsets else 0.0
    std_offset = float(np.std(line_offsets)) if len(line_offsets) > 1 else 0.0

    summary = _generate_summary(
        float(overall_score),
        float(line_alignment_score),
        float(pause_score),
        float(audio_coverage_score),
        float(avg_offset),
        float(std_offset),
        len(issues),
        total_lines,
    )

    return TimingReport(
        source_name=source_name,
        overall_score=float(overall_score),
        line_alignment_score=float(line_alignment_score),
        pause_alignment_score=float(pause_score),
        issues=issues,
        summary=summary,
        avg_line_offset=float(avg_offset),
        std_line_offset=float(std_offset),
        matched_onsets=matched_count,
        total_lines=total_lines,
        lyric_on_vocal_ratio=float(lyric_on_vocal_ratio),
        vocal_covered_by_lyrics_ratio=float(vocal_covered_by_lyrics_ratio),
        lyric_in_silence_ratio=float(lyric_in_silence_ratio),
    )


def _identify_ordering_issues(lines: List[Line]) -> List[TimingIssue]:
    return _identify_ordering_issues_impl(lines)


def _calculate_line_alignment_score(
    lines: List[Line], onset_times: np.ndarray
) -> Tuple[float, int, List[TimingIssue], List[float]]:
    return _calculate_line_alignment_score_impl(lines, onset_times)


def _calculate_audio_coverage_score(
    lines: List[Line], audio_features: AudioFeatures
) -> Tuple[float, List[TimingIssue], float, float, float]:
    """Calculate score based on how well lyrics cover vocal activity.

    Heuristic:
        - Lyrics exist to represent vocals.
        - Ideally, 100% of vocal activity should be covered by lyric duration (ignoring short breaths).
        - Ideally, 100% of lyric duration should overlap with vocal activity (no lyrics during silence).

    Metrics:
        - lyric_on_vocal_ratio: Fraction of lyric duration that overlaps with non-silent audio.
        - vocal_covered_by_lyrics_ratio: Fraction of non-silent audio covered by lyrics.
        - lyric_in_silence_ratio: Fraction of lyric duration that falls in silent regions (Bad).

    Issues:
        - 'uncovered_vocal_regions': If we detect >3 significant vocal bursts (>0.5s) with no lyrics,
          it suggests missing lines or massive desync.

    Returns:
        (score, issues, lyric_on_vocal, vocal_covered, lyric_in_silence)
    """
    issues = []
    (
        lyric_on_vocal_ratio,
        vocal_covered_by_lyrics_ratio,
        lyric_in_silence_ratio,
        uncovered_regions,
    ) = _compute_audio_coverage_metrics(lines, audio_features)

    if uncovered_regions >= 3:
        issues.append(
            TimingIssue(
                issue_type="uncovered_vocal_regions",
                line_index=-1,
                lyrics_time=audio_features.vocal_start,
                audio_time=None,
                delta=float(uncovered_regions),
                severity="moderate",
                description=(
                    f"Detected {uncovered_regions} non-silent vocal region(s) with no lyric coverage"
                ),
            )
        )

    score = 100.0 * (0.5 * lyric_on_vocal_ratio + 0.5 * vocal_covered_by_lyrics_ratio)
    return (
        score,
        issues,
        lyric_on_vocal_ratio,
        vocal_covered_by_lyrics_ratio,
        lyric_in_silence_ratio,
    )


def _find_closest_onset(
    target_time: float,
    onset_times: np.ndarray,
    max_distance: float = 2.0,
) -> Tuple[Optional[float], float]:
    return _find_closest_onset_impl(target_time, onset_times, max_distance)


def _append_line_spans_silence_issues(
    lines: List[Line],
    audio_features: AudioFeatures,
    issues: List[TimingIssue],
) -> None:
    _append_line_spans_silence_issues_impl(lines, audio_features, issues)


def _append_gap_issues(
    lines: List[Line],
    audio_features: AudioFeatures,
    issues: List[TimingIssue],
) -> None:
    _append_gap_issues_impl(
        lines,
        audio_features,
        issues,
        _check_vocal_activity_in_range,
        _check_for_silence_in_range,
    )


def _append_unexpected_pause_issues(
    lines: List[Line],
    audio_features: AudioFeatures,
    issues: List[TimingIssue],
) -> None:
    _append_unexpected_pause_issues_impl(lines, audio_features, issues)


def _check_pause_alignment(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> List[TimingIssue]:
    return _check_pause_alignment_impl(
        lines,
        audio_features,
        _check_vocal_activity_in_range,
        _check_for_silence_in_range,
    )


def _calculate_pause_score_with_stats(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> Tuple[float, int, int]:
    return _calculate_pause_score_with_stats_impl(lines, audio_features)


def _calculate_pause_score(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> float:
    return _calculate_pause_score_impl(lines, audio_features)


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return _merge_intervals_impl(intervals)


def _clip_intervals(
    intervals: List[Tuple[float, float]], window_start: float, window_end: float
) -> List[Tuple[float, float]]:
    return _clip_intervals_impl(intervals, window_start, window_end)


def _complement_intervals(
    window_start: float,
    window_end: float,
    intervals: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    return _complement_intervals_impl(window_start, window_end, intervals)


def _intersect_intervals(
    left: List[Tuple[float, float]],
    right: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    return _intersect_intervals_impl(left, right)


def _total_duration(intervals: List[Tuple[float, float]]) -> float:
    return _total_duration_impl(intervals)


def _compute_audio_coverage_metrics(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> Tuple[float, float, float, int]:
    return _compute_audio_coverage_metrics_impl(lines, audio_features)


def _generate_summary(
    overall: float,
    line_score: float,
    pause_score: float,
    audio_coverage_score: float,
    avg_offset: float,
    std_offset: float,
    num_issues: int,
    total_lines: int,
) -> str:
    return _generate_summary_impl(
        overall,
        line_score,
        pause_score,
        audio_coverage_score,
        avg_offset,
        std_offset,
        num_issues,
        total_lines,
    )
