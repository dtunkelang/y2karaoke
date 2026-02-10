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
    line_offsets: List[float] = []
    matched_count = 0

    onset_times = audio_features.onset_times

    # Enforce ordering constraints
    prev_start: Optional[float] = None
    for i, line in enumerate(lines):
        if not line.words:
            continue
        if line.end_time < line.start_time:
            issues.append(
                TimingIssue(
                    issue_type="negative_line_duration",
                    line_index=i,
                    lyrics_time=line.start_time,
                    audio_time=None,
                    delta=line.end_time - line.start_time,
                    severity="severe",
                    description=(
                        f"Line {i+1} has end before start ({line.start_time:.2f}s -> {line.end_time:.2f}s)"
                    ),
                )
            )
        if prev_start is not None and line.start_time < prev_start:
            issues.append(
                TimingIssue(
                    issue_type="out_of_order_line",
                    line_index=i,
                    lyrics_time=line.start_time,
                    audio_time=None,
                    delta=line.start_time - prev_start,
                    severity="severe",
                    description=(
                        f"Line {i+1} starts before previous line ({line.start_time:.2f}s < {prev_start:.2f}s)"
                    ),
                )
            )
        prev_start = line.start_time

    for i, line in enumerate(lines):
        if not line.words:
            continue

        line_start = line.start_time

        # Find closest onset to line start
        closest_onset, onset_delta = _find_closest_onset(line_start, onset_times)

        if closest_onset is not None:
            line_offsets.append(onset_delta)

            # Check if match is reasonable (within 0.5s)
            if abs(onset_delta) <= 0.5:
                matched_count += 1

            # Report significant timing issues
            if abs(onset_delta) > 0.3:
                severity = (
                    "minor"
                    if abs(onset_delta) < 0.5
                    else ("moderate" if abs(onset_delta) < 1.0 else "severe")
                )
                issue_type = "early_line" if onset_delta > 0 else "late_line"
                issues.append(
                    TimingIssue(
                        issue_type=issue_type,
                        line_index=i,
                        lyrics_time=line_start,
                        audio_time=closest_onset,
                        delta=onset_delta,
                        severity=severity,
                        description=f"Line {i+1} starts {abs(onset_delta):.2f}s"
                        f" {'before' if onset_delta > 0 else 'after'} detected onset",
                    )
                )

    # Check for pause alignment
    pause_issues = _check_pause_alignment(lines, audio_features)
    issues.extend(pause_issues)

    # Calculate scores
    total_lines = len([line for line in lines if line.words])

    # Line alignment score: based on how many lines match onsets within tolerance
    line_alignment_score = (
        (matched_count / total_lines * 100) if total_lines > 0 else 0.0
    )

    # Pause alignment score and coverage
    pause_score, matched_silences, total_silences = _calculate_pause_score_with_stats(
        lines, audio_features
    )

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
    audio_coverage_score = 100.0 * (
        0.5 * lyric_on_vocal_ratio + 0.5 * vocal_covered_by_lyrics_ratio
    )

    # Confidence weighting based on coverage
    line_confidence = (len(line_offsets) / total_lines) if total_lines > 0 else 1.0
    if total_silences > 0:
        pause_confidence = matched_silences / total_silences
    else:
        pause_confidence = 1.0
    confidence = 0.7 * line_confidence + 0.3 * pause_confidence

    # Overall score: weighted blend of onset, pause, and vocal coverage.
    overall_score = (
        0.5 * line_alignment_score + 0.2 * pause_score + 0.3 * audio_coverage_score
    ) * confidence

    # Calculate statistics
    avg_offset = float(np.mean(line_offsets)) if line_offsets else 0.0
    std_offset = float(np.std(line_offsets)) if len(line_offsets) > 1 else 0.0

    # Generate summary
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


def _find_closest_onset(
    target_time: float,
    onset_times: np.ndarray,
    max_distance: float = 2.0,
) -> Tuple[Optional[float], float]:
    """Find the closest onset to a target time."""
    if len(onset_times) == 0:
        return None, 0.0

    distances = np.abs(onset_times - target_time)
    min_idx = np.argmin(distances)

    if distances[min_idx] <= max_distance:
        return onset_times[min_idx], target_time - onset_times[min_idx]

    return None, 0.0


def _append_line_spans_silence_issues(
    lines: List[Line],
    audio_features: AudioFeatures,
    issues: List[TimingIssue],
) -> None:
    for i, line in enumerate(lines):
        if not line.words:
            continue
        line_start = line.start_time
        line_end = line.end_time

        for silence_start, silence_end in audio_features.silence_regions:
            silence_duration = silence_end - silence_start
            if silence_duration < 1.0:
                continue
            if silence_start > line_start + 0.5 and silence_end < line_end - 0.5:
                issues.append(
                    TimingIssue(
                        issue_type="line_spans_silence",
                        line_index=i,
                        lyrics_time=line_start,
                        audio_time=silence_start,
                        delta=silence_duration,
                        severity="severe",
                        description=(
                            f"Line {i+1} spans {silence_duration:.1f}s silence "
                            f"({silence_start:.1f}-{silence_end:.1f}s) - likely timing drift"
                        ),
                    )
                )


def _append_gap_issues(
    lines: List[Line],
    audio_features: AudioFeatures,
    issues: List[TimingIssue],
) -> None:
    for i in range(len(lines) - 1):
        if not lines[i].words or not lines[i + 1].words:
            continue

        line_end = lines[i].end_time
        next_start = lines[i + 1].start_time
        gap_duration = next_start - line_end

        if gap_duration <= 0.5:
            continue

        vocal_activity = _check_vocal_activity_in_range(
            line_end, next_start, audio_features
        )

        if vocal_activity > 0.5:
            if gap_duration > 1.0:
                has_silence = _check_for_silence_in_range(
                    line_end, next_start, audio_features, min_silence_duration=0.5
                )
                if not has_silence:
                    issues.append(
                        TimingIssue(
                            issue_type="split_phrase",
                            line_index=i,
                            lyrics_time=line_end,
                            audio_time=None,
                            delta=gap_duration,
                            severity="severe",
                            description=(
                                f"Gap of {gap_duration:.1f}s between lines {i+1} and {i+2} "
                                "has continuous vocals with no silence - likely a split phrase"
                            ),
                        )
                    )

            severity = (
                "severe"
                if vocal_activity > 0.8
                else ("moderate" if vocal_activity > 0.6 else "minor")
            )
            line_text = " ".join(w.text for w in lines[i].words)[:30]
            next_text = " ".join(w.text for w in lines[i + 1].words)[:30]
            issues.append(
                TimingIssue(
                    issue_type="spurious_gap",
                    line_index=i,
                    lyrics_time=line_end,
                    audio_time=None,
                    delta=gap_duration,
                    severity=severity,
                    description=(
                        f"Gap of {gap_duration:.1f}s between lines {i+1} and {i+2} "
                        f"has {vocal_activity*100:.0f}% vocal activity - likely continuous singing. "
                        f'Lines: "{line_text}..." → "{next_text}..."'
                    ),
                )
            )
            continue

        if gap_duration > 2.0:
            has_silence = any(
                silence_start < next_start and silence_end > line_end
                for silence_start, silence_end in audio_features.silence_regions
            )
            if not has_silence:
                issues.append(
                    TimingIssue(
                        issue_type="missing_pause",
                        line_index=i,
                        lyrics_time=line_end,
                        audio_time=None,
                        delta=gap_duration,
                        severity="moderate",
                        description=f"Gap of {gap_duration:.1f}s between lines"
                        f" {i+1} and {i+2} has no corresponding silence in audio",
                    )
                )


def _append_unexpected_pause_issues(
    lines: List[Line],
    audio_features: AudioFeatures,
    issues: List[TimingIssue],
) -> None:
    for silence_start, silence_end in audio_features.silence_regions:
        silence_duration = silence_end - silence_start
        if silence_duration < 2.0:
            continue

        covered = False
        for i in range(len(lines) - 1):
            if not lines[i].words or not lines[i + 1].words:
                continue
            line_end = lines[i].end_time
            next_start = lines[i + 1].start_time

            if line_end <= silence_start and next_start >= silence_end:
                covered = True
                break

        if not covered and silence_start > audio_features.vocal_start:
            issues.append(
                TimingIssue(
                    issue_type="unexpected_pause",
                    line_index=-1,
                    lyrics_time=silence_start,
                    audio_time=silence_start,
                    delta=silence_duration,
                    severity="minor",
                    description=f"Silence at {silence_start:.1f}s ({silence_duration:.1f}s) not reflected in lyrics timing",
                )
            )


def _check_pause_alignment(  # noqa: C901
    lines: List[Line],
    audio_features: AudioFeatures,
) -> List[TimingIssue]:
    """Check if gaps in lyrics align with silence/pauses in audio."""
    issues: List[TimingIssue] = []
    _append_line_spans_silence_issues(lines, audio_features, issues)
    _append_gap_issues(lines, audio_features, issues)
    _append_unexpected_pause_issues(lines, audio_features, issues)
    return issues


def _calculate_pause_score_with_stats(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> Tuple[float, int, int]:
    """Calculate pause score and return matching stats."""
    if len(audio_features.silence_regions) == 0:
        return 100.0, 0, 0

    matched_silences = 0
    total_silences = len(
        [
            s
            for s in audio_features.silence_regions
            if s[1] - s[0] >= 2.0 and s[0] > audio_features.vocal_start
        ]
    )

    if total_silences == 0:
        return 100.0, 0, 0

    for silence_start, silence_end in audio_features.silence_regions:
        if silence_end - silence_start < 2.0:
            continue
        if silence_start < audio_features.vocal_start:
            continue

        # Check if any lyrics gap covers this silence
        for i in range(len(lines) - 1):
            if not lines[i].words or not lines[i + 1].words:
                continue
            line_end = lines[i].end_time
            next_start = lines[i + 1].start_time

            # Allow some tolerance (0.5s)
            if line_end <= silence_start + 0.5 and next_start >= silence_end - 0.5:
                matched_silences += 1
                break

    score = (matched_silences / total_silences * 100) if total_silences > 0 else 100.0
    return score, matched_silences, total_silences


def _calculate_pause_score(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> float:
    """Calculate how well lyrics pauses align with audio silence."""
    score, _matched, _total = _calculate_pause_score_with_stats(lines, audio_features)
    return score


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda interval: interval[0])
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _clip_intervals(
    intervals: List[Tuple[float, float]], window_start: float, window_end: float
) -> List[Tuple[float, float]]:
    clipped: List[Tuple[float, float]] = []
    for start, end in intervals:
        s = max(start, window_start)
        e = min(end, window_end)
        if e > s:
            clipped.append((s, e))
    return _merge_intervals(clipped)


def _complement_intervals(
    window_start: float,
    window_end: float,
    intervals: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if window_end <= window_start:
        return []
    if not intervals:
        return [(window_start, window_end)]
    clipped = _clip_intervals(intervals, window_start, window_end)
    if not clipped:
        return [(window_start, window_end)]
    result: List[Tuple[float, float]] = []
    cursor = window_start
    for start, end in clipped:
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < window_end:
        result.append((cursor, window_end))
    return result


def _intersect_intervals(
    left: List[Tuple[float, float]],
    right: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if not left or not right:
        return []
    intersections: List[Tuple[float, float]] = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        a_start, a_end = left[i]
        b_start, b_end = right[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if end > start:
            intersections.append((start, end))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return intersections


def _total_duration(intervals: List[Tuple[float, float]]) -> float:
    return float(sum(max(0.0, end - start) for start, end in intervals))


def _compute_audio_coverage_metrics(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> Tuple[float, float, float, int]:
    lyric_intervals = [
        (line.start_time, line.end_time)
        for line in lines
        if line.words and line.end_time > line.start_time
    ]
    merged_lyrics = _merge_intervals(lyric_intervals)

    vocal_end = (
        audio_features.vocal_end
        if audio_features.vocal_end > audio_features.vocal_start
        else audio_features.duration
    )
    if merged_lyrics:
        window_start = max(audio_features.vocal_start, merged_lyrics[0][0] - 1.0)
        window_end = min(vocal_end, merged_lyrics[-1][1] + 1.0)
    else:
        window_start = audio_features.vocal_start
        window_end = vocal_end
    if window_end <= window_start:
        return 0.0, 0.0, 0.0, 0

    clipped_lyrics = _clip_intervals(merged_lyrics, window_start, window_end)
    silence = _clip_intervals(audio_features.silence_regions, window_start, window_end)
    non_silent = _complement_intervals(window_start, window_end, silence)

    lyric_duration = _total_duration(clipped_lyrics)
    non_silent_duration = _total_duration(non_silent)
    overlap = _total_duration(_intersect_intervals(clipped_lyrics, non_silent))

    lyric_on_vocal_ratio = overlap / lyric_duration if lyric_duration > 0 else 0.0
    vocal_covered_ratio = (
        overlap / non_silent_duration if non_silent_duration > 0 else 0.0
    )
    lyric_in_silence_ratio = (
        max(0.0, lyric_duration - overlap) / lyric_duration
        if lyric_duration > 0
        else 0.0
    )

    lyric_gaps = _complement_intervals(window_start, window_end, clipped_lyrics)
    uncovered = _intersect_intervals(non_silent, lyric_gaps)
    uncovered_regions = sum(1 for start, end in uncovered if end - start >= 0.5)
    return (
        float(lyric_on_vocal_ratio),
        float(vocal_covered_ratio),
        float(lyric_in_silence_ratio),
        uncovered_regions,
    )


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
    """Generate a human-readable summary."""
    quality = (
        "excellent"
        if overall >= 90
        else ("good" if overall >= 75 else ("fair" if overall >= 60 else "poor"))
    )

    lines = [
        f"Timing quality: {quality} ({overall:.1f}/100)",
        f"  Line alignment: {line_score:.1f}% (avg offset: {avg_offset:+.2f}s, std: {std_offset:.2f}s)",
        f"  Pause alignment: {pause_score:.1f}%",
        f"  Vocal coverage: {audio_coverage_score:.1f}%",
    ]

    if num_issues > 0:
        lines.append(f"  Issues found: {num_issues}")

    return "\n".join(lines)
