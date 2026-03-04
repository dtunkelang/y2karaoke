"""Scoring and summary helpers for timing evaluator core."""

from typing import List, Optional, Tuple
import numpy as np

from ...models import Line
from .timing_models import TimingIssue


def identify_ordering_issues(lines: List[Line]) -> List[TimingIssue]:
    """Check for lines that end before they start or are out of order."""
    issues = []
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
    return issues


def find_closest_onset(
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


def calculate_line_alignment_score(
    lines: List[Line], onset_times: np.ndarray
) -> Tuple[float, int, List[TimingIssue], List[float]]:
    """Calculate score based on line start alignment with audio onsets."""
    issues = []
    line_offsets = []
    matched_count = 0
    total_lines = len([line for line in lines if line.words])

    for i, line in enumerate(lines):
        if not line.words:
            continue

        line_start = line.start_time
        closest_onset, onset_delta = find_closest_onset(line_start, onset_times)

        if closest_onset is not None:
            line_offsets.append(onset_delta)

            if abs(onset_delta) <= 0.5:
                matched_count += 1

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

    score = (matched_count / total_lines * 100) if total_lines > 0 else 0.0
    return score, matched_count, issues, line_offsets


def generate_summary(
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
