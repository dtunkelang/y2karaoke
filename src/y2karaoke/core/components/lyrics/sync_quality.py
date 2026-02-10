"""Quality and validation helpers for synced lyrics."""

import re
from typing import Any, Dict, List, Optional, Tuple


def _has_timestamps(lrc_text: str) -> bool:
    """Check if LRC text contains timestamps."""
    if not lrc_text:
        return False
    timestamp_pattern = r"\[\d{1,2}:\d{2}[.:]\d{2,3}\]"
    return bool(re.search(timestamp_pattern, lrc_text))


def get_lrc_duration(lrc_text: str) -> Optional[int]:
    """Get the implied duration from LRC text based on timestamp span."""
    if not lrc_text or not _has_timestamps(lrc_text):
        return None

    from .lrc import parse_lrc_with_timing

    timings = parse_lrc_with_timing(lrc_text, "", "")
    if not timings or len(timings) < 2:
        return None

    first_ts = timings[0][0]
    last_ts = timings[-1][0]
    lyrics_span = last_ts - first_ts
    buffer = max(3, int(lyrics_span * 0.1))

    return int(last_ts + buffer)


def validate_lrc_quality(
    lrc_text: str, expected_duration: Optional[int] = None
) -> tuple[bool, str]:
    """Validate that an LRC file has sufficient quality for karaoke use."""
    if not lrc_text or not _has_timestamps(lrc_text):
        return False, "No timestamps found"

    from .lrc import parse_lrc_with_timing

    timings = parse_lrc_with_timing(lrc_text, "", "")

    if len(timings) < 5:
        return False, f"Too few timestamped lines ({len(timings)})"

    first_ts = timings[0][0]
    last_ts = timings[-1][0]
    lyrics_span = last_ts - first_ts

    if lyrics_span < 30:
        return False, f"Lyrics span too short ({lyrics_span:.0f}s)"

    density = len(timings) / (lyrics_span / 15) if lyrics_span > 0 else 0
    if density < 1.0:
        return False, f"Timestamp density too low ({density:.2f} per 15s)"

    if expected_duration and expected_duration > 0:
        coverage = lyrics_span / expected_duration
        if coverage < 0.6:
            return False, f"LRC covers only {coverage*100:.0f}% of expected duration"

    return True, ""


def _count_large_gaps(timings: List[Tuple[float, str]], threshold: float = 30.0) -> int:
    """Count gaps between consecutive timestamps exceeding threshold."""
    return sum(
        1
        for i in range(1, len(timings))
        if timings[i][0] - timings[i - 1][0] > threshold
    )


def _calculate_quality_score(report: Dict[str, Any], num_timings: int) -> float:
    """Calculate quality score based on metrics in report."""
    score = 100.0
    issues: List[str] = report["issues"]

    if report["coverage"] < 0.6:
        score -= 30
        issues.append(f"Low coverage ({report['coverage']*100:.0f}%)")
    elif report["coverage"] < 0.8:
        score -= 15

    if report["timestamp_density"] < 1.5:
        score -= 20
        issues.append(f"Low timestamp density ({report['timestamp_density']:.1f}/10s)")
    elif report["timestamp_density"] < 2.0:
        score -= 10

    if not report["duration_match"]:
        score -= 20

    if num_timings < 10:
        score -= 15
        issues.append(f"Only {num_timings} lines")

    return score


def get_lyrics_quality_report(
    lrc_text: str,
    source: str,
    target_duration: Optional[int] = None,
    sources_tried: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a quality report for fetched LRC lyrics."""
    from .lrc import parse_lrc_with_timing

    issues: List[str] = []
    report: Dict[str, Any] = {
        "quality_score": 0.0,
        "source": source,
        "sources_tried": sources_tried or [],
        "coverage": 0.0,
        "timestamp_density": 0.0,
        "duration": None,
        "duration_match": True,
        "issues": issues,
    }

    if not lrc_text or not _has_timestamps(lrc_text):
        issues.append("No synced lyrics found")
        return report

    timings = parse_lrc_with_timing(lrc_text, "", "")
    if not timings or len(timings) < 2:
        report["quality_score"] = 20.0
        issues.append("Too few timestamped lines")
        return report

    lyrics_span = timings[-1][0] - timings[0][0]
    lrc_duration = get_lrc_duration(lrc_text)
    report["duration"] = lrc_duration

    reference_duration = target_duration or lrc_duration or lyrics_span
    if reference_duration > 0:
        report["coverage"] = min(1.0, lyrics_span / reference_duration)

    if lyrics_span > 0:
        report["timestamp_density"] = len(timings) / (lyrics_span / 10.0)

    if target_duration and lrc_duration:
        diff = abs(lrc_duration - target_duration)
        report["duration_match"] = diff <= 8
        if diff > 20:
            issues.append(
                f"Duration mismatch: LRC={lrc_duration}s, target={target_duration}s"
            )

    score = _calculate_quality_score(report, len(timings))

    large_gaps = _count_large_gaps(timings)
    if large_gaps > 2:
        score -= 10
        issues.append(f"{large_gaps} large gaps (>30s)")

    report["quality_score"] = max(0.0, min(100.0, score))
    return report
