"""Lyrics source comparison and selection based on timing quality."""

import logging
from typing import Optional, Tuple, Dict

from ..utils.logging import get_logger
from .timing_models import TimingReport
from . import audio_analysis
from . import timing_evaluator_core

extract_audio_features = audio_analysis.extract_audio_features
evaluate_timing = timing_evaluator_core.evaluate_timing

logger = get_logger(__name__)


def compare_sources(
    title: str,
    artist: str,
    vocals_path: str,
) -> Dict[str, TimingReport]:
    """Compare timing quality across all available lyrics sources.

    Fetches lyrics from all available sources, evaluates each against
    the audio, and returns reports for comparison.

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio file

    Returns:
        Dict mapping source name to TimingReport
    """
    from .sync import fetch_from_all_sources
    from .lrc import parse_lrc_with_timing, create_lines_from_lrc

    # Extract audio features once
    audio_features = extract_audio_features(vocals_path)
    if audio_features is None:
        logger.error("Failed to extract audio features")
        return {}

    # Fetch from all sources
    sources = fetch_from_all_sources(title, artist)

    reports: Dict[str, TimingReport] = {}

    for source_name, (lrc_text, lrc_duration) in sources.items():
        if not lrc_text:
            continue

        try:
            # Parse and create lines
            lines = create_lines_from_lrc(
                lrc_text, romanize=False, title=title, artist=artist
            )

            # Apply timing from parsed LRC
            timings = parse_lrc_with_timing(lrc_text, title, artist)
            for i, line in enumerate(lines):
                if i < len(timings):
                    line_start = timings[i][0]
                    next_start = (
                        timings[i + 1][0] if i + 1 < len(timings) else line_start + 5.0
                    )
                    word_count = len(line.words)
                    if word_count > 0:
                        word_duration = (next_start - line_start) * 0.95 / word_count
                        for j, word in enumerate(line.words):
                            word.start_time = line_start + j * word_duration
                            word.end_time = word.start_time + word_duration

            # Evaluate timing
            report = evaluate_timing(lines, audio_features, source_name)
            reports[source_name] = report

            logger.info(f"{source_name}: {report.summary}")

        except Exception as e:
            logger.warning(f"Failed to evaluate {source_name}: {e}")

    return reports


def select_best_source(
    title: str,
    artist: str,
    vocals_path: str,
    target_duration: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str], Optional[TimingReport]]:
    """Select the best lyrics source based on timing quality.

    Compares all available sources and returns the one with the best
    timing alignment to the audio.

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio file
        target_duration: Expected track duration (for additional validation)

    Returns:
        Tuple of (lrc_text, source_name, timing_report) or (None, None, None)
    """
    from .sync import fetch_from_all_sources

    reports = compare_sources(title, artist, vocals_path)

    if not reports:
        logger.warning("No lyrics sources available for comparison")
        return None, None, None

    # Score each source
    best_source = None
    best_score = -1.0
    best_report = None
    best_duration_diff: Optional[float] = None

    sources = fetch_from_all_sources(title, artist)

    for source_name, report in reports.items():
        score = report.overall_score
        duration_diff = None

        # Bonus for matching target duration
        if target_duration:
            lrc_text, lrc_duration = sources.get(source_name, (None, None))
            if lrc_duration:
                duration_diff = abs(lrc_duration - target_duration)
                if duration_diff <= 10:
                    score += 10
                elif duration_diff <= 20:
                    score += 5

        if score > best_score:
            best_score = score
            best_source = source_name
            best_report = report
            best_duration_diff = duration_diff
        elif score == best_score:
            if duration_diff is not None and best_duration_diff is not None:
                if duration_diff < best_duration_diff:
                    best_source = source_name
                    best_report = report
                    best_duration_diff = duration_diff
            elif duration_diff is not None and best_duration_diff is None:
                best_source = source_name
                best_report = report
                best_duration_diff = duration_diff
            elif duration_diff is None and best_duration_diff is None and best_report:
                best_alignment = (
                    best_report.line_alignment_score + best_report.pause_alignment_score
                )
                current_alignment = (
                    report.line_alignment_score + report.pause_alignment_score
                )
                if current_alignment > best_alignment:
                    best_source = source_name
                    best_report = report
                    best_duration_diff = duration_diff

    if best_source and best_source in sources:
        lrc_text, _ = sources[best_source]
        logger.info(f"Selected best source: {best_source} (score: {best_score:.1f})")
        return lrc_text, best_source, best_report

    return None, None, None


def print_comparison_report(
    title: str,
    artist: str,
    vocals_path: str,
) -> None:
    """Print a detailed comparison report of all lyrics sources."""
    print(f"\n{'='*60}")
    print(f"Lyrics Timing Comparison: {artist} - {title}")
    print(f"{'='*60}\n")

    reports = compare_sources(title, artist, vocals_path)

    if not reports:
        print("No lyrics sources found.")
        return

    # Sort by score
    sorted_reports = sorted(
        reports.items(), key=lambda x: x[1].overall_score, reverse=True
    )

    for i, (source_name, report) in enumerate(sorted_reports):
        rank = "★" if i == 0 else f"{i+1}"
        print(f"\n[{rank}] {source_name}")
        print("-" * 40)
        print(report.summary)

        if report.issues:
            print("\nTop issues:")
            severe_issues = [iss for iss in report.issues if iss.severity == "severe"]
            moderate_issues = [
                iss for iss in report.issues if iss.severity == "moderate"
            ]

            for issue in (severe_issues + moderate_issues)[:5]:
                print(f"  [{issue.severity.upper()}] {issue.description}")

    print(f"\n{'='*60}")
    if sorted_reports:
        best_source = sorted_reports[0][0]
        best_score = sorted_reports[0][1].overall_score
        print(f"Recommended source: {best_source} (score: {best_score:.1f})")
    print(f"{'='*60}\n")
