"""Lyrics source comparison and selection based on timing quality."""

from difflib import SequenceMatcher
import re
from typing import Any, Optional, Tuple, Dict

from ....utils.logging import get_logger
from .timing_models import TimingReport
from ... import audio_analysis
from . import timing_evaluator_core

extract_audio_features = audio_analysis.extract_audio_features
evaluate_timing = timing_evaluator_core.evaluate_timing

logger = get_logger(__name__)


def _normalize_line_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"\[[^\]]*\]", " ", lowered)
    lowered = re.sub(r"\([^)]*\)", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _summarize_source(
    *,
    title: str,
    artist: str,
    source_name: str,
    lrc_text: Optional[str],
    duration: Optional[int],
) -> dict[str, Any]:
    from ..lyrics.lrc import parse_lrc_with_timing

    summary: dict[str, Any] = {
        "source_name": source_name,
        "has_lrc": bool(lrc_text),
        "duration": duration,
        "line_count": 0,
        "first_start": None,
        "last_start": None,
        "normalized_text": "",
        "comparable": False,
    }
    if not lrc_text:
        return summary
    try:
        timings = parse_lrc_with_timing(lrc_text, title, artist)
    except Exception:
        return summary
    if not timings:
        return summary

    normalized_lines = [
        _normalize_line_text(text)
        for _ts, text in timings
        if _normalize_line_text(text)
    ]
    summary.update(
        {
            "line_count": len(timings),
            "first_start": float(timings[0][0]),
            "last_start": float(timings[-1][0]),
            "normalized_text": "\n".join(normalized_lines),
            "comparable": True,
        }
    )
    return summary


def analyze_source_disagreement(
    title: str,
    artist: str,
    sources: Dict[str, Tuple[Optional[str], Optional[int]]],
) -> dict[str, Any]:
    """Summarize whether multiple timed-lyrics sources materially disagree."""
    source_summaries = [
        _summarize_source(
            title=title,
            artist=artist,
            source_name=source_name,
            lrc_text=lrc_text,
            duration=duration,
        )
        for source_name, (lrc_text, duration) in sources.items()
    ]
    comparable = [item for item in source_summaries if item["comparable"]]
    if len(comparable) < 2:
        return {
            "source_count": len(source_summaries),
            "comparable_source_count": len(comparable),
            "duration_spread_sec": 0.0,
            "line_count_spread": 0,
            "last_start_spread_sec": 0.0,
            "min_pairwise_text_similarity": 1.0 if comparable else 0.0,
            "flagged": False,
            "reasons": [],
            "sources": source_summaries,
        }

    durations = [float(item["duration"]) for item in comparable if item["duration"]]
    line_counts = [int(item["line_count"]) for item in comparable]
    last_starts = [float(item["last_start"]) for item in comparable]
    duration_spread = (max(durations) - min(durations)) if len(durations) >= 2 else 0.0
    line_count_spread = max(line_counts) - min(line_counts) if line_counts else 0
    last_start_spread = max(last_starts) - min(last_starts) if last_starts else 0.0

    text_similarities: list[float] = []
    for idx, current in enumerate(comparable):
        current_text = str(current["normalized_text"] or "")
        if not current_text:
            continue
        for other in comparable[idx + 1 :]:
            other_text = str(other["normalized_text"] or "")
            if not other_text:
                continue
            text_similarities.append(
                SequenceMatcher(a=current_text, b=other_text).ratio()
            )
    min_text_similarity = min(text_similarities) if text_similarities else 1.0

    reasons: list[str] = []
    if duration_spread >= 8.0:
        reasons.append(f"duration spread {duration_spread:.1f}s")
    if line_count_spread >= 3:
        reasons.append(f"line-count spread {line_count_spread}")
    if last_start_spread >= 8.0:
        reasons.append(f"tail-start spread {last_start_spread:.1f}s")
    if min_text_similarity <= 0.85:
        reasons.append(f"text similarity {min_text_similarity:.2f}")

    return {
        "source_count": len(source_summaries),
        "comparable_source_count": len(comparable),
        "duration_spread_sec": round(duration_spread, 3),
        "line_count_spread": line_count_spread,
        "last_start_spread_sec": round(last_start_spread, 3),
        "min_pairwise_text_similarity": round(min_text_similarity, 4),
        "flagged": bool(reasons),
        "reasons": reasons,
        "sources": source_summaries,
    }


def _score_with_duration_bonus(
    *,
    source_name: str,
    report: TimingReport,
    target_duration: Optional[int],
    sources: Dict[str, Tuple[Optional[str], Optional[int]]],
) -> tuple[float, Optional[float]]:
    score = report.overall_score
    duration_diff: Optional[float] = None
    if not target_duration:
        return score, duration_diff
    _lrc_text, lrc_duration = sources.get(source_name, (None, None))
    if not lrc_duration:
        return score, duration_diff
    duration_diff = abs(lrc_duration - target_duration)
    if duration_diff <= 10:
        score += 10
    elif duration_diff <= 20:
        score += 5
    return score, duration_diff


def _prefer_current_report_on_tie(
    *,
    current_report: TimingReport,
    current_duration_diff: Optional[float],
    best_report: Optional[TimingReport],
    best_duration_diff: Optional[float],
) -> bool:
    if current_duration_diff is not None and best_duration_diff is not None:
        return current_duration_diff < best_duration_diff
    if current_duration_diff is not None and best_duration_diff is None:
        return True
    if current_duration_diff is None and best_duration_diff is not None:
        return False
    if current_duration_diff is None and best_duration_diff is None and best_report:
        best_alignment = (
            best_report.line_alignment_score + best_report.pause_alignment_score
        )
        current_alignment = (
            current_report.line_alignment_score + current_report.pause_alignment_score
        )
        return current_alignment > best_alignment
    return False


def compare_sources(
    title: str,
    artist: str,
    vocals_path: str,
    *,
    sources: Optional[Dict[str, Tuple[Optional[str], Optional[int]]]] = None,
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
    from ..lyrics.sync import fetch_from_all_sources
    from ..lyrics.lrc import parse_lrc_with_timing, create_lines_from_lrc

    # Extract audio features once
    audio_features = extract_audio_features(vocals_path)
    if audio_features is None:
        logger.error("Failed to extract audio features")
        return {}

    # Fetch from all sources
    sources = sources or fetch_from_all_sources(title, artist)

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
    *,
    sources: Optional[Dict[str, Tuple[Optional[str], Optional[int]]]] = None,
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
    from ..lyrics.sync import fetch_from_all_sources

    sources = sources or fetch_from_all_sources(title, artist)
    try:
        reports = compare_sources(title, artist, vocals_path, sources=sources)
    except TypeError as error:
        if "unexpected keyword argument 'sources'" not in str(error):
            raise
        reports = compare_sources(title, artist, vocals_path)

    if not reports:
        logger.warning("No lyrics sources available for comparison")
        return None, None, None

    # Score each source
    best_source = None
    best_score = -1.0
    best_report = None
    best_duration_diff: Optional[float] = None

    for source_name, report in reports.items():
        score, duration_diff = _score_with_duration_bonus(
            source_name=source_name,
            report=report,
            target_duration=target_duration,
            sources=sources,
        )

        if score > best_score:
            best_score = score
            best_source = source_name
            best_report = report
            best_duration_diff = duration_diff
            continue
        if score != best_score:
            continue
        if _prefer_current_report_on_tie(
            current_report=report,
            current_duration_diff=duration_diff,
            best_report=best_report,
            best_duration_diff=best_duration_diff,
        ):
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
