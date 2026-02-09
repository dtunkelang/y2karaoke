"""Lyrics timing evaluation against audio analysis.

This module compares lyrics timing (from LRC files) against actual audio
characteristics to evaluate timing quality and identify inconsistencies.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

from ..utils.logging import get_logger
from .models import Line
from .timing_models import (  # noqa: F401
    TimingIssue,
    AudioFeatures,
    TimingReport,
    TranscriptionWord,
    TranscriptionSegment,
)
from .phonetic_utils import (  # noqa: F401
    _VOWEL_REGEX,
    _normalize_text_for_matching,
    _normalize_text_for_phonetic,
    _consonant_skeleton,
    _get_epitran,
    _get_panphon_distance,
    _get_panphon_ft,
    _is_vowel,
    _get_ipa,
    _get_ipa_segs,
    _phonetic_similarity,
    _text_similarity_basic,
    _text_similarity,
    _whisper_lang_to_epitran,
    _epitran_cache,
    _ipa_cache,
    _ipa_segs_cache,
)
from .audio_analysis import (  # noqa: F401
    extract_audio_features,
    _get_audio_features_cache_path,
    _load_audio_features_cache,
    _save_audio_features_cache,
    _find_silence_regions,
    _compute_silence_overlap,
    _is_time_in_silence,
    _find_vocal_start,
    _find_vocal_end,
    _check_vocal_activity_in_range,
    _check_for_silence_in_range,
)
from .whisper_integration import *  # noqa: F401, F403
from .whisper_integration import (  # noqa: F401
    _get_whisper_cache_path,
    _load_whisper_cache,
    _save_whisper_cache,
    _find_best_cached_whisper_model,
    _find_best_whisper_match,
    _extract_lrc_words,
    _compute_phonetic_costs,
    _extract_alignments_from_path,
    _apply_dtw_alignments,
    _align_dtw_whisper_with_data,
    _compute_dtw_alignment_metrics,
    _retime_lines_from_dtw_alignments,
    _merge_lines_to_whisper_segments,
    _retime_adjacent_lines_to_whisper_window,
    _retime_adjacent_lines_to_segment_window,
    _pull_next_line_into_segment_window,
    _pull_next_line_into_same_segment,
    _merge_short_following_line_into_segment,
    _pull_lines_near_segment_end,
    _clamp_repeated_line_duration,
    _tighten_lines_to_whisper_segments,
    _apply_offset_to_line,
    _calculate_drift_correction,
    _fix_ordering_violations,
    _find_best_whisper_segment,
    _assess_lrc_quality,
    _pull_lines_to_best_segments,
    _model_index,
    _MODEL_ORDER,
    _fill_vocal_activity_gaps,
    _pull_lines_forward_for_continuous_vocals,
    _merge_first_two_lines_if_segment_matches,
    transcribe_vocals,
    align_dtw_whisper,
    align_lyrics_to_transcription,
    align_words_to_whisper,
    correct_timing_with_whisper,
    align_lrc_text_to_whisper_timings,
)

logger = get_logger(__name__)


def evaluate_timing(
    lines: List[Line],
    audio_features: AudioFeatures,
    source_name: str = "unknown",
) -> TimingReport:
    """Evaluate lyrics timing against audio features.

    Compares line start times against detected onsets and
    identifies timing inconsistencies.

    Args:
        lines: Lyrics lines with timing
        audio_features: Extracted audio features
        source_name: Name of the lyrics source for reporting

    Returns:
        TimingReport with quality scores and identified issues
    """
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
                        " {'before' if onset_delta > 0 else 'after'} detected onset",
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

    # Confidence weighting based on coverage
    line_confidence = (len(line_offsets) / total_lines) if total_lines > 0 else 1.0
    if total_silences > 0:
        pause_confidence = matched_silences / total_silences
    else:
        pause_confidence = 1.0
    confidence = 0.7 * line_confidence + 0.3 * pause_confidence

    # Overall score: weighted combination
    overall_score = (0.7 * line_alignment_score + 0.3 * pause_score) * confidence

    # Calculate statistics
    avg_offset = float(np.mean(line_offsets)) if line_offsets else 0.0
    std_offset = float(np.std(line_offsets)) if len(line_offsets) > 1 else 0.0

    # Generate summary
    summary = _generate_summary(
        float(overall_score),
        float(line_alignment_score),
        float(pause_score),
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
    )


def _find_closest_onset(
    target_time: float,
    onset_times: np.ndarray,
    max_distance: float = 2.0,
) -> Tuple[Optional[float], float]:
    """Find the closest onset to a target time.

    Returns:
        Tuple of (onset_time, delta) where delta = target_time - onset_time
        Returns (None, 0) if no onset within max_distance
    """
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
                        " {i+1} and {i+2} has no corresponding silence in audio",
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
        return 100.0, 0, 0  # No silence to match

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


def _generate_summary(
    overall: float,
    line_score: float,
    pause_score: float,
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
    ]

    if num_issues > 0:
        lines.append(f"  Issues found: {num_issues}")

    return "\n".join(lines)


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


def _find_best_onset_for_phrase_end(
    onset_times: np.ndarray,
    line_start: float,
    prev_line_audio_end: float,
    audio_features: AudioFeatures,
) -> Optional[float]:
    """Find best onset when LRC timestamp is at end of phrase (silence follows)."""
    search_start = max(0, prev_line_audio_end + 0.2)
    candidate_onsets = onset_times[
        (onset_times >= search_start) & (onset_times < line_start)
    ]

    for onset in candidate_onsets:
        onset_silence_before = _check_vocal_activity_in_range(
            max(0, onset - 0.5), onset - 0.1, audio_features
        )
        if onset_silence_before < 0.3:
            return onset
    return None


def _find_best_onset_proximity(
    onset_times: np.ndarray,
    line_start: float,
    max_correction: float,
    audio_features: AudioFeatures,
) -> Optional[float]:
    """Find best onset using proximity-based scoring."""
    search_start = line_start - max_correction
    search_end = line_start + max_correction
    candidate_onsets = onset_times[
        (onset_times >= search_start) & (onset_times <= search_end)
    ]

    best_onset = None
    best_score = float("inf")

    for onset in candidate_onsets:
        distance = abs(onset - line_start)
        onset_silence_before = _check_vocal_activity_in_range(
            max(0, onset - 0.5), onset - 0.1, audio_features
        )

        score = distance
        if onset_silence_before < 0.3:
            score -= 1.0  # Bonus for following silence
        if onset < line_start - 0.5:
            score += 1.0  # Penalty for being too early

        if score < best_score:
            best_score = score
            best_onset = onset

    return best_onset


def _find_best_onset_during_silence(
    onset_times: np.ndarray,
    line_start: float,
    prev_line_audio_end: float,
    max_correction: float,
    audio_features: AudioFeatures,
) -> Optional[float]:
    """Find best onset when LRC timestamp falls during silence."""
    search_after = max(prev_line_audio_end + 0.2, line_start - max_correction)
    search_before = line_start + max_correction
    candidate_onsets = onset_times[
        (onset_times >= search_after) & (onset_times <= search_before)
    ]

    for onset in candidate_onsets:
        silence_before = _check_vocal_activity_in_range(
            max(0, onset - 0.5), onset - 0.1, audio_features
        )
        if silence_before < 0.3:
            return onset
    return None


def correct_line_timestamps(
    lines: List[Line],
    audio_features: Optional[AudioFeatures],
    max_correction: float = 3.0,
) -> Tuple[List[Line], List[str]]:
    """Correct line timestamps to align with detected vocal onsets.

    Uses a two-tier approach:
    1. If LRC timestamp has singing nearby, use proximity-based correction (normal case)
    2. If LRC timestamp falls during silence, find next phrase after previous line (fallback)

    Args:
        lines: Original lyrics lines with timing
        audio_features: Extracted audio features with onset times
        max_correction: Maximum correction for normal cases (seconds)

    Returns:
        Tuple of (corrected_lines, list of correction descriptions)
    """
    from .models import Line, Word

    if not lines:
        return lines, []
    if audio_features is None:
        return lines, ["Audio features unavailable; skipping onset corrections"]

    corrected_lines: List[Line] = []
    corrections: List[str] = []
    onset_times = audio_features.onset_times
    prev_line_audio_end = 0.0

    # Global shift if lyrics start well before detected vocals
    first_line = next((line for line in lines if line.words), None)
    if first_line is not None:
        first_start = first_line.start_time
        vocal_start = audio_features.vocal_start
        if vocal_start > 0 and first_start < vocal_start - 0.5:
            global_offset = vocal_start - first_start
            shifted_lines: List[Line] = []
            for line in lines:
                if not line.words:
                    shifted_lines.append(line)
                    continue
                new_words = [
                    Word(
                        text=word.text,
                        start_time=word.start_time + global_offset,
                        end_time=word.end_time + global_offset,
                        singer=word.singer,
                    )
                    for word in line.words
                ]
                shifted_lines.append(Line(words=new_words, singer=line.singer))
            lines = shifted_lines
            corrections.append(
                f"Global shift {global_offset:+.1f}s to align with vocal start"
            )

    for i, line in enumerate(lines):
        if not line.words:
            corrected_lines.append(line)
            continue

        line_start = line.start_time
        singing_at_lrc_time = (
            _check_vocal_activity_in_range(
                line_start - 0.5, line_start + 0.5, audio_features
            )
            > 0.3
        )

        best_onset = None

        if len(onset_times) > 0:
            if singing_at_lrc_time:
                silence_after = (
                    _check_vocal_activity_in_range(
                        line_start + 0.1, line_start + 0.6, audio_features
                    )
                    < 0.3
                )
                silence_before = (
                    _check_vocal_activity_in_range(
                        max(0, line_start - 0.6), line_start - 0.1, audio_features
                    )
                    < 0.3
                )

                if silence_after and not silence_before:
                    best_onset = _find_best_onset_for_phrase_end(
                        onset_times, line_start, prev_line_audio_end, audio_features
                    )
                else:
                    best_onset = _find_best_onset_proximity(
                        onset_times, line_start, max_correction, audio_features
                    )
            else:
                best_onset = _find_best_onset_during_silence(
                    onset_times,
                    line_start,
                    prev_line_audio_end,
                    max_correction,
                    audio_features,
                )

            if best_onset is not None:
                offset = best_onset - line_start

                if abs(offset) > 0.3:
                    new_words = [
                        Word(
                            text=word.text,
                            start_time=word.start_time + offset,
                            end_time=word.end_time + offset,
                            singer=word.singer,
                        )
                        for word in line.words
                    ]

                    corrected_lines.append(Line(words=new_words, singer=line.singer))
                    prev_line_audio_end = _find_phrase_end(
                        best_onset,
                        best_onset + 30.0,
                        audio_features,
                        min_silence_duration=0.3,
                    )

                    line_text = " ".join(w.text for w in line.words)[:30]
                    corrections.append(
                        f'Line {i+1} shifted {offset:+.1f}s: "{line_text}..."'
                    )
                    continue

        corrected_lines.append(line)
        prev_line_audio_end = _find_phrase_end(
            line_start, line_start + 30.0, audio_features, min_silence_duration=0.3
        )

    return corrected_lines, corrections




def _find_phrase_end(
    start_time: float,
    max_end_time: float,
    audio_features: AudioFeatures,
    min_silence_duration: float = 0.3,
) -> float:
    """Find where a phrase actually ends by detecting silence.

    Searches for the first silence region after start_time that lasts
    at least min_silence_duration seconds.

    Args:
        start_time: When the phrase starts
        max_end_time: Don't search beyond this time
        audio_features: Audio features with energy envelope
        min_silence_duration: Minimum silence to consider phrase end

    Returns:
        Estimated end time of the phrase
    """
    times = audio_features.energy_times
    energy = audio_features.energy_envelope

    # Use 2% of peak as silence threshold (consistent with _check_vocal_activity_in_range)
    peak_level = np.max(energy) if len(energy) > 0 else 1.0
    silence_threshold = 0.02 * peak_level

    # Find start index
    start_idx = np.searchsorted(times, start_time)

    # Track silence duration
    in_silence = False
    silence_start_time = 0.0

    for i in range(start_idx, len(times)):
        t = times[i]
        if t > max_end_time:
            break

        is_silent = energy[i] < silence_threshold

        if is_silent and not in_silence:
            # Entering silence
            in_silence = True
            silence_start_time = t
        elif not is_silent and in_silence:
            # Exiting silence - check if it was long enough
            silence_duration = t - silence_start_time
            if silence_duration >= min_silence_duration:
                # Found real phrase end
                return silence_start_time
            in_silence = False

    # If still in silence at max_end_time, check duration
    if in_silence:
        silence_duration = min(max_end_time, times[-1]) - silence_start_time
        if silence_duration >= min_silence_duration:
            return silence_start_time

    # No silence found - use max_end_time
    return max_end_time


def fix_spurious_gaps(  # noqa: C901
    lines: List[Line],
    audio_features: AudioFeatures,
    activity_threshold: float = 0.5,
) -> Tuple[List[Line], List[str]]:
    """Fix spurious gaps by merging lines that should be continuous.

    When a gap between lines has significant vocal activity (indicating
    continuous singing), merge the lines into one. Uses audio analysis
    to find the actual end of the merged phrase rather than using
    potentially incorrect LRC timestamps.

    Args:
        lines: Original lyrics lines
        audio_features: Extracted audio features
        activity_threshold: Minimum vocal activity fraction to trigger merge

    Returns:
        Tuple of (fixed_lines, list of fix descriptions)
    """
    from .models import Line

    if not lines:
        return lines, []

    fixed_lines: List[Line] = []
    fixes: List[str] = []
    i = 0

    while i < len(lines):
        current_line = lines[i]

        if not current_line.words:
            fixed_lines.append(current_line)
            i += 1
            continue

        lines_to_merge, j = _collect_lines_to_merge(
            lines, i, audio_features, activity_threshold
        )

        if len(lines_to_merge) > 1:
            next_line_start = (
                lines[j].start_time if j < len(lines) and lines[j].words else None
            )
            merged_line, phrase_end = _merge_lines_with_audio(
                lines_to_merge, next_line_start, audio_features
            )
            fixed_lines.append(merged_line)

            phrase_start = lines_to_merge[0].start_time
            merged_texts = [
                " ".join(w.text for w in line.words)[:20] for line in lines_to_merge
            ]
            fixes.append(
                f"Merged {len(lines_to_merge)} lines ({i+1}-{i+len(lines_to_merge)}): "
                f"duration {phrase_end - phrase_start:.1f}s - "
                f"\"{' + '.join(merged_texts)}...\""
            )

            i = j  # Skip all merged lines
            continue

        # No merge needed, keep original
        fixed_lines.append(current_line)
        i += 1

    return fixed_lines, fixes


def _collect_lines_to_merge(
    lines: List["Line"],
    start_idx: int,
    audio_features: AudioFeatures,
    activity_threshold: float,
) -> Tuple[List["Line"], int]:
    current_line = lines[start_idx]
    lines_to_merge = [current_line]
    j = start_idx + 1

    while j < len(lines) and lines[j].words:
        next_line = lines[j]
        prev_line = lines_to_merge[-1]
        gap_start = prev_line.end_time
        gap_end = next_line.start_time

        if _should_merge_gap(gap_start, gap_end, audio_features, activity_threshold):
            lines_to_merge.append(next_line)
            j += 1
            continue
        break

    return lines_to_merge, j


def _should_merge_gap(
    gap_start: float,
    gap_end: float,
    audio_features: AudioFeatures,
    activity_threshold: float,
) -> bool:
    gap_duration = gap_end - gap_start
    if gap_duration <= 0:
        return False

    if gap_duration > 2.0:
        mid_activity = _check_vocal_activity_in_range(
            gap_start, gap_end, audio_features
        )
        has_silence = _check_for_silence_in_range(
            gap_start, gap_end, audio_features, min_silence_duration=0.5
        )
        return mid_activity > max(0.6, activity_threshold) and not has_silence

    if gap_duration > 0.3:
        mid_start = gap_start + 0.15
        mid_end = gap_end - 0.15
        if mid_end > mid_start:
            mid_activity = _check_vocal_activity_in_range(
                mid_start, mid_end, audio_features
            )
            return mid_activity > max(0.7, activity_threshold)

    return False


def _merge_lines_with_audio(
    lines_to_merge: List["Line"],
    next_line_start: Optional[float],
    audio_features: AudioFeatures,
) -> Tuple["Line", float]:
    merged_words = []
    for line in lines_to_merge:
        merged_words.extend(list(line.words))

    phrase_start = lines_to_merge[0].start_time
    search_end = next_line_start if next_line_start is not None else phrase_start + 30.0
    phrase_end = _find_phrase_end(
        phrase_start, search_end, audio_features, min_silence_duration=0.3
    )

    min_duration = len(merged_words) * 0.3
    if phrase_end - phrase_start < min_duration:
        phrase_end = phrase_start + min_duration

    total_duration = phrase_end - phrase_start
    word_count = len(merged_words)

    if word_count <= 0:
        return Line(words=[], singer=lines_to_merge[0].singer), phrase_end

    word_spacing = total_duration / word_count
    new_words = []
    for k, word in enumerate(merged_words):
        new_start = phrase_start + k * word_spacing
        new_end = new_start + (word_spacing * 0.9)
        new_words.append(
            Word(
                text=word.text,
                start_time=new_start,
                end_time=new_end,
                singer=word.singer,
            )
        )

    return Line(words=new_words, singer=lines_to_merge[0].singer), phrase_end


# ============================================================================
# Whisper-based transcription and alignment
# ============================================================================


def print_comparison_report(
    title: str,
    artist: str,
    vocals_path: str,
) -> None:
    """Print a detailed comparison report of all lyrics sources.

    Useful for debugging and understanding timing quality differences.
    """
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
