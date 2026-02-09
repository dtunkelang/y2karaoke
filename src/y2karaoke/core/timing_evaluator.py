"""Lyrics timing evaluation against audio analysis.

This module compares lyrics timing (from LRC files) against actual audio
characteristics to evaluate timing quality and identify inconsistencies.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

from ..utils.logging import get_logger
from .models import Line, Word
from .lrc import parse_lrc_with_timing
from .timing_models import (
    TimingIssue,
    AudioFeatures,
    TimingReport,
    TranscriptionWord,
    TranscriptionSegment,
)
from .phonetic_utils import (
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
from .audio_analysis import (
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
from .whisper_integration import (
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

__all__ = [
    "evaluate_timing",
    "correct_line_timestamps",
    "fix_spurious_gaps",
    "compare_sources",
    "select_best_source",
    "parse_lrc_with_timing",
    "TimingIssue",
    "AudioFeatures",
    "TimingReport",
    "TranscriptionWord",
    "TranscriptionSegment",
    "_VOWEL_REGEX",
    "_normalize_text_for_matching",
    "_normalize_text_for_phonetic",
    "_consonant_skeleton",
    "_get_epitran",
    "_get_panphon_distance",
    "_get_panphon_ft",
    "_is_vowel",
    "_get_ipa",
    "_get_ipa_segs",
    "_phonetic_similarity",
    "_text_similarity_basic",
    "_text_similarity",
    "_whisper_lang_to_epitran",
    "_epitran_cache",
    "_ipa_cache",
    "_ipa_segs_cache",
    "_load_audio_features_cache",
    "_save_audio_features_cache",
    "_find_silence_regions",
    "_compute_silence_overlap",
    "_is_time_in_silence",
    "_find_vocal_start",
    "_find_vocal_end",
    "_check_vocal_activity_in_range",
    "_check_for_silence_in_range",
    "_get_whisper_cache_path",
    "_load_whisper_cache",
    "_save_whisper_cache",
    "_find_best_cached_whisper_model",
    "_find_best_whisper_match",
    "_extract_lrc_words",
    "_compute_phonetic_costs",
    "_extract_alignments_from_path",
    "_apply_dtw_alignments",
    "_align_dtw_whisper_with_data",
    "_compute_dtw_alignment_metrics",
    "_retime_lines_from_dtw_alignments",
    "_merge_lines_to_whisper_segments",
    "_retime_adjacent_lines_to_whisper_window",
    "_retime_adjacent_lines_to_segment_window",
    "_pull_next_line_into_segment_window",
    "_pull_next_line_into_same_segment",
    "_merge_short_following_line_into_segment",
    "_pull_lines_near_segment_end",
    "_clamp_repeated_line_duration",
    "_tighten_lines_to_whisper_segments",
    "_apply_offset_to_line",
    "_calculate_drift_correction",
    "_fix_ordering_violations",
    "_find_best_whisper_segment",
    "_assess_lrc_quality",
    "_pull_lines_to_best_segments",
    "_model_index",
    "_MODEL_ORDER",
    "_fill_vocal_activity_gaps",
    "_pull_lines_forward_for_continuous_vocals",
    "_merge_first_two_lines_if_segment_matches",
    "transcribe_vocals",
    "align_dtw_whisper",
    "align_lyrics_to_transcription",
    "align_words_to_whisper",
    "correct_timing_with_whisper",
    "align_lrc_text_to_whisper_timings",
]


def evaluate_timing(
    lines: List[Line],
    audio_features: AudioFeatures,
    source_name: str = "unknown",
) -> TimingReport:
    """Evaluate lyrics timing against audio features.

    Compares line start times against detected onsets and
    identifies timing inconsistencies.

    Args:
        lines: Lyrics lines with word-level timing
        audio_features: Extracted audio features with onset times
        source_name: Name of the lyrics source being evaluated

    Returns:
        TimingReport with score and identified issues
    """
    if not lines:
        return TimingReport(overall_score=0.0, issues=[])

    issues = []
    # 1. Onset alignment score
    onset_coverage, onset_issues = _check_onset_alignment(lines, audio_features)
    issues.extend(onset_issues)

    # 2. Pause/silence alignment score
    pause_score, pause_issues = _check_pause_alignment(lines, audio_features)
    issues.extend(pause_issues)

    # 3. Text consistency check (if available)
    # TODO: Add text consistency checks

    # Calculate overall score (weighted average)
    # Onset coverage is primary indicator (70%), pause alignment is secondary (30%)
    overall_score = (onset_coverage * 70.0) + (pause_score * 30.0)

    # Apply penalties for specific issues
    # Out of order lines are a severe issue
    order_violations = _check_line_ordering(lines)
    if order_violations:
        overall_score -= min(overall_score, 40.0)
        issues.append(TimingIssue(
            type="order_violation",
            message=f"Found {len(order_violations)} lines out of chronological order",
            severity="high"
        ))

    return TimingReport(
        overall_score=max(0.0, min(100.0, overall_score)),
        onset_coverage=onset_coverage,
        pause_alignment_score=pause_score,
        issues=issues
    )


def _check_onset_alignment(
    lines: List[Line],
    audio_features: AudioFeatures
) -> Tuple[float, List[TimingIssue]]:
    """Check how well line starts align with audio onsets."""
    if audio_features.onset_times is None or len(audio_features.onset_times) == 0:
        return 0.0, [TimingIssue(
            type="no_onsets",
            message="No audio onsets detected for evaluation",
            severity="medium"
        )]

    matched_onsets = 0
    total_valid_lines = 0
    issues = []

    for i, line in enumerate(lines):
        if not line.words:
            continue
        
        total_valid_lines += 1
        line_start = line.start_time
        
        # Find closest onset within 500ms
        closest_onset = _find_closest_onset(line_start, audio_features.onset_times)
        if closest_onset is not None:
            diff = abs(closest_onset - line_start)
            if diff <= 0.5:
                matched_onsets += 1
            if diff > 1.0:
                issues.append(TimingIssue(
                    type="onset_mismatch",
                    message=f"Line {i+1} starts {diff:.2f}s away from nearest onset",
                    severity="low" if diff < 2.0 else "medium",
                    line_index=i,
                    timestamp=line_start
                ))
        else:
            issues.append(TimingIssue(
                type="no_nearby_onset",
                message=f"No audio onset found near line {i+1} start",
                severity="medium",
                line_index=i,
                timestamp=line_start
            ))

    coverage = matched_onsets / total_valid_lines if total_valid_lines > 0 else 1.0
    return coverage, issues


def _check_line_ordering(lines: List[Line]) -> List[int]:
    """Identify lines that are out of chronological order."""
    violations = []
    prev_start = -1.0
    for i, line in enumerate(lines):
        if not line.words:
            continue
        if line.start_time < prev_start:
            violations.append(i)
        prev_start = line.start_time
    return violations


def _find_closest_onset(target_time: float, onset_times: np.ndarray) -> Optional[float]:
    """Find the onset time closest to target_time."""
    if len(onset_times) == 0:
        return None
    idx = np.abs(onset_times - target_time).argmin()
    return float(onset_times[idx])


def _check_pause_alignment(
    lines: List[Line],
    audio_features: AudioFeatures
) -> Tuple[float, List[TimingIssue]]:
    """Check if lyrics pauses align with audio silence."""
    # Implementation of pause score logic
    # This checks for vocals in gaps and silence during lines
    score = 1.0
    issues = []
    
    # Check for vocals in gaps between lines
    for i in range(len(lines) - 1):
        gap_start = lines[i].end_time
        gap_end = lines[i+1].start_time
        
        if gap_end - gap_start > 1.5:
            # Significant gap - should be silent
            activity = _check_vocal_activity_in_range(gap_start, gap_end, audio_features)
            if activity > 0.5:
                score -= 0.05
                issues.append(TimingIssue(
                    type="vocal_in_gap",
                    message=f"Vocal activity detected in {gap_end - gap_start:.1f}s gap after line {i+1}",
                    severity="medium",
                    line_index=i,
                    timestamp=gap_start
                ))

    return max(0.0, score), issues


def correct_line_timestamps(
    lines: List[Line],
    audio_features: AudioFeatures,
    max_correction: float = 3.0
) -> Tuple[List[Line], List[str]]:
    """Correct line start times based on audio onsets during silence."""
    if not lines:
        return lines, []
    if audio_features is None:
        return lines, ["Audio features unavailable; skipping onset corrections"]

    corrected_lines: List[Line] = []
    corrections: List[str] = []

    for i, line in enumerate(lines):
        if not line.words:
            corrected_lines.append(line)
            continue

        line_start = line.start_time

        # Logic: If line starts during or near silence, snap to nearest onset
        is_silent = _is_time_in_silence(line_start, audio_features.silence_regions)

        if is_silent:
            # Find best onset to snap to
            best_onset = _find_best_onset_during_silence(line_start, audio_features)
            if best_onset is not None:
                offset = best_onset - line_start
                if abs(offset) <= max_correction:
                    # Apply offset to all words in line
                    new_words = [
                        Word(
                            text=w.text,
                            start_time=w.start_time + offset,
                            end_time=w.end_time + offset,
                            singer=w.singer
                        ) for w in line.words
                    ]
                    corrected_lines.append(Line(words=new_words, singer=line.singer))
                    corrections.append(f"Line {i+1} snapped {offset:+.2f}s to audio onset")
                    continue
        
        corrected_lines.append(line)
        
    return corrected_lines, corrections


def _find_best_onset_during_silence(
    target_time: float,
    audio_features: AudioFeatures
) -> Optional[float]:
    """Find the most likely vocal onset near target_time during silence."""
    # Search window: 2s before to 1s after
    search_start = target_time - 2.0
    search_end = target_time + 1.0
    
    candidates = audio_features.onset_times[
        (audio_features.onset_times >= search_start) & 
        (audio_features.onset_times <= search_end)
    ]
    
    if len(candidates) == 0:
        return None
        
    # Pick the one closest to target_time
    idx = np.abs(candidates - target_time).argmin()
    return float(candidates[idx])


def fix_spurious_gaps(
    lines: List[Line],
    audio_features: AudioFeatures,
    threshold: float = 0.6
) -> Tuple[List[Line], List[str]]:
    """Identify and fix large gaps that actually contain continuous vocals."""
    fixed_lines = list(lines)
    fixes = []
    
    for i in range(len(fixed_lines) - 1):
        line1 = fixed_lines[i]
        line2 = fixed_lines[i+1]
        
        if not line1.words or not line2.words:
            continue
            
        gap_start = line1.end_time
        gap_end = line2.start_time
        
        if gap_end - gap_start > 2.0:
            # Check vocal activity in this gap
            activity = _check_vocal_activity_in_range(gap_start, gap_end, audio_features)
            
            if activity > threshold:
                # Vocals present! Merge these lines or extend line1
                # For now, let's just log it as a fix candidate
                fixes.append(f"Detected continuous vocals in {gap_end-gap_start:.1f}s gap after line {i+1}")
                
                # Simple fix: extend line1 to close half the gap
                mid = (gap_start + gap_end) / 2
                line1.words[-1].end_time = mid
                
    return fixed_lines, fixes


def _find_phrase_end(
    start_time: float,
    max_end_time: float,
    audio_features: AudioFeatures,
    min_silence_duration: float = 0.3
) -> float:
    """Find the likely end of a phrase by searching for silence."""
    # Implementation of phrase end detection
    silence_regions = audio_features.silence_regions
    if not silence_regions:
        return max_end_time
        
    for s_start, s_end in silence_regions:
        if s_start >= start_time and s_start < max_end_time:
            if s_end - s_start >= min_silence_duration:
                return s_start
                
    return max_end_time


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

    # Redistribute words within (phrase_start, phrase_end)
    word_count = len(merged_words)
    if word_count > 0:
        duration = phrase_end - phrase_start
        spacing = duration / word_count
        for i, word in enumerate(merged_words):
            word.start_time = phrase_start + i * spacing
            word.end_time = word.start_time + spacing * 0.9

    return Line(words=merged_words, singer=lines_to_merge[0].singer), phrase_end


def compare_sources(
    title: str,
    artist: str,
    vocals_path: str,
    target_duration: Optional[int] = None,
) -> List[Tuple[str, str, TimingReport]]:
    """Fetch lyrics from all sources and evaluate their timing."""
    from .sync import fetch_from_all_sources
    from .audio_analysis import extract_audio_features
    
    audio_features = extract_audio_features(vocals_path)
    if not audio_features:
        logger.warning("Could not extract audio features for evaluation")
        return []
        
    results = []
    sources = fetch_from_all_sources(title, artist)
    
    for source_name, lrc_text in sources.items():
        try:
            # Parse and evaluate
            lines = parse_lrc_with_timing(lrc_text, title, artist)
            report = evaluate_timing(lines, audio_features, source_name)
            results.append((source_name, lrc_text, report))
        except Exception as e:
            logger.debug(f"Evaluation failed for source {source_name}: {e}")
            
    return results


def select_best_source(
    title: str,
    artist: str,
    vocals_path: str,
    target_duration: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str], Optional[TimingReport]]:
    """Select the best lyrics source based on timing evaluation."""
    reports = compare_sources(title, artist, vocals_path, target_duration)
    
    if not reports:
        return None, None, None
        
    # Sort by overall score descending
    reports.sort(key=lambda x: x[2].overall_score, reverse=True)
    
    best_source, best_lrc, best_report = reports[0]
    return best_lrc, best_source, best_report


def print_comparison_report(
    title: str,
    artist: str,
    vocals_path: str,
    target_duration: Optional[int] = None,
) -> None:
    """Print a summary of timing evaluation for all sources."""
    reports = compare_sources(title, artist, vocals_path, target_duration)
    
    print(f"\nTiming evaluation for: {artist} - {title}")
    print("-" * 60)
    print(f"{'Source':<20} | {'Score':<10} | {'Issues':<10}")
    print("-" * 60)
    
    for source, _, report in reports:
        print(f"{source:<20} | {report.overall_score:>10.1f} | {len(report.issues):>10}")
        
    if reports:
        best = max(reports, key=lambda x: x[2].overall_score)
        print("-" * 60)
        print(f"Best source: {best[0]} ({best[2].overall_score:.1f})")