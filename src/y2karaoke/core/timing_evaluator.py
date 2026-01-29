"""Lyrics timing evaluation against audio analysis.

This module compares lyrics timing (from LRC files) against actual audio
characteristics to evaluate timing quality and identify inconsistencies.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np

from ..utils.logging import get_logger
from .models import Line

logger = get_logger(__name__)


@dataclass
class TimingIssue:
    """Represents a timing inconsistency between lyrics and audio."""
    issue_type: str  # "early_line", "late_line", "missing_pause", "unexpected_pause"
    line_index: int
    lyrics_time: float
    audio_time: Optional[float]
    delta: float  # positive = lyrics ahead of audio
    severity: str  # "minor", "moderate", "severe"
    description: str


@dataclass
class AudioFeatures:
    """Extracted audio features for timing evaluation."""
    onset_times: np.ndarray  # Detected onset times
    silence_regions: List[Tuple[float, float]]  # (start, end) of silent regions
    vocal_start: float  # First vocal onset
    vocal_end: float  # Last vocal activity
    duration: float  # Total audio duration
    energy_envelope: np.ndarray  # RMS energy over time
    energy_times: np.ndarray  # Time axis for energy envelope


@dataclass
class TimingReport:
    """Comprehensive timing evaluation report."""
    source_name: str
    overall_score: float  # 0-100, higher is better
    line_alignment_score: float  # How well line starts match onsets
    pause_alignment_score: float  # How well pauses match silence
    issues: List[TimingIssue] = field(default_factory=list)
    summary: str = ""

    # Detailed metrics
    avg_line_offset: float = 0.0  # Average offset between lyrics and audio
    std_line_offset: float = 0.0  # Standard deviation of offsets
    matched_onsets: int = 0  # Lines that matched an onset
    total_lines: int = 0


def extract_audio_features(
    vocals_path: str,
    min_silence_duration: float = 0.5,
) -> Optional[AudioFeatures]:
    """Extract audio features for timing evaluation.

    Args:
        vocals_path: Path to vocals audio file
        min_silence_duration: Minimum duration (seconds) to consider as silence

    Returns:
        AudioFeatures object or None if extraction fails
    """
    try:
        import librosa

        # Load audio
        y, sr = librosa.load(vocals_path, sr=22050)
        duration = len(y) / sr
        hop_length = 512

        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length, backtrack=True, units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        # RMS energy for silence detection
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # Compute silence threshold
        noise_floor = np.percentile(rms, 10)
        peak_level = np.percentile(rms, 90)
        silence_threshold = noise_floor + 0.15 * (peak_level - noise_floor)

        # Find silence regions
        is_silent = rms < silence_threshold
        silence_regions = _find_silence_regions(is_silent, rms_times, min_silence_duration)

        # Find vocal start and end
        vocal_start = _find_vocal_start(onset_times, rms, rms_times, silence_threshold)
        vocal_end = _find_vocal_end(rms, rms_times, silence_threshold)

        return AudioFeatures(
            onset_times=onset_times,
            silence_regions=silence_regions,
            vocal_start=vocal_start,
            vocal_end=vocal_end,
            duration=duration,
            energy_envelope=rms,
            energy_times=rms_times,
        )

    except Exception as e:
        logger.error(f"Failed to extract audio features: {e}")
        return None


def _find_silence_regions(
    is_silent: np.ndarray,
    times: np.ndarray,
    min_duration: float,
) -> List[Tuple[float, float]]:
    """Find regions of sustained silence."""
    regions = []
    in_silence = False
    silence_start = 0.0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            silence_start = times[i]
        elif not silent and in_silence:
            in_silence = False
            silence_end = times[i]
            if silence_end - silence_start >= min_duration:
                regions.append((silence_start, silence_end))

    # Handle trailing silence
    if in_silence:
        silence_end = times[-1]
        if silence_end - silence_start >= min_duration:
            regions.append((silence_start, silence_end))

    return regions


def _find_vocal_start(
    onset_times: np.ndarray,
    rms: np.ndarray,
    rms_times: np.ndarray,
    threshold: float,
    min_duration: float = 0.3,
) -> float:
    """Find where vocals actually start."""
    if len(onset_times) == 0:
        return 0.0

    # Validate onsets with sustained energy
    for onset_time in onset_times:
        onset_idx = np.searchsorted(rms_times, onset_time)
        if onset_idx >= len(rms):
            continue

        # Check for sustained energy after onset
        min_frames = int(min_duration / (rms_times[1] - rms_times[0]) if len(rms_times) > 1 else 10)
        end_idx = min(onset_idx + min_frames, len(rms))

        if end_idx > onset_idx:
            frames_above = rms[onset_idx:end_idx] > threshold
            if np.mean(frames_above) > 0.5:
                return onset_time

    return onset_times[0] if len(onset_times) > 0 else 0.0


def _find_vocal_end(
    rms: np.ndarray,
    rms_times: np.ndarray,
    threshold: float,
    min_silence: float = 0.5,
) -> float:
    """Find where vocals end (last sustained activity)."""
    if len(rms) == 0:
        return 0.0

    # Find last frame above threshold with sustained activity before it
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] > threshold:
            return rms_times[i]

    return rms_times[-1]


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
                severity = "minor" if abs(onset_delta) < 0.5 else (
                    "moderate" if abs(onset_delta) < 1.0 else "severe"
                )
                issue_type = "early_line" if onset_delta > 0 else "late_line"
                issues.append(TimingIssue(
                    issue_type=issue_type,
                    line_index=i,
                    lyrics_time=line_start,
                    audio_time=closest_onset,
                    delta=onset_delta,
                    severity=severity,
                    description=f"Line {i+1} starts {abs(onset_delta):.2f}s {'before' if onset_delta > 0 else 'after'} detected onset"
                ))

    # Check for pause alignment
    pause_issues = _check_pause_alignment(lines, audio_features)
    issues.extend(pause_issues)

    # Calculate scores
    total_lines = len([l for l in lines if l.words])

    # Line alignment score: based on how many lines match onsets within tolerance
    line_alignment_score = (matched_count / total_lines * 100) if total_lines > 0 else 0

    # Pause alignment score
    pause_score = _calculate_pause_score(lines, audio_features)

    # Overall score: weighted combination
    overall_score = 0.7 * line_alignment_score + 0.3 * pause_score

    # Calculate statistics
    avg_offset = np.mean(line_offsets) if line_offsets else 0.0
    std_offset = np.std(line_offsets) if len(line_offsets) > 1 else 0.0

    # Generate summary
    summary = _generate_summary(
        overall_score, line_alignment_score, pause_score,
        avg_offset, std_offset, len(issues), total_lines
    )

    return TimingReport(
        source_name=source_name,
        overall_score=overall_score,
        line_alignment_score=line_alignment_score,
        pause_alignment_score=pause_score,
        issues=issues,
        summary=summary,
        avg_line_offset=avg_offset,
        std_line_offset=std_offset,
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


def _check_pause_alignment(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> List[TimingIssue]:
    """Check if gaps in lyrics align with silence/pauses in audio.

    Detects two types of issues:
    1. Spurious gaps: Lyrics have a gap but audio shows continuous singing
    2. Missing gaps: Audio has silence but lyrics don't reflect it
    """
    issues = []

    for i in range(len(lines) - 1):
        if not lines[i].words or not lines[i+1].words:
            continue

        line_end = lines[i].end_time
        next_start = lines[i+1].start_time
        gap_duration = next_start - line_end

        # Check any gap > 0.5 seconds for spurious pauses
        # (shorter gaps are normal between lines)
        if gap_duration > 0.5:
            # Check if there's vocal activity during this gap
            vocal_activity = _check_vocal_activity_in_range(
                line_end, next_start, audio_features
            )

            if vocal_activity > 0.5:  # More than 50% of gap has vocal activity
                # This is a spurious gap - lyrics say pause but audio shows singing
                severity = "severe" if vocal_activity > 0.8 else (
                    "moderate" if vocal_activity > 0.6 else "minor"
                )
                line_text = " ".join(w.text for w in lines[i].words)[:30]
                next_text = " ".join(w.text for w in lines[i+1].words)[:30]
                issues.append(TimingIssue(
                    issue_type="spurious_gap",
                    line_index=i,
                    lyrics_time=line_end,
                    audio_time=None,
                    delta=gap_duration,
                    severity=severity,
                    description=(
                        f"Gap of {gap_duration:.1f}s between lines {i+1} and {i+2} "
                        f"has {vocal_activity*100:.0f}% vocal activity - likely continuous singing. "
                        f"Lines: \"{line_text}...\" → \"{next_text}...\""
                    )
                ))
            elif gap_duration > 2.0:
                # Large gap - check if there's corresponding silence
                has_silence = any(
                    silence_start < next_start and silence_end > line_end
                    for silence_start, silence_end in audio_features.silence_regions
                )
                if not has_silence:
                    issues.append(TimingIssue(
                        issue_type="missing_pause",
                        line_index=i,
                        lyrics_time=line_end,
                        audio_time=None,
                        delta=gap_duration,
                        severity="moderate",
                        description=f"Gap of {gap_duration:.1f}s between lines {i+1} and {i+2} has no corresponding silence in audio"
                    ))

    # Check for silence regions not covered by lyrics gaps
    for silence_start, silence_end in audio_features.silence_regions:
        silence_duration = silence_end - silence_start
        if silence_duration < 2.0:
            continue

        # Check if any lyrics gap covers this silence
        covered = False
        for i in range(len(lines) - 1):
            if not lines[i].words or not lines[i+1].words:
                continue
            line_end = lines[i].end_time
            next_start = lines[i+1].start_time

            if line_end <= silence_start and next_start >= silence_end:
                covered = True
                break

        if not covered and silence_start > audio_features.vocal_start:
            issues.append(TimingIssue(
                issue_type="unexpected_pause",
                line_index=-1,
                lyrics_time=silence_start,
                audio_time=silence_start,
                delta=silence_duration,
                severity="minor",
                description=f"Silence at {silence_start:.1f}s ({silence_duration:.1f}s) not reflected in lyrics timing"
            ))

    return issues


def _check_vocal_activity_in_range(
    start_time: float,
    end_time: float,
    audio_features: AudioFeatures,
) -> float:
    """Check how much vocal activity exists in a time range.

    Looks for TRUE SILENCE (energy near zero) vs any vocal activity.
    True silence has energy < 2% of peak. Anything above that is activity.

    Returns:
        Fraction of the time range that has vocal activity (0.0 to 1.0)
    """
    # Find frames in this range
    mask = (audio_features.energy_times >= start_time) & (audio_features.energy_times <= end_time)
    range_energy = audio_features.energy_envelope[mask]

    if len(range_energy) == 0:
        return 0.0

    # Normalize to peak energy
    peak_level = np.percentile(audio_features.energy_envelope, 99)
    if peak_level == 0:
        return 0.0

    range_energy_norm = range_energy / peak_level

    # True silence threshold: < 2% of peak energy
    # This catches actual pauses but not quiet singing
    silence_threshold = 0.02

    # Count frames above silence threshold (i.e., frames with any vocal activity)
    active_frames = np.sum(range_energy_norm > silence_threshold)
    return active_frames / len(range_energy)


def _calculate_pause_score(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> float:
    """Calculate how well lyrics pauses align with audio silence."""
    if len(audio_features.silence_regions) == 0:
        return 100.0  # No silence to match

    matched_silences = 0
    total_silences = len([s for s in audio_features.silence_regions
                         if s[1] - s[0] >= 2.0 and s[0] > audio_features.vocal_start])

    if total_silences == 0:
        return 100.0

    for silence_start, silence_end in audio_features.silence_regions:
        if silence_end - silence_start < 2.0:
            continue
        if silence_start < audio_features.vocal_start:
            continue

        # Check if any lyrics gap covers this silence
        for i in range(len(lines) - 1):
            if not lines[i].words or not lines[i+1].words:
                continue
            line_end = lines[i].end_time
            next_start = lines[i+1].start_time

            # Allow some tolerance (0.5s)
            if line_end <= silence_start + 0.5 and next_start >= silence_end - 0.5:
                matched_silences += 1
                break

    return (matched_silences / total_silences * 100) if total_silences > 0 else 100.0


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
    quality = "excellent" if overall >= 90 else (
        "good" if overall >= 75 else (
            "fair" if overall >= 60 else "poor"
        )
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
            lines = create_lines_from_lrc(lrc_text, romanize=False, title=title, artist=artist)

            # Apply timing from parsed LRC
            timings = parse_lrc_with_timing(lrc_text, title, artist)
            for i, line in enumerate(lines):
                if i < len(timings):
                    line_start = timings[i][0]
                    next_start = timings[i + 1][0] if i + 1 < len(timings) else line_start + 5.0
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
    from .sync import fetch_from_all_sources, get_lrc_duration

    reports = compare_sources(title, artist, vocals_path)

    if not reports:
        logger.warning("No lyrics sources available for comparison")
        return None, None, None

    # Score each source
    best_source = None
    best_score = -1
    best_report = None

    sources = fetch_from_all_sources(title, artist)

    for source_name, report in reports.items():
        score = report.overall_score

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

    if best_source and best_source in sources:
        lrc_text, _ = sources[best_source]
        logger.info(f"Selected best source: {best_source} (score: {best_score:.1f})")
        return lrc_text, best_source, best_report

    return None, None, None


def correct_line_timestamps(
    lines: List[Line],
    audio_features: AudioFeatures,
    max_correction: float = 3.0,
) -> Tuple[List[Line], List[str]]:
    """Correct line timestamps to align with detected vocal onsets.

    For each line, finds the best matching vocal onset considering:
    1. Proximity to the LRC timestamp
    2. Whether there's silence before the onset (indicating a phrase start)

    Args:
        lines: Original lyrics lines with timing
        audio_features: Extracted audio features with onset times
        max_correction: Maximum allowed timestamp correction in seconds

    Returns:
        Tuple of (corrected_lines, list of correction descriptions)
    """
    from .models import Line, Word

    if not lines:
        return lines, []

    corrected_lines: List[Line] = []
    corrections: List[str] = []
    onset_times = audio_features.onset_times

    for i, line in enumerate(lines):
        if not line.words:
            corrected_lines.append(line)
            continue

        line_start = line.start_time
        line_duration = line.end_time - line_start

        # Find best onset: prefer onsets that follow silence (phrase boundaries)
        if len(onset_times) > 0:
            best_onset = None
            best_score = float('inf')

            # Check onsets within search window
            search_start = line_start - max_correction
            search_end = line_start + max_correction

            candidate_onsets = onset_times[
                (onset_times >= search_start) & (onset_times <= search_end)
            ]

            for onset in candidate_onsets:
                # Score based on distance from LRC timestamp
                distance = abs(onset - line_start)

                # Check if there's silence before this onset (phrase boundary indicator)
                silence_before = _check_vocal_activity_in_range(
                    max(0, onset - 0.5), onset - 0.1, audio_features
                )

                # Lower score is better
                # Prefer onsets after silence (silence_before < 0.3)
                # Penalize onsets that are much earlier than LRC timestamp
                score = distance
                if silence_before < 0.3:
                    score -= 1.0  # Bonus for following silence
                if onset < line_start - 0.5:
                    score += 1.0  # Penalty for being too early

                if score < best_score:
                    best_score = score
                    best_onset = onset

            if best_onset is not None:
                offset = best_onset - line_start

                # Only correct if offset is significant
                if 0.3 < abs(offset) <= max_correction:
                    # Apply correction - shift all words
                    new_words = []
                    for word in line.words:
                        new_words.append(Word(
                            text=word.text,
                            start_time=word.start_time + offset,
                            end_time=word.end_time + offset,
                            singer=word.singer,
                        ))

                    corrected_line = Line(words=new_words, singer=line.singer)
                    corrected_lines.append(corrected_line)

                    line_text = " ".join(w.text for w in line.words)[:30]
                    corrections.append(
                        f"Line {i+1} shifted {offset:+.1f}s: \"{line_text}...\""
                    )
                    continue

        # No correction needed
        corrected_lines.append(line)

    return corrected_lines, corrections


def fix_spurious_gaps(
    lines: List[Line],
    audio_features: AudioFeatures,
    activity_threshold: float = 0.5,
) -> Tuple[List[Line], List[str]]:
    """Fix spurious gaps by merging lines that should be continuous.

    When a gap between lines has significant vocal activity (indicating
    continuous singing), merge the lines into one.

    Args:
        lines: Original lyrics lines
        audio_features: Extracted audio features
        activity_threshold: Minimum vocal activity fraction to trigger merge

    Returns:
        Tuple of (fixed_lines, list of fix descriptions)
    """
    from .models import Line, Word

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

        # Check if we should merge with next line
        if i + 1 < len(lines) and lines[i + 1].words:
            next_line = lines[i + 1]
            line_end = current_line.end_time
            next_start = next_line.start_time
            gap_duration = next_start - line_end

            if gap_duration > 0.3:  # Only check meaningful gaps
                vocal_activity = _check_vocal_activity_in_range(
                    line_end, next_start, audio_features
                )

                if vocal_activity > activity_threshold:
                    # Merge lines - combine words and redistribute timing
                    merged_words = list(current_line.words) + list(next_line.words)

                    # Redistribute timing across all words
                    total_start = current_line.start_time
                    total_end = next_line.end_time
                    total_duration = total_end - total_start
                    word_count = len(merged_words)

                    if word_count > 0:
                        word_spacing = total_duration / word_count
                        new_words = []
                        for j, word in enumerate(merged_words):
                            new_start = total_start + j * word_spacing
                            new_end = new_start + (word_spacing * 0.9)
                            new_words.append(Word(
                                text=word.text,
                                start_time=new_start,
                                end_time=new_end,
                                singer=word.singer,
                            ))

                        merged_line = Line(words=new_words, singer=current_line.singer)
                        fixed_lines.append(merged_line)

                        current_text = " ".join(w.text for w in current_line.words)[:25]
                        next_text = " ".join(w.text for w in next_line.words)[:25]
                        fixes.append(
                            f"Merged lines {i+1} and {i+2}: "
                            f"\"{current_text}...\" + \"{next_text}...\""
                        )

                        i += 2  # Skip both lines
                        continue

        # No merge needed, keep original
        fixed_lines.append(current_line)
        i += 1

    return fixed_lines, fixes


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
    sorted_reports = sorted(reports.items(), key=lambda x: x[1].overall_score, reverse=True)

    for i, (source_name, report) in enumerate(sorted_reports):
        rank = "★" if i == 0 else f"{i+1}"
        print(f"\n[{rank}] {source_name}")
        print("-" * 40)
        print(report.summary)

        if report.issues:
            print(f"\nTop issues:")
            severe_issues = [iss for iss in report.issues if iss.severity == "severe"]
            moderate_issues = [iss for iss in report.issues if iss.severity == "moderate"]

            for issue in (severe_issues + moderate_issues)[:5]:
                print(f"  [{issue.severity.upper()}] {issue.description}")

    print(f"\n{'='*60}")
    if sorted_reports:
        best_source = sorted_reports[0][0]
        best_score = sorted_reports[0][1].overall_score
        print(f"Recommended source: {best_source} (score: {best_score:.1f})")
    print(f"{'='*60}\n")
