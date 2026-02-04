"""Lyrics timing evaluation against audio analysis.

This module compares lyrics timing (from LRC files) against actual audio
characteristics to evaluate timing quality and identify inconsistencies.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
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


def _get_audio_features_cache_path(vocals_path: str) -> Optional[str]:
    """Get the cache file path for audio features."""
    from pathlib import Path

    vocals_file = Path(vocals_path)
    if not vocals_file.exists():
        return None

    cache_name = f"{vocals_file.stem}_audio_features.npz"
    return str(vocals_file.parent / cache_name)


def _load_audio_features_cache(cache_path: str) -> Optional[AudioFeatures]:
    """Load cached audio features if available."""
    from pathlib import Path

    cache_file = Path(cache_path)
    if not cache_file.exists():
        return None

    try:
        data = np.load(cache_file, allow_pickle=True)

        # Reconstruct AudioFeatures
        features = AudioFeatures(
            onset_times=data["onset_times"],
            silence_regions=list(data["silence_regions"]),
            vocal_start=float(data["vocal_start"]),
            vocal_end=float(data["vocal_end"]),
            duration=float(data["duration"]),
            energy_envelope=data["energy_envelope"],
            energy_times=data["energy_times"],
        )

        logger.debug(
            f"Loaded cached audio features: {len(features.onset_times)} onsets"
        )
        return features

    except Exception as e:
        logger.debug(f"Failed to load audio features cache: {e}")
        return None


def _save_audio_features_cache(cache_path: str, features: AudioFeatures) -> None:
    """Save audio features to cache."""
    try:
        np.savez(
            cache_path,
            onset_times=features.onset_times,
            silence_regions=np.array(features.silence_regions, dtype=object),
            vocal_start=features.vocal_start,
            vocal_end=features.vocal_end,
            duration=features.duration,
            energy_envelope=features.energy_envelope,
            energy_times=features.energy_times,
        )
        logger.debug(f"Saved audio features to cache: {cache_path}")

    except Exception as e:
        logger.debug(f"Failed to save audio features cache: {e}")


def extract_audio_features(
    vocals_path: str,
    min_silence_duration: float = 0.5,
) -> Optional[AudioFeatures]:
    """Extract audio features for timing evaluation.

    Results are cached to disk alongside the vocals file to avoid
    expensive re-extraction on subsequent runs.

    Args:
        vocals_path: Path to vocals audio file
        min_silence_duration: Minimum duration (seconds) to consider as silence

    Returns:
        AudioFeatures object or None if extraction fails
    """
    # Check cache first
    cache_path = _get_audio_features_cache_path(vocals_path)
    if cache_path:
        cached = _load_audio_features_cache(cache_path)
        if cached:
            return cached

    try:
        import librosa

        # Load audio
        y, sr = librosa.load(vocals_path, sr=22050)
        duration = len(y) / sr
        hop_length = 512

        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length, backtrack=True, units="frames"
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        # RMS energy for silence detection
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )

        # Compute silence threshold
        noise_floor = np.percentile(rms, 10)
        peak_level = np.percentile(rms, 90)
        silence_threshold = noise_floor + 0.15 * (peak_level - noise_floor)

        # Find silence regions
        is_silent = rms < silence_threshold
        silence_regions = _find_silence_regions(
            is_silent, rms_times, min_silence_duration
        )

        # Find vocal start and end
        vocal_start = _find_vocal_start(onset_times, rms, rms_times, silence_threshold)
        vocal_end = _find_vocal_end(rms, rms_times, silence_threshold)

        features = AudioFeatures(
            onset_times=onset_times,
            silence_regions=silence_regions,
            vocal_start=vocal_start,
            vocal_end=vocal_end,
            duration=duration,
            energy_envelope=rms,
            energy_times=rms_times,
        )

        # Save to cache
        if cache_path:
            _save_audio_features_cache(cache_path, features)

        return features

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
        min_frames = int(
            min_duration / (rms_times[1] - rms_times[0]) if len(rms_times) > 1 else 10
        )
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
    mask = (audio_features.energy_times >= start_time) & (
        audio_features.energy_times <= end_time
    )
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


def _check_for_silence_in_range(
    start_time: float,
    end_time: float,
    audio_features: AudioFeatures,
    min_silence_duration: float = 0.5,
) -> bool:
    """Check if there's a significant silence region within a time range.

    Even if there's some vocal activity in the range, if there's also
    a sustained silence period, it indicates a real phrase boundary.

    Args:
        start_time: Start of range to check
        end_time: End of range to check
        audio_features: Audio features with energy envelope
        min_silence_duration: Minimum silence duration to detect

    Returns:
        True if there's a silence region >= min_silence_duration
    """
    times = audio_features.energy_times
    energy = audio_features.energy_envelope

    # Use 2% of peak as silence threshold
    peak_level = np.max(energy) if len(energy) > 0 else 1.0
    silence_threshold = 0.02 * peak_level

    # Find indices for the range
    start_idx = int(np.searchsorted(times, start_time))
    end_idx = int(np.searchsorted(times, end_time))

    if start_idx >= end_idx or start_idx >= len(energy):
        return False

    # Scan for silence regions
    in_silence = False
    silence_start_time = 0.0

    for i in range(start_idx, min(end_idx, len(times))):
        t = times[i]
        is_silent = energy[i] < silence_threshold

        if is_silent and not in_silence:
            in_silence = True
            silence_start_time = t
        elif not is_silent and in_silence:
            silence_duration = t - silence_start_time
            if silence_duration >= min_silence_duration:
                return True
            in_silence = False

    # Check if still in silence at end of range
    if in_silence:
        silence_duration = (
            min(end_time, times[min(end_idx, len(times) - 1)]) - silence_start_time
        )
        if silence_duration >= min_silence_duration:
            return True

    return False


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
    from .models import Line, Word

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


@dataclass
class TranscriptionWord:
    """A word from Whisper transcription with timing."""

    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class TranscriptionSegment:
    """A segment from Whisper transcription."""

    start: float
    end: float
    text: str
    words: Optional[List[TranscriptionWord]] = None

    def __post_init__(self) -> None:
        if self.words is None:
            self.words = []


def _get_whisper_cache_path(
    vocals_path: str,
    model_size: str,
    language: Optional[str],
    aggressive: bool = False,
) -> Optional[str]:
    """Get the cache file path for Whisper transcription results.

    Cache is stored alongside the vocals file to ensure it's invalidated
    when the audio changes.
    """
    from pathlib import Path

    vocals_file = Path(vocals_path)
    if not vocals_file.exists():
        return None

    # Create cache filename with model and language info
    lang_suffix = f"_{language}" if language else "_auto"
    mode_suffix = "_aggr" if aggressive else ""
    cache_name = f"{vocals_file.stem}_whisper_{model_size}{lang_suffix}{mode_suffix}.json"
    return str(vocals_file.parent / cache_name)


def _load_whisper_cache(
    cache_path: str,
) -> Optional[Tuple[List[TranscriptionSegment], List[TranscriptionWord], str]]:
    """Load cached Whisper transcription if available."""
    import json
    from pathlib import Path

    cache_file = Path(cache_path)
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            data = json.load(f)

        # Reconstruct TranscriptionSegment and TranscriptionWord objects
        segments = []
        all_words = []
        for seg_data in data.get("segments", []):
            seg_words = []
            for w_data in seg_data.get("words", []):
                tw = TranscriptionWord(
                    start=w_data["start"],
                    end=w_data["end"],
                    text=w_data["text"],
                    probability=w_data.get("probability", 1.0),
                )
                seg_words.append(tw)
                all_words.append(tw)
            segments.append(
                TranscriptionSegment(
                    start=seg_data["start"],
                    end=seg_data["end"],
                    text=seg_data["text"],
                    words=seg_words,
                )
            )

        detected_lang = data.get("language", "")
        logger.info(
            f"Loaded cached Whisper transcription: {len(segments)} segments, {len(all_words)} words"
        )
        return segments, all_words, detected_lang

    except Exception as e:
        logger.debug(f"Failed to load Whisper cache: {e}")
        return None


def _save_whisper_cache(
    cache_path: str,
    segments: List[TranscriptionSegment],
    all_words: List[TranscriptionWord],
    language: str,
) -> None:
    """Save Whisper transcription to cache."""
    import json

    try:
        data = {
            "language": language,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "words": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "text": w.text,
                            "probability": w.probability,
                        }
                        for w in (seg.words or [])
                    ],
                }
                for seg in segments
            ],
        }

        with open(cache_path, "w") as f:
            json.dump(data, f)

        logger.debug(f"Saved Whisper transcription to cache: {cache_path}")

    except Exception as e:
        logger.debug(f"Failed to save Whisper cache: {e}")


def transcribe_vocals(
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
    aggressive: bool = False,
) -> Tuple[List[TranscriptionSegment], List[TranscriptionWord], str]:
    """Transcribe vocals using Whisper.

    Results are cached to disk alongside the vocals file to avoid
    expensive re-transcription on subsequent runs.

    Args:
        vocals_path: Path to vocals audio file
        language: Language code (e.g., 'fr', 'en'). Auto-detected if None.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')

    Returns:
        Tuple of (list of TranscriptionSegment, list of all TranscriptionWord, detected language code)
    """
    # Check cache first
    cache_path = _get_whisper_cache_path(
        vocals_path, model_size, language, aggressive
    )
    if cache_path:
        cached = _load_whisper_cache(cache_path)
        if cached:
            return cached

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster-whisper not installed, cannot transcribe")
        return [], [], ""

    try:
        logger.info(f"Loading Whisper model ({model_size})...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        logger.info(f"Transcribing vocals{f' in {language}' if language else ''}...")
        transcribe_kwargs: Dict[str, object] = {
            "language": language,
            "word_timestamps": True,
            "vad_filter": True,
        }
        if aggressive:
            transcribe_kwargs.update(
                {
                    "vad_filter": False,
                    "no_speech_threshold": 1.0,
                    "log_prob_threshold": -2.0,
                }
            )
        segments, info = model.transcribe(vocals_path, **transcribe_kwargs)

        # Convert to list of TranscriptionSegment with words
        result = []
        all_words = []
        for seg in segments:
            seg_words = []
            if seg.words:
                for w in seg.words:
                    tw = TranscriptionWord(
                        start=w.start,
                        end=w.end,
                        text=w.word.strip(),
                        probability=w.probability,
                    )
                    seg_words.append(tw)
                    all_words.append(tw)
            result.append(
                TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=seg_words,
                )
            )

        detected_lang = info.language
        logger.info(
            f"Transcribed {len(result)} segments, {len(all_words)} words (language: {detected_lang})"
        )

        # Save to cache
        if cache_path:
            _save_whisper_cache(cache_path, result, all_words, detected_lang)

        return result, all_words, detected_lang

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return [], [], ""


def _normalize_text_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching (basic normalization)."""
    import re
    import unicodedata

    # Convert to lowercase
    text = text.lower()

    # Normalize unicode (é -> e, etc.)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _normalize_text_for_phonetic(text: str, language: str) -> str:
    """Normalize text for phonetic matching with light English heuristics."""
    base = _normalize_text_for_matching(text)
    if not base:
        return base

    if not language.startswith("eng") and not language.startswith("en"):
        return base

    tokens = base.split()
    if not tokens:
        return base

    contractions = {
        "im": ["i", "am"],
        "ive": ["i", "have"],
        "ill": ["i", "will"],
        "id": ["i", "would"],
        "youre": ["you", "are"],
        "youve": ["you", "have"],
        "youll": ["you", "will"],
        "youd": ["you", "would"],
        "theyre": ["they", "are"],
        "theyve": ["they", "have"],
        "theyll": ["they", "will"],
        "theyd": ["they", "would"],
        "were": ["we", "are"],
        "weve": ["we", "have"],
        "well": ["we", "will"],
        "wed": ["we", "would"],
        "cant": ["can", "not"],
        "wont": ["will", "not"],
        "dont": ["do", "not"],
        "didnt": ["did", "not"],
        "doesnt": ["does", "not"],
        "isnt": ["is", "not"],
        "arent": ["are", "not"],
        "wasnt": ["was", "not"],
        "werent": ["were", "not"],
        "shouldnt": ["should", "not"],
        "couldnt": ["could", "not"],
        "wouldnt": ["would", "not"],
        "theres": ["there", "is"],
        "thats": ["that", "is"],
        "whats": ["what", "is"],
        "lets": ["let", "us"],
    }

    homophones = {
        "youre": "your",
        "your": "your",
        "theyre": "their",
        "their": "their",
        "there": "their",
        "to": "to",
        "too": "to",
        "two": "to",
        "for": "for",
        "four": "for",
        "its": "its",
    }

    filler_map = {
        "mmm": "mm",
        "mm": "mm",
        "mhm": "mm",
        "hmm": "mm",
        "uh": "uh",
        "uhh": "uh",
        "um": "um",
        "umm": "um",
    }

    expanded: List[str] = []
    for token in tokens:
        if token in contractions:
            expanded.extend(contractions[token])
            continue
        token = homophones.get(token, token)
        token = filler_map.get(token, token)
        expanded.append(token)

    return " ".join(expanded)


def _consonant_skeleton(text: str) -> str:
    """Create a consonant skeleton for quick approximate matching."""
    vowels = set("aeiouy")
    return "".join(ch for ch in text if ch not in vowels)


# Cache for epitran instances (they're expensive to create)
_epitran_cache: Dict[str, Any] = {}
_panphon_distance = None
_panphon_ft = None
_ipa_cache: Dict[str, str] = {}  # Cache for IPA transliterations
_ipa_segs_cache: Dict[str, List[str]] = {}  # Cache for IPA segments


def _get_epitran(language: str = "fra-Latn"):
    """Get or create an epitran instance for a language."""
    if language not in _epitran_cache:
        try:
            import epitran

            _epitran_cache[language] = epitran.Epitran(language)
        except ImportError:
            return None
        except Exception as e:
            logger.debug(f"Could not create epitran for {language}: {e}")
            return None
    return _epitran_cache[language]


def _get_panphon_distance():
    """Get or create a panphon distance calculator."""
    global _panphon_distance
    if _panphon_distance is None:
        try:
            import panphon.distance

            _panphon_distance = panphon.distance.Distance()
        except ImportError:
            return None
    return _panphon_distance


def _get_panphon_ft():
    """Get or create a panphon FeatureTable."""
    global _panphon_ft
    if _panphon_ft is None:
        try:
            import panphon.featuretable

            _panphon_ft = panphon.featuretable.FeatureTable()
        except ImportError:
            return None
    return _panphon_ft


def _get_ipa(text: str, language: str = "fra-Latn") -> Optional[str]:
    """Get IPA transliteration with caching."""
    norm = _normalize_text_for_phonetic(text, language)
    cache_key = f"{language}:{norm}"
    epi = _get_epitran(language)
    if epi is None:
        return None
    if cache_key in _ipa_cache:
        return _ipa_cache[cache_key]

    ipa = epi.transliterate(norm)
    _ipa_cache[cache_key] = ipa
    return ipa


def _get_ipa_segs(ipa: str) -> List[str]:
    """Get IPA segments with caching."""
    if ipa in _ipa_segs_cache:
        return _ipa_segs_cache[ipa]

    ft = _get_panphon_ft()
    if ft is None:
        return []

    segs = ft.ipa_segs(ipa)
    _ipa_segs_cache[ipa] = segs
    return segs


def _phonetic_similarity(text1: str, text2: str, language: str = "fra-Latn") -> float:
    """Calculate phonetic similarity using epitran and panphon.

    Strategy:
    1. Normalize text and convert to IPA via epitran (cached)
    2. Tokenize IPA into phonetic segments via panphon (cached)
    3. Compute weighted Levenshtein distance using panphon's feature_edit_distance
       (substitution cost = phonetic feature distance between segments)
    4. Normalize by segment count to get similarity score

    Args:
        text1: First text
        text2: Second text
        language: Epitran language code (default: French)

    Returns:
        Similarity score from 0.0 to 1.0
    """
    dst = _get_panphon_distance()

    if dst is None:
        return _text_similarity_basic(text1, text2, language)

    try:
        # Get cached IPA transliterations
        ipa1 = _get_ipa(text1, language)
        ipa2 = _get_ipa(text2, language)

        if not ipa1 or not ipa2:
            return _text_similarity_basic(text1, text2, language)

        # Get cached IPA segments
        segs1 = _get_ipa_segs(ipa1)
        segs2 = _get_ipa_segs(ipa2)

        if not segs1 or not segs2:
            return _text_similarity_basic(text1, text2, language)

        # Calculate feature edit distance (weighted Levenshtein)
        fed = dst.feature_edit_distance(ipa1, ipa2)

        # Normalize by max segment count
        # Feature distance is typically 0-1 per substitution, with
        # insertions/deletions costing ~1.0
        max_segs = max(len(segs1), len(segs2))

        # Convert normalized distance to similarity
        # Distance of 0 = identical, distance of max_segs = completely different
        normalized_distance = fed / max_segs
        similarity = max(0.0, 1.0 - normalized_distance)

        norm1 = _normalize_text_for_phonetic(text1, language)
        norm2 = _normalize_text_for_phonetic(text2, language)
        if norm1 and norm2 and norm1 == norm2:
            return max(similarity, 0.98)
        basic_sim = _text_similarity_basic(text1, text2, language)
        blended = max(similarity, basic_sim * 0.9)
        if (
            (language.startswith("eng") or language.startswith("en"))
            and " " not in norm1
            and " " not in norm2
        ):
            sk1 = _consonant_skeleton(norm1)
            sk2 = _consonant_skeleton(norm2)
            if sk1 and sk2:
                from difflib import SequenceMatcher

                sk_sim = SequenceMatcher(None, sk1, sk2).ratio()
                blended = max(blended, sk_sim * 0.95)

        return blended

    except Exception as e:
        logger.debug(f"Phonetic similarity failed: {e}")
        return _text_similarity_basic(text1, text2, language)


def _text_similarity_basic(
    text1: str, text2: str, language: Optional[str] = None
) -> float:
    """Basic text similarity using SequenceMatcher."""
    from difflib import SequenceMatcher

    if language:
        norm1 = _normalize_text_for_phonetic(text1, language)
        norm2 = _normalize_text_for_phonetic(text2, language)
    else:
        norm1 = _normalize_text_for_matching(text1)
        norm2 = _normalize_text_for_matching(text2)

    if not norm1 or not norm2:
        return 0.0

    return SequenceMatcher(None, norm1, norm2).ratio()


def _text_similarity(
    text1: str, text2: str, use_phonetic: bool = True, language: str = "fra-Latn"
) -> float:
    """Calculate similarity between two text strings.

    Args:
        text1: First text
        text2: Second text
        use_phonetic: If True, use phonetic similarity (epitran/panphon)
        language: Language code for phonetic comparison

    Returns:
        Similarity score from 0.0 to 1.0
    """
    if use_phonetic:
        return _phonetic_similarity(text1, text2, language)
    else:
        return _text_similarity_basic(text1, text2)


def align_lyrics_to_transcription(
    lines: List[Line],
    transcription: List[TranscriptionSegment],
    min_similarity: float = 0.4,
    max_time_shift: float = 10.0,
    language: str = "fra-Latn",
) -> Tuple[List[Line], List[str]]:
    """Align lyrics lines to Whisper transcription using fuzzy matching.

    For each lyrics line, finds the best matching transcription segment
    and adjusts timing accordingly. Only applies corrections that are
    within a reasonable time range to avoid catastrophic misalignment.

    Args:
        lines: Lyrics lines with potentially wrong timing
        transcription: Whisper transcription segments with correct timing
        min_similarity: Minimum text similarity to consider a match
        max_time_shift: Maximum time shift to apply (seconds)

    Returns:
        Tuple of (aligned lines, list of alignment descriptions)
    """
    from .models import Line, Word

    if not lines or not transcription:
        return lines, []

    aligned_lines: List[Line] = []
    alignments: List[str] = []

    # Track used segments to avoid double-matching
    used_segments: set = set()

    for i, line in enumerate(lines):
        if not line.words:
            aligned_lines.append(line)
            continue

        line_text = " ".join(w.text for w in line.words)
        line_start = line.start_time

        # Find best matching transcription segment within reasonable time range
        best_match_idx = None
        best_score = -float("inf")
        best_segment = None
        best_similarity = 0.0

        for j, seg in enumerate(transcription):
            if j in used_segments:
                continue

            # Only consider segments within max_time_shift of current position
            time_diff = abs(seg.start - line_start)
            if time_diff > max_time_shift:
                continue

            similarity = _text_similarity(
                line_text, seg.text, use_phonetic=True, language=language
            )

            if similarity < min_similarity:
                continue

            # Score: prioritize similarity, with small time bonus
            time_bonus = max(0, (max_time_shift - time_diff) / max_time_shift) * 0.1
            score = similarity + time_bonus

            if score > best_score:
                best_score = score
                best_similarity = similarity
                best_match_idx = j
                best_segment = seg

        if best_segment is not None and best_similarity >= min_similarity:
            used_segments.add(best_match_idx)

            # Calculate timing adjustment
            offset = best_segment.start - line_start
            new_duration = best_segment.end - best_segment.start

            # Only adjust if offset is significant but not too large
            if 0.3 < abs(offset) <= max_time_shift:
                # Redistribute words across the new duration
                word_count = len(line.words)
                word_spacing = new_duration / word_count if word_count > 0 else 0

                new_words = []
                for k, word in enumerate(line.words):
                    new_start = best_segment.start + k * word_spacing
                    new_end = new_start + (word_spacing * 0.9)
                    new_words.append(
                        Word(
                            text=word.text,
                            start_time=new_start,
                            end_time=new_end,
                            singer=word.singer,
                        )
                    )

                aligned_line = Line(words=new_words, singer=line.singer)
                aligned_lines.append(aligned_line)

                alignments.append(
                    f"Line {i+1} aligned to transcription: {offset:+.1f}s "
                    f'(similarity: {best_similarity:.0%}) "{line_text[:30]}..."'
                )
                continue

        # No good match found or no adjustment needed
        aligned_lines.append(line)

    return aligned_lines, alignments


def _whisper_lang_to_epitran(lang: str) -> str:
    """Map Whisper language code to epitran language code."""
    mapping = {
        "fr": "fra-Latn",
        "en": "eng-Latn",
        "es": "spa-Latn",
        "de": "deu-Latn",
        "it": "ita-Latn",
        "pt": "por-Latn",
        "nl": "nld-Latn",
        "pl": "pol-Latn",
        "ru": "rus-Cyrl",
        "ja": "jpn-Hira",
        "ko": "kor-Hang",
        "zh": "cmn-Hans",
    }
    return mapping.get(lang, "eng-Latn")  # Default to English


def _find_best_whisper_match(
    lrc_text: str,
    lrc_start: float,
    sorted_whisper: List[TranscriptionWord],
    used_indices: set,
    min_similarity: float,
    max_time_shift: float,
    language: str,
) -> Tuple[Optional[TranscriptionWord], Optional[int], float]:
    """Find the best matching Whisper word for an LRC word.

    Returns:
        Tuple of (best_match, best_match_index, best_similarity)
    """
    best_match = None
    best_match_idx = None
    best_similarity = 0.0

    for i, ww in enumerate(sorted_whisper):
        if i in used_indices:
            continue

        time_diff = abs(ww.start - lrc_start)
        if time_diff > max_time_shift:
            if ww.start > lrc_start + max_time_shift:
                break
            continue

        basic_sim = _text_similarity_basic(lrc_text, ww.text, language)
        if basic_sim < 0.25:
            continue

        similarity = _phonetic_similarity(lrc_text, ww.text, language)
        if similarity > best_similarity and similarity >= min_similarity:
            best_similarity = similarity
            best_match = ww
            best_match_idx = i

    return best_match, best_match_idx, best_similarity


def align_words_to_whisper(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    min_similarity: float = 0.5,
    max_time_shift: float = 5.0,
    language: str = "fra-Latn",
) -> Tuple[List[Line], List[str]]:
    """Align individual LRC words to Whisper word timestamps using phonetic matching.

    Args:
        lines: Lyrics lines with word-level timing
        whisper_words: All Whisper transcription words with timestamps
        min_similarity: Minimum phonetic similarity to consider a match
        max_time_shift: Maximum time difference to search for matches
        language: Epitran language code for phonetic matching

    Returns:
        Tuple of (aligned lines, list of correction descriptions)
    """
    from .models import Line, Word

    if not lines or not whisper_words:
        return lines, []

    # Pre-compute IPA for all Whisper words
    logger.debug(f"Pre-computing IPA for {len(whisper_words)} Whisper words...")
    for ww in whisper_words:
        _get_ipa(ww.text, language)

    sorted_whisper = sorted(whisper_words, key=lambda w: w.start)
    used_whisper_indices: set = set()
    aligned_lines: List[Line] = []
    corrections: List[str] = []

    for line in lines:
        if not line.words:
            aligned_lines.append(line)
            continue

        new_words: List[Word] = []
        line_corrections = 0

        for word in line.words:
            lrc_text = word.text.strip()
            if len(lrc_text) < 2:
                new_words.append(word)
                continue

            best_match, best_idx, _ = _find_best_whisper_match(
                lrc_text,
                word.start_time,
                sorted_whisper,
                used_whisper_indices,
                min_similarity,
                max_time_shift,
                language,
            )

            if best_match is not None:
                used_whisper_indices.add(best_idx)
                time_shift = best_match.start - word.start_time
                if abs(time_shift) > 0.15:
                    new_words.append(
                        Word(
                            text=word.text,
                            start_time=best_match.start,
                            end_time=best_match.end,
                            singer=word.singer,
                        )
                    )
                    line_corrections += 1
                else:
                    new_words.append(word)
            else:
                new_words.append(word)

        aligned_lines.append(Line(words=new_words, singer=line.singer))
        if line_corrections > 0:
            line_text = " ".join(w.text for w in line.words)[:40]
            corrections.append(
                f'Line aligned {line_corrections} word(s): "{line_text}..."'
            )

    return aligned_lines, corrections


def _assess_lrc_quality(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    tolerance: float = 1.5,
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """Assess LRC timing quality by comparing against Whisper.

    Returns:
        Tuple of (quality_score 0-1, list of (line_idx, lrc_time, best_whisper_time))
    """
    if not lines or not whisper_words:
        return 1.0, []

    assessments = []
    good_count = 0

    for line_idx, line in enumerate(lines):
        if not line.words:
            continue

        # Get first significant word in line
        first_word = None
        for w in line.words:
            if len(w.text.strip()) >= 2:
                first_word = w
                break
        if not first_word:
            continue

        lrc_time = first_word.start_time
        lrc_text = first_word.text

        # Find best matching Whisper word
        best_whisper_time = None
        best_similarity = 0.0

        for ww in whisper_words:
            # Only consider words within 20s
            if abs(ww.start - lrc_time) > 20:
                continue

            sim = _phonetic_similarity(lrc_text, ww.text, language)
            if sim > best_similarity:
                best_similarity = sim
                best_whisper_time = ww.start

        if best_whisper_time is not None and best_similarity >= 0.5:
            time_diff = abs(best_whisper_time - lrc_time)
            assessments.append((line_idx, lrc_time, best_whisper_time))
            if time_diff <= tolerance:
                good_count += 1

    quality = good_count / len(assessments) if assessments else 1.0
    return quality, assessments


def _extract_lrc_words(lines: List[Line]) -> List[Dict]:
    """Extract all LRC words with their line/word indices."""
    lrc_words = []
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line.words):
            if len(word.text.strip()) >= 2:
                lrc_words.append(
                    {
                        "line_idx": line_idx,
                        "word_idx": word_idx,
                        "text": word.text,
                        "start": word.start_time,
                        "end": word.end_time,
                        "word": word,
                    }
                )
    return lrc_words


def _compute_phonetic_costs(
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
) -> Dict[Tuple[int, int], float]:
    """Compute sparse phonetic cost matrix for DTW."""
    from collections import defaultdict

    phonetic_costs = defaultdict(lambda: 1.0)

    for i, lw in enumerate(lrc_words):
        lrc_time = lw["start"]
        for j, ww in enumerate(whisper_words):
            time_diff = abs(ww.start - lrc_time)
            if time_diff > 20:
                continue
            sim = _phonetic_similarity(lw["text"], ww.text, language)
            if sim >= min_similarity:
                phonetic_costs[(i, j)] = 1.0 - sim

    return phonetic_costs


def _compute_phonetic_costs_unbounded(
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
) -> Dict[Tuple[int, int], float]:
    """Compute phonetic costs without time-window constraints."""
    from collections import defaultdict

    phonetic_costs = defaultdict(lambda: 1.0)

    for i, lw in enumerate(lrc_words):
        for j, ww in enumerate(whisper_words):
            sim = _phonetic_similarity(lw["text"], ww.text, language)
            if sim >= min_similarity:
                phonetic_costs[(i, j)] = 1.0 - sim

    return phonetic_costs


def _extract_best_alignment_map(
    path: List[Tuple[int, int]],
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
) -> Dict[int, Tuple[int, float]]:
    """Extract best whisper index per LRC word index from a DTW path."""
    alignments: Dict[int, Tuple[int, float]] = {}
    for lrc_idx, whisper_idx in path:
        lw = lrc_words[lrc_idx]
        ww = whisper_words[whisper_idx]
        sim = _phonetic_similarity(lw["text"], ww.text, language)
        if lrc_idx not in alignments or sim > alignments[lrc_idx][1]:
            alignments[lrc_idx] = (whisper_idx, sim)
    return alignments


def _extract_lrc_words_all(lines: List[Line]) -> List[Dict]:
    """Extract all LRC words (including short tokens) with line/word indices."""
    lrc_words = []
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line.words):
            if not word.text.strip():
                continue
            lrc_words.append(
                {
                    "line_idx": line_idx,
                    "word_idx": word_idx,
                    "text": word.text,
                    "start": word.start_time,
                    "end": word.end_time,
                    "word": word,
                }
            )
    return lrc_words


def _build_dtw_path(
    lrc_words: List[Dict],
    all_words: List[TranscriptionWord],
    phonetic_costs: Dict[Tuple[int, int], float],
    language: str,
) -> List[Tuple[int, int]]:
    """Build a DTW path between LRC words and Whisper words."""
    try:
        from fastdtw import fastdtw

        lrc_seq = np.arange(len(lrc_words)).reshape(-1, 1)
        whisper_seq = np.arange(len(all_words)).reshape(-1, 1)

        def dtw_dist(a, b):
            i = int(a[0])
            j = int(b[0])
            return phonetic_costs[(i, j)]

        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)
        return path
    except ImportError:
        logger.warning("fastdtw not available, falling back to greedy alignment")
        path = []
        whisper_idx = 0
        for lrc_idx in range(len(lrc_words)):
            best_idx = whisper_idx
            best_sim = -1.0
            for j in range(whisper_idx, min(whisper_idx + 6, len(all_words))):
                sim = _phonetic_similarity(
                    lrc_words[lrc_idx]["text"], all_words[j].text, language
                )
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            path.append((lrc_idx, best_idx))
            whisper_idx = best_idx
        return path


def _map_lrc_words_to_whisper(
    lines: List[Line],
    lrc_words: List[Dict],
    all_words: List[TranscriptionWord],
    lrc_assignments: Dict[int, List[int]],
    language: str,
) -> Tuple[List[Line], int, float, set]:
    """Build mapped lines using Whisper timings based on LRC assignments."""
    from .models import Line, Word

    mapped_lines: List[Line] = []
    mapped_count = 0
    total_similarity = 0.0
    mapped_lines_set = set()

    lrc_index_by_loc = {
        (lw["line_idx"], lw["word_idx"]): idx for idx, lw in enumerate(lrc_words)
    }

    for line_idx, line in enumerate(lines):
        if not line.words:
            mapped_lines.append(line)
            continue

        new_words: List[Word] = []
        for word_idx, word in enumerate(line.words):
            lrc_idx_opt = lrc_index_by_loc.get((line_idx, word_idx))
            if lrc_idx_opt is None:
                new_words.append(word)
                continue
            assigned = lrc_assignments.get(lrc_idx_opt)
            if not assigned:
                new_words.append(word)
                continue
            whisper_words = [all_words[i] for i in assigned]
            start = min(w.start for w in whisper_words)
            end = max(w.end for w in whisper_words)
            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
            mapped_count += 1
            mapped_lines_set.add(line_idx)
            best_sim = max(
                _phonetic_similarity(word.text, ww.text, language)
                for ww in whisper_words
            )
            total_similarity += best_sim

        mapped_lines.append(Line(words=new_words, singer=line.singer))

    return mapped_lines, mapped_count, total_similarity, mapped_lines_set


def _shift_repeated_lines_to_next_whisper(
    mapped_lines: List[Line],
    all_words: List[TranscriptionWord],
) -> List[Line]:
    """Ensure repeated lines don't overlap by shifting to later Whisper words."""
    from .models import Line, Word

    prev_text = None
    prev_end = None
    adjusted_lines: List[Line] = []
    for line in mapped_lines:
        if not line.words:
            adjusted_lines.append(line)
            continue

        text_norm = line.text.strip().lower() if getattr(line, "text", "") else ""
        if prev_text and text_norm == prev_text and prev_end is not None:
            if line.start_time <= prev_end + 0.2:
                required_time = prev_end + 0.21
                start_idx = next(
                    (
                        idx
                        for idx, ww in enumerate(all_words)
                        if ww.start >= required_time
                    ),
                    None,
                )
                if start_idx is not None:
                    adjusted_words: List[Word] = []
                    for word_idx, w in enumerate(line.words):
                        new_idx = min(start_idx + word_idx, len(all_words) - 1)
                        ww = all_words[new_idx]
                        adjusted_words.append(
                            Word(
                                text=w.text,
                                start_time=ww.start,
                                end_time=ww.end,
                                singer=w.singer,
                            )
                        )
                    line = Line(words=adjusted_words, singer=line.singer)
        adjusted_lines.append(line)
        if line.words:
            prev_text = text_norm
            prev_end = line.end_time
    return adjusted_lines


def _enforce_monotonic_line_starts_whisper(
    mapped_lines: List[Line],
    all_words: List[TranscriptionWord],
) -> List[Line]:
    """Ensure line starts are monotonic by shifting backwards lines forward."""
    from .models import Line, Word

    prev_start = None
    prev_end = None
    monotonic_lines: List[Line] = []
    for line in mapped_lines:
        if not line.words:
            monotonic_lines.append(line)
            continue

        if prev_start is not None and line.start_time < prev_start:
            required_time = (prev_end or line.start_time) + 0.01
            start_idx = next(
                (idx for idx, ww in enumerate(all_words) if ww.start >= required_time),
                None,
            )
            if start_idx is not None:
                adjusted_words_2: List[Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words_2.append(
                        Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = Line(words=adjusted_words_2, singer=line.singer)

        monotonic_lines.append(line)
        if line.words:
            prev_start = line.start_time
            prev_end = line.end_time

    return monotonic_lines


def align_lrc_text_to_whisper_timings(  # noqa: C901
    lines: List[Line],
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
    aggressive: bool = False,
    min_similarity: float = 0.15,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
    """Align LRC text to Whisper timings using phonetic DTW (timings fixed).

    This maps each LRC word onto a Whisper word timestamp without changing
    the Whisper timing. The alignment is purely phonetic and monotonic.
    """
    transcription, all_words, detected_lang = transcribe_vocals(
        vocals_path, language, model_size, aggressive
    )

    if not transcription or not all_words:
        logger.warning("No transcription available, skipping Whisper timing map")
        return lines, [], {}

    epitran_lang = _whisper_lang_to_epitran(detected_lang)
    logger.debug(
        f"Using epitran language: {epitran_lang} (from Whisper: {detected_lang})"
    )

    lrc_words = _extract_lrc_words_all(lines)
    if not lrc_words:
        return lines, [], {}

    logger.debug(
        f"DTW-phonetic: Pre-computing IPA for {len(all_words)} Whisper words..."
    )
    for ww in all_words:
        _get_ipa(ww.text, epitran_lang)
    for lw in lrc_words:
        _get_ipa(lw["text"], epitran_lang)

    logger.debug(
        f"DTW-phonetic: Building cost matrix ({len(lrc_words)} x {len(all_words)})..."
    )
    phonetic_costs = _compute_phonetic_costs_unbounded(
        lrc_words, all_words, epitran_lang, min_similarity
    )

    logger.debug("DTW-phonetic: Running alignment...")
    path = _build_dtw_path(lrc_words, all_words, phonetic_costs, epitran_lang)

    # Build LRC->Whisper assignments from DTW path (use Whisper timings directly)
    lrc_assignments: Dict[int, List[int]] = {}
    for lrc_idx, whisper_idx in path:
        lrc_assignments.setdefault(lrc_idx, []).append(whisper_idx)

    # Build mapped lines with whisper timings
    corrections: List[str] = []
    mapped_lines, mapped_count, total_similarity, mapped_lines_set = (
        _map_lrc_words_to_whisper(
            lines, lrc_words, all_words, lrc_assignments, epitran_lang
        )
    )

    mapped_lines = _shift_repeated_lines_to_next_whisper(mapped_lines, all_words)
    mapped_lines = _enforce_monotonic_line_starts_whisper(mapped_lines, all_words)

    matched_ratio = mapped_count / len(lrc_words) if lrc_words else 0.0
    avg_similarity = total_similarity / mapped_count if mapped_count else 0.0
    line_coverage = (
        len(mapped_lines_set) / sum(1 for line in lines if line.words) if lines else 0.0
    )

    metrics = {
        "matched_ratio": matched_ratio,
        "avg_similarity": avg_similarity,
        "line_coverage": line_coverage,
        "dtw_used": 1.0,
        "dtw_mode": 1.0,
    }

    if mapped_count:
        corrections.append(f"DTW-phonetic mapped {mapped_count} word(s) to Whisper")
    return mapped_lines, corrections, metrics


def _extract_alignments_from_path(
    path: List[Tuple[int, int]],
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
) -> Dict[int, Tuple[TranscriptionWord, float]]:
    """Extract validated alignments from DTW path."""
    alignments_map = {}

    for lrc_idx, whisper_idx in path:
        if lrc_idx not in alignments_map:
            ww = whisper_words[whisper_idx]
            lw = lrc_words[lrc_idx]
            sim = _phonetic_similarity(lw["text"], ww.text, language)
            if sim >= min_similarity:
                alignments_map[lrc_idx] = (ww, sim)

    return alignments_map


def _apply_dtw_alignments(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
) -> Tuple[List[Line], List[str]]:
    """Apply DTW alignments to create corrected lines."""
    from .models import Line, Word

    corrections = []
    aligned_lines = []

    for line_idx, line in enumerate(lines):
        if not line.words:
            aligned_lines.append(line)
            continue

        new_words = []
        line_corrections = 0

        for word_idx, word in enumerate(line.words):
            lrc_word_idx = None
            for i, lw in enumerate(lrc_words):
                if lw["line_idx"] == line_idx and lw["word_idx"] == word_idx:
                    lrc_word_idx = i
                    break

            if lrc_word_idx is not None and lrc_word_idx in alignments_map:
                ww, sim = alignments_map[lrc_word_idx]
                time_shift = ww.start - word.start_time

                if abs(time_shift) > 1.0:
                    new_words.append(
                        Word(
                            text=word.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=word.singer,
                        )
                    )
                    line_corrections += 1
                else:
                    new_words.append(word)
            else:
                new_words.append(word)

        aligned_lines.append(Line(words=new_words, singer=line.singer))

        if line_corrections > 0:
            line_text = " ".join(w.text for w in line.words)[:40]
            corrections.append(
                f'DTW aligned {line_corrections} word(s) in line {line_idx}: "{line_text}..."'
            )

    return aligned_lines, corrections


def _compute_dtw_alignment_metrics(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
) -> Dict[str, float]:
    if not lrc_words:
        return {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0}

    total_words = len(lrc_words)
    matched_words = len(alignments_map)
    matched_ratio = matched_words / total_words if total_words else 0.0

    total_similarity = 0.0
    for _, (_ww, sim) in alignments_map.items():
        total_similarity += sim
    avg_similarity = total_similarity / matched_words if matched_words else 0.0

    total_lines = sum(1 for line in lines if line.words)
    matched_lines = {
        lrc_words[lrc_idx]["line_idx"] for lrc_idx in alignments_map.keys()
    }
    line_coverage = len(matched_lines) / total_lines if total_lines > 0 else 0.0

    return {
        "matched_ratio": matched_ratio,
        "avg_similarity": avg_similarity,
        "line_coverage": line_coverage,
    }


def _retime_lines_from_dtw_alignments(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
    min_word_duration: float = 0.05,
) -> Tuple[List[Line], List[str]]:
    from .models import Line, Word

    aligned_by_line: Dict[int, List[Tuple[int, TranscriptionWord]]] = {}
    for lrc_idx, (ww, _sim) in alignments_map.items():
        lw = lrc_words[lrc_idx]
        aligned_by_line.setdefault(lw["line_idx"], []).append((lw["word_idx"], ww))

    retimed_lines: List[Line] = []
    corrections: List[str] = []

    for line_idx, line in enumerate(lines):
        if not line.words:
            retimed_lines.append(line)
            continue

        matches = aligned_by_line.get(line_idx, [])
        if not matches:
            retimed_lines.append(line)
            continue

        matches.sort(key=lambda item: item[0])
        min_start = min(ww.start for _i, ww in matches)
        max_end = max(ww.end for _i, ww in matches)
        if max_end - min_start < min_word_duration:
            max_end = min_start + min_word_duration

        old_start = line.start_time
        old_end = line.end_time
        old_duration = max(old_end - old_start, min_word_duration)
        target_duration = max(max_end - min_start, min_word_duration)

        new_words: List[Word] = []
        for word in line.words:
            rel_start = (word.start_time - old_start) / old_duration
            rel_end = (word.end_time - old_start) / old_duration
            new_start = min_start + rel_start * target_duration
            new_end = min_start + rel_end * target_duration
            if new_end < new_start + min_word_duration:
                new_end = new_start + min_word_duration
            new_words.append(
                Word(
                    text=word.text,
                    start_time=new_start,
                    end_time=new_end,
                    singer=word.singer,
                )
            )

        retimed_lines.append(Line(words=new_words, singer=line.singer))
        corrections.append(f"DTW retimed line {line_idx} from matched words")

    return retimed_lines, corrections


def _align_dtw_whisper_with_data(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    min_similarity: float = 0.4,
) -> Tuple[
    List[Line],
    List[str],
    Dict[str, float],
    List[Dict],
    Dict[int, Tuple[TranscriptionWord, float]],
]:
    """Align LRC to Whisper using DTW and return alignment data for confidence gating."""
    if not lines or not whisper_words:
        return (
            lines,
            [],
            {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0},
            [],
            {},
        )

    lrc_words = _extract_lrc_words(lines)
    if not lrc_words:
        return (
            lines,
            [],
            {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0},
            [],
            {},
        )

    # Pre-compute IPA
    logger.debug(f"DTW: Pre-computing IPA for {len(whisper_words)} Whisper words...")
    for ww in whisper_words:
        _get_ipa(ww.text, language)
    for lw in lrc_words:
        _get_ipa(lw["text"], language)

    logger.debug(
        f"DTW: Building cost matrix ({len(lrc_words)} x {len(whisper_words)})..."
    )
    phonetic_costs = _compute_phonetic_costs(
        lrc_words, whisper_words, language, min_similarity
    )

    # Run DTW
    logger.debug("DTW: Running alignment...")
    try:
        from fastdtw import fastdtw

        lrc_times = np.array([lw["start"] for lw in lrc_words])
        whisper_times = np.array([ww.start for ww in whisper_words])

        lrc_seq = np.column_stack([np.arange(len(lrc_words)), lrc_times])
        whisper_seq = np.column_stack([np.arange(len(whisper_words)), whisper_times])

        def dtw_dist(a, b):
            i, lrc_t = int(a[0]), a[1]
            j, whisper_t = int(b[0]), b[1]
            phon_cost = phonetic_costs[(i, j)]
            time_diff = abs(whisper_t - lrc_t)
            time_penalty = min(time_diff / 20.0, 1.0)
            return 0.7 * phon_cost + 0.3 * time_penalty

        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)

    except ImportError:
        logger.warning("fastdtw not available, falling back to greedy alignment")
        return (
            lines,
            [],
            {"matched_ratio": 0.0, "avg_similarity": 0.0, "line_coverage": 0.0},
            lrc_words,
            {},
        )

    alignments_map = _extract_alignments_from_path(
        path, lrc_words, whisper_words, language, min_similarity
    )

    aligned_lines, corrections = _apply_dtw_alignments(lines, lrc_words, alignments_map)
    metrics = _compute_dtw_alignment_metrics(lines, lrc_words, alignments_map)

    logger.info(f"DTW alignment complete: {len(corrections)} lines modified")
    return aligned_lines, corrections, metrics, lrc_words, alignments_map


def align_dtw_whisper(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    min_similarity: float = 0.4,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
    """Align LRC to Whisper using Dynamic Time Warping.

    DTW finds globally optimal alignment, which handles cases where
    LRC timing errors compound over time.

    Args:
        lines: LRC lines with word-level timing
        whisper_words: Whisper words with timestamps
        language: Epitran language code
        min_similarity: Minimum similarity for valid alignment

    Returns:
        Tuple of (aligned lines, list of corrections, metrics)
    """
    aligned_lines, corrections, metrics, _lrc_words, _alignments = (
        _align_dtw_whisper_with_data(
            lines, whisper_words, language=language, min_similarity=min_similarity
        )
    )
    return aligned_lines, corrections, metrics


def correct_timing_with_whisper(  # noqa: C901
    lines: List[Line],
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
    aggressive: bool = False,
    trust_lrc_threshold: float = 1.0,
    correct_lrc_threshold: float = 1.5,
    force_dtw: bool = False,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
    """Correct lyrics timing using Whisper transcription (adaptive approach).

    Strategy:
    1. Transcribe vocals with Whisper
    2. Assess LRC timing quality (what % of lines are within tolerance of Whisper)
    3. If quality > 70%: LRC is good, only fix individual bad lines
    4. If quality 40-70%: Use hybrid approach (fix bad sections, keep good ones)
    5. If quality < 40%: LRC is broken, use DTW for global alignment

    Args:
        lines: Lyrics lines with potentially wrong timing
        vocals_path: Path to vocals audio
        language: Language code (auto-detected if None)
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        trust_lrc_threshold: If timing error < this, trust LRC (default: 1.0s)
        correct_lrc_threshold: If timing error > this, use Whisper (default: 1.5s)

    Returns:
        Tuple of (corrected lines, list of corrections, metrics)
    """
    # Transcribe vocals (returns segments, all_words, and language)
    transcription, all_words, detected_lang = transcribe_vocals(
        vocals_path, language, model_size, aggressive
    )

    if not transcription:
        logger.warning("No transcription available, skipping Whisper alignment")
        return lines, [], {}

    # Map to epitran language code for phonetic matching
    epitran_lang = _whisper_lang_to_epitran(detected_lang)
    logger.debug(
        f"Using epitran language: {epitran_lang} (from Whisper: {detected_lang})"
    )

    # Pre-compute IPA for Whisper words
    logger.debug(f"Pre-computing IPA for {len(all_words)} Whisper words...")
    for w in all_words:
        _get_ipa(w.text, epitran_lang)

    # Assess LRC quality
    quality, assessments = _assess_lrc_quality(
        lines, all_words, epitran_lang, tolerance=1.5
    )
    logger.info(f"LRC timing quality: {quality:.0%} of lines within 1.5s of Whisper")

    metrics: Dict[str, float] = {}
    if not force_dtw and quality >= 0.7:
        # LRC is mostly good - only fix individual bad lines using hybrid approach
        logger.info("LRC timing is good, using targeted corrections only")
        aligned_lines, alignments = align_hybrid_lrc_whisper(
            lines,
            transcription,
            all_words,
            language=epitran_lang,
            trust_threshold=trust_lrc_threshold,
            correct_threshold=correct_lrc_threshold,
        )
    elif not force_dtw and quality >= 0.4:
        # Mixed quality - use hybrid approach
        logger.info("LRC timing is mixed, using hybrid Whisper alignment")
        aligned_lines, alignments = align_hybrid_lrc_whisper(
            lines,
            transcription,
            all_words,
            language=epitran_lang,
            trust_threshold=trust_lrc_threshold,
            correct_threshold=correct_lrc_threshold,
        )
    else:
        # LRC is broken - use DTW for global alignment
        logger.info("LRC timing is poor, using DTW global alignment")
        aligned_lines, alignments, metrics, lrc_words, alignments_map = (
            _align_dtw_whisper_with_data(
                lines,
                all_words,
                language=epitran_lang,
            )
        )
        metrics["dtw_used"] = 1.0

        matched_ratio = metrics.get("matched_ratio", 0.0)
        avg_similarity = metrics.get("avg_similarity", 0.0)
        line_coverage = metrics.get("line_coverage", 0.0)
        confidence_ok = (
            matched_ratio >= 0.6 and avg_similarity >= 0.5 and line_coverage >= 0.6
        )

        if confidence_ok and lrc_words and alignments_map:
            dtw_lines, dtw_fixes = _retime_lines_from_dtw_alignments(
                lines, lrc_words, alignments_map
            )
            aligned_lines = dtw_lines
            alignments.extend(dtw_fixes)
            metrics["dtw_confidence_passed"] = 1.0
        else:
            metrics["dtw_confidence_passed"] = 0.0
            alignments.append(
                "DTW confidence gating failed; keeping conservative word shifts"
            )

    # Post-process: tighten/merge to Whisper segment boundaries for broken LRC
    if quality < 0.4 or force_dtw:
        aligned_lines, merged_first = _merge_first_two_lines_if_segment_matches(
            aligned_lines, transcription, epitran_lang
        )
        if merged_first:
            alignments.append("Merged first two lines via Whisper segment")
        aligned_lines, pair_retimed = _retime_adjacent_lines_to_whisper_window(
            aligned_lines, transcription, epitran_lang
        )
        if pair_retimed:
            alignments.append(
                f"Retimed {pair_retimed} adjacent line pair(s) to Whisper window"
            )
        aligned_lines, pair_windowed = _retime_adjacent_lines_to_segment_window(
            aligned_lines, transcription, epitran_lang
        )
        if pair_windowed:
            alignments.append(
                f"Retimed {pair_windowed} adjacent line pair(s) to Whisper segment window"
            )
        aligned_lines, pulled_next = _pull_next_line_into_segment_window(
            aligned_lines, transcription, epitran_lang
        )
        if pulled_next:
            alignments.append(
                f"Pulled {pulled_next} line(s) into adjacent segment window"
            )
        aligned_lines, pulled_near_end = _pull_lines_near_segment_end(
            aligned_lines, transcription, epitran_lang
        )
        if pulled_near_end:
            alignments.append(f"Pulled {pulled_near_end} line(s) near segment ends")
        aligned_lines, pulled_same = _pull_next_line_into_same_segment(
            aligned_lines, transcription
        )
        if pulled_same:
            alignments.append(f"Pulled {pulled_same} line(s) into same segment")
        # Re-apply adjacent retiming to keep pairs together after pulls.
        aligned_lines, pair_retimed_after = _retime_adjacent_lines_to_whisper_window(
            aligned_lines,
            transcription,
            epitran_lang,
            max_window_duration=4.5,
            max_start_offset=1.0,
        )
        if pair_retimed_after:
            alignments.append(
                f"Retimed {pair_retimed_after} adjacent line pair(s) after pulls"
            )
        aligned_lines, merged = _merge_lines_to_whisper_segments(
            aligned_lines, transcription, epitran_lang
        )
        if merged:
            alignments.append(f"Merged {merged} line pair(s) via Whisper segments")
        aligned_lines, tightened = _tighten_lines_to_whisper_segments(
            aligned_lines, transcription, epitran_lang
        )
        if tightened:
            alignments.append(f"Tightened {tightened} line(s) to Whisper segments")
        aligned_lines, pulled = _pull_lines_to_best_segments(
            aligned_lines, transcription, epitran_lang
        )
        if pulled:
            alignments.append(f"Pulled {pulled} line(s) to Whisper segments")

    # Post-process: reject corrections that break line ordering
    aligned_lines, alignments = _fix_ordering_violations(
        lines, aligned_lines, alignments
    )
    aligned_lines = _normalize_line_word_timings(aligned_lines)
    aligned_lines = _enforce_monotonic_line_starts(aligned_lines)
    if force_dtw:
        aligned_lines, pulled_near_end = _pull_lines_near_segment_end(
            aligned_lines, transcription, epitran_lang
        )
        if pulled_near_end:
            alignments.append(
                f"Pulled {pulled_near_end} line(s) near segment ends (post-order)"
            )
        aligned_lines, merged_short = _merge_short_following_line_into_segment(
            aligned_lines, transcription
        )
        if merged_short:
            alignments.append(
                f"Merged {merged_short} short line(s) into prior segments"
            )
        aligned_lines, clamped_repeat = _clamp_repeated_line_duration(aligned_lines)
        if clamped_repeat:
            alignments.append(f"Clamped {clamped_repeat} repeated line(s) duration")
    aligned_lines, deduped = _drop_duplicate_lines(
        aligned_lines, transcription, epitran_lang
    )
    if deduped:
        alignments.append(f"Dropped {deduped} duplicate line(s)")
    before_drop = len(aligned_lines)
    aligned_lines = [line for line in aligned_lines if line.words]
    if len(aligned_lines) != before_drop:
        alignments.append("Dropped empty lines after Whisper merges")
    aligned_lines, timing_deduped = _drop_duplicate_lines_by_timing(aligned_lines)
    if timing_deduped:
        alignments.append(
            f"Dropped {timing_deduped} duplicate line(s) by timing overlap"
        )

    if alignments:
        logger.info(f"Whisper hybrid alignment: {len(alignments)} lines corrected")

    return aligned_lines, alignments, metrics


def _enforce_monotonic_line_starts(
    lines: List[Line], min_gap: float = 0.01
) -> List[Line]:
    """Shift lines forward so start times are non-decreasing."""
    from .models import Line, Word

    adjusted: List[Line] = []
    prev_start = None
    for line in lines:
        if not line.words:
            adjusted.append(line)
            continue

        start = line.start_time
        if prev_start is not None and start < prev_start:
            shift = (prev_start - start) + min_gap
            new_words = [
                Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in line.words
            ]
            line = Line(words=new_words, singer=line.singer)

        adjusted.append(line)
        prev_start = line.start_time

    return adjusted


def _merge_lines_to_whisper_segments(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.55,
) -> Tuple[List[Line], int]:
    """Merge consecutive lines that map to the same Whisper segment."""
    from .models import Line, Word

    if not lines or not segments:
        return lines, 0

    merged_lines: List[Line] = []
    merged_count = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.words:
            merged_lines.append(line)
            idx += 1
            continue

        if idx + 1 >= len(lines):
            merged_lines.append(line)
            break

        next_line = lines[idx + 1]
        if not next_line.words:
            merged_lines.append(line)
            idx += 1
            continue

        seg, sim, _ = _find_best_whisper_segment(
            line.text, line.start_time, sorted_segments, language, min_similarity
        )
        next_seg, next_sim, _ = _find_best_whisper_segment(
            next_line.text,
            next_line.start_time,
            sorted_segments,
            language,
            min_similarity,
        )

        if seg is None:
            merged_lines.append(line)
            idx += 1
            continue

        if next_line.text and next_line.text in line.text and sim >= 0.8:
            merged_words = list(line.words)
            word_count = max(len(merged_words), 1)
            total_duration = max(seg.end - seg.start, 0.2)
            spacing = total_duration / word_count
            new_words = []
            for i, word in enumerate(merged_words):
                start = seg.start + i * spacing
                end = start + spacing * 0.9
                new_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            merged_lines.append(Line(words=new_words, singer=line.singer))
            merged_count += 1
            idx += 2
            continue

        combined_text = f"{line.text} {next_line.text}".strip()
        combined_seg, combined_sim, _ = _find_best_whisper_segment(
            combined_text, line.start_time, sorted_segments, language, min_similarity
        )
        if combined_seg is None:
            combined_sim = _phonetic_similarity(combined_text, seg.text, language)

        same_segment = next_seg is seg and next_sim >= min_similarity

        should_merge = False
        if same_segment and sim >= min_similarity:
            should_merge = True
        elif combined_sim >= min_similarity and sim >= min_similarity:
            # Accept merge when the combined line matches the segment better.
            if combined_sim >= max(sim, next_sim) + 0.05:
                should_merge = True
        elif (
            combined_sim >= min_similarity
            and seg.start <= line.start_time + 2.0
            and seg.end >= line.end_time
        ):
            should_merge = True
        elif (
            combined_seg is not None
            and combined_sim >= 0.8
            and combined_seg.start <= line.start_time + 3.0
            and (next_line.start_time - line.end_time) > 2.0
        ):
            seg = combined_seg
            should_merge = True

        if should_merge:
            merged_words = list(line.words) + list(next_line.words)
            word_count = max(len(merged_words), 1)
            total_duration = max(seg.end - seg.start, 0.2)
            spacing = total_duration / word_count
            new_words = []
            for i, word in enumerate(merged_words):
                start = seg.start + i * spacing
                end = start + spacing * 0.9
                new_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            merged_lines.append(Line(words=new_words, singer=line.singer))
            merged_count += 1
            idx += 2
            continue

        merged_lines.append(line)
        idx += 1

    return merged_lines, merged_count


def _retime_adjacent_lines_to_whisper_window(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.35,
    max_gap: float = 2.5,
    min_similarity_late: float = 0.25,
    max_late: float = 3.0,
    max_late_short: float = 6.0,
    max_time_window: float = 15.0,
    max_window_duration: Optional[float] = None,
    max_start_offset: Optional[float] = None,
) -> Tuple[List[Line], int]:
    """Retune adjacent lines into a single Whisper window without merging them."""
    from .models import Line, Word

    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        combined_text = f"{line.text} {next_line.text}".strip()
        prev_end = None
        if idx > 0 and adjusted[idx - 1].words:
            prev_end = adjusted[idx - 1].end_time

        seg, combined_sim, _ = _find_best_whisper_segment(
            combined_text, line.start_time, sorted_segments, language, min_similarity
        )
        if seg is None or combined_sim < min_similarity:
            # Fallback: if the pair is late, try the nearest segment by time.
            nearest = None
            nearest_gap = None
            for cand in sorted_segments:
                gap = abs(cand.start - line.start_time)
                if gap > max_time_window:
                    continue
                if nearest_gap is None or gap < nearest_gap:
                    nearest_gap = gap
                    nearest = cand
            if nearest is None:
                continue
            if line.start_time - nearest.start < max_late:
                continue

            combined_sim = _phonetic_similarity(combined_text, nearest.text, language)
            seg = nearest

            # If the nearest segment is after the line start, try the prior segment instead.
            prior = None
            prior_gap = None
            for cand in sorted_segments:
                if (
                    cand.end <= line.start_time
                    and cand.start >= line.start_time - max_time_window
                ):
                    gap = line.start_time - cand.end
                    if prior_gap is None or gap < prior_gap:
                        prior_gap = gap
                        prior = cand
            if prior is not None and prior_gap is not None:
                prior_sim = _phonetic_similarity(combined_text, prior.text, language)
                if (
                    prior_gap <= max_late_short
                    and (line.start_time - prior.start) > max_late
                    and prior_sim >= min_similarity_late
                ):
                    seg = prior
                    combined_sim = prior_sim

            if combined_sim < min_similarity_late:
                continue
        if (
            max_window_duration is not None
            and (seg.end - seg.start) > max_window_duration
        ):
            continue
        if (
            max_start_offset is not None
            and abs(line.start_time - seg.start) > max_start_offset
        ):
            continue

        total_words = len(line.words) + len(next_line.words)
        if total_words <= 0:
            continue

        window_start = seg.start
        if prev_end is not None:
            window_start = max(window_start, prev_end + 0.01)
        if window_start >= seg.end:
            continue
        if (
            max_window_duration is not None
            and (seg.end - window_start) > max_window_duration
        ):
            continue

        total_duration = max(seg.end - window_start, 0.2)
        spacing = total_duration / total_words
        new_line_words = []
        new_next_words = []

        for i, word in enumerate(line.words):
            start = window_start + i * spacing
            end = start + spacing * 0.9
            new_line_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )

        for j, word in enumerate(next_line.words):
            idx_in_seg = len(line.words) + j
            start = window_start + idx_in_seg * spacing
            end = start + spacing * 0.9
            new_next_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )

        gap = new_next_words[0].start_time - new_line_words[-1].end_time
        if gap > max_gap:
            continue

        adjusted[idx] = Line(words=new_line_words, singer=line.singer)
        adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
        fixes += 1

    return adjusted, fixes


def _retime_adjacent_lines_to_segment_window(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.35,
    max_gap: float = 2.5,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Retune adjacent lines into a two-segment Whisper window."""
    from .models import Line, Word

    if not lines or len(segments) < 2:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        combined_text = f"{line.text} {next_line.text}".strip()

        for s_idx in range(len(sorted_segments) - 1):
            seg_a = sorted_segments[s_idx]
            seg_b = sorted_segments[s_idx + 1]
            if seg_b.start - seg_a.end > 1.0:
                continue
            if abs(seg_a.start - line.start_time) > max_time_window:
                continue

            window_text = f"{seg_a.text} {seg_b.text}".strip()
            combined_sim = _phonetic_similarity(combined_text, window_text, language)
            if combined_sim < min_similarity:
                continue

            total_words = len(line.words) + len(next_line.words)
            if total_words <= 0:
                continue

            window_start = seg_a.start
            window_end = seg_b.end
            total_duration = max(window_end - window_start, 0.2)
            spacing = total_duration / total_words

            new_line_words = []
            new_next_words = []
            for i, word in enumerate(line.words):
                start = window_start + i * spacing
                end = start + spacing * 0.9
                new_line_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            for j, word in enumerate(next_line.words):
                idx_in_seg = len(line.words) + j
                start = window_start + idx_in_seg * spacing
                end = start + spacing * 0.9
                new_next_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )

            gap = new_next_words[0].start_time - new_line_words[-1].end_time
            if gap > max_gap:
                continue

            adjusted[idx] = Line(words=new_line_words, singer=line.singer)
            adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
            fixes += 1
            break

    return adjusted, fixes


def _pull_next_line_into_segment_window(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.2,
    max_late: float = 6.0,
    min_gap: float = 0.05,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Pull a late-following line into the second segment of a window."""
    if not lines or len(segments) < 2:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        nearest = None
        nearest_gap = None
        for cand in sorted_segments:
            gap = abs(cand.start - line.start_time)
            if gap > max_time_window:
                continue
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest = cand
        seg_a = nearest
        if seg_a is None:
            continue

        seg_idx = sorted_segments.index(seg_a)
        if seg_idx + 1 >= len(sorted_segments):
            continue

        seg_b = sorted_segments[seg_idx + 1]
        if seg_b.start - seg_a.end > 1.0:
            continue

        if next_line.start_time < seg_b.end:
            continue

        late_by = next_line.start_time - seg_b.end
        if late_by > max_late:
            continue
        if 0 <= late_by <= 0.5:
            adjusted[idx + 1] = _retime_line_to_segment(next_line, seg_b)
            fixes += 1
            continue

        next_start = None
        if idx + 2 < len(adjusted) and adjusted[idx + 2].words:
            next_start = adjusted[idx + 2].start_time

        if next_start is not None and seg_b.end >= next_start - min_gap:
            continue

        nearest_next = None
        nearest_next_gap = None
        for cand in sorted_segments:
            gap = abs(cand.start - next_line.start_time)
            if gap > max_time_window:
                continue
            if nearest_next_gap is None or gap < nearest_next_gap:
                nearest_next_gap = gap
                nearest_next = cand
        if (
            nearest_next is not None
            and abs(nearest_next.start - seg_b.start) < 1e-6
            and abs(nearest_next.end - seg_b.end) < 1e-6
            and next_line.start_time > seg_b.end
        ):
            adjusted[idx + 1] = _retime_line_to_segment(next_line, seg_b)
            fixes += 1
            continue

        word_count = len(next_line.words)
        sim_b = _phonetic_similarity(next_line.text, seg_b.text, language)
        if sim_b < min_similarity and word_count > 4:
            continue

        adjusted[idx + 1] = _retime_line_to_segment(next_line, seg_b)
        fixes += 1

    return adjusted, fixes


def _pull_next_line_into_same_segment(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    max_late: float = 6.0,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Pull the next line into the same segment as the current line when late."""
    from .models import Line, Word

    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        nearest = None
        nearest_gap = None
        for cand in sorted_segments:
            gap = abs(cand.start - line.start_time)
            if gap > max_time_window:
                continue
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest = cand
        if nearest is None:
            continue

        late_by = next_line.start_time - nearest.end
        if late_by < 0 or late_by > max_late:
            continue

        min_start = line.end_time + 0.01
        # If the next line is short and starts late, retime both into the same segment.
        if len(next_line.words) <= 3 and late_by > 0:
            total_words = len(line.words) + len(next_line.words)
            if total_words <= 0:
                continue
            window_start = nearest.start
            if idx > 0 and adjusted[idx - 1].words:
                window_start = max(window_start, adjusted[idx - 1].end_time + 0.01)
            if window_start >= nearest.end:
                continue
            total_duration = max(nearest.end - window_start, 0.2)
            spacing = total_duration / total_words
            new_line_words = []
            new_next_words = []
            for i, word in enumerate(line.words):
                start = window_start + i * spacing
                end = start + spacing * 0.9
                new_line_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            for j, word in enumerate(next_line.words):
                idx_in_seg = len(line.words) + j
                start = window_start + idx_in_seg * spacing
                end = start + spacing * 0.9
                new_next_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            adjusted[idx] = Line(words=new_line_words, singer=line.singer)
            adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
        elif min_start >= nearest.end:
            # Not enough room: retime both lines into the segment window.
            total_words = len(line.words) + len(next_line.words)
            if total_words <= 0:
                continue
            window_start = nearest.start
            total_duration = max(nearest.end - window_start, 0.2)
            spacing = total_duration / total_words
            new_line_words = []
            new_next_words = []
            for i, word in enumerate(line.words):
                start = window_start + i * spacing
                end = start + spacing * 0.9
                new_line_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            for j, word in enumerate(next_line.words):
                idx_in_seg = len(line.words) + j
                start = window_start + idx_in_seg * spacing
                end = start + spacing * 0.9
                new_next_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            adjusted[idx] = Line(words=new_line_words, singer=line.singer)
            adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
        else:
            adjusted[idx + 1] = _retime_line_to_segment_with_min_start(
                next_line, nearest, min_start
            )
        fixes += 1

    return adjusted, fixes


def _merge_short_following_line_into_segment(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    max_late: float = 6.0,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Merge a short following line into the same segment window as the current line."""
    from .models import Line, Word

    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        if len(next_line.words) > 3:
            continue

        nearest = None
        nearest_gap = None
        for cand in sorted_segments:
            gap = abs(cand.start - line.start_time)
            if gap > max_time_window:
                continue
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest = cand
        if nearest is None:
            continue

        late_by = next_line.start_time - nearest.end
        if late_by < 0 or late_by > max_late:
            continue

        total_words = len(line.words) + len(next_line.words)
        if total_words <= 0:
            continue

        window_start = nearest.start
        if idx > 0 and adjusted[idx - 1].words:
            window_start = max(window_start, adjusted[idx - 1].end_time + 0.01)
        if window_start >= nearest.end:
            continue

        total_duration = max(nearest.end - window_start, 0.2)
        spacing = total_duration / total_words
        new_line_words = []
        new_next_words = []
        for i, word in enumerate(line.words):
            start = window_start + i * spacing
            end = start + spacing * 0.9
            new_line_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
        for j, word in enumerate(next_line.words):
            idx_in_seg = len(line.words) + j
            start = window_start + idx_in_seg * spacing
            end = start + spacing * 0.9
            new_next_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )

        adjusted[idx] = Line(words=new_line_words, singer=line.singer)
        adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
        fixes += 1

    return adjusted, fixes


def _pull_lines_near_segment_end(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    max_late: float = 0.5,
    max_late_short: float = 6.0,
    min_similarity: float = 0.35,
    min_short_duration: float = 0.35,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Pull lines that start just after a nearby segment end back into it."""
    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx, line in enumerate(adjusted):
        if not line.words:
            continue
        prev_end = None
        if idx > 0 and adjusted[idx - 1].words:
            prev_end = adjusted[idx - 1].end_time
        next_start = None
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_start = adjusted[idx + 1].start_time
        nearest = None
        nearest_late = None
        for cand in sorted_segments:
            if abs(cand.start - line.start_time) > max_time_window:
                continue
            late_by = line.start_time - cand.end
            if late_by < 0:
                continue
            if nearest_late is None or late_by < nearest_late:
                nearest_late = late_by
                nearest = cand
        if nearest is None:
            continue

        late_by = line.start_time - nearest.end
        if late_by < 0:
            continue

        allow_late = late_by <= max_late
        if not allow_late:
            word_count = len(line.words)
            # Short lines can be pulled if they're only modestly late.
            if word_count <= 6 and late_by <= max_late_short:
                allow_late = True
            else:
                sim = _phonetic_similarity(line.text, nearest.text, language)
                if (
                    word_count <= 3
                    and sim >= min_similarity
                    and late_by <= max_late_short
                ):
                    allow_late = True

        if not allow_late:
            continue

        min_start = (prev_end + 0.01) if prev_end is not None else nearest.start
        if len(line.words) <= 2:
            target_end = max(nearest.end, min_start + min_short_duration)
            if next_start is not None:
                target_end = min(target_end, next_start - 0.01)
            adjusted[idx] = _retime_line_to_window(line, min_start, target_end)
        else:
            adjusted[idx] = _retime_line_to_segment_with_min_start(
                line, nearest, min_start
            )
        fixes += 1

    return adjusted, fixes


def _clamp_repeated_line_duration(
    lines: List[Line],
    max_duration: float = 1.5,
    min_gap: float = 0.01,
) -> Tuple[List[Line], int]:
    """Clamp duration for immediately repeated lines so they don't hang too long."""
    from .models import Line, Word

    if not lines:
        return lines, 0

    adjusted = list(lines)
    fixes = 0

    for idx in range(1, len(adjusted)):
        prev = adjusted[idx - 1]
        line = adjusted[idx]
        if not prev.words or not line.words:
            continue

        if prev.text.strip() != line.text.strip():
            continue

        duration = line.end_time - line.start_time
        if duration <= max_duration:
            continue

        new_end = line.start_time + max_duration
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_start = adjusted[idx + 1].start_time
            new_end = min(new_end, next_start - min_gap)
        if new_end <= line.start_time:
            continue

        total_duration = max(new_end - line.start_time, 0.2)
        word_count = max(len(line.words), 1)
        spacing = total_duration / word_count
        new_words = []
        for i, word in enumerate(line.words):
            start = line.start_time + i * spacing
            end = start + spacing * 0.9
            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
        adjusted[idx] = Line(words=new_words, singer=line.singer)
        fixes += 1

    return adjusted, fixes


def _merge_first_two_lines_if_segment_matches(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.8,
) -> Tuple[List[Line], bool]:
    """Merge the first two non-empty lines if Whisper sees them as one phrase."""
    from .models import Line, Word

    if not lines or not segments:
        return lines, False

    first_idx = None
    second_idx = None
    for idx, line in enumerate(lines):
        if line.words:
            if first_idx is None:
                first_idx = idx
            else:
                second_idx = idx
                break

    if first_idx is None or second_idx is None:
        return lines, False

    first_line = lines[first_idx]
    second_line = lines[second_idx]
    combined_text = f"{first_line.text} {second_line.text}".strip()

    seg, sim, _ = _find_best_whisper_segment(
        combined_text,
        first_line.start_time,
        sorted(segments, key=lambda s: s.start),
        language,
        min_similarity,
    )

    if seg is None or sim < min_similarity:
        return lines, False

    merged_words = list(first_line.words) + list(second_line.words)
    word_count = max(len(merged_words), 1)
    total_duration = max(seg.end - seg.start, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(merged_words):
        start = seg.start + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )

    lines[first_idx] = Line(words=new_words, singer=first_line.singer)
    lines[second_idx] = Line(words=[], singer=second_line.singer)
    return lines, True


def _tighten_lines_to_whisper_segments(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.5,
) -> Tuple[List[Line], int]:
    """Pull consecutive lines to Whisper segment boundaries when phrased as one."""
    from .models import Line, Word

    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(1, len(adjusted)):
        prev_line = adjusted[idx - 1]
        line = adjusted[idx]
        if not prev_line.words or not line.words:
            continue

        line_text = line.text.strip()
        if not line_text:
            continue

        prev_seg, prev_sim, _ = _find_best_whisper_segment(
            prev_line.text,
            prev_line.start_time,
            sorted_segments,
            language,
            min_similarity,
        )
        seg, sim, _ = _find_best_whisper_segment(
            line_text, line.start_time, sorted_segments, language, min_similarity
        )

        if seg is None or prev_seg is None:
            continue

        # If both lines map to the same segment, pull line start to segment end.
        if seg is prev_seg:
            shift = seg.end - line.start_time
        else:
            # If line maps to next segment but starts far after it, pull to segment start.
            if line.start_time - seg.start <= 0:
                continue
            if seg.start - prev_seg.end > 2.5:
                continue
            if sim < min_similarity:
                continue
            shift = seg.start - line.start_time

        if shift >= -0.2:
            continue

        new_words = [
            Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        adjusted[idx] = Line(words=new_words, singer=line.singer)
        fixes += 1

    return adjusted, fixes


def _retime_line_to_segment(line: Line, seg: TranscriptionSegment) -> Line:
    """Retime a line's words evenly within a Whisper segment."""
    from .models import Line, Word

    if not line.words:
        return line

    word_count = max(len(line.words), 1)
    total_duration = max(seg.end - seg.start, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(line.words):
        start = seg.start + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _retime_line_to_segment_with_min_start(
    line: Line, seg: TranscriptionSegment, min_start: float
) -> Line:
    """Retime a line within a segment, respecting a minimum start time."""
    from .models import Line, Word

    if not line.words:
        return line

    start_base = max(seg.start, min_start)
    if start_base >= seg.end:
        return line

    word_count = max(len(line.words), 1)
    total_duration = max(seg.end - start_base, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(line.words):
        start = start_base + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _retime_line_to_window(line: Line, window_start: float, window_end: float) -> Line:
    """Retime a line's words evenly within a custom window."""
    from .models import Line, Word

    if not line.words:
        return line

    if window_end <= window_start:
        return line

    word_count = max(len(line.words), 1)
    total_duration = max(window_end - window_start, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(line.words):
        start = window_start + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _pull_lines_to_best_segments(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.7,
    min_shift: float = 0.75,
    max_shift: float = 12.0,
    min_gap: float = 0.05,
) -> Tuple[List[Line], int]:
    """Pull lines toward their best Whisper segment when alignment drift is large."""
    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixed = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx, line in enumerate(adjusted):
        if not line.words:
            continue

        word_count = len(line.words)
        local_min_similarity = min_similarity
        if word_count <= 6:
            local_min_similarity = min(0.6, min_similarity)

        prev_end = None
        if idx > 0 and adjusted[idx - 1].words:
            prev_end = adjusted[idx - 1].end_time
        next_start = None
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_start = adjusted[idx + 1].start_time

        if word_count <= 6:
            nearest_start_seg = None
            nearest_start_gap = None
            for cand in sorted_segments:
                gap = abs(cand.start - line.start_time)
                if gap > 6.0:
                    continue
                if prev_end is not None and cand.start <= prev_end + min_gap:
                    continue
                if next_start is not None and cand.end >= next_start - min_gap:
                    continue
                if nearest_start_gap is None or gap < nearest_start_gap:
                    nearest_start_gap = gap
                    nearest_start_seg = cand
            if (
                nearest_start_seg is not None
                and nearest_start_seg.start <= line.start_time - 1.0
            ):
                adjusted[idx] = _retime_line_to_segment(line, nearest_start_seg)
                fixed += 1
                continue

        if word_count <= 4:
            end_aligned = None
            end_gap = None
            for cand in sorted_segments:
                gap_to_end = line.start_time - cand.end
                if gap_to_end < 0 or gap_to_end > 0.75:
                    continue
                if line.start_time - cand.start < 2.5:
                    continue
                if prev_end is not None and cand.start <= prev_end + min_gap:
                    if abs(cand.start - prev_end) > 0.1:
                        continue
                if next_start is not None and cand.end >= next_start - min_gap:
                    continue
                if end_gap is None or gap_to_end < end_gap:
                    end_gap = gap_to_end
                    end_aligned = cand
            if end_aligned is not None:
                adjusted[idx] = _retime_line_to_segment(line, end_aligned)
                fixed += 1
                continue

        seg, sim, _ = _find_best_whisper_segment(
            line.text,
            line.start_time,
            sorted_segments,
            language,
            local_min_similarity,
        )
        if seg is None or sim < local_min_similarity:
            # Fallback: allow weaker match if line is significantly late
            seg, sim, _ = _find_best_whisper_segment(
                line.text,
                line.start_time,
                sorted_segments,
                language,
                min_similarity=0.3,
            )
            if seg is None:
                # Secondary fallback: align to a segment that ends right at the line start
                if word_count > 4:
                    continue
                nearest = None
                nearest_gap = None
                for cand in sorted_segments:
                    gap_to_end = line.start_time - cand.end
                    if gap_to_end < 0 or gap_to_end > 0.75:
                        continue
                    if line.start_time - cand.start < 2.5:
                        continue
                    if nearest_gap is None or gap_to_end < nearest_gap:
                        nearest_gap = gap_to_end
                        nearest = cand
                if nearest is None:
                    continue
                seg = nearest
                sim = 0.0

        start_delta = seg.start - line.start_time
        end_delta = seg.end - line.end_time
        if abs(start_delta) < min_shift and abs(end_delta) < min_shift:
            continue

        late_and_ordered = (
            sim >= 0.3
            and start_delta <= -1.0
            and (prev_end is None or seg.start >= prev_end + min_gap)
            and (next_start is None or seg.end <= next_start - min_gap)
        )

        if sim < local_min_similarity:
            if word_count > 4 and not late_and_ordered:
                continue
            if start_delta > -3.0:
                continue

        if sim < local_min_similarity and word_count <= 6:
            if -3.0 <= start_delta <= -1.0:
                if prev_end is not None and seg.start <= prev_end + min_gap:
                    continue
                if next_start is not None and seg.end >= next_start - min_gap:
                    continue
                adjusted[idx] = _retime_line_to_segment(line, seg)
                fixed += 1
                continue

        if abs(start_delta) > max_shift:
            continue

        if sim < local_min_similarity:
            if prev_end is not None and seg.start <= prev_end + min_gap:
                continue
            if next_start is not None and seg.end >= next_start - min_gap:
                continue

        if prev_end is not None and seg.start <= prev_end + min_gap:
            continue
        if next_start is not None and seg.end >= next_start - min_gap:
            continue

        adjusted[idx] = _retime_line_to_segment(line, seg)
        fixed += 1

    return adjusted, fixed


def _drop_duplicate_lines(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.7,
) -> Tuple[List[Line], int]:
    """Drop adjacent duplicate lines when they map to the same Whisper segment."""
    if not lines:
        return lines, 0

    sorted_segments = sorted(segments, key=lambda s: s.start) if segments else []
    deduped: List[Line] = []
    dropped = 0

    for line in lines:
        if not deduped:
            deduped.append(line)
            continue

        prev = deduped[-1]
        if not prev.words or not line.words:
            deduped.append(line)
            continue

        if prev.text.strip() != line.text.strip():
            deduped.append(line)
            continue

        prev_seg, prev_sim, _ = _find_best_whisper_segment(
            prev.text, prev.start_time, sorted_segments, language, min_similarity
        )
        seg, sim, _ = _find_best_whisper_segment(
            line.text, line.start_time, sorted_segments, language, min_similarity
        )

        if (
            prev_seg is not None
            and seg is not None
            and prev_seg is seg
            and prev_sim >= min_similarity
            and sim >= min_similarity
        ):
            dropped += 1
            continue

        deduped.append(line)

    return deduped, dropped


def _drop_duplicate_lines_by_timing(
    lines: List[Line],
    max_gap: float = 0.2,
) -> Tuple[List[Line], int]:
    """Drop adjacent duplicate lines that overlap or are nearly contiguous."""
    if not lines:
        return lines, 0

    deduped: List[Line] = []
    dropped = 0

    for line in lines:
        if not deduped:
            deduped.append(line)
            continue

        prev = deduped[-1]
        if not prev.words or not line.words:
            deduped.append(line)
            continue

        if prev.text.strip() != line.text.strip():
            deduped.append(line)
            continue

        gap = line.start_time - prev.end_time
        if gap <= max_gap:
            dropped += 1
            continue

        deduped.append(line)

    return deduped, dropped


def _normalize_line_word_timings(
    lines: List[Line],
    min_word_duration: float = 0.05,
    min_gap: float = 0.01,
) -> List[Line]:
    """Ensure each line has monotonic word timings with valid durations."""
    from .models import Line, Word

    normalized: List[Line] = []
    for line in lines:
        if not line.words:
            normalized.append(line)
            continue

        new_words = []
        last_end = None
        for word in line.words:
            start = float(word.start_time)
            end = float(word.end_time)

            if end < start:
                end = start + min_word_duration

            if last_end is not None and start < last_end:
                shift = (last_end - start) + min_gap
                start += shift
                end += shift

            if end < start:
                end = start + min_word_duration

            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
            last_end = end

        normalized.append(Line(words=new_words, singer=line.singer))

    return normalized


def _find_best_whisper_segment(
    line_text: str,
    line_start: float,
    sorted_segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float,
) -> Tuple[Optional[TranscriptionSegment], float, float]:
    """Find best matching Whisper segment for a line."""
    best_segment = None
    best_similarity = 0.0
    best_offset = 0.0

    for seg in sorted_segments:
        if seg.start > line_start + 15 or seg.end < line_start - 15:
            continue

        similarity = _phonetic_similarity(line_text, seg.text, language)
        if similarity > best_similarity and similarity >= min_similarity:
            best_similarity = similarity
            best_segment = seg
            best_offset = seg.start - line_start

    return best_segment, best_similarity, best_offset


def _apply_offset_to_line(line: Line, offset: float) -> Line:
    """Apply timing offset to all words in a line."""
    from .models import Line, Word

    new_words = [
        Word(
            text=word.text,
            start_time=word.start_time + offset,
            end_time=word.end_time + offset,
            singer=word.singer,
        )
        for word in line.words
    ]
    return Line(words=new_words, singer=line.singer)


def _calculate_drift_correction(
    recent_offsets: List[float], trust_threshold: float
) -> Optional[float]:
    """Calculate drift correction from recent offsets if consistent drift exists."""
    if len(recent_offsets) < 2:
        return None

    # Check recent lines for consistent drift
    recent_nonzero = [o for o in recent_offsets[-5:] if abs(o) > 0.5]
    if len(recent_nonzero) >= 2:
        avg_drift = sum(recent_nonzero) / len(recent_nonzero)
        if abs(avg_drift) > trust_threshold:
            return avg_drift
    return None


def align_hybrid_lrc_whisper(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    words: List[TranscriptionWord],
    language: str = "fra-Latn",
    trust_threshold: float = 1.0,
    correct_threshold: float = 1.5,
    min_similarity: float = 0.4,
) -> Tuple[List[Line], List[str]]:
    """Hybrid alignment: preserve good LRC timing, use Whisper for broken sections.

    Strategy:
    1. First pass: Match each LRC line to best Whisper segment, measure timing error
    2. Lines with small error (< trust_threshold): keep LRC timing
    3. Lines with large error (> correct_threshold): use Whisper timing
    4. Track cumulative drift for lines without good matches
    5. Apply word-level alignment only to lines that need correction

    Args:
        lines: LRC lines with word-level timing
        segments: Whisper segments (sentence level)
        words: Whisper words (for word-level refinement)
        language: Epitran language code
        trust_threshold: Keep LRC if timing error below this (seconds)
        correct_threshold: Use Whisper if timing error above this (seconds)
        min_similarity: Minimum similarity to consider a match

    Returns:
        Tuple of (aligned lines, list of corrections)
    """
    if not lines or not segments:
        return lines, []

    logger.debug(f"Pre-computing IPA for {len(words)} Whisper words...")
    for w in words:
        _get_ipa(w.text, language)

    aligned_lines: List[Line] = []
    corrections: List[str] = []
    recent_offsets: List[float] = []
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for line_idx, line in enumerate(lines):
        if not line.words:
            aligned_lines.append(line)
            continue

        line_text = " ".join(w.text for w in line.words)
        line_start = line.start_time

        best_segment, best_similarity, best_offset = _find_best_whisper_segment(
            line_text, line_start, sorted_segments, language, min_similarity
        )

        timing_error = abs(best_offset) if best_segment else float("inf")

        # Case 1: LRC timing is good - keep it
        if best_segment and timing_error < trust_threshold:
            aligned_lines.append(line)
            recent_offsets.append(0.0)
            continue

        # Case 2: LRC timing is significantly off - use Whisper
        if (
            best_segment
            and timing_error >= correct_threshold
            and best_similarity >= 0.5
        ):
            aligned_lines.append(_apply_offset_to_line(line, best_offset))
            corrections.append(
                f'Line {line_idx} shifted {best_offset:+.1f}s (similarity: {best_similarity:.0%}): "{line_text[:35]}..."'
            )
            recent_offsets.append(best_offset)
            continue

        # Case 3: Intermediate timing error - check for consistent drift
        if timing_error >= trust_threshold and timing_error < correct_threshold:
            if len(recent_offsets) >= 2 and all(
                abs(o) > 0.5 for o in recent_offsets[-2:]
            ):
                avg_drift = sum(recent_offsets[-3:]) / len(recent_offsets[-3:])
                if abs(avg_drift) > trust_threshold:
                    aligned_lines.append(_apply_offset_to_line(line, avg_drift))
                    corrections.append(
                        f'Line {line_idx} drift-corrected {avg_drift:+.1f}s: "{line_text[:35]}..."'
                    )
                    recent_offsets.append(avg_drift)
                    continue

            aligned_lines.append(line)
            recent_offsets.append(0.0)
            continue

        # Case 4: No good match - apply drift correction if available
        drift = _calculate_drift_correction(recent_offsets, trust_threshold)
        if drift is not None:
            aligned_lines.append(_apply_offset_to_line(line, drift))
            corrections.append(
                f'Line {line_idx} drift-corrected {drift:+.1f}s (no match): "{line_text[:35]}..."'
            )
            recent_offsets.append(drift)
        else:
            aligned_lines.append(line)
            recent_offsets.append(0.0)

    return aligned_lines, corrections


def _fix_ordering_violations(
    original_lines: List[Line],
    aligned_lines: List[Line],
    alignments: List[str],
) -> Tuple[List[Line], List[str]]:
    """Fix lines that were moved out of order by Whisper alignment.

    If a line's Whisper-aligned timing would cause it to overlap with
    or come before the previous line, revert to the original timing.
    """
    from .models import Line

    if not aligned_lines:
        return aligned_lines, alignments

    fixed_lines: List[Line] = []
    fixed_alignments: List[str] = []
    prev_end_time = 0.0
    prev_start_time = 0.0
    reverted_count = 0

    for i, (orig, aligned) in enumerate(zip(original_lines, aligned_lines)):
        if not aligned.words:
            fixed_lines.append(aligned)
            continue

        aligned_start = aligned.start_time

        # Check if this line would start before previous line (or end) with tolerance
        if (
            aligned_start < prev_start_time - 0.01
            or aligned_start < prev_end_time - 0.1
        ):
            # Revert to original timing
            fixed_lines.append(orig)
            if orig.words:
                prev_end_time = orig.end_time
                prev_start_time = orig.start_time
            reverted_count += 1
        else:
            # Keep the aligned timing
            fixed_lines.append(aligned)
            prev_end_time = aligned.end_time
            prev_start_time = aligned.start_time

    # Update alignments list (remove reverted ones)
    if reverted_count > 0:
        logger.debug(
            f"Reverted {reverted_count} Whisper alignments due to ordering violations"
        )
        # Filter alignments to only include successful ones
        fixed_alignments = [a for a in alignments if True]  # Keep all for now
        # Actually recount
        actual_corrections = len(alignments) - reverted_count
        fixed_alignments = (
            alignments[:actual_corrections] if actual_corrections > 0 else []
        )

    return fixed_lines, fixed_alignments if reverted_count > 0 else alignments


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
