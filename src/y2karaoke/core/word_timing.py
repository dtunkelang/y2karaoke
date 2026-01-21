"""Word-level timing using audio analysis.

This module scores and fixes word timing, and provides fallback onset/energy-based estimation.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..utils.logging import get_logger
from .models import Word, Line
from .refine import refine_word_timing  # <- moved refinement logic here

logger = get_logger(__name__)


# ----------------------
# Quality scoring
# ----------------------
@dataclass
class TimingQuality:
    """Quality metrics for timing."""
    score: float  # 0.0 (bad) to 1.0 (good)
    short_words: int
    overlapping_words: int
    large_gaps: int
    out_of_order: int
    issues: List[str]


def score_timing_quality(
    lines: List[Line],
    min_word_duration: float = 0.08,
    max_gap: float = 1.0,
) -> TimingQuality:
    """Score the quality of word timing."""
    issues = []
    short_words = 0
    overlapping_words = 0
    large_gaps = 0
    out_of_order = 0
    total_words = 0
    prev_end = 0.0

    for line in lines:
        for word in line.words:
            total_words += 1
            duration = word.end_time - word.start_time

            if duration < min_word_duration:
                short_words += 1
                if short_words <= 3:
                    issues.append(f"Short word: '{word.text}' ({duration:.2f}s) at {word.start_time:.2f}s")

            if word.start_time < prev_end - 0.01:
                overlapping_words += 1
                if overlapping_words <= 3:
                    issues.append(f"Overlap: '{word.text}' starts at {word.start_time:.2f}s, prev ended at {prev_end:.2f}s")

            gap = word.start_time - prev_end
            if gap > max_gap and prev_end > 0:
                large_gaps += 1
                if large_gaps <= 3:
                    issues.append(f"Large gap: {gap:.2f}s before '{word.text}'")

            if word.start_time < prev_end - 0.5:
                out_of_order += 1

            prev_end = word.end_time

    if total_words == 0:
        return TimingQuality(0.0, 0, 0, 0, 0, ["No words found"])

    penalty = 0.0
    penalty += min(0.3, short_words / total_words)
    penalty += min(0.3, overlapping_words / total_words * 2)
    penalty += min(0.2, large_gaps / max(1, len(lines)) * 0.1)
    penalty += min(0.2, out_of_order / total_words * 5)
    score = max(0.0, 1.0 - penalty)

    return TimingQuality(score, short_words, overlapping_words, large_gaps, out_of_order, issues)


# ----------------------
# Fix timing issues
# ----------------------
def fix_timing_issues(
    lines: List[Line],
    min_word_duration: float = 0.08,
) -> List[Line]:
    """Fix short words, overlapping words, and ordering."""
    fixed_lines = []

    for line in lines:
        if not line.words:
            fixed_lines.append(line)
            continue

        fixed_words = []
        prev_end = line.start_time

        for i, word in enumerate(line.words):
            new_start = max(word.start_time, prev_end)
            new_end = word.end_time

            if new_end - new_start < min_word_duration:
                new_end = new_start + min_word_duration

            if i < len(line.words) - 1:
                new_end = min(new_end, line.end_time)

            fixed_words.append(Word(
                text=word.text,
                start_time=new_start,
                end_time=new_end,
                singer=word.singer,
            ))
            prev_end = new_end

        fixed_lines.append(Line(words=fixed_words, singer=line.singer))

    return fixed_lines


# ----------------------
# Onset / energy fallback utilities
# ----------------------
def detect_line_onsets(vocals_path: str, line_start: float, line_end: float) -> List[float]:
    """Detect onsets within a specific line segment."""
    try:
        import librosa

        y, sr = librosa.load(
            vocals_path,
            sr=22050,
            offset=max(0, line_start - 0.1),
            duration=(line_end - line_start) + 0.2
        )
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=512, backtrack=True, units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        onset_times += max(0, line_start - 0.1)
        onset_times = onset_times[(onset_times >= line_start) & (onset_times <= line_end)]
        return list(onset_times)

    except Exception as e:
        logger.warning(f"Onset detection failed for line: {e}")
        return []


def estimate_word_boundaries_from_energy(
    vocals_path: str,
    line_start: float,
    line_end: float,
    num_words: int,
) -> List[Tuple[float, float]]:
    """Estimate word boundaries using RMS energy peaks."""
    try:
        import librosa
        from scipy.signal import find_peaks

        y, sr = librosa.load(vocals_path, sr=22050, offset=line_start, duration=line_end - line_start)
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) + line_start
        peaks, _ = find_peaks(rms, distance=int(0.1 * sr / hop_length))

        if len(peaks) < num_words:
            duration = line_end - line_start
            word_duration = duration / num_words
            return [(line_start + i * word_duration, line_start + (i + 1) * word_duration) for i in range(num_words)]

        boundaries = []
        peak_times = times[peaks]
        for i in range(num_words):
            if i < len(peak_times):
                center = peak_times[i]
                start = line_start if i == 0 else (peak_times[i - 1] + center) / 2
                end = line_end if i >= len(peak_times) - 1 else (center + peak_times[i + 1]) / 2
                boundaries.append((start, end))
            else:
                remaining = num_words - i
                remaining_duration = line_end - boundaries[-1][1] if boundaries else line_end - line_start
                word_duration = remaining_duration / remaining
                start = boundaries[-1][1] if boundaries else line_start
                boundaries.append((start, start + word_duration))

        return boundaries

    except Exception as e:
        logger.warning(f"Energy-based boundary estimation failed: {e}")
        duration = line_end - line_start
        word_duration = duration / num_words
        return [(line_start + i * word_duration, line_start + (i + 1) * word_duration) for i in range(num_words)]
