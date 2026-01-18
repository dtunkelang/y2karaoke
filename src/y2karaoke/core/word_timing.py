"""Word-level timing using audio analysis.

This module refines word timing within line boundaries using onset detection
and energy analysis of the vocals audio.
"""

from typing import List, Tuple, Optional
import numpy as np

from ..utils.logging import get_logger
from .models import Word, Line

logger = get_logger(__name__)


def refine_word_timing(
    lines: List[Line],
    vocals_path: str,
    respect_line_boundaries: bool = True,
) -> List[Line]:
    """
    Refine word timing using audio onset detection.

    Takes lines with evenly-distributed word timing and refines them
    based on detected onsets in the vocals audio.

    Args:
        lines: List of Line objects with initial word timing
        vocals_path: Path to vocals audio file
        respect_line_boundaries: If True, words stay within their line's time window

    Returns:
        List of Line objects with refined word timing
    """
    try:
        import librosa

        # Load audio
        y, sr = librosa.load(vocals_path, sr=22050)

        # Detect onsets globally
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr,
            hop_length=512,
            backtrack=True,
            units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

        logger.info(f"Detected {len(onset_times)} onsets in vocals")

        # Compute energy envelope for duration estimation
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

        # Refine each line
        refined_lines = []
        for line in lines:
            refined_line = _refine_line_timing(
                line, onset_times, rms, rms_times, respect_line_boundaries
            )
            refined_lines.append(refined_line)

        return refined_lines

    except Exception as e:
        logger.warning(f"Word timing refinement failed: {e}, using original timing")
        return lines


def _refine_line_timing(
    line: Line,
    onset_times: np.ndarray,
    rms: np.ndarray,
    rms_times: np.ndarray,
    respect_boundaries: bool,
) -> Line:
    """
    Refine word timing for a single line using onsets.

    Args:
        line: Line object with initial word timing
        onset_times: Array of onset times from audio
        rms: RMS energy envelope
        rms_times: Times corresponding to RMS frames
        respect_boundaries: Keep words within line boundaries

    Returns:
        Line with refined word timing
    """
    if not line.words:
        return line

    line_start = line.start_time
    line_end = line.end_time

    # Find onsets within this line's time window (with small buffer)
    buffer = 0.1  # 100ms buffer
    line_onsets = onset_times[
        (onset_times >= line_start - buffer) &
        (onset_times <= line_end + buffer)
    ]

    # If we have fewer onsets than words, use original timing
    if len(line_onsets) < len(line.words):
        logger.debug(f"Line has {len(line.words)} words but only {len(line_onsets)} onsets, keeping original")
        return line

    # Match words to onsets
    refined_words = _match_words_to_onsets(
        line.words, line_onsets, line_start, line_end, respect_boundaries
    )

    return Line(words=refined_words, singer=line.singer)


def _match_words_to_onsets(
    words: List[Word],
    onsets: np.ndarray,
    line_start: float,
    line_end: float,
    respect_boundaries: bool,
) -> List[Word]:
    """
    Match words to detected onsets using greedy assignment.

    Strategy:
    1. For each word, find the closest onset to its expected start time
    2. Assign onset to word, mark onset as used
    3. Estimate word duration based on gap to next onset or next word

    Args:
        words: List of Word objects
        onsets: Array of onset times
        line_start: Line start time (boundary)
        line_end: Line end time (boundary)
        respect_boundaries: Keep words within boundaries

    Returns:
        List of Word objects with refined timing
    """
    if len(onsets) == 0:
        return words

    refined_words = []
    used_onsets = set()

    for i, word in enumerate(words):
        # Find closest unused onset to word's expected start
        expected_start = word.start_time

        best_onset_idx = None
        best_distance = float('inf')

        for j, onset in enumerate(onsets):
            if j in used_onsets:
                continue
            distance = abs(onset - expected_start)
            if distance < best_distance:
                best_distance = distance
                best_onset_idx = j

        # Use onset if found and reasonably close (within 0.5s)
        if best_onset_idx is not None and best_distance < 0.5:
            used_onsets.add(best_onset_idx)
            new_start = onsets[best_onset_idx]
        else:
            new_start = expected_start

        # Apply boundary constraints
        if respect_boundaries:
            new_start = max(line_start, min(new_start, line_end - 0.1))

        # Estimate end time
        if i + 1 < len(words):
            # End before next word starts
            next_expected = words[i + 1].start_time
            # Find next onset after this one
            future_onsets = onsets[onsets > new_start + 0.05]
            if len(future_onsets) > 0:
                next_onset = future_onsets[0]
                new_end = min(next_onset - 0.02, next_expected)
            else:
                new_end = next_expected - 0.02
        else:
            # Last word - end at line end or estimated duration
            original_duration = word.end_time - word.start_time
            new_end = min(new_start + original_duration, line_end)

        # Ensure minimum duration
        if new_end - new_start < 0.05:
            new_end = new_start + 0.1

        if respect_boundaries:
            new_end = min(new_end, line_end)

        refined_words.append(Word(
            text=word.text,
            start_time=new_start,
            end_time=new_end,
            singer=word.singer,
        ))

    return refined_words


def detect_line_onsets(
    vocals_path: str,
    line_start: float,
    line_end: float,
) -> List[float]:
    """
    Detect onsets within a specific time range.

    Args:
        vocals_path: Path to vocals audio
        line_start: Start time in seconds
        line_end: End time in seconds

    Returns:
        List of onset times within the range
    """
    try:
        import librosa

        # Load just the segment we need
        y, sr = librosa.load(
            vocals_path,
            sr=22050,
            offset=max(0, line_start - 0.1),
            duration=(line_end - line_start) + 0.2
        )

        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr,
            hop_length=512,
            backtrack=True,
            units='frames'
        )

        # Convert to absolute times
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        onset_times = onset_times + max(0, line_start - 0.1)

        # Filter to line boundaries
        onset_times = onset_times[
            (onset_times >= line_start) & (onset_times <= line_end)
        ]

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
    """
    Estimate word boundaries using energy envelope.

    Useful when onset detection doesn't find enough onsets.
    Finds energy peaks and valleys to estimate word boundaries.

    Args:
        vocals_path: Path to vocals audio
        line_start: Start time in seconds
        line_end: End time in seconds
        num_words: Expected number of words

    Returns:
        List of (start, end) tuples for each word
    """
    try:
        import librosa
        from scipy.signal import find_peaks

        # Load segment
        y, sr = librosa.load(
            vocals_path,
            sr=22050,
            offset=line_start,
            duration=line_end - line_start
        )

        # Compute energy envelope
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        times = times + line_start

        # Find energy peaks (likely word centers)
        peaks, _ = find_peaks(rms, distance=int(0.1 * sr / hop_length))  # Min 100ms between peaks

        if len(peaks) < num_words:
            # Not enough peaks - fall back to even distribution
            duration = line_end - line_start
            word_duration = duration / num_words
            return [
                (line_start + i * word_duration, line_start + (i + 1) * word_duration)
                for i in range(num_words)
            ]

        # Use peaks as word centers, find boundaries at energy minima between them
        boundaries = []
        peak_times = times[peaks]

        for i in range(num_words):
            if i < len(peak_times):
                center = peak_times[i]

                # Start: midpoint to previous peak or line start
                if i == 0:
                    start = line_start
                else:
                    start = (peak_times[i-1] + center) / 2

                # End: midpoint to next peak or line end
                if i >= len(peak_times) - 1:
                    end = line_end
                else:
                    end = (center + peak_times[i+1]) / 2

                boundaries.append((start, end))
            else:
                # Ran out of peaks - estimate remaining
                remaining = num_words - i
                remaining_duration = line_end - boundaries[-1][1] if boundaries else line_end - line_start
                word_duration = remaining_duration / remaining
                start = boundaries[-1][1] if boundaries else line_start
                boundaries.append((start, start + word_duration))

        return boundaries

    except Exception as e:
        logger.warning(f"Energy-based boundary estimation failed: {e}")
        # Fall back to even distribution
        duration = line_end - line_start
        word_duration = duration / num_words
        return [
            (line_start + i * word_duration, line_start + (i + 1) * word_duration)
            for i in range(num_words)
        ]
