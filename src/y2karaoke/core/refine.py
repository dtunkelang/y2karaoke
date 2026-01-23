"""Word timing refinement using onsets and energy analysis."""

from typing import List
import numpy as np
from ..utils.logging import get_logger
from .models import Word, Line

logger = get_logger(__name__)


def refine_word_timing(lines: List[Line], vocals_path: str, respect_line_boundaries: bool = True) -> List[Line]:
    """Refine word timing using audio onset detection."""
    try:
        import librosa

        # Load vocals audio
        y, sr = librosa.load(vocals_path, sr=22050)
        hop_length = 512
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        logger.debug(f"Detected {len(onset_times)} onsets in vocals")

        # Compute RMS energy for vocal end detection
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # Compute energy threshold for silence detection
        noise_floor = np.percentile(rms, 10)
        peak_level = np.percentile(rms, 90)
        silence_threshold = noise_floor + 0.15 * (peak_level - noise_floor)

        # Refine each line
        refined_lines = []
        for line in lines:
            refined_lines.append(
                _refine_line_timing(
                    line, onset_times, rms, rms_times,
                    silence_threshold, respect_line_boundaries
                )
            )

        return refined_lines

    except Exception as e:
        logger.warning(f"Word timing refinement failed: {e}, using original timing")
        return lines


# ----------------------
# Internal helpers
# ----------------------
def _refine_line_timing(
    line: Line,
    onset_times: np.ndarray,
    rms: np.ndarray,
    rms_times: np.ndarray,
    silence_threshold: float,
    respect_boundaries: bool
) -> Line:
    """Refine timing for a single line."""
    if not line.words:
        return line

    line_start, line_end = line.start_time, line.end_time
    buffer = 0.1
    line_onsets = onset_times[(onset_times >= line_start - buffer) & (onset_times <= line_end + buffer)]

    # Detect actual vocal end time (when energy drops to silence)
    vocal_end = _detect_vocal_end(
        line_start, line_end, rms, rms_times, silence_threshold, len(line.words)
    )

    if len(line_onsets) < len(line.words):
        # Not enough onsets - still apply vocal end detection to last word
        logger.debug(f"Line has {len(line.words)} words but only {len(line_onsets)} onsets, applying vocal end only")
        words = list(line.words)
        if words and vocal_end < words[-1].end_time:
            last_word = words[-1]
            # Ensure minimum word duration
            new_end = max(vocal_end, last_word.start_time + 0.1)
            words[-1] = Word(
                text=last_word.text,
                start_time=last_word.start_time,
                end_time=new_end,
                singer=last_word.singer
            )
        return Line(words=words, singer=line.singer)

    refined_words = _match_words_to_onsets(
        line.words, line_onsets, line_start, vocal_end, respect_boundaries
    )
    return Line(words=refined_words, singer=line.singer)


def _detect_vocal_end(
    line_start: float,
    line_end: float,
    rms: np.ndarray,
    rms_times: np.ndarray,
    silence_threshold: float,
    word_count: int,
    min_silence_duration: float = 0.4
) -> float:
    """
    Detect when vocals actually end within a line by finding sustained silence.

    Searches backwards from line_end to find where energy drops below threshold
    for at least min_silence_duration seconds.

    Returns the detected vocal end time, or line_end if no clear end found.
    """
    # Get RMS frames within the line
    mask = (rms_times >= line_start) & (rms_times <= line_end)
    line_rms = rms[mask]
    line_times = rms_times[mask]

    if len(line_rms) == 0:
        return line_end

    # Work backwards to find where sustained silence begins
    # We want to find the last moment of vocal activity
    is_silent = line_rms < silence_threshold

    # Find runs of silence from the end
    frame_duration = line_times[1] - line_times[0] if len(line_times) > 1 else 0.023
    min_silent_frames = int(min_silence_duration / frame_duration)

    # Scan backwards for sustained silence
    silent_count = 0
    vocal_end_idx = len(line_rms) - 1

    for i in range(len(line_rms) - 1, -1, -1):
        if is_silent[i]:
            silent_count += 1
        else:
            # Found vocal activity - check if preceding silence was sustained
            if silent_count >= min_silent_frames:
                vocal_end_idx = i
                break
            silent_count = 0
            vocal_end_idx = i

    # Add a small buffer after the last detected vocal activity
    detected_end = line_times[vocal_end_idx] + 0.15

    # Sanity check: ensure we have enough time for the words
    # Assume minimum ~0.15s per word for fast singing
    min_duration_for_words = word_count * 0.15
    min_end = line_start + min_duration_for_words

    # Don't trim if it would make the line unreasonably short
    if detected_end < min_end:
        detected_end = line_end

    # Don't extend beyond original line_end
    detected_end = min(detected_end, line_end)

    return detected_end


def _match_words_to_onsets(
    words: List[Word],
    onsets: np.ndarray,
    line_start: float,
    vocal_end: float,
    respect_boundaries: bool
) -> List[Word]:
    """Greedy assignment of words to closest onsets.

    Args:
        words: Words to refine timing for
        onsets: Detected onset times in the audio
        line_start: Start time of the line
        vocal_end: Detected end of vocals (used only for last word's end time)
        respect_boundaries: Whether to clamp times within line boundaries
    """
    refined_words = []
    used_onsets = set()
    is_last_word = False

    for i, word in enumerate(words):
        is_last_word = (i == len(words) - 1)
        expected_start = word.start_time
        best_onset_idx, best_distance = None, float('inf')

        for j, onset in enumerate(onsets):
            if j in used_onsets:
                continue
            distance = abs(onset - expected_start)
            if distance < best_distance:
                best_distance, best_onset_idx = distance, j

        # Choose onset if close enough
        new_start = onsets[best_onset_idx] if best_onset_idx is not None and best_distance < 0.5 else expected_start
        if best_onset_idx is not None:
            used_onsets.add(best_onset_idx)

        if respect_boundaries:
            new_start = max(line_start, new_start)

        # Estimate end time
        if i + 1 < len(words):
            next_expected = words[i + 1].start_time
            future_onsets = onsets[onsets > new_start + 0.05]
            if len(future_onsets) > 0:
                next_onset = future_onsets[0]
                new_end = min(next_onset - 0.02, next_expected)
            else:
                new_end = next_expected - 0.02
        else:
            # Last word - end at detected vocal end, not original line end
            # Use a reasonable duration based on word length, capped at vocal_end
            char_count = len(word.text)
            estimated_duration = max(0.15, min(0.1 * char_count, 0.8))
            new_end = min(new_start + estimated_duration, vocal_end)

        if new_end - new_start < 0.05:
            new_end = new_start + 0.1

        # Only apply vocal_end clamping to the last word
        if respect_boundaries and is_last_word:
            new_end = min(new_end, vocal_end)

        refined_words.append(Word(text=word.text, start_time=new_start, end_time=new_end, singer=word.singer))

    return refined_words
