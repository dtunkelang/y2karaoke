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

        # Refine word timing within each line
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
    # Use a small buffer before line start but NO buffer after line end
    # to avoid matching onsets that belong to the next line
    line_onsets = onset_times[(onset_times >= line_start - 0.1) & (onset_times <= line_end)]

    # Detect actual vocal end time (when energy drops to silence)
    # But never extend beyond line_end - the estimated duration is our upper bound
    vocal_end = _detect_vocal_end(
        line_start, line_end, rms, rms_times, silence_threshold, len(line.words)
    )
    vocal_end = min(vocal_end, line_end)  # Never extend beyond estimated duration

    if len(line_onsets) < len(line.words):
        # Not enough onsets - redistribute words evenly within detected vocal duration
        actual_duration = vocal_end - line_start
        line_text = " ".join(w.text for w in line.words)
        logger.debug(f"Line '{line_text[:30]}...' ({len(line.words)} words): duration={actual_duration:.2f}s")
        words = list(line.words)
        if words:
            word_count = len(words)
            if actual_duration > 0 and word_count > 0:
                word_spacing = actual_duration / word_count
                new_words = []
                for i, word in enumerate(words):
                    new_start = line_start + i * word_spacing
                    new_end = new_start + (word_spacing * 0.9)  # 90% of slot
                    # Ensure last word ends at vocal_end
                    if i == word_count - 1:
                        new_end = vocal_end
                    new_words.append(Word(
                        text=word.text,
                        start_time=new_start,
                        end_time=new_end,
                        singer=word.singer
                    ))
                return Line(words=new_words, singer=line.singer)
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
    """Order-preserving assignment of words to onsets.

    Uses dynamic programming to find the best assignment that respects
    temporal order (word i must be assigned to an onset before word i+1).
    This handles pauses within lines correctly.

    Args:
        words: Words to refine timing for
        onsets: Detected onset times in the audio
        line_start: Start time of the line
        vocal_end: Detected end of vocals (used only for last word's end time)
        respect_boundaries: Whether to clamp times within line boundaries
    """
    if len(onsets) == 0:
        return list(words)

    # Sort onsets chronologically
    sorted_onsets = np.sort(onsets)
    n_words = len(words)
    n_onsets = len(sorted_onsets)

    # Match words to onsets with order preservation
    # Key insight: after matching a word to an onset, skip any nearby onsets
    # (within 0.3s) as they're likely syllables of the same word
    SYLLABLE_GAP = 0.3  # Onsets closer than this are same word

    word_to_onset = []
    min_next_onset_idx = 0  # Don't consider onsets before this index

    for i, word in enumerate(words):
        expected_start = word.start_time
        best_onset_idx = None
        best_score = float('inf')

        # Search for best onset starting from min_next_onset_idx
        for j in range(min_next_onset_idx, n_onsets):
            onset = sorted_onsets[j]

            # Score based on distance to expected time
            distance = abs(onset - expected_start)

            # Strong penalty for onset significantly before expected
            if onset < expected_start - 0.5:
                distance += 2.0

            # Check if we'd leave enough onsets for remaining words
            remaining_words = n_words - i - 1
            remaining_onsets = n_onsets - j - 1
            if remaining_words > 0 and remaining_onsets < remaining_words:
                # Not enough onsets left, must use this one or earlier
                pass
            elif distance > 2.0:
                # If this onset is far from expected and we have onsets to spare, skip it
                continue

            if distance < best_score:
                best_score = distance
                best_onset_idx = j

        if best_onset_idx is not None and best_score < 3.0:
            word_to_onset.append(best_onset_idx)
            # Skip past this onset and any nearby ones (same word syllables)
            skip_until = sorted_onsets[best_onset_idx] + SYLLABLE_GAP
            min_next_onset_idx = best_onset_idx + 1
            while min_next_onset_idx < n_onsets and sorted_onsets[min_next_onset_idx] < skip_until:
                min_next_onset_idx += 1
        else:
            # No good onset found - will use expected time
            word_to_onset.append(None)

    # First pass: calculate start times for all words
    # For words without onset match, we'll fix them in second pass
    word_starts = []
    for i, word in enumerate(words):
        onset_idx = word_to_onset[i]
        if onset_idx is not None:
            word_starts.append(sorted_onsets[onset_idx])
        else:
            word_starts.append(None)  # Will fix in second pass

    # Second pass: fix words without onset matches
    # For unmatched words, place them relative to the next matched word
    for i in range(n_words - 1, -1, -1):
        if word_starts[i] is None:
            # Find next word with a matched onset
            next_matched_time = None
            for j in range(i + 1, n_words):
                if word_starts[j] is not None:
                    next_matched_time = word_starts[j]
                    break

            if next_matched_time is not None:
                # Place this word just before the next matched word
                # Estimate duration based on character count
                char_count = len(words[i].text)
                est_duration = max(0.15, min(0.08 * char_count, 0.5))
                word_starts[i] = next_matched_time - est_duration
            else:
                # No future matched word - use previous word's end
                if i > 0 and word_starts[i-1] is not None:
                    char_count = len(words[i-1].text)
                    est_duration = max(0.15, min(0.08 * char_count, 0.5))
                    word_starts[i] = word_starts[i-1] + est_duration
                else:
                    word_starts[i] = words[i].start_time

    # Ensure monotonically increasing start times
    for i in range(1, n_words):
        if word_starts[i] <= word_starts[i-1]:
            word_starts[i] = word_starts[i-1] + 0.1

    # Build refined words list
    refined_words = []
    for i, word in enumerate(words):
        is_last_word = (i == len(words) - 1)
        new_start = word_starts[i]

        if respect_boundaries:
            new_start = max(line_start, new_start)

        # Estimate end time
        if i + 1 < len(words):
            new_end = word_starts[i + 1] - 0.02
        else:
            # Last word - extend to vocal_end to fill gap to next line
            # This prevents pauses when lyrics flow continuously
            new_end = vocal_end

        # Ensure minimum duration
        min_duration = 0.1
        if new_end - new_start < min_duration:
            new_end = new_start + min_duration

        # For last word, clamp to vocal_end but ensure minimum duration
        if respect_boundaries and is_last_word:
            new_end = max(new_start + min_duration, min(new_end, vocal_end))

        refined_words.append(Word(text=word.text, start_time=new_start, end_time=new_end, singer=word.singer))

    return refined_words
