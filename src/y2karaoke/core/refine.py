from typing import List
import numpy as np
from .models import Word, Line
from ..utils.logging import get_logger

logger = get_logger(__name__)


def refine_word_timing(
    lines: List[Line],
    vocals_path: str,
    respect_line_boundaries: bool = True,
) -> List[Line]:
    """
    Refine word timing using audio onset detection.
    """
    try:
        import librosa

        y, sr = librosa.load(vocals_path, sr=22050)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        logger.info(f"Detected {len(onset_times)} onsets in vocals")

        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

        refined_lines = [
            _refine_line_timing(line, onset_times, rms, rms_times, respect_line_boundaries)
            for line in lines
        ]
        return refined_lines

    except Exception as e:
        logger.warning(f"Word timing refinement failed: {e}, using original timing")
        return lines


def _refine_line_timing(line: Line, onset_times: np.ndarray, rms: np.ndarray, rms_times: np.ndarray, respect_boundaries: bool) -> Line:
    if not line.words:
        return line

    line_start, line_end = line.start_time, line.end_time
    buffer = 0.1
    line_onsets = onset_times[(onset_times >= line_start - buffer) & (onset_times <= line_end + buffer)]

    if len(line_onsets) < len(line.words):
        logger.debug(f"Line has {len(line.words)} words but only {len(line_onsets)} onsets, keeping original")
        return line

    refined_words = _match_words_to_onsets(line.words, line_onsets, line_start, line_end, respect_boundaries)
    return Line(words=refined_words, singer=line.singer)


def _match_words_to_onsets(words: List[Word], onsets: np.ndarray, line_start: float, line_end: float, respect_boundaries: bool) -> List[Word]:
    refined_words, used_onsets = [], set()

    for i, word in enumerate(words):
        expected_start = word.start_time
        best_onset_idx, best_distance = None, float('inf')

        for j, onset in enumerate(onsets):
            if j in used_onsets:
                continue
            dist = abs(onset - expected_start)
            if dist < best_distance:
                best_distance, best_onset_idx = dist, j

        new_start = onsets[best_onset_idx] if best_onset_idx is not None and best_distance < 0.5 else expected_start
        if respect_boundaries:
            new_start = max(line_start, min(new_start, line_end - 0.1))

        if i + 1 < len(words):
            next_expected = words[i + 1].start_time
            future_onsets = onsets[onsets > new_start + 0.05]
            if len(future_onsets) > 0:
                next_onset = future_onsets[0]
                new_end = min(next_onset - 0.02, next_expected)
            else:
                new_end = next_expected - 0.02
        else:
            new_end = min(new_start + (word.end_time - word.start_time), line_end)

        if new_end - new_start < 0.05:
            new_end = new_start + 0.1
        if respect_boundaries:
            new_end = min(new_end, line_end)

        refined_words.append(Word(text=word.text, start_time=new_start, end_time=new_end, singer=word.singer))

    return refined_words
