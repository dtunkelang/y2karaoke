"""Line-level helper utilities for Whisper timing refinement."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ...models import Line, Word


def line_tokens(text: str) -> List[str]:
    tokens = []
    for raw in text.lower().split():
        token = "".join(ch for ch in raw if ch.isalpha())
        if token:
            tokens.append(token)
    return tokens


def token_overlap(a: str, b: str) -> float:
    tokens_a = line_tokens(a)
    tokens_b = line_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    return len(set_a & set_b) / max(len(set_a), len(set_b))


def shift_line_words(line: Line, shift: float) -> Line:
    shifted_words = [
        Word(
            text=word.text,
            start_time=word.start_time + shift,
            end_time=word.end_time + shift,
            singer=word.singer,
        )
        for word in line.words
    ]
    return Line(words=shifted_words, singer=line.singer)


def rebuild_line_with_target_end(line: Line, target_end: float) -> Optional[Line]:
    if not line.words:
        return None
    if target_end <= line.start_time + 0.2:
        return None
    spacing = (target_end - line.start_time) / len(line.words)
    compact_words = []
    for word_idx, word in enumerate(line.words):
        start = line.start_time + word_idx * spacing
        end = start + spacing * 0.9
        if word_idx == len(line.words) - 1:
            end = target_end
        compact_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=compact_words, singer=line.singer)


def compact_short_line_if_needed(
    line: Line,
    *,
    max_duration: float,
    next_start: Optional[float] = None,
) -> Line:
    if not line.words:
        return line
    duration = line.end_time - line.start_time
    if duration <= max_duration:
        return line
    target_end = line.start_time + max_duration
    if next_start is not None:
        target_end = min(target_end, next_start - 0.05)
    compacted = rebuild_line_with_target_end(line, target_end)
    return compacted if compacted is not None else line


def find_internal_silences(
    line_start: float,
    line_end: float,
    normalized_silences: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    return [
        (start, end)
        for start, end in normalized_silences
        if start >= line_start and end <= (line_end + 2.0)
    ]


def has_near_start_silence(
    line_start: float, internal_silences: List[Tuple[float, float]]
) -> bool:
    return any(
        start <= line_start + 0.7 and end >= line_start - 0.1
        for start, end in internal_silences
    )


def first_onset_after(onset_times, *, start: float, window: float) -> Optional[float]:
    candidate_onsets = onset_times[
        (onset_times >= start) & (onset_times <= start + window)
    ]
    if len(candidate_onsets) == 0:
        return None
    return float(candidate_onsets[0])


def is_uniform_word_timing(line: Line) -> bool:
    if not line.words or len(line.words) < 3:
        return False
    durations = np.array(
        [max(0.0, word.end_time - word.start_time) for word in line.words], dtype=float
    )
    mean_duration = float(np.mean(durations)) if len(durations) else 0.0
    if mean_duration <= 0.0:
        return False
    coeff_var = float(np.std(durations) / mean_duration)
    unique_rounded = len(set(round(float(d), 3) for d in durations))
    return coeff_var < 0.08 or unique_rounded <= 2


def retime_line_words_to_onsets(
    line: Line,
    onset_times,
    *,
    min_word_duration: float = 0.08,
) -> Optional[Line]:
    if not line.words or len(line.words) < 3:
        return None
    line_start = line.start_time
    line_end = line.end_time
    if line_end <= line_start + min_word_duration * len(line.words):
        return None

    n_words = len(line.words)
    if n_words < 2:
        return None

    candidate_onsets = _candidate_onsets_for_line(
        onset_times, line_start=line_start, line_end=line_end
    )
    target_word_count = n_words - 1
    if len(candidate_onsets) < target_word_count:
        return None

    expected_starts = _expected_word_starts(line.words, line_start, line_end)
    chosen_indices = _choose_onset_indices(
        candidate_onsets=candidate_onsets,
        expected_starts=expected_starts,
        target_word_count=target_word_count,
    )
    if chosen_indices is None:
        return None

    starts = [line_start] + [float(candidate_onsets[idx]) for idx in chosen_indices]
    adjusted_starts = _adjust_monotonic_starts(starts, min_step=0.02)

    if adjusted_starts[-1] >= line_end - 0.01:
        return None

    new_words = _build_words_from_starts(
        line.words,
        adjusted_starts,
        line_end=line_end,
        min_word_duration=min_word_duration,
        boundary_gap=0.02,
    )
    if new_words is None:
        return None
    return Line(words=new_words, singer=line.singer)


def _candidate_onsets_for_line(onset_times, *, line_start: float, line_end: float):
    return np.sort(
        onset_times[
            (onset_times >= line_start + 0.04) & (onset_times <= line_end - 0.01)
        ]
    )


def _expected_word_starts(
    words: List[Word], line_start: float, line_end: float
) -> List[float]:
    original_duration = max(line_end - line_start, 1e-3)
    expected_starts: List[float] = []
    for word in words[1:]:
        normalized = max(
            0.0, min(1.0, (word.start_time - line_start) / original_duration)
        )
        expected_starts.append(line_start + normalized * original_duration)
    return expected_starts


def _choose_onset_indices(
    *,
    candidate_onsets,
    expected_starts: List[float],
    target_word_count: int,
) -> Optional[List[int]]:
    chosen_indices: List[int] = []
    prev_idx = -1
    for i in range(target_word_count):
        min_idx = prev_idx + 1
        max_idx = len(candidate_onsets) - (target_word_count - i)
        if max_idx < min_idx:
            return None
        target = expected_starts[i]
        best_idx = min(
            range(min_idx, max_idx + 1),
            key=lambda idx: abs(float(candidate_onsets[idx]) - target),
        )
        chosen_indices.append(best_idx)
        prev_idx = best_idx
    return chosen_indices


def _adjust_monotonic_starts(starts: List[float], *, min_step: float) -> List[float]:
    adjusted_starts = [starts[0]]
    for start in starts[1:]:
        adjusted_starts.append(max(start, adjusted_starts[-1] + min_step))
    return adjusted_starts


def _build_words_from_starts(
    words: List[Word],
    adjusted_starts: List[float],
    *,
    line_end: float,
    min_word_duration: float,
    boundary_gap: float,
) -> Optional[List[Word]]:
    new_words: List[Word] = []
    for i, word in enumerate(words):
        start = adjusted_starts[i]
        if i + 1 < len(adjusted_starts):
            end = min(line_end, adjusted_starts[i + 1] - boundary_gap)
        else:
            end = line_end
        if end - start < min_word_duration:
            end = min(line_end, start + min_word_duration)
        if end <= start:
            return None
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return new_words
