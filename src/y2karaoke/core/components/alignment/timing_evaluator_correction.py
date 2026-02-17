"""Functions for correcting lyrics timing based on audio analysis."""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Callable, Iterator, List, Optional, Tuple
import numpy as np

from ...models import Line, Word
from .timing_models import AudioFeatures
from ...audio_analysis import (
    _check_vocal_activity_in_range,
    _check_for_silence_in_range,
)
from .timing_evaluator_correction_pipeline import (
    correct_line_timestamps_impl,
    fix_spurious_gaps_impl,
)

logger = logging.getLogger(__name__)


@dataclass
class TimingCorrectionHooks:
    """Optional runtime overrides for timing correction collaborators."""

    check_vocal_activity_in_range_fn: Optional[Callable[..., float]] = None
    check_for_silence_in_range_fn: Optional[Callable[..., bool]] = None
    find_phrase_end_fn: Optional[Callable[..., float]] = None
    find_best_onset_for_phrase_end_fn: Optional[Callable[..., Optional[float]]] = None
    find_best_onset_proximity_fn: Optional[Callable[..., Optional[float]]] = None
    find_best_onset_during_silence_fn: Optional[Callable[..., Optional[float]]] = None


_ACTIVE_HOOKS = TimingCorrectionHooks()


@contextmanager
def use_timing_correction_hooks(
    *,
    check_vocal_activity_in_range_fn: Optional[Callable[..., float]] = None,
    check_for_silence_in_range_fn: Optional[Callable[..., bool]] = None,
    find_phrase_end_fn: Optional[Callable[..., float]] = None,
    find_best_onset_for_phrase_end_fn: Optional[Callable[..., Optional[float]]] = None,
    find_best_onset_proximity_fn: Optional[Callable[..., Optional[float]]] = None,
    find_best_onset_during_silence_fn: Optional[Callable[..., Optional[float]]] = None,
) -> Iterator[None]:
    """Temporarily override timing correction collaborators for tests."""
    global _ACTIVE_HOOKS

    previous = _ACTIVE_HOOKS
    _ACTIVE_HOOKS = TimingCorrectionHooks(
        check_vocal_activity_in_range_fn=(
            check_vocal_activity_in_range_fn
            if check_vocal_activity_in_range_fn is not None
            else previous.check_vocal_activity_in_range_fn
        ),
        check_for_silence_in_range_fn=(
            check_for_silence_in_range_fn
            if check_for_silence_in_range_fn is not None
            else previous.check_for_silence_in_range_fn
        ),
        find_phrase_end_fn=(
            find_phrase_end_fn
            if find_phrase_end_fn is not None
            else previous.find_phrase_end_fn
        ),
        find_best_onset_for_phrase_end_fn=(
            find_best_onset_for_phrase_end_fn
            if find_best_onset_for_phrase_end_fn is not None
            else previous.find_best_onset_for_phrase_end_fn
        ),
        find_best_onset_proximity_fn=(
            find_best_onset_proximity_fn
            if find_best_onset_proximity_fn is not None
            else previous.find_best_onset_proximity_fn
        ),
        find_best_onset_during_silence_fn=(
            find_best_onset_during_silence_fn
            if find_best_onset_during_silence_fn is not None
            else previous.find_best_onset_during_silence_fn
        ),
    )
    try:
        yield
    finally:
        _ACTIVE_HOOKS = previous


def _check_vocal_activity(
    start: float, end: float, audio_features: AudioFeatures
) -> float:
    fn = (
        _ACTIVE_HOOKS.check_vocal_activity_in_range_fn or _check_vocal_activity_in_range
    )
    return fn(start, end, audio_features)


def _check_silence(
    start: float, end: float, audio_features: AudioFeatures, min_silence_duration: float
) -> bool:
    fn = _ACTIVE_HOOKS.check_for_silence_in_range_fn or _check_for_silence_in_range
    return fn(start, end, audio_features, min_silence_duration=min_silence_duration)


def _find_phrase_end_for_state(
    start_time: float,
    max_end_time: float,
    audio_features: AudioFeatures,
    min_silence_duration: float = 0.3,
) -> float:
    fn = _ACTIVE_HOOKS.find_phrase_end_fn
    if fn is not None:
        return fn(
            start_time,
            max_end_time,
            audio_features,
            min_silence_duration=min_silence_duration,
        )
    return _find_phrase_end(
        start_time,
        max_end_time,
        audio_features,
        min_silence_duration=min_silence_duration,
    )


def _find_best_onset_for_phrase_end_for_state(
    onset_times: np.ndarray,
    line_start: float,
    prev_line_audio_end: float,
    audio_features: AudioFeatures,
) -> Optional[float]:
    fn = _ACTIVE_HOOKS.find_best_onset_for_phrase_end_fn
    if fn is not None:
        return fn(onset_times, line_start, prev_line_audio_end, audio_features)
    return _find_best_onset_for_phrase_end(
        onset_times, line_start, prev_line_audio_end, audio_features
    )


def _find_best_onset_proximity_for_state(
    onset_times: np.ndarray,
    line_start: float,
    max_correction: float,
    audio_features: AudioFeatures,
) -> Optional[float]:
    fn = _ACTIVE_HOOKS.find_best_onset_proximity_fn
    if fn is not None:
        return fn(onset_times, line_start, max_correction, audio_features)
    return _find_best_onset_proximity(
        onset_times, line_start, max_correction, audio_features
    )


def _find_best_onset_during_silence_for_state(
    onset_times: np.ndarray,
    line_start: float,
    prev_line_audio_end: float,
    max_correction: float,
    audio_features: AudioFeatures,
) -> Optional[float]:
    fn = _ACTIVE_HOOKS.find_best_onset_during_silence_fn
    if fn is not None:
        return fn(
            onset_times,
            line_start,
            prev_line_audio_end,
            max_correction,
            audio_features,
        )
    return _find_best_onset_during_silence(
        onset_times,
        line_start,
        prev_line_audio_end,
        max_correction,
        audio_features,
    )


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
        onset_silence_before = _check_vocal_activity(
            max(0, onset - 0.5), onset - 0.1, audio_features
        )
        if onset_silence_before < 0.3:
            return float(onset)
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
        onset_silence_before = _check_vocal_activity(
            max(0, onset - 0.5), onset - 0.1, audio_features
        )

        score = distance
        if onset_silence_before < 0.3:
            score -= 1.0  # Bonus for following silence
        if onset < line_start - 0.5:
            score += 1.0  # Penalty for being too early

        if score < best_score:
            best_score = score
            best_onset = float(onset)

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
        silence_before = _check_vocal_activity(
            max(0, onset - 0.5), onset - 0.1, audio_features
        )
        if silence_before < 0.3:
            return float(onset)
    return None


def correct_line_timestamps(
    lines: List[Line],
    audio_features: Optional[AudioFeatures],
    max_correction: float = 3.0,
) -> Tuple[List[Line], List[str]]:
    """Correct line timestamps to align with detected vocal onsets.

    Heuristic:
        - Lyric start times often drift or are inaccurate in LRC files.
        - Audio onsets (sudden energy increases) are ground truth for syllable starts.
        - We try to snap the first word of a line to a nearby audio onset.

    Strategy:
        1. Proximity: If a strong onset is within `max_correction` window, snap to it.
           - Penalty for snapping to onsets preceded by vocal activity (likely mid-phrase).
           - Bonus for snapping to onsets preceded by silence (likely phrase start).
        2. Silence Fallback: If the LRC start time falls in a silent region (impossible),
           we search for the first valid onset *after* the previous phrase ended.

    Args:
        lines: Original lyrics lines with timing
        audio_features: Extracted audio features with onset times
        max_correction: Maximum correction window (seconds)

    Returns:
        Tuple of (corrected_lines, list of correction descriptions)
    """
    return correct_line_timestamps_impl(
        lines,
        audio_features,
        max_correction,
        check_vocal_activity_fn=_check_vocal_activity,
        find_best_onset_for_phrase_end_fn=_find_best_onset_for_phrase_end_for_state,
        find_best_onset_proximity_fn=_find_best_onset_proximity_for_state,
        find_best_onset_during_silence_fn=_find_best_onset_during_silence_for_state,
        find_phrase_end_fn=_find_phrase_end_for_state,
    )


def _find_phrase_end(
    start_time: float,
    max_end_time: float,
    audio_features: AudioFeatures,
    min_silence_duration: float = 0.3,
) -> float:
    """Find where a phrase actually ends by detecting silence.

    Heuristic:
        - Phrases end when the singer stops singing (silence).
        - We look for the first sustained silence (> min_silence_duration) after start_time.
        - If no silence is found before max_end_time, we assume the phrase fills the duration.

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

    Heuristic:
        - If two lines are separated by a gap, but the audio shows continuous singing
          (high vocal activity, no silence), they are likely part of the same phrase.
        - This happens when LRC splitters break lines arbitrarily or for visual layout.
        - Merging them allows us to re-calculate word timings over the full phrase duration.

    Args:
        lines: Original lyrics lines
        audio_features: Extracted audio features
        activity_threshold: Minimum vocal activity fraction to trigger merge

    Returns:
        Tuple of (fixed_lines, list of fix descriptions)
    """
    return fix_spurious_gaps_impl(
        lines,
        audio_features,
        activity_threshold,
        collect_lines_to_merge_fn=_collect_lines_to_merge,
        merge_lines_with_audio_fn=_merge_lines_with_audio,
    )


def _collect_lines_to_merge(
    lines: List[Line],
    start_idx: int,
    audio_features: AudioFeatures,
    activity_threshold: float,
) -> Tuple[List[Line], int]:
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
        mid_activity = _check_vocal_activity(gap_start, gap_end, audio_features)
        has_silence = _check_silence(
            gap_start, gap_end, audio_features, min_silence_duration=0.5
        )
        return mid_activity > max(0.6, activity_threshold) and not has_silence

    if gap_duration > 0.3:
        mid_start = gap_start + 0.15
        mid_end = gap_end - 0.15
        if mid_end > mid_start:
            mid_activity = _check_vocal_activity(mid_start, mid_end, audio_features)
            return mid_activity > max(0.7, activity_threshold)

    return False


def _merge_lines_with_audio(
    lines_to_merge: List[Line],
    next_line_start: Optional[float],
    audio_features: AudioFeatures,
) -> Tuple[Line, float]:
    merged_words = []
    for line in lines_to_merge:
        merged_words.extend(list(line.words))

    phrase_start = lines_to_merge[0].start_time
    search_end = next_line_start if next_line_start is not None else phrase_start + 30.0
    phrase_end = _find_phrase_end_for_state(
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
