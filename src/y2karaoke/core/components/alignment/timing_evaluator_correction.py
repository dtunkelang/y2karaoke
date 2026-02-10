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
    from ...models import Line, Word

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
            _check_vocal_activity(line_start - 0.5, line_start + 0.5, audio_features)
            > 0.3
        )

        best_onset = None

        if len(onset_times) > 0:
            if singing_at_lrc_time:
                silence_after = (
                    _check_vocal_activity(
                        line_start + 0.1, line_start + 0.6, audio_features
                    )
                    < 0.3
                )
                silence_before = (
                    _check_vocal_activity(
                        max(0, line_start - 0.6), line_start - 0.1, audio_features
                    )
                    < 0.3
                )

                if silence_after and not silence_before:
                    best_onset = _find_best_onset_for_phrase_end_for_state(
                        onset_times, line_start, prev_line_audio_end, audio_features
                    )
                else:
                    best_onset = _find_best_onset_proximity_for_state(
                        onset_times, line_start, max_correction, audio_features
                    )
            else:
                best_onset = _find_best_onset_during_silence_for_state(
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
                    prev_line_audio_end = _find_phrase_end_for_state(
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
        prev_line_audio_end = _find_phrase_end_for_state(
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
    from ...models import Line

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
