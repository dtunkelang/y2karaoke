"""Short-line silence/onset refinement helpers for Whisper alignment."""

from __future__ import annotations

from typing import Callable, List, Optional, Set, Tuple

from ...models import Line


def _short_line_silence_shift_candidate(
    line: Line,
    normalized_silences: List[Tuple[float, float]],
    onset_times,
    *,
    find_internal_silences_fn: Callable[
        [float, float, List[Tuple[float, float]]], List[Tuple[float, float]]
    ],
    has_near_start_silence_fn: Callable[[float, List[Tuple[float, float]]], bool],
    first_onset_after_fn: Callable[..., Optional[float]],
) -> Optional[float]:
    internal_silences = find_internal_silences_fn(
        line.start_time, line.end_time, normalized_silences
    )
    if not internal_silences or not has_near_start_silence_fn(
        line.start_time, internal_silences
    ):
        return None
    _, silence_end = internal_silences[-1]
    target_start = first_onset_after_fn(onset_times, start=silence_end, window=1.8)
    if target_start is None:
        return None
    desired_shift = target_start - line.start_time
    if desired_shift < 0.8:
        return None
    return desired_shift


def _short_line_run_end(lines: List[Line], start_idx: int) -> int:
    run_end = start_idx + 1
    while run_end < len(lines):
        prev_line = lines[run_end - 1]
        run_line = lines[run_end]
        if not prev_line.words or not run_line.words:
            break
        if len(run_line.words) > 4:
            break
        if run_line.start_time - prev_line.end_time > 2.2:
            break
        if run_end - start_idx >= 2:
            break
        run_end += 1
    return run_end


def _available_shift_for_run(
    lines: List[Line], start_idx: int, run_end: int, desired_shift: float
) -> float:
    if run_end < len(lines) and lines[run_end].words:
        return min(
            desired_shift,
            max(0.0, lines[run_end].start_time - 0.05 - lines[run_end - 1].end_time),
        )
    return desired_shift


def _apply_shift_to_short_run(
    lines: List[Line],
    start_idx: int,
    run_end: int,
    shift: float,
    *,
    shift_line_words_fn: Callable[[Line, float], Line],
    compact_short_line_if_needed_fn: Callable[..., Line],
) -> None:
    for run_idx in range(start_idx, run_end):
        run_line = lines[run_idx]
        shifted_line = shift_line_words_fn(run_line, shift)
        if len(run_line.words) <= 4 and shifted_line.end_time > shifted_line.start_time:
            max_duration = 1.25 if len(run_line.words) <= 3 else 1.6
            next_start = None
            if run_idx + 1 < run_end and lines[run_idx + 1].words:
                next_start = lines[run_idx + 1].start_time + shift
            elif run_idx + 1 < len(lines) and lines[run_idx + 1].words:
                next_start = lines[run_idx + 1].start_time
            shifted_line = compact_short_line_if_needed_fn(
                shifted_line, max_duration=max_duration, next_start=next_start
            )
        lines[run_idx] = shifted_line


def _shift_short_line_runs_after_silence(
    lines: List[Line],
    normalized_silences: List[Tuple[float, float]],
    onset_times,
    *,
    shift_line_words_fn: Callable[[Line, float], Line],
    compact_short_line_if_needed_fn: Callable[..., Line],
    find_internal_silences_fn: Callable[
        [float, float, List[Tuple[float, float]]], List[Tuple[float, float]]
    ],
    has_near_start_silence_fn: Callable[[float, List[Tuple[float, float]]], bool],
    first_onset_after_fn: Callable[..., Optional[float]],
) -> int:
    fixes = 0
    shifted_indices: Set[int] = set()
    idx = 1
    while idx < len(lines) - 1:
        if idx in shifted_indices:
            idx += 1
            continue
        line = lines[idx]
        if not line.words or len(line.words) > 4:
            idx += 1
            continue
        if line.end_time <= line.start_time + 0.2:
            idx += 1
            continue

        desired_shift = _short_line_silence_shift_candidate(
            line,
            normalized_silences,
            onset_times,
            find_internal_silences_fn=find_internal_silences_fn,
            has_near_start_silence_fn=has_near_start_silence_fn,
            first_onset_after_fn=first_onset_after_fn,
        )
        if desired_shift is None:
            idx += 1
            continue

        run_end = _short_line_run_end(lines, idx)
        available = _available_shift_for_run(lines, idx, run_end, desired_shift)
        if available < 0.4:
            idx += 1
            continue

        _apply_shift_to_short_run(
            lines,
            idx,
            run_end,
            available,
            shift_line_words_fn=shift_line_words_fn,
            compact_short_line_if_needed_fn=compact_short_line_if_needed_fn,
        )
        shifted_indices.update(range(idx, run_end))
        fixes += 1
        idx += 1
    return fixes


def _shift_single_short_lines_after_silence(
    lines: List[Line],
    normalized_silences: List[Tuple[float, float]],
    onset_times,
    *,
    shift_line_words_fn: Callable[[Line, float], Line],
    compact_short_line_if_needed_fn: Callable[..., Line],
    find_internal_silences_fn: Callable[
        [float, float, List[Tuple[float, float]]], List[Tuple[float, float]]
    ],
    has_near_start_silence_fn: Callable[[float, List[Tuple[float, float]]], bool],
    first_onset_after_fn: Callable[..., Optional[float]],
) -> int:
    fixes = 0
    for idx in range(1, len(lines) - 1):
        line = lines[idx]
        if not line.words or len(line.words) > 4:
            continue
        next_line = lines[idx + 1]
        if not next_line.words:
            continue

        desired_shift = _short_line_silence_shift_candidate(
            line,
            normalized_silences,
            onset_times,
            find_internal_silences_fn=find_internal_silences_fn,
            has_near_start_silence_fn=has_near_start_silence_fn,
            first_onset_after_fn=first_onset_after_fn,
        )
        if desired_shift is None:
            continue
        available = max(0.0, next_line.start_time - 0.05 - line.end_time)
        shift = min(desired_shift, available)
        if shift < 0.4:
            continue
        shifted_line = shift_line_words_fn(line, shift)
        if len(line.words) <= 4 and shifted_line.end_time > shifted_line.start_time:
            max_duration = 1.25 if len(line.words) <= 3 else 1.6
            shifted_line = compact_short_line_if_needed_fn(
                shifted_line,
                max_duration=max_duration,
                next_start=next_line.start_time,
            )
        lines[idx] = shifted_line
        fixes += 1
    return fixes


def _compact_short_lines_near_silence(
    lines: List[Line],
    normalized_silences: List[Tuple[float, float]],
    *,
    compact_short_line_if_needed_fn: Callable[..., Line],
    find_internal_silences_fn: Callable[
        [float, float, List[Tuple[float, float]]], List[Tuple[float, float]]
    ],
    has_near_start_silence_fn: Callable[[float, List[Tuple[float, float]]], bool],
) -> int:
    fixes = 0
    for idx in range(1, len(lines) - 1):
        line = lines[idx]
        if not line.words or len(line.words) > 4:
            continue
        next_line = lines[idx + 1]
        if not next_line.words:
            continue
        internal_silences = find_internal_silences_fn(
            line.start_time, line.end_time, normalized_silences
        )
        if not internal_silences or not has_near_start_silence_fn(
            line.start_time, internal_silences
        ):
            continue
        max_duration = 1.25 if len(line.words) <= 3 else 1.6
        compacted = compact_short_line_if_needed_fn(
            line,
            max_duration=max_duration,
            next_start=next_line.start_time,
        )
        if compacted is line:
            continue
        lines[idx] = compacted
        fixes += 1
    return fixes


def _stretch_similar_adjacent_short_lines(
    lines: List[Line],
    normalized_silences: List[Tuple[float, float]],
    *,
    token_overlap_fn: Callable[[str, str], float],
    rebuild_line_with_target_end_fn: Callable[[Line, float], Optional[Line]],
) -> int:
    fixes = 0
    for idx in range(1, len(lines)):
        prev_line = lines[idx - 1]
        line = lines[idx]
        if not prev_line.words or not line.words:
            continue
        if len(prev_line.words) > 5 or len(line.words) > 4:
            continue
        if token_overlap_fn(prev_line.text, line.text) < 0.25:
            continue
        gap = line.start_time - prev_line.end_time
        if gap <= 1.0:
            continue
        target_end = line.start_time - 0.05
        latest_silence_start = None
        for start, end in normalized_silences:
            if start >= prev_line.end_time and end <= line.start_time:
                latest_silence_start = start
        if latest_silence_start is not None:
            target_end = min(target_end, latest_silence_start - 0.05)
        if target_end <= prev_line.end_time + 0.6:
            continue
        stretched = rebuild_line_with_target_end_fn(prev_line, target_end)
        if stretched is None:
            continue
        if target_end - prev_line.start_time <= 0.4:
            continue
        lines[idx - 1] = stretched
        fixes += 1
    return fixes


def _cap_isolated_short_lines(
    lines: List[Line],
    *,
    rebuild_line_with_target_end_fn: Callable[[Line, float], Optional[Line]],
) -> int:
    fixes = 0
    for idx in range(1, len(lines) - 1):
        line = lines[idx]
        prev_line = lines[idx - 1]
        next_line = lines[idx + 1]
        if not line.words or len(line.words) > 3:
            continue
        if not prev_line.words or not next_line.words:
            continue
        duration = line.end_time - line.start_time
        if duration <= 1.25:
            continue
        prev_gap = line.start_time - prev_line.end_time
        next_gap = next_line.start_time - line.end_time
        if prev_gap <= 0.5 or next_gap <= 1.0:
            continue
        target_end = min(line.start_time + 1.25, next_line.start_time - 0.05)
        compacted = rebuild_line_with_target_end_fn(line, target_end)
        if compacted is None:
            continue
        lines[idx] = compacted
        fixes += 1
    return fixes
