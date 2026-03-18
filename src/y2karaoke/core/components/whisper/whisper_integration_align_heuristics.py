"""Heuristic line-shape helpers for Whisper alignment refinement."""

from __future__ import annotations

import re
from typing import Any, List, Optional

from ... import models
from ..alignment import timing_models
from .whisper_integration_align_experimental import (
    count_non_vocal_words_near_time as _count_non_vocal_words_near_time,
    local_lexical_overlap_ratio as _local_lexical_overlap_ratio,
    normalized_prefix_tokens as _normalized_prefix_tokens,
    normalized_tokens as _normalized_tokens,
    rescale_line_to_new_start as _rescale_line_to_new_start,
)
from .whisper_integration_late_run import (
    late_run_is_restorable,
    late_run_shift_for_baseline_restore,
)


def _line_set_end(lines: List[models.Line]) -> float:
    end_time = 0.0
    for line in lines:
        if line.words:
            end_time = max(end_time, line.end_time)
    return end_time


def _extend_last_word_end(line: models.Line, target_end: float) -> models.Line:
    words = [
        models.Word(
            text=w.text,
            start_time=w.start_time,
            end_time=(target_end if idx == len(line.words) - 1 else w.end_time),
            singer=w.singer,
        )
        for idx, w in enumerate(line.words)
    ]
    return models.Line(words=words, singer=line.singer)


def _retime_line_to_window(
    line: models.Line,
    *,
    window_start: float,
    window_end: float,
) -> models.Line:
    total_duration = max(window_end - window_start, 0.2)
    spacing = total_duration / len(line.words)
    new_words = []
    for word_idx, w in enumerate(line.words):
        start = window_start + word_idx * spacing
        end = start + spacing * 0.9
        new_words.append(
            models.Word(
                text=w.text,
                start_time=start,
                end_time=end,
                singer=w.singer,
            )
        )
    return models.Line(words=new_words, singer=line.singer)


def _extend_interjection_line_end(
    line: models.Line,
    *,
    target_end: float,
) -> models.Line:
    total_duration = max(target_end - line.start_time, 0.2)
    spacing = total_duration / len(line.words)
    new_words = []
    for word_idx, w in enumerate(line.words):
        start = line.start_time + word_idx * spacing
        end = start + spacing * 0.9
        new_words.append(
            models.Word(
                text=w.text,
                start_time=start,
                end_time=end,
                singer=w.singer,
            )
        )
    return models.Line(words=new_words, singer=line.singer)


def _restore_consistently_late_runs_from_baseline(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    *,
    min_run_length: int = 3,
    min_shift_sec: float = 0.8,
    max_shift_sec: float = 2.2,
    max_shift_spread_sec: float = 0.8,
    min_median_shift_sec: float = 1.15,
) -> tuple[List[models.Line], int]:
    limit = min(len(mapped_lines), len(baseline_lines))
    shifts = [
        late_run_shift_for_baseline_restore(
            mapped_lines[idx],
            baseline_lines[idx],
            min_shift_sec=min_shift_sec,
            max_shift_sec=max_shift_sec,
        )
        for idx in range(limit)
    ]

    repaired = list(mapped_lines)
    restored = 0
    idx = 0
    while idx < limit:
        if shifts[idx] is None:
            idx += 1
            continue
        run_start = idx
        run_values: list[float] = []
        while idx < limit and shifts[idx] is not None:
            shift_value = shifts[idx]
            assert shift_value is not None
            run_values.append(shift_value)
            idx += 1
        run_end = idx
        if not late_run_is_restorable(
            run_values,
            min_run_length=min_run_length,
            max_shift_spread_sec=max_shift_spread_sec,
            min_median_shift_sec=min_median_shift_sec,
        ):
            continue
        for line_idx in range(run_start, run_end):
            repaired[line_idx] = baseline_lines[line_idx]
            restored += 1
    return repaired, restored


def _restore_late_enumeration_lines_from_baseline(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
) -> tuple[List[models.Line], int]:
    repaired = list(mapped_lines)
    restored = 0
    limit = min(len(mapped_lines), len(baseline_lines))
    for idx in range(limit):
        mapped = mapped_lines[idx]
        baseline = baseline_lines[idx]
        if not mapped.words or not baseline.words:
            continue
        tokens = [re.sub(r"[^a-z0-9']+", "", w.text.lower()) for w in mapped.words]
        nonempty = [t for t in tokens if t]
        if not (3 <= len(nonempty) <= 6):
            continue
        short_count = sum(1 for t in nonempty if len(t) <= 3)
        if short_count < max(3, len(nonempty) - 1):
            continue
        if mapped.text.count(",") < 3:
            continue
        start_shift = mapped.start_time - baseline.start_time
        baseline_duration = baseline.end_time - baseline.start_time
        mapped_duration = mapped.end_time - mapped.start_time
        if start_shift < 1.5:
            continue
        if baseline_duration <= 0.0 or mapped_duration < baseline_duration * 1.5:
            continue
        repaired[idx] = baseline
        restored += 1
    return repaired, restored


def _rescale_line_to_new_end(line: models.Line, target_end: float) -> models.Line:
    old_duration = line.end_time - line.start_time
    new_duration = target_end - line.start_time
    span = old_duration if old_duration > 0 else 1.0
    rescaled_words: list[models.Word] = []
    for word in line.words:
        rel_start = (word.start_time - line.start_time) / span
        rel_end = (word.end_time - line.start_time) / span
        rescaled_words.append(
            models.Word(
                text=word.text,
                start_time=line.start_time + rel_start * new_duration,
                end_time=line.start_time + rel_end * new_duration,
                singer=word.singer,
            )
        )
    return models.Line(words=rescaled_words, singer=line.singer)


def _choose_parenthetical_tail_extension_end(
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 6:
        return None
    if ")" not in line.words[-1].text:
        return None
    if _normalized_prefix_tokens(next_line)[:2] != ["i", "said"]:
        return None
    if _count_non_vocal_words_near_time(whisper_words, line.start_time, window_sec=1.0):
        return None
    gap_after = next_line.start_time - line.end_time
    if gap_after < 1.2 or gap_after > 2.4:
        return None
    target_end = next_line.start_time - 0.25
    if target_end <= line.end_time + 0.8:
        return None
    return target_end


def _extend_unsupported_parenthetical_tails(
    mapped_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
) -> tuple[List[models.Line], int]:
    updated = list(mapped_lines)
    applied = 0
    for idx in range(len(updated) - 1):
        line = updated[idx]
        next_line = updated[idx + 1]
        target_end = _choose_parenthetical_tail_extension_end(
            line, next_line, whisper_words
        )
        if target_end is None:
            continue
        updated[idx] = _extend_last_word_end(line, target_end)
        applied += 1
    return updated, applied


def _choose_i_said_tail_extension_end(
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 7:
        return None
    if _normalized_prefix_tokens(line)[:2] != ["i", "said"]:
        return None
    nearby_count = _count_non_vocal_words_near_time(
        whisper_words,
        line.start_time,
        window_sec=1.0,
    )
    if nearby_count > 1:
        return None
    next_tokens = _normalized_prefix_tokens(next_line)
    if not next_tokens or next_tokens[0] != "no":
        return None
    gap_after = next_line.start_time - line.end_time
    min_gap = 0.8 if nearby_count == 1 else 1.2
    if gap_after < min_gap or gap_after > 2.0:
        return None
    target_end = next_line.start_time - 0.22
    if target_end <= line.end_time + 0.7:
        return None
    return target_end


def _extend_unsupported_i_said_tails(
    mapped_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
) -> tuple[List[models.Line], int]:
    updated = list(mapped_lines)
    applied = 0
    for idx in range(len(updated) - 1):
        line = updated[idx]
        next_line = updated[idx + 1]
        target_end = _choose_i_said_tail_extension_end(line, next_line, whisper_words)
        if target_end is None:
            continue
        updated[idx] = _rescale_line_to_new_end(line, target_end)
        applied += 1
    return updated, applied


def _choose_weak_opening_extension_end(
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 7:
        return None
    tokens = _normalized_prefix_tokens(line)
    if not tokens or tokens[0] not in {"oh", "maybe", "no", "cause"}:
        return None
    if tokens[0] == "no" and _normalized_prefix_tokens(next_line)[:2] == ["i", "said"]:
        return None
    if _count_non_vocal_words_near_time(whisper_words, line.start_time, window_sec=1.0):
        return None
    gap_after = next_line.start_time - line.end_time
    if gap_after < 0.8 or gap_after > 1.8:
        return None
    target_end = next_line.start_time - 0.3
    if target_end <= line.end_time + 0.5:
        return None
    return target_end


def _extend_unsupported_weak_opening_lines(
    mapped_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
) -> tuple[List[models.Line], int]:
    updated = list(mapped_lines)
    applied = 0
    for idx in range(len(updated) - 1):
        line = updated[idx]
        next_line = updated[idx + 1]
        target_end = _choose_weak_opening_extension_end(line, next_line, whisper_words)
        if target_end is None:
            continue
        updated[idx] = _rescale_line_to_new_end(line, target_end)
        applied += 1
    return updated, applied


def _restore_zero_support_parenthetical_late_start_expansions(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
) -> tuple[List[models.Line], int]:
    updated = list(mapped_lines)
    applied = 0
    for idx, line in enumerate(updated):
        if idx >= len(baseline_lines):
            break
        baseline_line = baseline_lines[idx]
        if not line.words or not baseline_line.words or len(line.words) < 5:
            continue
        if ")" not in line.words[-1].text:
            continue
        start_shift = line.start_time - baseline_line.start_time
        duration_growth = (line.end_time - line.start_time) - (
            baseline_line.end_time - baseline_line.start_time
        )
        if start_shift < 0.75 or duration_growth < 0.6:
            continue
        if _count_non_vocal_words_near_time(
            whisper_words,
            line.start_time,
            window_sec=1.0,
        ):
            continue
        updated[idx] = _rescale_line_to_new_start(line, baseline_line.start_time)
        applied += 1
    return updated, applied


def _choose_interjection_window_from_onsets(
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    onset_times: Any,
) -> Optional[tuple[float, float, bool]]:
    tokens = _normalized_tokens(line)
    if not line.words or len(line.words) > 3 or not tokens:
        return None
    if set(tokens) - {"hey", "oh", "ooh", "ah", "yeah"}:
        return None
    if _count_non_vocal_words_near_time(whisper_words, line.start_time, window_sec=1.0):
        return None
    gap_after = next_line.start_time - line.end_time
    if gap_after < 4.0:
        return None
    candidate_onsets = onset_times[
        (onset_times >= line.start_time + 0.5)
        & (onset_times <= min(next_line.start_time - 0.2, line.start_time + 2.5))
    ]
    if len(candidate_onsets) < 2:
        return None
    target_start = float(candidate_onsets[0])
    target_end = float(candidate_onsets[-1])
    onset_span = target_end - target_start
    if onset_span < 1.0:
        shift = target_start - line.start_time
        very_sparse_hey_ok = (
            len(candidate_onsets) == 2
            and onset_span >= 0.3
            and shift > 0.9
            and shift <= 2.0
            and gap_after >= 9.5
            and set(tokens) == {"hey"}
        )
        if not very_sparse_hey_ok and (
            onset_span < 0.6
            or shift > 0.9
            or gap_after < 8.0
            or len(candidate_onsets) != 2
        ):
            return None
        if very_sparse_hey_ok:
            target_end = min(next_line.start_time - 0.2, target_end + 0.6)
            return line.start_time, target_end, True
        target_end = min(next_line.start_time - 0.2, target_start + 1.5)
    return target_start, target_end, False


def _reanchor_unsupported_interjection_lines_to_onsets(
    mapped_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    audio_features: Optional[timing_models.AudioFeatures],
    *,
    max_gap_before_sec: float = 2.0,
) -> tuple[List[models.Line], int]:
    if audio_features is None or audio_features.onset_times is None:
        return mapped_lines, 0
    onset_times = audio_features.onset_times
    if len(onset_times) == 0:
        return mapped_lines, 0

    updated = list(mapped_lines)
    applied = 0
    for idx in range(len(updated) - 1):
        line = updated[idx]
        next_line = updated[idx + 1]
        if idx > 0 and updated[idx - 1].words:
            gap_before = line.start_time - updated[idx - 1].end_time
            if gap_before > max_gap_before_sec:
                continue
        window = _choose_interjection_window_from_onsets(
            line,
            next_line,
            whisper_words,
            onset_times,
        )
        if window is None:
            continue
        if window[2]:
            updated[idx] = _extend_interjection_line_end(
                line,
                target_end=window[1],
            )
        else:
            updated[idx] = _retime_line_to_window(
                line,
                window_start=window[0],
                window_end=window[1],
            )
        applied += 1
    return updated, applied


def _choose_long_line_pre_weak_opening_extension_end(
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 7:
        return None
    next_tokens = _normalized_tokens(next_line)
    if not next_tokens or next_tokens[0] not in {"oh", "maybe", "no", "cause"}:
        return None
    if _count_non_vocal_words_near_time(whisper_words, line.start_time, window_sec=1.0):
        return None
    gap_after = next_line.start_time - line.end_time
    if gap_after < 0.8 or gap_after > 1.6:
        return None
    target_end = next_line.start_time - 0.1
    if target_end <= line.end_time + 0.5:
        return None
    return target_end


def _extend_unsupported_long_lines_before_weak_opening(
    mapped_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
) -> tuple[List[models.Line], int]:
    updated = list(mapped_lines)
    applied = 0
    for idx in range(len(updated) - 1):
        line = updated[idx]
        next_line = updated[idx + 1]
        target_end = _choose_long_line_pre_weak_opening_extension_end(
            line,
            next_line,
            whisper_words,
        )
        if target_end is None:
            continue
        updated[idx] = _rescale_line_to_new_end(line, target_end)
        applied += 1
    return updated, applied


def _choose_pre_i_said_extension_end(
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 6:
        return None
    if _normalized_prefix_tokens(next_line)[:2] != ["i", "said"]:
        return None
    if _normalized_prefix_tokens(line)[:2] == ["i", "said"]:
        return None
    gap_after = next_line.start_time - line.end_time
    local_density = _count_non_vocal_words_near_time(
        whisper_words, line.start_time, window_sec=0.5
    )
    overlap_ratio = _local_lexical_overlap_ratio(line, whisper_words)
    if gap_after < 1.0 or gap_after > 2.2:
        return None
    if local_density == 0:
        return None
    max_overlap_ratio = 0.25 if local_density <= 2 else 0.2
    if overlap_ratio > max_overlap_ratio:
        return None
    target_end = next_line.start_time - 0.2
    if target_end <= line.end_time + 0.5:
        return None
    return target_end


def _extend_misaligned_lines_before_i_said(
    mapped_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
) -> tuple[List[models.Line], int]:
    updated = list(mapped_lines)
    applied = 0
    for idx in range(len(updated) - 1):
        line = updated[idx]
        next_line = updated[idx + 1]
        target_end = _choose_pre_i_said_extension_end(line, next_line, whisper_words)
        if target_end is None:
            continue
        updated[idx] = _rescale_line_to_new_end(line, target_end)
        applied += 1
    return updated, applied
