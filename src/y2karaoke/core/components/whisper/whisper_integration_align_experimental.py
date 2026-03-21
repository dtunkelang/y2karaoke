"""Experimental follow-up helpers for Whisper alignment."""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional

from ... import models
from ..alignment import timing_models


def enable_low_support_onset_reanchor() -> bool:
    return os.getenv("Y2K_WHISPER_ENABLE_LOW_SUPPORT_ONSET_REANCHOR", "0") == "1"


def enable_repeat_cadence_reanchor() -> bool:
    return os.getenv("Y2K_WHISPER_ENABLE_REPEAT_CADENCE_REANCHOR", "0") == "1"


def enable_restored_run_onset_shift() -> bool:
    return os.getenv("Y2K_WHISPER_ENABLE_RESTORED_RUN_ONSET_SHIFT", "0") == "1"


def count_non_vocal_words_near_time(
    words: List[timing_models.TranscriptionWord],
    center_time: float,
    *,
    window_sec: float = 1.0,
) -> int:
    lo = center_time - window_sec
    hi = center_time + window_sec
    count = 0
    for word in words:
        if word.text == "[VOCAL]":
            continue
        if lo <= word.start <= hi:
            count += 1
    return count


def normalized_prefix_tokens(line: models.Line, *, limit: int = 3) -> list[str]:
    return [
        re.sub(r"[^a-z]+", "", w.text.lower())
        for w in line.words[:limit]
        if re.sub(r"[^a-z]+", "", w.text.lower())
    ]


def normalized_tokens(line: models.Line) -> list[str]:
    return [
        re.sub(r"[^a-z]+", "", w.text.lower())
        for w in line.words
        if re.sub(r"[^a-z]+", "", w.text.lower())
    ]


def rescale_line_to_new_start(line: models.Line, target_start: float) -> models.Line:
    old_duration = line.end_time - line.start_time
    new_duration = line.end_time - target_start
    span = old_duration if old_duration > 0 else 1.0
    reanchored_words: list[models.Word] = []
    for word in line.words:
        rel_start = (word.start_time - line.start_time) / span
        rel_end = (word.end_time - line.start_time) / span
        reanchored_words.append(
            models.Word(
                text=word.text,
                start_time=target_start + rel_start * new_duration,
                end_time=target_start + rel_end * new_duration,
                singer=word.singer,
            )
        )
    return models.Line(words=reanchored_words, singer=line.singer)


def line_text_key(line: models.Line) -> str:
    return " ".join(normalized_tokens(line))


def local_lexical_overlap_ratio(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    pad_before: float = 0.8,
    pad_after: float = 0.8,
) -> float:
    line_tokens = set(normalized_tokens(line))
    if not line_tokens:
        return 0.0
    nearby_tokens = {
        re.sub(r"[^a-z]+", "", word.text.lower())
        for word in whisper_words
        if line.start_time - pad_before <= word.start <= line.end_time + pad_after
        and word.text != "[VOCAL]"
    }
    nearby_tokens.discard("")
    if not nearby_tokens:
        return 0.0
    overlap = len(line_tokens & nearby_tokens)
    return overlap / max(len(line_tokens), len(nearby_tokens))


def reanchor_late_supported_lines_to_earlier_whisper(
    lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_shift: float = 0.18,
    max_shift: float = 0.95,
    min_prefix_matches: int = 2,
    min_gap: float = 0.05,
    max_prev_overlap: float = 0.18,
    support_window: float = 1.2,
) -> tuple[List[models.Line], int]:
    adjusted = list(lines)
    applied = 0
    normalized_whisper = [
        re.sub(r"[^a-z]+", "", word.text.lower()) for word in whisper_words
    ]
    for idx, line in enumerate(adjusted):
        if len(line.words) < max(2, min_prefix_matches):
            continue
        prefix = normalized_prefix_tokens(line, limit=3)
        if len(prefix) < min_prefix_matches:
            continue

        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        search_start = line.start_time - max_shift
        search_end = line.start_time - min_shift
        best_target: float | None = None
        best_match_count = 0

        for word_idx, word in enumerate(whisper_words):
            if word.text == "[VOCAL]":
                continue
            if word.start < search_start or word.start > search_end:
                continue
            if normalized_whisper[word_idx] != prefix[0]:
                continue
            match_count = 1
            cursor = word_idx + 1
            last_end = word.end
            for token in prefix[1:]:
                while cursor < len(whisper_words):
                    candidate = whisper_words[cursor]
                    candidate_norm = normalized_whisper[cursor]
                    if candidate.start > word.start + support_window:
                        cursor = len(whisper_words)
                        break
                    cursor += 1
                    if candidate.text == "[VOCAL]" or not candidate_norm:
                        continue
                    if candidate.start + 1e-6 < last_end:
                        continue
                    if candidate_norm != token:
                        continue
                    match_count += 1
                    last_end = candidate.end
                    break
                else:
                    break

            if match_count < min_prefix_matches:
                continue
            target_start = word.start
            if prev_end is not None:
                target_start = max(target_start, prev_end + min_gap)
                overlap = prev_end - word.start
                if overlap > max_prev_overlap:
                    continue
            if target_start > line.start_time - min_shift:
                continue
            if match_count > best_match_count or (
                match_count == best_match_count
                and (best_target is None or target_start < best_target)
            ):
                best_target = target_start
                best_match_count = match_count

        if best_target is None:
            continue
        adjusted[idx] = rescale_line_to_new_start(line, best_target)
        applied += 1

    return adjusted, applied


def _choose_i_said_reanchor_start(
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    onset_times: Any,
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 6:
        return None
    if normalized_prefix_tokens(line)[:2] != ["i", "said"]:
        return None
    if count_non_vocal_words_near_time(whisper_words, line.start_time, window_sec=0.9):
        return None
    gap_after = next_line.start_time - line.end_time
    if gap_after < 0.0 or gap_after > 1.8:
        return None
    candidate_onsets = onset_times[
        (onset_times >= line.start_time + 0.35)
        & (onset_times <= min(line.start_time + 1.2, line.end_time - 3.5))
    ]
    if len(candidate_onsets) == 0:
        return None
    target_start = float(candidate_onsets[0])
    old_duration = line.end_time - line.start_time
    new_duration = line.end_time - target_start
    if new_duration < 3.5 or new_duration < old_duration * 0.72:
        return None
    return target_start


def choose_low_support_reanchor_start(  # noqa: C901
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    onset_times: Any,
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 4:
        return None
    tokens = normalized_tokens(line)
    if len(tokens) < 3:
        return None
    if normalized_prefix_tokens(line)[:2] == ["i", "said"]:
        return None
    if set(tokens) <= {"hey", "oh", "ooh", "ah", "yeah"}:
        return None

    gap_after = next_line.start_time - line.end_time
    if gap_after < 0.0 or gap_after > 2.5:
        return None

    local_density = count_non_vocal_words_near_time(
        whisper_words,
        line.start_time,
        window_sec=0.9,
    )
    if local_density > 1:
        return None
    overlap_ratio = local_lexical_overlap_ratio(
        line,
        whisper_words,
        pad_before=0.7,
        pad_after=0.7,
    )
    if overlap_ratio > 0.2:
        return None

    max_target_start = min(line.start_time + 1.4, line.end_time - 1.0)
    candidate_onsets = onset_times[
        (onset_times >= line.start_time + 0.25) & (onset_times <= max_target_start)
    ]
    if len(candidate_onsets) == 0:
        return None
    target_start = float(candidate_onsets[0])
    if target_start - line.start_time < 0.25:
        return None

    old_duration = line.end_time - line.start_time
    new_duration = line.end_time - target_start
    min_duration = max(1.2, 0.18 * len(line.words))
    if new_duration < min_duration or new_duration < old_duration * 0.68:
        return None
    return target_start


def reanchor_repeated_cadence_lines(  # noqa: C901
    mapped_lines: List[models.Line],
    *,
    min_word_count: int = 5,
    min_shift_sec: float = 0.25,
    max_shift_sec: float = 0.6,
    min_pair_gap_sec: float = 1.0,
) -> tuple[List[models.Line], int]:
    adjusted = list(mapped_lines)
    text_to_indices: dict[str, list[int]] = {}
    for idx, line in enumerate(adjusted):
        if not line.words:
            continue
        key = line_text_key(line)
        if key:
            text_to_indices.setdefault(key, []).append(idx)

    applied = 0
    for idx in range(1, len(adjusted)):
        line = adjusted[idx]
        prev_line = adjusted[idx - 1]
        if not line.words or not prev_line.words or len(line.words) < min_word_count:
            continue
        key = line_text_key(line)
        prev_key = line_text_key(prev_line)
        if not key or not prev_key:
            continue
        later_line_indices = [i for i in text_to_indices.get(key, []) if i > idx]
        later_prev_idx = next(
            (
                later_idx - 1
                for later_idx in later_line_indices
                if later_idx - 1 >= 0
                and line_text_key(adjusted[later_idx - 1]) == prev_key
            ),
            None,
        )
        if later_prev_idx is None:
            continue
        later_idx = later_prev_idx + 1

        current_gap = line.start_time - prev_line.start_time
        later_gap = adjusted[later_idx].start_time - adjusted[later_prev_idx].start_time
        desired_shift = later_gap - current_gap
        if desired_shift < min_shift_sec or desired_shift > max_shift_sec:
            continue
        if later_gap < min_pair_gap_sec or current_gap < min_pair_gap_sec:
            continue

        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )
        max_allowed = desired_shift
        if next_start is not None:
            max_allowed = min(
                max_allowed,
                max(0.0, (next_start - 0.05) - line.end_time),
            )
        if max_allowed < min_shift_sec:
            continue

        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + max_allowed,
                end_time=w.end_time + max_allowed,
                singer=w.singer,
            )
            for w in line.words
        ]
        adjusted[idx] = models.Line(words=shifted_words, singer=line.singer)
        applied += 1
    return adjusted, applied


def shift_restored_low_support_runs_to_onset(  # noqa: C901
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    audio_features: Optional[timing_models.AudioFeatures],
    *,
    baseline_anchor_tolerance: float = 0.08,
    max_shift_sec: float = 0.65,
    min_shift_sec: float = 0.18,
    onset_window_sec: float = 0.8,
    max_run_size: int = 4,
    max_relative_index_ratio: float = 0.5,
    max_lexical_overlap_ratio: float = 0.12,
    moderate_support_skip_min: int = 2,
    moderate_support_skip_max: int = 12,
    moderate_support_long_line_word_count: int = 8,
) -> tuple[List[models.Line], int]:
    if audio_features is None or audio_features.onset_times is None:
        return mapped_lines, 0
    onset_times = audio_features.onset_times
    if len(onset_times) == 0:
        return mapped_lines, 0

    adjusted = list(mapped_lines)
    applied = 0
    idx = 0
    limit = min(len(adjusted), len(baseline_lines))
    while idx < limit:
        if idx / max(limit, 1) > max_relative_index_ratio:
            break
        line = adjusted[idx]
        base = baseline_lines[idx]
        if (
            not line.words
            or not base.words
            or len(line.words) < 4
            or abs(line.start_time - base.start_time) > baseline_anchor_tolerance
            or local_lexical_overlap_ratio(
                line, whisper_words, pad_before=0.7, pad_after=0.7
            )
            > max_lexical_overlap_ratio
        ):
            idx += 1
            continue
        support_count = count_non_vocal_words_near_time(
            whisper_words,
            line.start_time,
            window_sec=0.9,
        )
        if (
            moderate_support_skip_min <= support_count <= moderate_support_skip_max
            and len(line.words) >= moderate_support_long_line_word_count
        ):
            idx += 1
            continue

        run_end = idx + 1
        while run_end < limit and run_end - idx < max_run_size:
            cur = adjusted[run_end]
            cur_base = baseline_lines[run_end]
            prev = adjusted[run_end - 1]
            if (
                not cur.words
                or not cur_base.words
                or len(cur.words) < 4
                or abs(cur.start_time - cur_base.start_time) > baseline_anchor_tolerance
                or cur.start_time - prev.end_time > 0.9
                or local_lexical_overlap_ratio(
                    cur, whisper_words, pad_before=0.7, pad_after=0.7
                )
                > 0.2
            ):
                break
            run_end += 1

        candidates = onset_times[
            (onset_times >= line.start_time + min_shift_sec)
            & (onset_times <= line.start_time + onset_window_sec)
        ]
        if len(candidates) == 0:
            idx = run_end
            continue
        desired_shift = min(float(candidates[0]) - line.start_time, max_shift_sec)
        if desired_shift < min_shift_sec:
            idx = run_end
            continue
        next_start = (
            adjusted[run_end].start_time
            if run_end < len(adjusted) and adjusted[run_end].words
            else None
        )
        max_allowed = desired_shift
        if next_start is not None:
            max_allowed = min(
                max_allowed,
                max(0.0, (next_start - 0.05) - adjusted[run_end - 1].end_time),
            )
        if max_allowed < min_shift_sec:
            idx = run_end
            continue

        for run_idx in range(idx, run_end):
            run_line = adjusted[run_idx]
            shifted_words = [
                models.Word(
                    text=w.text,
                    start_time=w.start_time + max_allowed,
                    end_time=w.end_time + max_allowed,
                    singer=w.singer,
                )
                for w in run_line.words
            ]
            adjusted[run_idx] = models.Line(words=shifted_words, singer=run_line.singer)
        applied += run_end - idx
        idx = run_end
    return adjusted, applied


def reanchor_unsupported_i_said_lines_to_later_onset(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    audio_features: Optional[timing_models.AudioFeatures],
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
        baseline_line = baseline_lines[idx] if idx < len(baseline_lines) else None
        if (
            baseline_line is not None
            and baseline_line.words
            and line.start_time - baseline_line.start_time > 0.75
        ):
            continue
        target_start = _choose_i_said_reanchor_start(
            line, next_line, whisper_words, onset_times
        )
        if target_start is None:
            continue
        updated[idx] = rescale_line_to_new_start(line, target_start)
        applied += 1
    return updated, applied


def reanchor_low_support_lines_to_later_onset(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    audio_features: Optional[timing_models.AudioFeatures],
    *,
    baseline_anchor_tolerance: float = 0.08,
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
        baseline_line = baseline_lines[idx] if idx < len(baseline_lines) else None
        if (
            baseline_line is not None
            and baseline_line.words
            and abs(line.start_time - baseline_line.start_time)
            > baseline_anchor_tolerance
        ):
            continue
        target_start = choose_low_support_reanchor_start(
            line,
            next_line,
            whisper_words,
            onset_times,
        )
        if target_start is None:
            continue
        updated[idx] = rescale_line_to_new_start(line, target_start)
        applied += 1
    return updated, applied
