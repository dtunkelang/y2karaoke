"""Local forced-alignment repair helpers shared by fallback integration."""

from __future__ import annotations

import re
from typing import List

from ... import models
from ..alignment import timing_models
from .whisper_forced_prefix_repairs import (
    find_exact_whisper_sequence_match as _find_exact_whisper_sequence_match,
)
from .whisper_forced_refrain_repairs import _neighbor_line_bounds
from .whisper_integration_align_heuristics import _retime_line_to_window

_LIGHT_LEADING_TOKENS = {"the", "a", "an"}
_LOW_SIGNAL_TOKENS = {
    "a",
    "an",
    "and",
    "i",
    "i'm",
    "if",
    "is",
    "it",
    "of",
    "oh",
    "the",
    "to",
    "uh",
    "you",
}
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _low_signal_penalty(token: str) -> int:
    return 1 if token in _LOW_SIGNAL_TOKENS else 0


def normalize_token(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def non_placeholder_whisper_word_count(
    whisper_words: List[timing_models.TranscriptionWord] | None,
) -> int:
    if not whisper_words:
        return 0
    count = 0
    for word in whisper_words:
        normalized = normalize_token(word.text)
        if not normalized or normalized == "vocal":
            continue
        count += 1
    return count


def mean_nearest_onset_distance(
    lines: List[models.Line],
    audio_features: timing_models.AudioFeatures | None,
) -> float | None:
    if (
        audio_features is None
        or audio_features.onset_times is None
        or len(audio_features.onset_times) == 0
    ):
        return None
    populated = [line for line in lines if line.words]
    if not populated:
        return None
    distances: list[float] = []
    onset_times = audio_features.onset_times
    for line in populated:
        distances.append(float(min(abs(onset_times - line.start_time))))
    if not distances:
        return None
    return sum(distances) / len(distances)


def shift_line(line: models.Line, delta: float) -> models.Line:
    return models.Line(
        words=[
            models.Word(
                text=word.text,
                start_time=word.start_time + delta,
                end_time=word.end_time + delta,
                singer=word.singer,
            )
            for word in line.words
        ],
        singer=line.singer,
    )


def _count_leading_light_tokens(normalized_tokens: List[str]) -> int:
    count = 0
    for token in normalized_tokens:
        if token in _LIGHT_LEADING_TOKENS:
            count += 1
            continue
        break
    return count


def _find_local_content_anchor_start(
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    content_token: str,
    line: models.Line,
    lookback_sec: float,
    lookahead_sec: float,
) -> float | None:
    nearby_matches = [
        word.start
        for word in whisper_words
        if normalize_token(word.text) == content_token
        and line.start_time - lookback_sec
        <= word.start
        <= line.end_time + lookahead_sec
    ]
    if not nearby_matches:
        return None
    return min(nearby_matches)


def _reanchor_delta_for_line(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_shift_sec: float,
    max_shift_sec: float,
    lookback_sec: float,
    lookahead_sec: float,
) -> float | None:
    if len(line.words) < 2:
        return None
    normalized = [normalize_token(word.text) for word in line.words]
    leading_count = _count_leading_light_tokens(normalized)
    if leading_count == 0 or leading_count >= len(line.words):
        return None

    content_token = normalized[leading_count]
    if not content_token:
        return None

    target_start = _find_local_content_anchor_start(
        whisper_words,
        content_token=content_token,
        line=line,
        lookback_sec=lookback_sec,
        lookahead_sec=lookahead_sec,
    )
    if target_start is None:
        return None

    delta = target_start - line.words[leading_count].start_time
    if delta < min_shift_sec or delta > max_shift_sec:
        return None
    return delta


def _can_apply_reanchored_line(
    adjusted: List[models.Line], idx: int, shifted_line: models.Line
) -> bool:
    if idx + 1 >= len(adjusted) or not adjusted[idx + 1].words:
        return True
    next_start = adjusted[idx + 1].start_time
    return shifted_line.end_time <= next_start - 0.04


def can_apply_reanchored_line(
    adjusted: List[models.Line], idx: int, shifted_line: models.Line
) -> bool:
    return _can_apply_reanchored_line(adjusted, idx, shifted_line)


def reanchor_forced_lines_to_local_content_words(
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    *,
    min_shift_sec: float = 1.0,
    max_shift_sec: float = 3.5,
    lookback_sec: float = 1.0,
    lookahead_sec: float = 4.0,
) -> tuple[List[models.Line], int]:
    if not whisper_words:
        return forced_lines, 0

    adjusted = list(forced_lines)
    shifted = 0
    for idx in range(len(adjusted) - 1, -1, -1):
        line = adjusted[idx]
        delta = _reanchor_delta_for_line(
            line,
            whisper_words,
            min_shift_sec=min_shift_sec,
            max_shift_sec=max_shift_sec,
            lookback_sec=lookback_sec,
            lookahead_sec=lookahead_sec,
        )
        if delta is None:
            continue

        shifted_line = shift_line(line, delta)
        if not _can_apply_reanchored_line(adjusted, idx, shifted_line):
            continue
        adjusted[idx] = shifted_line
        shifted += 1
    return adjusted, shifted


def restore_forced_leading_unmatched_prefix_starts_from_source(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    *,
    min_word_count: int = 6,
    min_leading_unmatched_tokens: int = 3,
    min_start_gain_sec: float = 0.25,
    max_start_gain_sec: float = 0.7,
    max_anchor_delta_sec: float = 0.2,
    lookback_sec: float = 0.25,
    lookahead_sec: float = 1.0,
) -> tuple[List[models.Line], int]:
    if not whisper_words:
        return forced_lines, 0

    repaired = list(forced_lines)
    restored = 0
    limit = min(len(baseline_lines), len(forced_lines))
    for idx in range(limit):
        baseline = baseline_lines[idx]
        forced = repaired[idx]
        if (
            not baseline.words
            or not forced.words
            or len(forced.words) != len(baseline.words)
            or len(forced.words) < min_word_count
        ):
            continue

        start_gain = baseline.start_time - forced.start_time
        if start_gain < min_start_gain_sec or start_gain > max_start_gain_sec:
            continue

        anchor = _best_leading_prefix_restore_anchor(
            baseline=baseline,
            forced=forced,
            whisper_words=whisper_words,
            lookback_sec=lookback_sec,
            lookahead_sec=lookahead_sec,
        )
        if anchor is None:
            continue
        best_anchor_idx, best_anchor_start = anchor
        if best_anchor_idx < min_leading_unmatched_tokens:
            continue
        if abs(baseline.start_time - best_anchor_start) > max_anchor_delta_sec:
            continue

        repaired[idx] = _retime_line_to_window(
            forced,
            window_start=baseline.start_time,
            window_end=forced.end_time,
        )
        restored += 1
    return repaired, restored


def _nearby_whisper_token_starts(
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    window_start: float,
    window_end: float,
) -> list[tuple[str, float]]:
    return [
        (normalize_token(word.text), float(word.start))
        for word in whisper_words
        if window_start <= word.start <= window_end
    ]


def _best_leading_prefix_restore_anchor(
    *,
    baseline: models.Line,
    forced: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    lookback_sec: float,
    lookahead_sec: float,
) -> tuple[int, float] | None:
    nearby = _nearby_whisper_token_starts(
        whisper_words,
        window_start=forced.start_time - lookback_sec,
        window_end=forced.end_time + lookahead_sec,
    )
    if not nearby:
        return None

    best_anchor: tuple[int, float] | None = None
    best_anchor_score: tuple[float, int, int] | None = None
    for token_idx, word in enumerate(forced.words):
        token = normalize_token(word.text)
        if not token:
            continue
        matches = [start for observed, start in nearby if observed == token]
        if not matches:
            continue
        anchor_start = min(matches, key=lambda start: abs(baseline.start_time - start))
        anchor_score = (
            abs(baseline.start_time - anchor_start),
            _low_signal_penalty(token),
            -token_idx,
        )
        if best_anchor_score is None or anchor_score < best_anchor_score:
            best_anchor = (token_idx, anchor_start)
            best_anchor_score = anchor_score
    return best_anchor


def _three_word_suffix_retime_tokens(line: models.Line) -> list[str] | None:
    if len(line.words) != 3:
        return None
    normalized = [normalize_token(word.text) for word in line.words]
    if any(not token for token in normalized):
        return None
    return normalized


def _three_word_suffix_retime_window(
    *,
    line: models.Line,
    suffix_match: tuple[float, float],
    prev_end: float | None,
    next_start: float | None,
    min_gap: float,
    min_start_gain_sec: float,
    min_suffix_span_sec: float,
    max_prefix_slot_sec: float,
) -> tuple[float, float, float, float, float] | None:
    suffix_start, suffix_end = suffix_match
    suffix_span = suffix_end - suffix_start
    if suffix_span < min_suffix_span_sec:
        return None

    prefix_slot = min(max_prefix_slot_sec, suffix_span / 2.0)
    target_start = suffix_start - prefix_slot
    if prev_end is not None:
        target_start = max(target_start, prev_end + min_gap)
    if target_start - line.start_time < min_start_gain_sec:
        return None
    if next_start is not None and suffix_end >= next_start - min_gap:
        return None

    prefix_end = min(suffix_start - 0.02, target_start + prefix_slot * 0.9)
    if prefix_end <= target_start:
        return None

    final_end = max(suffix_end, line.words[2].end_time)
    if next_start is not None:
        final_end = min(final_end, next_start - min_gap)
    if final_end <= suffix_start + 0.08:
        return None

    suffix_break = min(final_end - 0.02, suffix_start + suffix_span * 0.55)
    third_word_start = max(
        suffix_break,
        final_end - max(line.words[2].end_time - line.words[2].start_time, 0.18),
    )
    if third_word_start >= final_end:
        return None
    return target_start, prefix_end, suffix_start, suffix_break, final_end


def _rebuild_three_word_suffix_retimed_line(
    line: models.Line,
    *,
    target_start: float,
    prefix_end: float,
    suffix_start: float,
    suffix_break: float,
    final_end: float,
) -> models.Line:
    return models.Line(
        words=[
            models.Word(
                text=line.words[0].text,
                start_time=target_start,
                end_time=prefix_end,
                singer=line.words[0].singer,
            ),
            models.Word(
                text=line.words[1].text,
                start_time=suffix_start,
                end_time=suffix_break,
                singer=line.words[1].singer,
            ),
            models.Word(
                text=line.words[2].text,
                start_time=max(
                    suffix_break,
                    final_end
                    - max(line.words[2].end_time - line.words[2].start_time, 0.18),
                ),
                end_time=final_end,
                singer=line.words[2].singer,
            ),
        ],
        singer=line.singer,
    )


def retime_three_word_lines_from_suffix_matches(
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    *,
    min_gap: float = 0.05,
    search_lookback_sec: float = 0.3,
    search_lookahead_sec: float = 1.4,
    min_start_gain_sec: float = 0.35,
    min_suffix_span_sec: float = 0.35,
    max_prefix_slot_sec: float = 0.4,
    min_prefix_share: float = 0.55,
    min_relaxed_prefix_share: float = 0.2,
    min_relaxed_suffix_start_gain_sec: float = 0.8,
) -> tuple[List[models.Line], int]:
    if not whisper_words:
        return forced_lines, 0

    adjusted = list(forced_lines)
    restored = 0
    for idx, line in enumerate(adjusted):
        normalized = _three_word_suffix_retime_tokens(
            line,
        )
        if normalized is None:
            continue
        suffix_match = _find_exact_whisper_sequence_match(
            whisper_words,
            tokens=normalized[1:],
            search_start=line.start_time - search_lookback_sec,
            search_end=line.end_time + search_lookahead_sec,
            normalize_token_fn=normalize_token,
        )
        if suffix_match is None:
            continue
        line_duration = line.end_time - line.start_time
        if line_duration <= 0.0:
            continue
        prefix_duration = line.words[0].end_time - line.words[0].start_time
        prefix_share = prefix_duration / line_duration
        suffix_start = suffix_match[0]
        if prefix_share < min_prefix_share:
            if prefix_share < min_relaxed_prefix_share:
                continue
            if suffix_start - line.start_time < min_relaxed_suffix_start_gain_sec:
                continue

        prev_end, next_start = _neighbor_line_bounds(adjusted, idx)
        target_window = _three_word_suffix_retime_window(
            line=line,
            suffix_match=suffix_match,
            prev_end=prev_end,
            next_start=next_start,
            min_gap=min_gap,
            min_start_gain_sec=min_start_gain_sec,
            min_suffix_span_sec=min_suffix_span_sec,
            max_prefix_slot_sec=max_prefix_slot_sec,
        )
        if target_window is None:
            continue

        adjusted[idx] = _rebuild_three_word_suffix_retimed_line(
            line,
            target_start=target_window[0],
            prefix_end=target_window[1],
            suffix_start=target_window[2],
            suffix_break=target_window[3],
            final_end=target_window[4],
        )
        restored += 1
    return adjusted, restored
