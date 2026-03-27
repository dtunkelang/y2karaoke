"""Helpers for transcript-constrained WhisperX fallback alignment."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_forced_refrain_repairs import (
    _coerce_forced_segments,
    _enforce_repeated_short_refrain_followup_gap,
    _neighbor_line_bounds,
    _restore_short_refrains_from_aligned_segment_words,
)
from .whisper_forced_prefix_repairs import (
    find_exact_whisper_sequence_match as _find_exact_whisper_sequence_match,
    reanchor_medium_lines_to_earlier_exact_prefixes as _reanchor_medium_lines_to_earlier_exact_prefixes_impl,  # noqa: E501
)
from .whisper_forced_tail_repairs import (
    extend_low_score_forced_line_tails_from_source as _extend_low_score_forced_line_tails_from_source,  # noqa: E501
)
from .whisper_forced_sparse_followup_repairs import (
    restore_sparse_forced_followup_lines_from_source as _restore_sparse_forced_followup_lines_from_source,  # noqa: E501
)
from .whisper_forced_sparse_support_repairs import (
    redistribute_sparse_support_sustained_words as _redistribute_sparse_support_sustained_words,  # noqa: E501
    restore_compact_two_word_lines_from_source as _restore_compact_two_word_lines_from_source,  # noqa: E501
    restore_sparse_support_line_durations_from_source as _restore_sparse_support_line_durations_from_source,  # noqa: E501
    restore_sparse_support_line_starts_from_source as _restore_sparse_support_line_starts_from_source,  # noqa: E501
    shift_sparse_support_lines_toward_better_onsets as _shift_sparse_support_lines_toward_better_onsets,  # noqa: E501
)
from .whisper_forced_trace import (
    capture_forced_trace_snapshot as _capture_forced_trace_snapshot,
    maybe_write_forced_trace_snapshot_file as _maybe_write_forced_trace_snapshot_file,
    parse_forced_trace_line_range as _parse_forced_trace_line_range,
)
from .whisper_forced_advisory_trace import (
    maybe_write_forced_advisory_trace as _maybe_write_forced_advisory_trace,
)
from .whisper_forced_advisory_nudges import (
    apply_forced_advisory_start_nudges as _apply_forced_advisory_start_nudges,  # noqa: E501
)
from .whisper_split_refrain_restore import (
    restore_split_short_refrains_to_matching_segments as _restore_split_short_refrains_to_matching_segments,  # noqa: E501
)

_LIGHT_LEADING_TOKENS = {"the", "a", "an"}
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _normalize_token(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def _non_placeholder_whisper_word_count(
    whisper_words: List[timing_models.TranscriptionWord] | None,
) -> int:
    if not whisper_words:
        return 0
    count = 0
    for word in whisper_words:
        normalized = _normalize_token(word.text)
        if not normalized or normalized == "vocal":
            continue
        count += 1
    return count


def _mean_nearest_onset_distance(
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


def _shift_line(line: models.Line, delta: float) -> models.Line:
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
        if _normalize_token(word.text) == content_token
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
    normalized = [_normalize_token(word.text) for word in line.words]
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


def _count_sustained_line_degradations(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    min_baseline_duration_sec: float = 3.5,
    max_duration_ratio: float = 0.6,
) -> tuple[int, int]:
    compared = 0
    degraded = 0
    for baseline_line, forced_line in zip(baseline_lines, forced_lines):
        if not baseline_line.words or not forced_line.words:
            continue
        baseline_duration = baseline_line.end_time - baseline_line.start_time
        if baseline_duration < min_baseline_duration_sec:
            continue
        compared += 1
        forced_duration = forced_line.end_time - forced_line.start_time
        if forced_duration < baseline_duration * max_duration_ratio:
            degraded += 1
    return degraded, compared


def _should_rollback_sustained_line_degradation(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    min_degraded_lines: int = 2,
    min_degraded_ratio: float = 0.5,
) -> tuple[bool, int, int]:
    degraded, compared = _count_sustained_line_degradations(
        baseline_lines,
        forced_lines,
    )
    if compared == 0:
        return False, degraded, compared
    rollback = (
        degraded >= min_degraded_lines and degraded / compared >= min_degraded_ratio
    )
    return rollback, degraded, compared


def _restore_sustained_line_durations_from_source(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    min_baseline_duration_sec: float = 3.5,
    max_duration_ratio: float = 0.6,
    exact_start_restore_ratio: float = 0.25,
    exact_start_restore_min_shift_sec: float = 1.5,
    compact_recovered_word_count: int = 3,
    compact_recovered_shift_coeff: float = 0.12,
    compact_recovered_max_shift_sec: float = 0.55,
    compact_followed_by_longer_line_extra_shift_sec: float = 0.28,
    compact_followed_by_longer_line_min_words: int = 5,
    inter_line_gap_sec: float = 0.05,
) -> tuple[List[models.Line], int]:
    repaired = list(forced_lines)
    restored = 0
    for idx, (baseline_line, forced_line) in enumerate(
        zip(baseline_lines, forced_lines)
    ):
        sustained_restore = _restored_sustained_line(
            baseline_lines=baseline_lines,
            repaired_lines=repaired,
            idx=idx,
            baseline_line=baseline_line,
            forced_line=forced_line,
            min_baseline_duration_sec=min_baseline_duration_sec,
            max_duration_ratio=max_duration_ratio,
            exact_start_restore_ratio=exact_start_restore_ratio,
            exact_start_restore_min_shift_sec=exact_start_restore_min_shift_sec,
            compact_recovered_word_count=compact_recovered_word_count,
            compact_recovered_shift_coeff=compact_recovered_shift_coeff,
            compact_recovered_max_shift_sec=compact_recovered_max_shift_sec,
            compact_followed_by_longer_line_extra_shift_sec=(
                compact_followed_by_longer_line_extra_shift_sec
            ),
            compact_followed_by_longer_line_min_words=(
                compact_followed_by_longer_line_min_words
            ),
            inter_line_gap_sec=inter_line_gap_sec,
        )
        if sustained_restore is None:
            continue
        repaired[idx] = sustained_restore
        restored += 1
    return repaired, restored


def _restored_sustained_line(
    *,
    baseline_lines: List[models.Line],
    repaired_lines: List[models.Line],
    idx: int,
    baseline_line: models.Line,
    forced_line: models.Line,
    min_baseline_duration_sec: float,
    max_duration_ratio: float,
    exact_start_restore_ratio: float,
    exact_start_restore_min_shift_sec: float,
    compact_recovered_word_count: int,
    compact_recovered_shift_coeff: float,
    compact_recovered_max_shift_sec: float,
    compact_followed_by_longer_line_extra_shift_sec: float,
    compact_followed_by_longer_line_min_words: int,
    inter_line_gap_sec: float,
) -> models.Line | None:
    if not baseline_line.words or not forced_line.words:
        return None
    baseline_duration = baseline_line.end_time - baseline_line.start_time
    if baseline_duration < min_baseline_duration_sec:
        return None
    forced_duration = forced_line.end_time - forced_line.start_time
    if forced_duration >= baseline_duration * max_duration_ratio:
        return None
    start_shift = forced_line.start_time - baseline_line.start_time
    if (
        forced_duration <= baseline_duration * exact_start_restore_ratio
        and abs(start_shift) >= exact_start_restore_min_shift_sec
    ):
        return baseline_line

    repaired_line = _shift_line(baseline_line, start_shift)
    if idx <= 0 or len(baseline_line.words) != compact_recovered_word_count:
        return repaired_line
    later_shift = min(
        compact_recovered_max_shift_sec,
        max(0.0, (baseline_duration - forced_duration) * compact_recovered_shift_coeff),
    )
    if (
        idx + 1 < len(baseline_lines)
        and len(baseline_lines[idx + 1].words)
        >= compact_followed_by_longer_line_min_words
    ):
        later_shift = min(
            compact_recovered_max_shift_sec,
            later_shift + compact_followed_by_longer_line_extra_shift_sec,
        )
    if later_shift <= 0.0:
        return repaired_line
    return _shift_compact_recovered_line(
        repaired_line=repaired_line,
        repaired_lines=repaired_lines,
        idx=idx,
        later_shift=later_shift,
        inter_line_gap_sec=inter_line_gap_sec,
    )


def _shift_compact_recovered_line(
    *,
    repaired_line: models.Line,
    repaired_lines: List[models.Line],
    idx: int,
    later_shift: float,
    inter_line_gap_sec: float,
) -> models.Line:
    shifted_candidate = _shift_line(repaired_line, later_shift)
    if idx + 1 < len(repaired_lines) and repaired_lines[idx + 1].words:
        next_start = repaired_lines[idx + 1].start_time
        if shifted_candidate.end_time > next_start - inter_line_gap_sec:
            available = next_start - inter_line_gap_sec - repaired_line.end_time
            if available > 0.0:
                shifted_candidate = _shift_line(
                    repaired_line,
                    min(later_shift, available),
                )
    return shifted_candidate


def _count_compact_line_drift(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    max_baseline_duration_sec: float = 2.5,
    min_late_shift_sec: float = 1.5,
) -> tuple[int, int]:
    compared = 0
    degraded = 0
    for baseline_line, forced_line in zip(baseline_lines, forced_lines):
        if not baseline_line.words or not forced_line.words:
            continue
        baseline_duration = baseline_line.end_time - baseline_line.start_time
        if baseline_duration > max_baseline_duration_sec:
            continue
        compared += 1
        if forced_line.start_time - baseline_line.start_time >= min_late_shift_sec:
            degraded += 1
    return degraded, compared


def _should_rollback_compact_line_drift(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    min_degraded_lines: int = 3,
    min_degraded_ratio: float = 0.4,
) -> tuple[bool, int, int]:
    degraded, compared = _count_compact_line_drift(
        baseline_lines,
        forced_lines,
    )
    if compared == 0:
        return False, degraded, compared
    rollback = (
        degraded >= min_degraded_lines and degraded / compared >= min_degraded_ratio
    )
    return rollback, degraded, compared


def _count_compact_line_duration_collapse(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    max_baseline_words: int = 3,
    min_baseline_duration_sec: float = 0.9,
    max_duration_ratio: float = 0.45,
) -> tuple[int, int]:
    compared = 0
    degraded = 0
    for baseline_line, forced_line in zip(baseline_lines, forced_lines):
        if not baseline_line.words or not forced_line.words:
            continue
        if len(baseline_line.words) != len(forced_line.words):
            continue
        if len(baseline_line.words) > max_baseline_words:
            continue
        baseline_duration = baseline_line.end_time - baseline_line.start_time
        if baseline_duration < min_baseline_duration_sec:
            continue
        compared += 1
        forced_duration = forced_line.end_time - forced_line.start_time
        if (
            forced_duration <= 0.0
            or forced_duration <= baseline_duration * max_duration_ratio
        ):
            degraded += 1
    return degraded, compared


def _should_rollback_compact_line_duration_collapse(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    min_degraded_lines: int = 2,
    min_degraded_ratio: float = 0.4,
) -> tuple[bool, int, int]:
    degraded, compared = _count_compact_line_duration_collapse(
        baseline_lines,
        forced_lines,
    )
    if compared == 0:
        return False, degraded, compared
    rollback = (
        degraded >= min_degraded_lines and degraded / compared >= min_degraded_ratio
    )
    return rollback, degraded, compared


def _reject_compact_line_duration_collapse_if_needed(
    *,
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    logger: Any,
) -> bool:
    rollback_compact_duration, compact_duration_degraded, compact_duration_compared = (
        _should_rollback_compact_line_duration_collapse(
            baseline_lines,
            forced_lines,
        )
    )
    if not rollback_compact_duration:
        return False
    logger.warning(
        (
            "Discarded WhisperX forced alignment due to compact-line duration "
            "collapse (%d/%d)"
        ),
        compact_duration_degraded,
        compact_duration_compared,
    )
    return True


def _reanchor_forced_lines_to_local_content_words(
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

        shifted_line = _shift_line(line, delta)
        if not _can_apply_reanchored_line(adjusted, idx, shifted_line):
            continue
        adjusted[idx] = shifted_line
        shifted += 1
    return adjusted, shifted


def _three_word_suffix_retime_tokens(
    line: models.Line,
    *,
    min_prefix_share: float,
) -> list[str] | None:
    if len(line.words) != 3:
        return None
    line_duration = line.end_time - line.start_time
    if line_duration <= 0.0:
        return None
    prefix_duration = line.words[0].end_time - line.words[0].start_time
    if prefix_duration / line_duration < min_prefix_share:
        return None
    normalized = [_normalize_token(word.text) for word in line.words]
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


def _retime_three_word_lines_from_suffix_matches(
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
) -> tuple[List[models.Line], int]:
    if not whisper_words:
        return forced_lines, 0

    adjusted = list(forced_lines)
    restored = 0
    for idx, line in enumerate(adjusted):
        normalized = _three_word_suffix_retime_tokens(
            line,
            min_prefix_share=min_prefix_share,
        )
        if normalized is None:
            continue
        suffix_match = _find_exact_whisper_sequence_match(
            whisper_words,
            tokens=normalized[1:],
            search_start=line.start_time - search_lookback_sec,
            search_end=line.end_time + search_lookahead_sec,
            normalize_token_fn=_normalize_token,
        )
        if suffix_match is None:
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


def _forced_coverage_ok(
    *,
    logger: Any,
    forced_word_coverage: float,
    forced_line_coverage: float,
    min_forced_word_coverage: float,
    min_forced_line_coverage: float,
) -> bool:
    if (
        forced_word_coverage >= min_forced_word_coverage
        and forced_line_coverage >= min_forced_line_coverage
    ):
        return True
    logger.warning(
        (
            "Discarded WhisperX forced alignment due to low forced coverage "
            "(word=%.2f line=%.2f)"
        ),
        forced_word_coverage,
        forced_line_coverage,
    )
    return False


def _forced_alignment_hurts_sparse_onsets(
    *,
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    audio_features: timing_models.AudioFeatures | None,
    logger: Any,
) -> bool:
    lexical_word_count = _non_placeholder_whisper_word_count(whisper_words)
    baseline_onset_distance = _mean_nearest_onset_distance(
        baseline_lines, audio_features
    )
    forced_onset_distance = _mean_nearest_onset_distance(forced_lines, audio_features)
    if not (
        lexical_word_count <= 3
        and baseline_onset_distance is not None
        and forced_onset_distance is not None
        and baseline_onset_distance <= 0.2
        and forced_onset_distance >= baseline_onset_distance + 0.05
    ):
        return False
    logger.info(
        (
            "Discarded WhisperX forced alignment because sparse lexical support "
            "did not improve onset proximity (baseline=%.3f forced=%.3f)"
        ),
        baseline_onset_distance,
        forced_onset_distance,
    )
    return True


def _repair_short_line_degradation_if_possible(
    *,
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    logger: Any,
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
) -> List[models.Line] | None:
    rollback, short_before, short_after = should_rollback_short_line_degradation_fn(
        baseline_lines, forced_lines
    )
    if not rollback:
        return forced_lines
    repaired_lines, restored_count = restore_implausibly_short_lines_fn(
        baseline_lines, forced_lines
    )
    repaired_rollback, _, repaired_after = should_rollback_short_line_degradation_fn(
        baseline_lines, repaired_lines
    )
    if restored_count > 0 and not repaired_rollback:
        logger.info(
            (
                "Kept WhisperX forced alignment after restoring %d short "
                "baseline line(s) (%d -> %d)"
            ),
            restored_count,
            short_after,
            repaired_after,
        )
        return repaired_lines
    logger.warning(
        "Discarded WhisperX forced alignment due to short-line degradation (%d -> %d)",
        short_before,
        short_after,
    )
    return None


def _repair_sustained_line_degradation_if_possible(
    *,
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    logger: Any,
) -> List[models.Line] | None:
    rollback_sustained, sustained_degraded, sustained_compared = (
        _should_rollback_sustained_line_degradation(
            baseline_lines,
            forced_lines,
        )
    )
    if not rollback_sustained:
        return forced_lines
    repaired_lines, restored_count = _restore_sustained_line_durations_from_source(
        baseline_lines,
        forced_lines,
    )
    (
        repaired_rollback_sustained,
        repaired_sustained_degraded,
        repaired_sustained_compared,
    ) = _should_rollback_sustained_line_degradation(
        baseline_lines,
        repaired_lines,
    )
    if restored_count > 0 and not repaired_rollback_sustained:
        logger.info(
            (
                "Kept WhisperX forced alignment after restoring %d sustained line(s) "
                "from source duration (%d/%d -> %d/%d)"
            ),
            restored_count,
            sustained_degraded,
            sustained_compared,
            repaired_sustained_degraded,
            repaired_sustained_compared,
        )
        return repaired_lines
    logger.warning(
        "Discarded WhisperX forced alignment due to sustained-line compression (%d/%d)",
        sustained_degraded,
        sustained_compared,
    )
    return None


def _post_normalize_sparse_support_repairs(
    *,
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    audio_features: timing_models.AudioFeatures | None,
    logger: Any,
    normalize_line_word_timings_fn: Callable[..., Any] | None,
) -> List[models.Line]:
    if normalize_line_word_timings_fn is None:
        return forced_lines
    forced_lines = normalize_line_word_timings_fn(forced_lines)
    forced_lines, sparse_duration_restored_count = (
        _restore_sparse_support_line_durations_from_source(
            baseline_lines,
            forced_lines,
            whisper_words,
            non_placeholder_whisper_word_count_fn=_non_placeholder_whisper_word_count,
            shift_line_fn=_shift_line,
        )
    )
    if sparse_duration_restored_count:
        logger.info(
            (
                "Restored %d sparse-support line duration(s) from source "
                "after normalization"
            ),
            sparse_duration_restored_count,
        )
    forced_lines, sparse_onset_shift_count = (
        _shift_sparse_support_lines_toward_better_onsets(
            baseline_lines,
            forced_lines,
            whisper_words,
            audio_features,
            non_placeholder_whisper_word_count_fn=_non_placeholder_whisper_word_count,
            shift_line_fn=_shift_line,
        )
    )
    if sparse_onset_shift_count:
        logger.info(
            (
                "Shifted %d sparse-support line(s) to better local onsets "
                "after normalization"
            ),
            sparse_onset_shift_count,
        )
    forced_lines, sparse_start_restore_count = (
        _restore_sparse_support_line_starts_from_source(
            baseline_lines,
            forced_lines,
            whisper_words,
            audio_features,
            non_placeholder_whisper_word_count_fn=_non_placeholder_whisper_word_count,
            shift_line_fn=_shift_line,
        )
    )
    if sparse_start_restore_count:
        logger.info(
            "Restored %d sparse-support line start(s) from source after normalization",
            sparse_start_restore_count,
        )
    forced_lines, restored_two_word_compact_count = (
        _restore_compact_two_word_lines_from_source(
            baseline_lines,
            forced_lines,
        )
    )
    if restored_two_word_compact_count:
        logger.info(
            "Restored %d compact two-word line(s) from source after normalization",
            restored_two_word_compact_count,
        )
    forced_lines, sustained_word_redistributed_count = (
        _redistribute_sparse_support_sustained_words(
            baseline_lines,
            forced_lines,
            whisper_words,
            non_placeholder_whisper_word_count_fn=_non_placeholder_whisper_word_count,
        )
    )
    if sustained_word_redistributed_count:
        logger.info(
            "Redistributed %d sparse-support sustained line(s) for held final words",
            sustained_word_redistributed_count,
        )
    return forced_lines


def _apply_pre_finalize_forced_refrain_repairs(
    *,
    forced_lines: List[models.Line],
    logger: Any,
) -> tuple[List[models.Line], int]:
    forced_lines, shifted_refrain_gaps = _enforce_repeated_short_refrain_followup_gap(
        forced_lines
    )
    if shifted_refrain_gaps:
        logger.info(
            (
                "Shifted %d repeated short refrain line(s) later after "
                "long preceding lines"
            ),
            shifted_refrain_gaps,
        )
    return forced_lines, shifted_refrain_gaps


def _apply_post_finalize_forced_refrain_repairs(
    *,
    forced_lines: List[models.Line],
    aligned_segments: Any,
    forced_segments: List[timing_models.TranscriptionSegment],
    transcription: List[timing_models.TranscriptionSegment] | None,
    logger: Any,
) -> tuple[List[models.Line], int, int]:
    forced_lines, restored_word_sequence_refrains = (
        _restore_short_refrains_from_aligned_segment_words(
            forced_lines,
            aligned_segments,
        )
    )
    if restored_word_sequence_refrains:
        logger.info(
            "Restored %d short refrain line(s) from WhisperX aligned word sequences",
            restored_word_sequence_refrains,
        )
    forced_lines, restored_split_refrains = (
        _restore_split_short_refrains_to_matching_segments(
            forced_lines,
            forced_segments or transcription or [],
        )
    )
    if restored_split_refrains:
        logger.info(
            "Restored %d split short refrain line(s) after WhisperX forced alignment",
            restored_split_refrains,
        )
    return forced_lines, restored_word_sequence_refrains, restored_split_refrains


def _build_forced_payload(
    *,
    forced_word_coverage: float,
    forced_line_coverage: float,
    shifted_refrain_gaps: int,
    restored_word_sequence_refrains: int,
    restored_split_refrains: int,
    used_model: str,
    advisory_start_nudges: int = 0,
) -> Dict[str, Any]:
    return {
        "matched_ratio": forced_word_coverage,
        "word_coverage": forced_word_coverage,
        "avg_similarity": 1.0,
        "line_coverage": forced_line_coverage,
        "phonetic_similarity_coverage": forced_word_coverage,
        "high_similarity_ratio": 1.0,
        "exact_match_ratio": 0.0,
        "unmatched_ratio": 1.0 - forced_word_coverage,
        "dtw_used": 0.0,
        "dtw_mode": 0.0,
        "whisperx_forced": 1.0,
        "shifted_refrain_followup_gaps": float(shifted_refrain_gaps),
        "restored_word_sequence_refrains": float(restored_word_sequence_refrains),
        "restored_split_short_refrains": float(restored_split_refrains),
        "forced_advisory_start_nudges": float(advisory_start_nudges),
        "whisper_model": used_model,
    }


def _reject_compact_line_drift_if_needed(
    *,
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    logger: Any,
) -> bool:
    rollback_compact, compact_degraded, compact_compared = (
        _should_rollback_compact_line_drift(
            baseline_lines,
            forced_lines,
        )
    )
    if not rollback_compact:
        return False
    logger.warning(
        "Discarded WhisperX forced alignment due to compact-line drift (%d/%d)",
        compact_degraded,
        compact_compared,
    )
    return True


def _finalize_forced_line_timing(
    *,
    forced_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord] | None,
    audio_features: timing_models.AudioFeatures | None,
    logger: Any,
    normalize_line_word_timings_fn: Callable[..., Any] | None,
    enforce_monotonic_line_starts_fn: Callable[..., Any] | None,
    enforce_non_overlapping_lines_fn: Callable[..., Any] | None,
) -> List[models.Line]:
    forced_lines, reanchored_count = _reanchor_forced_lines_to_local_content_words(
        forced_lines,
        whisper_words,
    )
    if reanchored_count:
        logger.info(
            (
                "Reanchored %d forced-aligned line(s) to local content-word "
                "Whisper anchors"
            ),
            reanchored_count,
        )
    forced_lines, prefix_reanchored_count = (
        _reanchor_medium_lines_to_earlier_exact_prefixes_impl(
            forced_lines,
            whisper_words,
            normalize_token_fn=_normalize_token,
            can_apply_reanchored_line_fn=_can_apply_reanchored_line,
        )
    )
    if prefix_reanchored_count:
        logger.info(
            (
                "Reanchored %d medium forced-aligned line(s) to earlier exact "
                "Whisper prefixes"
            ),
            prefix_reanchored_count,
        )
    forced_lines, suffix_retimed_count = _retime_three_word_lines_from_suffix_matches(
        forced_lines,
        whisper_words,
    )
    if suffix_retimed_count:
        logger.info(
            (
                "Retimed %d compact forced-aligned line(s) from exact Whisper "
                "suffix matches"
            ),
            suffix_retimed_count,
        )

    forced_lines = _post_normalize_sparse_support_repairs(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        whisper_words=whisper_words,
        audio_features=audio_features,
        logger=logger,
        normalize_line_word_timings_fn=normalize_line_word_timings_fn,
    )
    if enforce_monotonic_line_starts_fn is not None:
        forced_lines = enforce_monotonic_line_starts_fn(forced_lines)
    if enforce_non_overlapping_lines_fn is not None:
        forced_lines = enforce_non_overlapping_lines_fn(forced_lines)
    return forced_lines


def _trace_forced_rejection(
    *,
    trace_path: str,
    trace_snapshots: list[dict[str, Any]],
    trace_metadata: dict[str, Any],
    status: str,
) -> None:
    _maybe_write_forced_trace_snapshot_file(
        trace_path=trace_path,
        snapshots=trace_snapshots,
        metadata={**trace_metadata, "status": status},
    )


def _apply_forced_pre_finalize_repairs(
    *,
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    logger: Any,
    trace_path: str,
    trace_snapshots: list[dict[str, Any]],
    trace_metadata: dict[str, Any],
    trace_line_range: tuple[int, int] | None,
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
) -> List[models.Line] | None:
    repaired_short_lines = _repair_short_line_degradation_if_possible(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        logger=logger,
        should_rollback_short_line_degradation_fn=(
            should_rollback_short_line_degradation_fn
        ),
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
    )
    if repaired_short_lines is None:
        _trace_forced_rejection(
            trace_path=trace_path,
            trace_snapshots=trace_snapshots,
            trace_metadata=trace_metadata,
            status="rejected_short_line_repair",
        )
        return None
    forced_lines = repaired_short_lines
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="after_short_line_repair",
        lines=forced_lines,
        line_range=trace_line_range,
    )

    repaired_sustained_lines = _repair_sustained_line_degradation_if_possible(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        logger=logger,
    )
    if repaired_sustained_lines is None:
        _trace_forced_rejection(
            trace_path=trace_path,
            trace_snapshots=trace_snapshots,
            trace_metadata=trace_metadata,
            status="rejected_sustained_line_repair",
        )
        return None
    forced_lines = repaired_sustained_lines
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="after_sustained_line_repair",
        lines=forced_lines,
        line_range=trace_line_range,
    )

    if _reject_compact_line_drift_if_needed(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        logger=logger,
    ):
        _trace_forced_rejection(
            trace_path=trace_path,
            trace_snapshots=trace_snapshots,
            trace_metadata=trace_metadata,
            status="rejected_compact_line_drift",
        )
        return None
    if _reject_compact_line_duration_collapse_if_needed(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        logger=logger,
    ):
        _trace_forced_rejection(
            trace_path=trace_path,
            trace_snapshots=trace_snapshots,
            trace_metadata=trace_metadata,
            status="rejected_pre_finalize_collapse",
        )
        return None
    return forced_lines


def _apply_forced_post_finalize_repairs(
    *,
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    aligned_segments: List[timing_models.TranscriptionSegment],
    forced_segments: List[timing_models.TranscriptionSegment],
    transcription: List[timing_models.TranscriptionSegment] | None,
    whisper_words: List[timing_models.TranscriptionWord] | None,
    vocals_path: str,
    language: str | None,
    detected_lang: str | None,
    used_model: str,
    logger: Any,
    trace_path: str,
    trace_snapshots: list[dict[str, Any]],
    trace_metadata: dict[str, Any],
    trace_line_range: tuple[int, int] | None,
) -> tuple[List[models.Line], int, int, int] | None:
    forced_lines, restored_sparse_followups = (
        _restore_sparse_forced_followup_lines_from_source(
            baseline_lines,
            forced_lines,
            aligned_segments,
        )
    )
    if restored_sparse_followups:
        logger.info(
            "Restored %d sparse forced followup line(s) from source timing",
            restored_sparse_followups,
        )
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="after_restore_sparse_followups",
        lines=forced_lines,
        line_range=trace_line_range,
    )

    forced_lines, extended_low_score_tails = (
        _extend_low_score_forced_line_tails_from_source(
            baseline_lines,
            forced_lines,
            aligned_segments,
        )
    )
    if extended_low_score_tails:
        logger.info(
            "Extended %d low-score forced-aligned line tail(s) toward source timing",
            extended_low_score_tails,
        )
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="after_extend_low_score_tails",
        lines=forced_lines,
        line_range=trace_line_range,
    )

    (
        forced_lines,
        restored_word_sequence_refrains,
        restored_split_refrains,
    ) = _apply_post_finalize_forced_refrain_repairs(
        forced_lines=forced_lines,
        aligned_segments=aligned_segments,
        forced_segments=forced_segments,
        transcription=transcription,
        logger=logger,
    )
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="after_post_finalize_refrain_repairs",
        lines=forced_lines,
        line_range=trace_line_range,
    )
    if _reject_compact_line_duration_collapse_if_needed(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        logger=logger,
    ):
        _trace_forced_rejection(
            trace_path=trace_path,
            trace_snapshots=trace_snapshots,
            trace_metadata=trace_metadata,
            status="rejected_post_finalize_collapse",
        )
        return None

    forced_lines, advisory_start_nudges = _apply_forced_advisory_start_nudges(
        lines=forced_lines,
        current_segments=transcription,
        current_words=whisper_words,
        vocals_path=vocals_path,
        language=language or detected_lang,
        model_size=used_model,
        logger=logger,
    )
    if advisory_start_nudges:
        logger.info(
            "Applied %d advisory start nudge(s) from alternate transcription support",
            advisory_start_nudges,
        )
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="after_advisory_start_nudges",
        lines=forced_lines,
        line_range=trace_line_range,
    )
    _maybe_write_forced_advisory_trace(
        lines=forced_lines,
        current_segments=transcription,
        current_words=whisper_words,
        vocals_path=vocals_path,
        language=language or detected_lang,
        model_size=used_model,
        logger=logger,
    )
    return (
        forced_lines,
        restored_word_sequence_refrains,
        restored_split_refrains,
        advisory_start_nudges,
    )


def attempt_whisperx_forced_alignment(
    *,
    lines: List[models.Line],
    baseline_lines: List[models.Line],
    vocals_path: str,
    language: str | None,
    detected_lang: str | None,
    logger: Any,
    used_model: str,
    reason: str,
    align_lines_with_whisperx_fn: Callable[..., Any],
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    whisper_words: List[timing_models.TranscriptionWord] | None = None,
    transcription: List[timing_models.TranscriptionSegment] | None = None,
    audio_features: timing_models.AudioFeatures | None = None,
    normalize_line_word_timings_fn: Callable[..., Any] | None = None,
    enforce_monotonic_line_starts_fn: Callable[..., Any] | None = None,
    enforce_non_overlapping_lines_fn: Callable[..., Any] | None = None,
    min_forced_word_coverage: float = 0.2,
    min_forced_line_coverage: float = 0.2,
) -> Optional[Tuple[List[models.Line], List[str], Dict[str, Any]]]:
    trace_path = os.environ.get("Y2K_TRACE_FORCED_FALLBACK_STAGES_JSON", "").strip()
    trace_line_range = _parse_forced_trace_line_range()
    trace_snapshots: list[dict[str, Any]] = []
    trace_metadata: dict[str, Any] = {
        "reason": reason,
        "used_model": used_model,
        "vocals_path": vocals_path,
    }
    forced_result = _load_forced_alignment_result(
        lines=lines,
        vocals_path=vocals_path,
        language=language,
        detected_lang=detected_lang,
        logger=logger,
        align_lines_with_whisperx_fn=align_lines_with_whisperx_fn,
    )
    if forced_result is None:
        return None
    (
        forced_lines,
        forced_word_coverage,
        forced_line_coverage,
        aligned_segments,
        forced_segments,
    ) = forced_result
    trace_metadata.update(
        {
            "forced_word_coverage": forced_word_coverage,
            "forced_line_coverage": forced_line_coverage,
            "aligned_segment_count": (
                len(aligned_segments) if isinstance(aligned_segments, list) else 0
            ),
        }
    )
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="loaded_forced_alignment",
        lines=forced_lines,
        line_range=trace_line_range,
    )
    if not _forced_alignment_is_usable(
        logger=logger,
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        forced_word_coverage=forced_word_coverage,
        forced_line_coverage=forced_line_coverage,
        min_forced_word_coverage=min_forced_word_coverage,
        min_forced_line_coverage=min_forced_line_coverage,
        whisper_words=whisper_words,
        audio_features=audio_features,
    ):
        _trace_forced_rejection(
            trace_path=trace_path,
            trace_snapshots=trace_snapshots,
            trace_metadata=trace_metadata,
            status="rejected_before_repair",
        )
        return None
    repaired_lines = _apply_forced_pre_finalize_repairs(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        logger=logger,
        trace_path=trace_path,
        trace_snapshots=trace_snapshots,
        trace_metadata=trace_metadata,
        trace_line_range=trace_line_range,
        should_rollback_short_line_degradation_fn=(
            should_rollback_short_line_degradation_fn
        ),
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
    )
    if repaired_lines is None:
        return None
    forced_lines = repaired_lines

    forced_lines, shifted_refrain_gaps = _apply_pre_finalize_forced_refrain_repairs(
        forced_lines=forced_lines,
        logger=logger,
    )
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="after_pre_finalize_refrain_repairs",
        lines=forced_lines,
        line_range=trace_line_range,
    )
    forced_lines = _finalize_forced_line_timing(
        forced_lines=forced_lines,
        baseline_lines=baseline_lines,
        whisper_words=whisper_words,
        audio_features=audio_features,
        logger=logger,
        normalize_line_word_timings_fn=normalize_line_word_timings_fn,
        enforce_monotonic_line_starts_fn=enforce_monotonic_line_starts_fn,
        enforce_non_overlapping_lines_fn=enforce_non_overlapping_lines_fn,
    )
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="after_finalize_forced_line_timing",
        lines=forced_lines,
        line_range=trace_line_range,
    )
    post_finalize_result = _apply_forced_post_finalize_repairs(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        aligned_segments=aligned_segments,
        forced_segments=forced_segments,
        transcription=transcription,
        whisper_words=whisper_words,
        vocals_path=vocals_path,
        language=language,
        detected_lang=detected_lang,
        used_model=used_model,
        logger=logger,
        trace_path=trace_path,
        trace_snapshots=trace_snapshots,
        trace_metadata=trace_metadata,
        trace_line_range=trace_line_range,
    )
    if post_finalize_result is None:
        return None
    (
        forced_lines,
        restored_word_sequence_refrains,
        restored_split_refrains,
        advisory_start_nudges,
    ) = post_finalize_result

    forced_payload = _build_forced_payload(
        forced_word_coverage=forced_word_coverage,
        forced_line_coverage=forced_line_coverage,
        shifted_refrain_gaps=shifted_refrain_gaps,
        restored_word_sequence_refrains=restored_word_sequence_refrains,
        restored_split_refrains=restored_split_refrains,
        used_model=used_model,
        advisory_start_nudges=advisory_start_nudges,
    )
    _capture_forced_trace_snapshot(
        trace_snapshots,
        stage="final_forced_lines",
        lines=forced_lines,
        line_range=trace_line_range,
    )
    _maybe_write_forced_trace_snapshot_file(
        trace_path=trace_path,
        snapshots=trace_snapshots,
        metadata={**trace_metadata, "status": "accepted", **forced_payload},
    )
    return (
        forced_lines,
        [f"Applied WhisperX transcript-constrained forced alignment due to {reason}"],
        forced_payload,
    )


def _load_forced_alignment_result(
    *,
    lines: List[models.Line],
    vocals_path: str,
    language: str | None,
    detected_lang: str | None,
    logger: Any,
    align_lines_with_whisperx_fn: Callable[..., Any],
) -> (
    tuple[
        List[models.Line],
        float,
        float,
        Any,
        List[timing_models.TranscriptionSegment],
    ]
    | None
):
    forced_language = language or detected_lang
    forced = align_lines_with_whisperx_fn(lines, vocals_path, forced_language, logger)
    if forced is None:
        return None
    forced_lines, forced_metrics = forced
    aligned_segments = forced_metrics.get("aligned_segments")
    return (
        forced_lines,
        float(forced_metrics.get("forced_word_coverage", 0.0)),
        float(forced_metrics.get("forced_line_coverage", 0.0)),
        aligned_segments,
        _coerce_forced_segments(aligned_segments),
    )


def _forced_alignment_is_usable(
    *,
    logger: Any,
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    forced_word_coverage: float,
    forced_line_coverage: float,
    min_forced_word_coverage: float,
    min_forced_line_coverage: float,
    whisper_words: List[timing_models.TranscriptionWord] | None,
    audio_features: timing_models.AudioFeatures | None,
) -> bool:
    if not _forced_coverage_ok(
        logger=logger,
        forced_word_coverage=forced_word_coverage,
        forced_line_coverage=forced_line_coverage,
        min_forced_word_coverage=min_forced_word_coverage,
        min_forced_line_coverage=min_forced_line_coverage,
    ):
        return False
    return not _forced_alignment_hurts_sparse_onsets(
        baseline_lines=baseline_lines,
        forced_lines=forced_lines,
        whisper_words=whisper_words,
        audio_features=audio_features,
        logger=logger,
    )
