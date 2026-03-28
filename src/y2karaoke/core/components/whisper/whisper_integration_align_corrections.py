"""Correction-pass orchestration for Whisper alignment refinement."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ... import models
from ..alignment import timing_models
from . import whisper_integration_align_experimental as _align_experimental
from .whisper_integration_align_heuristics import (
    _extend_misaligned_lines_before_i_said,
    _extend_unsupported_i_said_tails,
    _extend_unsupported_long_lines_before_weak_opening,
    _extend_unsupported_parenthetical_tails,
    _extend_unsupported_weak_opening_lines,
    _line_set_end,
    _retime_line_to_window,
    _reanchor_unsupported_interjection_lines_to_onsets,
    _restore_consistently_late_runs_from_baseline,
    _restore_late_enumeration_lines_from_baseline,
    _restore_zero_support_parenthetical_late_start_expansions,
)
from .whisper_mapping_post_text import (
    _normalize_match_token,
    _soft_token_overlap_ratio,
)
from .whisper_integration_align_trace import (
    capture_trace_snapshot as _capture_trace_snapshot,
)
from .whisper_integration_finalize import _restore_pairwise_inversions_from_source
from .whisper_integration_stages import (
    _shift_weak_opening_lines_past_phrase_carryover,
)
from . import whisper_integration_weak_evidence as _weak_evidence
from .whisper_runtime_config import WhisperRuntimeConfig

_reanchor_late_compact_tail = (
    _align_experimental.reanchor_late_compact_repetitive_tail_lines_to_later_onsets
)
_reanchor_late_supported_lines_to_earlier_whisper = (
    _align_experimental.reanchor_late_supported_lines_to_earlier_whisper
)
_reanchor_light_leading_lines_to_content_words = (
    _align_experimental.reanchor_light_leading_lines_to_content_words
)
_reanchor_low_support_lines_to_later_onset = (
    _align_experimental.reanchor_low_support_lines_to_later_onset
)
_reanchor_repeated_cadence_lines = _align_experimental.reanchor_repeated_cadence_lines
_reanchor_truncated_followup_lines_from_phonetic_variants = (
    _align_experimental.reanchor_truncated_followup_lines_from_phonetic_variants
)
_rebalance_short_followup_boundaries_from_whisper = (
    _align_experimental.rebalance_short_followup_boundaries_from_whisper
)
_reanchor_unsupported_i_said_lines_to_later_onset = (
    _align_experimental.reanchor_unsupported_i_said_lines_to_later_onset
)
_shift_restored_low_support_runs_to_onset = (
    _align_experimental.shift_restored_low_support_runs_to_onset
)
_restore_adjacent_near_threshold_late_shifts = (
    _weak_evidence.restore_adjacent_near_threshold_late_shifts
)
_restore_weak_evidence_large_start_shifts = (
    _weak_evidence.restore_weak_evidence_large_start_shifts
)
_restore_unsupported_early_duplicate_shifts = (
    _weak_evidence.restore_unsupported_early_duplicate_shifts
)


def _capture_stage_lines(
    *,
    mapped_lines: List[models.Line],
    trace_snapshots: list[dict[str, Any]],
    trace_line_range: Any,
    stage: str,
) -> None:
    _capture_trace_snapshot(
        trace_snapshots,
        stage=stage,
        lines=mapped_lines,
        line_range=trace_line_range,
    )


def _append_correction_if_any(
    corrections: List[str],
    count: int,
    message: str,
) -> None:
    if count:
        corrections.append(message.format(count=count))


def _normalized_line_tokens(line: models.Line) -> list[str]:
    return [
        token for word in line.words if (token := _normalize_match_token(word.text))
    ]


def _is_alternating_three_word_hook_pair(
    prev_tokens: list[str], cur_tokens: list[str]
) -> bool:
    return (
        len(prev_tokens) == 3
        and len(cur_tokens) == 3
        and prev_tokens[0] == cur_tokens[0]
        and prev_tokens[1] == cur_tokens[2]
        and prev_tokens[2] == cur_tokens[1]
        and prev_tokens[1] != prev_tokens[2]
    )


def _find_exact_phrase_window(
    whisper_words: List[timing_models.TranscriptionWord],
    tokens: list[str],
) -> tuple[float, float] | None:
    filtered_words = [
        (token, word)
        for word in whisper_words
        if word.text != "[VOCAL]"
        if (token := _normalize_match_token(word.text))
    ]
    token_count = len(tokens)
    if token_count == 0 or len(filtered_words) < token_count:
        return None
    for idx in range(0, len(filtered_words) - token_count + 1):
        window_tokens = [filtered_words[idx + off][0] for off in range(token_count)]
        if window_tokens != tokens:
            continue
        return (
            float(filtered_words[idx][1].start),
            float(filtered_words[idx + token_count - 1][1].end),
        )
    return None


def _alternating_middle_hook_has_shape(
    *,
    prev_base: models.Line,
    base: models.Line,
    mapped: models.Line,
    next_line: models.Line,
    min_baseline_duration_sec: float,
    max_start_delta_sec: float,
    max_next_overlap_ratio: float,
) -> bool:
    if not prev_base.words or not base.words or not mapped.words or not next_line.words:
        return False
    prev_tokens = _normalized_line_tokens(prev_base)
    cur_tokens = _normalized_line_tokens(base)
    next_tokens = _normalized_line_tokens(next_line)
    if not _is_alternating_three_word_hook_pair(prev_tokens, cur_tokens):
        return False
    if _soft_token_overlap_ratio(cur_tokens, next_tokens) > max_next_overlap_ratio:
        return False
    baseline_duration = base.end_time - base.start_time
    mapped_duration = mapped.end_time - mapped.start_time
    if baseline_duration < min_baseline_duration_sec or mapped_duration <= 0.0:
        return False
    if mapped_duration >= baseline_duration * 0.85:
        return False
    return abs(mapped.start_time - base.start_time) <= max_start_delta_sec


def _compute_alternating_middle_hook_target_window(
    *,
    base: models.Line,
    mapped: models.Line,
    next_line: models.Line,
    phrase_window: tuple[float, float] | None,
    max_early_pull_sec: float,
    min_end_gain_sec: float,
) -> tuple[float, float] | None:
    if phrase_window is None:
        return None
    window_start, window_end = phrase_window
    if window_end <= mapped.end_time + min_end_gain_sec:
        return None
    if window_start >= mapped.start_time:
        return None
    target_start = max(mapped.start_time, base.start_time - max_early_pull_sec)
    baseline_duration = base.end_time - base.start_time
    target_end = max(target_start + baseline_duration, window_end)
    target_end = min(target_end, next_line.start_time - 0.05)
    if target_end <= mapped.end_time + min_end_gain_sec:
        return None
    return target_start, target_end


def _restore_alternating_middle_hook_from_phrase_window(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    max_early_pull_sec: float = 0.4,
    min_baseline_duration_sec: float = 3.0,
    max_start_delta_sec: float = 0.6,
    min_end_gain_sec: float = 0.6,
    max_next_overlap_ratio: float = 0.34,
) -> tuple[List[models.Line], int]:
    repaired = list(mapped_lines)
    restored = 0
    limit = min(len(mapped_lines), len(baseline_lines))
    for idx in range(1, limit - 1):
        prev_base = baseline_lines[idx - 1]
        base = baseline_lines[idx]
        mapped = repaired[idx]
        next_line = repaired[idx + 1]
        if not _alternating_middle_hook_has_shape(
            prev_base=prev_base,
            base=base,
            mapped=mapped,
            next_line=next_line,
            min_baseline_duration_sec=min_baseline_duration_sec,
            max_start_delta_sec=max_start_delta_sec,
            max_next_overlap_ratio=max_next_overlap_ratio,
        ):
            continue
        target_window = _compute_alternating_middle_hook_target_window(
            base=base,
            mapped=mapped,
            next_line=next_line,
            phrase_window=_find_exact_phrase_window(
                whisper_words,
                _normalized_line_tokens(base),
            ),
            max_early_pull_sec=max_early_pull_sec,
            min_end_gain_sec=min_end_gain_sec,
        )
        if target_window is None:
            continue
        target_start, target_end = target_window
        repaired[idx] = _retime_line_to_window(
            mapped,
            window_start=target_start,
            window_end=target_end,
        )
        restored += 1
    return repaired, restored


def _compact_exact_phrase_late_start_is_eligible(
    *,
    base: models.Line,
    mapped: models.Line,
    next_line: models.Line,
    min_duration_sec: float,
) -> bool:
    if not base.words or not mapped.words or not next_line.words:
        return False
    if len(mapped.words) < 3 or len(mapped.words) > 4:
        return False
    if len(_normalized_line_tokens(base)) != len(mapped.words):
        return False
    mapped_duration = mapped.end_time - mapped.start_time
    return mapped_duration >= min_duration_sec


def _compact_exact_phrase_late_start_target_window(
    *,
    base: models.Line,
    mapped: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    min_start_gain_sec: float,
    max_start_gain_sec: float,
    max_baseline_anchor_delta_sec: float,
    max_end_delta_sec: float,
) -> tuple[float, float] | None:
    phrase_window = _find_exact_phrase_window(
        whisper_words,
        _normalized_line_tokens(base),
    )
    if phrase_window is None:
        return None
    phrase_start, phrase_end = phrase_window
    start_gain = phrase_start - mapped.start_time
    if start_gain < min_start_gain_sec or start_gain > max_start_gain_sec:
        return None
    if abs(mapped.start_time - base.start_time) > max_baseline_anchor_delta_sec:
        return None
    if abs(phrase_end - mapped.end_time) > max_end_delta_sec:
        return None
    if phrase_end > next_line.start_time - 0.05:
        return None
    return phrase_start, phrase_end


def _restore_compact_exact_phrase_late_starts(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_start_gain_sec: float = 0.25,
    max_start_gain_sec: float = 0.65,
    max_baseline_anchor_delta_sec: float = 0.6,
    max_end_delta_sec: float = 0.2,
    min_duration_sec: float = 2.4,
) -> tuple[List[models.Line], int]:
    repaired = list(mapped_lines)
    restored = 0
    limit = min(len(mapped_lines), len(baseline_lines))
    for idx in range(limit - 1):
        base = baseline_lines[idx]
        mapped = repaired[idx]
        next_line = repaired[idx + 1]
        if not _compact_exact_phrase_late_start_is_eligible(
            base=base,
            mapped=mapped,
            next_line=next_line,
            min_duration_sec=min_duration_sec,
        ):
            continue
        target_window = _compact_exact_phrase_late_start_target_window(
            base=base,
            mapped=mapped,
            next_line=next_line,
            whisper_words=whisper_words,
            min_start_gain_sec=min_start_gain_sec,
            max_start_gain_sec=max_start_gain_sec,
            max_baseline_anchor_delta_sec=max_baseline_anchor_delta_sec,
            max_end_delta_sec=max_end_delta_sec,
        )
        if target_window is None:
            continue
        phrase_start, phrase_end = target_window
        repaired[idx] = _retime_line_to_window(
            mapped,
            window_start=phrase_start,
            window_end=phrase_end,
        )
        restored += 1
    return repaired, restored


def _apply_baseline_constraint_and_snap(
    *,
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    corrections: List[str],
    constrain_line_starts_to_baseline_fn: Callable[..., Any],
    snap_first_word_to_whisper_onset_fn: Callable[..., Any],
    trace_snapshots: list[dict[str, Any]],
    trace_line_range: Any,
    apply_baseline_constraint: bool,
    snap_first_word_max_shift: float,
) -> tuple[List[models.Line], List[str]]:
    if apply_baseline_constraint:
        mapped_lines = constrain_line_starts_to_baseline_fn(
            mapped_lines, baseline_lines
        )
        _capture_stage_lines(
            mapped_lines=mapped_lines,
            trace_snapshots=trace_snapshots,
            trace_line_range=trace_line_range,
            stage="after_initial_baseline_constraint",
        )
    else:
        corrections.append(
            "Skipped baseline start constraint due to strong global Whisper "
            "shift evidence"
        )

    try:
        mapped_lines = snap_first_word_to_whisper_onset_fn(
            mapped_lines,
            all_words,
            max_shift=snap_first_word_max_shift,
        )
    except TypeError:
        mapped_lines = snap_first_word_to_whisper_onset_fn(mapped_lines, all_words)
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_snap_first_word_to_whisper_onset",
    )
    if apply_baseline_constraint:
        mapped_lines = constrain_line_starts_to_baseline_fn(
            mapped_lines, baseline_lines
        )
        _capture_stage_lines(
            mapped_lines=mapped_lines,
            trace_snapshots=trace_snapshots,
            trace_line_range=trace_line_range,
            stage="after_second_baseline_constraint",
        )
    return mapped_lines, corrections


def _apply_baseline_restore_shift_passes(
    *,
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    corrections: List[str],
    trace_snapshots: list[dict[str, Any]],
    trace_line_range: Any,
    apply_baseline_constraint: bool,
) -> tuple[List[models.Line], List[str]]:
    mapped_lines, restored_weak = _restore_weak_evidence_large_start_shifts(
        mapped_lines,
        baseline_lines,
        all_words,
    )
    _append_correction_if_any(
        corrections,
        restored_weak,
        "Restored {count} weak-evidence large start shift line(s) to baseline",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_restore_weak_evidence_large_start_shifts",
    )

    mapped_lines, restored_alternating_hooks = (
        _restore_alternating_middle_hook_from_phrase_window(
            mapped_lines,
            baseline_lines,
            all_words,
        )
    )
    _append_correction_if_any(
        corrections,
        restored_alternating_hooks,
        "Restored {count} alternating middle hook line(s) from phrase windows",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_restore_alternating_middle_hook_from_phrase_window",
    )

    mapped_lines, restored_adjacent_late = _restore_adjacent_near_threshold_late_shifts(
        mapped_lines,
        baseline_lines,
        all_words,
    )
    _append_correction_if_any(
        corrections,
        restored_adjacent_late,
        "Restored {count} adjacent near-threshold late line(s) to baseline",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_restore_adjacent_near_threshold_late_shifts",
    )

    mapped_lines, restored_compact_exact_phrase = (
        _restore_compact_exact_phrase_late_starts(
            mapped_lines,
            baseline_lines,
            all_words,
        )
    )
    _append_correction_if_any(
        corrections,
        restored_compact_exact_phrase,
        "Restored {count} compact exact-phrase late start line(s)",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_restore_compact_exact_phrase_late_starts",
    )

    if not apply_baseline_constraint:
        mapped_lines, restored_late_runs = (
            _restore_consistently_late_runs_from_baseline(
                mapped_lines,
                baseline_lines,
            )
        )
        _append_correction_if_any(
            corrections,
            restored_late_runs,
            "Restored {count} consistently late line(s) from baseline timing",
        )
        _capture_stage_lines(
            mapped_lines=mapped_lines,
            trace_snapshots=trace_snapshots,
            trace_line_range=trace_line_range,
            stage="after_restore_consistently_late_runs_from_baseline",
        )
    return mapped_lines, corrections


def _apply_baseline_restore_cleanup_passes(
    *,
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    corrections: List[str],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    trace_snapshots: list[dict[str, Any]],
    trace_line_range: Any,
) -> tuple[List[models.Line], List[str]]:
    mapped_lines, restored_early_duplicates = (
        _restore_unsupported_early_duplicate_shifts(
            mapped_lines,
            baseline_lines,
            all_words,
        )
    )
    _append_correction_if_any(
        corrections,
        restored_early_duplicates,
        "Restored {count} unsupported early duplicate line(s) to baseline",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_restore_unsupported_early_duplicate_shifts",
    )

    mapped_lines, restored_short = restore_implausibly_short_lines_fn(
        baseline_lines, mapped_lines
    )
    _append_correction_if_any(
        corrections,
        restored_short,
        "Restored {count} short compressed lines from baseline timing",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_restore_implausibly_short_lines",
    )

    mapped_lines, restored_inversions = _restore_pairwise_inversions_from_source(
        baseline_lines,
        mapped_lines,
        min_inversion_gap=0.25,
        min_ahead_shift=2.5,
    )
    _append_correction_if_any(
        corrections,
        restored_inversions,
        "Restored {count} inversion outlier line(s) from baseline timing",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_restore_pairwise_inversions_from_source",
    )

    mapped_lines, restored_zero_support_late = (
        _restore_zero_support_parenthetical_late_start_expansions(
            mapped_lines,
            baseline_lines,
            all_words,
        )
    )
    _append_correction_if_any(
        corrections,
        restored_zero_support_late,
        "Restored {count} zero-support parenthetical late expanded "
        "line(s) to baseline starts",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_restore_zero_support_parenthetical_late_start_expansions",
    )

    mapped_lines, restored_enumerations = _restore_late_enumeration_lines_from_baseline(
        mapped_lines,
        baseline_lines,
    )
    _append_correction_if_any(
        corrections,
        restored_enumerations,
        "Restored {count} late enumeration line(s) from baseline timing",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_restore_late_enumeration_lines_from_baseline",
    )
    return mapped_lines, corrections


def _apply_audio_reanchor_corrections(
    *,
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    audio_features: timing_models.AudioFeatures,
    corrections: List[str],
    trace_snapshots: list[dict[str, Any]],
    trace_line_range: Any,
    runtime_config: WhisperRuntimeConfig,
) -> tuple[List[models.Line], List[str]]:
    mapped_lines, carryover_fixes = _shift_weak_opening_lines_past_phrase_carryover(
        mapped_lines,
        audio_features,
        all_words,
    )
    _append_correction_if_any(
        corrections,
        carryover_fixes,
        "Shifted {count} weak-opening line(s) past prior-phrase carryover",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_shift_weak_opening_lines_past_phrase_carryover",
    )

    repeated_cadence_reanchors = 0
    if runtime_config.repeat_cadence_reanchor:
        mapped_lines, repeated_cadence_reanchors = _reanchor_repeated_cadence_lines(
            mapped_lines
        )
    _append_correction_if_any(
        corrections,
        repeated_cadence_reanchors,
        "Reanchored {count} repeated-cadence line(s)",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_reanchor_repeated_cadence_lines",
    )

    restored_run_onset_shifts = 0
    if runtime_config.restored_run_onset_shift:
        mapped_lines, restored_run_onset_shifts = (
            _shift_restored_low_support_runs_to_onset(
                mapped_lines,
                baseline_lines,
                all_words,
                audio_features,
            )
        )
    _append_correction_if_any(
        corrections,
        restored_run_onset_shifts,
        "Shifted {count} restored low-support line(s) to nearby onset runs",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_shift_restored_low_support_runs_to_onset",
    )

    mapped_lines, late_compact_tail_reanchors = _reanchor_late_compact_tail(
        mapped_lines,
        baseline_lines,
        all_words,
        audio_features,
    )
    _append_correction_if_any(
        corrections,
        late_compact_tail_reanchors,
        "Reanchored {count} late compact repetitive tail line(s) to later audio onsets",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_reanchor_late_compact_repetitive_tail_lines_to_later_onsets",
    )

    mapped_lines, earlier_whisper_reanchors = (
        _reanchor_late_supported_lines_to_earlier_whisper(
            mapped_lines,
            all_words,
        )
    )
    _append_correction_if_any(
        corrections,
        earlier_whisper_reanchors,
        "Reanchored {count} late supported line(s) to earlier Whisper starts",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_reanchor_late_supported_lines_to_earlier_whisper",
    )

    mapped_lines, light_leading_reanchors = (
        _reanchor_light_leading_lines_to_content_words(
            mapped_lines,
            all_words,
        )
    )
    _append_correction_if_any(
        corrections,
        light_leading_reanchors,
        "Reanchored {count} light-leading line(s) to local content words",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_reanchor_light_leading_lines_to_content_words",
    )

    mapped_lines, short_followup_rebalances = (
        _rebalance_short_followup_boundaries_from_whisper(
            mapped_lines,
            all_words,
        )
    )
    _append_correction_if_any(
        corrections,
        short_followup_rebalances,
        "Rebalanced {count} short followup line boundary/boundaries "
        "from Whisper support",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_rebalance_short_followup_boundaries_from_whisper",
    )

    mapped_lines, truncated_followup_reanchors = (
        _reanchor_truncated_followup_lines_from_phonetic_variants(
            mapped_lines,
            all_words,
        )
    )
    _append_correction_if_any(
        corrections,
        truncated_followup_reanchors,
        "Reanchored {count} truncated followup line(s) from phonetic Whisper variants",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_reanchor_truncated_followup_lines_from_phonetic_variants",
    )

    mapped_lines, said_reanchors = _reanchor_unsupported_i_said_lines_to_later_onset(
        mapped_lines,
        baseline_lines,
        all_words,
        audio_features,
    )
    _append_correction_if_any(
        corrections,
        said_reanchors,
        "Reanchored {count} unsupported 'I said' line(s) to later audio onsets",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_reanchor_unsupported_i_said_lines_to_later_onset",
    )

    low_support_reanchors = 0
    if runtime_config.low_support_onset_reanchor:
        mapped_lines, low_support_reanchors = (
            _reanchor_low_support_lines_to_later_onset(
                mapped_lines,
                baseline_lines,
                all_words,
                audio_features,
            )
        )
    _append_correction_if_any(
        corrections,
        low_support_reanchors,
        "Reanchored {count} low-support line(s) to later audio onsets",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_reanchor_low_support_lines_to_later_onset",
    )

    mapped_lines, interjection_reanchors = (
        _reanchor_unsupported_interjection_lines_to_onsets(
            mapped_lines,
            all_words,
            audio_features,
        )
    )
    _append_correction_if_any(
        corrections,
        interjection_reanchors,
        "Reanchored {count} unsupported interjection line(s) to audio onsets",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_reanchor_unsupported_interjection_lines_to_onsets",
    )
    return mapped_lines, corrections


def _apply_audio_extension_corrections(
    *,
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    audio_features: timing_models.AudioFeatures,
    corrections: List[str],
    trace_snapshots: list[dict[str, Any]],
    trace_line_range: Any,
) -> tuple[List[models.Line], List[str]]:
    _ = audio_features

    mapped_lines, tail_extensions = _extend_unsupported_parenthetical_tails(
        mapped_lines,
        all_words,
    )
    _append_correction_if_any(
        corrections,
        tail_extensions,
        "Extended {count} unsupported parenthetical tail(s)",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_extend_unsupported_parenthetical_tails",
    )

    mapped_lines, i_said_tail_extensions = _extend_unsupported_i_said_tails(
        mapped_lines,
        all_words,
    )
    _append_correction_if_any(
        corrections,
        i_said_tail_extensions,
        "Extended {count} unsupported 'I said' tail(s)",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_extend_unsupported_i_said_tails",
    )

    mapped_lines, pre_i_said_extensions = _extend_misaligned_lines_before_i_said(
        mapped_lines,
        all_words,
    )
    _append_correction_if_any(
        corrections,
        pre_i_said_extensions,
        "Extended {count} lexically mismatched line(s) before "
        "unsupported 'I said' lines",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_extend_misaligned_lines_before_i_said",
    )

    mapped_lines, pre_weak_opening_extensions = (
        _extend_unsupported_long_lines_before_weak_opening(
            mapped_lines,
            all_words,
        )
    )
    _append_correction_if_any(
        corrections,
        pre_weak_opening_extensions,
        "Extended {count} unsupported line(s) before weak openings",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_extend_unsupported_long_lines_before_weak_opening",
    )

    mapped_lines, weak_opening_extensions = _extend_unsupported_weak_opening_lines(
        mapped_lines,
        all_words,
    )
    _append_correction_if_any(
        corrections,
        weak_opening_extensions,
        "Extended {count} unsupported weak-opening line(s)",
    )
    _capture_stage_lines(
        mapped_lines=mapped_lines,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        stage="after_extend_unsupported_weak_opening_lines",
    )
    return mapped_lines, corrections


def _apply_baseline_restore_corrections(
    *,
    lines: List[models.Line],
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    corrections: List[str],
    constrain_line_starts_to_baseline_fn: Callable[..., Any],
    snap_first_word_to_whisper_onset_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    trace_snapshots: list[dict[str, Any]],
    trace_line_range: Any,
    matched_ratio: float,
    line_coverage: float,
    config,
    should_apply_baseline_constraint_fn: Callable[..., Any],
) -> tuple[List[models.Line], List[str], float, float, float, bool]:
    _ = lines
    whisper_end = max((w.end for w in all_words), default=0.0)
    baseline_end = _line_set_end(baseline_lines)
    mapped_end = _line_set_end(mapped_lines)
    baseline_timeline_ratio = baseline_end / whisper_end if whisper_end > 0.0 else 1.0
    mapped_timeline_ratio = mapped_end / whisper_end if whisper_end > 0.0 else 1.0
    apply_baseline_constraint, median_global_shift = (
        should_apply_baseline_constraint_fn(
            mapped_lines,
            baseline_lines,
            matched_ratio=matched_ratio,
            line_coverage=line_coverage,
        )
    )
    mapped_lines, corrections = _apply_baseline_constraint_and_snap(
        mapped_lines=mapped_lines,
        baseline_lines=baseline_lines,
        all_words=all_words,
        corrections=corrections,
        constrain_line_starts_to_baseline_fn=constrain_line_starts_to_baseline_fn,
        snap_first_word_to_whisper_onset_fn=snap_first_word_to_whisper_onset_fn,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        apply_baseline_constraint=apply_baseline_constraint,
        snap_first_word_max_shift=config.snap_first_word_max_shift,
    )
    mapped_lines, corrections = _apply_baseline_restore_shift_passes(
        mapped_lines=mapped_lines,
        baseline_lines=baseline_lines,
        all_words=all_words,
        corrections=corrections,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        apply_baseline_constraint=apply_baseline_constraint,
    )
    mapped_lines, corrections = _apply_baseline_restore_cleanup_passes(
        mapped_lines=mapped_lines,
        baseline_lines=baseline_lines,
        all_words=all_words,
        corrections=corrections,
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
    )
    return (
        mapped_lines,
        corrections,
        baseline_timeline_ratio,
        mapped_timeline_ratio,
        median_global_shift,
        apply_baseline_constraint,
    )


def _apply_audio_alignment_corrections(
    *,
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    audio_features: Optional[timing_models.AudioFeatures],
    corrections: List[str],
    trace_snapshots: list[dict[str, Any]],
    trace_line_range: Any,
    runtime_config: WhisperRuntimeConfig,
) -> tuple[List[models.Line], List[str]]:
    if audio_features is None:
        return mapped_lines, corrections

    mapped_lines, corrections = _apply_audio_reanchor_corrections(
        mapped_lines=mapped_lines,
        baseline_lines=baseline_lines,
        all_words=all_words,
        audio_features=audio_features,
        corrections=corrections,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        runtime_config=runtime_config,
    )
    return _apply_audio_extension_corrections(
        mapped_lines=mapped_lines,
        all_words=all_words,
        audio_features=audio_features,
        corrections=corrections,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
    )
