"""Correction-pass orchestration for Whisper alignment refinement."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ... import models
from ..alignment import timing_models
from .whisper_integration_align_experimental import (
    reanchor_late_compact_repetitive_tail_lines_to_later_onsets as _reanchor_late_compact_tail,
    reanchor_late_supported_lines_to_earlier_whisper as _reanchor_late_supported_lines_to_earlier_whisper,
    reanchor_light_leading_lines_to_content_words as _reanchor_light_leading_lines_to_content_words,
    reanchor_low_support_lines_to_later_onset as _reanchor_low_support_lines_to_later_onset,
    reanchor_repeated_cadence_lines as _reanchor_repeated_cadence_lines,
    reanchor_truncated_followup_lines_from_phonetic_variants as _reanchor_truncated_followup_lines_from_phonetic_variants,
    rebalance_short_followup_boundaries_from_whisper as _rebalance_short_followup_boundaries_from_whisper,
    reanchor_unsupported_i_said_lines_to_later_onset as _reanchor_unsupported_i_said_lines_to_later_onset,
    shift_restored_low_support_runs_to_onset as _shift_restored_low_support_runs_to_onset,
)
from .whisper_integration_align_heuristics import (
    _extend_misaligned_lines_before_i_said,
    _extend_unsupported_i_said_tails,
    _extend_unsupported_long_lines_before_weak_opening,
    _extend_unsupported_parenthetical_tails,
    _extend_unsupported_weak_opening_lines,
    _line_set_end,
    _reanchor_unsupported_interjection_lines_to_onsets,
    _restore_consistently_late_runs_from_baseline,
    _restore_late_enumeration_lines_from_baseline,
    _restore_zero_support_parenthetical_late_start_expansions,
)
from .whisper_integration_align_trace import (
    capture_trace_snapshot as _capture_trace_snapshot,
)
from .whisper_integration_finalize import _restore_pairwise_inversions_from_source
from .whisper_integration_stages import _shift_weak_opening_lines_past_phrase_carryover
from .whisper_integration_weak_evidence import (
    restore_adjacent_near_threshold_late_shifts as _restore_adjacent_near_threshold_late_shifts,
    restore_weak_evidence_large_start_shifts as _restore_weak_evidence_large_start_shifts,
    restore_unsupported_early_duplicate_shifts as _restore_unsupported_early_duplicate_shifts,
)
from .whisper_runtime_config import WhisperRuntimeConfig


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
            "Skipped baseline start constraint due to strong global Whisper shift evidence"
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
        "Restored {count} zero-support parenthetical late expanded line(s) to baseline starts",
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
        "Rebalanced {count} short followup line boundary/boundaries from Whisper support",
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
        "Extended {count} lexically mismatched line(s) before unsupported 'I said' lines",
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
