"""Finalization/post-pass orchestration helpers for Whisper integration."""

from __future__ import annotations

import re
import time
from typing import Any, Callable, List, Optional, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_split_refrain_restore import (
    restore_split_short_refrains_to_matching_segments as _restore_split_short_refrains_to_matching_segments_impl,
)

_FINALIZE_TOKEN_RE = re.compile(r"[^a-z0-9\s]")


def _clone_line(line: models.Line) -> models.Line:
    return models.Line(
        words=[
            models.Word(
                text=w.text,
                start_time=w.start_time,
                end_time=w.end_time,
                singer=w.singer,
            )
            for w in line.words
        ],
        singer=line.singer,
    )


def _normalize_finalize_text(text: str) -> str:
    return _FINALIZE_TOKEN_RE.sub("", text.lower()).strip()


def _line_text_token_overlap(a: str, b: str) -> float:
    def _tokens(text: str) -> set[str]:
        out = set()
        for raw in text.lower().split():
            token = "".join(ch for ch in raw if ch.isalpha())
            if token:
                out.add(token)
        return out

    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _text_divergence_stats(
    source_lines: List[models.Line],
    aligned_lines: List[models.Line],
    *,
    min_overlap: float = 0.45,
) -> Tuple[int, int]:
    compared = 0
    diverged = 0
    limit = min(len(source_lines), len(aligned_lines))
    for idx in range(limit):
        src = source_lines[idx]
        dst = aligned_lines[idx]
        if not src.words or not dst.words:
            continue
        compared += 1
        if _line_text_token_overlap(src.text, dst.text) < min_overlap:
            diverged += 1
    return diverged, compared


def _restore_pairwise_inversions_from_source(
    source_lines: List[models.Line],
    aligned_lines: List[models.Line],
    *,
    min_inversion_gap: float = 1.0,
    min_ahead_shift: float = 8.0,
    source_order_tolerance: float = 0.1,
) -> Tuple[List[models.Line], int]:
    """Repair local pairwise inversions by restoring obviously over-shifted lines."""
    if len(aligned_lines) < 2 or len(source_lines) < 2:
        return aligned_lines, 0

    repaired = list(aligned_lines)
    restored = 0
    limit = min(len(source_lines), len(aligned_lines)) - 1
    for idx in range(limit):
        curr = repaired[idx]
        nxt = repaired[idx + 1]
        src_curr = source_lines[idx]
        src_next = source_lines[idx + 1]
        if not curr.words or not nxt.words or not src_curr.words or not src_next.words:
            continue
        inversion = curr.start_time > nxt.start_time + min_inversion_gap
        source_ordered = (
            src_curr.start_time <= src_next.start_time + source_order_tolerance
        )
        shifted_far_ahead = (curr.start_time - src_curr.start_time) >= min_ahead_shift
        if inversion and source_ordered and shifted_far_ahead:
            repaired[idx] = _clone_line(src_curr)
            restored += 1
    return repaired, restored


def _restore_repeated_compact_runs_from_source(
    source_lines: List[models.Line],
    aligned_lines: List[models.Line],
    *,
    max_source_duration: float = 2.5,
    min_late_shift: float = 1.5,
    min_pair_overlap: float = 0.6,
    min_run_length: int = 2,
) -> Tuple[List[models.Line], int]:
    if len(aligned_lines) < min_run_length or len(source_lines) < min_run_length:
        return aligned_lines, 0

    repaired = list(aligned_lines)
    restored = 0
    limit = min(len(source_lines), len(aligned_lines))
    idx = 0
    while idx < limit:
        run_end = _find_repeated_compact_run_end(
            source_lines=source_lines,
            repaired_lines=repaired,
            start_idx=idx,
            limit=limit,
            max_source_duration=max_source_duration,
            min_late_shift=min_late_shift,
            min_pair_overlap=min_pair_overlap,
        )
        if run_end - idx < min_run_length:
            idx += 1
            continue

        for restore_idx in range(idx, run_end):
            repaired[restore_idx] = _clone_line(source_lines[restore_idx])
            restored += 1
        idx = run_end

    return repaired, restored


def _find_repeated_compact_run_end(
    *,
    source_lines: List[models.Line],
    repaired_lines: List[models.Line],
    start_idx: int,
    limit: int,
    max_source_duration: float,
    min_late_shift: float,
    min_pair_overlap: float,
) -> int:
    src = source_lines[start_idx]
    dst = repaired_lines[start_idx]
    if not src.words or not dst.words:
        return start_idx
    src_duration = src.end_time - src.start_time
    if src_duration > max_source_duration:
        return start_idx
    if dst.start_time - src.start_time < min_late_shift:
        return start_idx

    run_end = start_idx + 1
    while run_end < limit:
        prev_src = source_lines[run_end - 1]
        cur_src = source_lines[run_end]
        cur_dst = repaired_lines[run_end]
        if not prev_src.words or not cur_src.words or not cur_dst.words:
            break
        cur_duration = cur_src.end_time - cur_src.start_time
        if cur_duration > max_source_duration:
            break
        if cur_dst.start_time - cur_src.start_time < min_late_shift:
            break
        if _line_text_token_overlap(prev_src.text, cur_src.text) < min_pair_overlap:
            break
        run_end += 1
    return run_end


def _restore_split_short_refrains_to_matching_segments(
    aligned_lines: List[models.Line],
    transcription: List[timing_models.TranscriptionSegment],
    *,
    min_gap: float = 0.05,
    max_words: int = 4,
    min_late_shift: float = 0.8,
    max_late_shift: float = 3.0,
) -> Tuple[List[models.Line], int]:
    return _restore_split_short_refrains_to_matching_segments_impl(
        aligned_lines,
        transcription,
        min_gap=min_gap,
        max_words=max_words,
        min_late_shift=min_late_shift,
        max_late_shift=max_late_shift,
    )


def _apply_low_quality_segment_postpasses(
    *,
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    merge_first_two_lines_if_segment_matches_fn: Callable[..., Any],
    retime_adjacent_lines_to_whisper_window_fn: Callable[..., Any],
    retime_adjacent_lines_to_segment_window_fn: Callable[..., Any],
    pull_next_line_into_segment_window_fn: Callable[..., Any],
    pull_lines_near_segment_end_fn: Callable[..., Any],
    pull_next_line_into_same_segment_fn: Callable[..., Any],
    merge_lines_to_whisper_segments_fn: Callable[..., Any],
    tighten_lines_to_whisper_segments_fn: Callable[..., Any],
    pull_lines_to_best_segments_fn: Callable[..., Any],
    stage_metrics: Optional[dict[str, float]] = None,
) -> Tuple[List[models.Line], List[str]]:
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=merge_first_two_lines_if_segment_matches_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Merged first two lines via Whisper segment",
        metric_key="postpass_merge_first_two_lines",
        stage_metrics=stage_metrics,
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=retime_adjacent_lines_to_whisper_window_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Retimed {count} adjacent line pair(s) to Whisper window",
        metric_key="postpass_retime_adjacent_whisper_window",
        stage_metrics=stage_metrics,
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=retime_adjacent_lines_to_segment_window_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Retimed {count} adjacent line pair(s) to Whisper segment window",
        metric_key="postpass_retime_adjacent_segment_window",
        stage_metrics=stage_metrics,
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=pull_next_line_into_segment_window_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Pulled {count} line(s) into adjacent segment window",
        metric_key="postpass_pull_next_into_segment_window",
        stage_metrics=stage_metrics,
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=pull_lines_near_segment_end_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Pulled {count} line(s) near segment ends",
        metric_key="postpass_pull_near_segment_end",
        stage_metrics=stage_metrics,
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=pull_next_line_into_same_segment_fn,
        postpass_args=(transcription,),
        message_template="Pulled {count} line(s) into same segment",
        metric_key="postpass_pull_next_into_same_segment",
        stage_metrics=stage_metrics,
    )
    aligned_lines, pair_retimed_after = retime_adjacent_lines_to_whisper_window_fn(
        aligned_lines,
        transcription,
        epitran_lang,
        max_window_duration=4.5,
        max_start_offset=1.0,
    )
    _append_counted_alignment(
        alignments,
        pair_retimed_after,
        "Retimed {count} adjacent line pair(s) after pulls",
    )
    _record_stage_metric(
        stage_metrics, "postpass_retime_adjacent_after_pulls", pair_retimed_after
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=merge_lines_to_whisper_segments_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Merged {count} line pair(s) via Whisper segments",
        metric_key="postpass_merge_lines_to_segments",
        stage_metrics=stage_metrics,
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=tighten_lines_to_whisper_segments_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Tightened {count} line(s) to Whisper segments",
        metric_key="postpass_tighten_lines_to_segments",
        stage_metrics=stage_metrics,
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=pull_lines_to_best_segments_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Pulled {count} line(s) to Whisper segments",
        metric_key="postpass_pull_lines_to_best_segments",
        stage_metrics=stage_metrics,
    )
    return aligned_lines, alignments


def _finalize_whisper_line_set(
    *,
    source_lines: List[models.Line],
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    force_dtw: bool,
    audio_features: Optional[timing_models.AudioFeatures],
    fix_ordering_violations_fn: Callable[..., Any],
    normalize_line_word_timings_fn: Callable[..., Any],
    enforce_monotonic_line_starts_fn: Callable[..., Any],
    enforce_non_overlapping_lines_fn: Callable[..., Any],
    pull_lines_near_segment_end_fn: Callable[..., Any],
    merge_short_following_line_into_segment_fn: Callable[..., Any],
    clamp_repeated_line_duration_fn: Callable[..., Any],
    drop_duplicate_lines_fn: Callable[..., Any],
    drop_duplicate_lines_by_timing_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    stage_metrics: Optional[dict[str, float]] = None,
) -> Tuple[List[models.Line], List[str]]:
    aligned_lines, restored_inversions = _restore_pairwise_inversions_from_source(
        source_lines,
        aligned_lines,
    )
    if restored_inversions:
        alignments.append(
            f"Restored {restored_inversions} inversion outlier line(s) from source timing"
        )
    _record_stage_metric(
        stage_metrics, "finalize_restored_inversions_from_source", restored_inversions
    )
    aligned_lines, restored_repeated_runs = _restore_repeated_compact_runs_from_source(
        source_lines,
        aligned_lines,
    )
    if restored_repeated_runs:
        alignments.append(
            f"Restored {restored_repeated_runs} repeated compact line(s) from source timing"
        )
    _record_stage_metric(
        stage_metrics,
        "finalize_restored_repeated_compact_runs_from_source",
        restored_repeated_runs,
    )
    aligned_lines, alignments = fix_ordering_violations_fn(
        source_lines, aligned_lines, alignments
    )
    aligned_lines = normalize_line_word_timings_fn(aligned_lines)
    aligned_lines = enforce_monotonic_line_starts_fn(aligned_lines)
    aligned_lines = enforce_non_overlapping_lines_fn(aligned_lines)

    if force_dtw:
        aligned_lines, alignments = _apply_force_dtw_finalize_passes(
            aligned_lines=aligned_lines,
            alignments=alignments,
            transcription=transcription,
            epitran_lang=epitran_lang,
            pull_lines_near_segment_end_fn=pull_lines_near_segment_end_fn,
            merge_short_following_line_into_segment_fn=merge_short_following_line_into_segment_fn,
            clamp_repeated_line_duration_fn=clamp_repeated_line_duration_fn,
            stage_metrics=stage_metrics,
        )

    aligned_lines, alignments = _apply_dedup_finalize_passes(
        aligned_lines=aligned_lines,
        alignments=alignments,
        transcription=transcription,
        epitran_lang=epitran_lang,
        drop_duplicate_lines_fn=drop_duplicate_lines_fn,
        drop_duplicate_lines_by_timing_fn=drop_duplicate_lines_by_timing_fn,
        stage_metrics=stage_metrics,
    )

    if audio_features is not None:
        aligned_lines, alignments = _apply_continuous_vocal_finalize_pass(
            aligned_lines=aligned_lines,
            alignments=alignments,
            audio_features=audio_features,
            pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
            stage_metrics=stage_metrics,
        )

    aligned_lines, restored_split_refrains = (
        _restore_split_short_refrains_to_matching_segments(
            aligned_lines,
            transcription,
        )
    )
    if restored_split_refrains:
        alignments.append(
            f"Restored {restored_split_refrains} split short refrain line(s) to matching Whisper segments"
        )
    _record_stage_metric(
        stage_metrics,
        "finalize_restored_split_short_refrains_to_matching_segments",
        restored_split_refrains,
    )

    aligned_lines = enforce_monotonic_line_starts_fn(aligned_lines)
    aligned_lines = enforce_non_overlapping_lines_fn(aligned_lines)

    diverged_count, compared_count = _text_divergence_stats(source_lines, aligned_lines)
    if compared_count >= 8 and diverged_count >= max(3, int(compared_count * 0.25)):
        alignments.append(
            "Rolled back Whisper timing due to high per-line text divergence from source lyrics"
        )
        _record_stage_metric(stage_metrics, "finalize_text_divergence_rollback", 1)
        return [_clone_line(line) for line in source_lines], alignments
    _record_stage_metric(stage_metrics, "finalize_text_divergence_rollback", 0)
    return aligned_lines, alignments


def _append_counted_alignment(alignments: List[str], count: int, template: str) -> None:
    if count:
        alignments.append(template.format(count=count))


def _run_counted_postpass(
    *,
    aligned_lines: List[models.Line],
    alignments: List[str],
    postpass_fn: Callable[..., Any],
    postpass_args: tuple[Any, ...],
    message_template: str,
    metric_key: str,
    stage_metrics: Optional[dict[str, float]] = None,
) -> Tuple[List[models.Line], List[str]]:
    start = time.perf_counter()
    aligned_lines, count = postpass_fn(aligned_lines, *postpass_args)
    elapsed = time.perf_counter() - start
    _append_counted_alignment(alignments, count, message_template)
    _record_stage_metric(stage_metrics, metric_key, count)
    _record_stage_metric(stage_metrics, f"{metric_key}_sec", elapsed)
    return aligned_lines, alignments


def _record_stage_metric(
    stage_metrics: Optional[dict[str, float]],
    metric_key: str,
    value: float,
) -> None:
    if stage_metrics is None:
        return
    stage_metrics[metric_key] = float(value)


def _apply_force_dtw_finalize_passes(
    *,
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    pull_lines_near_segment_end_fn: Callable[..., Any],
    merge_short_following_line_into_segment_fn: Callable[..., Any],
    clamp_repeated_line_duration_fn: Callable[..., Any],
    stage_metrics: Optional[dict[str, float]] = None,
) -> Tuple[List[models.Line], List[str]]:
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=pull_lines_near_segment_end_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Pulled {count} line(s) near segment ends (post-order)",
        metric_key="finalize_force_dtw_pull_near_end",
        stage_metrics=stage_metrics,
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=merge_short_following_line_into_segment_fn,
        postpass_args=(transcription,),
        message_template="Merged {count} short line(s) into prior segments",
        metric_key="finalize_force_dtw_merge_short_following",
        stage_metrics=stage_metrics,
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=clamp_repeated_line_duration_fn,
        postpass_args=(),
        message_template="Clamped {count} repeated line(s) duration",
        metric_key="finalize_force_dtw_clamp_repeated",
        stage_metrics=stage_metrics,
    )
    return aligned_lines, alignments


def _apply_dedup_finalize_passes(
    *,
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    drop_duplicate_lines_fn: Callable[..., Any],
    drop_duplicate_lines_by_timing_fn: Callable[..., Any],
    stage_metrics: Optional[dict[str, float]] = None,
) -> Tuple[List[models.Line], List[str]]:
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=drop_duplicate_lines_fn,
        postpass_args=(transcription, epitran_lang),
        message_template="Dropped {count} duplicate line(s)",
        metric_key="finalize_drop_duplicate_lines",
        stage_metrics=stage_metrics,
    )
    before_drop = len(aligned_lines)
    aligned_lines = [line for line in aligned_lines if line.words]
    if len(aligned_lines) != before_drop:
        alignments.append("Dropped empty lines after Whisper merges")
    _record_stage_metric(
        stage_metrics,
        "finalize_drop_empty_lines_after_merge",
        before_drop - len(aligned_lines),
    )
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=drop_duplicate_lines_by_timing_fn,
        postpass_args=(),
        message_template="Dropped {count} duplicate line(s) by timing overlap",
        metric_key="finalize_drop_duplicate_lines_by_timing",
        stage_metrics=stage_metrics,
    )
    return aligned_lines, alignments


def _apply_continuous_vocal_finalize_pass(
    *,
    aligned_lines: List[models.Line],
    alignments: List[str],
    audio_features: timing_models.AudioFeatures,
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    stage_metrics: Optional[dict[str, float]] = None,
) -> Tuple[List[models.Line], List[str]]:
    aligned_lines, alignments = _run_counted_postpass(
        aligned_lines=aligned_lines,
        alignments=alignments,
        postpass_fn=pull_lines_forward_for_continuous_vocals_fn,
        postpass_args=(audio_features,),
        message_template="Pulled {count} line(s) forward for continuous vocals",
        metric_key="finalize_pull_lines_forward_continuous_vocals",
        stage_metrics=stage_metrics,
    )
    return aligned_lines, alignments
