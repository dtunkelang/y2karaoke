"""Finalization/post-pass orchestration helpers for Whisper integration."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from ... import models
from ..alignment import timing_models


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
) -> Tuple[List[models.Line], List[str]]:
    aligned_lines, merged_first = merge_first_two_lines_if_segment_matches_fn(
        aligned_lines, transcription, epitran_lang
    )
    if merged_first:
        alignments.append("Merged first two lines via Whisper segment")
    aligned_lines, pair_retimed = retime_adjacent_lines_to_whisper_window_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pair_retimed:
        alignments.append(
            f"Retimed {pair_retimed} adjacent line pair(s) to Whisper window"
        )
    aligned_lines, pair_windowed = retime_adjacent_lines_to_segment_window_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pair_windowed:
        alignments.append(
            f"Retimed {pair_windowed} adjacent line pair(s) to Whisper segment window"
        )
    aligned_lines, pulled_next = pull_next_line_into_segment_window_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pulled_next:
        alignments.append(f"Pulled {pulled_next} line(s) into adjacent segment window")
    aligned_lines, pulled_near_end = pull_lines_near_segment_end_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pulled_near_end:
        alignments.append(f"Pulled {pulled_near_end} line(s) near segment ends")
    aligned_lines, pulled_same = pull_next_line_into_same_segment_fn(
        aligned_lines, transcription
    )
    if pulled_same:
        alignments.append(f"Pulled {pulled_same} line(s) into same segment")
    aligned_lines, pair_retimed_after = retime_adjacent_lines_to_whisper_window_fn(
        aligned_lines,
        transcription,
        epitran_lang,
        max_window_duration=4.5,
        max_start_offset=1.0,
    )
    if pair_retimed_after:
        alignments.append(
            f"Retimed {pair_retimed_after} adjacent line pair(s) after pulls"
        )
    aligned_lines, merged = merge_lines_to_whisper_segments_fn(
        aligned_lines, transcription, epitran_lang
    )
    if merged:
        alignments.append(f"Merged {merged} line pair(s) via Whisper segments")
    aligned_lines, tightened = tighten_lines_to_whisper_segments_fn(
        aligned_lines, transcription, epitran_lang
    )
    if tightened:
        alignments.append(f"Tightened {tightened} line(s) to Whisper segments")
    aligned_lines, pulled = pull_lines_to_best_segments_fn(
        aligned_lines, transcription, epitran_lang
    )
    if pulled:
        alignments.append(f"Pulled {pulled} line(s) to Whisper segments")
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
) -> Tuple[List[models.Line], List[str]]:
    aligned_lines, restored_inversions = _restore_pairwise_inversions_from_source(
        source_lines,
        aligned_lines,
    )
    if restored_inversions:
        alignments.append(
            f"Restored {restored_inversions} inversion outlier line(s) from source timing"
        )
    aligned_lines, alignments = fix_ordering_violations_fn(
        source_lines, aligned_lines, alignments
    )
    aligned_lines = normalize_line_word_timings_fn(aligned_lines)
    aligned_lines = enforce_monotonic_line_starts_fn(aligned_lines)
    aligned_lines = enforce_non_overlapping_lines_fn(aligned_lines)

    if force_dtw:
        aligned_lines, pulled_near_end = pull_lines_near_segment_end_fn(
            aligned_lines, transcription, epitran_lang
        )
        if pulled_near_end:
            alignments.append(
                f"Pulled {pulled_near_end} line(s) near segment ends (post-order)"
            )
        aligned_lines, merged_short = merge_short_following_line_into_segment_fn(
            aligned_lines, transcription
        )
        if merged_short:
            alignments.append(
                f"Merged {merged_short} short line(s) into prior segments"
            )
        aligned_lines, clamped_repeat = clamp_repeated_line_duration_fn(aligned_lines)
        if clamped_repeat:
            alignments.append(f"Clamped {clamped_repeat} repeated line(s) duration")

    aligned_lines, deduped = drop_duplicate_lines_fn(
        aligned_lines, transcription, epitran_lang
    )
    if deduped:
        alignments.append(f"Dropped {deduped} duplicate line(s)")
    before_drop = len(aligned_lines)
    aligned_lines = [line for line in aligned_lines if line.words]
    if len(aligned_lines) != before_drop:
        alignments.append("Dropped empty lines after Whisper merges")
    aligned_lines, timing_deduped = drop_duplicate_lines_by_timing_fn(aligned_lines)
    if timing_deduped:
        alignments.append(
            f"Dropped {timing_deduped} duplicate line(s) by timing overlap"
        )

    if audio_features is not None:
        aligned_lines, continuous_fixes = pull_lines_forward_for_continuous_vocals_fn(
            aligned_lines, audio_features
        )
        if continuous_fixes:
            alignments.append(
                f"Pulled {continuous_fixes} line(s) forward for continuous vocals"
            )

    aligned_lines = enforce_monotonic_line_starts_fn(aligned_lines)
    aligned_lines = enforce_non_overlapping_lines_fn(aligned_lines)
    return aligned_lines, alignments
