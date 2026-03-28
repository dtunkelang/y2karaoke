"""High-level refinement and hybrid alignment logic."""

import os
import logging
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Set

from ...models import Line
from ..alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
    AudioFeatures,
)
from ...phonetic_utils import _get_ipa
from ... import audio_analysis
from .whisper_alignment_base import (
    _apply_offset_to_line,
    _line_duration,
    _shift_line,
)
from . import whisper_alignment_activity as _alignment_activity_helpers
from . import whisper_alignment_guard_stats as _guard_stats_helpers
from . import whisper_alignment_hybrid as _hybrid_helpers
from . import whisper_alignment_line_helpers as _line_helpers
from . import whisper_alignment_short_lines as _short_line_helpers
from .whisper_alignment_segments import _find_best_whisper_segment

_check_vocal_activity_in_range = audio_analysis._check_vocal_activity_in_range
_check_for_silence_in_range = audio_analysis._check_for_silence_in_range

logger = logging.getLogger(__name__)

_CONTINUOUS_VOCALS_TRACE_CALL_COUNT = 0
_CURRENT_CONTINUOUS_VOCALS_CALL_INDEX = 0
_LAST_LONG_GAP_SHIFTED_INDICES: set[int] = set()


def _maybe_write_long_gap_shift_trace(rows: list[dict[str, Any]]) -> None:
    trace_path = os.environ.get("Y2K_TRACE_LONG_GAP_SHIFT_JSON", "").strip()
    if not trace_path:
        return
    existing_rows: list[dict[str, Any]] = []
    if os.path.exists(trace_path):
        try:
            with open(trace_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            existing_rows = list(payload.get("rows", []))
        except (OSError, json.JSONDecodeError, AttributeError):
            existing_rows = []
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": existing_rows + rows}, fh, indent=2)


def _maybe_write_active_gap_trace(rows: list[dict]) -> None:
    trace_path = os.environ.get("Y2K_TRACE_ACTIVE_GAP_EXTENSION_JSON", "").strip()
    if not trace_path:
        return
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2)


def _capture_continuous_vocals_stage(
    rows: list[dict[str, Any]],
    *,
    call_index: int,
    stage: str,
    lines: List[Line],
) -> None:
    trace_line_range = os.environ.get("Y2K_TRACE_WHISPER_LINE_RANGE", "").strip()
    if not trace_line_range:
        return
    try:
        start_s, end_s = trace_line_range.split("-", 1)
        start_idx = int(start_s)
        end_idx = int(end_s)
    except ValueError:
        return
    rows.append(
        {
            "call_index": call_index,
            "stage": stage,
            "lines": [
                {
                    "line_index": idx,
                    "text": line.text,
                    "start": round(line.start_time, 3),
                    "end": round(line.end_time, 3),
                }
                for idx, line in enumerate(lines, start=1)
                if start_idx <= idx <= end_idx and line.words
            ],
        }
    )


def _maybe_write_continuous_vocals_trace(rows: list[dict[str, Any]]) -> None:
    trace_path = os.environ.get("Y2K_TRACE_CONTINUOUS_VOCALS_JSON", "").strip()
    if not trace_path:
        return
    existing_rows: list[dict[str, Any]] = []
    if os.path.exists(trace_path):
        try:
            with open(trace_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            existing_rows = list(payload.get("rows", []))
        except (OSError, json.JSONDecodeError, AttributeError):
            existing_rows = []
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": existing_rows + rows}, fh, indent=2)


@dataclass(frozen=True)
class _ContinuousVocalsRefinementConfig:
    enable_silence_short_line_refinement: bool
    enable_shift_long_activity_gaps: bool
    enable_extend_active_gaps: bool


def _default_continuous_vocals_refinement_config() -> _ContinuousVocalsRefinementConfig:
    env_flag = os.getenv("Y2K_WHISPER_SILENCE_REFINEMENT", "1").strip().lower()
    shift_long_gaps = (
        os.getenv("Y2K_WHISPER_CONTINUOUS_SHIFT_LONG_GAPS", "1").strip().lower()
    )
    extend_active_gaps = (
        os.getenv("Y2K_WHISPER_CONTINUOUS_EXTEND_ACTIVE_GAPS", "1").strip().lower()
    )
    return _ContinuousVocalsRefinementConfig(
        enable_silence_short_line_refinement=env_flag
        not in {"0", "false", "off", "no"},
        enable_shift_long_activity_gaps=shift_long_gaps
        not in {"0", "false", "off", "no"},
        enable_extend_active_gaps=extend_active_gaps not in {"0", "false", "off", "no"},
    )


@contextmanager
def use_alignment_refinement_hooks(
    *,
    find_best_whisper_segment_fn=None,
    get_ipa_fn=None,
    check_vocal_activity_in_range_fn=None,
    check_for_silence_in_range_fn=None,
):
    """Temporarily override hybrid-alignment collaborators for tests."""
    global _find_best_whisper_segment, _get_ipa
    global _check_vocal_activity_in_range, _check_for_silence_in_range

    prev_find_best = _find_best_whisper_segment
    prev_get_ipa = _get_ipa
    prev_check_activity = _check_vocal_activity_in_range
    prev_check_silence = _check_for_silence_in_range

    if find_best_whisper_segment_fn is not None:
        _find_best_whisper_segment = find_best_whisper_segment_fn
    if get_ipa_fn is not None:
        _get_ipa = get_ipa_fn
    if check_vocal_activity_in_range_fn is not None:
        _check_vocal_activity_in_range = check_vocal_activity_in_range_fn
    if check_for_silence_in_range_fn is not None:
        _check_for_silence_in_range = check_for_silence_in_range_fn

    try:
        yield
    finally:
        _find_best_whisper_segment = prev_find_best
        _get_ipa = prev_get_ipa
        _check_vocal_activity_in_range = prev_check_activity
        _check_for_silence_in_range = prev_check_silence


def _calculate_drift_correction(
    recent_offsets: List[float], trust_threshold: float
) -> Optional[float]:
    return _hybrid_helpers.calculate_drift_correction(recent_offsets, trust_threshold)


def _interpolate_unmatched_lines(
    mapped_lines: List[Line], matched_lines: Set[int]
) -> List[Line]:
    return _hybrid_helpers.interpolate_unmatched_lines(
        mapped_lines,
        matched_lines,
        line_duration_fn=_line_duration,
        shift_line_fn=_shift_line,
    )


def _refine_unmatched_lines_with_onsets(
    mapped_lines: List[Line],
    matched_lines: Set[int],
    vocals_path: str,
) -> List[Line]:
    from ...refine import refine_word_timing

    return _hybrid_helpers.refine_unmatched_lines_with_onsets(
        mapped_lines,
        matched_lines,
        vocals_path,
        refine_word_timing_fn=refine_word_timing,
        logger=logger,
    )


def align_hybrid_lrc_whisper(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    words: List[TranscriptionWord],
    language: str = "fra-Latn",
    trust_threshold: float = 1.0,
    correct_threshold: float = 1.5,
    min_similarity: float = 0.4,
) -> Tuple[List[Line], List[str]]:
    logger.debug(f"Pre-computing IPA for {len(words)} Whisper words...")
    return _hybrid_helpers.align_hybrid_lrc_whisper(
        lines,
        segments,
        words,
        language=language,
        trust_threshold=trust_threshold,
        correct_threshold=correct_threshold,
        min_similarity=min_similarity,
        get_ipa_fn=_get_ipa,
        find_best_whisper_segment_fn=_find_best_whisper_segment,
        apply_offset_to_line_fn=_apply_offset_to_line,
        calculate_drift_correction_fn=_calculate_drift_correction,
    )


def _fix_ordering_violations(
    original_lines: List[Line],
    aligned_lines: List[Line],
    alignments: List[str],
) -> Tuple[List[Line], List[str]]:
    return _hybrid_helpers.fix_ordering_violations(
        original_lines,
        aligned_lines,
        alignments,
        logger=logger,
    )


def _line_tokens(text: str) -> List[str]:
    return _line_helpers.line_tokens(text)


def _token_overlap(a: str, b: str) -> float:
    return _line_helpers.token_overlap(a, b)


def _shift_line_words(line: Line, shift: float) -> Line:
    return _line_helpers.shift_line_words(line, shift)


def _rebuild_line_with_target_end(line: Line, target_end: float) -> Optional[Line]:
    return _line_helpers.rebuild_line_with_target_end(line, target_end)


def _compact_short_line_if_needed(
    line: Line,
    *,
    max_duration: float,
    next_start: Optional[float] = None,
) -> Line:
    return _line_helpers.compact_short_line_if_needed(
        line,
        max_duration=max_duration,
        next_start=next_start,
    )


def _find_internal_silences(
    line_start: float,
    line_end: float,
    normalized_silences: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    return _line_helpers.find_internal_silences(
        line_start,
        line_end,
        normalized_silences,
    )


def _has_near_start_silence(
    line_start: float, internal_silences: List[Tuple[float, float]]
) -> bool:
    return _line_helpers.has_near_start_silence(line_start, internal_silences)


def _first_onset_after(onset_times, *, start: float, window: float) -> Optional[float]:
    return _line_helpers.first_onset_after(onset_times, start=start, window=window)


def _shift_lines_across_long_activity_gaps(
    lines: List[Line], audio_features: AudioFeatures, max_gap: float, onset_times
) -> int:
    global _LAST_LONG_GAP_SHIFTED_INDICES
    _LAST_LONG_GAP_SHIFTED_INDICES = set()
    fixes = 0
    prev_shift: Optional[float] = None
    trace_rows: list[dict[str, Any]] = []
    for idx in range(1, len(lines)):
        prev_line = lines[idx - 1]
        line = lines[idx]
        row = {
            "call_index": _CURRENT_CONTINUOUS_VOCALS_CALL_INDEX,
            "line_index": idx + 1,
            "prev_text": prev_line.text,
            "text": line.text,
            "prev_end": round(prev_line.end_time, 3),
            "start": round(line.start_time, 3),
            "end": round(line.end_time, 3),
        }
        if not prev_line.words or not line.words:
            row["decision"] = "skip_missing_words"
            trace_rows.append(row)
            prev_shift = None
            continue
        gap = line.start_time - prev_line.end_time
        row["gap"] = round(gap, 3)
        if gap <= max_gap:
            row["decision"] = "skip_gap_within_max"
            trace_rows.append(row)
            prev_shift = None
            continue
        if (
            prev_shift is not None
            and prev_shift <= -2.0
            and _token_overlap(prev_line.text, line.text) < 0.5
        ):
            row["decision"] = "skip_after_nonmatching_large_shift"
            row["prev_shift"] = round(prev_shift, 3)
            trace_rows.append(row)
            prev_shift = None
            continue
        activity = _check_vocal_activity_in_range(
            prev_line.end_time, line.start_time, audio_features
        )
        row["activity"] = round(float(activity), 3)
        has_silence = _check_for_silence_in_range(
            prev_line.end_time,
            line.start_time,
            audio_features,
            min_silence_duration=0.5,
        )
        row["has_silence"] = bool(has_silence)
        if activity <= 0.6 or has_silence:
            row["decision"] = "skip_low_activity_or_silence"
            trace_rows.append(row)
            continue
        candidate_onsets = onset_times[
            (onset_times >= prev_line.end_time) & (onset_times <= line.start_time)
        ]
        row["candidate_onsets"] = [round(float(onset), 3) for onset in candidate_onsets]
        if len(candidate_onsets) == 0:
            row["decision"] = "skip_no_candidate_onsets"
            trace_rows.append(row)
            prev_shift = None
            continue
        new_start = max(float(candidate_onsets[0]), prev_line.end_time + 0.05)
        shift = new_start - line.start_time
        row["chosen_onset"] = round(new_start, 3)
        row["shift"] = round(shift, 3)
        if shift > -0.3:
            row["decision"] = "skip_small_shift"
            trace_rows.append(row)
            prev_shift = None
            continue
        lines[idx] = _shift_line_words(line, shift)
        _LAST_LONG_GAP_SHIFTED_INDICES.add(idx)
        prev_shift = shift
        fixes += 1
        row["decision"] = "shift"
        row["new_start"] = round(lines[idx].start_time, 3)
        row["new_end"] = round(lines[idx].end_time, 3)
        trace_rows.append(row)
    _maybe_write_long_gap_shift_trace(trace_rows)
    return fixes


def _extend_line_ends_across_active_gaps(
    lines: List[Line],
    audio_features: AudioFeatures,
    *,
    min_gap: float = 1.25,
    min_extension: float = 0.25,
    max_extension: float = 2.5,
    activity_threshold: float = 0.65,
    silence_min_duration: float = 0.35,
) -> int:
    """Extend prior line ends when a gap has sustained vocal activity and no silence."""
    fixes = 0
    trace_rows: list[dict] = []
    for idx in range(1, len(lines)):
        prev_line = lines[idx - 1]
        next_line = lines[idx]
        if not prev_line.words or not next_line.words:
            continue
        if (idx - 1) in _LAST_LONG_GAP_SHIFTED_INDICES or (
            idx - 2
        ) in _LAST_LONG_GAP_SHIFTED_INDICES:
            continue
        gap = next_line.start_time - prev_line.end_time
        if gap < min_gap:
            continue
        activity = _check_vocal_activity_in_range(
            prev_line.end_time, next_line.start_time, audio_features
        )
        row = {
            "line_index": idx,
            "prev_text": prev_line.text,
            "next_text": next_line.text,
            "prev_end": round(prev_line.end_time, 3),
            "next_start": round(next_line.start_time, 3),
            "gap": round(gap, 3),
            "activity": round(float(activity), 3),
        }
        if activity < activity_threshold:
            row["decision"] = "skip_low_activity"
            trace_rows.append(row)
            continue
        has_silence = _check_for_silence_in_range(
            prev_line.end_time,
            next_line.start_time,
            audio_features,
            min_silence_duration=silence_min_duration,
        )
        if has_silence:
            row["decision"] = "skip_silence"
            trace_rows.append(row)
            continue

        extension = min(gap - 0.05, max_extension)
        if extension < min_extension:
            row["decision"] = "skip_small_extension"
            trace_rows.append(row)
            continue
        target_end = prev_line.end_time + extension
        target_end = min(target_end, next_line.start_time - 0.05)
        stretched = _rebuild_line_with_target_end(prev_line, target_end)
        if stretched is None:
            row["decision"] = "skip_rebuild_failed"
            trace_rows.append(row)
            continue
        if stretched.end_time <= prev_line.end_time + min_extension:
            row["decision"] = "skip_insufficient_growth"
            trace_rows.append(row)
            continue
        lines[idx - 1] = stretched
        row["decision"] = "extend"
        row["target_end"] = round(target_end, 3)
        row["new_end"] = round(stretched.end_time, 3)
        fixes += 1
        trace_rows.append(row)
    _maybe_write_active_gap_trace(trace_rows)
    return fixes


def _is_uniform_word_timing(line: Line) -> bool:
    return _line_helpers.is_uniform_word_timing(line)


def _retime_line_words_to_onsets(
    line: Line,
    onset_times,
    *,
    min_word_duration: float = 0.08,
) -> Optional[Line]:
    return _line_helpers.retime_line_words_to_onsets(
        line,
        onset_times,
        min_word_duration=min_word_duration,
    )


def _retime_uniform_word_lines_with_onsets(
    lines: List[Line],
    onset_times,
) -> int:
    fixes = 0
    for idx, line in enumerate(lines):
        if not _is_uniform_word_timing(line):
            continue
        retimed = _retime_line_words_to_onsets(line, onset_times)
        if retimed is None:
            continue
        delta = sum(
            abs(retimed.words[i].start_time - line.words[i].start_time)
            for i in range(len(line.words))
        )
        if delta < 0.15:
            continue
        lines[idx] = retimed
        fixes += 1
    return fixes


def _short_line_silence_shift_candidate(
    line: Line,
    normalized_silences: List[Tuple[float, float]],
    onset_times,
) -> Optional[float]:
    return _short_line_helpers._short_line_silence_shift_candidate(
        line,
        normalized_silences,
        onset_times,
        find_internal_silences_fn=_find_internal_silences,
        has_near_start_silence_fn=_has_near_start_silence,
        first_onset_after_fn=_first_onset_after,
    )


def _short_line_run_end(lines: List[Line], start_idx: int) -> int:
    return _short_line_helpers._short_line_run_end(lines, start_idx)


def _shift_short_line_runs_after_silence(
    lines: List[Line], normalized_silences: List[Tuple[float, float]], onset_times
) -> int:
    return _short_line_helpers._shift_short_line_runs_after_silence(
        lines,
        normalized_silences,
        onset_times,
        shift_line_words_fn=_shift_line_words,
        compact_short_line_if_needed_fn=_compact_short_line_if_needed,
        find_internal_silences_fn=_find_internal_silences,
        has_near_start_silence_fn=_has_near_start_silence,
        first_onset_after_fn=_first_onset_after,
    )


def _shift_single_short_lines_after_silence(
    lines: List[Line], normalized_silences: List[Tuple[float, float]], onset_times
) -> int:
    return _short_line_helpers._shift_single_short_lines_after_silence(
        lines,
        normalized_silences,
        onset_times,
        shift_line_words_fn=_shift_line_words,
        compact_short_line_if_needed_fn=_compact_short_line_if_needed,
        find_internal_silences_fn=_find_internal_silences,
        has_near_start_silence_fn=_has_near_start_silence,
        first_onset_after_fn=_first_onset_after,
    )


def _compact_short_lines_near_silence(
    lines: List[Line], normalized_silences: List[Tuple[float, float]]
) -> int:
    return _short_line_helpers._compact_short_lines_near_silence(
        lines,
        normalized_silences,
        compact_short_line_if_needed_fn=_compact_short_line_if_needed,
        find_internal_silences_fn=_find_internal_silences,
        has_near_start_silence_fn=_has_near_start_silence,
    )


def _stretch_similar_adjacent_short_lines(
    lines: List[Line], normalized_silences: List[Tuple[float, float]]
) -> int:
    return _short_line_helpers._stretch_similar_adjacent_short_lines(
        lines,
        normalized_silences,
        token_overlap_fn=_token_overlap,
        rebuild_line_with_target_end_fn=_rebuild_line_with_target_end,
    )


def _cap_isolated_short_lines(lines: List[Line]) -> int:
    return _short_line_helpers._cap_isolated_short_lines(
        lines, rebuild_line_with_target_end_fn=_rebuild_line_with_target_end
    )


def _clone_lines(lines: List[Line]) -> List[Line]:
    return _guard_stats_helpers.clone_lines(lines)


def _long_gap_stats(lines: List[Line], threshold: float = 20.0) -> Tuple[int, float]:
    return _guard_stats_helpers.long_gap_stats(lines, threshold=threshold)


def _ordering_inversion_stats(
    lines: List[Line], tolerance: float = 0.01
) -> Tuple[int, float]:
    return _guard_stats_helpers.ordering_inversion_stats(lines, tolerance=tolerance)


def _pull_lines_forward_for_continuous_vocals(
    lines: List[Line],
    audio_features: Optional[AudioFeatures],
    max_gap: float = 4.0,
    enable_silence_short_line_refinement: bool = True,
) -> Tuple[List[Line], int]:
    """Refine short-line placement from audio continuity/silence cues."""
    if not lines or audio_features is None:
        return lines, 0

    onset_times = audio_features.onset_times
    if onset_times is None or len(onset_times) == 0:
        return lines, 0

    original_lines = _clone_lines(lines)
    before_long_count, before_max_gap = _long_gap_stats(lines)
    before_inv_count, before_inv_drop = _ordering_inversion_stats(lines)
    global _CONTINUOUS_VOCALS_TRACE_CALL_COUNT, _CURRENT_CONTINUOUS_VOCALS_CALL_INDEX
    _CONTINUOUS_VOCALS_TRACE_CALL_COUNT += 1
    call_index = _CONTINUOUS_VOCALS_TRACE_CALL_COUNT
    _CURRENT_CONTINUOUS_VOCALS_CALL_INDEX = call_index
    stage_rows: list[dict[str, Any]] = []
    _capture_continuous_vocals_stage(
        stage_rows, call_index=call_index, stage="before", lines=lines
    )

    config = _default_continuous_vocals_refinement_config()
    fixes = 0
    if config.enable_shift_long_activity_gaps:
        fixes += _shift_lines_across_long_activity_gaps(
            lines, audio_features, max_gap, onset_times
        )
        _capture_continuous_vocals_stage(
            stage_rows,
            call_index=call_index,
            stage="after_shift_long_activity_gaps",
            lines=lines,
        )
    if config.enable_extend_active_gaps:
        fixes += _extend_line_ends_across_active_gaps(lines, audio_features)
        _capture_continuous_vocals_stage(
            stage_rows,
            call_index=call_index,
            stage="after_extend_active_gaps",
            lines=lines,
        )
    if not (
        enable_silence_short_line_refinement
        and config.enable_silence_short_line_refinement
    ):
        _maybe_write_continuous_vocals_trace(stage_rows)
        return lines, fixes

    silence_regions = getattr(audio_features, "silence_regions", None) or []
    normalized_silences = [
        (float(start), float(end))
        for start, end in silence_regions
        if float(end) - float(start) >= 0.8
    ]
    if normalized_silences:
        fixes += _shift_short_line_runs_after_silence(
            lines, normalized_silences, onset_times
        )
        _capture_continuous_vocals_stage(
            stage_rows,
            call_index=call_index,
            stage="after_shift_short_line_runs_after_silence",
            lines=lines,
        )
        fixes += _shift_single_short_lines_after_silence(
            lines, normalized_silences, onset_times
        )
        _capture_continuous_vocals_stage(
            stage_rows,
            call_index=call_index,
            stage="after_shift_single_short_lines_after_silence",
            lines=lines,
        )
        fixes += _compact_short_lines_near_silence(lines, normalized_silences)
        _capture_continuous_vocals_stage(
            stage_rows,
            call_index=call_index,
            stage="after_compact_short_lines_near_silence",
            lines=lines,
        )
        fixes += _stretch_similar_adjacent_short_lines(lines, normalized_silences)
        _capture_continuous_vocals_stage(
            stage_rows,
            call_index=call_index,
            stage="after_stretch_similar_adjacent_short_lines",
            lines=lines,
        )
        fixes += _cap_isolated_short_lines(lines)
        _capture_continuous_vocals_stage(
            stage_rows,
            call_index=call_index,
            stage="after_cap_isolated_short_lines",
            lines=lines,
        )

    after_long_count, after_max_gap = _long_gap_stats(lines)
    after_inv_count, after_inv_drop = _ordering_inversion_stats(lines)
    if after_long_count > before_long_count or after_max_gap > max(
        before_max_gap + 8.0, 20.0
    ):
        logger.debug(
            (
                "Reverting continuous-vocals refinement: long gaps worsened "
                "(%d→%d, %.2fs→%.2fs)"
            ),
            before_long_count,
            after_long_count,
            before_max_gap,
            after_max_gap,
        )
        _maybe_write_continuous_vocals_trace(stage_rows)
        return original_lines, 0
    if after_inv_count > before_inv_count and after_inv_drop > max(
        before_inv_drop + 0.75, 1.5
    ):
        logger.debug(
            (
                "Reverting continuous-vocals refinement: ordering inversions "
                "worsened (%d→%d, %.2fs→%.2fs)"
            ),
            before_inv_count,
            after_inv_count,
            before_inv_drop,
            after_inv_drop,
        )
        _maybe_write_continuous_vocals_trace(stage_rows)
        return original_lines, 0
    _maybe_write_continuous_vocals_trace(stage_rows)
    return lines, fixes


def _fill_vocal_activity_gaps(
    whisper_words: List[TranscriptionWord],
    audio_features: AudioFeatures,
    threshold: float = 0.3,
    min_gap: float = 1.0,
    chunk_duration: float = 0.5,
    segments: Optional[List[TranscriptionSegment]] = None,
) -> Tuple[List[TranscriptionWord], Optional[List[TranscriptionSegment]]]:
    return _alignment_activity_helpers._fill_vocal_activity_gaps(
        whisper_words,
        audio_features,
        check_vocal_activity_in_range_fn=_check_vocal_activity_in_range,
        threshold=threshold,
        min_gap=min_gap,
        chunk_duration=chunk_duration,
        segments=segments,
    )


def _drop_duplicate_lines_by_timing(
    lines: List[Line],
    max_gap: float = 0.2,
) -> Tuple[List[Line], int]:
    return _alignment_activity_helpers._drop_duplicate_lines_by_timing(
        lines, max_gap=max_gap
    )
