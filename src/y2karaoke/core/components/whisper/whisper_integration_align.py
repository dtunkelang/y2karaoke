"""DTW-based LRC-to-Whisper alignment orchestration for integration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import re

from ....utils.lex_lookup_installer import ensure_local_lex_lookup
from ... import models, phonetic_utils
from ..alignment import timing_models
from .whisper_forced_alignment import align_lines_with_whisperx
from .whisper_integration_align_experimental import (
    count_non_vocal_words_near_time as _count_non_vocal_words_near_time,
    local_lexical_overlap_ratio as _local_lexical_overlap_ratio,
    normalized_prefix_tokens as _normalized_prefix_tokens,
    normalized_tokens as _normalized_tokens,
    reanchor_low_support_lines_to_later_onset as _reanchor_low_support_lines_to_later_onset,
    reanchor_repeated_cadence_lines as _reanchor_repeated_cadence_lines,
    reanchor_unsupported_i_said_lines_to_later_onset as _reanchor_unsupported_i_said_lines_to_later_onset,
    rescale_line_to_new_start as _rescale_line_to_new_start,
    shift_restored_low_support_runs_to_onset as _shift_restored_low_support_runs_to_onset,
)
from .whisper_integration_finalize import _restore_pairwise_inversions_from_source
from .whisper_integration_forced_fallback import attempt_whisperx_forced_alignment
from .whisper_integration_align_trace import (
    capture_trace_snapshot as _capture_trace_snapshot,
    maybe_write_trace_snapshot_file as _maybe_write_trace_snapshot_file,
    parse_trace_line_range as _parse_trace_line_range,
)
from .whisper_integration_shift_guard import (
    should_apply_baseline_constraint as _should_apply_baseline_constraint,
)
from .whisper_integration_stages import _shift_weak_opening_lines_past_phrase_carryover
from .whisper_integration_late_run import (
    late_run_is_restorable,
    late_run_shift_for_baseline_restore,
)
from .whisper_integration_weak_evidence import (
    restore_adjacent_near_threshold_late_shifts as _restore_adjacent_near_threshold_late_shifts,
    restore_weak_evidence_large_start_shifts as _restore_weak_evidence_large_start_shifts,
    restore_unsupported_early_duplicate_shifts as _restore_unsupported_early_duplicate_shifts,
)
from .whisper_runtime_config import WhisperRuntimeConfig, load_whisper_runtime_config
from . import whisper_utils

_MIN_FORCED_WORD_COVERAGE = 0.2
_MIN_FORCED_LINE_COVERAGE = 0.2


@dataclass(frozen=True)
class _WhisperMappingDecisionConfig:
    sparse_word_threshold: int = 80
    sparse_segment_threshold: int = 4
    low_coverage_lrc_word_min: int = 20
    low_coverage_matched_ratio_max: float = 0.35
    low_coverage_line_coverage_max: float = 0.35
    snap_first_word_max_shift: float = 2.5


@dataclass
class _PreparedAlignmentInputs:
    baseline_lines: List[models.Line]
    transcription: List[timing_models.TranscriptionSegment]
    all_words: List[timing_models.TranscriptionWord]
    detected_lang: str
    used_model: str
    audio_features: Optional[timing_models.AudioFeatures]
    trace_path: str
    trace_line_range: Any
    trace_snapshots: list[dict[str, Any]]
    before_low_conf_filter: int
    whisper_words_after_filter: int


def _default_mapping_decision_config(
    runtime_config: WhisperRuntimeConfig | None = None,
) -> _WhisperMappingDecisionConfig:
    profile = (runtime_config or load_whisper_runtime_config()).profile
    if profile == "safe":
        return _WhisperMappingDecisionConfig(
            sparse_word_threshold=100,
            sparse_segment_threshold=5,
            low_coverage_lrc_word_min=24,
            low_coverage_matched_ratio_max=0.3,
            low_coverage_line_coverage_max=0.3,
            snap_first_word_max_shift=2.0,
        )
    if profile == "aggressive":
        return _WhisperMappingDecisionConfig(
            sparse_word_threshold=64,
            sparse_segment_threshold=3,
            low_coverage_lrc_word_min=16,
            low_coverage_matched_ratio_max=0.4,
            low_coverage_line_coverage_max=0.4,
            snap_first_word_max_shift=3.0,
        )
    return _WhisperMappingDecisionConfig()


def _line_set_end(lines: List[models.Line]) -> float:
    end_time = 0.0
    for line in lines:
        if line.words:
            end_time = max(end_time, line.end_time)
    return end_time


def _should_ignore_trimmed_transcript(
    *,
    trimmed_end: Optional[float],
    original_transcription: List[timing_models.TranscriptionSegment],
    lines: List[models.Line],
    min_cutoff_gap_sec: float = 10.0,
) -> bool:
    if trimmed_end is None:
        return False
    transcript_end = max(
        (whisper_utils._segment_end(seg) for seg in original_transcription),
        default=trimmed_end,
    )
    lyric_end = _line_set_end(lines)
    return (
        transcript_end - trimmed_end >= min_cutoff_gap_sec
        and lyric_end - trimmed_end >= min_cutoff_gap_sec
    )


def _should_force_whisperx_for_tail_shortfall(
    *,
    all_words: List[timing_models.TranscriptionWord],
    lines: List[models.Line],
    language: str | None,
    runtime_config: WhisperRuntimeConfig,
    detected_lang: str | None = None,
    min_total_words: int = 80,
    min_tail_shortfall_sec: float = 8.0,
    recent_window_sec: float = 20.0,
    max_recent_non_vocal_words: int = 8,
) -> bool:
    if not runtime_config.tail_shortfall_forced_fallback:
        return False
    lang_code = (language or detected_lang or "").split("-", 1)[0].strip().lower()
    if lang_code not in {"fr"}:
        return False
    if len(all_words) < min_total_words:
        return False

    transcript_end = max((word.end for word in all_words), default=0.0)
    lyric_end = _line_set_end(lines)
    if lyric_end - transcript_end < min_tail_shortfall_sec:
        return False

    cutoff = transcript_end - recent_window_sec
    recent_non_vocal_words = sum(
        1
        for word in all_words
        if word.end >= cutoff and word.text.strip().lower() != "[vocal]"
    )
    return recent_non_vocal_words <= max_recent_non_vocal_words


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


def _return_alignment_result_with_trace(
    *,
    result: Tuple[List[models.Line], List[str], Dict[str, float]],
    trace_snapshots: list[dict[str, Any]],
    trace_path: str,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    _maybe_write_trace_snapshot_file(
        snapshots=trace_snapshots,
        trace_path=trace_path,
    )
    return result


def _initialize_alignment_trace() -> tuple[str, Any, list[dict[str, Any]]]:
    return (
        os.environ.get("Y2K_TRACE_WHISPER_STAGES_JSON", "").strip(),
        _parse_trace_line_range(),
        [],
    )


def _attempt_forced_alignment_for_reason(
    *,
    enabled: bool,
    lines: List[models.Line],
    baseline_lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    detected_lang: str,
    logger,
    used_model: str,
    reason: str,
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    trace_snapshots: list[dict[str, Any]],
    trace_path: str,
) -> Optional[Tuple[List[models.Line], List[str], Dict[str, float]]]:
    if not enabled:
        return None
    forced_result = attempt_whisperx_forced_alignment(
        lines=lines,
        baseline_lines=baseline_lines,
        vocals_path=vocals_path,
        language=language,
        detected_lang=detected_lang,
        logger=logger,
        used_model=used_model,
        reason=reason,
        align_lines_with_whisperx_fn=align_lines_with_whisperx,
        should_rollback_short_line_degradation_fn=should_rollback_short_line_degradation_fn,
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
        min_forced_word_coverage=_MIN_FORCED_WORD_COVERAGE,
        min_forced_line_coverage=_MIN_FORCED_LINE_COVERAGE,
    )
    if forced_result is None:
        return None
    return _return_alignment_result_with_trace(
        result=forced_result,
        trace_snapshots=trace_snapshots,
        trace_path=trace_path,
    )


def _transcribe_and_trim_alignment_inputs(
    *,
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    audio_features: Optional[timing_models.AudioFeatures],
    transcribe_vocals_fn: Callable[..., Tuple[Any, Any, str, str]],
    extract_audio_features_fn: Callable[..., Optional[timing_models.AudioFeatures]],
    dedupe_whisper_segments_fn: Callable[..., Any],
    trim_whisper_transcription_by_lyrics_fn: Callable[..., Any],
    logger,
) -> tuple[
    List[timing_models.TranscriptionSegment],
    List[timing_models.TranscriptionWord],
    str,
    str,
    Optional[timing_models.AudioFeatures],
]:
    transcription, all_words, detected_lang, used_model = transcribe_vocals_fn(
        vocals_path, language, model_size, aggressive, temperature
    )
    if not audio_features:
        audio_features = extract_audio_features_fn(vocals_path)
    transcription = dedupe_whisper_segments_fn(transcription)
    original_transcription = transcription
    original_all_words = all_words

    line_texts = [line.text for line in lines if line.text.strip()]
    transcription, all_words, trimmed_end = trim_whisper_transcription_by_lyrics_fn(
        transcription, all_words, line_texts
    )
    if _should_ignore_trimmed_transcript(
        trimmed_end=trimmed_end,
        original_transcription=original_transcription,
        lines=lines,
    ):
        transcription = original_transcription
        all_words = original_all_words
        trimmed_end = None
    if trimmed_end:
        logger.info(
            "Truncated Whisper transcript to %.2f s (last matching lyric).", trimmed_end
        )
    return transcription, all_words, detected_lang, used_model, audio_features


def _finalize_prepared_alignment_inputs(
    *,
    baseline_lines: List[models.Line],
    transcription: List[timing_models.TranscriptionSegment],
    all_words: List[timing_models.TranscriptionWord],
    detected_lang: str,
    used_model: str,
    audio_features: Optional[timing_models.AudioFeatures],
    lenient_vocal_activity_threshold: float,
    low_word_confidence_threshold: float,
    fill_vocal_activity_gaps_fn: Callable[..., Any],
    filter_low_confidence_whisper_words_fn: Callable[..., Any],
    dedupe_whisper_words_fn: Callable[..., Any],
    trace_path: str,
    trace_line_range: Any,
    trace_snapshots: list[dict[str, Any]],
    logger,
) -> tuple[
    Optional[_PreparedAlignmentInputs],
    Optional[Tuple[List[models.Line], List[str], Dict[str, float]]],
]:
    if not transcription or not all_words:
        logger.warning("No transcription available, skipping Whisper timing map")
        return None, _return_alignment_result_with_trace(
            result=(baseline_lines, [], {}),
            trace_snapshots=trace_snapshots,
            trace_path=trace_path,
        )

    if audio_features:
        all_words, filled_segments = fill_vocal_activity_gaps_fn(
            all_words,
            audio_features,
            lenient_vocal_activity_threshold,
            segments=transcription,
        )
        if filled_segments is not None:
            transcription = filled_segments

    before_low_conf_filter = len(all_words)
    all_words = filter_low_confidence_whisper_words_fn(
        all_words,
        low_word_confidence_threshold,
    )
    if len(all_words) != before_low_conf_filter:
        logger.debug(
            "Filtered low-confidence Whisper words: %d -> %d (threshold=%.2f)",
            before_low_conf_filter,
            len(all_words),
            low_word_confidence_threshold,
        )
    all_words = dedupe_whisper_words_fn(all_words)

    return (
        _PreparedAlignmentInputs(
            baseline_lines=baseline_lines,
            transcription=transcription,
            all_words=all_words,
            detected_lang=detected_lang,
            used_model=used_model,
            audio_features=audio_features,
            trace_path=trace_path,
            trace_line_range=trace_line_range,
            trace_snapshots=trace_snapshots,
            before_low_conf_filter=before_low_conf_filter,
            whisper_words_after_filter=len(all_words),
        ),
        None,
    )


def _prepare_alignment_inputs(
    *,
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    audio_features: Optional[timing_models.AudioFeatures],
    lenient_vocal_activity_threshold: float,
    low_word_confidence_threshold: float,
    transcribe_vocals_fn: Callable[..., Tuple[Any, Any, str, str]],
    extract_audio_features_fn: Callable[..., Optional[timing_models.AudioFeatures]],
    dedupe_whisper_segments_fn: Callable[..., Any],
    trim_whisper_transcription_by_lyrics_fn: Callable[..., Any],
    fill_vocal_activity_gaps_fn: Callable[..., Any],
    dedupe_whisper_words_fn: Callable[..., Any],
    clone_lines_for_fallback_fn: Callable[..., Any],
    filter_low_confidence_whisper_words_fn: Callable[..., Any],
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    config: _WhisperMappingDecisionConfig,
    runtime_config: WhisperRuntimeConfig,
    logger,
) -> tuple[
    Optional[_PreparedAlignmentInputs],
    Optional[Tuple[List[models.Line], List[str], Dict[str, float]]],
]:
    baseline_lines = clone_lines_for_fallback_fn(lines)
    ensure_local_lex_lookup()
    (
        transcription,
        all_words,
        detected_lang,
        used_model,
        audio_features,
    ) = _transcribe_and_trim_alignment_inputs(
        lines=lines,
        vocals_path=vocals_path,
        language=language,
        model_size=model_size,
        aggressive=aggressive,
        temperature=temperature,
        audio_features=audio_features,
        transcribe_vocals_fn=transcribe_vocals_fn,
        extract_audio_features_fn=extract_audio_features_fn,
        dedupe_whisper_segments_fn=dedupe_whisper_segments_fn,
        trim_whisper_transcription_by_lyrics_fn=trim_whisper_transcription_by_lyrics_fn,
        logger=logger,
    )
    trace_path, trace_line_range, trace_snapshots = _initialize_alignment_trace()

    forced_result = _attempt_forced_alignment_for_reason(
        enabled=_should_force_whisperx_for_tail_shortfall(
            all_words=all_words,
            lines=lines,
            language=language,
            runtime_config=runtime_config,
            detected_lang=detected_lang,
        ),
        lines=lines,
        baseline_lines=baseline_lines,
        vocals_path=vocals_path,
        language=language,
        detected_lang=detected_lang,
        logger=logger,
        used_model=used_model,
        reason="early Whisper transcript tail shortfall",
        should_rollback_short_line_degradation_fn=should_rollback_short_line_degradation_fn,
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
        trace_snapshots=trace_snapshots,
        trace_path=trace_path,
    )
    if forced_result is not None:
        return None, forced_result

    sparse_whisper_output = (
        len(all_words) < config.sparse_word_threshold
        or len(transcription) <= config.sparse_segment_threshold
    )
    forced_result = _attempt_forced_alignment_for_reason(
        enabled=sparse_whisper_output,
        lines=lines,
        baseline_lines=baseline_lines,
        vocals_path=vocals_path,
        language=language,
        detected_lang=detected_lang,
        logger=logger,
        used_model=used_model,
        reason="sparse Whisper transcript",
        should_rollback_short_line_degradation_fn=should_rollback_short_line_degradation_fn,
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
        trace_snapshots=trace_snapshots,
        trace_path=trace_path,
    )
    if forced_result is not None:
        return None, forced_result

    return _finalize_prepared_alignment_inputs(
        baseline_lines=baseline_lines,
        transcription=transcription,
        all_words=all_words,
        detected_lang=detected_lang,
        used_model=used_model,
        audio_features=audio_features,
        lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
        low_word_confidence_threshold=low_word_confidence_threshold,
        fill_vocal_activity_gaps_fn=fill_vocal_activity_gaps_fn,
        filter_low_confidence_whisper_words_fn=filter_low_confidence_whisper_words_fn,
        dedupe_whisper_words_fn=dedupe_whisper_words_fn,
        trace_path=trace_path,
        trace_line_range=trace_line_range,
        trace_snapshots=trace_snapshots,
        logger=logger,
    )


def _build_lrc_assignments(
    *,
    lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    transcription: List[timing_models.TranscriptionSegment],
    extract_lrc_words_all_fn: Callable[..., Any],
    build_phoneme_tokens_from_lrc_words_fn: Callable[..., Any],
    build_phoneme_tokens_from_whisper_words_fn: Callable[..., Any],
    build_syllable_tokens_from_phonemes_fn: Callable[..., Any],
    build_segment_text_overlap_assignments_fn: Callable[..., Any],
    build_phoneme_dtw_path_fn: Callable[..., Any],
    build_word_assignments_from_phoneme_path_fn: Callable[..., Any],
    build_block_segmented_syllable_assignments_fn: Callable[..., Any],
    min_segment_overlap_coverage: float,
    epitran_lang: str,
    logger,
) -> tuple[List[Any], Any]:
    lrc_words = extract_lrc_words_all_fn(lines)
    if not lrc_words:
        return [], {}

    logger.debug(
        "DTW-phonetic: Pre-computing IPA for %d Whisper words...", len(all_words)
    )
    phonetic_utils._prewarm_ipa_cache(
        [ww.text for ww in all_words] + [lw["text"] for lw in lrc_words],
        epitran_lang,
    )

    logger.debug(
        "DTW-phonetic: Preparing phoneme sequences for %d lyrics words and %d Whisper words...",
        len(lrc_words),
        len(all_words),
    )
    lrc_phonemes = build_phoneme_tokens_from_lrc_words_fn(lrc_words, epitran_lang)
    whisper_phonemes = build_phoneme_tokens_from_whisper_words_fn(
        all_words, epitran_lang
    )
    lrc_syllables = build_syllable_tokens_from_phonemes_fn(lrc_phonemes)
    whisper_syllables = build_syllable_tokens_from_phonemes_fn(whisper_phonemes)

    lrc_assignments = build_segment_text_overlap_assignments_fn(
        lrc_words,
        all_words,
        transcription,
    )
    seg_coverage = len(lrc_assignments) / len(lrc_words) if lrc_words else 0
    if seg_coverage >= min_segment_overlap_coverage:
        return lrc_words, lrc_assignments

    logger.debug(
        "Segment overlap coverage %.0f%% below %.0f%% threshold, falling back to DTW",
        seg_coverage * 100,
        min_segment_overlap_coverage * 100,
    )
    if not lrc_syllables or not whisper_syllables:
        if not lrc_phonemes or not whisper_phonemes:
            logger.warning("No phoneme/syllable data; skipping mapping")
            return lrc_words, {}
        path = build_phoneme_dtw_path_fn(
            lrc_phonemes,
            whisper_phonemes,
            epitran_lang,
        )
        return lrc_words, build_word_assignments_from_phoneme_path_fn(
            path, lrc_phonemes, whisper_phonemes
        )

    return lrc_words, build_block_segmented_syllable_assignments_fn(
        lrc_words,
        all_words,
        lrc_syllables,
        whisper_syllables,
        epitran_lang,
    )


def _finalize_alignment_result(
    *,
    baseline_lines: List[models.Line],
    mapped_lines: List[models.Line],
    corrections: List[str],
    metrics: Dict[str, Any],
    mapped_count: int,
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    logger,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    if mapped_count:
        corrections.append(f"DTW-phonetic mapped {mapped_count} word(s) to Whisper")
    rollback, short_before, short_after = should_rollback_short_line_degradation_fn(
        baseline_lines, mapped_lines
    )
    if not rollback:
        return mapped_lines, corrections, metrics

    repaired_lines, restored_count = restore_implausibly_short_lines_fn(
        baseline_lines, mapped_lines
    )
    repaired_rollback, _, repaired_after = should_rollback_short_line_degradation_fn(
        baseline_lines, repaired_lines
    )
    if restored_count > 0 and not repaired_rollback:
        logger.info(
            "Recovered Whisper map by restoring %d short baseline line(s) (%d -> %d)",
            restored_count,
            short_after,
            repaired_after,
        )
        corrections.append(
            f"Restored {restored_count} short compressed lines from baseline timing"
        )
        return repaired_lines, corrections, metrics

    logger.warning(
        "Rolling back Whisper map: implausibly short multi-word lines worsened (%d -> %d)",
        short_before,
        short_after,
    )
    corrections.append(
        "Ignored Whisper timing map due to short-line compression artifacts"
    )
    return baseline_lines, corrections, metrics


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
    config: _WhisperMappingDecisionConfig,
) -> tuple[List[models.Line], List[str], float, float, float, bool]:
    whisper_end = max((w.end for w in all_words), default=0.0)
    baseline_end = _line_set_end(baseline_lines)
    mapped_end = _line_set_end(mapped_lines)
    baseline_timeline_ratio = baseline_end / whisper_end if whisper_end > 0.0 else 1.0
    mapped_timeline_ratio = mapped_end / whisper_end if whisper_end > 0.0 else 1.0
    apply_baseline_constraint, median_global_shift = _should_apply_baseline_constraint(
        mapped_lines,
        baseline_lines,
        matched_ratio=matched_ratio,
        line_coverage=line_coverage,
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


def _run_initial_mapping_and_postpasses(
    *,
    lines: List[models.Line],
    lrc_words: List[Any],
    all_words: List[timing_models.TranscriptionWord],
    lrc_assignments: Any,
    epitran_lang: str,
    transcription: List[timing_models.TranscriptionSegment],
    audio_features: Optional[timing_models.AudioFeatures],
    vocals_path: str,
    corrections: List[str],
    map_lrc_words_to_whisper_fn: Callable[..., Any],
    run_mapped_line_postpasses_fn: Callable[..., Any],
    interpolate_unmatched_lines_fn: Callable[..., Any],
    refine_unmatched_lines_with_onsets_fn: Callable[..., Any],
    shift_repeated_lines_to_next_whisper_fn: Callable[..., Any],
    extend_line_to_trailing_whisper_matches_fn: Callable[..., Any],
    pull_late_lines_to_matching_segments_fn: Callable[..., Any],
    retime_short_interjection_lines_fn: Callable[..., Any],
    snap_first_word_to_whisper_onset_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any],
    resolve_line_overlaps_fn: Callable[..., Any],
    trace_snapshots: list[dict[str, Any]],
    trace_line_range: Any,
) -> tuple[List[models.Line], int, float, set[int], List[str], float, float]:
    mapping_start = time.perf_counter()
    mapped_lines, mapped_count, total_similarity, mapped_lines_set = (
        map_lrc_words_to_whisper_fn(
            lines,
            lrc_words,
            all_words,
            lrc_assignments,
            epitran_lang,
            transcription,
        )
    )
    _capture_trace_snapshot(
        trace_snapshots,
        stage="after_map_lrc_words_to_whisper",
        lines=mapped_lines,
        line_range=trace_line_range,
    )
    mapping_elapsed = time.perf_counter() - mapping_start

    postpass_start = time.perf_counter()
    mapped_lines, corrections = run_mapped_line_postpasses_fn(
        mapped_lines=mapped_lines,
        mapped_lines_set=mapped_lines_set,
        all_words=all_words,
        transcription=transcription,
        audio_features=audio_features,
        vocals_path=vocals_path,
        epitran_lang=epitran_lang,
        corrections=corrections,
        interpolate_unmatched_lines_fn=interpolate_unmatched_lines_fn,
        refine_unmatched_lines_with_onsets_fn=refine_unmatched_lines_with_onsets_fn,
        shift_repeated_lines_to_next_whisper_fn=shift_repeated_lines_to_next_whisper_fn,
        extend_line_to_trailing_whisper_matches_fn=extend_line_to_trailing_whisper_matches_fn,
        pull_late_lines_to_matching_segments_fn=pull_late_lines_to_matching_segments_fn,
        retime_short_interjection_lines_fn=retime_short_interjection_lines_fn,
        snap_first_word_to_whisper_onset_fn=snap_first_word_to_whisper_onset_fn,
        pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
        capture_trace_snapshot_fn=_capture_trace_snapshot,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
    )
    _capture_trace_snapshot(
        trace_snapshots,
        stage="after_run_mapped_line_postpasses",
        lines=mapped_lines,
        line_range=trace_line_range,
    )
    postpass_elapsed = time.perf_counter() - postpass_start
    return (
        mapped_lines,
        mapped_count,
        total_similarity,
        mapped_lines_set,
        corrections,
        mapping_elapsed,
        postpass_elapsed,
    )


def _maybe_force_low_coverage_alignment(
    *,
    lines: List[models.Line],
    lrc_words: List[Any],
    matched_ratio: float,
    line_coverage: float,
    baseline_lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    detected_lang: str,
    used_model: str,
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    trace_snapshots: list[dict[str, Any]],
    trace_path: str,
    config: _WhisperMappingDecisionConfig,
    logger,
) -> Optional[Tuple[List[models.Line], List[str], Dict[str, float]]]:
    if len(lrc_words) < config.low_coverage_lrc_word_min:
        return None
    if (
        matched_ratio >= config.low_coverage_matched_ratio_max
        and line_coverage >= config.low_coverage_line_coverage_max
    ):
        return None
    forced_result = attempt_whisperx_forced_alignment(
        lines=lines,
        baseline_lines=baseline_lines,
        vocals_path=vocals_path,
        language=language,
        detected_lang=detected_lang,
        logger=logger,
        used_model=used_model,
        reason="low DTW mapping coverage",
        align_lines_with_whisperx_fn=align_lines_with_whisperx,
        should_rollback_short_line_degradation_fn=should_rollback_short_line_degradation_fn,
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
        min_forced_word_coverage=_MIN_FORCED_WORD_COVERAGE,
        min_forced_line_coverage=_MIN_FORCED_LINE_COVERAGE,
    )
    if forced_result is None:
        return None
    return _return_alignment_result_with_trace(
        result=forced_result,
        trace_snapshots=trace_snapshots,
        trace_path=trace_path,
    )


def align_lrc_text_to_whisper_timings_impl(  # noqa: C901
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    min_similarity: float,
    audio_features: Optional[timing_models.AudioFeatures],
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    *,
    transcribe_vocals_fn: Callable[..., Tuple[Any, Any, str, str]],
    extract_audio_features_fn: Callable[..., Optional[timing_models.AudioFeatures]],
    dedupe_whisper_segments_fn: Callable[..., Any],
    trim_whisper_transcription_by_lyrics_fn: Callable[..., Any],
    fill_vocal_activity_gaps_fn: Callable[..., Any],
    extract_lrc_words_all_fn: Callable[..., Any],
    build_phoneme_tokens_from_lrc_words_fn: Callable[..., Any],
    build_phoneme_tokens_from_whisper_words_fn: Callable[..., Any],
    build_syllable_tokens_from_phonemes_fn: Callable[..., Any],
    build_segment_text_overlap_assignments_fn: Callable[..., Any],
    build_phoneme_dtw_path_fn: Callable[..., Any],
    build_word_assignments_from_phoneme_path_fn: Callable[..., Any],
    build_block_segmented_syllable_assignments_fn: Callable[..., Any],
    map_lrc_words_to_whisper_fn: Callable[..., Any],
    dedupe_whisper_words_fn: Callable[..., Any],
    interpolate_unmatched_lines_fn: Callable[..., Any],
    refine_unmatched_lines_with_onsets_fn: Callable[..., Any],
    shift_repeated_lines_to_next_whisper_fn: Callable[..., Any],
    extend_line_to_trailing_whisper_matches_fn: Callable[..., Any],
    pull_late_lines_to_matching_segments_fn: Callable[..., Any],
    retime_short_interjection_lines_fn: Callable[..., Any],
    snap_first_word_to_whisper_onset_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any],
    resolve_line_overlaps_fn: Callable[..., Any],
    run_mapped_line_postpasses_fn: Callable[..., Any],
    constrain_line_starts_to_baseline_fn: Callable[..., Any],
    should_rollback_short_line_degradation_fn: Callable[..., Any],
    restore_implausibly_short_lines_fn: Callable[..., Any],
    clone_lines_for_fallback_fn: Callable[..., Any],
    filter_low_confidence_whisper_words_fn: Callable[..., Any],
    min_segment_overlap_coverage: float,
    runtime_config: WhisperRuntimeConfig | None = None,
    logger,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Align LRC text to Whisper timings via DTW-style phonetic assignment."""
    resolved_runtime_config = runtime_config or load_whisper_runtime_config()
    config = _default_mapping_decision_config(resolved_runtime_config)
    _ = min_similarity  # reserved for potential future tuning hooks
    _ = lenient_activity_bonus  # consumed by downstream scoring in related paths

    overall_start = time.perf_counter()
    prepared, early_result = _prepare_alignment_inputs(
        lines=lines,
        vocals_path=vocals_path,
        language=language,
        model_size=model_size,
        aggressive=aggressive,
        temperature=temperature,
        audio_features=audio_features,
        lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
        low_word_confidence_threshold=low_word_confidence_threshold,
        transcribe_vocals_fn=transcribe_vocals_fn,
        extract_audio_features_fn=extract_audio_features_fn,
        dedupe_whisper_segments_fn=dedupe_whisper_segments_fn,
        trim_whisper_transcription_by_lyrics_fn=trim_whisper_transcription_by_lyrics_fn,
        fill_vocal_activity_gaps_fn=fill_vocal_activity_gaps_fn,
        dedupe_whisper_words_fn=dedupe_whisper_words_fn,
        clone_lines_for_fallback_fn=clone_lines_for_fallback_fn,
        filter_low_confidence_whisper_words_fn=filter_low_confidence_whisper_words_fn,
        should_rollback_short_line_degradation_fn=should_rollback_short_line_degradation_fn,
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
        config=config,
        runtime_config=resolved_runtime_config,
        logger=logger,
    )
    if early_result is not None:
        return early_result

    assert prepared is not None
    baseline_lines = prepared.baseline_lines
    transcription = prepared.transcription
    all_words = prepared.all_words
    detected_lang = prepared.detected_lang
    used_model = prepared.used_model
    audio_features = prepared.audio_features
    trace_path = prepared.trace_path
    trace_line_range = prepared.trace_line_range
    trace_snapshots = prepared.trace_snapshots
    before_low_conf_filter = prepared.before_low_conf_filter
    whisper_words_after_filter = prepared.whisper_words_after_filter

    epitran_lang = phonetic_utils._whisper_lang_to_epitran(detected_lang)
    logger.debug(
        "Using epitran language: %s (from Whisper: %s)", epitran_lang, detected_lang
    )

    lrc_words, lrc_assignments = _build_lrc_assignments(
        lines=lines,
        all_words=all_words,
        transcription=transcription,
        extract_lrc_words_all_fn=extract_lrc_words_all_fn,
        build_phoneme_tokens_from_lrc_words_fn=build_phoneme_tokens_from_lrc_words_fn,
        build_phoneme_tokens_from_whisper_words_fn=build_phoneme_tokens_from_whisper_words_fn,
        build_syllable_tokens_from_phonemes_fn=build_syllable_tokens_from_phonemes_fn,
        build_segment_text_overlap_assignments_fn=build_segment_text_overlap_assignments_fn,
        build_phoneme_dtw_path_fn=build_phoneme_dtw_path_fn,
        build_word_assignments_from_phoneme_path_fn=build_word_assignments_from_phoneme_path_fn,
        build_block_segmented_syllable_assignments_fn=build_block_segmented_syllable_assignments_fn,
        min_segment_overlap_coverage=min_segment_overlap_coverage,
        epitran_lang=epitran_lang,
        logger=logger,
    )
    if not lrc_words:
        return lines, [], {}

    corrections: List[str] = []
    (
        mapped_lines,
        mapped_count,
        total_similarity,
        mapped_lines_set,
        corrections,
        mapping_elapsed,
        postpass_elapsed,
    ) = _run_initial_mapping_and_postpasses(
        lines=lines,
        lrc_words=lrc_words,
        all_words=all_words,
        lrc_assignments=lrc_assignments,
        epitran_lang=epitran_lang,
        transcription=transcription,
        audio_features=audio_features,
        vocals_path=vocals_path,
        corrections=corrections,
        map_lrc_words_to_whisper_fn=map_lrc_words_to_whisper_fn,
        run_mapped_line_postpasses_fn=run_mapped_line_postpasses_fn,
        interpolate_unmatched_lines_fn=interpolate_unmatched_lines_fn,
        refine_unmatched_lines_with_onsets_fn=refine_unmatched_lines_with_onsets_fn,
        shift_repeated_lines_to_next_whisper_fn=shift_repeated_lines_to_next_whisper_fn,
        extend_line_to_trailing_whisper_matches_fn=extend_line_to_trailing_whisper_matches_fn,
        pull_late_lines_to_matching_segments_fn=pull_late_lines_to_matching_segments_fn,
        retime_short_interjection_lines_fn=retime_short_interjection_lines_fn,
        snap_first_word_to_whisper_onset_fn=snap_first_word_to_whisper_onset_fn,
        pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
    )

    matched_ratio = mapped_count / len(lrc_words) if lrc_words else 0.0
    avg_similarity = total_similarity / mapped_count if mapped_count else 0.0
    line_coverage = (
        len(mapped_lines_set) / sum(1 for line in lines if line.words) if lines else 0.0
    )
    forced_result = _maybe_force_low_coverage_alignment(
        lines=lines,
        lrc_words=lrc_words,
        matched_ratio=matched_ratio,
        line_coverage=line_coverage,
        baseline_lines=baseline_lines,
        vocals_path=vocals_path,
        language=language,
        detected_lang=detected_lang,
        used_model=used_model,
        should_rollback_short_line_degradation_fn=should_rollback_short_line_degradation_fn,
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
        trace_snapshots=trace_snapshots,
        trace_path=trace_path,
        config=config,
        logger=logger,
    )
    if forced_result is not None:
        return forced_result

    (
        mapped_lines,
        corrections,
        baseline_timeline_ratio,
        mapped_timeline_ratio,
        median_global_shift,
        apply_baseline_constraint,
    ) = _apply_baseline_restore_corrections(
        lines=lines,
        mapped_lines=mapped_lines,
        baseline_lines=baseline_lines,
        all_words=all_words,
        corrections=corrections,
        constrain_line_starts_to_baseline_fn=constrain_line_starts_to_baseline_fn,
        snap_first_word_to_whisper_onset_fn=snap_first_word_to_whisper_onset_fn,
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        matched_ratio=matched_ratio,
        line_coverage=line_coverage,
        config=config,
    )
    mapped_lines, corrections = _apply_audio_alignment_corrections(
        mapped_lines=mapped_lines,
        baseline_lines=baseline_lines,
        all_words=all_words,
        audio_features=audio_features,
        corrections=corrections,
        trace_snapshots=trace_snapshots,
        trace_line_range=trace_line_range,
        runtime_config=resolved_runtime_config,
    )
    _maybe_write_trace_snapshot_file(
        snapshots=trace_snapshots,
        trace_path=trace_path,
    )
    metrics: Dict[str, Any] = {
        "matched_ratio": matched_ratio,
        "word_coverage": matched_ratio,
        "avg_similarity": avg_similarity,
        "line_coverage": line_coverage,
        "baseline_timeline_ratio": baseline_timeline_ratio,
        "mapped_timeline_ratio": mapped_timeline_ratio,
        "median_global_start_shift_sec": median_global_shift,
        "baseline_constraint_applied": 1.0 if apply_baseline_constraint else 0.0,
        "phonetic_similarity_coverage": matched_ratio * avg_similarity,
        "high_similarity_ratio": avg_similarity,
        "exact_match_ratio": 0.0,
        "unmatched_ratio": 1.0 - matched_ratio,
        "dtw_used": 1.0,
        "dtw_mode": 1.0,
        "whisper_model": used_model,
        "whisper_transcription_segment_count": float(len(transcription)),
        "whisper_word_count_before_filter": float(before_low_conf_filter),
        "whisper_word_count_after_filter": float(whisper_words_after_filter),
        "lrc_word_count": float(len(lrc_words)),
        "mapped_line_count": float(len(mapped_lines_set)),
        "mapping_stage_sec": float(mapping_elapsed),
        "mapped_postpasses_sec": float(postpass_elapsed),
        "alignment_total_sec": float(time.perf_counter() - overall_start),
    }
    return _finalize_alignment_result(
        baseline_lines=baseline_lines,
        mapped_lines=mapped_lines,
        corrections=corrections,
        metrics=metrics,
        mapped_count=mapped_count,
        should_rollback_short_line_degradation_fn=should_rollback_short_line_degradation_fn,
        restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
        logger=logger,
    )
