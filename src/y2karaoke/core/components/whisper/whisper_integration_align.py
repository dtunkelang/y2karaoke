"""DTW-based LRC-to-Whisper alignment orchestration for integration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import re

from ....utils.lex_lookup_installer import ensure_local_lex_lookup
from ... import models, phonetic_utils
from ..alignment import timing_models
from .whisper_forced_alignment import align_lines_with_whisperx
from .whisper_integration_finalize import _restore_pairwise_inversions_from_source
from .whisper_integration_forced_fallback import (
    attempt_whisperx_forced_alignment,
)
from .whisper_integration_shift_guard import (
    should_apply_baseline_constraint as _should_apply_baseline_constraint,
)
from .whisper_integration_stages import (
    _shift_weak_opening_lines_past_phrase_carryover,
)
from .whisper_integration_weak_evidence import (
    restore_weak_evidence_large_start_shifts as _restore_weak_evidence_large_start_shifts,
    restore_unsupported_early_duplicate_shifts as _restore_unsupported_early_duplicate_shifts,
)
from .whisper_profile import get_whisper_profile

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


def _default_mapping_decision_config() -> _WhisperMappingDecisionConfig:
    profile = get_whisper_profile()
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


def _count_non_vocal_words_near_time(
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


def _normalized_prefix_tokens(line: models.Line, *, limit: int = 3) -> list[str]:
    return [
        re.sub(r"[^a-z]+", "", w.text.lower())
        for w in line.words[:limit]
        if re.sub(r"[^a-z]+", "", w.text.lower())
    ]


def _normalized_tokens(line: models.Line) -> list[str]:
    return [
        re.sub(r"[^a-z]+", "", w.text.lower())
        for w in line.words
        if re.sub(r"[^a-z]+", "", w.text.lower())
    ]


def _rescale_line_to_new_start(line: models.Line, target_start: float) -> models.Line:
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


def _choose_i_said_reanchor_start(
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    onset_times: Any,
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 6:
        return None
    if _normalized_prefix_tokens(line)[:2] != ["i", "said"]:
        return None
    if _count_non_vocal_words_near_time(whisper_words, line.start_time, window_sec=0.9):
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


def _local_lexical_overlap_ratio(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    pad_before: float = 0.8,
    pad_after: float = 0.8,
) -> float:
    line_tokens = set(_normalized_tokens(line))
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


def _reanchor_unsupported_i_said_lines_to_later_onset(
    mapped_lines: List[models.Line],
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
        target_start = _choose_i_said_reanchor_start(
            line, next_line, whisper_words, onset_times
        )
        if target_start is None:
            continue
        updated[idx] = _rescale_line_to_new_start(line, target_start)
        applied += 1
    return updated, applied


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
    logger,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Align LRC text to Whisper timings via DTW-style phonetic assignment."""
    config = _default_mapping_decision_config()
    _ = min_similarity  # reserved for potential future tuning hooks
    _ = lenient_activity_bonus  # consumed by downstream scoring in related paths

    baseline_lines = clone_lines_for_fallback_fn(lines)
    overall_start = time.perf_counter()
    ensure_local_lex_lookup()
    transcription, all_words, detected_lang, used_model = transcribe_vocals_fn(
        vocals_path, language, model_size, aggressive, temperature
    )
    if not audio_features:
        audio_features = extract_audio_features_fn(vocals_path)
    transcription = dedupe_whisper_segments_fn(transcription)

    line_texts = [line.text for line in lines if line.text.strip()]
    transcription, all_words, trimmed_end = trim_whisper_transcription_by_lyrics_fn(
        transcription, all_words, line_texts
    )
    if trimmed_end:
        logger.info(
            "Truncated Whisper transcript to %.2f s (last matching lyric).", trimmed_end
        )

    sparse_whisper_output = (
        len(all_words) < config.sparse_word_threshold
        or len(transcription) <= config.sparse_segment_threshold
    )
    if sparse_whisper_output:
        forced_result = attempt_whisperx_forced_alignment(
            lines=lines,
            baseline_lines=baseline_lines,
            vocals_path=vocals_path,
            language=language,
            logger=logger,
            used_model=used_model,
            reason="sparse Whisper transcript",
            align_lines_with_whisperx_fn=align_lines_with_whisperx,
            should_rollback_short_line_degradation_fn=should_rollback_short_line_degradation_fn,
            restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
            min_forced_word_coverage=_MIN_FORCED_WORD_COVERAGE,
            min_forced_line_coverage=_MIN_FORCED_LINE_COVERAGE,
        )
        if forced_result is not None:
            return forced_result

    if not transcription or not all_words:
        logger.warning("No transcription available, skipping Whisper timing map")
        return lines, [], {}

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
    whisper_words_after_filter = len(all_words)

    epitran_lang = phonetic_utils._whisper_lang_to_epitran(detected_lang)
    logger.debug(
        "Using epitran language: %s (from Whisper: %s)", epitran_lang, detected_lang
    )

    lrc_words = extract_lrc_words_all_fn(lines)
    if not lrc_words:
        return lines, [], {}

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
    if seg_coverage < min_segment_overlap_coverage:
        logger.debug(
            "Segment overlap coverage %.0f%% below %.0f%% threshold, falling back to DTW",
            seg_coverage * 100,
            min_segment_overlap_coverage * 100,
        )
        if not lrc_syllables or not whisper_syllables:
            if not lrc_phonemes or not whisper_phonemes:
                logger.warning("No phoneme/syllable data; skipping mapping")
                return lines, [], {}
            path = build_phoneme_dtw_path_fn(
                lrc_phonemes,
                whisper_phonemes,
                epitran_lang,
            )
            lrc_assignments = build_word_assignments_from_phoneme_path_fn(
                path, lrc_phonemes, whisper_phonemes
            )
        else:
            lrc_assignments = build_block_segmented_syllable_assignments_fn(
                lrc_words,
                all_words,
                lrc_syllables,
                whisper_syllables,
                epitran_lang,
            )

    corrections: List[str] = []
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
    )
    postpass_elapsed = time.perf_counter() - postpass_start

    matched_ratio = mapped_count / len(lrc_words) if lrc_words else 0.0
    avg_similarity = total_similarity / mapped_count if mapped_count else 0.0
    line_coverage = (
        len(mapped_lines_set) / sum(1 for line in lines if line.words) if lines else 0.0
    )
    if len(lrc_words) >= config.low_coverage_lrc_word_min and (
        matched_ratio < config.low_coverage_matched_ratio_max
        or line_coverage < config.low_coverage_line_coverage_max
    ):
        forced_result = attempt_whisperx_forced_alignment(
            lines=lines,
            baseline_lines=baseline_lines,
            vocals_path=vocals_path,
            language=language,
            logger=logger,
            used_model=used_model,
            reason="low DTW mapping coverage",
            align_lines_with_whisperx_fn=align_lines_with_whisperx,
            should_rollback_short_line_degradation_fn=should_rollback_short_line_degradation_fn,
            restore_implausibly_short_lines_fn=restore_implausibly_short_lines_fn,
            min_forced_word_coverage=_MIN_FORCED_WORD_COVERAGE,
            min_forced_line_coverage=_MIN_FORCED_LINE_COVERAGE,
        )
        if forced_result is not None:
            return forced_result

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
    if apply_baseline_constraint:
        mapped_lines = constrain_line_starts_to_baseline_fn(
            mapped_lines, baseline_lines
        )
    else:
        corrections.append(
            "Skipped baseline start constraint due to strong global Whisper shift evidence"
        )

    try:
        mapped_lines = snap_first_word_to_whisper_onset_fn(
            mapped_lines,
            all_words,
            max_shift=config.snap_first_word_max_shift,
        )
    except TypeError:
        mapped_lines = snap_first_word_to_whisper_onset_fn(mapped_lines, all_words)
    if apply_baseline_constraint:
        mapped_lines = constrain_line_starts_to_baseline_fn(
            mapped_lines, baseline_lines
        )
    mapped_lines, restored_weak = _restore_weak_evidence_large_start_shifts(
        mapped_lines,
        baseline_lines,
        all_words,
    )
    if restored_weak:
        corrections.append(
            f"Restored {restored_weak} weak-evidence large start shift line(s) to baseline"
        )
    mapped_lines, restored_early_duplicates = (
        _restore_unsupported_early_duplicate_shifts(
            mapped_lines,
            baseline_lines,
            all_words,
        )
    )
    if restored_early_duplicates:
        corrections.append(
            "Restored "
            f"{restored_early_duplicates} unsupported early duplicate line(s) to baseline"
        )
    mapped_lines, restored_short = restore_implausibly_short_lines_fn(
        baseline_lines, mapped_lines
    )
    if restored_short:
        corrections.append(
            f"Restored {restored_short} short compressed lines from baseline timing"
        )
    mapped_lines, restored_inversions = _restore_pairwise_inversions_from_source(
        baseline_lines,
        mapped_lines,
        min_inversion_gap=0.25,
        min_ahead_shift=2.5,
    )
    if restored_inversions:
        corrections.append(
            f"Restored {restored_inversions} inversion outlier line(s) from baseline timing"
        )
    if audio_features is not None:
        mapped_lines, carryover_fixes = _shift_weak_opening_lines_past_phrase_carryover(
            mapped_lines,
            audio_features,
        )
        if carryover_fixes:
            corrections.append(
                "Shifted "
                f"{carryover_fixes} weak-opening line(s) past prior-phrase carryover"
            )
        mapped_lines, said_reanchors = (
            _reanchor_unsupported_i_said_lines_to_later_onset(
                mapped_lines,
                all_words,
                audio_features,
            )
        )
        if said_reanchors:
            corrections.append(
                "Reanchored "
                f"{said_reanchors} unsupported 'I said' line(s) to later audio onsets"
            )
        mapped_lines, tail_extensions = _extend_unsupported_parenthetical_tails(
            mapped_lines,
            all_words,
        )
        if tail_extensions:
            corrections.append(
                "Extended " f"{tail_extensions} unsupported parenthetical tail(s)"
            )
        mapped_lines, i_said_tail_extensions = _extend_unsupported_i_said_tails(
            mapped_lines,
            all_words,
        )
        if i_said_tail_extensions:
            corrections.append(
                "Extended " f"{i_said_tail_extensions} unsupported 'I said' tail(s)"
            )
        mapped_lines, interjection_reanchors = (
            _reanchor_unsupported_interjection_lines_to_onsets(
                mapped_lines,
                all_words,
                audio_features,
            )
        )
        if interjection_reanchors:
            corrections.append(
                "Reanchored "
                f"{interjection_reanchors} unsupported interjection line(s) to audio onsets"
            )
        mapped_lines, pre_i_said_extensions = _extend_misaligned_lines_before_i_said(
            mapped_lines,
            all_words,
        )
        if pre_i_said_extensions:
            corrections.append(
                "Extended "
                f"{pre_i_said_extensions} lexically mismatched line(s) before unsupported 'I said' lines"
            )
        mapped_lines, pre_weak_opening_extensions = (
            _extend_unsupported_long_lines_before_weak_opening(
                mapped_lines,
                all_words,
            )
        )
        if pre_weak_opening_extensions:
            corrections.append(
                "Extended "
                f"{pre_weak_opening_extensions} unsupported line(s) before weak openings"
            )
        mapped_lines, weak_opening_extensions = _extend_unsupported_weak_opening_lines(
            mapped_lines,
            all_words,
        )
        if weak_opening_extensions:
            corrections.append(
                "Extended "
                f"{weak_opening_extensions} unsupported weak-opening line(s)"
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

    if mapped_count:
        corrections.append(f"DTW-phonetic mapped {mapped_count} word(s) to Whisper")
    rollback, short_before, short_after = should_rollback_short_line_degradation_fn(
        baseline_lines, mapped_lines
    )
    if rollback:
        repaired_lines, restored_count = restore_implausibly_short_lines_fn(
            baseline_lines, mapped_lines
        )
        repaired_rollback, _, repaired_after = (
            should_rollback_short_line_degradation_fn(baseline_lines, repaired_lines)
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
    return mapped_lines, corrections, metrics
