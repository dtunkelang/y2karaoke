"""Stage orchestration helpers for Whisper integration mapping."""

from __future__ import annotations

import re
import os
import statistics
from typing import Any, Callable, List, Optional, Set, Tuple

from ... import models
from ..alignment import timing_models


def _group_repeated_line_indices(lines_in: List[models.Line]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, line in enumerate(lines_in):
        if not line.words:
            continue
        text = (getattr(line, "text", "") or "").strip().lower()
        if not text:
            continue
        groups.setdefault(text, []).append(idx)
    return groups


def _normalized_line_duration(
    adjusted: List[models.Line],
    idx: int,
    target_duration: float,
) -> float:
    line = adjusted[idx]
    start = line.start_time
    next_start = (
        adjusted[idx + 1].start_time
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words
        else None
    )
    capped_duration = target_duration
    if next_start is not None:
        capped_duration = min(
            capped_duration,
            max(0.2, next_start - 0.05 - start),
        )
    return capped_duration


def _scaled_line_duration(
    line: models.Line, target_duration: float
) -> models.Line | None:
    current_duration = max(0.2, line.end_time - line.start_time)
    if abs(target_duration - current_duration) < 0.12 or target_duration <= 0.2:
        return None
    scale = target_duration / current_duration
    return models.Line(
        words=[
            models.Word(
                text=w.text,
                start_time=w.start_time,
                end_time=w.start_time + ((w.end_time - w.start_time) * scale),
                singer=w.singer,
            )
            for w in line.words
        ],
        singer=line.singer,
    )


def _normalize_repeated_line_durations(
    lines_in: List[models.Line],
    *,
    min_repeats: int = 2,
) -> List[models.Line]:
    if os.getenv("Y2K_REPEAT_DURATION_NORMALIZE", "0") != "1":
        return lines_in

    adjusted = list(lines_in)
    for indices in _group_repeated_line_indices(lines_in).values():
        if len(indices) < min_repeats:
            continue
        durations = [
            max(0.2, adjusted[idx].end_time - adjusted[idx].start_time)
            for idx in indices
            if adjusted[idx].words
        ]
        if len(durations) < min_repeats:
            continue
        target_duration = float(statistics.median(durations))
        for idx in indices:
            line = adjusted[idx]
            if not line.words:
                continue
            capped_duration = _normalized_line_duration(adjusted, idx, target_duration)
            scaled_line = _scaled_line_duration(line, capped_duration)
            if scaled_line is not None:
                adjusted[idx] = scaled_line
    return adjusted


def _shift_weak_opening_lines_past_phrase_carryover(
    lines_in: List[models.Line],
    audio_features: Optional[timing_models.AudioFeatures],
    whisper_words: Optional[List[timing_models.TranscriptionWord]] = None,
    *,
    min_gap: float = 0.05,
    carryover_buffer: float = 0.7,
    min_local_overlap_to_keep: float = 0.35,
) -> tuple[List[models.Line], int]:
    if audio_features is None or audio_features.onset_times is None:
        return lines_in, 0
    onset_times = audio_features.onset_times
    if len(onset_times) == 0:
        return lines_in, 0
    shifted = list(lines_in)
    applied = 0
    for idx in range(1, len(shifted)):
        prev_line = shifted[idx - 1]
        line = shifted[idx]
        if not _should_shift_weak_opening_line(
            prev_line,
            line,
            whisper_words=whisper_words,
            min_local_overlap_to_keep=min_local_overlap_to_keep,
        ):
            continue
        next_start = (
            shifted[idx + 1].start_time
            if idx + 1 < len(shifted) and shifted[idx + 1].words
            else None
        )
        max_allowed = (
            max(0.0, next_start - min_gap - line.end_time)
            if next_start is not None
            else 1.5
        )
        if max_allowed < 0.35:
            continue
        candidate_onsets = onset_times[
            (
                onset_times
                >= max(
                    line.start_time + 0.25,
                    prev_line.end_time + carryover_buffer,
                )
            )
            & (onset_times <= line.start_time + min(1.4, max_allowed + 0.1))
        ]
        if len(candidate_onsets) == 0:
            continue
        target_start = float(candidate_onsets[0])
        shift = target_start - line.start_time
        if shift < 0.25 or shift > max_allowed + 1e-6:
            continue
        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        shifted[idx] = models.Line(words=shifted_words, singer=line.singer)
        applied += 1
    return shifted, applied


def _should_shift_weak_opening_line(
    prev_line: models.Line,
    line: models.Line,
    *,
    whisper_words: Optional[List[timing_models.TranscriptionWord]],
    min_local_overlap_to_keep: float,
) -> bool:
    if not prev_line.words or not line.words or len(line.words) < 6:
        return False
    if line.start_time - prev_line.end_time > 0.25:
        return False
    tokens = _normalized_line_tokens(line)
    if not tokens or tokens[0] not in {"no", "maybe", "oh", "cause"}:
        return False
    if whisper_words is None:
        return True
    return _local_line_token_overlap(line, whisper_words) < min_local_overlap_to_keep


def _normalized_line_tokens(line: models.Line) -> list[str]:
    return [
        token
        for word in line.words
        if (token := re.sub(r"[^a-z]+", "", word.text.lower()))
    ]


def _local_line_token_overlap(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    pad_before: float = 0.8,
    pad_after: float = 0.8,
) -> float:
    line_tokens = {
        re.sub(r"[^a-z]+", "", word.text.lower())
        for word in line.words
        if re.sub(r"[^a-z]+", "", word.text.lower())
    }
    if not line_tokens:
        return 0.0
    nearby_tokens = {
        re.sub(r"[^a-z]+", "", word.text.lower())
        for word in whisper_words
        if line.start_time - pad_before <= word.start <= line.end_time + pad_after
        and word.text != "[VOCAL]"
        and re.sub(r"[^a-z]+", "", word.text.lower())
    }
    if not nearby_tokens:
        return 0.0
    overlap = len(line_tokens & nearby_tokens)
    return overlap / max(len(line_tokens), len(nearby_tokens))


def _enforce_mapped_line_stage_invariants(
    lines_in: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any],
    resolve_line_overlaps_fn: Callable[..., Any],
) -> List[models.Line]:
    """Settle monotonic/overlap constraints across dense mapped lyric sections."""
    out = lines_in
    for _ in range(2):
        out = enforce_monotonic_line_starts_whisper_fn(out, all_words)
        out = resolve_line_overlaps_fn(out)
    return out


def _run_mapped_line_postpasses(
    *,
    mapped_lines: List[models.Line],
    mapped_lines_set: Set[int],
    all_words: List[timing_models.TranscriptionWord],
    transcription: List[timing_models.TranscriptionSegment],
    audio_features: Optional[timing_models.AudioFeatures],
    vocals_path: str,
    epitran_lang: str,
    corrections: List[str],
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
) -> Tuple[List[models.Line], List[str]]:
    mapped_lines = interpolate_unmatched_lines_fn(mapped_lines, mapped_lines_set)
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )

    mapped_lines = refine_unmatched_lines_with_onsets_fn(
        mapped_lines,
        mapped_lines_set,
        vocals_path,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )

    mapped_lines = shift_repeated_lines_to_next_whisper_fn(mapped_lines, all_words)
    mapped_lines = _normalize_repeated_line_durations(mapped_lines)
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = extend_line_to_trailing_whisper_matches_fn(
        mapped_lines,
        all_words,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = pull_late_lines_to_matching_segments_fn(
        mapped_lines,
        transcription,
        epitran_lang,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = retime_short_interjection_lines_fn(
        mapped_lines,
        transcription,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = snap_first_word_to_whisper_onset_fn(
        mapped_lines,
        all_words,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    if audio_features is not None:
        mapped_lines, continuous_fixes = pull_lines_forward_for_continuous_vocals_fn(
            mapped_lines,
            audio_features,
        )
        if continuous_fixes:
            corrections.append(
                f"Pulled {continuous_fixes} line(s) forward for continuous vocals"
            )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    mapped_lines = pull_late_lines_to_matching_segments_fn(
        mapped_lines,
        transcription,
        epitran_lang,
    )
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    try:
        mapped_lines = snap_first_word_to_whisper_onset_fn(
            mapped_lines,
            all_words,
            max_shift=2.5,
        )
    except TypeError:
        mapped_lines = snap_first_word_to_whisper_onset_fn(mapped_lines, all_words)
    mapped_lines = _enforce_mapped_line_stage_invariants(
        mapped_lines,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )
    if audio_features is not None:
        mapped_lines, late_audio_fixes = pull_lines_forward_for_continuous_vocals_fn(
            mapped_lines,
            audio_features,
        )
        if late_audio_fixes:
            corrections.append(
                f"Applied {late_audio_fixes} late audio onset/silence adjustment(s)"
            )
        mapped_lines = _enforce_mapped_line_stage_invariants(
            mapped_lines,
            all_words,
            enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
            resolve_line_overlaps_fn=resolve_line_overlaps_fn,
        )
        try:
            mapped_lines = snap_first_word_to_whisper_onset_fn(
                mapped_lines,
                all_words,
                max_shift=2.5,
            )
        except TypeError:
            mapped_lines = snap_first_word_to_whisper_onset_fn(mapped_lines, all_words)
        mapped_lines = _enforce_mapped_line_stage_invariants(
            mapped_lines,
            all_words,
            enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
            resolve_line_overlaps_fn=resolve_line_overlaps_fn,
        )
        mapped_lines, carryover_fixes = _shift_weak_opening_lines_past_phrase_carryover(
            mapped_lines,
            audio_features,
            all_words,
        )
        if carryover_fixes:
            corrections.append(
                f"Shifted {carryover_fixes} weak-opening line(s) past prior-phrase carryover"
            )
            mapped_lines = _enforce_mapped_line_stage_invariants(
                mapped_lines,
                all_words,
                enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
                resolve_line_overlaps_fn=resolve_line_overlaps_fn,
            )

    return mapped_lines, corrections
