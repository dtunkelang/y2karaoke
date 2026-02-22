"""Stage orchestration helpers for Whisper integration mapping."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Set, Tuple

from ... import models
from ..alignment import timing_models


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

    return mapped_lines, corrections
