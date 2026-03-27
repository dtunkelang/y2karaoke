"""Advisory start nudges for accepted forced alignment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

from ... import models
from ..alignment import timing_models
from .whisper_forced_advisory_trace import (
    _load_or_transcribe_aggressive_variant,
    _resolve_advisory_audio_path,
    collect_forced_advisory_start_candidates,
)


def _forced_advisory_start_nudge_enabled() -> bool:
    raw = os.environ.get("Y2K_DISABLE_FORCED_ADVISORY_START_NUDGE", "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def _pull_first_word_to_new_start(
    line: models.Line,
    *,
    new_start: float,
) -> models.Line:
    if not line.words or new_start >= line.start_time:
        return line
    words = list(line.words)
    first = words[0]
    first_end = max(first.end_time, new_start + 0.06)
    if len(words) > 1 and first_end > words[1].start_time - 0.01:
        first_end = max(new_start + 0.06, words[1].start_time - 0.01)
    words[0] = models.Word(
        text=first.text,
        start_time=new_start,
        end_time=first_end,
        singer=first.singer,
    )
    return models.Line(words=words, singer=line.singer)


def _load_forced_advisory_candidates(
    *,
    lines: Sequence[models.Line],
    current_segments: Sequence[timing_models.TranscriptionSegment],
    current_words: Sequence[timing_models.TranscriptionWord],
    vocals_path: str,
    language: str | None,
    model_size: str,
    logger: Any,
    load_aggressive_variant_fn: Callable[..., Any],
) -> list[dict[str, object]]:
    if not Path(_resolve_advisory_audio_path(vocals_path)).exists():
        return []
    aggressive_segments, aggressive_words, _aggressive_language = (
        load_aggressive_variant_fn(
            vocals_path=vocals_path,
            language=language,
            model_size=model_size,
            logger=logger,
        )
    )
    return collect_forced_advisory_start_candidates(
        lines=lines,
        current_segments=current_segments,
        current_words=current_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )


def _advisory_candidate_is_eligible(
    *,
    candidate: dict[str, object],
    adjusted_lines: Sequence[models.Line],
    idx: int,
) -> bool:
    if idx < 0 or idx >= len(adjusted_lines):
        return False
    line = adjusted_lines[idx]
    target_start = candidate.get("aggressive_segment_start")
    aggressive_overlap = candidate.get("aggressive_best_overlap")
    default_overlap = candidate.get("default_best_overlap")
    current_window_word_count = candidate.get("current_window_word_count")
    if not isinstance(target_start, (int, float)):
        return False
    if not isinstance(aggressive_overlap, (int, float)):
        return False
    if not isinstance(default_overlap, (int, float)):
        return False
    if not isinstance(current_window_word_count, int):
        return False
    if candidate["bucket"] != "medium_confidence":
        return False
    if len(line.words) != 3:
        return False
    if float(aggressive_overlap) < 0.99:
        return False
    if float(default_overlap) > 0.0:
        return False
    if current_window_word_count > 3:
        return False
    delta = float(target_start) - line.start_time
    if delta > -0.25 or delta < -1.0:
        return False
    prev_end = (
        adjusted_lines[idx - 1].end_time
        if idx > 0 and adjusted_lines[idx - 1].words
        else None
    )
    return prev_end is None or float(target_start) > prev_end + 0.2


def apply_forced_advisory_start_nudges(
    *,
    lines: Sequence[models.Line],
    current_segments: Sequence[timing_models.TranscriptionSegment] | None,
    current_words: Sequence[timing_models.TranscriptionWord] | None,
    vocals_path: str,
    language: str | None,
    model_size: str,
    logger: Any,
    load_aggressive_variant_fn: Callable[
        ..., Any
    ] = _load_or_transcribe_aggressive_variant,
) -> tuple[list[models.Line], int]:
    if not _forced_advisory_start_nudge_enabled():
        return list(lines), 0
    if not current_segments or current_words is None:
        return list(lines), 0
    candidates = _load_forced_advisory_candidates(
        lines=lines,
        current_segments=current_segments,
        current_words=current_words,
        vocals_path=vocals_path,
        language=language,
        model_size=model_size,
        logger=logger,
        load_aggressive_variant_fn=load_aggressive_variant_fn,
    )
    if not candidates:
        return list(lines), 0
    adjusted = list(lines)
    nudged = 0
    for candidate in candidates:
        raw_index = candidate.get("index")
        target_start = candidate.get("aggressive_segment_start")
        if not isinstance(raw_index, int):
            continue
        if not isinstance(target_start, (int, float)):
            continue
        idx = raw_index - 1
        if not _advisory_candidate_is_eligible(
            candidate=candidate,
            adjusted_lines=adjusted,
            idx=idx,
        ):
            continue
        line = adjusted[idx]
        adjusted[idx] = _pull_first_word_to_new_start(
            line,
            new_start=float(target_start),
        )
        nudged += 1
    return adjusted, nudged
