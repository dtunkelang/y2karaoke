"""Runtime advisory support tracing for accepted forced alignment."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from ... import models
from ..alignment import timing_models
from .whisper_advisory_support import (
    advisory_candidate_bucket,
    advisory_candidate_score,
    summarize_line_support,
    token_overlap_score,
    window_words,
)
from .whisper_cache import (
    _get_whisper_cache_path,
    _load_whisper_cache,
    _save_whisper_cache,
)
from .whisper_integration_transcribe import (
    _convert_whisper_segments,
    _run_whisper_transcription,
)


def _load_whisper_model_class() -> Any:
    from faster_whisper import WhisperModel  # type: ignore

    return WhisperModel


def _resolve_advisory_audio_path(vocals_path: str) -> str:
    vocals_file = Path(vocals_path)
    suffix = "_(Vocals)_htdemucs_ft"
    if vocals_file.stem.endswith(suffix):
        clip_stem = vocals_file.stem[: -len(suffix)]
        clip_path = vocals_file.with_name(f"{clip_stem}{vocals_file.suffix}")
        if clip_path.exists():
            return str(clip_path)
    return vocals_path


def _build_report_like_lines(
    lines: Sequence[models.Line],
    current_words: Sequence[timing_models.TranscriptionWord],
) -> list[dict[str, Any]]:
    report_lines: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if not line.words:
            continue
        start = float(line.start_time)
        end = float(line.end_time)
        next_start = (
            float(lines[idx + 1].start_time)
            if idx + 1 < len(lines) and lines[idx + 1].words
            else end + 2.0
        )
        report_lines.append(
            {
                "index": idx + 1,
                "text": line.text,
                "start": start,
                "end": end,
                "whisper_window_word_count": len(
                    window_words(current_words, start=start - 1.0, end=next_start)
                ),
            }
        )
    return report_lines


def _best_segment_match_with_bounds(
    segments: Sequence[timing_models.TranscriptionSegment],
    line_text: str,
    *,
    start: float,
    end: float,
) -> tuple[str, float, float | None, float | None]:
    best_text = ""
    best_score = 0.0
    best_start: float | None = None
    best_end: float | None = None
    for segment in segments:
        seg_start = float(segment.start)
        seg_end = float(segment.end)
        if seg_end < start - 1.0 or seg_start > end + 1.0:
            continue
        score = token_overlap_score(line_text, segment.text)
        if score > best_score:
            best_score = score
            best_text = segment.text
            best_start = seg_start
            best_end = seg_end
    return best_text, round(best_score, 3), best_start, best_end


def _aggressive_cache_is_usable(
    segments: Sequence[timing_models.TranscriptionSegment],
    words: Sequence[timing_models.TranscriptionWord],
) -> bool:
    if not segments or len(words) < 3:
        return False
    if len(segments) == 1 and len(words) >= 8:
        return False
    max_segment_words = max(
        (len(segment.words or []) for segment in segments),
        default=0,
    )
    if len(segments) <= 2 and max_segment_words >= max(8, int(len(words) * 0.7)):
        return False
    return True


def _load_or_transcribe_aggressive_variant(
    *,
    vocals_path: str,
    language: str | None,
    model_size: str,
    logger: Any,
) -> tuple[
    list[timing_models.TranscriptionSegment],
    list[timing_models.TranscriptionWord],
    str,
]:
    advisory_audio_path = _resolve_advisory_audio_path(vocals_path)
    cache_path = _get_whisper_cache_path(
        advisory_audio_path,
        model_size,
        language,
        aggressive=True,
        temperature=0.0,
    )
    if cache_path:
        cached = _load_whisper_cache(cache_path)
        if cached is not None:
            segments, words, detected_lang = cached
            if _aggressive_cache_is_usable(segments, words):
                return segments, words, detected_lang
            logger.info(
                "Ignoring over-merged aggressive advisory cache: %d segments, %d words",
                len(segments),
                len(words),
            )

    whisper_model_class = _load_whisper_model_class()
    model = whisper_model_class(model_size, device="cpu", compute_type="int8")
    raw_segments, info = _run_whisper_transcription(
        model=model,
        vocals_path=advisory_audio_path,
        language=language,
        aggressive=True,
        temperature=0.0,
    )
    segments, words = _convert_whisper_segments(raw_segments)
    detected_lang = str(info.language)
    if cache_path:
        _save_whisper_cache(
            cache_path,
            segments,
            words,
            detected_lang,
            model_size,
            True,
            0.0,
        )
    logger.info(
        "Loaded aggressive advisory transcription: %d segments, %d words",
        len(segments),
        len(words),
    )
    return segments, words, detected_lang


def collect_forced_advisory_start_candidates(
    *,
    lines: Sequence[models.Line],
    current_segments: Sequence[timing_models.TranscriptionSegment],
    current_words: Sequence[timing_models.TranscriptionWord],
    aggressive_segments: Sequence[timing_models.TranscriptionSegment],
    aggressive_words: Sequence[timing_models.TranscriptionWord],
) -> list[dict[str, Any]]:
    report_lines = _build_report_like_lines(lines, current_words)
    summaries = summarize_line_support(
        report_lines=report_lines,
        default_segments=current_segments,
        default_words=current_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )
    candidates: list[dict[str, Any]] = []
    for idx, (summary, report_line) in enumerate(zip(summaries, report_lines)):
        bucket = advisory_candidate_bucket(summary)
        if bucket is None:
            continue
        _text, _score, aggressive_start, aggressive_end = (
            _best_segment_match_with_bounds(
                aggressive_segments,
                summary.text,
                start=float(report_line["start"]) - 1.0,
                end=(
                    float(report_lines[idx + 1]["start"])
                    if idx + 1 < len(report_lines)
                    else float(report_line["end"]) + 2.0
                ),
            )
        )
        candidates.append(
            {
                "bucket": bucket,
                "score": round(advisory_candidate_score(summary), 3),
                "aggressive_segment_start": aggressive_start,
                "aggressive_segment_end": aggressive_end,
                **asdict(summary),
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["index"])))
    return candidates


def build_forced_advisory_trace_payload(
    *,
    lines: Sequence[models.Line],
    current_segments: Sequence[timing_models.TranscriptionSegment],
    current_words: Sequence[timing_models.TranscriptionWord],
    aggressive_segments: Sequence[timing_models.TranscriptionSegment],
    aggressive_words: Sequence[timing_models.TranscriptionWord],
) -> dict[str, Any]:
    report_lines = _build_report_like_lines(lines, current_words)
    summaries = summarize_line_support(
        report_lines=report_lines,
        default_segments=current_segments,
        default_words=current_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )
    candidates = collect_forced_advisory_start_candidates(
        lines=lines,
        current_segments=current_segments,
        current_words=current_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )
    return {
        "lines": [asdict(summary) for summary in summaries],
        "candidates": candidates,
    }


def maybe_write_forced_advisory_trace(
    *,
    lines: Sequence[models.Line],
    current_segments: Sequence[timing_models.TranscriptionSegment] | None,
    current_words: Sequence[timing_models.TranscriptionWord] | None,
    vocals_path: str,
    language: str | None,
    model_size: str,
    logger: Any,
    load_aggressive_variant_fn: Any = _load_or_transcribe_aggressive_variant,
) -> dict[str, Any] | None:
    trace_path = os.environ.get("Y2K_TRACE_FORCED_ADVISORY_JSON", "").strip()
    if not trace_path or not current_segments or current_words is None:
        return None

    aggressive_segments, aggressive_words, aggressive_language = (
        load_aggressive_variant_fn(
            vocals_path=vocals_path,
            language=language,
            model_size=model_size,
            logger=logger,
        )
    )
    payload = build_forced_advisory_trace_payload(
        lines=lines,
        current_segments=current_segments,
        current_words=current_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )
    payload.update(
        {
            "vocals_path": vocals_path,
            "advisory_audio_path": _resolve_advisory_audio_path(vocals_path),
            "language": language,
            "model_size": model_size,
            "aggressive_language": aggressive_language,
        }
    )
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload
