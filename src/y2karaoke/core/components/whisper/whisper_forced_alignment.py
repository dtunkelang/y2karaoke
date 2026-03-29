"""Transcript-constrained WhisperX forced alignment fallback."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ... import models
from .whisperx_compat import patch_torchaudio_for_whisperx

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_LIGHT_LEADING_TOKENS = {"the", "a", "an", "if"}


def _trace_whisperx_char_alignments_enabled() -> bool:
    raw = os.environ.get("Y2K_TRACE_WHISPERX_FORCED_CHAR_ALIGN", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _maybe_write_whisperx_trace(payload: Dict[str, Any]) -> None:
    trace_path = os.environ.get("Y2K_TRACE_WHISPERX_FORCED_JSON", "").strip()
    if not trace_path:
        return
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _norm_token(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def _safe_segment_bounds(
    line: models.Line, max_audio_end: float
) -> Tuple[float, float]:
    start = max(0.0, line.start_time - 0.45)
    end = max(start + 0.25, line.end_time + 0.8)
    if max_audio_end > 0.0:
        end = min(end, max_audio_end)
        if end <= start:
            end = min(max_audio_end, start + 0.25)
    return start, end


def _distribute_line_words(
    line: models.Line, start: float, end: float
) -> List[models.Word]:
    words = line.words
    if not words:
        return []
    span = max(0.08 * len(words), end - start, 0.12)
    left = start
    right = left + span
    if right <= left:
        right = left + 0.12
    weights = [max(1.0, float(len(w.text.strip()))) for w in words]
    total = sum(weights) or float(len(words))
    out: List[models.Word] = []
    cursor = left
    for idx, src in enumerate(words):
        if idx == len(words) - 1:
            nxt = right
        else:
            nxt = cursor + span * (weights[idx] / total)
        nxt = max(nxt, cursor + 0.06)
        out.append(
            models.Word(
                text=src.text,
                start_time=cursor,
                end_time=nxt,
                singer=src.singer,
            )
        )
        cursor = nxt
    out[-1].end_time = max(out[-1].start_time + 0.06, right)
    return out


def _find_monotonic_token_matches(
    line: models.Line, seg_words: Sequence[Tuple[float, float, str]]
) -> Dict[int, int]:
    norm_targets = [_norm_token(w.text) for w in line.words]
    norm_observed = [_norm_token(w[2]) for w in seg_words]
    match_idx: Dict[int, int] = {}
    cursor = 0
    for target_idx, token in enumerate(norm_targets):
        if not token:
            continue
        for observed_idx in range(cursor, len(norm_observed)):
            if norm_observed[observed_idx] == token:
                match_idx[target_idx] = observed_idx
                cursor = observed_idx + 1
                break
    return match_idx


def _estimate_anchor_for_unmatched_word(
    target_idx: int,
    match_idx: Dict[int, int],
    seg_words: Sequence[Tuple[float, float, str]],
    seg_start: float,
    seg_end: float,
) -> Tuple[float, float]:
    prev = [k for k in match_idx.keys() if k < target_idx]
    nxt = [k for k in match_idx.keys() if k > target_idx]
    if prev and nxt:
        prev_idx = max(prev)
        next_idx = min(nxt)
        prev_word = match_idx[prev_idx]
        next_word = match_idx[next_idx]
        frac = (target_idx - prev_idx) / max(1.0, float(next_idx - prev_idx))
        start = (
            seg_words[prev_word][0]
            + (seg_words[next_word][0] - seg_words[prev_word][0]) * frac
        )
        end = (
            seg_words[prev_word][1]
            + (seg_words[next_word][1] - seg_words[prev_word][1]) * frac
        )
        return start, end
    if prev:
        prev_word = match_idx[max(prev)]
        start = seg_words[prev_word][1]
        return start, start + 0.12
    if nxt:
        next_word = match_idx[min(nxt)]
        end = seg_words[next_word][0]
        return end - 0.12, end
    return seg_start, seg_end


def _build_words_from_anchors(
    line: models.Line,
    starts: Sequence[float],
    ends: Sequence[float],
    seg_start: float,
    seg_end: float,
) -> List[models.Word]:
    out: List[models.Word] = []
    cursor_time = seg_start
    for idx, src in enumerate(line.words):
        start = max(cursor_time, starts[idx], seg_start)
        end = max(start + 0.06, ends[idx])
        if idx + 1 < len(line.words):
            next_start = starts[idx + 1]
            if end > next_start and next_start > start + 0.04:
                end = max(start + 0.04, next_start - 0.01)
        out.append(
            models.Word(
                text=src.text,
                start_time=start,
                end_time=end,
                singer=src.singer,
            )
        )
        cursor_time = out[-1].end_time
    out[-1].end_time = max(out[-1].end_time, seg_end)
    return out


def _tighten_leading_light_token_anchors(
    starts: List[float],
    ends: List[float],
    norm_targets: Sequence[str],
) -> None:
    if len(norm_targets) < 2:
        return
    leading_count = 0
    for token in norm_targets:
        if token in _LIGHT_LEADING_TOKENS:
            leading_count += 1
            continue
        break
    if leading_count == 0 or leading_count >= len(norm_targets):
        return

    anchor_start = starts[leading_count]
    current_start = starts[0]
    if anchor_start <= current_start + 0.35:
        return

    total_window = min(
        0.32 * leading_count,
        max(0.16, anchor_start - current_start - 0.02),
    )
    if total_window <= 0.08:
        return

    target_start = anchor_start - total_window
    word_span = total_window / leading_count
    for idx in range(leading_count):
        start = target_start + idx * word_span
        if idx == leading_count - 1:
            end = anchor_start - 0.02
        else:
            end = start + word_span * 0.82
        starts[idx] = start
        ends[idx] = max(start + 0.06, end)


def _serialize_words(words: Sequence[models.Word]) -> List[Dict[str, Any]]:
    return [
        {
            "text": word.text,
            "start": word.start_time,
            "end": word.end_time,
        }
        for word in words
    ]


def _build_line_mapping_debug(
    line: models.Line,
    seg_words: Sequence[Tuple[float, float, str]],
    match_idx: Dict[int, int],
    mapped_words: Sequence[models.Word],
    min_required: int,
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "line_text": line.text,
        "line_tokens": [word.text for word in line.words],
        "segment_words": [
            {"start": start, "end": end, "text": text} for start, end, text in seg_words
        ],
        "normalized_line_tokens": [_norm_token(word.text) for word in line.words],
        "match_index_by_line_token": {
            str(target_idx): observed_idx
            for target_idx, observed_idx in match_idx.items()
        },
        "match_count": len(match_idx),
        "min_required_match_count": min_required,
        "fallback_reason": fallback_reason,
        "mapped_words": _serialize_words(mapped_words),
    }


def _map_segment_words_to_line_with_debug(
    line: models.Line,
    seg_words: Sequence[Tuple[float, float, str]],
    seg_start: float,
    seg_end: float,
) -> Tuple[List[models.Word], Dict[str, Any]]:
    if not line.words:
        return [], _build_line_mapping_debug(
            line=line,
            seg_words=seg_words,
            match_idx={},
            mapped_words=[],
            min_required=0,
            fallback_reason="empty_line",
        )
    if not seg_words:
        mapped_words = _distribute_line_words(line, seg_start, seg_end)
        return mapped_words, _build_line_mapping_debug(
            line=line,
            seg_words=[],
            match_idx={},
            mapped_words=mapped_words,
            min_required=max(1, math.ceil(len(line.words) * 0.35)),
            fallback_reason="empty_segment_words",
        )

    match_idx = _find_monotonic_token_matches(line, seg_words)
    min_required = max(1, math.ceil(len(line.words) * 0.35))
    if len(match_idx) < min_required:
        mapped_words = _distribute_line_words(line, seg_start, seg_end)
        return mapped_words, _build_line_mapping_debug(
            line=line,
            seg_words=seg_words,
            match_idx=match_idx,
            mapped_words=mapped_words,
            min_required=min_required,
            fallback_reason="insufficient_matches",
        )

    norm_targets = [_norm_token(w.text) for w in line.words]
    starts: List[float] = []
    ends: List[float] = []
    for ti in range(len(line.words)):
        if ti in match_idx:
            oi = match_idx[ti]
            starts.append(seg_words[oi][0])
            ends.append(seg_words[oi][1])
            continue
        s, e = _estimate_anchor_for_unmatched_word(
            ti, match_idx, seg_words, seg_start, seg_end
        )
        starts.append(s)
        ends.append(e)
    _tighten_leading_light_token_anchors(starts, ends, norm_targets)
    mapped_words = _build_words_from_anchors(line, starts, ends, seg_start, seg_end)
    return mapped_words, _build_line_mapping_debug(
        line=line,
        seg_words=seg_words,
        match_idx=match_idx,
        mapped_words=mapped_words,
        min_required=min_required,
    )


def _map_segment_words_to_line(
    line: models.Line,
    seg_words: Sequence[Tuple[float, float, str]],
    seg_start: float,
    seg_end: float,
) -> List[models.Word]:
    mapped_words, _debug = _map_segment_words_to_line_with_debug(
        line, seg_words, seg_start, seg_end
    )
    return mapped_words


def _count_line_overlaps(lines: List[models.Line]) -> int:
    overlaps = 0
    prev_end = None
    for line in lines:
        if not line.words:
            continue
        if prev_end is not None and line.start_time < prev_end - 0.02:
            overlaps += 1
        prev_end = line.end_time
    return overlaps


def _index_aligned_segments(
    aligned_segments: Sequence[dict], non_empty: Sequence[int]
) -> Dict[int, dict]:
    seg_by_idx: Dict[int, dict] = {}
    fallback_pairs = zip(non_empty, aligned_segments)
    for fallback_idx, seg in fallback_pairs:
        idx_raw = seg.get("id")
        if isinstance(idx_raw, int):
            seg_by_idx[idx_raw] = seg
        elif fallback_idx not in seg_by_idx:
            seg_by_idx[fallback_idx] = seg
    return seg_by_idx


def _min_required_timed_lines(non_empty_line_count: int) -> int:
    if non_empty_line_count <= 0:
        return 0
    if non_empty_line_count <= 3:
        return non_empty_line_count
    return max(3, int(non_empty_line_count * 0.5))


def align_lines_with_whisperx(  # noqa: C901
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    logger,
) -> Optional[Tuple[List[models.Line], Dict[str, float]]]:
    """Force-align known lyric lines with WhisperX and return aligned lines."""
    trace: Dict[str, Any] = {
        "status": "started",
        "language": language,
        "vocals_path": vocals_path,
    }
    patch_torchaudio_for_whisperx()
    try:
        import whisperx  # type: ignore
    except Exception as exc:
        trace.update({"status": "import_failed", "error": repr(exc)})
        _maybe_write_whisperx_trace(trace)
        return None

    non_empty = [
        idx for idx, line in enumerate(lines) if line.words and line.text.strip()
    ]
    trace["non_empty_line_count"] = len(non_empty)
    if len(non_empty) < 2:
        trace["status"] = "too_few_non_empty_lines"
        _maybe_write_whisperx_trace(trace)
        return None

    try:
        audio = whisperx.load_audio(vocals_path)
        audio_end = (len(audio) / 16000.0) if hasattr(audio, "__len__") else 0.0
        lang_code = (language or "en").split("-", 1)[0].strip() or "en"
        trace["language_code"] = lang_code
        trace["audio_end"] = audio_end
        align_model, metadata = whisperx.load_align_model(
            language_code=lang_code,
            device="cpu",
        )
        return_char_alignments = _trace_whisperx_char_alignments_enabled()
        trace["return_char_alignments"] = return_char_alignments
        segs = []
        for idx in non_empty:
            line = lines[idx]
            start, end = _safe_segment_bounds(line, audio_end)
            segs.append(
                {
                    "id": idx,
                    "start": float(start),
                    "end": float(end),
                    "text": line.text,
                }
            )
        trace["requested_segments"] = segs
        aligned = whisperx.align(
            segs,
            align_model,
            metadata,
            audio,
            device="cpu",
            return_char_alignments=return_char_alignments,
        )
    except Exception as exc:
        logger.debug("whisperx forced alignment failed: %s", exc)
        trace.update({"status": "align_failed", "error": repr(exc)})
        _maybe_write_whisperx_trace(trace)
        return None

    aligned_segments = aligned.get("segments", [])
    trace["aligned_segment_count"] = len(aligned_segments)
    trace["aligned_segments"] = aligned_segments
    if not aligned_segments:
        trace["status"] = "no_aligned_segments"
        _maybe_write_whisperx_trace(trace)
        return None

    seg_by_idx = _index_aligned_segments(aligned_segments, non_empty)

    forced_lines: List[models.Line] = []
    trace_line_mappings: List[Dict[str, Any]] = []
    timed_lines = 0
    total_words = 0
    timed_words = 0
    for idx, line in enumerate(lines):
        if not line.words:
            forced_lines.append(line)
            continue
        total_words += len(line.words)
        seg = seg_by_idx.get(idx)
        if seg is None:
            forced_lines.append(line)
            continue
        raw_words = []
        for w in seg.get("words", []):
            ws = w.get("start")
            we = w.get("end")
            if not isinstance(ws, (int, float)) or not isinstance(we, (int, float)):
                continue
            text = str(w.get("word") or w.get("text") or "").strip()
            if not text:
                continue
            raw_words.append((float(ws), float(we), text))
        seg_start = float(seg.get("start", line.start_time))
        seg_end = float(seg.get("end", line.end_time))
        mapped_words, line_mapping = _map_segment_words_to_line_with_debug(
            line, raw_words, seg_start, seg_end
        )
        line_mapping.update(
            {
                "line_index": idx,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "segment_text": str(seg.get("text") or ""),
            }
        )
        trace_line_mappings.append(line_mapping)
        if mapped_words:
            timed_lines += 1
            timed_words += len(mapped_words)
            forced_lines.append(models.Line(words=mapped_words, singer=line.singer))
        else:
            forced_lines.append(line)

    min_timed_lines = _min_required_timed_lines(len(non_empty))
    trace["timed_line_count"] = timed_lines
    trace["timed_word_count"] = timed_words
    trace["min_timed_lines_required"] = min_timed_lines
    if timed_lines < min_timed_lines:
        trace["status"] = "insufficient_timed_lines"
        _maybe_write_whisperx_trace(trace)
        return None

    overlaps = _count_line_overlaps(forced_lines)
    timed_ratio = timed_lines / max(1, len(non_empty))
    word_ratio = timed_words / max(1, total_words)
    trace["timed_ratio"] = timed_ratio
    trace["word_ratio"] = word_ratio
    trace["overlaps"] = overlaps
    if timed_ratio < 0.7 or word_ratio < 0.7:
        trace["status"] = "insufficient_coverage"
        _maybe_write_whisperx_trace(trace)
        return None
    if overlaps > max(2, int(len(non_empty) * 0.08)):
        trace["status"] = "too_many_overlaps"
        _maybe_write_whisperx_trace(trace)
        return None

    metrics = {
        "forced_line_coverage": timed_ratio,
        "forced_word_coverage": word_ratio,
        "forced_line_overlaps": float(overlaps),
        "aligned_segments": aligned_segments,
    }
    trace.update(
        {
            "status": "accepted",
            "metrics": metrics,
            "line_mappings": trace_line_mappings,
        }
    )
    _maybe_write_whisperx_trace(trace)
    logger.info(
        "WhisperX forced alignment accepted: %.0f%% lines, %.0f%% words",
        timed_ratio * 100.0,
        word_ratio * 100.0,
    )
    return forced_lines, metrics
