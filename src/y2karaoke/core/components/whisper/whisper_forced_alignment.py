"""Transcript-constrained WhisperX forced alignment fallback."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ... import models
from .whisperx_compat import patch_torchaudio_for_whisperx

_TOKEN_RE = re.compile(r"[a-z0-9']+")


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


def _map_segment_words_to_line(
    line: models.Line,
    seg_words: Sequence[Tuple[float, float, str]],
    seg_start: float,
    seg_end: float,
) -> List[models.Word]:
    if not line.words:
        return []
    if not seg_words:
        return _distribute_line_words(line, seg_start, seg_end)

    match_idx = _find_monotonic_token_matches(line, seg_words)

    min_required = max(1, math.ceil(len(line.words) * 0.35))
    if len(match_idx) < min_required:
        return _distribute_line_words(line, seg_start, seg_end)

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
    return _build_words_from_anchors(line, starts, ends, seg_start, seg_end)


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


def align_lines_with_whisperx(  # noqa: C901
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    logger,
) -> Optional[Tuple[List[models.Line], Dict[str, float]]]:
    """Force-align known lyric lines with WhisperX and return aligned lines."""
    patch_torchaudio_for_whisperx()
    try:
        import whisperx  # type: ignore
    except Exception:
        return None

    non_empty = [
        idx for idx, line in enumerate(lines) if line.words and line.text.strip()
    ]
    if len(non_empty) < 4:
        return None

    try:
        audio = whisperx.load_audio(vocals_path)
        audio_end = (len(audio) / 16000.0) if hasattr(audio, "__len__") else 0.0
        lang_code = (language or "en").split("-", 1)[0].strip() or "en"
        align_model, metadata = whisperx.load_align_model(
            language_code=lang_code,
            device="cpu",
        )
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
        aligned = whisperx.align(
            segs,
            align_model,
            metadata,
            audio,
            device="cpu",
            return_char_alignments=False,
        )
    except Exception as exc:
        logger.debug("whisperx forced alignment failed: %s", exc)
        return None

    aligned_segments = aligned.get("segments", [])
    if not aligned_segments:
        return None

    seg_by_idx: Dict[int, dict] = {}
    for seg in aligned_segments:
        idx_raw = seg.get("id")
        if isinstance(idx_raw, int):
            seg_by_idx[idx_raw] = seg

    forced_lines: List[models.Line] = []
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
        mapped_words = _map_segment_words_to_line(line, raw_words, seg_start, seg_end)
        if mapped_words:
            timed_lines += 1
            timed_words += len(mapped_words)
            forced_lines.append(models.Line(words=mapped_words, singer=line.singer))
        else:
            forced_lines.append(line)

    if timed_lines < max(3, int(len(non_empty) * 0.5)):
        return None

    overlaps = _count_line_overlaps(forced_lines)
    timed_ratio = timed_lines / max(1, len(non_empty))
    word_ratio = timed_words / max(1, total_words)
    if timed_ratio < 0.7 or word_ratio < 0.7:
        return None
    if overlaps > max(2, int(len(non_empty) * 0.08)):
        return None

    metrics = {
        "forced_line_coverage": timed_ratio,
        "forced_word_coverage": word_ratio,
        "forced_line_overlaps": float(overlaps),
    }
    logger.info(
        "WhisperX forced alignment accepted: %.0f%% lines, %.0f%% words",
        timed_ratio * 100.0,
        word_ratio * 100.0,
    )
    return forced_lines, metrics
