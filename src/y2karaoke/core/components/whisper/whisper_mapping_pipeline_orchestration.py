"""Top-level line orchestration for LRC-to-Whisper mapping pipeline."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_assignment_confidence import build_assignment_confidence_profile
from .whisper_dtw import _LineMappingContext
from .whisper_mapping_helpers import _SPEECH_BLOCK_GAP, _build_word_to_segment_index
from . import whisper_utils


def _parse_trace_line_range() -> tuple[int, int] | None:
    raw = os.environ.get("Y2K_TRACE_MAPPER_LINE_RANGE", "").strip()
    if not raw:
        return None
    try:
        start_s, end_s = raw.split("-", 1)
        start = int(start_s)
        end = int(end_s)
    except (TypeError, ValueError):
        return None
    if start <= 0 or end < start:
        return None
    return start, end


def _append_mapper_trace_row(
    rows: list[dict[str, Any]],
    *,
    ctx: _LineMappingContext,
    line: models.Line,
    line_index: int,
    text: str,
    line_anchor_time: float,
    line_shift: float,
    line_segment: Optional[int],
    assignment_profile,
    assigned_segs: Dict[int, int],
    lrc_index_by_loc: Dict[tuple[int, int], int],
    lrc_assignments: Dict[int, List[int]],
    line_matches: List[Tuple[int, Tuple[float, float]]],
    mapped_line: models.Line,
) -> None:
    assigned_words = []
    for word_idx, word in enumerate(line.words):
        lrc_idx = lrc_index_by_loc.get((line_index - 1, word_idx))
        assigned_indices = (
            lrc_assignments.get(lrc_idx, []) if lrc_idx is not None else []
        )
        assigned_words.append(
            {
                "word_index": word_idx,
                "text": word.text,
                "start": round(word.start_time, 3),
                "assigned": [
                    {
                        "index": wi,
                        "text": ctx.all_words[wi].text,
                        "start": round(ctx.all_words[wi].start, 3),
                        "end": round(ctx.all_words[wi].end, 3),
                        "segment": ctx.word_segment_idx.get(wi),
                    }
                    for wi in assigned_indices
                ],
            }
        )
    rows.append(
        {
            "line_index": line_index,
            "text": text,
            "line_anchor_time": round(line_anchor_time, 3),
            "line_shift": round(line_shift, 3),
            "line_segment": line_segment,
            "assignment_confidence": {
                "total_assigned": assignment_profile.total_assigned,
                "lexical_overlap_ratio": round(
                    assignment_profile.lexical_overlap_ratio, 4
                ),
                "placeholder_ratio": round(assignment_profile.placeholder_ratio, 4),
                "median_start_drift_sec": round(
                    assignment_profile.median_start_drift_sec, 4
                ),
                "max_start_drift_sec": round(assignment_profile.max_start_drift_sec, 4),
                "unique_segment_count": assignment_profile.unique_segment_count,
                "low_confidence": assignment_profile.low_confidence,
            },
            "assigned_segment_votes": dict(sorted(assigned_segs.items())),
            "assigned_words": assigned_words,
            "match_count": len(line_matches),
            "matches": [
                {
                    "word_index": word_idx,
                    "start": round(interval[0], 3),
                    "end": round(interval[1], 3),
                }
                for word_idx, interval in line_matches
            ],
            "mapped_start": (
                round(mapped_line.start_time, 3) if mapped_line.words else None
            ),
            "mapped_end": round(mapped_line.end_time, 3) if mapped_line.words else None,
        }
    )


def _maybe_write_mapper_trace(rows: list[dict[str, Any]]) -> None:
    trace_path = os.environ.get("Y2K_TRACE_MAPPER_DETAILS_JSON", "").strip()
    if not trace_path:
        return
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump({"lines": rows}, fh, indent=2)


def _line_override_segment_votes(
    *,
    line_idx: int,
    line: models.Line,
    lrc_index_by_loc: Dict[tuple[int, int], int],
    lrc_assignments: Dict[int, List[int]],
    word_segment_idx: Dict[int, int],
) -> Dict[int, int]:
    votes: Dict[int, int] = {}
    for word_idx in range(len(line.words)):
        lrc_idx = lrc_index_by_loc.get((line_idx, word_idx))
        if lrc_idx is None:
            continue
        for wi in lrc_assignments.get(lrc_idx, []):
            si = word_segment_idx.get(wi)
            if si is not None:
                votes[si] = votes.get(si, 0) + 1
    return votes


def _maybe_override_line_segment(
    *,
    ctx: _LineMappingContext,
    line: models.Line,
    line_segment: Optional[int],
    line_anchor_time: float,
    line_shift: float,
    votes: Dict[int, int],
    should_override_line_segment_fn: Callable[..., bool],
) -> tuple[Optional[int], float, float]:
    if not votes:
        return line_segment, line_anchor_time, line_shift
    override_seg = max(votes, key=votes.get)  # type: ignore[arg-type]
    should_override = line_segment != override_seg and should_override_line_segment_fn(
        current_segment=line_segment,
        override_segment=override_seg,
        override_hits=votes[override_seg],
        line_word_count=len(line.words),
        line_anchor_time=line_anchor_time,
        segments=ctx.segments,
    )
    if not should_override:
        return line_segment, line_anchor_time, line_shift
    line_segment = override_seg
    if ctx.segments and override_seg < len(ctx.segments):
        seg_start = whisper_utils._segment_start(ctx.segments[override_seg])
        line_anchor_time = max(seg_start, ctx.prev_line_end)
        line_shift = line_anchor_time - line.start_time
    return line_segment, line_anchor_time, line_shift


def _map_lrc_words_to_whisper(
    lines: List[models.Line],
    lrc_words: List[Dict],
    all_words: List[timing_models.TranscriptionWord],
    lrc_assignments: Dict[int, List[int]],
    language: str,
    segments: Sequence[Any],
    *,
    prepare_line_context_fn: Callable[..., Tuple[Optional[int], float, float]],
    should_override_line_segment_fn: Callable[..., bool],
    match_assigned_words_fn: Callable[..., None],
    fill_unmatched_gaps_fn: Callable[..., None],
    assemble_mapped_line_fn: Callable[..., models.Line],
    logger,
) -> Tuple[List[models.Line], int, float, set]:
    """Build mapped lines using Whisper timings based on LRC assignments."""
    segments = segments or []
    speech_blocks = whisper_utils._compute_speech_blocks(all_words)
    if speech_blocks:
        logger.debug(
            "Speech blocks: %d (gaps >= %.1fs)",
            len(speech_blocks),
            _SPEECH_BLOCK_GAP,
        )
    ctx = _LineMappingContext(
        all_words=all_words,
        segments=segments,
        word_segment_idx=_build_word_to_segment_index(all_words, segments),
        language=language,
        total_lrc_words=len(lrc_words),
        total_whisper_words=len(all_words),
        speech_blocks=speech_blocks,
    )
    lrc_index_by_loc = {
        (lw["line_idx"], lw["word_idx"]): idx for idx, lw in enumerate(lrc_words)
    }
    mapped_lines: List[models.Line] = []
    mapper_trace_range = _parse_trace_line_range()
    mapper_trace_rows: list[dict[str, Any]] = []

    for line_idx, line in enumerate(lines):
        if not line.words:
            mapped_lines.append(line)
            continue

        line_segment, line_anchor_time, line_shift = prepare_line_context_fn(ctx, line)
        assignment_profile = build_assignment_confidence_profile(
            line_idx=line_idx,
            line=line,
            lrc_index_by_loc=lrc_index_by_loc,
            lrc_assignments=lrc_assignments,
            all_words=ctx.all_words,
            word_segment_idx=ctx.word_segment_idx,
        )

        assigned_segs = _line_override_segment_votes(
            line_idx=line_idx,
            line=line,
            lrc_index_by_loc=lrc_index_by_loc,
            lrc_assignments=lrc_assignments,
            word_segment_idx=ctx.word_segment_idx,
        )
        line_segment, line_anchor_time, line_shift = _maybe_override_line_segment(
            ctx=ctx,
            line=line,
            line_segment=line_segment,
            line_anchor_time=line_anchor_time,
            line_shift=line_shift,
            votes=assigned_segs,
            should_override_line_segment_fn=should_override_line_segment_fn,
        )

        line_matches: List[Tuple[int, Tuple[float, float]]] = []
        line_match_intervals: Dict[int, Tuple[float, float]] = {}
        line_last_idx_ref: List[Optional[int]] = [None]

        match_assigned_words_fn(
            ctx,
            line_idx,
            line,
            lrc_index_by_loc,
            lrc_assignments,
            line_segment,
            line_anchor_time,
            line_shift,
            line_matches,
            line_match_intervals,
            line_last_idx_ref,
        )
        fill_unmatched_gaps_fn(
            ctx,
            line_idx,
            line,
            lrc_index_by_loc,
            lrc_assignments,
            line_segment,
            line_anchor_time,
            line_shift,
            line_matches,
            line_match_intervals,
            line_last_idx_ref,
        )
        mapped_line = assemble_mapped_line_fn(
            ctx,
            line_idx,
            line,
            line_matches,
            line_match_intervals,
            line_anchor_time,
            line_segment,
            line_last_idx_ref,
            lines[line_idx + 1].start_time if line_idx + 1 < len(lines) else None,
        )
        mapped_lines.append(mapped_line)
        if (
            mapper_trace_range is not None
            and mapper_trace_range[0] <= line_idx + 1 <= mapper_trace_range[1]
        ):
            _append_mapper_trace_row(
                mapper_trace_rows,
                ctx=ctx,
                line=line,
                line_index=line_idx + 1,
                text=line.text,
                line_anchor_time=line_anchor_time,
                line_shift=line_shift,
                line_segment=line_segment,
                assignment_profile=assignment_profile,
                assigned_segs=assigned_segs,
                lrc_index_by_loc=lrc_index_by_loc,
                lrc_assignments=lrc_assignments,
                line_matches=line_matches,
                mapped_line=mapped_line,
            )

    _maybe_write_mapper_trace(mapper_trace_rows)
    return mapped_lines, ctx.mapped_count, ctx.total_similarity, ctx.mapped_lines_set
