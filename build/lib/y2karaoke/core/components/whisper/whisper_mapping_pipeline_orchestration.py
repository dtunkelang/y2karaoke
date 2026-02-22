"""Top-level line orchestration for LRC-to-Whisper mapping pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_dtw import _LineMappingContext
from .whisper_mapping_helpers import _SPEECH_BLOCK_GAP, _build_word_to_segment_index
from . import whisper_utils


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

    for line_idx, line in enumerate(lines):
        if not line.words:
            mapped_lines.append(line)
            continue

        line_segment, line_anchor_time, line_shift = prepare_line_context_fn(ctx, line)

        assigned_segs: Dict[int, int] = {}
        for word_idx in range(len(line.words)):
            lrc_idx = lrc_index_by_loc.get((line_idx, word_idx))
            if lrc_idx is None:
                continue
            for wi in lrc_assignments.get(lrc_idx, []):
                si = ctx.word_segment_idx.get(wi)
                if si is not None:
                    assigned_segs[si] = assigned_segs.get(si, 0) + 1
        if assigned_segs:
            override_seg = max(assigned_segs, key=assigned_segs.get)  # type: ignore[arg-type]
            if line_segment != override_seg and should_override_line_segment_fn(
                current_segment=line_segment,
                override_segment=override_seg,
                override_hits=assigned_segs[override_seg],
                line_word_count=len(line.words),
                line_anchor_time=line_anchor_time,
                segments=ctx.segments,
            ):
                line_segment = override_seg
                if ctx.segments and override_seg < len(ctx.segments):
                    seg_start = whisper_utils._segment_start(ctx.segments[override_seg])
                    line_anchor_time = max(seg_start, ctx.prev_line_end)
                    line_shift = line_anchor_time - line.start_time

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

    return mapped_lines, ctx.mapped_count, ctx.total_similarity, ctx.mapped_lines_set
