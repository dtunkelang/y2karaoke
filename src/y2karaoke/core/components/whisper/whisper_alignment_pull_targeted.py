"""Targeted line-pull heuristics for Whisper segment alignment."""

from typing import Callable, List, Tuple

from ...models import Line
from ..alignment.timing_models import TranscriptionSegment
from ...phonetic_utils import _phonetic_similarity
from .whisper_alignment_utils import _find_best_whisper_segment
from .whisper_alignment_retime import (
    _retime_line_to_segment,
    _retime_line_to_segment_with_min_start,
    _retime_line_to_window,
)
from .whisper_alignment_pull_helpers import (
    line_neighbors,
    nearest_prior_segment_by_end,
)


def pull_lines_near_segment_end(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    max_late: float = 0.5,
    max_late_short: float = 6.0,
    min_similarity: float = 0.35,
    min_short_duration: float = 0.35,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Pull lines that start just after a nearby segment end back into it."""
    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx, line in enumerate(adjusted):
        if not line.words:
            continue
        prev_end, next_start = line_neighbors(adjusted, idx)
        nearest_pair = nearest_prior_segment_by_end(
            line.start_time, sorted_segments, max_time_window
        )
        if nearest_pair is None:
            continue
        nearest, late_by = nearest_pair

        allow_late = late_by <= max_late
        if not allow_late:
            word_count = len(line.words)
            if word_count <= 6 and late_by <= max_late_short:
                allow_late = True
            else:
                sim = _phonetic_similarity(line.text, nearest.text, language)
                if (
                    word_count <= 3
                    and sim >= min_similarity
                    and late_by <= max_late_short
                ):
                    allow_late = True

        if not allow_late:
            continue

        min_start = (prev_end + 0.01) if prev_end is not None else nearest.start
        if len(line.words) <= 2:
            target_end = max(nearest.end, min_start + min_short_duration)
            if next_start is not None:
                target_end = min(target_end, next_start - 0.01)
            adjusted[idx] = _retime_line_to_window(line, min_start, target_end)
        else:
            adjusted[idx] = _retime_line_to_segment_with_min_start(
                line, nearest, min_start
            )
        fixes += 1

    return adjusted, fixes


def pull_lines_to_best_segments(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    segment_window_available_fn: Callable[
        [TranscriptionSegment, float | None, float | None, float], bool
    ],
    retime_line_to_segment_with_neighbors_fn: Callable[
        [Line, TranscriptionSegment, float | None, float | None, float], Line | None
    ],
    min_similarity: float = 0.7,
    min_shift: float = 0.75,
    max_shift: float = 12.0,
    min_gap: float = 0.05,
) -> Tuple[List[Line], int]:
    """Pull lines toward their best Whisper segment when alignment drift is large."""
    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixed = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx, line in enumerate(adjusted):
        if not line.words:
            continue

        word_count = len(line.words)
        local_min_similarity = min_similarity
        if word_count <= 6:
            local_min_similarity = min(0.6, min_similarity)

        prev_end, next_start = line_neighbors(adjusted, idx)

        if word_count <= 6:
            nearest_start_seg = None
            nearest_start_gap = None
            for cand in sorted_segments:
                gap = abs(cand.start - line.start_time)
                if gap > 6.0:
                    continue
                if not segment_window_available_fn(cand, prev_end, next_start, min_gap):
                    continue
                if nearest_start_gap is None or gap < nearest_start_gap:
                    nearest_start_gap = gap
                    nearest_start_seg = cand
            if (
                nearest_start_seg is not None
                and nearest_start_seg.start <= line.start_time - 1.0
            ):
                retimed = retime_line_to_segment_with_neighbors_fn(
                    line, nearest_start_seg, prev_end, next_start, min_gap
                )
                if retimed is not None:
                    adjusted[idx] = retimed
                    fixed += 1
                continue

        if word_count <= 4:
            end_aligned = None
            end_gap = None
            for cand in sorted_segments:
                gap_to_end = line.start_time - cand.end
                if gap_to_end < 0 or gap_to_end > 0.75:
                    continue
                if line.start_time - cand.start < 2.5:
                    continue
                if not segment_window_available_fn(cand, prev_end, next_start, min_gap):
                    continue
                if end_gap is None or gap_to_end < end_gap:
                    end_gap = gap_to_end
                    end_aligned = cand
            if end_aligned is not None:
                chosen_seg = end_aligned
                end_aligned_sim = _phonetic_similarity(
                    line.text, end_aligned.text, language
                )
                best_prior = None
                best_prior_sim = end_aligned_sim
                for cand in sorted_segments:
                    if cand is end_aligned or cand.end > line.start_time:
                        continue
                    if cand.start >= end_aligned.start:
                        continue
                    if not segment_window_available_fn(
                        cand, prev_end, next_start, min_gap
                    ):
                        continue
                    cand_sim = _phonetic_similarity(line.text, cand.text, language)
                    if cand_sim > best_prior_sim:
                        best_prior_sim = cand_sim
                        best_prior = cand
                if best_prior is not None and best_prior_sim >= max(
                    0.3, end_aligned_sim + 0.2
                ):
                    chosen_seg = best_prior

                retimed = retime_line_to_segment_with_neighbors_fn(
                    line, chosen_seg, prev_end, next_start, min_gap
                )
                if retimed is not None:
                    adjusted[idx] = retimed
                    fixed += 1
                continue

        seg, sim, _ = _find_best_whisper_segment(
            line.text,
            line.start_time,
            sorted_segments,
            language,
            local_min_similarity,
        )
        if seg is None or sim < local_min_similarity:
            seg, sim, _ = _find_best_whisper_segment(
                line.text,
                line.start_time,
                sorted_segments,
                language,
                min_similarity=0.3,
            )
            if seg is None:
                if word_count > 4:
                    continue
                nearest = None
                nearest_gap = None
                for cand in sorted_segments:
                    gap_to_end = line.start_time - cand.end
                    if gap_to_end < 0 or gap_to_end > 0.75:
                        continue
                    if line.start_time - cand.start < 2.5:
                        continue
                    if nearest_gap is None or gap_to_end < nearest_gap:
                        nearest_gap = gap_to_end
                        nearest = cand
                if nearest is None:
                    continue
                seg = nearest
                sim = 0.0

        start_delta = seg.start - line.start_time
        end_delta = seg.end - line.end_time
        if abs(start_delta) < min_shift and abs(end_delta) < min_shift:
            continue

        late_and_ordered = (
            sim >= 0.3
            and start_delta <= -1.0
            and segment_window_available_fn(seg, prev_end, next_start, min_gap)
        )

        if sim < local_min_similarity:
            if word_count > 4 and not late_and_ordered:
                continue
            if late_and_ordered:
                if start_delta > -1.0:
                    continue
            elif start_delta > -3.0:
                continue

        if sim < local_min_similarity and word_count <= 6:
            if -3.0 <= start_delta <= -1.0:
                if prev_end is not None and seg.start <= prev_end + min_gap:
                    continue
                if next_start is not None and seg.end >= next_start - min_gap:
                    continue
                adjusted[idx] = _retime_line_to_segment(line, seg)
                fixed += 1
                continue

        if abs(start_delta) > max_shift:
            continue

        if sim < local_min_similarity:
            if not segment_window_available_fn(seg, prev_end, next_start, min_gap):
                continue

        retimed = retime_line_to_segment_with_neighbors_fn(
            line, seg, prev_end, next_start, min_gap
        )
        if retimed is None:
            continue
        adjusted[idx] = retimed
        fixed += 1

    return adjusted, fixed
