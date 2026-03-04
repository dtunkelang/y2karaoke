"""Functions for pulling and merging lyrics lines based on Whisper segments."""

import logging
from typing import List, Tuple

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
    nearest_segment_by_start,
    reflow_two_lines_to_segment,
    reflow_words_to_window,
)
from . import whisper_alignment_pull_targeted as _targeted_pull_helpers

logger = logging.getLogger(__name__)


def _segment_window_available(
    seg: TranscriptionSegment,
    prev_end: float | None,
    next_start: float | None,
    min_gap: float,
) -> bool:
    min_start = seg.start
    if prev_end is not None:
        min_start = max(min_start, prev_end + min_gap)

    max_end = seg.end
    if next_start is not None:
        max_end = min(max_end, next_start - min_gap)

    return (max_end - min_start) > 0.05


def _retime_line_to_segment_with_neighbors(
    line: Line,
    seg: TranscriptionSegment,
    prev_end: float | None,
    next_start: float | None,
    min_gap: float,
) -> Line | None:
    """Retime into a segment while respecting neighboring line boundaries."""
    min_start = seg.start
    if prev_end is not None:
        min_start = max(min_start, prev_end + min_gap)

    max_end = seg.end
    if next_start is not None:
        max_end = min(max_end, next_start - min_gap)

    if max_end - min_start <= 0.05:
        return None

    return _retime_line_to_window(line, min_start, max_end)


def _merge_lines_to_whisper_segments(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.55,
) -> Tuple[List[Line], int]:
    """Merge consecutive lines that map to the same Whisper segment."""
    if not lines or not segments:
        return lines, 0

    merged_lines: List[Line] = []
    merged_count = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.words:
            merged_lines.append(line)
            idx += 1
            continue

        if idx + 1 >= len(lines):
            merged_lines.append(line)
            break

        next_line = lines[idx + 1]
        if not next_line.words:
            merged_lines.append(line)
            idx += 1
            continue

        seg, sim, _ = _find_best_whisper_segment(
            line.text, line.start_time, sorted_segments, language, min_similarity
        )
        next_seg, next_sim, _ = _find_best_whisper_segment(
            next_line.text,
            next_line.start_time,
            sorted_segments,
            language,
            min_similarity,
        )

        if seg is None:
            merged_lines.append(line)
            idx += 1
            continue

        if next_line.text and next_line.text in line.text and sim >= 0.8:
            merged_words = list(line.words)
            new_words = reflow_words_to_window(merged_words, seg.start, seg.end)
            merged_lines.append(Line(words=new_words, singer=line.singer))
            merged_count += 1
            idx += 2
            continue

        combined_text = f"{line.text} {next_line.text}".strip()
        combined_seg, combined_sim, _ = _find_best_whisper_segment(
            combined_text, line.start_time, sorted_segments, language, min_similarity
        )
        if combined_seg is None:
            combined_sim = _phonetic_similarity(combined_text, seg.text, language)

        same_segment = next_seg is seg and next_sim >= min_similarity

        should_merge = False
        if same_segment and sim >= min_similarity:
            should_merge = True
        elif combined_sim >= min_similarity and sim >= min_similarity:
            # Accept merge when the combined line matches the segment better.
            if combined_sim >= max(sim, next_sim) + 0.05:
                should_merge = True
        elif (
            combined_sim >= min_similarity
            and seg.start <= line.start_time + 2.0
            and seg.end >= line.end_time
        ):
            should_merge = True
        elif (
            combined_seg is not None
            and combined_sim >= 0.8
            and combined_seg.start <= line.start_time + 3.0
            and (next_line.start_time - line.end_time) > 2.0
        ):
            seg = combined_seg
            should_merge = True

        if should_merge:
            merged_words = list(line.words) + list(next_line.words)
            new_words = reflow_words_to_window(merged_words, seg.start, seg.end)
            merged_lines.append(Line(words=new_words, singer=line.singer))
            merged_count += 1
            idx += 2
            continue

        merged_lines.append(line)
        idx += 1

    return merged_lines, merged_count


def _pull_next_line_into_segment_window(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.2,
    max_late: float = 6.0,
    min_gap: float = 0.05,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Pull a late-following line into the second segment of a window."""
    if not lines or len(segments) < 2:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        seg_a = nearest_segment_by_start(
            line.start_time, sorted_segments, max_time_window
        )
        if seg_a is None:
            continue

        seg_idx = sorted_segments.index(seg_a)
        if seg_idx + 1 >= len(sorted_segments):
            continue

        seg_b = sorted_segments[seg_idx + 1]
        if seg_b.start - seg_a.end > 1.0:
            continue

        if next_line.start_time < seg_b.end:
            continue

        late_by = next_line.start_time - seg_b.end
        if late_by > max_late:
            continue
        if 0 <= late_by <= 0.5:
            adjusted[idx + 1] = _retime_line_to_segment(next_line, seg_b)
            fixes += 1
            continue

        next_start = None
        if idx + 2 < len(adjusted) and adjusted[idx + 2].words:
            next_start = adjusted[idx + 2].start_time

        if next_start is not None and seg_b.end >= next_start - min_gap:
            continue

        nearest_next = nearest_segment_by_start(
            next_line.start_time, sorted_segments, max_time_window
        )
        if (
            nearest_next is not None
            and abs(nearest_next.start - seg_b.start) < 1e-6
            and abs(nearest_next.end - seg_b.end) < 1e-6
            and next_line.start_time > seg_b.end
        ):
            adjusted[idx + 1] = _retime_line_to_segment(next_line, seg_b)
            fixes += 1
            continue

        word_count = len(next_line.words)
        sim_b = _phonetic_similarity(next_line.text, seg_b.text, language)
        if sim_b < min_similarity and word_count > 4:
            continue

        adjusted[idx + 1] = _retime_line_to_segment(next_line, seg_b)
        fixes += 1

    return adjusted, fixes


def _pull_next_line_into_same_segment(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    max_late: float = 6.0,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Pull the next line into the same segment as the current line when late."""
    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        nearest = nearest_segment_by_start(
            line.start_time, sorted_segments, max_time_window
        )
        if nearest is None:
            continue

        late_by = next_line.start_time - nearest.end
        if late_by < 0 or late_by > max_late:
            continue

        min_start = line.end_time + 0.01
        # If the next line is short and starts late, retime both into the same segment.
        if len(next_line.words) <= 3 and late_by > 0:
            prev_end, _ = line_neighbors(adjusted, idx)
            reflowed = reflow_two_lines_to_segment(line, next_line, nearest, prev_end)
            if reflowed is None:
                continue
            adjusted[idx], adjusted[idx + 1] = reflowed
        elif min_start >= nearest.end:
            # Not enough room: retime both lines into the segment window.
            reflowed = reflow_two_lines_to_segment(line, next_line, nearest, None)
            if reflowed is None:
                continue
            adjusted[idx], adjusted[idx + 1] = reflowed
        else:
            adjusted[idx + 1] = _retime_line_to_segment_with_min_start(
                next_line, nearest, min_start
            )
        fixes += 1

    return adjusted, fixes


def _merge_short_following_line_into_segment(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    max_late: float = 6.0,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Merge a short following line into the same segment window as the current line."""
    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(len(adjusted) - 1):
        line = adjusted[idx]
        next_line = adjusted[idx + 1]
        if not line.words or not next_line.words:
            continue

        if len(next_line.words) > 3:
            continue

        nearest = nearest_segment_by_start(
            line.start_time, sorted_segments, max_time_window
        )
        if nearest is None:
            continue

        late_by = next_line.start_time - nearest.end
        if late_by < 0 or late_by > max_late:
            continue

        prev_end, _ = line_neighbors(adjusted, idx)
        reflowed = reflow_two_lines_to_segment(line, next_line, nearest, prev_end)
        if reflowed is None:
            continue
        adjusted[idx], adjusted[idx + 1] = reflowed
        fixes += 1

    return adjusted, fixes


def _pull_lines_near_segment_end(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    max_late: float = 0.5,
    max_late_short: float = 6.0,
    min_similarity: float = 0.35,
    min_short_duration: float = 0.35,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    return _targeted_pull_helpers.pull_lines_near_segment_end(
        lines,
        segments,
        language,
        max_late=max_late,
        max_late_short=max_late_short,
        min_similarity=min_similarity,
        min_short_duration=min_short_duration,
        max_time_window=max_time_window,
    )


def _pull_lines_to_best_segments(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.7,
    min_shift: float = 0.75,
    max_shift: float = 12.0,
    min_gap: float = 0.05,
) -> Tuple[List[Line], int]:
    return _targeted_pull_helpers.pull_lines_to_best_segments(
        lines,
        segments,
        language,
        segment_window_available_fn=_segment_window_available,
        retime_line_to_segment_with_neighbors_fn=_retime_line_to_segment_with_neighbors,
        min_similarity=min_similarity,
        min_shift=min_shift,
        max_shift=max_shift,
        min_gap=min_gap,
    )
