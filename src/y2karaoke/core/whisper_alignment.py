"""Whisper-based alignment and timing correction logic."""

import logging
from typing import List, Optional, Tuple, Set

from .models import Line, Word
from .timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
    AudioFeatures,
)
from .phonetic_utils import _phonetic_similarity, _get_ipa
from .audio_analysis import _check_vocal_activity_in_range, _check_for_silence_in_range

logger = logging.getLogger(__name__)


def _enforce_monotonic_line_starts(
    lines: List[Line], min_gap: float = 0.01
) -> List[Line]:
    """Shift lines forward so start times are non-decreasing."""
    adjusted: List[Line] = []
    prev_start = None
    for line in lines:
        if not line.words:
            adjusted.append(line)
            continue

        start = line.start_time
        if prev_start is not None and start < prev_start:
            shift = (prev_start - start) + min_gap
            new_words = [
                Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in line.words
            ]
            line = Line(words=new_words, singer=line.singer)

        adjusted.append(line)
        prev_start = line.start_time

    return adjusted


def _scale_line_to_duration(line: Line, target_duration: float) -> Line:
    """Scale word timings so the line fits within target_duration."""
    if not line.words:
        return line

    start = line.start_time
    end = line.end_time
    duration = end - start
    if duration <= 0 or target_duration <= 0:
        return line

    scale = target_duration / duration
    new_words: List[Word] = []
    for w in line.words:
        rel_start = w.start_time - start
        rel_end = w.end_time - start
        new_start = start + rel_start * scale
        new_end = start + rel_end * scale
        new_words.append(
            Word(
                text=w.text,
                start_time=new_start,
                end_time=new_end,
                singer=w.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _enforce_non_overlapping_lines(
    lines: List[Line], min_gap: float = 0.02, min_duration: float = 0.2
) -> List[Line]:
    """Shift/scale lines to avoid overlaps while preserving order."""
    if not lines:
        return lines

    adjusted: List[Line] = []
    prev_end: Optional[float] = None
    # Precompute next starts for lookahead
    next_starts: List[Optional[float]] = [None] * len(lines)
    next_start = None
    for i in range(len(lines) - 1, -1, -1):
        next_starts[i] = next_start
        if lines[i].words:
            next_start = lines[i].start_time

    for i, line in enumerate(lines):
        if not line.words:
            adjusted.append(line)
            continue

        start = line.start_time
        if prev_end is not None and start < prev_end + min_gap:
            shift = (prev_end + min_gap) - start
            new_words = [
                Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in line.words
            ]
            line = Line(words=new_words, singer=line.singer)

        next_start = next_starts[i]
        if next_start is not None and line.end_time >= next_start - min_gap:
            target = max(next_start - min_gap - line.start_time, min_duration)
            if target < (line.end_time - line.start_time):
                line = _scale_line_to_duration(line, target)

        adjusted.append(line)
        prev_end = line.end_time

    return adjusted


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
            word_count = max(len(merged_words), 1)
            total_duration = max(seg.end - seg.start, 0.2)
            spacing = total_duration / word_count
            new_words = []
            for i, word in enumerate(merged_words):
                start = seg.start + i * spacing
                end = start + spacing * 0.9
                new_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
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
            word_count = max(len(merged_words), 1)
            total_duration = max(seg.end - seg.start, 0.2)
            spacing = total_duration / word_count
            new_words = []
            for i, word in enumerate(merged_words):
                start = seg.start + i * spacing
                end = start + spacing * 0.9
                new_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            merged_lines.append(Line(words=new_words, singer=line.singer))
            merged_count += 1
            idx += 2
            continue

        merged_lines.append(line)
        idx += 1

    return merged_lines, merged_count


def _retime_adjacent_lines_to_whisper_window(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.35,
    max_gap: float = 2.5,
    min_similarity_late: float = 0.25,
    max_late: float = 3.0,
    max_late_short: float = 6.0,
    max_time_window: float = 15.0,
    max_window_duration: Optional[float] = None,
    max_start_offset: Optional[float] = None,
) -> Tuple[List[Line], int]:
    """Retune adjacent lines into a single Whisper window without merging them."""
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

        combined_text = f"{line.text} {next_line.text}".strip()
        prev_end = None
        if idx > 0 and adjusted[idx - 1].words:
            prev_end = adjusted[idx - 1].end_time

        seg, combined_sim, _ = _find_best_whisper_segment(
            combined_text, line.start_time, sorted_segments, language, min_similarity
        )
        if seg is None or combined_sim < min_similarity:
            # Fallback: if the pair is late, try the nearest segment by time.
            nearest = None
            nearest_gap = None
            for cand in sorted_segments:
                gap = abs(cand.start - line.start_time)
                if gap > max_time_window:
                    continue
                if nearest_gap is None or gap < nearest_gap:
                    nearest_gap = gap
                    nearest = cand
            if nearest is None:
                continue
            if line.start_time - nearest.start < max_late:
                continue

            combined_sim = _phonetic_similarity(combined_text, nearest.text, language)
            seg = nearest

            # If the nearest segment is after the line start, try the prior segment instead.
            prior = None
            prior_gap = None
            for cand in sorted_segments:
                if (
                    cand.end <= line.start_time
                    and cand.start >= line.start_time - max_time_window
                ):
                    gap = line.start_time - cand.end
                    if prior_gap is None or gap < prior_gap:
                        prior_gap = gap
                        prior = cand
            if prior is not None and prior_gap is not None:
                prior_sim = _phonetic_similarity(combined_text, prior.text, language)
                if (
                    prior_gap <= max_late_short
                    and (line.start_time - prior.start) > max_late
                    and prior_sim >= min_similarity_late
                ):
                    seg = prior
                    combined_sim = prior_sim

            if combined_sim < min_similarity_late:
                continue
        if (
            max_window_duration is not None
            and (seg.end - seg.start) > max_window_duration
        ):
            continue
        if (
            max_start_offset is not None
            and abs(line.start_time - seg.start) > max_start_offset
        ):
            continue

        total_words = len(line.words) + len(next_line.words)
        if total_words <= 0:
            continue

        window_start = seg.start
        if prev_end is not None:
            window_start = max(window_start, prev_end + 0.01)
        if window_start >= seg.end:
            continue
        if (
            max_window_duration is not None
            and (seg.end - window_start) > max_window_duration
        ):
            continue

        total_duration = max(seg.end - window_start, 0.2)
        spacing = total_duration / total_words
        new_line_words = []
        new_next_words = []

        for i, word in enumerate(line.words):
            start = window_start + i * spacing
            end = start + spacing * 0.9
            new_line_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )

        for j, word in enumerate(next_line.words):
            idx_in_seg = len(line.words) + j
            start = window_start + idx_in_seg * spacing
            end = start + spacing * 0.9
            new_next_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )

        gap = new_next_words[0].start_time - new_line_words[-1].end_time
        if gap > max_gap:
            continue

        adjusted[idx] = Line(words=new_line_words, singer=line.singer)
        adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
        fixes += 1

    return adjusted, fixes


def _retime_adjacent_lines_to_segment_window(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.35,
    max_gap: float = 2.5,
    max_time_window: float = 15.0,
) -> Tuple[List[Line], int]:
    """Retune adjacent lines into a two-segment Whisper window."""
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

        combined_text = f"{line.text} {next_line.text}".strip()

        for s_idx in range(len(sorted_segments) - 1):
            seg_a = sorted_segments[s_idx]
            seg_b = sorted_segments[s_idx + 1]
            if seg_b.start - seg_a.end > 1.0:
                continue
            if abs(seg_a.start - line.start_time) > max_time_window:
                continue

            window_text = f"{seg_a.text} {seg_b.text}".strip()
            combined_sim = _phonetic_similarity(combined_text, window_text, language)
            if combined_sim < min_similarity:
                continue

            total_words = len(line.words) + len(next_line.words)
            if total_words <= 0:
                continue

            window_start = seg_a.start
            window_end = seg_b.end
            total_duration = max(window_end - window_start, 0.2)
            spacing = total_duration / total_words

            new_line_words = []
            new_next_words = []
            for i, word in enumerate(line.words):
                start = window_start + i * spacing
                end = start + spacing * 0.9
                new_line_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            for j, word in enumerate(next_line.words):
                idx_in_seg = len(line.words) + j
                start = window_start + idx_in_seg * spacing
                end = start + spacing * 0.9
                new_next_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )

            gap = new_next_words[0].start_time - new_line_words[-1].end_time
            if gap > max_gap:
                continue

            adjusted[idx] = Line(words=new_line_words, singer=line.singer)
            adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
            fixes += 1
            break

    return adjusted, fixes


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

        nearest = None
        nearest_gap = None
        for cand in sorted_segments:
            gap = abs(cand.start - line.start_time)
            if gap > max_time_window:
                continue
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest = cand
        seg_a = nearest
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

        nearest_next = None
        nearest_next_gap = None
        for cand in sorted_segments:
            gap = abs(cand.start - next_line.start_time)
            if gap > max_time_window:
                continue
            if nearest_next_gap is None or gap < nearest_next_gap:
                nearest_next_gap = gap
                nearest_next = cand
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

        nearest = None
        nearest_gap = None
        for cand in sorted_segments:
            gap = abs(cand.start - line.start_time)
            if gap > max_time_window:
                continue
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest = cand
        if nearest is None:
            continue

        late_by = next_line.start_time - nearest.end
        if late_by < 0 or late_by > max_late:
            continue

        min_start = line.end_time + 0.01
        # If the next line is short and starts late, retime both into the same segment.
        if len(next_line.words) <= 3 and late_by > 0:
            total_words = len(line.words) + len(next_line.words)
            if total_words <= 0:
                continue
            window_start = nearest.start
            if idx > 0 and adjusted[idx - 1].words:
                window_start = max(window_start, adjusted[idx - 1].end_time + 0.01)
            if window_start >= nearest.end:
                continue
            total_duration = max(nearest.end - window_start, 0.2)
            spacing = total_duration / total_words
            new_line_words = []
            new_next_words = []
            for i, word in enumerate(line.words):
                start = window_start + i * spacing
                end = start + spacing * 0.9
                new_line_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            for j, word in enumerate(next_line.words):
                idx_in_seg = len(line.words) + j
                start = window_start + idx_in_seg * spacing
                end = start + spacing * 0.9
                new_next_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            adjusted[idx] = Line(words=new_line_words, singer=line.singer)
            adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
        elif min_start >= nearest.end:
            # Not enough room: retime both lines into the segment window.
            total_words = len(line.words) + len(next_line.words)
            if total_words <= 0:
                continue
            window_start = nearest.start
            total_duration = max(nearest.end - window_start, 0.2)
            spacing = total_duration / total_words
            new_line_words = []
            new_next_words = []
            for i, word in enumerate(line.words):
                start = window_start + i * spacing
                end = start + spacing * 0.9
                new_line_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            for j, word in enumerate(next_line.words):
                idx_in_seg = len(line.words) + j
                start = window_start + idx_in_seg * spacing
                end = start + spacing * 0.9
                new_next_words.append(
                    Word(
                        text=word.text,
                        start_time=start,
                        end_time=end,
                        singer=word.singer,
                    )
                )
            adjusted[idx] = Line(words=new_line_words, singer=line.singer)
            adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
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

        nearest = None
        nearest_gap = None
        for cand in sorted_segments:
            gap = abs(cand.start - line.start_time)
            if gap > max_time_window:
                continue
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest = cand
        if nearest is None:
            continue

        late_by = next_line.start_time - nearest.end
        if late_by < 0 or late_by > max_late:
            continue

        total_words = len(line.words) + len(next_line.words)
        if total_words <= 0:
            continue

        window_start = nearest.start
        if idx > 0 and adjusted[idx - 1].words:
            window_start = max(window_start, adjusted[idx - 1].end_time + 0.01)
        if window_start >= nearest.end:
            continue

        total_duration = max(nearest.end - window_start, 0.2)
        spacing = total_duration / total_words
        new_line_words = []
        new_next_words = []
        for i, word in enumerate(line.words):
            start = window_start + i * spacing
            end = start + spacing * 0.9
            new_line_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
        for j, word in enumerate(next_line.words):
            idx_in_seg = len(line.words) + j
            start = window_start + idx_in_seg * spacing
            end = start + spacing * 0.9
            new_next_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )

        adjusted[idx] = Line(words=new_line_words, singer=line.singer)
        adjusted[idx + 1] = Line(words=new_next_words, singer=next_line.singer)
        fixes += 1

    return adjusted, fixes


def _pull_lines_near_segment_end(  # noqa: C901
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
        prev_end = None
        if idx > 0 and adjusted[idx - 1].words:
            prev_end = adjusted[idx - 1].end_time
        next_start = None
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_start = adjusted[idx + 1].start_time
        nearest = None
        nearest_late = None
        for cand in sorted_segments:
            if abs(cand.start - line.start_time) > max_time_window:
                continue
            late_by = line.start_time - cand.end
            if late_by < 0:
                continue
            if nearest_late is None or late_by < nearest_late:
                nearest_late = late_by
                nearest = cand
        if nearest is None:
            continue

        late_by = line.start_time - nearest.end
        if late_by < 0:
            continue

        allow_late = late_by <= max_late
        if not allow_late:
            word_count = len(line.words)
            # Short lines can be pulled if they're only modestly late.
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


def _clamp_repeated_line_duration(
    lines: List[Line],
    max_duration: float = 1.5,
    min_gap: float = 0.01,
) -> Tuple[List[Line], int]:
    """Clamp duration for immediately repeated lines so they don't hang too long."""
    if not lines:
        return lines, 0

    adjusted = list(lines)
    fixes = 0

    for idx in range(1, len(adjusted)):
        prev = adjusted[idx - 1]
        line = adjusted[idx]
        if not prev.words or not line.words:
            continue

        if prev.text.strip() != line.text.strip():
            continue

        duration = line.end_time - line.start_time
        if duration <= max_duration:
            continue

        new_end = line.start_time + max_duration
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_start = adjusted[idx + 1].start_time
            new_end = min(new_end, next_start - min_gap)
        if new_end <= line.start_time:
            continue

        total_duration = max(new_end - line.start_time, 0.2)
        word_count = max(len(line.words), 1)
        spacing = total_duration / word_count
        new_words = []
        for i, word in enumerate(line.words):
            start = line.start_time + i * spacing
            end = start + spacing * 0.9
            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
        adjusted[idx] = Line(words=new_words, singer=line.singer)
        fixes += 1

    return adjusted, fixes


def _merge_first_two_lines_if_segment_matches(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.8,
) -> Tuple[List[Line], bool]:
    """Merge the first two non-empty lines if Whisper sees them as one phrase."""
    if not lines or not segments:
        return lines, False

    first_idx = None
    second_idx = None
    for idx, line in enumerate(lines):
        if line.words:
            if first_idx is None:
                first_idx = idx
            else:
                second_idx = idx
                break

    if first_idx is None or second_idx is None:
        return lines, False

    first_line = lines[first_idx]
    second_line = lines[second_idx]
    combined_text = f"{first_line.text} {second_line.text}".strip()

    seg, sim, _ = _find_best_whisper_segment(
        combined_text,
        first_line.start_time,
        sorted(segments, key=lambda s: s.start),
        language,
        min_similarity,
    )

    if seg is None or sim < min_similarity:
        return lines, False

    merged_words = list(first_line.words) + list(second_line.words)
    word_count = max(len(merged_words), 1)
    total_duration = max(seg.end - seg.start, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(merged_words):
        start = seg.start + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )

    lines[first_idx] = Line(words=new_words, singer=first_line.singer)
    lines[second_idx] = Line(words=[], singer=second_line.singer)
    return lines, True


def _tighten_lines_to_whisper_segments(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.5,
) -> Tuple[List[Line], int]:
    """Pull consecutive lines to Whisper segment boundaries when phrased as one."""
    if not lines or not segments:
        return lines, 0

    adjusted = list(lines)
    fixes = 0
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx in range(1, len(adjusted)):
        prev_line = adjusted[idx - 1]
        line = adjusted[idx]
        if not prev_line.words or not line.words:
            continue

        line_text = line.text.strip()
        if not line_text:
            continue

        prev_seg, prev_sim, _ = _find_best_whisper_segment(
            prev_line.text,
            prev_line.start_time,
            sorted_segments,
            language,
            min_similarity,
        )
        seg, sim, _ = _find_best_whisper_segment(
            line_text, line.start_time, sorted_segments, language, min_similarity
        )

        if seg is None or prev_seg is None:
            continue

        # If both lines map to the same segment, pull line start to segment end.
        if seg is prev_seg:
            shift = seg.end - line.start_time
        else:
            # If line maps to next segment but starts far after it, pull to segment start.
            if line.start_time - seg.start <= 0:
                continue
            if seg.start - prev_seg.end > 2.5:
                continue
            if sim < min_similarity:
                continue
            shift = seg.start - line.start_time

        if shift >= -0.2:
            continue

        new_words = [
            Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        adjusted[idx] = Line(words=new_words, singer=line.singer)
        fixes += 1

    return adjusted, fixes


def _retime_line_to_segment(line: Line, seg: TranscriptionSegment) -> Line:
    """Retime a line's words evenly within a Whisper segment."""
    if not line.words:
        return line

    word_count = max(len(line.words), 1)
    total_duration = max(seg.end - seg.start, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(line.words):
        start = seg.start + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _retime_line_to_segment_with_min_start(
    line: Line, seg: TranscriptionSegment, min_start: float
) -> Line:
    """Retime a line within a segment, respecting a minimum start time."""
    start_base = max(seg.start, min_start)
    if start_base >= seg.end:
        return line

    word_count = max(len(line.words), 1)
    total_duration = max(seg.end - start_base, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(line.words):
        start = start_base + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _retime_line_to_window(line: Line, window_start: float, window_end: float) -> Line:
    """Retime a line's words evenly within a custom window."""
    if window_end <= window_start:
        return line

    if not line.words:
        return line

    word_count = max(len(line.words), 1)
    total_duration = max(window_end - window_start, 0.2)
    spacing = total_duration / word_count
    new_words = []
    for i, word in enumerate(line.words):
        start = window_start + i * spacing
        end = start + spacing * 0.9
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _pull_lines_to_best_segments(  # noqa: C901
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
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

        prev_end = None
        if idx > 0 and adjusted[idx - 1].words:
            prev_end = adjusted[idx - 1].end_time
        next_start = None
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_start = adjusted[idx + 1].start_time

        if word_count <= 6:
            nearest_start_seg = None
            nearest_start_gap = None
            for cand in sorted_segments:
                gap = abs(cand.start - line.start_time)
                if gap > 6.0:
                    continue
                if prev_end is not None and cand.start <= prev_end + min_gap:
                    continue
                if next_start is not None and cand.end >= next_start - min_gap:
                    continue
                if nearest_start_gap is None or gap < nearest_start_gap:
                    nearest_start_gap = gap
                    nearest_start_seg = cand
            if (
                nearest_start_seg is not None
                and nearest_start_seg.start <= line.start_time - 1.0
            ):
                adjusted[idx] = _retime_line_to_segment(line, nearest_start_seg)
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
                if prev_end is not None and cand.start <= prev_end + min_gap:
                    if abs(cand.start - prev_end) > 0.1:
                        continue
                if next_start is not None and cand.end >= next_start - min_gap:
                    continue
                if end_gap is None or gap_to_end < end_gap:
                    end_gap = gap_to_end
                    end_aligned = cand
            if end_aligned is not None:
                adjusted[idx] = _retime_line_to_segment(line, end_aligned)
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
            # Fallback: allow weaker match if line is significantly late
            seg, sim, _ = _find_best_whisper_segment(
                line.text,
                line.start_time,
                sorted_segments,
                language,
                min_similarity=0.3,
            )
            if seg is None:
                # Secondary fallback: align to a segment that ends right at the line start
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
            and (prev_end is None or seg.start >= prev_end + min_gap)
            and (next_start is None or seg.end <= next_start - min_gap)
        )

        if sim < local_min_similarity:
            if word_count > 4 and not late_and_ordered:
                continue
            if start_delta > -3.0:
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
            if prev_end is not None and seg.start <= prev_end + min_gap:
                continue
            if next_start is not None and seg.end >= next_start - min_gap:
                continue

        if prev_end is not None and seg.start <= prev_end + min_gap:
            continue
        if next_start is not None and seg.end >= next_start - min_gap:
            continue

        adjusted[idx] = _retime_line_to_segment(line, seg)
        fixed += 1

    return adjusted, fixed


def _drop_duplicate_lines(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float = 0.7,
) -> Tuple[List[Line], int]:
    """Drop adjacent duplicate lines when they map to the same Whisper segment."""
    if not lines:
        return lines, 0

    sorted_segments = sorted(segments, key=lambda s: s.start) if segments else []
    deduped: List[Line] = []
    dropped = 0

    for line in lines:
        if not deduped:
            deduped.append(line)
            continue

        prev = deduped[-1]
        if not prev.words or not line.words:
            deduped.append(line)
            continue

        if prev.text.strip() != line.text.strip():
            deduped.append(line)
            continue

        prev_seg, prev_sim, _ = _find_best_whisper_segment(
            prev.text, prev.start_time, sorted_segments, language, min_similarity
        )
        seg, sim, _ = _find_best_whisper_segment(
            line.text, line.start_time, sorted_segments, language, min_similarity
        )

        if (
            prev_seg is not None
            and seg is not None
            and prev_seg is seg
            and prev_sim >= min_similarity
            and sim >= min_similarity
        ):
            dropped += 1
            continue

        deduped.append(line)

    return deduped, dropped


def _drop_duplicate_lines_by_timing(
    lines: List[Line],
    max_gap: float = 0.2,
) -> Tuple[List[Line], int]:
    """Drop adjacent duplicate lines that overlap or are nearly contiguous."""
    if not lines:
        return lines, 0

    deduped: List[Line] = []
    dropped = 0

    for line in lines:
        if not deduped:
            deduped.append(line)
            continue

        prev = deduped[-1]
        if not prev.words or not line.words:
            deduped.append(line)
            continue

        if prev.text.strip() != line.text.strip():
            deduped.append(line)
            continue

        gap = line.start_time - prev.end_time
        if gap <= max_gap:
            dropped += 1
            continue

        deduped.append(line)

    return deduped, dropped


def _normalize_line_word_timings(
    lines: List[Line],
    min_word_duration: float = 0.05,
    min_gap: float = 0.01,
) -> List[Line]:
    """Ensure each line has monotonic word timings with valid durations."""
    normalized: List[Line] = []
    for line in lines:
        if not line.words:
            normalized.append(line)
            continue

        new_words = []
        last_end = None
        for word in line.words:
            start = float(word.start_time)
            end = float(word.end_time)

            if end < start:
                end = start + min_word_duration

            if last_end is not None and start < last_end:
                shift = (last_end - start) + min_gap
                start += shift
                end += shift

            if end < start:
                end = start + min_word_duration

            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
            last_end = end

        normalized.append(Line(words=new_words, singer=line.singer))

    return normalized


def _find_best_whisper_segment(
    line_text: str,
    line_start: float,
    sorted_segments: List[TranscriptionSegment],
    language: str,
    min_similarity: float,
) -> Tuple[Optional[TranscriptionSegment], float, float]:
    """Find best matching Whisper segment for a line."""
    best_segment = None
    best_similarity = 0.0
    best_offset = 0.0

    for seg in sorted_segments:
        if seg.start > line_start + 15 or seg.end < line_start - 15:
            continue

        similarity = _phonetic_similarity(line_text, seg.text, language)
        if similarity > best_similarity and similarity >= min_similarity:
            best_similarity = similarity
            best_segment = seg
            best_offset = seg.start - line_start

    return best_segment, best_similarity, best_offset


def _apply_offset_to_line(line: Line, offset: float) -> Line:
    """Apply timing offset to all words in a line."""
    new_words = [
        Word(
            text=word.text,
            start_time=word.start_time + offset,
            end_time=word.end_time + offset,
            singer=word.singer,
        )
        for word in line.words
    ]
    return Line(words=new_words, singer=line.singer)


def _calculate_drift_correction(
    recent_offsets: List[float], trust_threshold: float
) -> Optional[float]:
    """Calculate drift correction from recent offsets if consistent drift exists."""
    if len(recent_offsets) < 2:
        return None

    # Check recent lines for consistent drift
    recent_nonzero = [o for o in recent_offsets[-5:] if abs(o) > 0.5]
    if len(recent_nonzero) >= 2:
        avg_drift = sum(recent_nonzero) / len(recent_nonzero)
        if abs(avg_drift) > trust_threshold:
            return avg_drift
    return None


def _shift_line(line: Line, shift: float) -> Line:
    if not line.words:
        return line
    shifted_words = [
        Word(
            text=w.text,
            start_time=w.start_time + shift,
            end_time=w.end_time + shift,
            singer=w.singer,
        )
        for w in line.words
    ]
    return Line(words=shifted_words, singer=line.singer)


def _line_duration(line: Line) -> float:
    duration = line.end_time - line.start_time
    return duration if duration > 0.01 else 0.5


def _interpolate_unmatched_lines(
    mapped_lines: List[Line], matched_lines: Set[int]
) -> List[Line]:
    """Spread lines without matches between their nearest matched neighbors."""
    total = len(mapped_lines)
    prev_end = None
    idx = 0
    while idx < total:
        if idx in matched_lines:
            prev_end = mapped_lines[idx].end_time
            idx += 1
            continue
        run_start = idx
        while idx < total and idx not in matched_lines:
            idx += 1
        run_end = idx
        run_indices = list(range(run_start, run_end))
        if not run_indices:
            continue
        start_anchor = (
            prev_end if prev_end is not None else mapped_lines[run_start].start_time
        )
        next_anchor = (
            mapped_lines[run_end].start_time
            if run_end < total
            else start_anchor
            + sum(_line_duration(mapped_lines[i]) for i in run_indices)
            + 0.5
        )
        total_duration = sum(_line_duration(mapped_lines[i]) for i in run_indices)
        total_duration = max(total_duration, 0.5 * len(run_indices))
        available_gap = max(next_anchor - start_anchor, total_duration)
        duration_scale = available_gap / total_duration if total_duration > 0 else 1.0
        # Don't spread unmatched lines across a large instrumental gap;
        # keep them compact near the previous anchor.
        if duration_scale > 2.0:
            duration_scale = 2.0
        current = start_anchor
        for line_idx in run_indices:
            line = mapped_lines[line_idx]
            duration = _line_duration(line) * duration_scale
            line_shift = current - line.start_time
            mapped_lines[line_idx] = _shift_line(line, line_shift)
            current += duration + 0.01
        prev_end = current
    return mapped_lines


def _refine_unmatched_lines_with_onsets(
    mapped_lines: List[Line],
    matched_lines: Set[int],
    vocals_path: str,
) -> List[Line]:
    """Re-apply onset refinement to lines without Whisper word matches.

    Lines with no Whisper matches have correct segment-anchored start/end
    times but evenly-spaced words.  Audio onsets give better word
    boundaries even when Whisper can't identify the text.
    """
    unmatched = [
        i
        for i in range(len(mapped_lines))
        if i not in matched_lines and mapped_lines[i].words
    ]
    if not unmatched:
        return mapped_lines

    from .refine import refine_word_timing

    # Build a list of just the unmatched lines and refine them.
    subset = [mapped_lines[i] for i in unmatched]
    try:
        refined = refine_word_timing(subset, vocals_path)
    except Exception:
        return mapped_lines

    for idx, line_idx in enumerate(unmatched):
        mapped_lines[line_idx] = refined[idx]
    logger.debug(
        "Onset-refined %d unmatched line(s)",
        len(unmatched),
    )
    return mapped_lines


def align_hybrid_lrc_whisper(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    words: List[TranscriptionWord],
    language: str = "fra-Latn",
    trust_threshold: float = 1.0,
    correct_threshold: float = 1.5,
    min_similarity: float = 0.4,
) -> Tuple[List[Line], List[str]]:
    """Hybrid alignment: preserve good LRC timing, use Whisper for broken sections.

    Strategy:
    1. First pass: Match each LRC line to best Whisper segment, measure timing error
    2. Lines with small error (< trust_threshold): keep LRC timing
    3. Lines with large error (> correct_threshold): use Whisper timing
    4. Track cumulative drift for lines without good matches
    5. Apply word-level alignment only to lines that need correction

    Args:
        lines: LRC lines with word-level timing
        segments: Whisper segments (sentence level)
        words: Whisper words (for word-level refinement)
        language: Epitran language code
        trust_threshold: Keep LRC if timing error below this (seconds)
        correct_threshold: Use Whisper if timing error above this (seconds)
        min_similarity: Minimum similarity to consider a match

    Returns:
        Tuple of (aligned lines, list of corrections)
    """
    if not lines or not segments:
        return lines, []

    logger.debug(f"Pre-computing IPA for {len(words)} Whisper words...")
    for w in words:
        _get_ipa(w.text, language)

    aligned_lines: List[Line] = []
    corrections: List[str] = []
    recent_offsets: List[float] = []
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for line_idx, line in enumerate(lines):
        if not line.words:
            aligned_lines.append(line)
            continue

        line_text = " ".join(w.text for w in line.words)
        line_start = line.start_time

        best_segment, best_similarity, best_offset = _find_best_whisper_segment(
            line_text, line_start, sorted_segments, language, min_similarity
        )

        timing_error = abs(best_offset) if best_segment else float("inf")

        # Case 1: LRC timing is good - keep it
        if best_segment and timing_error < trust_threshold:
            aligned_lines.append(line)
            recent_offsets.append(0.0)
            continue

        # Case 2: LRC timing is significantly off - use Whisper
        if (
            best_segment
            and timing_error >= correct_threshold
            and best_similarity >= 0.5
        ):
            aligned_lines.append(_apply_offset_to_line(line, best_offset))
            corrections.append(
                f'Line {line_idx} shifted {best_offset:+.1f}s (similarity: {best_similarity:.0%}): "{line_text[:35]}..."'
            )
            recent_offsets.append(best_offset)
            continue

        # Case 3: Intermediate timing error - check for consistent drift
        if timing_error >= trust_threshold and timing_error < correct_threshold:
            if len(recent_offsets) >= 2 and all(
                abs(o) > 0.5 for o in recent_offsets[-2:]
            ):
                avg_drift = sum(recent_offsets[-3:]) / len(recent_offsets[-3:])
                if abs(avg_drift) > trust_threshold:
                    aligned_lines.append(_apply_offset_to_line(line, avg_drift))
                    corrections.append(
                        f'Line {line_idx} drift-corrected {avg_drift:+.1f}s: "{line_text[:35]}..."'
                    )
                    recent_offsets.append(avg_drift)
                    continue

            aligned_lines.append(line)
            recent_offsets.append(0.0)
            continue

        # Case 4: No good match - apply drift correction if available
        drift = _calculate_drift_correction(recent_offsets, trust_threshold)
        if drift is not None:
            aligned_lines.append(_apply_offset_to_line(line, drift))
            corrections.append(
                f'Line {line_idx} drift-corrected {drift:+.1f}s (no match): "{line_text[:35]}..."'
            )
            recent_offsets.append(drift)
        else:
            aligned_lines.append(line)
            recent_offsets.append(0.0)

    return aligned_lines, corrections


def _fix_ordering_violations(
    original_lines: List[Line],
    aligned_lines: List[Line],
    alignments: List[str],
) -> Tuple[List[Line], List[str]]:
    """Fix lines that were moved out of order by Whisper alignment.

    If a line's Whisper-aligned timing would cause it to overlap with
    or come before the previous line, revert to the original timing.
    """
    if not aligned_lines:
        return aligned_lines, alignments

    fixed_lines: List[Line] = []
    fixed_alignments: List[str] = []
    prev_end_time = 0.0
    prev_start_time = 0.0
    reverted_count = 0

    for i, (orig, aligned) in enumerate(zip(original_lines, aligned_lines)):
        if not aligned.words:
            fixed_lines.append(aligned)
            continue

        aligned_start = aligned.start_time

        # Check if this line would start before previous line (or end) with tolerance
        if (
            aligned_start < prev_start_time - 0.01
            or aligned_start < prev_end_time - 0.1
        ):
            # Revert to original timing
            fixed_lines.append(orig)
            if orig.words:
                prev_end_time = orig.end_time
                prev_start_time = orig.start_time
            reverted_count += 1
        else:
            # Keep the aligned timing
            fixed_lines.append(aligned)
            prev_end_time = aligned.end_time
            prev_start_time = aligned.start_time

    # Update alignments list (remove reverted ones)
    if reverted_count > 0:
        logger.debug(
            f"Reverted {reverted_count} Whisper alignments due to ordering violations"
        )
        # Filter alignments to only include successful ones
        fixed_alignments = [a for a in alignments if True]  # Keep all for now
        # Actually recount
        actual_corrections = len(alignments) - reverted_count
        fixed_alignments = (
            alignments[:actual_corrections] if actual_corrections > 0 else []
        )

    return fixed_lines, fixed_alignments if reverted_count > 0 else alignments


def _pull_lines_forward_for_continuous_vocals(
    lines: List[Line],
    audio_features: Optional[AudioFeatures],
    max_gap: float = 4.0,
) -> Tuple[List[Line], int]:
    """Pull lines earlier when a long gap contains continuous vocal activity."""
    fixes = 0
    if not lines or audio_features is None:
        return lines, fixes

    onset_times = audio_features.onset_times
    if onset_times is None or len(onset_times) == 0:
        return lines, fixes

    for idx in range(1, len(lines)):
        prev_line = lines[idx - 1]
        line = lines[idx]
        if not prev_line.words or not line.words:
            continue

        gap = line.start_time - prev_line.end_time
        if gap <= max_gap:
            continue

        activity = _check_vocal_activity_in_range(
            prev_line.end_time, line.start_time, audio_features
        )
        has_silence = _check_for_silence_in_range(
            prev_line.end_time,
            line.start_time,
            audio_features,
            min_silence_duration=0.5,
        )
        if activity <= 0.6 or has_silence:
            continue

        candidate_onsets = onset_times[
            (onset_times >= prev_line.end_time) & (onset_times <= line.start_time)
        ]
        if len(candidate_onsets) == 0:
            continue

        new_start = float(candidate_onsets[0])
        new_start = max(new_start, prev_line.end_time + 0.05)
        shift = new_start - line.start_time
        if shift > -0.3:
            continue

        new_words = [
            Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]

        lines[idx] = Line(words=new_words, singer=line.singer)
        fixes += 1

    return lines, fixes


def _fill_vocal_activity_gaps(
    whisper_words: List[TranscriptionWord],
    audio_features: AudioFeatures,
    threshold: float = 0.3,
    min_gap: float = 1.0,
    chunk_duration: float = 0.5,
    segments: Optional[List[TranscriptionSegment]] = None,
) -> Tuple[List[TranscriptionWord], Optional[List[TranscriptionSegment]]]:
    """Inject pseudo-words and segments where vocal activity is high but transcription missing.

    This helps DTW and hybrid alignment when Whisper misses words but vocals are audible.
    """
    if not whisper_words:
        return whisper_words, segments

    filled_words = list(whisper_words)
    filled_words.sort(key=lambda w: w.start)

    filled_segments = list(segments) if segments is not None else None
    if filled_segments:
        filled_segments.sort(key=lambda s: s.start)

    new_words = []
    new_segments = []

    def add_vocal_block(start, end):
        curr = start
        seg_words = []
        while curr + chunk_duration <= end:
            w = TranscriptionWord(
                start=curr,
                end=curr + chunk_duration,
                text="[VOCAL]",
                probability=0.0,
            )
            new_words.append(w)
            seg_words.append(w)
            curr += chunk_duration

        if seg_words and filled_segments is not None:
            new_segments.append(
                TranscriptionSegment(
                    start=seg_words[0].start,
                    end=seg_words[-1].end,
                    text="[VOCAL]",
                    words=seg_words,
                )
            )

    # 1. Check gap before first word
    vocal_start = audio_features.vocal_start
    if filled_words[0].start - vocal_start >= min_gap:
        activity = _check_vocal_activity_in_range(
            vocal_start, filled_words[0].start, audio_features
        )
        if activity > threshold:
            add_vocal_block(vocal_start, filled_words[0].start)

    # 2. Check gaps between words
    for i in range(len(filled_words) - 1):
        gap_start = filled_words[i].end
        gap_end = filled_words[i + 1].start

        if gap_end - gap_start >= min_gap:
            activity = _check_vocal_activity_in_range(
                gap_start, gap_end, audio_features
            )
            if activity > threshold:
                add_vocal_block(gap_start, gap_end)

    # 3. Check gap after last word
    vocal_end = audio_features.vocal_end
    if vocal_end - filled_words[-1].end >= min_gap:
        activity = _check_vocal_activity_in_range(
            filled_words[-1].end, vocal_end, audio_features
        )
        if activity > threshold:
            add_vocal_block(filled_words[-1].end, vocal_end)

    if new_words:
        logger.info(f"Vocal gap filler: injected {len(new_words)} [VOCAL] pseudo-words")
        filled_words.extend(new_words)
        filled_words.sort(key=lambda w: w.start)

        if filled_segments is not None and new_segments:
            filled_segments.extend(new_segments)
            filled_segments.sort(key=lambda s: s.start)

    return filled_words, filled_segments
