"""LRC to Whisper segment mapping logic."""

import logging
from typing import List, Optional, Tuple

from ...models import Line, Word
from ...phonetic_utils import _phonetic_similarity
from ..alignment.timing_models import TranscriptionSegment
from .lyrics_whisper_map_helpers import (
    _build_whisper_word_list,
    _clamp_line_end_to_next_start,
    _find_best_segment_for_line,
    _line_duration_from_lrc,
    _map_line_words_to_segment,
    _norm_token,
    _resample_slots_to_line,
    _select_window_sequence,
    _select_window_words_for_line,
    _shift_words,
    _slots_from_sequence,
)
from .lyrics_whisper_map_weighting import (
    _apply_lrc_weighted_timing,
    _apply_weighted_slots_to_line,
    _whisper_durations_for_line,
)

logger = logging.getLogger(__name__)


def _create_lines_from_whisper(
    transcription: List[TranscriptionSegment],
) -> List[Line]:
    """Create Line objects directly from Whisper transcription."""
    lines: List[Line] = []
    for segment in transcription:
        if segment is None:
            continue
        words: List[Word] = []
        if segment.words:
            for w in segment.words:
                text = (w.text or "").strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        start_time=float(w.start),
                        end_time=float(w.end),
                        singer="",
                    )
                )
        else:
            tokens = [t for t in segment.text.strip().split() if t]
            if tokens:
                duration = max(segment.end - segment.start, 0.2)
                spacing = duration / len(tokens)
                for i, token in enumerate(tokens):
                    start = segment.start + i * spacing
                    end = start + spacing * 0.9
                    words.append(
                        Word(text=token, start_time=start, end_time=end, singer="")
                    )
        if not words:
            continue
        lines.append(Line(words=words))
    return lines


def _map_lrc_lines_to_whisper_segments(  # noqa: C901
    lines: List[Line],
    transcription: List[TranscriptionSegment],
    language: str,
    lrc_line_starts: Optional[List[float]] = None,
    min_similarity: float = 0.35,
    min_similarity_fallback: float = 0.2,
    max_time_offset: float = 4.0,
    lookahead: int = 6,
) -> Tuple[List[Line], int, List[str]]:  # noqa: C901
    """Map LRC lines onto Whisper segment timing without reordering."""
    if not lines or not transcription:
        return lines, 0, []

    sorted_segments = sorted(transcription, key=lambda s: s.start)
    all_words = _build_whisper_word_list(transcription)
    adjusted: List[Line] = []
    fixes = 0
    issues: List[str] = []
    seg_idx = 0
    last_end = None
    min_gap = 0.01
    prev_text = None

    for line_idx, line in enumerate(lines):
        if not line.words:
            adjusted.append(line)
            continue

        best_idx, best_sim, window_end = _find_best_segment_for_line(
            line,
            sorted_segments,
            seg_idx,
            lookahead,
            language,
            _phonetic_similarity,
        )

        text_norm = line.text.strip().lower() if line.text else ""
        gap_required = 0.21 if prev_text and text_norm == prev_text else min_gap

        desired_start = (
            lrc_line_starts[line_idx]
            if lrc_line_starts and line_idx < len(lrc_line_starts)
            else None
        )
        next_lrc_start = (
            lrc_line_starts[line_idx + 1]
            if lrc_line_starts and line_idx + 1 < len(lrc_line_starts)
            else None
        )
        if (
            best_idx is None
            or best_sim < min_similarity_fallback
            or (
                desired_start is not None
                and best_idx is not None
                and abs(sorted_segments[best_idx].start - desired_start)
                > max_time_offset
            )
        ):
            new_words: List[Word]
            window_fallback = False
            if desired_start is not None and next_lrc_start is not None and all_words:
                window_words = _select_window_words_for_line(
                    all_words,
                    line,
                    desired_start,
                    next_lrc_start,
                    language,
                    _phonetic_similarity,
                )
                if window_words:
                    if window_words[0].start is not None:
                        whisper_start = window_words[0].start
                        start_delta = desired_start - whisper_start
                        if start_delta > 0.4:
                            desired_start = max(
                                whisper_start, desired_start - min(start_delta, 0.8)
                            )
                        elif start_delta < -0.4:
                            desired_start = min(
                                whisper_start, desired_start + min(-start_delta, 2.5)
                            )
                    whisper_duration = None
                    if (
                        window_words[-1].end is not None
                        and window_words[0].start is not None
                    ):
                        whisper_duration = max(
                            window_words[-1].end - window_words[0].start, 0.5
                        )
                    elif (
                        window_words[-1].start is not None
                        and window_words[0].start is not None
                    ):
                        whisper_duration = max(
                            window_words[-1].start - window_words[0].start, 0.5
                        )
                    window_text = " ".join(w.text for w in window_words)
                    window_sim = _phonetic_similarity(line.text, window_text, language)
                    if window_sim >= min_similarity_fallback:
                        new_words, desired_start = _apply_lrc_weighted_timing(
                            line,
                            desired_start,
                            next_lrc_start,
                            lrc_line_starts,
                            line_idx,
                            all_words,
                            None,
                            language,
                            _phonetic_similarity,
                            line_duration_override=whisper_duration,
                        )
                        window_fallback = True
                        issues.append(
                            f"Used window-only mapping for line '{line.text[:30]}...' "
                            f"(segment offset {sorted_segments[best_idx].start - desired_start:+.2f}s)"
                            if best_idx is not None and desired_start is not None
                            else f"Used window-only mapping for line '{line.text[:30]}...'"
                        )
            if not window_fallback:
                new_words = list(line.words)
            if desired_start is not None and new_words:
                shift = desired_start - new_words[0].start_time
                new_words = _shift_words(new_words, shift)
            new_words = _clamp_line_end_to_next_start(
                line, new_words, lrc_line_starts, line_idx
            )
            if last_end is not None and new_words:
                if new_words[0].start_time < last_end + gap_required:
                    shift = (last_end + gap_required) - new_words[0].start_time
                    new_words = _shift_words(new_words, shift)
            if new_words:
                adjusted.append(Line(words=new_words, singer=line.singer))
                last_end = new_words[-1].end_time
                if window_fallback:
                    fixes += 1
            else:
                adjusted.append(line)
                last_end = line.end_time
            if not window_fallback:
                if best_sim < min_similarity_fallback and best_idx is not None:
                    issues.append(
                        f"Skipped Whisper mapping for line '{line.text[:30]}...' "
                        f"(sim={best_sim:.2f})"
                    )
                if (
                    desired_start is not None
                    and best_idx is not None
                    and abs(sorted_segments[best_idx].start - desired_start)
                    > max_time_offset
                ):
                    issues.append(
                        f"Skipped Whisper mapping for line '{line.text[:30]}...' "
                        f"(segment offset {sorted_segments[best_idx].start - desired_start:+.2f}s)"
                    )
            prev_text = text_norm
            continue

        if best_idx is None:
            adjusted.append(line)
            prev_text = text_norm
            continue
        seg = sorted_segments[best_idx]
        forced_fallback = False
        if last_end is not None and seg.start < last_end + gap_required:
            next_idx: Optional[int] = None
            best_future_idx: Optional[int] = None
            best_future_sim: Optional[float] = None
            for idx in range(best_idx, window_end):
                seg_candidate = sorted_segments[idx]
                if seg_candidate.start < last_end + gap_required:
                    continue
                next_idx = idx
                sim_candidate = _phonetic_similarity(
                    line.text, seg_candidate.text, language
                )
                if best_future_sim is None or sim_candidate > best_future_sim:
                    best_future_sim = sim_candidate
                    best_future_idx = idx
            if best_future_idx is not None and best_future_sim is not None:
                if best_future_sim >= min_similarity_fallback:
                    best_idx = best_future_idx
                    seg = sorted_segments[best_idx]
                else:
                    if next_idx is None:
                        adjusted.append(line)
                        prev_text = text_norm
                        continue
                    best_idx = next_idx
                    seg = sorted_segments[best_idx]
                    forced_fallback = True
        new_words = _map_line_words_to_segment(line, seg)

        if desired_start is not None and line.words:
            new_words, desired_start = _apply_lrc_weighted_timing(
                line,
                desired_start,
                next_lrc_start,
                lrc_line_starts,
                line_idx,
                all_words,
                seg,
                language,
                _phonetic_similarity,
            )

        if desired_start is not None and new_words:
            shift = desired_start - new_words[0].start_time
            new_words = _shift_words(new_words, shift)

        new_words = _clamp_line_end_to_next_start(
            line, new_words, lrc_line_starts, line_idx
        )

        if (
            last_end is not None
            and new_words
            and new_words[0].start_time < last_end + gap_required
        ):
            shift = (last_end + gap_required) - new_words[0].start_time
            new_words = _shift_words(new_words, shift)
        adjusted.append(Line(words=new_words, singer=line.singer))
        fixes += 1
        if best_sim < min_similarity:
            issues.append(
                f"Low similarity mapping for line '{line.text[:30]}...' (sim={best_sim:.2f})"
            )
        if forced_fallback:
            issues.append(
                f"Forced forward mapping for line '{line.text[:30]}...' to avoid overlap"
            )
        last_end = new_words[-1].end_time if new_words else last_end
        seg_idx = best_idx + 1 if best_idx is not None else seg_idx
        prev_text = text_norm

    return adjusted, fixes, issues


__all__ = [
    "_norm_token",
    "_build_whisper_word_list",
    "_select_window_sequence",
    "_slots_from_sequence",
    "_resample_slots_to_line",
    "_whisper_durations_for_line",
    "_apply_weighted_slots_to_line",
    "_shift_words",
    "_find_best_segment_for_line",
    "_map_line_words_to_segment",
    "_select_window_words_for_line",
    "_line_duration_from_lrc",
    "_apply_lrc_weighted_timing",
    "_clamp_line_end_to_next_start",
    "_create_lines_from_whisper",
    "_map_lrc_lines_to_whisper_segments",
]
