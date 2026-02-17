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


class _LineMapper:
    """Helper class to encapsulate state for LRC-to-Whisper mapping."""

    def __init__(
        self,
        transcription: List[TranscriptionSegment],
        language: str,
        lrc_line_starts: Optional[List[float]],
        min_similarity: float,
        min_similarity_fallback: float,
        max_time_offset: float,
        lookahead: int,
    ):
        self.sorted_segments = sorted(transcription, key=lambda s: s.start)
        self.all_words = _build_whisper_word_list(transcription)
        self.language = language
        self.lrc_line_starts = lrc_line_starts
        self.min_similarity = min_similarity
        self.min_similarity_fallback = min_similarity_fallback
        self.max_time_offset = max_time_offset
        self.lookahead = lookahead

        # State
        self.seg_idx = 0
        self.last_end: Optional[float] = None
        self.prev_text: Optional[str] = None
        self.adjusted: List[Line] = []
        self.fixes = 0
        self.issues: List[str] = []
        self.min_gap = 0.01

    def process_line(self, line_idx: int, line: Line) -> None:  # noqa: C901
        if not line.words:
            self.adjusted.append(line)
            return

        best_idx, best_sim, window_end = _find_best_segment_for_line(
            line,
            self.sorted_segments,
            self.seg_idx,
            self.lookahead,
            self.language,
            _phonetic_similarity,
        )

        text_norm = line.text.strip().lower() if line.text else ""
        gap_required = (
            0.21 if self.prev_text and text_norm == self.prev_text else self.min_gap
        )

        desired_start = (
            self.lrc_line_starts[line_idx]
            if self.lrc_line_starts and line_idx < len(self.lrc_line_starts)
            else None
        )
        next_lrc_start = (
            self.lrc_line_starts[line_idx + 1]
            if self.lrc_line_starts and line_idx + 1 < len(self.lrc_line_starts)
            else None
        )

        should_fallback = (
            best_idx is None
            or best_sim < self.min_similarity_fallback
            or (
                desired_start is not None
                and best_idx is not None
                and abs(self.sorted_segments[best_idx].start - desired_start)
                > self.max_time_offset
            )
        )

        if should_fallback:
            self._handle_fallback(
                line,
                line_idx,
                desired_start,
                next_lrc_start,
                best_idx,
                best_sim,
                gap_required,
            )
            self.prev_text = text_norm
            return

        # Explicit check for None best_idx to satisfy type checker, though implied by should_fallback logic
        if best_idx is None:
            self.adjusted.append(line)
            self.prev_text = text_norm
            return

        self._handle_match(
            line,
            line_idx,
            desired_start,
            next_lrc_start,
            best_idx,
            best_sim,
            window_end,
            gap_required,
        )
        self.prev_text = text_norm

    def _handle_fallback(
        self,
        line: Line,
        line_idx: int,
        desired_start: Optional[float],
        next_lrc_start: Optional[float],
        best_idx: Optional[int],
        best_sim: float,
        gap_required: float,
    ) -> None:
        new_words: List[Word] = []
        window_fallback = False

        if desired_start is not None and next_lrc_start is not None and self.all_words:
            window_words = _select_window_words_for_line(
                self.all_words,
                line,
                desired_start,
                next_lrc_start,
                self.language,
                _phonetic_similarity,
            )
            if window_words:
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
                window_sim = _phonetic_similarity(line.text, window_text, self.language)
                if window_sim >= self.min_similarity_fallback:
                    new_words, desired_start = _apply_lrc_weighted_timing(
                        line,
                        desired_start,
                        next_lrc_start,
                        self.lrc_line_starts,
                        line_idx,
                        self.all_words,
                        None,
                        self.language,
                        _phonetic_similarity,
                        line_duration_override=whisper_duration,
                    )
                    window_fallback = True
                    offset_msg = (
                        f"(segment offset {self.sorted_segments[best_idx].start - desired_start:+.2f}s)"
                        if best_idx is not None and desired_start is not None
                        else ""
                    )
                    self.issues.append(
                        f"Used window-only mapping for line '{line.text[:30]}...' {offset_msg}".strip()
                    )

        if not window_fallback:
            new_words = list(line.words)

        if desired_start is not None and new_words:
            shift = desired_start - new_words[0].start_time
            new_words = _shift_words(new_words, shift)

        new_words = _clamp_line_end_to_next_start(
            line, new_words, self.lrc_line_starts, line_idx
        )

        if self.last_end is not None and new_words:
            if new_words[0].start_time < self.last_end + gap_required:
                shift = (self.last_end + gap_required) - new_words[0].start_time
                new_words = _shift_words(new_words, shift)

        if new_words:
            self.adjusted.append(Line(words=new_words, singer=line.singer))
            self.last_end = new_words[-1].end_time
            if window_fallback:
                self.fixes += 1
        else:
            self.adjusted.append(line)
            self.last_end = line.end_time

        if not window_fallback:
            if best_sim < self.min_similarity_fallback and best_idx is not None:
                self.issues.append(
                    f"Skipped Whisper mapping for line '{line.text[:30]}...' (sim={best_sim:.2f})"
                )
            if (
                desired_start is not None
                and best_idx is not None
                and abs(self.sorted_segments[best_idx].start - desired_start)
                > self.max_time_offset
            ):
                self.issues.append(
                    f"Skipped Whisper mapping for line '{line.text[:30]}...' "
                    f"(segment offset {self.sorted_segments[best_idx].start - desired_start:+.2f}s)"
                )

    def _handle_match(
        self,
        line: Line,
        line_idx: int,
        desired_start: Optional[float],
        next_lrc_start: Optional[float],
        best_idx: int,
        best_sim: float,
        window_end: int,
        gap_required: float,
    ) -> None:
        seg = self.sorted_segments[best_idx]
        forced_fallback = False

        if self.last_end is not None and seg.start < self.last_end + gap_required:
            next_idx: Optional[int] = None
            best_future_idx: Optional[int] = None
            best_future_sim: Optional[float] = None
            for idx in range(best_idx, window_end):
                seg_candidate = self.sorted_segments[idx]
                if seg_candidate.start < self.last_end + gap_required:
                    continue
                next_idx = idx
                sim_candidate = _phonetic_similarity(
                    line.text, seg_candidate.text, self.language
                )
                if best_future_sim is None or sim_candidate > best_future_sim:
                    best_future_sim = sim_candidate
                    best_future_idx = idx
            if best_future_idx is not None and best_future_sim is not None:
                if best_future_sim >= self.min_similarity_fallback:
                    best_idx = best_future_idx
                    seg = self.sorted_segments[best_idx]
                else:
                    if next_idx is None:
                        self.adjusted.append(line)
                        return
                    best_idx = next_idx
                    seg = self.sorted_segments[best_idx]
                    forced_fallback = True

        new_words = _map_line_words_to_segment(line, seg)

        if desired_start is not None and line.words:
            new_words, desired_start = _apply_lrc_weighted_timing(
                line,
                desired_start,
                next_lrc_start,
                self.lrc_line_starts,
                line_idx,
                self.all_words,
                seg,
                self.language,
                _phonetic_similarity,
            )

        if desired_start is not None and new_words:
            shift = desired_start - new_words[0].start_time
            new_words = _shift_words(new_words, shift)

        new_words = _clamp_line_end_to_next_start(
            line, new_words, self.lrc_line_starts, line_idx
        )

        if (
            self.last_end is not None
            and new_words
            and new_words[0].start_time < self.last_end + gap_required
        ):
            shift = (self.last_end + gap_required) - new_words[0].start_time
            new_words = _shift_words(new_words, shift)

        self.adjusted.append(Line(words=new_words, singer=line.singer))
        self.fixes += 1
        if best_sim < self.min_similarity:
            self.issues.append(
                f"Low similarity mapping for line '{line.text[:30]}...' (sim={best_sim:.2f})"
            )
        if forced_fallback:
            self.issues.append(
                f"Forced forward mapping for line '{line.text[:30]}...' to avoid overlap"
            )
        self.last_end = new_words[-1].end_time if new_words else self.last_end
        self.seg_idx = best_idx + 1


def _map_lrc_lines_to_whisper_segments(
    lines: List[Line],
    transcription: List[TranscriptionSegment],
    language: str,
    lrc_line_starts: Optional[List[float]] = None,
    min_similarity: float = 0.35,
    min_similarity_fallback: float = 0.2,
    max_time_offset: float = 4.0,
    lookahead: int = 6,
) -> Tuple[List[Line], int, List[str]]:
    """Map LRC lines onto Whisper segment timing without reordering."""
    if not lines or not transcription:
        return lines, 0, []

    mapper = _LineMapper(
        transcription,
        language,
        lrc_line_starts,
        min_similarity,
        min_similarity_fallback,
        max_time_offset,
        lookahead,
    )

    for line_idx, line in enumerate(lines):
        mapper.process_line(line_idx, line)

    return mapper.adjusted, mapper.fixes, mapper.issues


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
