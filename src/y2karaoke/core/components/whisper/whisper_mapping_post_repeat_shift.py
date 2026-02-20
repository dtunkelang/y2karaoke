"""Repeated-line shift and monotonic-start helpers for Whisper post-mapping."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ... import models
from ..alignment import timing_models


def _shift_repeated_lines_to_next_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    is_placeholder_whisper_token_fn: Callable[[str], bool],
) -> List[models.Line]:
    """Ensure repeated lines reserve later Whisper words when they reappear."""
    adjusted_lines: List[models.Line] = []
    last_idx_by_text: Dict[str, int] = {}
    last_end_time: Dict[str, float] = {}
    lexical_indices = [
        i
        for i, ww in enumerate(all_words)
        if not is_placeholder_whisper_token_fn(ww.text)
    ]

    for idx, line in enumerate(mapped_lines):
        if not line.words:
            adjusted_lines.append(line)
            continue

        text_norm = line.text.strip().lower() if getattr(line, "text", "") else ""
        prev_idx = last_idx_by_text.get(text_norm)
        prev_end = last_end_time.get(text_norm)
        assigned_end_idx: Optional[int] = None

        if prev_idx is not None and prev_end is not None:
            required_time = max(prev_end + 0.4, line.start_time)
            start_idx = next(
                (
                    wi
                    for wi in lexical_indices
                    if wi > prev_idx and all_words[wi].start >= required_time
                ),
                None,
            )
            if start_idx is None:
                start_idx = next((wi for wi in lexical_indices if wi > prev_idx), None)
            max_repeat_jump = 4.0
            if start_idx is not None and (
                all_words[start_idx].start - prev_end > max_repeat_jump
            ):
                start_idx = None
            if start_idx is not None:
                adjusted_words: List[models.Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words.append(
                        models.Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = models.Line(words=adjusted_words, singer=line.singer)
                assigned_end_idx = min(
                    start_idx + len(line.words) - 1, len(all_words) - 1
                )

        if (
            assigned_end_idx is None
            and prev_end is not None
            and len(line.words) <= 5
            and adjusted_lines
            and text_norm
            and adjusted_lines[-1].words
            and adjusted_lines[-1].text.strip().lower() == text_norm
        ):
            gap = line.start_time - prev_end
            if 1.0 < gap <= 4.0:
                duration = max(0.25, min(line.end_time - line.start_time, 1.8))
                next_start = (
                    mapped_lines[idx + 1].start_time
                    if idx + 1 < len(mapped_lines) and mapped_lines[idx + 1].words
                    else None
                )
                target_start = prev_end + 0.18
                if next_start is not None:
                    target_start = min(
                        target_start,
                        max(prev_end + 0.05, next_start - 0.05 - duration),
                    )
                if target_start < line.start_time - 0.2:
                    shift = target_start - line.start_time
                    shifted_words = [
                        models.Word(
                            text=w.text,
                            start_time=w.start_time + shift,
                            end_time=w.end_time + shift,
                            singer=w.singer,
                        )
                        for w in line.words
                    ]
                    line = models.Line(words=shifted_words, singer=line.singer)

        adjusted_lines.append(line)
        if line.words:
            if assigned_end_idx is None:
                assigned_end_idx = next(
                    (
                        wi
                        for wi in lexical_indices
                        if abs(all_words[wi].start - line.words[-1].start_time) < 0.05
                    ),
                    len(all_words) - 1,
                )
            last_idx_by_text[text_norm] = assigned_end_idx
            last_end_time[text_norm] = line.end_time

    return adjusted_lines


def _enforce_monotonic_line_starts_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
) -> List[models.Line]:
    """Ensure line starts are monotonic by shifting backwards lines forward."""
    prev_start = None
    prev_end = None
    monotonic_lines: List[models.Line] = []
    for line in mapped_lines:
        if not line.words:
            monotonic_lines.append(line)
            continue

        if prev_start is not None and line.start_time < prev_start:
            required_time = (prev_end or line.start_time) + 0.01
            start_idx = next(
                (idx for idx, ww in enumerate(all_words) if ww.start >= required_time),
                None,
            )
            if start_idx is not None and (
                all_words[start_idx].start - required_time <= 10.0
            ):
                adjusted_words_2: List[models.Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words_2.append(
                        models.Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = models.Line(words=adjusted_words_2, singer=line.singer)
            else:
                shift = required_time - line.start_time
                shifted_words: List[models.Word] = [
                    models.Word(
                        text=w.text,
                        start_time=w.start_time + shift,
                        end_time=w.end_time + shift,
                        singer=w.singer,
                    )
                    for w in line.words
                ]
                line = models.Line(words=shifted_words, singer=line.singer)

        monotonic_lines.append(line)
        if line.words:
            prev_start = line.start_time
            prev_end = line.end_time

    return monotonic_lines
