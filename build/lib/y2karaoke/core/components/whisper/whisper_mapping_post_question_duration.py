"""Question-line duration helpers for Whisper repetition post-processing."""

from __future__ import annotations

from typing import List

from ... import models


def _enforce_min_duration_for_short_question_lines(  # noqa: C901
    lines: List[models.Line],
    *,
    min_duration: float = 1.0,
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Prevent short question lines from flashing too quickly."""
    if not lines:
        return lines
    adjusted = list(lines)
    for idx, line in enumerate(adjusted):
        if not line.words or not line.text.strip().endswith("?"):
            continue
        if len(line.words) > 6:
            continue
        duration = line.end_time - line.start_time
        if duration >= min_duration:
            continue
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )
        target_end = line.start_time + min_duration
        if next_start is not None:
            target_end = min(target_end, next_start - min_gap)
            if target_end <= line.end_time + 0.05 and idx + 1 < len(adjusted):
                next_line = adjusted[idx + 1]
                if next_line.words and next_line.text.strip().endswith("?"):
                    need = min_duration - duration
                    if need > 0:
                        after_next_start = (
                            adjusted[idx + 2].start_time
                            if idx + 2 < len(adjusted) and adjusted[idx + 2].words
                            else None
                        )
                        available = (
                            max(0.0, after_next_start - next_line.end_time)
                            if after_next_start is not None
                            else 0.0
                        )
                        shift = min(need, available)
                        if shift > 0.05:
                            j = idx + 1
                            while j < len(adjusted):
                                run_line = adjusted[j]
                                if not run_line.words:
                                    break
                                if j > idx + 1 and not run_line.text.strip().endswith(
                                    "?"
                                ):
                                    break
                                shifted_words = [
                                    models.Word(
                                        text=w.text,
                                        start_time=w.start_time + shift,
                                        end_time=w.end_time + shift,
                                        singer=w.singer,
                                    )
                                    for w in run_line.words
                                ]
                                adjusted[j] = models.Line(
                                    words=shifted_words, singer=run_line.singer
                                )
                                j += 1
                            next_start = adjusted[idx + 1].start_time
                            target_end = min(
                                line.start_time + min_duration, next_start - min_gap
                            )
        if target_end <= line.end_time + 0.05:
            continue
        spacing = (target_end - line.start_time) / len(line.words)
        new_words: List[models.Word] = []
        for w_idx, w in enumerate(line.words):
            ws = line.start_time + w_idx * spacing
            we = ws + spacing * 0.9
            if w_idx == len(line.words) - 1:
                we = target_end
            new_words.append(
                models.Word(
                    text=w.text,
                    start_time=ws,
                    end_time=we,
                    singer=w.singer,
                )
            )
        adjusted[idx] = models.Line(words=new_words, singer=line.singer)
    return adjusted
