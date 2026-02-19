"""Overlap-resolution helpers for Whisper post-processing."""

from __future__ import annotations

from typing import List

from ... import models


def _resolve_line_overlaps(lines: List[models.Line]) -> List[models.Line]:  # noqa: C901
    """Ensure consecutive lines never overlap in time."""
    from ...models import Line as LineModel

    def _rebalance_degenerate_word_timings(
        line: models.Line,
        *,
        min_word_duration: float,
    ) -> models.Line:
        if not line.words or len(line.words) <= 1:
            return line
        if line.end_time <= line.start_time + (min_word_duration * len(line.words)):
            return line

        starts = [w.start_time for w in line.words]
        is_stacked = any(starts[i] >= starts[i + 1] for i in range(len(starts) - 1))
        if not is_stacked:
            return line

        span = line.end_time - line.start_time
        step = span / len(line.words)
        new_words: List[models.Word] = []
        for idx, w in enumerate(line.words):
            start = line.start_time + idx * step
            end = line.start_time + (idx + 1) * step
            if idx == len(line.words) - 1:
                end = line.end_time
            if end - start < min_word_duration:
                end = min(line.end_time, start + min_word_duration)
            new_words.append(
                models.Word(
                    text=w.text,
                    start_time=start,
                    end_time=end,
                    singer=w.singer,
                )
            )
        return LineModel(words=new_words, singer=line.singer)

    min_word_duration = 0.06
    resolved: List[models.Line] = []
    for line in lines:
        if not line.words:
            resolved.append(line)
            continue
        ordered = sorted(line.words, key=lambda w: (w.start_time, w.end_time))
        resolved.append(LineModel(words=ordered, singer=line.singer))
    for i in range(len(resolved) - 1):
        cur = resolved[i]
        if not cur.words:
            continue
        next_idx = None
        for j in range(i + 1, len(resolved)):
            if resolved[j].words:
                next_idx = j
                break
        if next_idx is None:
            continue
        nxt = resolved[next_idx]
        if cur.end_time > nxt.start_time:
            gap_point = nxt.start_time
            new_words: List[models.Word] = []
            for w in cur.words:
                if w.start_time >= gap_point:
                    new_words.append(
                        models.Word(
                            text=w.text,
                            start_time=max(
                                cur.start_time, gap_point - min_word_duration
                            ),
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                elif w.end_time > gap_point:
                    clipped_start = min(w.start_time, gap_point - min_word_duration)
                    new_words.append(
                        models.Word(
                            text=w.text,
                            start_time=clipped_start,
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                else:
                    new_words.append(w)
            if new_words:
                last = new_words[-1]
                if (last.end_time - last.start_time) < min_word_duration:
                    new_start = max(cur.start_time, last.end_time - min_word_duration)
                    if len(new_words) >= 2 and new_words[-2].end_time > new_start:
                        prev = new_words[-2]
                        new_words[-2] = models.Word(
                            text=prev.text,
                            start_time=prev.start_time,
                            end_time=max(
                                prev.start_time + min_word_duration, new_start
                            ),
                            singer=prev.singer,
                        )
                        new_start = max(new_words[-2].end_time, new_start)
                    new_words[-1] = models.Word(
                        text=last.text,
                        start_time=new_start,
                        end_time=last.end_time,
                        singer=last.singer,
                    )
            resolved[i] = LineModel(words=new_words, singer=cur.singer)
    for i, line in enumerate(resolved):
        if not line.words:
            continue
        resolved[i] = _rebalance_degenerate_word_timings(
            line,
            min_word_duration=min_word_duration,
        )
    return resolved
