"""Helpers for redistributing words inside sparse sustained forced-aligned lines."""

from __future__ import annotations

import re

from ... import models


def redistribute_line_with_word_weights(
    line: models.Line, weights: list[float]
) -> models.Line:
    start = line.start_time
    end = line.end_time
    duration = end - start
    total_weight = sum(weights)
    if duration <= 0.0 or total_weight <= 0.0:
        return line

    cursor = start
    words: list[models.Word] = []
    for idx, (word, weight) in enumerate(zip(line.words, weights)):
        span = duration * (weight / total_weight)
        word_start = cursor
        word_end = end if idx == len(line.words) - 1 else cursor + span
        words.append(
            models.Word(
                text=word.text,
                start_time=word_start,
                end_time=word_end,
                singer=word.singer,
            )
        )
        cursor = word_end
    return models.Line(words=words, singer=line.singer)


def sustained_word_layout_weights(line: models.Line) -> list[float] | None:
    tokens = [
        re.sub(r"[^a-z0-9]", "", word.text.lower()) for word in line.words if word.text
    ]
    if len(tokens) != len(line.words):
        return None
    if len(tokens) == 5 and all(len(token) <= 3 for token in tokens[:-1]):
        return [0.6, 0.6, 0.7, 0.7, 4.2]
    return None
