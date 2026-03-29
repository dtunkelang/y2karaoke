from __future__ import annotations

from y2karaoke.core import models
from y2karaoke.core.components.whisper import (
    whisper_forced_repeated_short_repairs as module,
)


def _line(text: str, start: float, end: float) -> models.Line:
    words = text.split()
    step = (end - start) / max(len(words), 1)
    built = []
    cursor = start
    for word in words:
        nxt = cursor + step
        built.append(
            models.Word(
                text=word,
                start_time=cursor,
                end_time=nxt,
                singer="",
            )
        )
        cursor = nxt
    return models.Line(words=built, singer="")


def test_restore_leading_repeated_short_line_tails_repairs_clocks_shape() -> None:
    baseline = [
        _line("You are", 0.859, 6.512),
        _line("You are", 8.545, 13.727),
        _line("Confusion that never stops", 29.123, 31.782),
    ]
    forced = [
        _line("You are", 0.81, 6.289),
        _line("You are", 8.295, 9.738),
        _line("Confusion that never stops", 29.114, 32.0),
    ]

    repaired, count = module.restore_leading_repeated_short_line_tails_from_baseline(
        baseline,
        forced,
    )

    assert count == 1
    assert repaired[0].start_time == forced[0].start_time
    assert repaired[0].end_time == baseline[0].end_time
    assert repaired[1].start_time == forced[1].start_time


def test_restore_leading_repeated_short_line_tails_skips_unsafe_gap() -> None:
    baseline = [
        _line("You are", 0.859, 6.512),
        _line("You are", 6.7, 9.0),
        _line("Confusion that never stops", 12.0, 15.0),
    ]
    forced = [
        _line("You are", 0.81, 6.289),
        _line("You are", 6.85, 7.8),
        _line("Confusion that never stops", 12.0, 15.0),
    ]

    repaired, count = module.restore_leading_repeated_short_line_tails_from_baseline(
        baseline,
        forced,
    )

    assert count == 0
    assert repaired[0].end_time == forced[0].end_time
