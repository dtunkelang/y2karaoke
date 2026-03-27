"""Shared text and feature helpers for Whisper alignment experiments."""

from __future__ import annotations

import os
import re
from typing import List

from ... import models
from ..alignment import timing_models


def enable_low_support_onset_reanchor() -> bool:
    return os.getenv("Y2K_WHISPER_ENABLE_LOW_SUPPORT_ONSET_REANCHOR", "0") == "1"


def enable_repeat_cadence_reanchor() -> bool:
    return os.getenv("Y2K_WHISPER_ENABLE_REPEAT_CADENCE_REANCHOR", "0") == "1"


def enable_restored_run_onset_shift() -> bool:
    return os.getenv("Y2K_WHISPER_ENABLE_RESTORED_RUN_ONSET_SHIFT", "0") == "1"


def count_non_vocal_words_near_time(
    words: List[timing_models.TranscriptionWord],
    center_time: float,
    *,
    window_sec: float = 1.0,
) -> int:
    lo = center_time - window_sec
    hi = center_time + window_sec
    count = 0
    for word in words:
        if word.text == "[VOCAL]":
            continue
        if lo <= word.start <= hi:
            count += 1
    return count


def normalized_prefix_tokens(line: models.Line, *, limit: int = 3) -> list[str]:
    return [
        re.sub(r"[^a-z]+", "", w.text.lower())
        for w in line.words[:limit]
        if re.sub(r"[^a-z]+", "", w.text.lower())
    ]


def normalized_tokens(line: models.Line) -> list[str]:
    return [
        re.sub(r"[^a-z]+", "", w.text.lower())
        for w in line.words
        if re.sub(r"[^a-z]+", "", w.text.lower())
    ]


def line_text_key(line: models.Line) -> str:
    return " ".join(normalized_tokens(line))
