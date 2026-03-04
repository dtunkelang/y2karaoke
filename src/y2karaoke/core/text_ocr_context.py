"""OCR token-context normalization helpers."""

from __future__ import annotations

import re
from typing import List


def regularize_short_chant_alternation(tokens: List[str]) -> List[str]:
    """Stabilize short chant runs made only of 'oh'/'I' atoms."""
    if len(tokens) < 4 or len(tokens) > 10:
        return tokens

    lowered = [t.lower() for t in tokens]
    allowed = {"oh", "i"}
    if any(tok not in allowed for tok in lowered):
        return tokens
    if len(set(lowered)) < 2:
        return tokens

    has_adjacent_dup = any(lowered[i] == lowered[i - 1] for i in range(1, len(lowered)))
    if not has_adjacent_dup:
        return tokens

    start = lowered[0]
    other = "i" if start == "oh" else "oh"
    out: List[str] = []
    for i, _ in enumerate(tokens):
        atom = start if i % 2 == 0 else other
        if atom == "oh":
            out.append("Oh" if i == 0 else "oh")
        else:
            out.append("I")
    return out


def contextual_ocr_token_replacement(  # noqa: C901
    low: str, prev_low: str, next_low: str
) -> str | None:
    if low != "drinking" and re.fullmatch(r"d[a-z]{0,4}(?:inking|nking)", low):
        return "drinking"
    if low in {"ony", "om"} and prev_low == "come":
        return "on"

    if low == "lite" and prev_low == "for":
        return "life"
    if low == "thythm":
        return "rhythm"
    if low == "vou":
        return "you"
    if low == "bue":
        return "me"
    if low == "walls" and prev_low == "you":
        return "want"
    if low == "walls" and prev_low == "vou":
        return "want"

    if low == "corne" and next_low in {"me", "on", "baby", "with"}:
        return "come"
    if low == "cance" and (prev_low in {"on", "come"} or next_low in {"with", "me"}):
        return "dance"
    if low == "starlich" and prev_low in {"my", "youre", "you're"}:
        return "starlight"
    if low == "tonish":
        return "tonight"
    if low == "metonight" and prev_low == "with":
        return "tonight"
    if low == "youtre":
        return "you're"
    if low == "ctric":
        return "electric"
    if low == "cuting":
        return "cutting"

    if (
        low in {"l", "loh", "loll", "lohl", "lohlohlohl", "lohlohl"}
        and prev_low == "oh"
    ):
        return "I"
    if (
        low in {"loh", "loll", "lohl", "lohlohlohl", "lohlohl", "ohl"}
        and prev_low == "i"
    ):
        return "oh"
    if low == "l" and prev_low in {"i", "im", "i'm"}:
        return "oh"

    if low == "l" and next_low in {
        "may",
        "want",
        "know",
        "be",
        "love",
        "got",
        "dont",
        "don't",
        "could",
        "would",
        "do",
    }:
        return "I"

    if low == "loh":
        if prev_low == "in" and next_low == "with":
            return "love"
        if prev_low == "with" and next_low == "body":
            return "your"
        if prev_low == "your":
            return "body"

    return None
