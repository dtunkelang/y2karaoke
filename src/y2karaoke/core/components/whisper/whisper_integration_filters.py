"""Filtering helpers for Whisper integration pipeline."""

from __future__ import annotations

from typing import List

from ..alignment import timing_models


def _filter_low_confidence_whisper_words(
    words: List[timing_models.TranscriptionWord],
    threshold: float,
    *,
    min_keep_ratio: float = 0.6,
    min_keep_words: int = 20,
) -> List[timing_models.TranscriptionWord]:
    """Drop low-confidence Whisper words when enough confident words remain."""
    if not words:
        return words
    if threshold <= 0.0:
        return words

    noisy_short_tokens = {
        "ah",
        "eh",
        "ha",
        "huh",
        "la",
        "mm",
        "mmm",
        "na",
        "oh",
        "ooh",
        "uh",
        "woo",
        "yeah",
        "yo",
    }

    def keep_word(word: timing_models.TranscriptionWord) -> bool:
        if word.text == "[VOCAL]":
            return True
        prob = float(getattr(word, "probability", 1.0))
        if prob >= threshold:
            return True
        normalized = "".join(ch for ch in word.text.lower() if ch.isalpha())
        if not normalized:
            return False
        if normalized in noisy_short_tokens:
            return False
        return len(normalized) >= 4

    filtered = [w for w in words if keep_word(w)]
    if not filtered:
        return words
    if len(filtered) < min_keep_words:
        return words
    if (len(filtered) / len(words)) < min_keep_ratio:
        return words
    return filtered
