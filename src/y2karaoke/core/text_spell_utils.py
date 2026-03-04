"""Spell-correction helpers used by OCR text normalization."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


def is_correctly_spelled(word: str, checker: Any) -> bool:
    """Check if a word is correctly spelled according to the checker."""
    if not checker:
        return True
    # missed.length == 0 means it's correctly spelled
    return checker.checkSpellingOfString_startingAt_(word, 0).length == 0


def get_ocr_variants(word: str) -> list[str]:
    """Generate potential corrections based on common OCR confusions.

    Returns a deterministic BFS-ordered list (closest/simple rewrites first).
    """
    # (confusion, target) - we want to go from OCR error to real word
    rules = [
        ("rn", "m"),
        ("in", "m"),
        ("ni", "m"),
        ("ri", "m"),
        ("li", "h"),
        ("ii", "h"),
        ("vv", "w"),
        ("cl", "d"),
        ("ci", "d"),
        ("ii", "ll"),
        ("ai", "al"),
        ("i", "m"),
        ("m", "in"),
        ("m", "rn"),
        ("m", "ri"),
        ("m", "ni"),
        ("h", "li"),
        ("d", "cl"),
        ("l", "I"),
        ("I", "l"),
        ("0", "O"),
        ("O", "0"),
        ("ti", "th"),
        ("nt", "m"),
        ("t", "f"),
        ("in", "m"),
        ("v", "y"),
        ("b", "m"),
        ("ll", "n"),
    ]

    seed = word.lower()
    seen = {seed}
    variants = [seed]

    # Try applying rules recursively up to a depth of 4 (breadth-first)
    current_tier = [seed]
    for _ in range(4):
        next_tier: list[str] = []
        for w in current_tier:
            for conf, target in rules:
                if conf in w:
                    candidate = w.replace(conf, target)
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    next_tier.append(candidate)
        if not next_tier:
            break
        variants.extend(next_tier)
        current_tier = next_tier

    return variants


def visual_spell_correction_mode() -> str:
    mode = os.environ.get("Y2K_VISUAL_SPELL_CORRECTION_MODE", "").strip().lower()
    if mode in {"full", "no-guesses", "off", "auto"}:
        return mode
    if os.environ.get("Y2K_VISUAL_DISABLE_SPELL_CORRECT", "0") == "1":
        return "off"
    if os.environ.get("Y2K_VISUAL_DISABLE_SPELL_GUESSES", "0") == "1":
        return "no-guesses"
    return "full"


def looks_ocr_suspicious(word: str) -> bool:
    low = word.lower()
    if len(low) < 3:
        return False
    if any(ch.isdigit() for ch in low) and any(ch.isalpha() for ch in low):
        return True
    if "|" in low:
        return True
    # Common OCR confusion clusters. Avoid overly common patterns like "in".
    if any(cluster in low for cluster in ("vv", "ii")):
        return True
    if low.startswith(("rn", "ri", "ni")):
        return True
    # Many narrow glyphs often indicate OCR confusion (e.g. lI1 noise).
    narrow_count = sum(1 for ch in low if ch in {"l", "i", "1"})
    if len(low) >= 4 and narrow_count >= 3:
        return True
    return False


@lru_cache(maxsize=4096)
def spell_correct_cached(text: str, mode: str) -> str:
    """Mode-aware cached spell correction.

    The mode is part of the cache key to avoid cross-mode contamination.
    """
    if not text:
        return text
    if mode == "off":
        return text

    try:
        from AppKit import NSSpellChecker

        checker = NSSpellChecker.sharedSpellChecker()
    except Exception:
        checker = None

    words = text.split()
    corrected = []
    for w in words:
        low_w = w.lower()
        if len(low_w) < 3 or not checker:
            corrected.append(w)
            continue

        # 1. Already correct? (Highest priority)
        if is_correctly_spelled(low_w, checker):
            corrected.append(w)
            continue

        auto_mode = mode == "auto"
        if auto_mode and not looks_ocr_suspicious(low_w):
            corrected.append(w)
            continue

        # 2. Try OCR variant generation
        correction = None
        variants = get_ocr_variants(low_w)
        for variant in variants:
            if variant != low_w and is_correctly_spelled(variant, checker):
                correction = variant
                break

        # 3. Use correction or keep original
        if correction:
            if w.isupper():
                corrected.append(correction.upper())
            elif w[0].isupper():
                corrected.append(correction.capitalize())
            else:
                corrected.append(correction)
        else:
            corrected.append(w)

    return " ".join(corrected)
