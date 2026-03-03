"""OCR/spelling helper utilities for visual bootstrap postprocessing."""

from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Callable

OCR_SUB_CHAR_MAP = {
    "i": ("l",),
    "l": ("i",),
    "1": ("l", "i"),
    "|": ("l", "i"),
    "y": ("w",),
    "w": ("y",),
    "0": ("o",),
    "o": ("0",),
}


@lru_cache(maxsize=1)
def fallback_spell_checker() -> Any:
    try:
        from AppKit import NSSpellChecker

        return NSSpellChecker.sharedSpellChecker()
    except Exception:
        return None


def fallback_spell_guess(
    token: str, *, fallback_spell_checker_fn: Callable[[], Any]
) -> str | None:
    checker = fallback_spell_checker_fn()
    if checker is None:
        return None
    try:
        missed = checker.checkSpellingOfString_startingAt_(token, 0)
        if missed.length <= 0:
            return None
        guesses = checker.guessesForWordRange_inString_language_inSpellDocumentWithTag_(
            missed, token, "en", 0
        )
        if not guesses:
            return None
        return str(guesses[0])
    except Exception:
        return None


def is_safe_spell_guess_correction(source: str, guess: str) -> bool:
    s = source.lower()
    g = guess.lower()
    if not s.isalpha() or not g.isalpha():
        return False
    if s == g:
        return False
    if len(s) < 4 or len(g) < 4:
        return False
    if abs(len(s) - len(g)) > 1:
        return False
    if SequenceMatcher(None, s, g).ratio() < 0.75:
        return False
    if not (s[:2] == g[:2] or s[-3:] == g[-3:] or s[1:] == g[1:]):
        return False
    return True


def ocr_substitution_candidates(
    token: str,
    *,
    is_spelled_word_fn: Callable[[str], bool],
    sub_char_map: dict[str, tuple[str, ...]] = OCR_SUB_CHAR_MAP,
) -> list[str]:
    low = token.lower()
    if not low.isalpha() or len(low) < 3 or len(low) > 8:
        return []

    candidates: set[str] = set()
    chars = list(low)
    for i, ch in enumerate(chars):
        for repl in sub_char_map.get(ch, ()):
            if repl == ch:
                continue
            cand_chars = chars.copy()
            cand_chars[i] = repl
            candidates.add("".join(cand_chars))
    if len(low) <= 6 and not is_spelled_word_fn(low):
        one_edit = list(candidates)
        for base in one_edit:
            bchars = list(base)
            for i, ch in enumerate(bchars):
                for repl in sub_char_map.get(ch, ()):
                    if repl == ch:
                        continue
                    cand_chars = bchars.copy()
                    cand_chars[i] = repl
                    candidates.add("".join(cand_chars))

    out: list[str] = []
    for cand in sorted(candidates):
        if cand == low:
            continue
        if is_spelled_word_fn(cand):
            out.append(cand)
    return out


def ocr_insertion_candidates(
    token: str, *, is_spelled_word_fn: Callable[[str], bool]
) -> list[str]:
    low = token.lower()
    if not low.isalpha() or len(low) < 2 or len(low) > 5:
        return []
    if is_spelled_word_fn(low):
        return []
    alphabet = ("w", "l", "i", "e", "o", "a", "u", "r", "n")
    out: list[str] = []
    seen: set[str] = set()
    for i in range(len(low) + 1):
        for ch in alphabet:
            cand = low[:i] + ch + low[i:]
            if cand in seen:
                continue
            seen.add(cand)
            if is_spelled_word_fn(cand):
                out.append(cand)
    return out


def best_ocr_substitution(
    token: str,
    *,
    is_spelled_word_fn: Callable[[str], bool],
    case_like_fn: Callable[[str, str], str],
    ocr_substitution_candidates_fn: Callable[[str], list[str]],
    ocr_insertion_candidates_fn: Callable[[str], list[str]],
) -> str | None:
    low = token.lower()
    if is_spelled_word_fn(low):
        return None
    best: tuple[float, str] | None = None
    for cand in ocr_substitution_candidates_fn(low):
        score = SequenceMatcher(None, low, cand).ratio()
        if score < 0.58:
            continue
        if best is None or score > best[0]:
            best = (score, cand)
    if best is None:
        for cand in ocr_insertion_candidates_fn(low):
            score = SequenceMatcher(None, low, cand).ratio()
            if score < 0.72:
                continue
            if best is None or score > best[0]:
                best = (score, cand)
    if best is None:
        return None
    return case_like_fn(token, best[1])
