"""
Text utilities for slug generation, normalization, and karaoke-specific line cleaning.

These are general-purpose helpers used across the y2karaoke core modules.
"""

import re
from functools import lru_cache
from typing import Any, List, Tuple

from .text_normalization_utils import (
    canon_punct as _canon_punct_impl,
    normalize_text_basic as _normalize_text_basic_impl,
    text_similarity as _text_similarity_impl,
)
from .text_ocr_context import (
    contextual_ocr_token_replacement as _contextual_ocr_token_replacement_impl,
    regularize_short_chant_alternation as _regularize_short_chant_alternation_impl,
)
from .text_spell_utils import (
    get_ocr_variants as _get_ocr_variants_impl,
)
from .text_spell_utils import (
    is_correctly_spelled as _is_correctly_spelled_impl,
)
from .text_spell_utils import (
    looks_ocr_suspicious as _looks_ocr_suspicious_impl,
)
from .text_spell_utils import (
    visual_spell_correction_mode as _visual_spell_correction_mode_impl,
)
from .text_title_utils import (
    STOP_WORDS as _STOP_WORDS,
    clean_title_for_search as _clean_title_for_search_impl,
    filter_singer_only_lines as _filter_singer_only_lines_impl,
    make_slug as _make_slug_impl,
    normalize_title as _normalize_title_impl,
    strip_leading_artist_from_line as _strip_leading_artist_from_line_impl,
)

STOP_WORDS = _STOP_WORDS


# ----------------------
# Slug and title utilities
# ----------------------
def make_slug(text: str) -> str:
    return _make_slug_impl(text)


def clean_title_for_search(
    title: str, title_cleanup_patterns: List[str], youtube_suffixes: List[str]
) -> str:
    return _clean_title_for_search_impl(title, title_cleanup_patterns, youtube_suffixes)


# ----------------------
# Genius-specific line helpers
# ----------------------
def strip_leading_artist_from_line(text: str, artist: str) -> str:
    return _strip_leading_artist_from_line_impl(text, artist)


def filter_singer_only_lines(
    lines: List[Tuple[str, str]], known_singers: List[str]
) -> List[Tuple[str, str]]:
    return _filter_singer_only_lines_impl(lines, known_singers)


def normalize_title(title: str, remove_stopwords: bool = False) -> str:
    return _normalize_title_impl(title, remove_stopwords=remove_stopwords)


LYRIC_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "dont",
    "for",
    "from",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "im",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "not",
    "of",
    "oh",
    "on",
    "or",
    "she",
    "so",
    "the",
    "they",
    "this",
    "to",
    "was",
    "we",
    "with",
    "yeah",
    "you",
    "your",
}


def normalize_text_basic(text: str) -> str:
    return _normalize_text_basic_impl(text)


def text_similarity(a: str, b: str) -> float:
    return _text_similarity_impl(a, b)


def canon_punct(text: str) -> str:
    return _canon_punct_impl(text)


def _is_correctly_spelled(word: str, checker: Any) -> bool:
    return _is_correctly_spelled_impl(word, checker)


def _get_ocr_variants(word: str) -> list[str]:
    return _get_ocr_variants_impl(word)


def spell_correct(text: str) -> str:
    """Attempt spell correction using the active visual spell-correction mode."""
    mode = _visual_spell_correction_mode()
    return _spell_correct_cached(text, mode)


def _visual_spell_correction_mode() -> str:
    return _visual_spell_correction_mode_impl()


def _looks_ocr_suspicious(word: str) -> bool:
    return _looks_ocr_suspicious_impl(word)


@lru_cache(maxsize=4096)
def _spell_correct_cached(text: str, mode: str) -> str:
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
        if _is_correctly_spelled(low_w, checker):
            corrected.append(w)
            continue

        auto_mode = mode == "auto"
        if auto_mode and not _looks_ocr_suspicious(low_w):
            corrected.append(w)
            continue

        # 2. Misspelled? Try standard OCR confusion reversals
        found_variant = False
        variants = _get_ocr_variants(low_w)
        for variant in variants:
            if variant != low_w and _is_correctly_spelled(variant, checker):
                # We found a valid correction via OCR rules.
                corrected.append(_case_like(w, variant))
                found_variant = True
                break
        if found_variant:
            continue

        # 3. Fallback to standard system guesses
        if mode in {"no-guesses", "auto"}:
            corrected.append(w)
            continue
        missed = checker.checkSpellingOfString_startingAt_(w, 0)
        if missed.length > 0:
            guesses = (
                checker.guessesForWordRange_inString_language_inSpellDocumentWithTag_(
                    missed, w, "en", 0
                )
            )
            if guesses:
                corrected.append(guesses[0])
                continue

        corrected.append(w)
    return " ".join(corrected)


def normalize_ocr_line(text: str) -> str:
    """Clean up OCR output: punctuation, common typos, contractions."""
    text = canon_punct(text)
    if not text:
        return text
    if text.lower().startswith("have "):
        text = "I " + text
    toks = text.split()
    out: List[str] = []
    contractions = {"'ll", "'re", "'ve", "'m", "'d"}
    confusable_i = {"1", "|", "!"}
    for i, tok in enumerate(toks):
        prev_tok = out[-1] if out else ""
        next_tok = toks[i + 1] if i + 1 < len(toks) else ""
        if tok in contractions and prev_tok:
            out[-1] = prev_tok + tok
            continue
        if tok in confusable_i:
            if any(c.isalpha() for c in prev_tok) or any(c.isalpha() for c in next_tok):
                tok = "I"
        if tok.lower() == "l" and prev_tok.lower() in {"oh", "i", "loh", "ioh"}:
            tok = "I"
        chant_split = _split_confusable_chant_token(tok)
        if chant_split:
            out.extend(chant_split)
            continue
        out.append(tok)
    out = _regularize_short_chant_alternation(out)
    repaired = _repair_fused_ocr_words(" ".join(out))
    if _visual_spell_correction_mode() == "off":
        return repaired
    return spell_correct(repaired)


def _split_confusable_chant_token(token: str) -> List[str] | None:
    compact = "".join(ch for ch in token.lower() if ch.isalnum())
    if len(compact) < 3 or len(compact) > 16:
        return None
    if any(ch not in {"o", "h", "i", "l", "1"} for ch in compact):
        return None
    if "oh" not in compact:
        return None

    out: List[str] = []
    i = 0
    while i < len(compact):
        if compact.startswith("oh", i):
            out.append("oh")
            i += 2
            continue
        if compact[i] in {"i", "l", "1"}:
            out.append("I")
            i += 1
            continue
        return None

    # Split only when we recovered at least one "I" and one "oh" atom.
    if len(out) < 2 or "I" not in out or "oh" not in out:
        return None
    return out


_FUSED_SPLIT_ANCHORS = {
    "i",
    "a",
    "an",
    "am",
    "is",
    "are",
    "to",
    "my",
    "me",
    "we",
    "you",
    "he",
    "she",
    "it",
    "they",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
}


@lru_cache(maxsize=1)
def _lexicon_words() -> set[str]:
    try:
        import cmudict

        return set(cmudict.dict().keys())
    except Exception:
        return set()


def _case_like(source: str, token: str) -> str:
    if token.lower() == "i":
        return "I"
    if source.isupper():
        return token.upper()
    if source[:1].isupper() and source[1:].islower():
        return token.capitalize()
    if source.islower():
        return token.lower()
    return token


def _best_fused_split(token: str) -> Tuple[str, str] | None:
    if not token.isalpha() or len(token) < 4:
        return None

    lower = token.lower()
    lex = _lexicon_words()
    if not lex or lower in lex:
        return None

    best: Tuple[int, Tuple[str, str]] | None = None
    for i in range(1, len(lower)):
        left = lower[:i]
        right = lower[i:]
        if len(right) < 2:
            continue
        if left not in lex or right not in lex:
            continue

        # Favor splits anchored by common short function words.
        score = 0
        if left in _FUSED_SPLIT_ANCHORS:
            score += 6
        if right in _FUSED_SPLIT_ANCHORS:
            score += 4
        if len(left) <= 2:
            score += 2
        if len(right) >= 4:
            score += 1

        if best is None or score > best[0]:
            best = (score, (left, right))

    if best is None or best[0] < 5:
        return None
    return best[1]


def _repair_fused_ocr_words(text: str) -> str:
    if not text:
        return text

    toks = text.split()
    out: List[str] = []
    for tok in toks:
        split = _best_fused_split(tok)
        if split is None:
            out.append(tok)
            continue
        left, right = split
        out.append(_case_like(tok[: len(left)], left))
        out.append(_case_like(tok[len(left) :], right))
    return " ".join(out)


def _normalize_confusable_i_token(tok: str, prev_tok: str, next_tok: str) -> str:
    low = tok.lower()
    if low not in {"1", "|", "!"}:
        return tok
    if any(ch.isalpha() for ch in prev_tok) or any(ch.isalpha() for ch in next_tok):
        return "I"
    return tok


def normalize_ocr_tokens(tokens: List[str]) -> List[str]:
    """Normalize OCR token list and merge common contraction fragments.

    This intentionally preserves token count monotonicity (same or fewer tokens),
    so downstream word/ROI arrays remain safely indexable.
    """
    if not tokens:
        return []

    suffixes = {"re", "ve", "ll", "m", "d", "s", "t"}
    compact: List[str] = []
    for raw in tokens:
        tok = canon_punct(str(raw))
        tok = re.sub(r"[^A-Za-z0-9']+", "", tok)
        if not tok:
            continue
        compact.append(tok)

    out: List[str] = []
    i = 0
    while i < len(compact):
        tok = compact[i]
        low = tok.lower()
        prev_low = out[-1].lower() if out else ""
        next_low = compact[i + 1].lower() if i + 1 < len(compact) else ""

        if out and tok in {"'ll", "'re", "'ve", "'m", "'d", "'s", "'t"}:
            out[-1] = out[-1] + tok
            i += 1
            continue

        if (
            tok == "'"
            and out
            and i + 1 < len(compact)
            and compact[i + 1].lower() in suffixes
            and any(ch.isalpha() for ch in out[-1])
        ):
            out[-1] = out[-1] + "'" + compact[i + 1]
            i += 2
            continue

        if tok == "'" and out and any(ch.isalpha() for ch in out[-1]):
            out[-1] = out[-1] + "'"
            i += 1
            continue

        tok = _normalize_confusable_i_token(
            tok, out[-1] if out else "", compact[i + 1] if i + 1 < len(compact) else ""
        )
        low = tok.lower()

        # Common OCR split for "you" at line starts: "' ou come ..."
        if tok == "'" and i + 1 < len(compact) and compact[i + 1].lower() == "ou":
            out.append("you")
            i += 2
            continue

        # Context-aware repair for frequent "you" misread.
        if low == "ou" and (
            not out
            or prev_low in {"and", "but", "so", "then", "if", "when", "cause"}
            or next_low
            in {
                "come",
                "know",
                "want",
                "need",
                "got",
                "are",
                "were",
                "can",
                "could",
                "would",
                "should",
                "say",
                "said",
                "make",
                "made",
                "do",
                "did",
                "dont",
                "don't",
            }
        ):
            out.append("you")
            i += 1
            continue

        # Frequent OCR confusion in phrase "... start up".
        if low == "tup" and prev_low == "start":
            out.append("up")
            i += 1
            continue

        # Frequent OCR confusion in refrain phrase
        # "... come on be my baby come on ...".
        if low in {"one", "an"} and prev_low == "come":
            trailing_baby_phrase = (
                not next_low
                and len(out) >= 4
                and out[-1].lower() == "come"
                and [w.lower() for w in out[-4:-1]] == ["be", "my", "baby"]
            )
            if next_low in {"be", "my", "baby"} or trailing_baby_phrase:
                out.append("on")
                i += 1
                continue

        replacement = _contextual_ocr_token_replacement(low, prev_low, next_low)
        if replacement is not None:
            out.append(replacement)
            i += 1
            continue

        out.append(tok)
        i += 1

    return _regularize_short_chant_alternation(out)


def _regularize_short_chant_alternation(tokens: List[str]) -> List[str]:
    return _regularize_short_chant_alternation_impl(tokens)


def _contextual_ocr_token_replacement(
    low: str, prev_low: str, next_low: str
) -> str | None:
    return _contextual_ocr_token_replacement_impl(low, prev_low, next_low)
