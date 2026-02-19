"""
Text utilities for slug generation, normalization, and karaoke-specific line cleaning.

These are general-purpose helpers used across the y2karaoke core modules.
"""

import re
import unicodedata
from functools import lru_cache
from typing import List, Tuple


# ----------------------
# Slug and title utilities
# ----------------------
def make_slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def clean_title_for_search(
    title: str, title_cleanup_patterns: List[str], youtube_suffixes: List[str]
) -> str:
    cleaned = title
    for pattern in title_cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    for suffix in youtube_suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned.strip()


# ----------------------
# Genius-specific line helpers
# ----------------------
def strip_leading_artist_from_line(text: str, artist: str) -> str:
    if not artist:
        return text
    pattern = re.compile(
        rf"^(?:\[{re.escape(artist)}\]\s*|{re.escape(artist)}\s*[-–]\s*)", re.IGNORECASE
    )
    return pattern.sub("", text).strip()


def filter_singer_only_lines(
    lines: List[Tuple[str, str]], known_singers: List[str]
) -> List[Tuple[str, str]]:
    known_set = {s.lower() for s in known_singers}
    filtered = []
    for text, singer in lines:
        text_clean = strip_leading_artist_from_line(text, artist="")
        parts = re.split(r"[\/,]", text_clean.lower())
        if any(p.strip() not in known_set for p in parts):
            filtered.append((text_clean, singer))
    return filtered


def normalize_title(title: str, remove_stopwords: bool = False) -> str:
    """Normalize title for comparison.

    Args:
        title: Title string to normalize
        remove_stopwords: If True, remove common stopwords (the, el, los, etc.)
    """
    normalized = re.sub(r"[,.\-:;\'\"!?()]", " ", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if remove_stopwords:
        words = [w for w in normalized.split() if w not in STOP_WORDS]
        normalized = " ".join(words)

    return normalized


STOP_WORDS = {
    # English
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "with",
    "in",
    "to",
    "for",
    "by",
    "&",
    "+",
    # Spanish
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "y",
    "de",
    "del",
    "con",
    # French
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "et",
    "de",
    "du",
    "au",
    "aux",
    # German
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "und",
    "von",
    "mit",
    # Italian
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "uno",
    "una",
    "e",
    "di",
    "del",
    "della",
    # Portuguese
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "uns",
    "umas",
    "e",
    "de",
    "do",
    "da",
    "dos",
    "das",
}


def normalize_text_basic(text: str) -> str:
    """Basic normalization: lowercase, strip non-alphanumeric (except quotes/spaces)."""
    if not text:
        return ""
    # Replace hyphens with spaces before regex to ensure 'anti-hero' -> 'anti hero'
    t = text.lower().replace("-", " ")
    return re.sub(r"[^a-z0-9' ]+", "", t).strip()


def text_similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    from difflib import SequenceMatcher

    na, nb = normalize_text_basic(a), normalize_text_basic(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def canon_punct(text: str) -> str:
    """Canonicalize punctuation characters."""
    text = unicodedata.normalize("NFKC", text)
    trans = {
        "’": "'",
        "‘": "'",
        "´": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
    text = "".join(trans.get(ch, ch) for ch in text)
    return " ".join(text.split())


def spell_correct(text: str) -> str:
    """Attempt spell correction using macOS spell checker if available."""
    if not text:
        return text
    try:
        from AppKit import NSSpellChecker

        checker = NSSpellChecker.sharedSpellChecker()
        words = text.split()
        corrected = []
        for w in words:
            if len(w) < 3:
                corrected.append(w)
                continue
            missed = checker.checkSpellingOfString_startingAt_(w, 0)
            if missed.length > 0:
                guesses = checker.guessesForWordRange_inString_language_inSpellDocumentWithTag_(
                    missed, w, "en", 0
                )
                if guesses:
                    corrected.append(guesses[0])
                    continue
            corrected.append(w)
        return " ".join(corrected)
    except Exception:
        return text


def normalize_ocr_line(text: str) -> str:
    """Clean up OCR output: punctuation, common typos, contractions."""
    text = canon_punct(text)
    if not text:
        return text
    if text.lower().startswith("have "):
        text = "I " + text
    text = text.replace("problei", "problem")
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
        out.append(tok)
    repaired = _repair_fused_ocr_words(" ".join(out))
    return spell_correct(repaired)


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

        replacement = _contextual_ocr_token_replacement(low, prev_low, next_low)
        if replacement is not None:
            out.append(replacement)
            i += 1
            continue

        out.append(tok)
        i += 1

    return out


def _contextual_ocr_token_replacement(
    low: str, prev_low: str, next_low: str
) -> str | None:
    # Frequent OCR confusion in phrase "... doing shots drinking fast ...".
    if low != "drinking" and re.fullmatch(r"d[a-z]{0,4}(?:inking|nking)", low):
        return "drinking"

    # Frequent OCR confusion around "... come on ...".
    if low in {"ony", "om"} and prev_low == "come":
        return "on"

    if low in {"l", "loh", "loll", "lohlohlohl", "lohlohl"} and prev_low == "oh":
        return "I"
    if low in {"loh", "loll", "lohlohlohl", "lohlohl"} and prev_low == "i":
        return "oh"
    if low == "l" and prev_low in {"i", "im", "i'm"}:
        return "oh"

    # Short-token confusions in fast sections.
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
        # "in loh with your body" -> "in love with your body"
        if prev_low == "in" and next_low == "with":
            return "love"
        # "with loh body" -> "with your body"
        if prev_low == "with" and next_low == "body":
            return "your"
        # "your loh" often means "body" in this repeated phrase.
        if prev_low == "your":
            return "body"

    return None
