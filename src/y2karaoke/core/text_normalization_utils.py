"""Generic text normalization helpers."""

import re
import unicodedata
from difflib import SequenceMatcher


def normalize_text_basic(text: str) -> str:
    """Basic normalization: lowercase and strip punctuation-like noise."""
    if not text:
        return ""
    t = text.lower().replace("-", " ")
    return re.sub(r"[^a-z0-9' ]+", "", t).strip()


def text_similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
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
