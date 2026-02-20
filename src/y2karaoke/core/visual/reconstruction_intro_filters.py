from __future__ import annotations

from typing import Any

from ..text_utils import normalize_text_basic

_INTRO_META_KEYWORDS = {
    "karaoke",
    "singking",
    "version",
    "official",
    "records",
    "universal",
    "kobalt",
    "publishing",
    "ltd",
    "mca",
    "oconnell",
    "connell",
    "lyrics",
    "instrumental",
}

_LYRIC_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "dont",
    "for",
    "i",
    "im",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "oh",
    "on",
    "so",
    "the",
    "to",
    "we",
    "yeah",
    "you",
    "your",
}


def is_intro_artifact(entry: dict[str, Any]) -> bool:
    text = str(entry.get("text", ""))
    words = [w for w in entry.get("words", []) if str(w).strip()]
    duration = float(entry.get("last", 0.0)) - float(entry.get("first", 0.0))
    cleaned = normalize_text_basic(text).strip()
    compact_words = ["".join(ch for ch in w if ch.isalnum()) for w in cleaned.split()]
    compact_words = [w for w in compact_words if w]

    if not compact_words:
        return True

    text_l = cleaned.lower()
    if any(k in text_l for k in _INTRO_META_KEYWORDS):
        return True

    upper_chars = sum(1 for ch in text if ch.isalpha() and ch.isupper())
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    upper_ratio = (upper_chars / alpha_chars) if alpha_chars else 0.0

    short_caps = (
        len(compact_words) <= 2
        and max((len(w) for w in compact_words), default=0) <= 4
        and upper_ratio >= 0.75
    )
    if short_caps:
        return True

    first_t = float(entry.get("first", 0.0))
    if first_t < 12.0 and len(compact_words) >= 4:
        mid = len(compact_words) // 2
        if compact_words[:mid] == compact_words[mid:]:
            return True

    all_title_case = (
        bool(compact_words)
        and all(
            token[:1].isupper() and (token[1:].islower() if len(token) > 1 else True)
            for token in text.split()
            if any(ch.isalpha() for ch in token)
        )
        and len(compact_words) <= 3
    )
    if all_title_case and first_t < 12.0:
        words_l = [w.lower() for w in compact_words]
        if all(w not in _LYRIC_FUNCTION_WORDS for w in words_l):
            return True

    if len(words) <= 2 and duration < 1.1 and "'" not in text and "-" not in text:
        return True

    return False


def filter_intro_non_lyrics(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(entries) < 2:
        return entries

    anchor_idx = None
    candidates: list[int] = []
    for idx, ent in enumerate(entries):
        words = [w for w in ent.get("words", []) if str(w).strip()]
        duration = float(ent.get("last", 0.0)) - float(ent.get("first", 0.0))
        if len(words) >= 3 and duration >= 0.8 and not is_intro_artifact(ent):
            candidates.append(idx)

    if candidates:
        later = [
            idx for idx in candidates if float(entries[idx].get("first", 0.0)) >= 10.0
        ]
        anchor_idx = later[0] if later else candidates[0]

    if anchor_idx is None or anchor_idx == 0:
        return entries

    first_t = float(entries[0].get("first", 0.0))
    anchor_t = float(entries[anchor_idx].get("first", 0.0))
    if anchor_t - first_t < 6.0:
        return entries

    kept: list[dict[str, Any]] = []
    for idx, ent in enumerate(entries):
        if idx < anchor_idx and is_intro_artifact(ent):
            continue
        kept.append(ent)
    return kept


def suppress_bottom_fragment_families(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(entries) < 4:
        return entries

    ys = [float(ent.get("y", 0.0)) for ent in entries]
    y_min = min(ys)
    y_max = max(ys)
    if (y_max - y_min) < 80.0:
        return entries
    bottom_cut = y_min + 0.84 * (y_max - y_min)

    candidates: list[tuple[str, str]] = []
    for ent in entries:
        words = [str(w).strip() for w in ent.get("words", []) if str(w).strip()]
        if len(words) != 1:
            continue
        tok = words[0]
        compact = "".join(ch for ch in tok if ch.isalnum())
        if not compact or len(compact) > 4:
            continue
        upper_chars = sum(1 for ch in tok if ch.isalpha() and ch.isupper())
        alpha_chars = sum(1 for ch in tok if ch.isalpha())
        upper_ratio = (upper_chars / alpha_chars) if alpha_chars else 0.0
        if upper_ratio < 0.8:
            continue
        if float(ent.get("y", 0.0)) < bottom_cut:
            continue
        candidates.append((compact[0].lower(), compact.lower()))

    if not candidates:
        return entries

    counts: dict[str, int] = {}
    variants: dict[str, set[str]] = {}
    for initial, token in candidates:
        counts[initial] = counts.get(initial, 0) + 1
        variants.setdefault(initial, set()).add(token)

    noisy_initials = {
        initial
        for initial, count in counts.items()
        if count >= 4 and len(variants.get(initial, set())) >= 3
    }
    if not noisy_initials:
        return entries

    out: list[dict[str, Any]] = []
    for ent in entries:
        words = [str(w).strip() for w in ent.get("words", []) if str(w).strip()]
        if len(words) != 1:
            out.append(ent)
            continue
        tok = words[0]
        compact = "".join(ch for ch in tok if ch.isalnum())
        if not compact or len(compact) > 4:
            out.append(ent)
            continue
        initial = compact[0].lower()
        upper_chars = sum(1 for ch in tok if ch.isalpha() and ch.isupper())
        alpha_chars = sum(1 for ch in tok if ch.isalpha())
        upper_ratio = (upper_chars / alpha_chars) if alpha_chars else 0.0
        is_bottom = float(ent.get("y", 0.0)) >= bottom_cut
        if initial in noisy_initials and upper_ratio >= 0.8 and is_bottom:
            continue
        out.append(ent)
    return out
