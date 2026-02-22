from __future__ import annotations

from typing import Any, Optional

from ..text_utils import LYRIC_FUNCTION_WORDS, normalize_text_basic

_INTRO_META_KEYWORDS = {
    "karaoke",
    "singking",
    "karafun",
    "stingray",
    "megakaraoke",
    "cckaraoke",
    "zoomkaraoke",
    "sunfly",
    "songjam",
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
    "fun",
    "produced",
    "producer",
    "writer",
    "composed",
    "arranged",
    "distributed",
    "copyright",
    "reserved",
}


def is_intro_artifact(
    entry: dict[str, Any], artist: Optional[str] = None, is_pre_anchor: bool = True
) -> bool:
    text = str(entry.get("text", ""))
    raw_words = entry.get("words", [])
    words = []
    for w in raw_words:
        if isinstance(w, dict):
            words.append(str(w.get("text", "")))
        else:
            words.append(str(w))
    words = [w for w in words if w.strip()]

    duration = float(entry.get("last", entry.get("end", 0.0))) - float(
        entry.get("first", entry.get("start", 0.0))
    )
    cleaned = normalize_text_basic(text).strip()
    compact_words = ["".join(ch for ch in w if ch.isalnum()) for w in cleaned.split()]
    compact_words = [w for w in compact_words if w]

    if not compact_words:
        return True

    text_l = cleaned.lower()
    if _is_metadata_keyword_line(text_l):
        return True

    if _is_pure_numeric_artifact(compact_words):
        return True

    if artist and _is_artist_name_artifact(text_l, compact_words, artist):
        return True

    # Aggressive filters only apply BEFORE we've found the true lyrics anchor.
    if not is_pre_anchor:
        return False

    upper_chars = sum(1 for ch in text if ch.isalpha() and ch.isupper())
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    upper_ratio = (upper_chars / alpha_chars) if alpha_chars else 0.0

    if _is_short_caps_artifact(compact_words, upper_ratio):
        return True

    first_t = float(entry.get("first", entry.get("start", 0.0)))
    if first_t < 12.0 and _is_repeated_intro_pattern(compact_words):
        return True

    if _is_title_case_credit_line(text, compact_words, first_t):
        return True

    if _is_generic_short_noise(words, compact_words, duration, first_t, text):
        return True

    return False


def _is_metadata_keyword_line(text_l: str) -> bool:
    if any(k in text_l for k in _INTRO_META_KEYWORDS):
        return True
    if any(k in text_l.replace(" ", "") for k in _INTRO_META_KEYWORDS):
        return True
    return False


def _is_pure_numeric_artifact(compact_words: list[str]) -> bool:
    if compact_words and all(w.isdigit() for w in compact_words):
        return True
    return False


def _is_artist_name_artifact(
    text_l: str, compact_words: list[str], artist: str
) -> bool:
    artist_l = normalize_text_basic(artist).lower()
    if text_l == artist_l:
        return True
    if len(compact_words) <= 2 and text_l in artist_l:
        return True
    return False


def _is_short_caps_artifact(compact_words: list[str], upper_ratio: float) -> bool:
    if len(compact_words) <= 2:
        max_len = max((len(w) for w in compact_words), default=0)
        if max_len <= 4 and upper_ratio >= 0.75:
            return True
    return False


def _is_repeated_intro_pattern(compact_words: list[str]) -> bool:
    if len(compact_words) >= 4:
        mid = len(compact_words) // 2
        if compact_words[:mid] == compact_words[mid:]:
            return True
    return False


def _is_title_case_credit_line(
    text: str, compact_words: list[str], first_t: float
) -> bool:
    if first_t >= 15.0:
        return False
    all_title_case = (
        bool(compact_words)
        and all(
            token[:1].isupper() and (token[1:].islower() if len(token) > 1 else True)
            for token in text.split()
            if any(ch.isalpha() for ch in token)
        )
        and len(compact_words) <= 6
    )
    if not all_title_case:
        return False

    words_l = [w.lower() for w in compact_words]
    if all(w not in LYRIC_FUNCTION_WORDS for w in words_l):
        # Only filter if it's not a common lyric interjection
        if not any(w in {"duh", "oh", "yeah", "hey", "ah"} for w in words_l):
            return True
    return False


def _is_generic_short_noise(
    words: list[str],
    compact_words: list[str],
    duration: float,
    first_t: float,
    text: str,
) -> bool:
    if (
        len(words) <= 2
        and duration < 1.1
        and "'" not in text
        and "-" not in text
        and first_t < 15.0
    ):
        words_l = [w.lower() for w in compact_words]
        if not any(w in LYRIC_FUNCTION_WORDS or w in {"duh"} for w in words_l):
            return True
    return False


def filter_intro_non_lyrics(
    entries: list[dict[str, Any]], artist: Optional[str] = None
) -> list[dict[str, Any]]:
    if len(entries) < 2:
        return entries

    anchor_idx = _find_lyrics_anchor_idx(entries, artist)

    if anchor_idx is None:
        return _handle_missing_anchor(entries)

    return _filter_artifacts_around_anchor(entries, anchor_idx, artist)


def _find_lyrics_anchor_idx(
    entries: list[dict[str, Any]], artist: Optional[str]
) -> Optional[int]:
    candidates: list[int] = []
    for idx, ent in enumerate(entries):
        raw_words = ent.get("words", [])
        words = []
        for w in raw_words:
            if isinstance(w, dict):
                words.append(str(w.get("text", "")))
            else:
                words.append(str(w))
        words = [w for w in words if w.strip()]

        start_t = float(ent.get("first", ent.get("start", 0.0)))
        end_t = float(ent.get("last", ent.get("end", 0.0)))
        duration = end_t - start_t

        if len(words) >= 3 and duration >= 0.8 and not is_intro_artifact(ent, artist):
            candidates.append(idx)

    if not candidates:
        return None

    later = [
        idx
        for idx in candidates
        if float(entries[idx].get("first", entries[idx].get("start", 0.0))) >= 10.0
    ]
    return later[0] if later else candidates[0]


def _handle_missing_anchor(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # If no anchor found, we might still want to drop artifacts,
    # but only if the video is long enough to have real lyrics.
    last_t = float(entries[-1].get("first", entries[-1].get("start", 0.0)))
    if last_t > 15.0:
        return _filter_artifacts_around_anchor(entries, len(entries), None)

    # Short snippet/test: don't filter anything unless it's a known brand keyword.
    kept: list[dict[str, Any]] = []
    for ent in entries:
        txt = normalize_text_basic(str(ent.get("text", ""))).lower()
        if _is_metadata_keyword_line(txt):
            continue
        kept.append(ent)
    return kept


def _filter_artifacts_around_anchor(
    entries: list[dict[str, Any]], anchor_idx: int, artist: Optional[str]
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    # Drop identified artifacts throughout the song.
    for idx, ent in enumerate(entries):
        is_pre = idx < anchor_idx
        if is_intro_artifact(ent, artist, is_pre_anchor=is_pre):
            if is_pre:
                continue

            # After anchor: only drop if it's metadata keywords or pure numeric
            text_l = (
                normalize_text_basic(str(ent.get("text", ""))).lower().replace(" ", "")
            )
            if any(k in text_l for k in _INTRO_META_KEYWORDS):
                continue

            compact_words = [
                "".join(ch for ch in w if ch.isalnum()) for w in text_l.split()
            ]
            if compact_words and all(w.isdigit() for w in compact_words):
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
